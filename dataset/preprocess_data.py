from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
import timm
from tqdm import tqdm
from configs.cfg import CFG
from dataset.biomass_dataset import *
from torch.utils.data import Dataset, DataLoader
from utils.augs import *
from configs.deterministic import *
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import os
import json

def check_splits(splitter, df):
    fold_stats = []
    for fold, (tr_idx, val_idx) in enumerate(splitter):
        train_fold = df.iloc[tr_idx]
        val_fold   = df.iloc[val_idx]
        n_train = len(train_fold)
        n_val   = len(val_fold)
        ratio   = n_val / (n_train + n_val) * 100

        val_mean_total_dry  = val_fold['Dry_Total_g'].mean()
        val_mean_green_dry  = val_fold['Dry_Green_g'].mean()
        val_mean_dead_dry   = val_fold['Dry_Dead_g'].mean()
        val_mean_clover_dry = val_fold['Dry_Clover_g'].mean()
        val_mean_gdm        = val_fold['GDM_g'].mean()
        val_mean_weighted   = (
            CFG.R2_WEIGHTS_VAL[4] * val_mean_total_dry  +
            CFG.R2_WEIGHTS_VAL[0] * val_mean_green_dry  +
            CFG.R2_WEIGHTS_VAL[1] * val_mean_dead_dry   +
            CFG.R2_WEIGHTS_VAL[2] * val_mean_clover_dry +
            CFG.R2_WEIGHTS_VAL[3] * val_mean_gdm
        )

        # State distribution in val fold
        state_counts = val_fold['State'].value_counts().to_dict()
        n_missions   = val_fold['group'].nunique()
        state_str    = " ".join(f"{s}:{c}" for s, c in sorted(state_counts.items()))

        print(
            f"{fold+1:<5} | {n_train:<12} | {n_val:<10} | {ratio:<6.2f}% | "
            f"Dry_Total_g:{val_mean_total_dry:<8.4f} | Dry_Green_g:{val_mean_green_dry:<8.4f} | "
            f"Dry_Dead_g:{val_mean_dead_dry:<8.4f} | Dry_Clover_g:{val_mean_clover_dry:<8.4f} | "
            f"GDM_g:{val_mean_gdm:<8.4f} | Weighted_g:{val_mean_weighted:<8.4f} | "
            f"missions:{n_missions:<3} | states:[{state_str}]"
        )
        fold_stats.append(n_val)

    fold_stats = np.array(fold_stats)
    mean_size  = np.mean(fold_stats)
    max_dev    = np.max(np.abs(fold_stats - mean_size)) / mean_size * 100
    print("-" * 65)
    print(f"Max deviation from ideal size: {max_dev:.2f}%")

def get_df():
    print("Loading data...")
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_wide = df_wide[['image_path'] + CFG.ALL_TARGET_COLS]
    print(f"{len(df_wide)} training images")
    aux_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    df_aux = df_long[aux_cols].drop_duplicates().reset_index(drop=True)
    df_wide = df_wide.merge(df_aux, on='image_path', how='left')
    df_wide['State_idx'],   STATE_MAP   = pd.factorize(df_wide['State'])
    df_wide['Species_idx'], SPECIES_MAP = pd.factorize(df_wide['Species'])
    df_wide['Sampling_Date'] = pd.to_datetime(df_wide['Sampling_Date'])
    df_wide['day_of_year'] = df_wide['Sampling_Date'].dt.dayofyear
    df_wide['day_sin'] = np.sin(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['day_cos'] = np.cos(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['group'] = df_wide['State'].astype(str) + "_" + df_wide['Sampling_Date'].astype(str)
    df_wide['biomass_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False)
    df_wide['Weighted_g'] = sum(
    df_wide[col] * w for col, w in zip(CFG.ALL_TARGET_COLS, CFG.R2_WEIGHTS_VAL))
    
    # missions = df_wide.groupby('group').size().sort_values(ascending=False)
    # print(f"Total missions: {len(missions)}")
    # print(f"Samples per mission — min: {missions.min()}, max: {missions.max()}, mean: {missions.mean():.1f}, median: {missions.median():.0f}")
    # print(missions)
    return df_wide

def create_hard_extrapolation_split(df, target_col='Dry_Total_g', percentile=0.80):
    cutoff_value = df[target_col].quantile(percentile)
    train_df = df[df[target_col] < cutoff_value].reset_index(drop=True)
    val_df   = df[df[target_col] >= cutoff_value].reset_index(drop=True)
    print(f"\n--- HARD EXTRAPOLATION SPLIT ---")
    print(f"Cutoff Value (80th percentile): {cutoff_value:.2f}")
    print(f"Train Set: {len(train_df)} images (Max Mass: {train_df[target_col].max():.2f})")
    print(f"Valid Set: {len(val_df)} images (Min Mass: {val_df[target_col].min():.2f})")
    print(f"The Validation set contains ONLY values unseen in Training.")
    return train_df, val_df

WEIGHTS_MAP = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g':  0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g':       0.2,
    'Dry_Total_g': 0.5
}

def calculate_weighted_score(df):
    weighted_col = np.zeros(len(df))
    for col, w in WEIGHTS_MAP.items():
        weighted_col += df[col] * w
    return weighted_col

def find_best_seed(df, n_seeds=2000):
    df_search = df.copy()
    df_search['Weighted_g'] = calculate_weighted_score(df_search)
    targets_to_balance = ['Weighted_g', 'Dry_Clover_g', 'Dry_Dead_g']
    best_seed = -1
    lowest_penalty = float('inf')
    print(f"🔍 Searching {n_seeds} seeds for the most balanced split...")
    for seed in tqdm(range(n_seeds)):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_means = {t: [] for t in targets_to_balance}
        try:
            for _, val_idx in sgkf.split(df_search, df_search['biomass_bin'], groups=df_search['group']):
                val_fold = df_search.iloc[val_idx]
                for t in targets_to_balance:
                    fold_means[t].append(val_fold[t].mean())
            current_penalty = 0
            for t in targets_to_balance:
                means_array = np.array(fold_means[t])
                global_mean = df_search[t].mean() + 1e-6
                cv = np.std(means_array) / global_mean
                current_penalty += cv
            if current_penalty < lowest_penalty:
                lowest_penalty = current_penalty
                best_seed = seed
        except ValueError:
            continue
    print("-" * 50)
    print(f"✅ Best Seed Found: {best_seed}")
    print(f"📉 Penalty Score: {lowest_penalty:.4f} (Lower is better)")
    return best_seed

def get_image_id(path):
    """Extract image ID (without extension) from a path."""
    return os.path.splitext(os.path.basename(path))[0]

def extract_features_organized(df, model, save_dir='embeddings', n_aug=20, batch_size=32, device='cuda'):
    """
    Extract DINOv3 embeddings for all images and save as compact single files.
    Uses dynamic list append (same pattern as working extract_features_to_disk)
    to avoid hardcoded dimension bugs.
    
    Output structure:
      embeddings/
        clean_embeddings.pt     # [N, D] — val transforms, no augmentation
        aug_embeddings.pt       # [N, n_aug, D] — augmented versions
        targets.pt              # [N, 5]
        image_ids.csv           # image_path → index mapping
        metadata.json           # config info for reproducibility
    """
    import shutil
    if os.path.exists(save_dir):
        print(f"Removing existing {save_dir}/...")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save targets and image IDs
    targets = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
    torch.save(torch.from_numpy(targets), os.path.join(save_dir, 'targets.pt'))
    df[['image_path']].to_csv(os.path.join(save_dir, 'image_ids.csv'), index=False)
    
    n_images = len(df)
    img_size = CFG.IMG_SIZE
    
    # Use augs.py transforms with CFG.IMG_SIZE
    val_transform = get_val_transforms(img_size)
    spatial_transform = get_spatial_transforms(img_size)
    photometric_transform = get_photometric_transforms(img_size)
    
    # --- Extract clean embeddings (multiplier=1, val transforms) ---
    print("Extracting CLEAN embeddings...")
    dataset_clean = BiomassDatasetBase(
        df, transform=None, photometric_transform=None,
        img_dir=CFG.TRAIN_IMAGE_DIR, multiplier=1,
        val_transform=val_transform
    )
    loader_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_clean = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for left, right, _ in tqdm(loader_clean, desc='Clean'):
            left, right = left.to(device), right.to(device)
            f_l = model(left)
            f_r = model(right)
            feats = torch.cat([f_l, f_r], dim=1).cpu()
            all_clean.append(feats)
    
    clean_embeddings = torch.cat(all_clean)
    torch.save(clean_embeddings, os.path.join(save_dir, 'clean_embeddings.pt'))
    print(f"  Saved: {os.path.join(save_dir, 'clean_embeddings.pt')} ({clean_embeddings.shape})")
    
    # --- Extract augmented embeddings (multiplier=n_aug, train transforms) ---
    print(f"Extracting {n_aug}× AUGMENTED embeddings...")
    
    dataset_aug = BiomassDatasetBase(
        df, transform=spatial_transform, photometric_transform=photometric_transform,
        img_dir=CFG.TRAIN_IMAGE_DIR, multiplier=n_aug,
        val_transform=val_transform
    )
    loader_aug = DataLoader(dataset_aug, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_aug = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for left, right, _ in tqdm(loader_aug, desc='Augmented'):
            left, right = left.to(device), right.to(device)
            f_l = model(left)
            f_r = model(right)
            feats = torch.cat([f_l, f_r], dim=1).cpu()
            all_aug.append(feats)
    
    aug_embeddings = torch.cat(all_aug).view(n_aug, n_images, -1).permute(1, 0, 2)
    torch.save(aug_embeddings, os.path.join(save_dir, 'aug_embeddings.pt'))
    print(f"  Saved: {os.path.join(save_dir, 'aug_embeddings.pt')} ({aug_embeddings.shape})")
    
    # Save metadata
    meta = {
        'n_images': n_images,
        'n_aug': n_aug,
        'img_size': img_size,
        'embedding_dim': clean_embeddings.shape[1],
        'backbone': CFG.MODEL_NAME,
        'target_cols': list(CFG.ALL_TARGET_COLS),
    }
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    total_mb = (clean_embeddings.numel() + aug_embeddings.numel()) * 4 / 1e6
    print(f"Done! {n_images} images × {1 + n_aug} versions = {n_images * (1 + n_aug)} embeddings ({total_mb:.1f} MB total)")
    print(f"Saved to {save_dir}/")
    return save_dir


def extract_features_to_disk(df, model, save_path, mode='val', device='cuda'):
    """
    Runs images through DINO -> Pools -> Saves Tensors to .pt file
    """
    multiplier = 20 if mode == 'train' else 1
    spatial_transform = get_spatial_transforms() if mode=='train' else None
    photometric_transform = get_photometric_transforms() if mode=='train' else get_val_transforms()
    dataset = BiomassDatasetBase(df, transform=spatial_transform, photometric_transform=photometric_transform, img_dir=CFG.TRAIN_IMAGE_DIR, multiplier=multiplier)
    g=get_generator()
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    
    all_features = []
    all_targets = []
    
    print(f"   -> Extracting {mode.upper()} set... ({len(dataset)} samples)")
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for left, right, targets in tqdm(loader, leave=False):
            left = left.to(device)
            right = right.to(device)
            f_l = model(left)
            f_r = model(right)
            x_all = torch.cat([f_l, f_r], dim=1)
            all_features.append(x_all.cpu())
            all_targets.append(targets)
    
    data_dict = {
        'features': torch.cat(all_features),
        'targets': torch.cat(all_targets)
    }
    torch.save(data_dict, save_path)
    print(f"      Saved to {save_path}")

def prepare_cached_folds(df, splits, model_name, base_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Backbone: {model_name}...")
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    backbone.to(device)
    backbone.eval()
    
    for fold, (tr_idx, val_idx) in enumerate(splits):
        print(f"\nProcessing FOLD {fold+1}...")
        fold_dir = os.path.join(base_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        tr_df = df.iloc[tr_idx]
        val_df = df.iloc[val_idx]
        extract_features_to_disk(tr_df, backbone, save_path=os.path.join(fold_dir, 'train.pt'), mode='train', device=device)
        extract_features_to_disk(val_df, backbone, save_path=os.path.join(fold_dir, 'val.pt'), mode='val', device=device)
    
    print("\nAll folds cached successfully!")
    del backbone
    torch.cuda.empty_cache()

from sklearn.neighbors import NearestNeighbors

def get_plant_neighbor_map(features, views_per_plant=20, k=1):
    num_samples, dim = features.shape
    num_plants = num_samples // views_per_plant
    features_reshaped = features.view(num_plants, views_per_plant, dim)
    centroids = features_reshaped.mean(dim=1).cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
    nbrs.fit(centroids)
    _, plant_indices = nbrs.kneighbors(centroids)
    nearest_plant_ids = plant_indices[:, 1]
    neighbor_map = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        current_plant_id = i // views_per_plant
        target_plant_id = nearest_plant_ids[current_plant_id]
        random_view = np.random.randint(0, views_per_plant)
        neighbor_idx = (target_plant_id * views_per_plant) + random_view
        neighbor_map[i] = neighbor_idx
    return neighbor_map

def load_embeddings_for_split(embed_dir, train_idx, val_idx, n_aug=20):
    """
    Load precomputed embeddings for train/val indices using compact single-file format.
    """
    targets = torch.load(os.path.join(embed_dir, 'targets.pt'))
    clean_embeddings = torch.load(os.path.join(embed_dir, 'clean_embeddings.pt'))
    
    return (
        clean_embeddings[train_idx],
        targets[train_idx],
        clean_embeddings[val_idx],
        targets[val_idx]
    )

class EmbeddingAugmentationDataset(Dataset):
    """
    Dataset that samples augmented embeddings on-the-fly from compact single-file storage.
    
    clean_embeddings.pt: [N, D] — val transforms
    aug_embeddings.pt:   [N, n_aug, D] — augmented versions
    
    For training: randomly samples one of (1 clean + n_aug aug) versions per __getitem__ call,
    matching the original image pipeline where copy 0 = clean and copies 1..n_aug = aug.
    For validation: always returns the clean embedding (deterministic).
    """
    def __init__(self, indices, embed_dir, n_aug=20, is_train=True):
        self.indices = indices
        self.embed_dir = embed_dir
        self.n_aug = n_aug
        self.is_train = is_train
        self.targets = torch.load(os.path.join(embed_dir, 'targets.pt'))
        self.clean_embeddings = torch.load(os.path.join(embed_dir, 'clean_embeddings.pt'))
        if os.path.exists(os.path.join(embed_dir, 'aug_embeddings.pt')):
            self.aug_embeddings = torch.load(os.path.join(embed_dir, 'aug_embeddings.pt'))
        else:
            self.aug_embeddings = None
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if self.is_train and self.aug_embeddings is not None:
            # Sample from 1 clean + n_aug augmented = n_aug + 1 total versions
            # aug_idx == 0  → clean embedding
            # aug_idx >= 1  → aug_embeddings[real_idx, aug_idx - 1]
            aug_idx = np.random.randint(0, self.n_aug + 1)
            if aug_idx == 0:
                feat = self.clean_embeddings[real_idx]
            else:
                feat = self.aug_embeddings[real_idx, aug_idx - 1]
        else:
            feat = self.clean_embeddings[real_idx]
        target = self.targets[real_idx]
        return feat, target