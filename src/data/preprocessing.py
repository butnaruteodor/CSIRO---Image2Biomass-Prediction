"""
Data loading, splitting, and feature extraction utilities.
Moved from dataset/preprocess_data.py.
"""
import os
import json
import gc
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import CFG
from src.data.dataset import BiomassDatasetBase, fast_slice_resize_image
from src.data.augmentations import *
from src.deterministic import *


def _get_amp_context(device):
    """Return appropriate autocast context for the device."""
    if device == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu")



# ============================================================================
# Data loading helpers
# ============================================================================

def check_splits(splitter, df):
    """Print fold statistics."""
    fold_stats = []
    for fold, (tr_idx, val_idx) in enumerate(splitter):
        train_fold = df.iloc[tr_idx]
        val_fold = df.iloc[val_idx]
        n_train = len(train_fold)
        n_val = len(val_fold)
        ratio = n_val / (n_train + n_val) * 100

        val_mean_total_dry = val_fold['Dry_Total_g'].mean()
        val_mean_green_dry = val_fold['Dry_Green_g'].mean()
        val_mean_dead_dry = val_fold['Dry_Dead_g'].mean()
        val_mean_clover_dry = val_fold['Dry_Clover_g'].mean()
        val_mean_gdm = val_fold['GDM_g'].mean()
        val_mean_weighted = sum(
            val_fold[col] * w for col, w in zip(CFG.ALL_TARGET_COLS, CFG.R2_WEIGHTS_VAL)
        ).mean()

        state_counts = val_fold['State'].value_counts().to_dict()
        n_missions = val_fold['group'].nunique()
        state_str = " ".join(f"{s}:{c}" for s, c in sorted(state_counts.items()))

        print(
            f"{fold+1:<5} | {n_train:<12} | {n_val:<10} | {ratio:<6.2f}% | "
            f"Dry_Total_g:{val_mean_total_dry:<8.4f} | Dry_Green_g:{val_mean_green_dry:<8.4f} | "
            f"Dry_Dead_g:{val_mean_dead_dry:<8.4f} | Dry_Clover_g:{val_mean_clover_dry:<8.4f} | "
            f"GDM_g:{val_mean_gdm:<8.4f} | Weighted_g:{val_mean_weighted:<8.4f} | "
            f"missions:{n_missions:<3} | states:[{state_str}]"
        )
        fold_stats.append(n_val)

    fold_stats = np.array(fold_stats)
    mean_size = np.mean(fold_stats)
    max_dev = np.max(np.abs(fold_stats - mean_size)) / mean_size * 100
    print("-" * 65)
    print(f"Max deviation from ideal size: {max_dev:.2f}%")


def get_df():
    """Load and preprocess the training dataframe."""
    print("Loading data...")
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_wide = df_wide[['image_path'] + CFG.ALL_TARGET_COLS]
    print(f"{len(df_wide)} training images")
    aux_cols = ['image_path', 'Sampling_Date', 'State', 'Species',
                'Pre_GSHH_NDVI', 'Height_Ave_cm']
    df_aux = df_long[aux_cols].drop_duplicates().reset_index(drop=True)
    df_wide = df_wide.merge(df_aux, on='image_path', how='left')
    df_wide['State_idx'], STATE_MAP = pd.factorize(df_wide['State'])
    df_wide['Species_idx'], SPECIES_MAP = pd.factorize(df_wide['Species'])
    df_wide['Sampling_Date'] = pd.to_datetime(df_wide['Sampling_Date'])
    df_wide['day_of_year'] = df_wide['Sampling_Date'].dt.dayofyear
    df_wide['day_sin'] = np.sin(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['day_cos'] = np.cos(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['group'] = df_wide['State'].astype(str) + "_" + df_wide['Sampling_Date'].astype(str)
    df_wide['biomass_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False)
    df_wide['Weighted_g'] = sum(
        df_wide[col] * w for col, w in zip(CFG.ALL_TARGET_COLS, CFG.R2_WEIGHTS_VAL))

    print(f"States: {df_wide['State'].unique()}")
    print(f"Species: {df_wide['Species'].unique()}")
    dates = df_wide['Sampling_Date'].nunique()
    print(f"Unique missions: {df_wide['group'].nunique()}  (Date x State)")
    print(f"Date range: {df_wide['Sampling_Date'].min().date()} to {df_wide['Sampling_Date'].max().date()}")

    # Store meta
    global _STATE_MAP, _SPECIES_MAP
    _STATE_MAP = STATE_MAP
    _SPECIES_MAP = SPECIES_MAP

    return df_wide


def get_test_df():
    """Load test CSV with ground truth for local evaluation."""
    df_long = pd.read_csv(CFG.BASE_PATH + '/test.csv')
    df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_wide = df_wide[['image_path'] + CFG.ALL_TARGET_COLS]
    print(f"{len(df_wide)} test images")
    aux = df_long[['image_path', 'Usage']].drop_duplicates().reset_index(drop=True)
    df_wide = df_wide.merge(aux, on='image_path', how='left')
    print(f"  Public:  {(df_wide['Usage'] == 'Public').sum()}")
    print(f"  Private: {(df_wide['Usage'] == 'Private').sum()}")
    return df_wide


# ============================================================================
# Feature extraction functions
# ============================================================================

def extract_features_to_disk(df, backbone, save_path, mode='train', device='cuda',
                             img_size=1008, batch_size=8, num_workers=4):
    """
    Extract embeddings from backbone and save to disk.
    mode='train' uses train + photometric augs; 'val' uses val transforms only.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if mode == 'train':
        spatial_tfm = get_spatial_transforms(size=img_size)
        photo_tfm = get_photometric_transforms(size=img_size)
        dataset = BiomassDatasetBase(df, spatial_tfm, photo_tfm,
                                     img_dir=CFG.TRAIN_IMAGE_DIR,
                                     multiplier=21,  # 1 val copy + 20 aug
                                     val_transform=get_val_transforms(size=img_size))
    else:
        dataset = BiomassDatasetBase(df, None, None,
                                     img_dir=CFG.TRAIN_IMAGE_DIR,
                                     multiplier=1,
                                     val_transform=get_val_transforms(size=img_size))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_features = []
    all_targets = []

    with torch.inference_mode():
        for left, right, targets in tqdm(loader, desc=f'Extracting ({mode})'):
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)

            with _get_amp_context(device):
                features = backbone(left, right)

            all_features.append(features.cpu())
            all_targets.append(targets.cpu())

    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)
    torch.save({'features': all_features, 'targets': all_targets}, save_path)
    print(f"Saved {save_path}: features {all_features.shape}, targets {all_targets.shape}")
    del all_features, all_targets, loader, dataset
    gc.collect()


def extract_test_embeddings(backbone, test_df, save_dir='embeddings',
                            img_size=768, device='cuda', batch_size=8):
    """
    Extract embeddings for test images using the backbone.
    Saves to save_dir/test_clean_embeddings.pt, test_targets.pt, test_image_ids.csv.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'test_clean_embeddings.pt')

    if os.path.exists(save_path):
        print(f"Test embeddings already exist at {save_path}, loading...")
        test_embeds = torch.load(save_path)
        test_targets = torch.load(os.path.join(save_dir, 'test_targets.pt'))
        test_ids = pd.read_csv(os.path.join(save_dir, 'test_image_ids.csv'))
        return test_embeds, test_targets, test_ids

    print("Extracting test embeddings...")
    from src.data.augmentations import get_val_transform_for_inference
    from src.data.dataset import TestBiomassDataset

    transform = get_val_transform_for_inference(img_size=img_size)
    dataset = TestBiomassDataset(test_df, transform,
                                 image_dir=CFG.BASE_PATH + '/test')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_features = []
    with torch.inference_mode():
        for left, right in tqdm(loader, desc='Extracting test embeddings'):
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            with _get_amp_context(device):
                features = backbone(left, right)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features).float()

    # Get targets
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    targets = torch.from_numpy(test_df[target_cols].values.astype(np.float32))

    # Save
    torch.save(all_features, save_path)
    torch.save(targets, os.path.join(save_dir, 'test_targets.pt'))
    test_df[['image_path']].to_csv(os.path.join(save_dir, 'test_image_ids.csv'), index=False)

    print(f"Saved test embeddings: {all_features.shape}")
    return all_features, targets, test_df[['image_path']]


# ============================================================================
# Split strategies
# ============================================================================

def get_random_stratified_splits(df, seed):
    """Random stratified 5-fold CV."""
    bins = pd.qcut(df['Dry_Total_g'], q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    check_splits(skf.split(df, bins), df)
    return list(skf.split(df, bins))


def get_date_grouped_splits(df, seed):
    """Date-grouped 5-fold CV."""
    dates = df['Sampling_Date'].astype(str)
    bins = pd.qcut(df['Dry_Total_g'], q=5, labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    check_splits(sgkf.split(df, bins, groups=dates), df)
    return list(sgkf.split(df, bins, groups=dates))


def get_date_location_grouped_splits(df, seed):
    """Date-location grouped 5-fold CV (primary protocol)."""
    groups = df['group']
    bins = pd.qcut(df['Dry_Total_g'], q=5, labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    check_splits(sgkf.split(df, bins, groups=groups), df)
    return list(sgkf.split(df, bins, groups=groups))


# ============================================================================
# Embedding helpers
# ============================================================================

def get_plant_neighbor_map(features, views_per_plant=20, k=1):
    """Build neighbor map for contrastive learning."""
    num_samples, dim = features.shape
    num_plants = num_samples // views_per_plant
    features_reshaped = features.view(num_plants, views_per_plant, dim)
    centroids = features_reshaped.mean(dim=1).cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
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
    """Load precomputed embeddings for train/val split."""
    targets = torch.load(os.path.join(embed_dir, 'targets.pt'))
    clean_embeddings = torch.load(os.path.join(embed_dir, 'clean_embeddings.pt'))
    return (
        clean_embeddings[train_idx],
        targets[train_idx],
        clean_embeddings[val_idx],
        targets[val_idx]
    )


# Global state/species maps (populated by get_df)
_STATE_MAP = None
_SPECIES_MAP = None
