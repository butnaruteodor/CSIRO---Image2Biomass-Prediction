#!/usr/bin/env python3
"""
Unified embedding extraction for train and test images.

Usage:
    python scripts/extract_embeddings.py --mode train
    python scripts/extract_embeddings.py --mode test
"""
import os, sys, gc, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def _no_collate(batch):
    """Do not collate — return the single sample as-is (batch_size must be 1)."""
    return batch[0]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG
from src.data.augmentations import get_val_transforms, get_spatial_transforms, get_photometric_transforms, get_tta_transforms
from src.models.backbone import BackboneModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8; NUM_WORKERS = 4; N_AUG = 15; N_TTA = 5
EMBED_DIR = "embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

class TrainEmbeddingDataset(Dataset):
    """Dataset yielding both val-transform and aug-transform versions."""
    def __init__(self, df, img_dir, img_size=1008, seed=42):
        self.df = df; self.img_dir = img_dir
        self.paths = df["image_path"].values
        self.targets = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
        self.val_transform = get_val_transforms(size=img_size, seed=seed)
        self.spatial_transform = get_spatial_transforms(size=img_size, seed=seed)
        self.photo_transform = get_photometric_transforms(size=img_size, seed=seed)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None: img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape; mid = w // 2
        left = img[:, :mid]; right = img[:, mid:]
        left_val = self.val_transform(image=left)["image"]
        right_val = self.val_transform(image=right)["image"]
        if self.spatial_transform:
            left_spat = self.spatial_transform(image=left)["image"]
            right_spat = self.spatial_transform(image=right)["image"]
        else:
            left_spat = left.copy(); right_spat = right.copy()
        left_aug = self.photo_transform(image=left_spat)["image"]
        right_aug = self.photo_transform(image=right_spat)["image"]
        target = torch.from_numpy(self.targets[idx])
        return left_val, right_val, left_aug, right_aug, target

class TestEmbeddingDataset(Dataset):
    """Dataset for test images with TTA transforms. Supports resume offset."""
    def __init__(self, df_wide, img_dir, img_size=1008, start_idx=0):
        self.df = df_wide; self.img_dir = img_dir
        self.paths = df_wide["image_path"].values
        self.tta_transforms = get_tta_transforms(img_size=img_size)
        self.start_idx = start_idx
    def __len__(self): return len(self.df) - self.start_idx
    def __getitem__(self, idx):
        idx_actual = idx + self.start_idx
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx_actual]))
        image = cv2.imread(str(path))
        if image is None: image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape; mid = w // 2
        img_left = image[:, :mid]; img_right = image[:, mid:]
        tta_results = [(t(image=img_left)["image"], t(image=img_right)["image"]) for t in self.tta_transforms]
        return tta_results, idx

def extract_train_embeddings():
    print("=" * 70)
    print("TRAIN EMBEDDING EXTRACTION")
    print("=" * 70)
    clean_path = os.path.join(EMBED_DIR, "clean_embeddings.pt")
    aug_path = os.path.join(EMBED_DIR, "aug_embeddings.pt")
    targets_path = os.path.join(EMBED_DIR, "targets.pt")
    if all(os.path.exists(p) for p in [clean_path, aug_path, targets_path]):
        print(f"All train embeddings already exist in {EMBED_DIR}/")
        return
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df = df_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
    df = df[["image_path"] + CFG.ALL_TARGET_COLS]
    N = len(df); print(f"Loading {N} training images...")
    print("Loading backbone...")
    backbone = BackboneModel(CFG.MODEL_NAME, pretrained=True)
    backbone.eval(); backbone.to(DEVICE)
    feature_dim = backbone.feature_dim; print(f"  Feature dimension: {feature_dim}")
    dataset = TrainEmbeddingDataset(df, CFG.TRAIN_IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    clean_embeds = torch.zeros(N, feature_dim, dtype=torch.float32)
    aug_embeds = torch.zeros(N, N_AUG, feature_dim, dtype=torch.float32)
    targets = torch.zeros(N, 5, dtype=torch.float32)
    t0 = time.time()
    print(f"Extracting embeddings ({N} images, {N_AUG} aug copies)...")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="extract train")):
            left_val, right_val, left_aug, right_aug, batch_targets = batch
            bs = left_val.shape[0]; start = batch_idx * BATCH_SIZE; end = start + bs
            left_val = left_val.to(DEVICE, non_blocking=True)
            right_val = right_val.to(DEVICE, non_blocking=True)
            clean_embeds[start:end] = backbone(left_val, right_val).cpu().float()
            targets[start:end] = batch_targets
            left_aug = left_aug.to(DEVICE, non_blocking=True)
            right_aug = right_aug.to(DEVICE, non_blocking=True)
            aug_embeds[start:end, 0] = backbone(left_aug, right_aug).cpu().float()
            for aug_idx in range(1, N_AUG):
                batch_indices = list(range(start, min(end, N)))
                batch_aug_feats = []
                for sample_idx in batch_indices:
                    _, _, al, ar, _ = dataset[sample_idx]
                    al = al.unsqueeze(0).to(DEVICE); ar = ar.unsqueeze(0).to(DEVICE)
                    batch_aug_feats.append(backbone(al, ar).cpu())
                stack = torch.cat(batch_aug_feats, dim=0)
                aug_embeds[start:end, aug_idx] = stack.float()
            if (batch_idx + 1) % 10 == 0:
                torch.save(clean_embeds[:end], clean_path + ".tmp")
                torch.save(aug_embeds[:end], aug_path + ".tmp")
                torch.save(targets[:end], targets_path + ".tmp")
                print(f"  Batch {batch_idx+1}/{len(loader)} ({end}/{N}) | {time.time()-t0:.0f}s")
    torch.save(clean_embeds, clean_path); torch.save(aug_embeds, aug_path); torch.save(targets, targets_path)
    for f in [clean_path+".tmp", aug_path+".tmp", targets_path+".tmp"]:
        if os.path.exists(f): os.remove(f)
    with open(os.path.join(EMBED_DIR, "metadata.json"), "w") as f:
        json.dump({"embedding_dim": feature_dim, "n_images": N, "n_aug": N_AUG}, f)
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"  clean_embeddings.pt: {clean_embeds.shape}")
    print(f"  aug_embeddings.pt:   {aug_embeds.shape}")
    del backbone; gc.collect(); torch.cuda.empty_cache()

def extract_test_embeddings():
    print("=" * 70)
    print("TEST EMBEDDING EXTRACTION")
    print("=" * 70)
    clean_path = os.path.join(EMBED_DIR, "test_clean_embeddings.pt")
    tta_path = os.path.join(EMBED_DIR, "test_tta_embeddings.pt")
    targets_path = os.path.join(EMBED_DIR, "test_targets.pt")
    ids_path = os.path.join(EMBED_DIR, "test_image_ids.csv")
    checkpoint_file = os.path.join(EMBED_DIR, "test_checkpoint.txt")

    # Resume from checkpoint if available
    resume_from = 0
    if all(os.path.exists(p) for p in [clean_path, tta_path, targets_path, ids_path]) and os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            resume_from = int(f.read().strip())
        if resume_from > 0:
            print(f"Resuming from checkpoint: {resume_from} images already processed")
    elif all(os.path.exists(p) for p in [clean_path, tta_path, targets_path, ids_path]):
        print(f"All test embeddings already exist in {EMBED_DIR}/")
        return
    test_path = os.path.join("csiro-biomass", "test.csv")
    df_long = pd.read_csv(test_path)
    df_wide = df_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
    df_wide = df_wide[["image_path"] + CFG.ALL_TARGET_COLS]
    N = len(df_wide); print(f"Loading {N} test images...")
    df_wide[["image_path"]].to_csv(ids_path, index=False)
    targets = torch.from_numpy(df_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32))
    torch.save(targets, targets_path); print(f"  Saved targets: {targets.shape}")
    print("Loading backbone...")
    backbone = BackboneModel(CFG.MODEL_NAME, pretrained=True)
    backbone.eval(); backbone.to(DEVICE)
    # Speed optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    feature_dim = backbone.feature_dim; print(f"  Feature dimension: {feature_dim}")
    img_dir = os.path.join("csiro-biomass", "test")
    dataset = TestEmbeddingDataset(df_wide, img_dir, start_idx=resume_from)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_no_collate)
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)

    clean_embeds = torch.zeros(N, feature_dim, dtype=torch.float32)
    tta_embeds = torch.zeros(N * N_TTA, feature_dim, dtype=torch.float32)
    # Load existing checkpoint tensors if resuming
    if resume_from > 0 and os.path.exists(clean_path) and os.path.exists(tta_path):
        clean_embeds = torch.load(clean_path)
        tta_embeds = torch.load(tta_path)
        print(f"  Loaded checkpoint: clean={clean_embeds.shape}, tta={tta_embeds.shape}")
    t0 = time.time()
    print(f"Extracting TTA embeddings ({N} images, {N_TTA} views each, starting at {resume_from})...")
    with torch.inference_mode(), autocast_ctx:
        for idx, (tta_results, _) in enumerate(tqdm(loader, desc="extract test")):
            actual_idx = idx + resume_from
            for view_idx, (left, right) in enumerate(tta_results):
                left = left.to(DEVICE).unsqueeze(0)
                right = right.to(DEVICE).unsqueeze(0)
                feats = backbone(left, right).cpu().float().squeeze(0)
                if view_idx == 0:
                    clean_embeds[actual_idx] = feats
                tta_embeds[actual_idx * N_TTA + view_idx] = feats
            # Periodic checkpoint — safe to interrupt after a checkpoint
            if (actual_idx + 1) % 50 == 0 or (actual_idx + 1) == N:
                torch.save(clean_embeds, clean_path)
                torch.save(tta_embeds, tta_path)
                with open(checkpoint_file, 'w') as f:
                    f.write(str(actual_idx + 1))
                torch.cuda.empty_cache()
                print(f"  CHECKPOINT: {actual_idx+1}/{N} | {time.time()-t0:.0f}s")
    torch.save(clean_embeds, clean_path); torch.save(tta_embeds, tta_path)
    print(f"\nSaved: test_clean_embeddings.pt: {clean_embeds.shape}")
    print(f"       test_tta_embeddings.pt:   {tta_embeds.shape}")
    del backbone; gc.collect(); torch.cuda.empty_cache()
    print(f"Done in {time.time()-t0:.0f}s")

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings for train or test")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--img-size", type=int, default=1008)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-aug", type=int, default=15)
    args = parser.parse_args()
    global BATCH_SIZE, N_AUG
    BATCH_SIZE, N_AUG = args.batch_size, args.n_aug
    if args.mode == "train": extract_train_embeddings()
    else: extract_test_embeddings()

if __name__ == "__main__":
    main()
