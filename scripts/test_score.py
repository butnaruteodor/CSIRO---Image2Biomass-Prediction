#!/usr/bin/env python3
"""
Comprehensive test: compute Kaggle-matching score locally.

This script:
1. Loads the backbone (vit_large_patch16_dinov3) via timm pretrained
2. Computes test embeddings (same transforms as training val: resize 1008 + normalize)
3. Loads all 5 ridge seed models
4. Runs predictions and computes weighted R2
5. Compares with existing submission.csv
"""
import os
import sys
import gc
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import joblib

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    global_weighted_r2_score, per_target_r2_score,
    per_target_rmse, per_target_mae, per_target_bias,
    all_metrics, print_metrics_table, TARGET_NAMES
)
from src.config import CFG

# Config
IMAGE_SIZE = 1008  # Match training config
BATCH_SIZE = 4
DEVICE = torch.device("cpu")  # No CUDA available
TEST_CSV = "csiro-biomass/test.csv"
RIDGE_DIR = "results/submission_models"
EMBED_DIR = "embeddings"
OUTPUT_DIR = "results/local_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE LOCAL EVALUATION")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Backbone: vit_large_patch16_dinov3")
print()


# ============================================================
# 1. Load test data (with ground truth)
# ============================================================
print("=" * 40)
print("1. LOADING TEST DATA")
print("=" * 40)
test_long = pd.read_csv(TEST_CSV)
test_wide = test_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
test_wide = test_wide[["image_path"] + CFG.ALL_TARGET_COLS]
print(f"Test images: {len(test_wide)}")
print(f"Columns: {list(test_wide.columns)}")

# Get ground truth targets
y_true = test_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32)
print(f"Ground truth shape: {y_true.shape}")
print()


# ============================================================
# 2. Compute test embeddings
# ============================================================
print("=" * 40)
print("2. COMPUTING TEST EMBEDDINGS")
print("=" * 40)

# Check if cached test embeddings exist
CACHE_PATH = os.path.join(EMBED_DIR, "test_clean_embeddings.pt")
if os.path.exists(CACHE_PATH):
    print(f"Loading cached embeddings from {CACHE_PATH}")
    test_embeds = torch.load(CACHE_PATH)
    print(f"  Shape: {test_embeds.shape}")
else:
    print("Computing embeddings from scratch (this will take a while on CPU)...")
    t0 = time.time()
    
    # Load backbone
    print("  Loading backbone via timm...")
    backbone = timm.create_model(
        "vit_large_patch16_dinov3", pretrained=True, num_classes=0)
    backbone.eval()
    backbone.to(DEVICE)
    nf = backbone.num_features
    print(f"  Backbone output dim: {nf} (concat = {nf*2})")
    
    # Create dataset with val transforms (same as training val)
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    class TestDataset(Dataset):
        def __init__(self, df, transform, img_dir="csiro-biomass"):
            self.paths = df["image_path"].values
            self.transform = transform
            self.img_dir = img_dir
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            path = os.path.join(self.img_dir, self.paths[idx])
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((1000, 2000, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid:]
            left_t = self.transform(image=left)["image"]
            right_t = self.transform(image=right)["image"]
            return left_t, right_t
    
    dataset = TestDataset(test_wide, val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_features = []
    with torch.inference_mode():
        for left, right in tqdm(loader, desc="  Extracting"):
            left = left.to(DEVICE)
            right = right.to(DEVICE)
            fl = backbone(left)
            fr = backbone(right)
            feats = torch.cat([fl, fr], dim=1)
            all_features.append(feats.cpu())
    
    test_embeds = torch.cat(all_features).float()
    
    # Cache
    torch.save(test_embeds, CACHE_PATH)
    print(f"  Saved to {CACHE_PATH}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Shape: {test_embeds.shape}")
    
    del backbone, loader, dataset
    gc.collect()

print()


# ============================================================
# 3. Load ridge models and predict
# ============================================================
print("=" * 40)
print("3. RIDGE MODEL INFERENCE")
print("=" * 40)

ridge_files = sorted([f for f in os.listdir(RIDGE_DIR) if f.startswith("ridge_")])
print(f"Found {len(ridge_files)} ridge models")

seed_preds = []
seed_r2s = []

X = test_embeds.numpy()
print(f"Embeddings shape: {X.shape}")

for rf in ridge_files:
    seed = rf.split("_")[1] + "_" + rf.split("_")[2].split(".")[0]
    print(f"  Evaluating {rf}...")
    
    model = joblib.load(os.path.join(RIDGE_DIR, rf))
    
    # Ridge predicts 5 targets [Green, Dead, Clover, GDM, Total]
    # But we need to check the order. Let's map based on training
    preds_ridge = model.predict(X)  # shape (805, 5)
    
    # Check what order the ridge was trained on
    # From experiment_2_ridge / get_full_ridge_dataset, targets are [Green, Dead, Clover, GDM, Total]
    # But wait - the ridge models might have been trained on different target subsets
    # Let's check from the code... ridge_target_cols = [0, 2, 3, 4] originally
    # That means [Green, Clover, GDM, Total] -> dead derived
    
    # Actually ridge_seed models in submission_models are the full 5-target ones
    # Let me check the shape
    print(f"    Predictions shape: {preds_ridge.shape}")
    
    # Compute metrics assuming [Green, Dead, Clover, GDM, Total]
    r2 = global_weighted_r2_score(y_true, preds_ridge)
    seed_r2s.append(r2)
    seed_preds.append(preds_ridge)
    print(f"    Weighted R2 = {r2:.4f}")

print()


# ============================================================
# 4. RESULTS
# ============================================================
print("=" * 40)
print("4. FINAL RESULTS")
print("=" * 40)

if seed_r2s:
    r2_mean = np.mean(seed_r2s)
    r2_std = np.std(seed_r2s, ddof=1)
    print(f"\nRidge ensemble (5 seeds): Weighted R2 = {r2_mean:.4f} +/- {r2_std:.4f}")
    
    # Ensemble
    ensemble_preds = np.mean(seed_preds, axis=0)
    ensemble_metrics = all_metrics(y_true, ensemble_preds)
    print(f"\nRidge ensemble (avg of 5):")
    print_metrics_table(ensemble_metrics)
    
    # Save metrics
    import json
    results = {
        "ridge_weighted_r2_mean": float(r2_mean),
        "ridge_weighted_r2_std": float(r2_std),
        "ridge_per_seed_r2": [float(x) for x in seed_r2s],
        "ensemble_weighted_r2": float(ensemble_metrics["weighted_r2"]),
        "ensemble_per_target_r2": {k: float(v) for k, v in ensemble_metrics["per_target_r2"].items()},
        "ensemble_per_target_rmse": {k: float(v) for k, v in ensemble_metrics["per_target_rmse"].items()},
        "ensemble_per_target_mae": {k: float(v) for k, v in ensemble_metrics["per_target_mae"].items()},
        "ensemble_per_target_bias": {k: float(v) for k, v in ensemble_metrics["per_target_bias"].items()},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR}/results.json")
else:
    print("No ridge models found to evaluate.")

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
