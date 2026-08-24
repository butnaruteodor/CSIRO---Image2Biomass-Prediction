No, let me do this differently - run 16 images at a time, checkpointing each chunk.
#!/usr/bin/env python3
"""Compute test embeddings + evaluate ridge models against test ground truth."""
import os, sys, gc, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, timm, joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG
from src.evaluation.metrics import global_weighted_r2_score, all_metrics, print_metrics_table

IMG_SIZE = 224
BATCH_SIZE = 8
DEVICE = torch.device("cpu")
EMBED_DIR = "embeddings"
CACHE_PATH = os.path.join(EMBED_DIR, f"test_clean_embeddings_{IMG_SIZE}.pt")
TARGET_PATH = os.path.join(EMBED_DIR, f"test_targets_{IMG_SIZE}.pt")
os.makedirs(EMBED_DIR, exist_ok=True)

print("=" * 70)
print(f"COMPUTE + EVALUATE (img={IMG_SIZE}, bs={BATCH_SIZE})")
print("=" * 70)
sys.stdout.flush()

# 1
print("\n1. Loading test data...")
test_long = pd.read_csv("csiro-biomass/test.csv")
test_wide = test_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
test_wide = test_wide[["image_path"] + CFG.ALL_TARGET_COLS]
N = len(test_wide)
y_true = test_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32)
print(f"   {N} images")
sys.stdout.flush()

# 2
if os.path.exists(CACHE_PATH) and os.path.exists(TARGET_PATH):
    print(f"\n2. Loading cached embeddings from {CACHE_PATH}")
    test_embeds = torch.load(CACHE_PATH)
    test_targets = torch.load(TARGET_PATH)
else:
    print(f"\n2. Computing embeddings at {IMG_SIZE}x{IMG_SIZE}...")
    t0 = time.time()
    backbone = timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0)
    backbone.eval(); backbone.to(DEVICE)
    print("   Backbone loaded")
    sys.stdout.flush()
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    class TestDS(Dataset):
        def __init__(self, df, transform):
            self.paths = df["image_path"].values
            self.tr = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            path = os.path.join("csiro-biomass", self.paths[idx])
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((1000, 2000, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape; mid = w // 2
            left = self.tr(image=img[:, :mid])["image"]
            right = self.tr(image=img[:, mid:])["image"]
            return left, right
    
    dataset = TestDS(test_wide, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_features = []
    
    for batch_idx, (left, right) in enumerate(loader):
        with torch.inference_mode():
            fl = backbone(left.to(DEVICE))
            fr = backbone(right.to(DEVICE))
            feats = torch.cat([fl, fr], dim=1)
        all_features.append(feats.cpu())
        if (batch_idx+1) % 15 == 0:
            d = (batch_idx+1) * BATCH_SIZE
            print(f"   Batch {batch_idx+1}/{len(loader)} ({d}/{N}) elapsed={time.time()-t0:.0f}s")
            sys.stdout.flush()
    
    test_embeds = torch.cat(all_features).float()
    test_targets = torch.from_numpy(y_true)
    torch.save(test_embeds, CACHE_PATH)
    torch.save(test_targets, TARGET_PATH)
    print(f"   Done in {time.time()-t0:.0f}s")
    del backbone, loader, dataset; gc.collect()
    sys.stdout.flush()

print(f"\n   Embeddings: {test_embeds.shape}")

# 3
print("\n3. Evaluating ridge models...")
RIDGE_DIR = "results/submission_models"
ridge_files = sorted([f for f in os.listdir(RIDGE_DIR) if f.startswith("ridge_")])
print(f"   Found {len(ridge_files)} models")

seed_r2s, all_preds = [], []
X = test_embeds.numpy()
y_true_np = test_targets.numpy()

for rf in ridge_files:
    model = joblib.load(os.path.join(RIDGE_DIR, rf))
    preds = model.predict(X)
    r2 = global_weighted_r2_score(y_true_np, preds)
    seed_r2s.append(r2); all_preds.append(preds)
    print(f"   {rf}: R2 = {r2:.4f}")
    sys.stdout.flush()

# 4
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
r2_mean = float(np.mean(seed_r2s))
r2_std = float(np.std(seed_r2s, ddof=1))
print(f"\nRidge (5 seeds): R2 = {r2_mean:.4f} +/- {r2_std:.4f}")
print(f"  Per seed: {[f'{x:.4f}' for x in seed_r2s]}")

ensemble_preds = np.mean(all_preds, axis=0)
ensemble_metrics = all_metrics(y_true_np, ensemble_preds)
print_metrics_table(ensemble_metrics, "Ridge Ensemble")

OUTPUT_DIR = "results/local_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {
    "config": {"img_size": IMG_SIZE},
    "ridge_wr2_mean": r2_mean, "ridge_wr2_std": r2_std,
    "ridge_per_seed_r2": [float(x) for x in seed_r2s],
    "ensemble": {
        "weighted_r2": float(ensemble_metrics["weighted_r2"]),
        "per_target_r2": {k: float(v) for k,v in ensemble_metrics["per_target_r2"].items()},
        "per_target_rmse": {k: float(v) for k,v in ensemble_metrics["per_target_rmse"].items()},
        "per_target_mae": {k: float(v) for k,v in ensemble_metrics["per_target_mae"].items()},
        "per_target_bias": {k: float(v) for k,v in ensemble_metrics["per_target_bias"].items()},
    },
}
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/results.json")
print("=" * 70)
print("DONE")
print("=" * 70)
