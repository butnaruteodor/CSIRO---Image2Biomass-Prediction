#!/usr/bin/env python3
"""Simple extraction: process images individually, collect results, then evaluate."""
import os, sys, gc, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, timm, joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG
from src.evaluation.metrics import global_weighted_r2_score, all_metrics, print_metrics_table

IMG_SIZE = 224; DEVICE = torch.device("cpu")
EMBED_DIR = "embeddings"; CACHE_FILE = os.path.join(EMBED_DIR, "test_embeds_simple.npy")
TARGETS_FILE = os.path.join(EMBED_DIR, "test_targets.pt")
PROGRESS_FILE = os.path.join(EMBED_DIR, "test_progress.txt")
os.makedirs(EMBED_DIR, exist_ok=True)

# Load test data
print("Loading test data..."); sys.stdout.flush()
test_long = pd.read_csv("csiro-biomass/test.csv")
test_wide = test_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
test_wide = test_wide[["image_path"] + CFG.ALL_TARGET_COLS]
N = len(test_wide)
print(f"  {N} images"); sys.stdout.flush()
torch.save(torch.from_numpy(test_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32)), TARGETS_FILE)

if os.path.exists(CACHE_FILE):
    print(f"Already done: {CACHE_FILE}")
    sys.exit(0)

# Resume?
start_idx = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        start_idx = int(f.read().strip())
    print(f"Resuming from image {start_idx}")
    sys.stdout.flush()

# Load previously saved features
if start_idx > 0:
    all_feats = [np.load(CACHE_FILE + ".part")]
else:
    all_feats = []

# Load backbone
print("Loading backbone..."); sys.stdout.flush()
backbone = timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0)
backbone.eval(); backbone.to(DEVICE)
print("Backbone loaded"); sys.stdout.flush()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

t0 = time.time()
batch_embeds = []

for idx in range(start_idx, N):
    row = test_wide.iloc[idx]
    pth = row["image_path"]
    full_path = os.path.join("csiro-biomass", pth)
    img = cv2.imread(full_path)
    if img is None:
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape; mid = w // 2
    left = transform(image=img[:, :mid])["image"].unsqueeze(0)
    right = transform(image=img[:, mid:])["image"].unsqueeze(0)
    
    with torch.inference_mode():
        fl = backbone(left.to(DEVICE))
        fr = backbone(right.to(DEVICE))
        feats = torch.cat([fl, fr], dim=1).cpu().numpy()
    batch_embeds.append(feats[0])
    
    if (idx + 1) % 40 == 0 or idx == N - 1:
        all_feats.append(np.array(batch_embeds))
        batch_embeds = []
        np.save(CACHE_FILE + ".part", np.concatenate(all_feats, axis=0))
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(idx))
        elapsed = time.time() - t0
        remaining = (elapsed / (idx + 1 - start_idx)) * (N - idx - 1) if idx > start_idx else 0
        print(f"  {idx+1}/{N} ({100*(idx+1)/N:.0f}%) elapsed={elapsed:.0f}s remain={remaining:.0f}s")
        sys.stdout.flush()

# Finalize
del backbone; gc.collect()
final = np.concatenate(all_feats, axis=0)
np.save(CACHE_FILE, final)
if os.path.exists(CACHE_FILE + ".part"):
    os.remove(CACHE_FILE + ".part")
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
print(f"Done in {time.time()-t0:.0f}s")
print(f"Saved to {CACHE_FILE}")

# Now evaluate
print("\nEvaluating ridge models..."); sys.stdout.flush()
test_embeds = torch.from_numpy(final)
test_targets = torch.load(TARGETS_FILE)

RIDGE_DIR = "results/submission_models"
ridge_files = sorted([f for f in os.listdir(RIDGE_DIR) if f.startswith("ridge_")])
print(f"Found {len(ridge_files)} ridge models")

seed_r2s, all_preds = [], []
X, yt = test_embeds.numpy(), test_targets.numpy()
for rf in ridge_files:
    m = joblib.load(os.path.join(RIDGE_DIR, rf)); p = m.predict(X)
    r2 = global_weighted_r2_score(yt, p); seed_r2s.append(r2); all_preds.append(p)
    print(f"  {rf}: R2 = {r2:.4f}")
    sys.stdout.flush()

r2m = float(np.mean(seed_r2s)); r2s = float(np.std(seed_r2s, ddof=1))
print(f"\nRidge (5 seeds): R2 = {r2m:.4f} +/- {r2s:.4f}")
print(f"  Per seed: {[f'{x:.4f}' for x in seed_r2s]}")
em = all_metrics(yt, np.mean(all_preds, axis=0))
print_metrics_table(em, "Ridge Ensemble on Test Set")

OUTPUT_DIR = "results/local_evaluation"; os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump({"ridge_wr2_mean": r2m, "ridge_wr2_std": r2s,
        "ridge_per_seed_r2": [float(x) for x in seed_r2s],
        "ensemble": {"weighted_r2": float(em["weighted_r2"]),
            "per_target_r2": {k:float(v) for k,v in em["per_target_r2"].items()},
            "per_target_rmse": {k:float(v) for k,v in em["per_target_rmse"].items()},
            "per_target_mae": {k:float(v) for k,v in em["per_target_mae"].items()},
            "per_target_bias": {k:float(v) for k,v in em["per_target_bias"].items()}}}, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/results.json")
print("DONE")
