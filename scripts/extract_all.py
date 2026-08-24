#!/usr/bin/env python3
"""Extract all test embeddings in one go, saving checkpoints."""
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

IMG_SIZE = 224; BATCH_SIZE = 8; DEVICE = torch.device("cpu")
EMBED_DIR = "embeddings"; OUTPUT = os.path.join(EMBED_DIR, "test_embeds_all.npy")
CHECKPOINT = os.path.join(EMBED_DIR, "test_embeds_progress.npy")
TARGETS_OUT = os.path.join(EMBED_DIR, "test_targets.pt")
os.makedirs(EMBED_DIR, exist_ok=True)

# Load test data
print("Loading test data...")
sys.stdout.flush()
test_long = pd.read_csv("csiro-biomass/test.csv")
test_wide = test_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
test_wide = test_wide[["image_path"] + CFG.ALL_TARGET_COLS]
N = len(test_wide)
print(f"  {N} images")

# Save targets
torch.save(torch.from_numpy(test_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32)), TARGETS_OUT)

# Check if already done
if os.path.exists(OUTPUT):
    print(f"Embeddings already exist at {OUTPUT}")
    sys.exit(0)

# Resume from checkpoint?
start_idx = 0
if os.path.exists(CHECKPOINT):
    prev = np.load(CHECKPOINT, allow_pickle=True).item()
    start_idx = prev.get("idx", 0) + 1
    print(f"Resuming from image {start_idx}")

# Init embedding array
embeddings = np.zeros((N, 2048), dtype=np.float32)

# Load backbone only if needed
if start_idx == 0:
    print("Loading backbone...")
    sys.stdout.flush()
backbone = timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0)
backbone.eval(); backbone.to(DEVICE)
print("Backbone loaded")
sys.stdout.flush()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

t0 = time.time()
batch_paths, batch_left, batch_right = [], [], []

def process_batch():
    if not batch_paths: return
    left_t = torch.stack(batch_left).to(DEVICE)
    right_t = torch.stack(batch_right).to(DEVICE)
    with torch.inference_mode():
        fl = backbone(left_t); fr = backbone(right_t)
        feats = torch.cat([fl, fr], dim=1).cpu().numpy()
    for bi, pth in enumerate(batch_paths):
        path_idx = test_wide[test_wide["image_path"] == pth].index[0]
        embeddings[path_idx] = feats[bi]
    batch_paths.clear(); batch_left.clear(); batch_right.clear()

print("Processing images...")
sys.stdout.flush()
for idx in range(start_idx, N):
    row = test_wide.iloc[idx]
    pth = row["image_path"]
    full_path = os.path.join("csiro-biomass", pth)
    img = cv2.imread(full_path)
    if img is None:
        print(f"  Warning: {full_path} not found, using zeros")
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape; mid = w // 2
    left_t = transform(image=img[:, :mid])["image"]
    right_t = transform(image=img[:, mid:])["image"]
    batch_paths.append(pth)
    batch_left.append(left_t)
    batch_right.append(right_t)
    
    if len(batch_paths) >= BATCH_SIZE:
        process_batch()
    
    if (idx + 1) % 40 == 0:
        elapsed = time.time() - t0
        remaining = (elapsed / (idx + 1 - start_idx)) * (N - idx - 1) if idx > start_idx else 0
        print(f"  {idx+1}/{N} ({100*(idx+1)/N:.0f}%) elapsed={elapsed:.0f}s remain={remaining:.0f}s")
        np.save(CHECKPOINT, {"idx": idx, "elapsed": elapsed})
        sys.stdout.flush()

# Final batch
if batch_paths:
    process_batch()

np.save(OUTPUT, embeddings)
elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Saved to {OUTPUT}")

# Cleanup
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
del backbone; gc.collect()
