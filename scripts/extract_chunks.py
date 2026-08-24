#!/usr/bin/env python3
"""Extract ALL test embeddings and evaluate. Runs until completion."""
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

IMG_SIZE = 224; BATCH_SIZE = 8; DEVICE = torch.device("cpu")
EMBED_DIR = "embeddings"; CHUNK_SIZE = 32
os.makedirs(EMBED_DIR, exist_ok=True)

paths_file = os.path.join(EMBED_DIR, "test_paths.csv")
targets_file = os.path.join(EMBED_DIR, "test_targets.pt")
chunks_file = os.path.join(EMBED_DIR, "test_chunks_done.npy")

if not os.path.exists(paths_file):
    print("Preparing test metadata...")
    test_long = pd.read_csv("csiro-biomass/test.csv")
    test_wide = test_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
    test_wide[["image_path"] + CFG.ALL_TARGET_COLS].to_csv(paths_file, index=False)
    targets = torch.from_numpy(test_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32))
    torch.save(targets, targets_file)

N = sum(1 for _ in open(paths_file)) - 1
n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE

if os.path.exists(chunks_file):
    chunks_done = set(np.load(chunks_file).tolist())
else:
    chunks_done = set()

print(f"Total: {N} images, {n_chunks} chunks of {CHUNK_SIZE}, {len(chunks_done)} done")

remaining = [c for c in range(n_chunks) if c not in chunks_done]

if remaining:
    print(f"Processing {len(remaining)} remaining chunks...")
    backbone = timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0)
    backbone.eval(); backbone.to(DEVICE)
    print(f"Backbone loaded")
    sys.stdout.flush()
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    for cid in remaining:
        st = cid * CHUNK_SIZE
        en = min((cid + 1) * CHUNK_SIZE, N)
        t0 = time.time()
        
        df = pd.read_csv(paths_file, skiprows=range(0, st + 1), nrows=en - st, header=None)
        n_imgs = len(df)
        
        class TempDS(Dataset):
            def __init__(self, df, transform):
                self.rows = df
                self.tr = transform
            def __len__(self): return len(self.rows)
            def __getitem__(self, idx):
                p = os.path.join("csiro-biomass", self.rows.iloc[idx][0])
                img = cv2.imread(p)
                if img is None: img = np.zeros((1000, 2000, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape; mid = w // 2
                return self.tr(image=img[:, :mid])["image"], self.tr(image=img[:, mid:])["image"]
        
        ds = TempDS(df, transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        all_feats = []
        for left, right in loader:
            with torch.inference_mode():
                fl = backbone(left.to(DEVICE))
                fr = backbone(right.to(DEVICE))
                all_feats.append(torch.cat([fl, fr], dim=1).cpu().numpy())
        
        chunk_data = np.concatenate(all_feats, axis=0)
        np.save(os.path.join(EMBED_DIR, f"test_embed_chunk_{cid}.npy"), chunk_data)
        chunks_done.add(cid)
        np.save(chunks_file, np.array(list(chunks_done)))
        print(f"  Chunk {cid}/{n_chunks}: {chunk_data.shape} in {time.time()-t0:.0f}s ({len(chunks_done)}/{n_chunks})")
        sys.stdout.flush()
    
    del backbone; gc.collect()
    print("All chunks processed!")

# Evaluate
print("\nRunning evaluation...")
all_embeds = [np.load(os.path.join(EMBED_DIR, f"test_embed_chunk_{c}.npy")) for c in range(n_chunks)]
test_embeds = torch.from_numpy(np.concatenate(all_embeds, axis=0))
test_targets = torch.load(targets_file)
print(f"Embeddings: {test_embeds.shape}")

from src.evaluation.metrics import global_weighted_r2_score, all_metrics, print_metrics_table
RIDGE_DIR = "results/submission_models"
ridge_files = sorted([f for f in os.listdir(RIDGE_DIR) if f.startswith("ridge_")])
print(f"Ridge models: {len(ridge_files)}")

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
