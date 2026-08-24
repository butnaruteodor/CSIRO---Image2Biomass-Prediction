#!/usr/bin/env python3
"""Train ridge on 224px train embeddings and evaluate on 224px test embeddings."""
import warnings, sys, os, json, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG
from src.evaluation.metrics import global_weighted_r2_score, all_metrics, print_metrics_table

IMG_SIZE = 224; BATCH_SIZE = 8; DEVICE = torch.device("cpu")

# Load train data
train_long = pd.read_csv("csiro-biomass/train.csv")
train_wide = train_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
train_wide = train_wide[["image_path"] + CFG.ALL_TARGET_COLS]
print(f"Train images: {len(train_wide)}")

# Load backbone
backbone = timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0)
backbone.eval(); backbone.to(DEVICE)
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class SimpleDS(Dataset):
    def __init__(self, df, transform):
        self.rows = df; self.tr = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        p = os.path.join("csiro-biomass", self.rows.iloc[idx]["image_path"])
        img = cv2.imread(p)
        if img is None: img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape; mid = w // 2
        return self.tr(image=img[:, :mid])["image"], self.tr(image=img[:, mid:])["image"]

print("Computing train embeddings at 224px...")
t0 = time.time()
ds = SimpleDS(train_wide, transform)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_feats = []
for left, right in loader:
    with torch.inference_mode():
        fl = backbone(left.to(DEVICE)); fr = backbone(right.to(DEVICE))
        train_feats.append(torch.cat([fl, fr], dim=1).cpu().numpy())
train_embeds = np.concatenate(train_feats, axis=0)
train_targets = train_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32)
print(f"Train embeddings: {train_embeds.shape}, time: {time.time()-t0:.0f}s")

# Train ridge
print("Training ridge...")
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
ridge = MultiOutputRegressor(Ridge(alpha=1.0)).fit(train_embeds, train_targets)

# Load test embeddings (already computed at 224px)
test_embeds = np.load("embeddings/test_embeds_simple.npy")
test_targets = torch.load("embeddings/test_targets.pt").numpy()

# Predict and evaluate
preds = ridge.predict(test_embeds)
r2 = global_weighted_r2_score(test_targets, preds)
print(f"\nRidge (224px train -> 224px test): R2 = {r2:.4f}")
em = all_metrics(test_targets, preds)
print_metrics_table(em, "224px Ridge (224px train -> 224px test)")

# Save results
OUTPUT_DIR = "results/local_evaluation"; os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "results_224px.json"), "w") as f:
    json.dump({
        "ridge_wr2": float(r2),
        "ensemble": {
            "weighted_r2": float(em["weighted_r2"]),
            "per_target_r2": {k:float(v) for k,v in em["per_target_r2"].items()},
            "per_target_rmse": {k:float(v) for k,v in em["per_target_rmse"].items()},
            "per_target_mae": {k:float(v) for k,v in em["per_target_mae"].items()},
            "per_target_bias": {k:float(v) for k,v in em["per_target_bias"].items()},
        }
    }, f, indent=2)
print(f"Saved to {OUTPUT_DIR}/results_224px.json")