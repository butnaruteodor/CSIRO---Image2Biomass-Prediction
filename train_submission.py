"""
train_submission.py — Train final submission models for Kaggle

Trains BiomassSimpleMLP on ALL 357 public samples for the median best epoch
from CV, repeated over 5 seeds. Produces model checkpoint files for later
inference on the hidden test set.

Usage:
    python train_submission.py              # After precompute_features.py and experiment_2.py

Output:
    results/submission_models/
        seed_13_final.pt
        seed_21_final.pt
        seed_42_final.pt
        seed_87_final.pt
        seed_101_final.pt

Reference: CSIRO PDF Section 2
    "train the selected model on all 357 public samples using the median best
     epoch from CV, repeated over 5 seeds"
"""

import os, sys, json, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.models import *
from dataset.biomass_dataset import *
from dataset.preprocess_data import *

# Project modules that DON'T import timm
from configs.cfg import CFG
from configs.deterministic import set_seed, seed_worker, get_generator
from utils.eval import weighted_biomass_loss

# ============================================================
# CONFIGURATION
# ============================================================
EMBED_DIR = 'embeddings'
RESULTS_DIR = 'results/experiment_2'
OUTPUT_DIR = 'results/submission_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LR = 1e-3
WD = 1e-2
EPOCHS = 80
WARMUP_EPOCHS = 5
BATCH_SIZE = 8
GRAD_ACC = 1
N_AUG = 15

SEEDS = [13, 21, 42, 87, 101]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(EMBED_DIR, 'metadata.json')) as f:
    _meta = json.load(f)
FEATURE_DIM = _meta['embedding_dim']
print(f"Embedding dimension: {FEATURE_DIM}")
print(f"Device: {DEVICE}")


# ============================================================
# LOAD MEDIAN BEST EPOCH FROM CV RESULTS
# ============================================================

def get_median_best_epoch(results_path):
    if not os.path.exists(results_path):
        print(f"WARNING: {results_path} not found. Using default EPOCHS={EPOCHS}.")
        return EPOCHS

    data = torch.load(results_path, map_location='cpu', weights_only=False)

    proto = 'date_location_grouped'
    if proto not in data:
        for key in data:
            if 'fold_results' in data[key]:
                proto = key
                break

    fold_results = data[proto]['fold_results']
    epochs = [r['best_epoch'] for r in fold_results]
    median_epoch = int(np.median(epochs))

    print(f"  Loaded {len(epochs)} fold results from '{proto}'")
    print(f"  Best epochs — min={min(epochs)}, max={max(epochs)}, "
          f"median={median_epoch}, mean={np.mean(epochs):.1f}")

    return median_epoch


# ============================================================
# TRAINING (no validation — train on all samples)
# ============================================================

def train_epoch_mlp(model, loader, optimizer, scaler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (feats, targets) in enumerate(tqdm(loader, desc='train', leave=False)):
        feats = feats.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with autocast('cuda', dtype=torch.bfloat16):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(
                p_total, p_gdm, p_green, p_clover, p_dead, targets)

        loss = loss / GRAD_ACC
        scaler.scale(loss).backward()
        running_loss += loss.item() * feats.size(0) * GRAD_ACC

        if (i + 1) % GRAD_ACC == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return running_loss / len(loader.dataset)


def train_final_model(all_indices, seed):
    set_seed(seed, deterministic=True)

    dataset = EmbeddingAugmentationDataset(
        all_indices, EMBED_DIR, n_aug=N_AUG, is_train=True)

    g = get_generator(seed)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=g)

    model = BiomassSimpleMLP(FEATURE_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch_mlp(model, loader, optimizer, scaler)
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == N_EPOCHS:
            print(f"  Epoch {epoch:02d}/{N_EPOCHS} | Loss {train_loss:.5f}")

    del loader, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    global N_EPOCHS
    print("=" * 70)
    print("TRAIN SUBMISSION MODELS")
    print("=" * 70)

    # 1. Get median best epoch from CV
    cv_results_path = os.path.join(RESULTS_DIR, 'full_results.pt')
    N_EPOCHS = get_median_best_epoch(cv_results_path)
    print(f"Training for {N_EPOCHS} epochs (median best epoch from CV)")

    # 2. Get all sample indices
    df = get_df()
    all_indices = np.arange(len(df))
    print(f"Training on all {len(all_indices)} samples\n")

    # 3. Train one model per seed
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'─' * 50}")
        print(f"Training seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print(f"{'─' * 50}")

        model = train_final_model(all_indices, seed)

        save_path = os.path.join(OUTPUT_DIR, f'seed_{seed}_final.pt')
        torch.save(model.state_dict(), save_path)
        print(f"  ✓ Saved to {save_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"All {len(SEEDS)} submission models saved to {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()