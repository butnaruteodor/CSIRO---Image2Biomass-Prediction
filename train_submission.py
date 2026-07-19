"""
train_submission.py — Train final Ridge submission models for Kaggle

Trains a MultiOutputRegressor(RidgeCV) on ALL public samples, repeated over 5 seeds.
Produces joblib model files for inference on the hidden test set.

Usage:
    python train_submission.py              # After precompute_features.py

Output:
    results/submission_models/
        ridge_seed_13.joblib
        ridge_seed_21.joblib
        ridge_seed_42.joblib
        ridge_seed_87.joblib
        ridge_seed_101.joblib
"""

import os, sys, json, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from joblib import dump as jl_dump
from tqdm import tqdm

# Light project modules (no torch/timm chain needed)
from configs.deterministic import set_seed
from dataset.preprocess_data import get_df

# ============================================================
# CONFIGURATION
# ============================================================
EMBED_DIR = 'embeddings'
OUTPUT_DIR = 'results/submission_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ridge regression hyperparameters (same as experiment_2.py)
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

SEEDS = [13, 21, 42, 87, 101]

# Read embedding dimension from metadata
with open(os.path.join(EMBED_DIR, 'metadata.json')) as f:
    _meta = json.load(f)
FEATURE_DIM = _meta['embedding_dim']
print(f"Embedding dimension: {FEATURE_DIM}")

TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']


def _load_embeddings(embed_dir):
    """Load clean (unaugmented) embeddings and targets."""
    import torch
    targets = torch.load(os.path.join(embed_dir, 'targets.pt'))
    clean_embeddings = torch.load(os.path.join(embed_dir, 'clean_embeddings.pt'))
    return targets, clean_embeddings


# ============================================================
# RIDGE TRAINING (train on ALL samples, no validation)
# ============================================================

def train_ridge_model(all_feats, all_targets, seed, seed_idx):
    """
    Train a MultiOutputRegressor(RidgeCV) on ALL data for one seed.
    
    Args:
        all_feats: numpy array of shape (N, FEATURE_DIM)
        all_targets: numpy array of shape (N, 5)
        seed: random seed
        seed_idx: index (0-based) for display
    
    Returns:
        Trained MultiOutputRegressor model
    """
    set_seed(seed, deterministic=True)
    
    print(f"\n  Training RidgeCV with alphas={RIDGE_ALPHAS}...")
    ridge = MultiOutputRegressor(RidgeCV(alphas=RIDGE_ALPHAS))
    ridge.fit(all_feats, all_targets)
    
    return ridge


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TRAIN SUBMISSION RIDGE MODELS")
    print("=" * 70)

    # 1. Load clean embeddings (no augmentation — Ridge trains on raw features)
    print("\nLoading clean embeddings...")
    targets, clean_embeddings = _load_embeddings(EMBED_DIR)
    all_feats = clean_embeddings.numpy()
    all_targets = targets.numpy()
    print(f"  Features: {all_feats.shape}")
    print(f"  Targets:  {all_targets.shape}")
    print(f"  Target columns: {TARGET_NAMES}")

    # 2. Train one Ridge model per seed
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'─' * 50}")
        print(f"Training seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print(f"{'─' * 50}")

        model = train_ridge_model(all_feats, all_targets, seed, seed_idx)

        save_path = os.path.join(OUTPUT_DIR, f'ridge_seed_{seed}.joblib')
        jl_dump(model, save_path)
        print(f"  ✓ Saved to {save_path}")

        del model
        gc.collect()

    print(f"\n{'=' * 70}")
    print(f"All {len(SEEDS)} Ridge models saved to {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()