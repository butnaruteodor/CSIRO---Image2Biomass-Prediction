"""
train_submission.py - Train final submission models for Kaggle

Trains BiomassSimpleMLP on ALL 357 public samples for the median best epoch
from CV, repeated over 5 seeds. Produces model checkpoint files for later
inference on the hidden test set.

Usage:
    python experiments/train_submission.py

Output:
    results/submission_models/
        seed_13_final.pt
        seed_21_final.pt
        seed_42_final.pt
        seed_87_final.pt
        seed_101_final.pt
"""

import os
import sys
import json
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.deterministic import set_seed, seed_worker, get_generator
from src.data.preprocessing import get_df, EmbeddingAugmentationDataset
from src.models.heads import BiomassSimpleMLP
from src.training.loss import weighted_biomass_loss
from src.training.trainer import train_epoch_mlp

# ============================================================
# CONFIGURATION
# ============================================================
EMBED_DIR = "embeddings"
RESULTS_DIR = "results/experiment_2"
OUTPUT_DIR = "results/submission_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LR = 1e-3
WD = 1e-2
EPOCHS = 80
BATCH_SIZE = 8
GRAD_ACC = 1
N_AUG = 15

SEEDS = [13, 21, 42, 87, 101]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(EMBED_DIR, "metadata.json")) as f:
    _meta = json.load(f)
FEATURE_DIM = _meta["embedding_dim"]
print(f"Embedding dimension: {FEATURE_DIM}")
print(f"Device: {DEVICE}")


def get_median_best_epoch(results_path):
    """Load median best epoch from CV results."""
    if not os.path.exists(results_path):
        print(f"WARNING: {results_path} not found. Using default EPOCHS={EPOCHS}.")
        return EPOCHS

    data = torch.load(results_path, map_location="cpu", weights_only=False)

    proto = "random_stratified"
    if proto not in data:
        for key in data:
            if "fold_results" in data[key]:
                proto = key
                break

    fold_results = data[proto]["fold_results"]
    epochs = [r["best_epoch"] for r in fold_results]
    median_epoch = int(np.median(epochs))

    print(f"  Loaded {len(epochs)} fold results from '{proto}'")
    print(f"  Best epochs - min={min(epochs)}, max={max(epochs)}, "
          f"median={median_epoch}, mean={np.mean(epochs):.1f}")

    return max(1, median_epoch)


def train_final_model(all_indices, seed):
    """Train a single final model on all samples."""
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
        optimizer, T_max=N_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == 'cuda' else None

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch_mlp(
            model, loader, optimizer, scaler, grad_acc=GRAD_ACC, device=DEVICE)
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == N_EPOCHS:
            print(f"  Epoch {epoch:02d}/{N_EPOCHS} | Loss {train_loss:.5f}")

    del loader, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return model


def main():
    global N_EPOCHS
    print("=" * 70)
    print("TRAIN SUBMISSION MODELS")
    print("=" * 70)

    # 1. Get median best epoch from CV
    cv_results_path = os.path.join(RESULTS_DIR, "full_results.pt")
    N_EPOCHS = get_median_best_epoch(cv_results_path)
    print(f"Training for {N_EPOCHS} epochs (median best epoch from CV)")

    # 2. Get all sample indices
    df = get_df()
    all_indices = np.arange(len(df))
    print(f"Training on all {len(all_indices)} samples")

    # 3. Train one model per seed
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'─' * 50}")
        print(f"Training seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print(f"{'─' * 50}")

        model = train_final_model(all_indices, seed)

        save_path = os.path.join(OUTPUT_DIR, f"seed_{seed}_final.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  Saved to {save_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"All {len(SEEDS)} submission models saved to {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    N_EPOCHS = EPOCHS  # Default, will be overwritten by main()
    main()
