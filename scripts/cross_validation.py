#!/usr/bin/env python3
"""
Cross-validation training for model selection.
Runs 5-fold CV for each seed using a chosen split strategy.
Saves fold predictions and metrics (no table generation).

Usage:
    # Date-location grouped CV (primary protocol) with MLP
    python scripts/cross_validation.py --split date_location --head mlp --seeds 13 21 42 87 101

    # Random stratified CV with Ridge
    python scripts/cross_validation.py --split random --head ridge
"""
import os, sys, gc, json, argparse, copy, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG
from src.deterministic import set_seed, seed_worker, get_generator
from src.data.preprocessing import get_df, get_random_stratified_splits, get_date_grouped_splits, get_date_location_grouped_splits
from src.data.dataset import EmbeddingAugmentationDataset
from src.models.factory import HeadFactory
from src.training.loss import weighted_biomass_loss
from src.evaluation.metrics import global_weighted_r2_score, per_target_r2_score, TARGET_NAMES
from src.training.trainer import train_epoch_mlp, valid_epoch_mlp

EMBED_DIR = "embeddings"
with open(os.path.join(EMBED_DIR, "metadata.json")) as f:
    FEATURE_DIM = json.load(f)["embedding_dim"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training defaults
LR = 1e-3; WD = 1e-2; EPOCHS = 80; PATIENCE = 15
BATCH_SIZE = 8; GRAD_ACC = 1; N_AUG = 15; N_FOLDS = 5

SPLIT_FACTORIES = {
    "random": get_random_stratified_splits,
    "date": get_date_grouped_splits,
    "date_location": get_date_location_grouped_splits,
}

def train_epoch_ridge(model, loader):
    """Sklearn models don't have epochs; collect then fit."""
    pass  # handled in main loop

def cv_seed_mlp(df, splits, seed):
    """Run 5-fold CV for one seed using MLP head."""
    set_seed(seed, deterministic=True)
    n_total = len(df)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}...")

        train_set = EmbeddingAugmentationDataset(train_idx, EMBED_DIR, n_aug=N_AUG, is_train=True)
        val_set = EmbeddingAugmentationDataset(val_idx, EMBED_DIR, n_aug=N_AUG, is_train=False)

        g = get_generator(seed)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True)

        model = HeadFactory.create("mlp", feature_dim=FEATURE_DIM, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

        best_r2 = -np.inf; best_epoch = 0; patience = 0
        best_preds = None; best_targets = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch_mlp(model, train_loader, optimizer, scaler, grad_acc=GRAD_ACC, device=DEVICE)
            val_loss, val_r2, _, preds, targets = valid_epoch_mlp(model, val_loader, device=DEVICE)
            scheduler.step()

            if val_r2 > best_r2:
                best_r2 = val_r2; best_epoch = epoch; patience = 0
                best_preds = preds.copy(); best_targets = targets.copy()
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

        fold_results.append({
            "fold": fold_idx, "best_epoch": best_epoch,
            "weighted_r2": float(best_r2),
            "preds": best_preds.tolist(),
            "targets": best_targets.tolist(),
            "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),
        })
        print(f"    Best epoch {best_epoch}: Weighted R2 = {best_r2:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()

    return fold_results

def cv_seed_ridge(df, splits, seed):
    """Run 5-fold CV for one seed using Ridge head."""
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    import joblib

    set_seed(seed, deterministic=True)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}...")

        train_set = EmbeddingAugmentationDataset(train_idx, EMBED_DIR, n_aug=1, is_train=False)
        val_set = EmbeddingAugmentationDataset(val_idx, EMBED_DIR, n_aug=1, is_train=False)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # Collect features
        X_train, y_train = [], []
        for feats, targets in train_loader:
            X_train.append(feats.numpy()); y_train.append(targets.numpy())
        X_val, y_val = [], []
        for feats, targets in val_loader:
            X_val.append(feats.numpy()); y_val.append(targets.numpy())

        X_train = np.concatenate(X_train); y_train = np.concatenate(y_train)
        X_val = np.concatenate(X_val); y_val = np.concatenate(y_val)

        model = MultiOutputRegressor(Ridge(alpha=1.0)).fit(X_train, y_train)
        preds = model.predict(X_val)
        r2 = global_weighted_r2_score(y_val, preds)

        fold_results.append({
            "fold": fold_idx, "best_epoch": 0,
            "weighted_r2": float(r2),
            "preds": preds.tolist(),
            "targets": y_val.tolist(),
            "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),
        })
        print(f"    Weighted R2 = {r2:.4f}")

    return fold_results

def main():
    parser = argparse.ArgumentParser(description="Cross-validation training")
    parser.add_argument("--split", type=str, default="date_location",
                        choices=["random", "date", "date_location"],
                        help="Split strategy (default: date_location)")
    parser.add_argument("--head", type=str, default="mlp", choices=["mlp", "ridge"],
                        help="Model head (default: mlp)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 101])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    global EPOCHS
    EPOCHS = args.epochs
    split_name = args.split
    strategy_key = {"random": "random_stratified", "date": "date_grouped", "date_location": "date_location_grouped"}[split_name]
    out_dir = args.output_dir or f"results/cv_{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Cross-Validation: {strategy_key} | Head: {args.head}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {out_dir}")
    print()

    df = get_df()
    all_results = {}

    split_fn = SPLIT_FACTORIES[split_name]

    cv_fn = cv_seed_mlp if args.head == "mlp" else cv_seed_ridge

    for seed in args.seeds:
        print(f"\n{'=' * 50}")
        print(f"Seed {seed}")
        print(f"{'=' * 50}")
        splits = split_fn(df, seed)
        fold_results = cv_fn(df, splits, seed)

        # Compute overall OOF R2 for this seed
        all_preds = np.concatenate([np.array(fr["preds"]) for fr in fold_results])
        all_targets = np.concatenate([np.array(fr["targets"]) for fr in fold_results])
        oof_r2 = global_weighted_r2_score(all_targets, all_preds)
        print(f"  OOF Weighted R2 = {oof_r2:.4f}")

        all_results[f"seed_{seed}"] = {
            "fold_results": fold_results,
            "oof_weighted_r2": float(oof_r2),
        }

    # Save results
    output = {
        "strategy": strategy_key,
        "head": args.head,
        "seeds": args.seeds,
        "fold_results_by_seed": all_results,
    }

    torch.save(output, os.path.join(out_dir, "full_results.pt"))
    print(f"\nSaved CV results to {out_dir}/full_results.pt")

    # Summary
    wr2s = [all_results[f"seed_{s}"]["oof_weighted_r2"] for s in args.seeds if f"seed_{s}" in all_results]
    if wr2s:
        print(f"\nSummary: OOF Weighted R2 = {np.mean(wr2s):.4f} ± {np.std(wr2s, ddof=1):.4f}")

if __name__ == "__main__":
    main()
