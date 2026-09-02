#!/usr/bin/env python3
"""
Leave-One-Period-Out (LOPO) temporal analysis.

Divides the 2015 public data into three consecutive, sample-balanced
temporal periods (early / middle / late) and holds out each period in turn
as the test set. Supports the MLP and Ridge heads.

Usage:
    # Ridge head (deterministic)
    python scripts/lopo_cv.py --head ridge

    # MLP head with the five standard seeds
    python scripts/lopo_cv.py --head mlp --seeds 13 21 42 87 101 --epochs 80
"""
import os, sys, gc, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deterministic import set_seed, seed_worker, get_generator
from src.data.preprocessing import get_df, get_lopo_splits
from src.data.dataset import EmbeddingAugmentationDataset
from src.models.factory import HeadFactory
from src.evaluation.metrics import (
    global_weighted_r2_score, per_target_r2_score,
    per_target_rmse, per_target_mae, per_target_bias, TARGET_NAMES
)
from src.training.trainer import train_epoch_mlp, valid_epoch_mlp

EMBED_DIR = "embeddings"
with open(os.path.join(EMBED_DIR, "metadata.json")) as f:
    FEATURE_DIM = json.load(f)["embedding_dim"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
N_AUG = 15

PERIOD_NAMES = ["early", "middle", "late"]


def _std(values):
    """Sample std; 0.0 for a single value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _collect(loader):
    """Collect features/targets from an embedding dataset (sklearn heads)."""
    X, y = [], []
    for feats, targets in loader:
        X.append(feats.numpy())
        y.append(targets.numpy())
    return np.concatenate(X), np.concatenate(y)


def run_period_mlp(train_idx, test_idx, seed, epochs):
    """Train the MLP on the remaining periods, predict the held-out period."""
    set_seed(seed, deterministic=True)

    train_set = EmbeddingAugmentationDataset(train_idx, EMBED_DIR, n_aug=N_AUG, is_train=True)
    test_set = EmbeddingAugmentationDataset(test_idx, EMBED_DIR, n_aug=N_AUG, is_train=False)

    g = get_generator(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = HeadFactory.create("mlp", feature_dim=FEATURE_DIM, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for epoch in range(1, epochs + 1):
        train_epoch_mlp(model, train_loader, optimizer, scaler, grad_acc=1, device=DEVICE)
        scheduler.step()

    _, _, _, preds, targets = valid_epoch_mlp(model, test_loader, device=DEVICE)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds, targets


def run_period_ridge(train_idx, test_idx):
    """Fit RidgeCV on the remaining periods, predict the held-out period."""
    train_set = EmbeddingAugmentationDataset(train_idx, EMBED_DIR, n_aug=1, is_train=False)
    test_set = EmbeddingAugmentationDataset(test_idx, EMBED_DIR, n_aug=1, is_train=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    X_train, y_train = _collect(train_loader)
    X_test, y_test = _collect(test_loader)

    model = HeadFactory.create("ridge")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    alphas = [float(est.alpha_) for est in model.estimators_]
    return preds, y_test, alphas


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Period-Out temporal analysis")
    parser.add_argument("--head", type=str, default="ridge", choices=["mlp", "ridge"],
                        help="Model head (default: ridge)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 101],
                        help="Random seeds (default: 13 21 42 87 101)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs for the MLP head (default: 80)")
    parser.add_argument("--year", type=int, default=2015,
                        help="Year used for the LOPO periods (default: 2015)")
    parser.add_argument("--output-dir", type=str, default="results/lopo")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"LOPO analysis | Head: {args.head} | Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")
    print()

    df = get_df()
    splits, period_info = get_lopo_splits(df, year=args.year)

    all_results = {"strategy": "leave_one_period_out", "head": args.head,
                   "seeds": args.seeds, "epochs": args.epochs,
                   "year": args.year, "period_info": period_info,
                   "period_results": {}}

    for period, (train_idx, test_idx) in zip(PERIOD_NAMES, splits):
        print(f"\n{'=' * 50}")
        print(f"Held-out period: {period}")
        print(f"{'=' * 50}")

        period_runs = []
        for seed in args.seeds:
            print(f"  Seed {seed}...")
            if args.head == "mlp":
                preds, targets = run_period_mlp(train_idx, test_idx, seed, args.epochs)
                selected_alphas = None
            else:
                preds, targets, selected_alphas = run_period_ridge(train_idx, test_idx)

            wr2 = global_weighted_r2_score(targets, preds)
            print(f"    Weighted R2 = {wr2:.4f}")

            period_runs.append({
                "seed": seed,
                "weighted_r2": float(wr2),
                "per_target_r2": per_target_r2_score(targets, preds),
                "per_target_rmse": per_target_rmse(targets, preds),
                "per_target_mae": per_target_mae(targets, preds),
                "per_target_bias": per_target_bias(targets, preds),
                "selected_alphas": selected_alphas,
                "preds": preds.tolist(),
                "targets": targets.tolist(),
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            })

        wr2s = [r["weighted_r2"] for r in period_runs]
        rmse_total = [r["per_target_rmse"]["Dry_Total_g"] for r in period_runs]
        mae_total = [r["per_target_mae"]["Dry_Total_g"] for r in period_runs]
        bias_total = [r["per_target_bias"]["Dry_Total_g"] for r in period_runs]
        print(f"\n  {period}: Weighted R2 = {np.mean(wr2s):.4f} +/- {_std(wr2s):.4f} | "
              f"RMSE(Total) = {np.mean(rmse_total):.2f} | MAE(Total) = {np.mean(mae_total):.2f} | "
              f"Bias(Total) = {np.mean(bias_total):.3f}")

        all_results["period_results"][period] = period_runs

    out_path = os.path.join(args.output_dir, "full_results.pt")
    torch.save(all_results, out_path)
    print(f"\nSaved LOPO results to {out_path}")

    # Overall summary across periods
    print(f"\n{'Period':<10} {'R2_w':>14} {'RMSE (Total)':>14} {'MAE (Total)':>14} {'Bias (Total)':>14}")
    print("-" * 70)
    for period in PERIOD_NAMES:
        runs = all_results["period_results"][period]
        wr2s = [r["weighted_r2"] for r in runs]
        rmse = [r["per_target_rmse"]["Dry_Total_g"] for r in runs]
        mae = [r["per_target_mae"]["Dry_Total_g"] for r in runs]
        bias = [r["per_target_bias"]["Dry_Total_g"] for r in runs]
        print(f"{period:<10} {np.mean(wr2s):>7.3f} +/- {_std(wr2s):.3f} "
              f"{np.mean(rmse):>8.2f} {np.mean(mae):>14.2f} {np.mean(bias):>14.3f}")


if __name__ == "__main__":
    main()
