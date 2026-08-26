#!/usr/bin/env python3
"""
Flexible full-data training on ALL training samples.
Supports different model heads via --head flag.

Usage:
    # Train MLP head (default)
    python scripts/train_model.py --head mlp --seeds 13 21 42 87 101

    # Train Ridge head
    python scripts/train_model.py --head ridge --seeds 13 21 42 87 101

    # Custom output dir
    python scripts/train_model.py --head mlp --output-dir out
"""
import os, sys, gc, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CFG
from src.deterministic import set_seed, seed_worker, get_generator
from src.data.preprocessing import get_df, EmbeddingAugmentationDataset
from src.models.factory import HeadFactory
from src.evaluation.metrics import global_weighted_r2_score, print_metrics_table, all_metrics

EMBED_DIR = "embeddings"
with open(os.path.join(EMBED_DIR, "metadata.json")) as f:
    FEATURE_DIM = json.load(f)["embedding_dim"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Training functions
# ============================================================================

def train_epoch_mlp(model, loader, optimizer, scaler, grad_acc=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (feats, targets) in enumerate(tqdm(loader, desc="train", leave=False)):
        feats = feats.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        from src.training.loss import weighted_biomass_loss
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else torch.amp.autocast("cpu"):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, targets)
        loss = loss / grad_acc
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.item() * feats.size(0) * grad_acc
        if (i + 1) % grad_acc == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
    return running_loss / len(loader.dataset)

def train_mlp(loader, epochs=80, lr=1e-3, wd=1e-2, seed=42):
    set_seed(seed, deterministic=True)
    model = HeadFactory.create("mlp", feature_dim=FEATURE_DIM, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_mlp(model, loader, optimizer, scaler)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss {train_loss:.5f}")
    return model

def train_ridge(loader, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    print("  Collecting all features for sklearn training...")
    all_feats, all_targets = [], []
    for feats, targets in tqdm(loader, desc="collect"):
        all_feats.append(feats.numpy())
        all_targets.append(targets.numpy())
    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_targets, axis=0)
    print(f"  Training Ridge on {X.shape[0]} samples...")
    model = MultiOutputRegressor(Ridge(alpha=alpha)).fit(X, y)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train model on all samples")
    parser.add_argument("--head", type=str, default="mlp", choices=["mlp", "ridge"],
                        help="Model head type (default: mlp)")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (MLP only)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (MLP only)")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay (MLP only)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 87, 101],
                        help="Random seeds (default: 13 21 42 87 101)")
    parser.add_argument("--output-dir", type=str, default="results/submission_models",
                        help="Output directory for trained models (default: results/submission_models)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-aug", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training {args.head.upper()} on ALL samples")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {DEVICE}")
    print()

    # Load all indices
    df = get_df()
    all_indices = np.arange(len(df))
    print(f"Training on all {len(all_indices)} samples\n")

    for seed_idx, seed in enumerate(args.seeds):
        print(f"{'=' * 50}")
        print(f"Seed {seed} ({seed_idx + 1}/{len(args.seeds)})")
        print(f"{'=' * 50}")

        dataset = EmbeddingAugmentationDataset(all_indices, EMBED_DIR, n_aug=args.n_aug, is_train=True)
        g = get_generator(seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        if args.head == "mlp":
            model = train_mlp(loader, epochs=args.epochs, lr=args.lr, wd=args.wd, seed=seed)
            save_path = os.path.join(args.output_dir, f"seed_{seed}_final.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved to {save_path}")
            del model
        else:  # ridge
            model = train_ridge(loader)
            import joblib
            save_path = os.path.join(args.output_dir, f"ridge_seed_{seed}.joblib")
            joblib.dump(model, save_path)
            print(f"  Saved to {save_path}")
            del model

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll {len(args.seeds)} models saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
