"""
experiment_2.py — Validation Protocol Comparison

Compares 5 validation strategies to see which best approximates hidden performance.
Produces Tables 8, 9, 10, 11 from the experimental plan.

Usage:
    python experiment_2.py           # After precompute_features.py has run

Output:
    results/experiment_2/
        table_8.csv, table_9.csv, table_10.csv, table_11.csv
        full_results.pt  # All predictions for downstream use
"""

import os, sys, json, copy, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold
from tqdm import tqdm

# Project modules
from configs.cfg import CFG
from configs.deterministic import set_seed, seed_worker, get_generator
from dataset.preprocess_data import get_df, EmbeddingAugmentationDataset
from models.models import BiomassSimpleMLP
from utils.eval import global_weighted_r2_score, per_target_r2_score, weighted_biomass_loss

# ============================================================
# CONFIGURATION (overriding CFG for PDF-specified params)
# ============================================================
EMBED_DIR = 'embeddings'
RESULTS_DIR = 'results/experiment_2'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training params matching PDF spec
LR = 1e-4
WD = 1e-4
EPOCHS = 80
WARMUP_EPOCHS = 5
PATIENCE = 12
BATCH_SIZE = 32  # MLP on features: much larger batches possible
GRAD_ACC = 1
N_FOLDS = 5
N_AUG = 20

SEEDS = [13, 21, 42, 87, 101]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_DIM = 2048  # 1024 left + 1024 right

# Weighted R2 weights (from CFG)
R2_WEIGHTS = torch.tensor(CFG.R2_WEIGHTS_VAL, dtype=torch.float32, device=DEVICE)
TARGET_NAMES = ['Dry_Green', 'Dry_Dead', 'Dry_Clover', 'GDM', 'Dry_Total']

# ============================================================
# SPLIT STRATEGIES
# ============================================================

def get_random_stratified_splits(df, seed):
    """Random stratified 5-fold CV (IID validation)."""
    bins = pd.qcut(df['Dry_Total_g'], q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(skf.split(df, bins))

def get_date_grouped_splits(df, seed):
    """Date-grouped 5-fold CV (unseen sampling dates)."""
    # Use unique dates as groups
    dates = df['Sampling_Date'].astype(str)
    gkf = GroupKFold(n_splits=5)
    return list(gkf.split(df, groups=dates))

def get_location_grouped_splits(df, seed):
    """Location-grouped 5-fold CV (unseen locations)."""
    locations = df['State']
    gkf = GroupKFold(n_splits=5)
    return list(gkf.split(df, groups=locations))

def get_date_location_grouped_splits(df, seed):
    """Date-location grouped 5-fold CV (unseen acquisition contexts).
    This is the PRIMARY validation protocol per the PDF."""
    groups = df['group']  # Already created in get_df() as State_Date
    gkf = GroupKFold(n_splits=5)
    return list(gkf.split(df, groups=groups))

def get_lopo_splits(df, seed):
    """Leave-one-period-out splits.
    Sort unique dates, split into Early/Middle/Late thirds, 
    train on 2 periods, test on 1."""
    unique_dates = sorted(df['Sampling_Date'].unique())
    n_dates = len(unique_dates)
    split1 = n_dates // 3
    split2 = 2 * n_dates // 3
    
    early_dates = set(unique_dates[:split1])
    middle_dates = set(unique_dates[split1:split2])
    late_dates = set(unique_dates[split2:])
    
    splits = []
    for held_out, name in [(early_dates, 'Early'), (middle_dates, 'Middle'), (late_dates, 'Late')]:
        train_idx = df[~df['Sampling_Date'].isin(held_out)].index.values
        val_idx = df[df['Sampling_Date'].isin(held_out)].index.values
        splits.append((train_idx, val_idx))
    
    return splits

SPLIT_STRATEGIES = {
    'random_stratified': get_random_stratified_splits,
    'date_grouped': get_date_grouped_splits,
    'location_grouped': get_location_grouped_splits,
    'date_location_grouped': get_date_location_grouped_splits,
    'leave_one_period_out': get_lopo_splits,
}

SPLIT_DISPLAY_NAMES = {
    'random_stratified': 'Random stratified 5-fold CV',
    'date_grouped': 'Date-grouped 5-fold CV',
    'location_grouped': 'Location-grouped 5-fold CV',
    'date_location_grouped': 'Date-location grouped 5-fold CV',
    'leave_one_period_out': 'Leave-one-period-out',
}

LOPO_PERIOD_NAMES = ['Early', 'Middle', 'Late']


# ============================================================
# TRAINING LOOP (adapted from train.py for precomputed features)
# ============================================================

def train_epoch_mlp(model, loader, optimizer, scaler):
    """One training epoch for MLP on precomputed features."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (feats, targets) in enumerate(tqdm(loader, desc='train', leave=False)):
        feats = feats.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        with autocast('cuda', dtype=torch.bfloat16):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, targets)
        
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


@torch.no_grad()
def valid_epoch_mlp(model, loader):
    """Validation epoch for MLP on precomputed features."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for feats, targets in tqdm(loader, desc='valid', leave=False):
        feats = feats.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        with autocast('cuda', dtype=torch.bfloat16):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
        
        # Stack in correct order: Green, Dead, Clover, GDM, Total
        preds = torch.stack([p_green, p_dead, p_clover, p_gdm, p_total], dim=1).squeeze(-1)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    weighted_r2 = global_weighted_r2_score(all_labels, all_preds)
    per_target = per_target_r2_score(all_labels, all_preds)
    
    # Compute RMSE, MAE, Bias (per target then weighted)
    errors = all_preds - all_labels
    per_rmse = np.sqrt(np.mean(errors**2, axis=0))
    per_mae = np.mean(np.abs(errors), axis=0)
    per_bias = np.mean(errors, axis=0)
    
    weights = CFG.R2_WEIGHTS_VAL
    weighted_rmse = np.sum(per_rmse * weights)
    weighted_mae = np.sum(per_mae * weights)
    weighted_bias = np.sum(per_bias * weights)
    
    return {
        'weighted_r2': weighted_r2,
        'per_target_r2': per_target,
        'weighted_rmse': weighted_rmse,
        'weighted_mae': weighted_mae,
        'weighted_bias': weighted_bias,
        'preds': all_preds,
        'targets': all_labels,
    }


def train_model(train_idx, val_idx, embed_dir, seed):
    """
    Train a single MLP model for one fold.
    Returns validation metrics and predictions.
    """
    set_seed(seed, deterministic=True)
    
    # Create datasets
    train_set = EmbeddingAugmentationDataset(
        train_idx, embed_dir, n_aug=N_AUG, is_train=True
    )
    val_set = EmbeddingAugmentationDataset(
        val_idx, embed_dir, n_aug=N_AUG, is_train=False
    )
    
    g = get_generator()
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    
    # Build MLP model
    model = BiomassSimpleMLP(FEATURE_DIM).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    
    # Warmup + Cosine scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler],
        milestones=[WARMUP_EPOCHS]
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_metrics = None
    best_score = -np.inf
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch_mlp(model, train_loader, optimizer, scaler)
        val_metrics = valid_epoch_mlp(model, val_loader)
        scheduler.step()
        
        val_r2 = val_metrics['weighted_r2']
        
        if val_r2 > best_score:
            best_score = val_r2
            best_metrics = val_metrics
            best_metrics['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'  Early stopping at epoch {epoch} (best: {best_score:.4f})')
                break
    
    # Cleanup
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_metrics


# ============================================================
# ORCHESTRATION
# ============================================================

def run_experiment():
    """Run all protocols × seeds × folds."""
    
    print("=" * 80)
    print("EXPERIMENT 2: Validation Protocol Comparison")
    print("=" * 80)
    
    # Load data
    df = get_df()
    print(f"Loaded {len(df)} images\n")
    
    # Store all results
    all_results = {}
    
    for protocol_name, split_fn in SPLIT_STRATEGIES.items():
        display_name = SPLIT_DISPLAY_NAMES[protocol_name]
        print(f"\n{'='*60}")
        print(f"Protocol: {display_name}")
        print(f"{'='*60}")
        
        protocol_results = {
            'seed_results': [],
            'fold_results': [],
            'per_target_results': [],
        }
        
        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  --- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")
            
            # Generate splits
            splits = split_fn(df, seed)
            n_splits = len(splits)
            
            seed_fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                print(f"\n  Fold {fold_idx+1}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")
                
                metrics = train_model(train_idx, val_idx, EMBED_DIR, seed)
                metrics['fold'] = fold_idx
                metrics['seed'] = seed
                metrics['n_train'] = len(train_idx)
                metrics['n_val'] = len(val_idx)
                seed_fold_results.append(metrics)
                
                print(f"    Weighted R2: {metrics['weighted_r2']:.4f} | "
                      f"RMSE: {metrics['weighted_rmse']:.4f} | "
                      f"MAE: {metrics['weighted_mae']:.4f} | "
                      f"Bias: {metrics['weighted_bias']:.4f}")
                for i, name in enumerate(['Green', 'Dead', 'Clover', 'GDM', 'Total']):
                    print(f"      {name}: R2={metrics['per_target_r2'][TARGET_NAMES[i]]:.4f}", end="")
                    if i < 4:
                        print(" |", end="")
                print()
            
            # Aggregate folds for this seed
            seed_avg_metrics = aggregate_fold_results(seed_fold_results)
            seed_avg_metrics['seed'] = seed
            protocol_results['seed_results'].append(seed_avg_metrics)
            protocol_results['fold_results'].extend(seed_fold_results)
            
            print(f"  Seed {seed} avg: Weighted R2={seed_avg_metrics['weighted_r2']:.4f} ± {seed_avg_metrics['std_weighted_r2']:.4f}")
        
        # Aggregate across seeds
        protocol_avg = aggregate_seed_results(protocol_results['seed_results'])
        protocol_results['aggregated'] = protocol_avg
        all_results[protocol_name] = protocol_results
        
        print(f"\n  >>> {display_name}: "
              f"R2={protocol_avg['weighted_r2']:.4f}±{protocol_avg['std_weighted_r2']:.4f} | "
              f"RMSE={protocol_avg['weighted_rmse']:.4f}±{protocol_avg['std_weighted_rmse']:.4f} | "
              f"MAE={protocol_avg['weighted_mae']:.4f}±{protocol_avg['std_weighted_mae']:.4f}")
    
    # Save results
    save_path = os.path.join(RESULTS_DIR, 'full_results.pt')
    torch.save(all_results, save_path)
    print(f"\nFull results saved to {save_path}")
    
    # Generate tables
    generate_table_8(all_results)
    generate_table_9(all_results)
    generate_table_10(all_results, df)
    generate_table_11(all_results, df)
    
    print(f"\nAll tables saved to {RESULTS_DIR}/")
    return all_results


# ============================================================
# AGGREGATION HELPERS
# ============================================================

def aggregate_fold_results(fold_results):
    """Average metrics across folds for one seed."""
    keys = ['weighted_r2', 'weighted_rmse', 'weighted_mae', 'weighted_bias']
    aggregated = {}
    
    for key in keys:
        values = [r[key] for r in fold_results]
        aggregated[key] = np.mean(values)
        aggregated[f'std_{key}'] = np.std(values)
    
    # Per-target R2
    per_target_keys = list(fold_results[0]['per_target_r2'].keys())
    per_target_means = {}
    per_target_stds = {}
    for tk in per_target_keys:
        vals = [r['per_target_r2'][tk] for r in fold_results]
        per_target_means[tk] = np.mean(vals)
        per_target_stds[tk] = np.std(vals)
    aggregated['per_target_r2'] = per_target_means
    aggregated['std_per_target_r2'] = per_target_stds
    
    return aggregated


def aggregate_seed_results(seed_results):
    """Average metrics across seeds."""
    keys = ['weighted_r2', 'weighted_rmse', 'weighted_mae', 'weighted_bias',
            'std_weighted_r2', 'std_weighted_rmse', 'std_weighted_mae', 'std_weighted_bias']
    
    aggregated = {}
    for key in keys:
        values = [r[key] for r in seed_results]
        aggregated[key] = np.mean(values)
        aggregated[f'std_{key}_across_seeds'] = np.std(values)
    
    # Per-target R2 across seeds
    per_target_keys = list(seed_results[0]['per_target_r2'].keys())
    per_target_means = {}
    per_target_stds = {}
    for tk in per_target_keys:
        vals = [r['per_target_r2'][tk] for r in seed_results]
        per_target_means[tk] = np.mean(vals)
        per_target_stds[tk] = np.std(vals)
    aggregated['per_target_r2'] = per_target_means
    aggregated['std_per_target_r2'] = per_target_stds
    
    return aggregated


# ============================================================
# TABLE GENERATION
# ============================================================

def generate_table_8(all_results):
    """
    Table 8: Effect of validation protocol on local performance estimates.
    Columns: Protocol, Local weighted R2↑, Std, Local RMSE↓, Local MAE↓, Distance to hidden↓
    """
    rows = []
    protocol_order = ['random_stratified', 'date_grouped', 'location_grouped', 
                      'date_location_grouped', 'leave_one_period_out']
    
    for protocol in protocol_order:
        agg = all_results[protocol]['aggregated']
        rows.append({
            'Validation protocol': SPLIT_DISPLAY_NAMES[protocol],
            'Local weighted R2 ↑': f"{agg['weighted_r2']:.4f}",
            'Std': f"{agg['std_weighted_r2']:.4f}",
            'Local RMSE ↓': f"{agg['weighted_rmse']:.4f}",
            'Local MAE ↓': f"{agg['weighted_mae']:.4f}",
            'Distance to hidden ↓': 'TBD',  # Requires Kaggle submission
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_8.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 8: Validation Protocol Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


def generate_table_9(all_results, df):
    """
    Table 9: Per-target R2 under different validation protocols.
    """
    rows = []
    protocol_order = ['random_stratified', 'date_grouped', 'location_grouped',
                      'date_location_grouped', 'leave_one_period_out']
    target_map = {'Dry_Green_g': 'Target 1', 'Dry_Dead_g': 'Target 2', 
                  'Dry_Clover_g': 'Target 3', 'GDM_g': 'Target 4', 'Dry_Total_g': 'Target 5'}
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    
    for protocol in protocol_order:
        agg = all_results[protocol]['aggregated']
        row = {'Validation protocol': SPLIT_DISPLAY_NAMES[protocol]}
        for tc in target_cols:
            row[target_map[tc]] = f"{agg['per_target_r2'][tc]:.4f}"
        row['Weighted R2'] = f"{agg['weighted_r2']:.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_9.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 9: Per-Target R2 under Different Validation Protocols")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


def generate_table_10(all_results, df):
    """
    Table 10: Leave-one-period-out temporal generalization.
    Extracts per-period results from the LOPO protocol.
    """
    lopo_results = all_results.get('leave_one_period_out')
    if lopo_results is None:
        return
    
    # Extract per-fold results which correspond to periods
    # Fold 0 = Early held out, Fold 1 = Middle, Fold 2 = Late
    fold_results_by_seed = {}
    for seed_result in lopo_results['seed_results']:
        seed = seed_result['seed']
        # Re-extract from fold results
        for fr in lopo_results['fold_results']:
            if fr['seed'] == seed:
                fold_idx = fr['fold']
                if fold_idx not in fold_results_by_seed:
                    fold_results_by_seed[fold_idx] = []
                fold_results_by_seed[fold_idx].append(fr)
    
    rows = []
    for period_idx, period_name in enumerate(['Early', 'Middle', 'Late']):
        if period_idx not in fold_results_by_seed:
            continue
        
        period_r2s = [r['weighted_r2'] for r in fold_results_by_seed[period_idx]]
        period_rmses = [r['weighted_rmse'] for r in fold_results_by_seed[period_idx]]
        period_maes = [r['weighted_mae'] for r in fold_results_by_seed[period_idx]]
        period_biases = [r['weighted_bias'] for r in fold_results_by_seed[period_idx]]
        
        # Get test samples count and mean
        val_idx = None
        # Find any fold result to get indices
        if lopo_results['fold_results']:
            fr = lopo_results['fold_results'][0]
        
        rows.append({
            'Held-out period': period_name,
            'Training periods': {'Early': 'Middle + Late', 'Middle': 'Early + Late', 'Late': 'Early + Middle'}[period_name],
            'Weighted R2 ↑': f"{np.mean(period_r2s):.4f}",
            'RMSE ↓': f"{np.mean(period_rmses):.4f}",
            'MAE ↓': f"{np.mean(period_maes):.4f}",
            'Bias': f"{np.mean(period_biases):.4f}",
        })
    
    # Add mean/std rows
    all_period_r2s = [r['weighted_r2'] for fr_list in fold_results_by_seed.values() for r in fr_list]
    all_period_rmses = [r['weighted_rmse'] for fr_list in fold_results_by_seed.values() for r in fr_list]
    all_period_maes = [r['weighted_mae'] for fr_list in fold_results_by_seed.values() for r in fr_list]
    
    rows.append({
        'Held-out period': 'Mean',
        'Training periods': '—',
        'Weighted R2 ↑': f"{np.mean(all_period_r2s):.4f}",
        'RMSE ↓': f"{np.mean(all_period_rmses):.4f}",
        'MAE ↓': f"{np.mean(all_period_maes):.4f}",
        'Bias': f"{np.mean([r['weighted_bias'] for fr_list in fold_results_by_seed.values() for r in fr_list]):.4f}",
    })
    rows.append({
        'Held-out period': 'Std',
        'Training periods': '—',
        'Weighted R2 ↑': f"{np.std(all_period_r2s):.4f}",
        'RMSE ↓': f"{np.std(all_period_rmses):.4f}",
        'MAE ↓': f"{np.std(all_period_maes):.4f}",
        'Bias': f"{np.std([r['weighted_bias'] for fr_list in fold_results_by_seed.values() for r in fr_list]):.4f}",
    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_10.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 10: Leave-One-Period-Out Temporal Generalization")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


def generate_table_11(all_results, df):
    """
    Table 11: Per-target R2 for leave-one-period-out.
    """
    lopo_results = all_results.get('leave_one_period_out')
    if lopo_results is None:
        return
    
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    target_map = {'Dry_Green_g': 'Target 1', 'Dry_Dead_g': 'Target 2',
                  'Dry_Clover_g': 'Target 3', 'GDM_g': 'Target 4', 'Dry_Total_g': 'Target 5'}
    
    # Group fold results by period
    fold_results_by_period = {}
    for fr in lopo_results['fold_results']:
        period = LOPO_PERIOD_NAMES[fr['fold']]
        if period not in fold_results_by_period:
            fold_results_by_period[period] = []
        fold_results_by_period[period].append(fr)
    
    rows = []
    for period in ['Early', 'Middle', 'Late']:
        if period not in fold_results_by_period:
            continue
        fr_list = fold_results_by_period[period]
        row = {'Held-out period': period}
        for tc in target_cols:
            vals = [fr['per_target_r2'][tc] for fr in fr_list]
            row[target_map[tc]] = f"{np.mean(vals):.4f}"
        r2_vals = [fr['weighted_r2'] for fr in fr_list]
        row['Weighted R2'] = f"{np.mean(r2_vals):.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_11.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 11: Per-Target R2 for Leave-One-Period-Out")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    run_experiment()