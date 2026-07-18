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
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from joblib import dump as jl_dump
from tqdm import tqdm

# Light project modules (no torch/timm/albumentations chain)
from configs.cfg import CFG
from configs.deterministic import set_seed
from dataset.preprocess_data import check_splits, get_df
from utils.eval import global_weighted_r2_score, per_target_r2_score

# ============================================================
# CONFIGURATION (overriding CFG for PDF-specified params)
# ============================================================
EMBED_DIR = 'embeddings'
RESULTS_DIR = 'results/experiment_2'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model selection
MODEL_TYPE = 'ridge'   # 'mlp' or 'ridge'

# Ridge regression hyperparameters
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

# Training params matching PDF spec
LR = 1e-3
WD = 1e-2
EPOCHS = 80
WARMUP_EPOCHS = 5
PATIENCE = 15
BATCH_SIZE = 8  # MLP on features: much larger batches possible
GRAD_ACC = 1
N_FOLDS = 5
N_AUG = 15

SEEDS = [13, 21, 42, 87, 101]

# Read embedding dimension from metadata
with open(os.path.join(EMBED_DIR, 'metadata.json')) as f:
    _meta = json.load(f)
FEATURE_DIM = _meta['embedding_dim']
print(f"Embedding dimension: {FEATURE_DIM}")

TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']


def _get_device():
    """Lazy import for torch and get device."""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_embeddings(embed_dir):
    """Lazy load torch for loading .pt files."""
    import torch
    targets = torch.load(os.path.join(embed_dir, 'targets.pt'))
    clean_embeddings = torch.load(os.path.join(embed_dir, 'clean_embeddings.pt'))
    return targets, clean_embeddings


# ============================================================
# SPLIT STRATEGIES
# ============================================================

def get_random_stratified_splits(df, seed):
    """Random stratified 5-fold CV (IID validation)."""
    bins = pd.qcut(df['Dry_Total_g'], q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    check_splits(skf.split(df, bins), df)
    
    return list(skf.split(df, bins))

def get_date_grouped_splits(df, seed):
    """Date-grouped 5-fold CV (unseen sampling dates)."""
    dates = df['Sampling_Date'].astype(str)
    bins = pd.qcut(df['Dry_Total_g'], q=5, labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    check_splits(sgkf.split(df, bins, groups=dates), df)

    return list(sgkf.split(df, bins, groups=dates))

def get_date_location_grouped_splits(df, seed):
    """Date-location grouped 5-fold CV (unseen acquisition contexts)."""
    groups = df['group']
    bins = pd.qcut(df['Dry_Total_g'], q=5, labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    check_splits(sgkf.split(df, bins, groups=groups), df)

    return list(sgkf.split(df, bins, groups=groups))

def get_date_location_grouped_splits_weighted(df, seed):
    """Same as primary but stratified on Weighted_g instead of Total."""
    groups = df['group']
    bins = pd.qcut(df['Weighted_g'], q=5, labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    check_splits(sgkf.split(df, bins, groups=groups), df)
    
    return list(sgkf.split(df, bins, groups=groups))

def get_lopo_splits(df, seed):
    """Leave-one-period-out splits with sample-balanced periods."""
    unique_dates = sorted(df['Sampling_Date'].unique())
    total_samples = len(df)
    target = total_samples // 3
    
    period_dates = {'Early': [], 'Middle': [], 'Late': []}
    period_names = ['Early', 'Middle', 'Late']
    cumulative = 0
    current_period_idx = 0
    
    for i, d in enumerate(unique_dates):
        count = len(df[df['Sampling_Date'] == d])
        cumulative += count
        period_dates[period_names[current_period_idx]].append(d)
        
        if cumulative >= target and current_period_idx < 2:
            remaining_dates = len(unique_dates) - i - 1
            remaining_periods = 2 - current_period_idx
            if remaining_dates >= remaining_periods:
                cumulative = 0
                current_period_idx += 1
    
    splits = []
    for period_name in period_names:
        held_out_dates = set(period_dates[period_name])
        train_idx = df[~df['Sampling_Date'].isin(held_out_dates)].index.values
        val_idx = df[df['Sampling_Date'].isin(held_out_dates)].index.values
        splits.append((train_idx, val_idx))
    
    print(f"\n  LOPO Split Report ({total_samples} total samples, {len(unique_dates)} dates):")
    print(f"  {'Period':<10} {'Dates':<8} {'Samples':<10} {'Date Range':<28} {'States':<20} {'Groups':<10}")
    print(f"  {'-'*76}")
    for name in period_names:
        dates = period_dates[name]
        sub = df[df['Sampling_Date'].isin(dates)]
        d0 = pd.Timestamp(dates[0]).strftime('%Y-%m-%d')
        d1 = pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')
        states = sorted(sub['State'].unique())
        groups = sub['group'].nunique()
        print(f"  {name:<10} {len(dates):<8} {len(sub):<10} {d0} to {d1:<14} {str(states):<20} {groups:<10}")
        for d in dates:
            cnt = len(df[df['Sampling_Date'] == d])
            state_str = str(sorted(df[df['Sampling_Date'] == d]['State'].unique()))
            print(f"    {pd.Timestamp(d).strftime('%Y-%m-%d')}: {cnt} samples {state_str}")
    
    all_split_dates = set()
    for name in period_names:
        for d in period_dates[name]:
            assert d not in all_split_dates, f"DUPLICATE DATE {d} in {name}!"
            all_split_dates.add(d)
    assert len(all_split_dates) == len(unique_dates), "Not all dates assigned!"
    
    return splits

def get_loso_splits(df, seed):
    """Leave-one-state-out: 4 fixed splits, no seeds needed."""
    splits = []
    for state in sorted(df['State'].unique()):
        val_idx   = df.index[df['State'] == state].tolist()
        train_idx = df.index[df['State'] != state].tolist()
        splits.append((train_idx, val_idx))
    return splits

SPLIT_STRATEGIES = {
    'random_stratified': get_random_stratified_splits,
    'date_grouped': get_date_grouped_splits,
    'date_location_grouped': get_date_location_grouped_splits,
    # 'date_location_grouped_splits_weighted' :get_date_location_grouped_splits_weighted,
    # 'leave_one_period_out': get_lopo_splits,
    # 'leave_one_state_out': get_loso_splits
}

SPLIT_DISPLAY_NAMES = {
    'random_stratified': 'Random stratified 5-fold CV',
    'date_grouped': 'Date-grouped 5-fold CV',
    'date_location_grouped': 'Date-location grouped 5-fold CV',
    # 'date_location_grouped_splits_weighted': 'Date-location grouped 5-fold CV stratified by weighted targets',
    # 'leave_one_period_out': 'Leave-one-period-out',
    # 'leave_one_state_out': 'Leave-one-state-out'
}

LOPO_PERIOD_NAMES = ['Early', 'Middle', 'Late']


# ============================================================
# MLP TRAINING (lazy imports — torch etc. loaded on demand)
# ============================================================

def _mlp_imports():
    """Lazy imports for MLP-only dependencies (torch, timm, albumentations chain)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from configs.deterministic import seed_worker, get_generator
    from models.models import BiomassSimpleMLP
    from utils.eval import weighted_biomass_loss
    from dataset.preprocess_data import EmbeddingAugmentationDataset
    return (torch, nn, F, autocast, DataLoader, seed_worker, get_generator,
            BiomassSimpleMLP, weighted_biomass_loss, EmbeddingAugmentationDataset)


def train_epoch_mlp(model, loader, optimizer, scaler):
    """One training epoch for MLP on precomputed features."""
    torch = __import__('torch')
    autocast = torch.amp.autocast
    from utils.eval import weighted_biomass_loss
    DEVICE = _get_device()
    
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


def valid_epoch_mlp(model, loader):
    """Validation epoch for MLP on precomputed features."""
    torch = __import__('torch')
    autocast = torch.amp.autocast
    DEVICE = _get_device()
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for feats, targets in tqdm(loader, desc='valid', leave=False):
        feats = feats.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
        
        preds = torch.stack([p_green, p_dead, p_clover, p_gdm, p_total], dim=1).squeeze(-1)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    weighted_r2 = global_weighted_r2_score(all_labels, all_preds)
    per_target = per_target_r2_score(all_labels, all_preds)
    
    errors = all_preds - all_labels
    per_rmse = np.sqrt(np.mean(errors**2, axis=0))
    per_mae = np.mean(np.abs(errors), axis=0)
    per_bias = np.mean(errors, axis=0)
    
    return {
        'weighted_r2': weighted_r2,
        'per_target_r2': per_target,
        'per_rmse': per_rmse,
        'per_mae': per_mae,
        'per_bias': per_bias,
        'preds': all_preds,
        'targets': all_labels,
    }


def train_mlp_model(train_idx, val_idx, embed_dir, seed, fold_idx, protocol_name):
    """
    Train a single MLP model for one fold.
    Returns validation metrics and predictions (with val_idx for OOF assembly).
    Saves the best model checkpoint to results/{protocol_name}/fold_{fold}_seed_{seed}.pt
    """
    (torch, nn, F, autocast, DataLoader, seed_worker, get_generator,
     BiomassSimpleMLP, weighted_biomass_loss, EmbeddingAugmentationDataset) = _mlp_imports()
    
    DEVICE = _get_device()
    set_seed(seed, deterministic=True)
    
    train_set = EmbeddingAugmentationDataset(train_idx, embed_dir, n_aug=N_AUG, is_train=True)
    val_set = EmbeddingAugmentationDataset(val_idx, embed_dir, n_aug=N_AUG, is_train=False)
    
    g = get_generator(seed)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    
    model = BiomassSimpleMLP(FEATURE_DIM).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    best_metrics = None
    best_score = -np.inf
    best_model_state = None
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
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'  Early stopping at epoch {epoch} (best: {best_score:.4f})')
                break
    
    best_metrics['val_idx'] = val_idx
    
    model_dir = os.path.join(RESULTS_DIR, protocol_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, f'fold_{fold_idx}_seed_{seed}.pt')
    torch.save(best_model_state, ckpt_path)
    print(f'  ✓ Model saved to {ckpt_path}')
    
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_metrics


# ============================================================
# RIDGE TRAINING (minimal deps — only needs torch for .pt loading)
# ============================================================

def train_ridge_model(train_idx, val_idx, embed_dir, seed, fold_idx, protocol_name):
    """
    Train a Ridge regression model for one fold.
    Uses clean embeddings only (no augmentation) and MultiOutputRegressor(RidgeCV).
    Saves the model to results/{protocol_name}/fold_{fold}_seed_{seed}.joblib.
    Returns validation metrics in the same format as train_mlp_model.
    """
    set_seed(seed, deterministic=True)
    
    targets, clean_embeddings = _load_embeddings(embed_dir)
    
    train_feats = clean_embeddings[train_idx].numpy()
    train_targets = targets[train_idx].numpy()
    val_feats = clean_embeddings[val_idx].numpy()
    val_targets = targets[val_idx].numpy()
    
    ridge = MultiOutputRegressor(RidgeCV(alphas=RIDGE_ALPHAS))
    ridge.fit(train_feats, train_targets)
    
    # Save the trained Ridge model
    model_dir = os.path.join(RESULTS_DIR, protocol_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'fold_{fold_idx}_seed_{seed}.joblib')
    jl_dump(ridge, model_path)
    print(f'  ✓ Model saved to {model_path}')
    
    val_preds = ridge.predict(val_feats)
    
    weighted_r2 = global_weighted_r2_score(val_targets, val_preds)
    per_target = per_target_r2_score(val_targets, val_preds)
    
    errors = val_preds - val_targets
    per_rmse = np.sqrt(np.mean(errors**2, axis=0))
    per_mae = np.mean(np.abs(errors), axis=0)
    per_bias = np.mean(errors, axis=0)
    
    return {
        'weighted_r2': weighted_r2,
        'per_target_r2': per_target,
        'per_rmse': per_rmse,
        'per_mae': per_mae,
        'per_bias': per_bias,
        'preds': val_preds,
        'targets': val_targets,
        'val_idx': val_idx,
    }


def train_model(train_idx, val_idx, embed_dir, seed, fold_idx, protocol_name):
    """
    Dispatch to the appropriate model training function based on MODEL_TYPE.
    """
    if MODEL_TYPE == 'ridge':
        return train_ridge_model(train_idx, val_idx, embed_dir, seed, fold_idx, protocol_name)
    else:
        return train_mlp_model(train_idx, val_idx, embed_dir, seed, fold_idx, protocol_name)


# ============================================================
# OOF METRIC COMPUTATION
# ============================================================

def assemble_oof(fold_results, n_total):
    """
    Assemble the complete OOF prediction array from fold-wise results.
    """
    n_targets = fold_results[0]['preds'].shape[1]
    oof_preds = np.full((n_total, n_targets), np.nan)
    oof_targets = np.full((n_total, n_targets), np.nan)
    
    for fr in fold_results:
        val_idx = fr['val_idx']
        oof_preds[val_idx] = fr['preds']
        oof_targets[val_idx] = fr['targets']
    
    missing = np.isnan(oof_preds).any(axis=1)
    if missing.any():
        n_missing = missing.sum()
        print(f"  WARNING: {n_missing}/{n_total} samples missing from OOF array!")
        return None, None
    
    return oof_preds, oof_targets


def compute_oof_metrics(oof_preds, oof_targets):
    """
    Compute ALL metrics on the FULL OOF array (357, 5).
    """
    weighted_r2 = global_weighted_r2_score(oof_targets, oof_preds)
    per_target_r2 = per_target_r2_score(oof_targets, oof_preds)
    
    errors = oof_preds - oof_targets
    per_rmse = np.sqrt(np.mean(errors**2, axis=0))
    per_mae = np.mean(np.abs(errors), axis=0)
    per_bias = np.mean(errors, axis=0)
    
    return {
        'weighted_r2': weighted_r2,
        'per_target_r2': per_target_r2,
        'per_rmse': per_rmse,
        'per_mae': per_mae,
        'per_bias': per_bias,
        'oos_preds': oof_preds,
        'oos_targets': oof_targets,
    }


# ============================================================
# ORCHESTRATION
# ============================================================

def run_experiment():
    """Run all protocols × seeds × folds."""
    
    print("=" * 80)
    print("EXPERIMENT 2: Validation Protocol Comparison")
    print(f"Model: {MODEL_TYPE}")
    print("=" * 80)
    
    df = get_df()
    print(f"Loaded {len(df)} images\n")
    
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
            
            splits = split_fn(df, seed)
            n_splits = len(splits)
            
            seed_fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                print(f"\n  Fold {fold_idx+1}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")
                
                metrics = train_model(train_idx, val_idx, EMBED_DIR, seed, fold_idx, protocol_name)
                metrics['fold'] = fold_idx
                metrics['seed'] = seed
                metrics['n_train'] = len(train_idx)
                metrics['n_val'] = len(val_idx)
                seed_fold_results.append(metrics)
                
                print(f"    Weighted R2: {metrics['weighted_r2']:.4f}")
                print(f"    Per-target RMSE: {', '.join(f'{v:.2f}' for v in metrics['per_rmse'])}")
                print(f"    Per-target MAE:  {', '.join(f'{v:.2f}' for v in metrics['per_mae'])}")
                print(f"    Per-target Bias: {', '.join(f'{v:.3f}' for v in metrics['per_bias'])}")
                for i, name in enumerate(['Green', 'Dead', 'Clover', 'GDM', 'Total']):
                    print(f"      {name}: R2={metrics['per_target_r2'][TARGET_NAMES[i]]:.4f}", end="")
                    if i < 4:
                        print(" |", end="")
                print()
            
            oof_preds, oof_targets = assemble_oof(seed_fold_results, len(df))
            
            if oof_preds is not None:
                oof_metrics = compute_oof_metrics(oof_preds, oof_targets)
                oof_metrics['seed'] = seed
            else:
                print("  WARNING: Could not assemble OOF array, using fold averages")
                oof_metrics = aggregate_fold_results(seed_fold_results)
                oof_metrics['seed'] = seed
            
            protocol_results['seed_results'].append(oof_metrics)
            protocol_results['fold_results'].extend(seed_fold_results)
            
            print(f"  Seed {seed} OOF: Weighted R2={oof_metrics['weighted_r2']:.4f}")
        
        protocol_avg = aggregate_seed_results(protocol_results['seed_results'])
        protocol_results['aggregated'] = protocol_avg
        all_results[protocol_name] = protocol_results
        
        per_rmse_avg = protocol_avg['per_rmse']
        per_rmse_std = protocol_avg['std_per_rmse_across_seeds']
        print(f"\n  >>> {display_name}: "
              f"R2={protocol_avg['weighted_r2']:.4f}±{protocol_avg['std_weighted_r2']:.4f} | "
              f"Per-target RMSE: {', '.join(f'{v:.2f}±{s:.2f}' for v,s in zip(per_rmse_avg, per_rmse_std))}")
    
    # Save results using torch (lazy import)
    import torch as _torch
    save_path = os.path.join(RESULTS_DIR, 'full_results.pt')
    _torch.save(all_results, save_path)
    print(f"\nFull results saved to {save_path}")
    
    generate_table_8(all_results)
    generate_table_9(all_results, df)
    generate_table_10(all_results, df)
    generate_table_11(all_results, df)
    
    print(f"\nAll tables saved to {RESULTS_DIR}/")
    return all_results


# ============================================================
# AGGREGATION HELPERS
# ============================================================

def aggregate_fold_results(fold_results):
    """Average metrics across folds for one seed."""
    aggregated = {}
    
    values = [r['weighted_r2'] for r in fold_results]
    aggregated['weighted_r2'] = np.mean(values)
    aggregated['std_weighted_r2'] = np.std(values, ddof=1)
    
    for metric in ['per_rmse', 'per_mae', 'per_bias']:
        vals = np.stack([r[metric] for r in fold_results])
        aggregated[metric] = vals.mean(axis=0)
        aggregated[f'std_{metric}'] = vals.std(axis=0, ddof=1)
    
    per_target_keys = list(fold_results[0]['per_target_r2'].keys())
    per_target_means = {}
    per_target_stds = {}
    for tk in per_target_keys:
        vals = [r['per_target_r2'][tk] for r in fold_results]
        per_target_means[tk] = np.mean(vals)
        per_target_stds[tk] = np.std(vals, ddof=1)
    aggregated['per_target_r2'] = per_target_means
    aggregated['std_per_target_r2'] = per_target_stds
    
    return aggregated


def aggregate_seed_results(seed_results):
    """Average metrics across seeds."""
    aggregated = {}
    
    values = [r['weighted_r2'] for r in seed_results]
    aggregated['weighted_r2'] = np.mean(values)
    aggregated['std_weighted_r2'] = np.std(values, ddof=1)
    
    for metric in ['per_rmse', 'per_mae', 'per_bias']:
        vals = np.stack([r[metric] for r in seed_results])
        aggregated[metric] = vals.mean(axis=0)
        aggregated[f'std_{metric}_across_seeds'] = vals.std(axis=0, ddof=1)
        intra_std_vals = np.stack([r.get(f'std_{metric}', np.zeros(5)) for r in seed_results])
        aggregated[f'std_{metric}'] = intra_std_vals.mean(axis=0)
    
    per_target_keys = list(seed_results[0]['per_target_r2'].keys())
    per_target_means = {}
    per_target_stds = {}
    for tk in per_target_keys:
        vals = [r['per_target_r2'][tk] for r in seed_results]
        per_target_means[tk] = np.mean(vals)
        per_target_stds[tk] = np.std(vals, ddof=1)
    aggregated['per_target_r2'] = per_target_means
    aggregated['std_per_target_r2'] = per_target_stds
    
    return aggregated


# ============================================================
# TABLE GENERATION
# ============================================================

def generate_table_8(all_results):
    """Table 8: Effect of validation protocol on local performance estimates."""
    rows = []
    protocol_order = [
                       'random_stratified', 
                       'date_grouped',
                       'date_location_grouped', 
                    #   'date_location_grouped_splits_weighted',
                    #   'leave_one_period_out',
                    #   'leave_one_state_out'
                      ]
    
    for protocol in protocol_order:
        agg = all_results[protocol]['aggregated']
        agg_per_rmse = agg['per_rmse']
        agg_per_mae = agg['per_mae']
        rows.append({
            'Validation protocol': SPLIT_DISPLAY_NAMES[protocol],
            'Local weighted R2 ↑': f"{agg['weighted_r2']:.4f}",
            'Std': f"{agg['std_weighted_r2']:.4f}",
            'Local RMSE ↓ (Total)': f"{agg_per_rmse[4]:.2f}",
            'Local MAE ↓ (Total)': f"{agg_per_mae[4]:.2f}",
            'Distance to hidden ↓': 'TBD',
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_8.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 8: Validation Protocol Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


def generate_table_9(all_results, df):
    """Table 9: Per-target R2 under different validation protocols."""
    rows = []
    protocol_order = [
                       'random_stratified', 
                       'date_grouped',
                       'date_location_grouped', 
                    #   'date_location_grouped_splits_weighted',
                    #   'leave_one_period_out',
                    #   'leave_one_state_out'
                      ]
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
    """Table 10: Leave-one-period-out temporal generalization."""
    lopo_results = all_results.get('leave_one_period_out')
    if lopo_results is None:
        return
    
    fold_results_by_seed = {}
    for seed_result in lopo_results['seed_results']:
        seed = seed_result['seed']
        for fr in lopo_results['fold_results']:
            if fr['seed'] == seed:
                fold_idx = fr['fold']
                if fold_idx not in fold_results_by_seed:
                    fold_results_by_seed[fold_idx] = []
                fold_results_by_seed[fold_idx].append(fr)
    
    def _total_metric(fr, key):
        return fr[key][4]
    
    rows = []
    for period_idx, period_name in enumerate(['Early', 'Middle', 'Late']):
        if period_idx not in fold_results_by_seed:
            continue
        
        fr_list = fold_results_by_seed[period_idx]
        period_r2s = [r['weighted_r2'] for r in fr_list]
        period_rmses = [_total_metric(r, 'per_rmse') for r in fr_list]
        period_maes = [_total_metric(r, 'per_mae') for r in fr_list]
        period_biases = [_total_metric(r, 'per_bias') for r in fr_list]
        
        rows.append({
            'Held-out period': period_name,
            'Training periods': {'Early': 'Middle + Late', 'Middle': 'Early + Late', 'Late': 'Early + Middle'}[period_name],
            'Weighted R2 ↑': f"{np.mean(period_r2s):.4f}",
            'RMSE ↓': f"{np.mean(period_rmses):.2f}",
            'MAE ↓': f"{np.mean(period_maes):.2f}",
            'Bias': f"{np.mean(period_biases):.3f}",
        })
    
    all_fr = [r for fr_list in fold_results_by_seed.values() for r in fr_list]
    all_period_r2s = [r['weighted_r2'] for r in all_fr]
    all_period_rmses = [_total_metric(r, 'per_rmse') for r in all_fr]
    all_period_maes = [_total_metric(r, 'per_mae') for r in all_fr]
    all_period_biases = [_total_metric(r, 'per_bias') for r in all_fr]
    
    rows.append({
        'Held-out period': 'Mean',
        'Training periods': '—',
        'Weighted R2 ↑': f"{np.mean(all_period_r2s):.4f}",
        'RMSE ↓': f"{np.mean(all_period_rmses):.2f}",
        'MAE ↓': f"{np.mean(all_period_maes):.2f}",
        'Bias': f"{np.mean(all_period_biases):.3f}",
    })
    rows.append({
        'Held-out period': 'Std',
        'Training periods': '—',
        'Weighted R2 ↑': f"{np.std(all_period_r2s):.4f}",
        'RMSE ↓': f"{np.std(all_period_rmses):.2f}",
        'MAE ↓': f"{np.std(all_period_maes):.2f}",
        'Bias': f"{np.std(all_period_biases):.3f}",
    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_10.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 10: Leave-One-Period-Out Temporal Generalization")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df


def generate_table_11(all_results, df):
    """Table 11: Per-target R2 for leave-one-period-out."""
    lopo_results = all_results.get('leave_one_period_out')
    if lopo_results is None:
        return
    
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    target_map = {'Dry_Green_g': 'Target 1', 'Dry_Dead_g': 'Target 2',
                  'Dry_Clover_g': 'Target 3', 'GDM_g': 'Target 4', 'Dry_Total_g': 'Target 5'}
    
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