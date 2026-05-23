"""
experiment_5.py — Error Analysis

Analyzes where the model fails by evaluating out-of-fold predictions
grouped by biomass range, temporal period, and location.
Uses date-location grouped CV (the best/primary protocol from Experiment 2).

Produces Tables 17, 18, 19 from the experimental plan.

Usage:
    python experiment_5.py           # After precompute_features.py

Output:
    results/experiment_5/
        table_17.csv  (Error by biomass range)
        table_18.csv  (Error by temporal period)
        table_19.csv  (Error by location)
"""

import os, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.cfg import CFG
from configs.deterministic import set_seed, seed_worker, get_generator
from dataset.preprocess_data import get_df, EmbeddingAugmentationDataset
from models.models import BiomassSimpleMLP
from utils.eval import global_weighted_r2_score, per_target_r2_score

# Reuse experiment_2's components
from experiment_2 import (
    get_date_location_grouped_splits, train_model, 
    SEEDS, EMBED_DIR, FEATURE_DIM, N_AUG, DEVICE, TARGET_NAMES
)

RESULTS_DIR = 'results/experiment_5'
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_NAMES_SHORT = ['Green', 'Dead', 'Clover', 'GDM', 'Total']
PERIOD_NAMES = ['Early', 'Middle', 'Late']
LOCATIONS = ['Tas', 'NSW', 'Vic', 'WA']


def compute_subgroup_metrics(y_true, y_pred, weights):
    """
    Compute metrics for a subset of data.
    
    Args:
        y_true: array of shape (N, 5)
        y_pred: array of shape (N, 5)
        weights: array of shape (5,) — R2_WEIGHTS_VAL
    
    Returns:
        dict with 'weighted_r2', 'rmse', 'mae', 'bias', 'target_mean', 'count'
    """
    if len(y_true) < 2:
        return {'weighted_r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 
                'bias': np.nan, 'target_mean': np.nan, 'count': len(y_true)}
    
    errors = y_pred - y_true
    
    # Weighted R2
    weights_matrix = np.tile(weights, (y_true.shape[0], 1))
    weighted_sum = np.sum(weights_matrix * y_true)
    total_weight = np.sum(weights_matrix)
    y_bar_w = weighted_sum / total_weight
    ss_res = np.sum(weights_matrix * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights_matrix * (y_true - y_bar_w) ** 2)
    weighted_r2 = 1 - (ss_res / ss_tot)
    
    # Weighted RMSE, MAE, Bias
    per_rmse = np.sqrt(np.mean(errors**2, axis=0))
    per_mae = np.mean(np.abs(errors), axis=0)
    per_bias = np.mean(errors, axis=0)
    
    weighted_rmse = np.sum(per_rmse * weights)
    weighted_mae = np.sum(per_mae * weights)
    weighted_bias = np.sum(per_bias * weights)
    
    # Target mean (weighted)
    target_mean = np.mean(y_true[:, 4])  # Dry_Total_g mean
    
    return {
        'weighted_r2': weighted_r2,
        'rmse': weighted_rmse,
        'mae': weighted_mae,
        'bias': weighted_bias,
        'target_mean': target_mean,
        'count': len(y_true),
    }


def collect_all_oof_predictions():
    """
    Collect out-of-fold predictions for date-location grouped CV across all 5 seeds.
    Same training procedure as experiment_2, but we save all OOF predictions.
    """
    df = get_df()
    all_oof_preds = []
    all_oof_targets = []
    all_oof_indices = []
    
    for seed in SEEDS:
        set_seed(seed, deterministic=True)
        print(f"\n--- Seed {seed} ---")
        
        splits = get_date_location_grouped_splits(df, seed)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {fold_idx+1}/5: training...")
            
            metrics = train_model(train_idx, val_idx, EMBED_DIR, seed)
            
            # Store OOF predictions with their original indices
            all_oof_preds.append(metrics['preds'])
            all_oof_targets.append(metrics['targets'])
            all_oof_indices.append(val_idx)
    
    # Concatenate all OOF predictions
    all_oof_preds = np.concatenate(all_oof_preds, axis=0)
    all_oof_targets = np.concatenate(all_oof_targets, axis=0)
    all_oof_indices = np.concatenate(all_oof_indices, axis=0)
    
    return all_oof_preds, all_oof_targets, all_oof_indices, df


def collect_single_oof_predictions(df, seed):
    """
    Collect OOF predictions for a single seed (used internally for per-seed aggregation).
    """
    set_seed(seed, deterministic=True)
    splits = get_date_location_grouped_splits(df, seed)
    
    oof_preds = []
    oof_targets = []
    oof_indices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {fold_idx+1}/5")
        metrics = train_model(train_idx, val_idx, EMBED_DIR, seed)
        oof_preds.append(metrics['preds'])
        oof_targets.append(metrics['targets'])
        oof_indices.append(val_idx)
    
    return {
        'preds': np.concatenate(oof_preds, axis=0),
        'targets': np.concatenate(oof_targets, axis=0),
        'indices': np.concatenate(oof_indices, axis=0),
    }


def date_to_period(date, unique_dates):
    """Classify a date into Early/Middle/Late period."""
    sorted_dates = sorted(unique_dates)
    n = len(sorted_dates)
    split1 = n // 3
    split2 = 2 * n // 3
    if date in sorted_dates[:split1]:
        return 'Early'
    elif date in sorted_dates[split1:split2]:
        return 'Middle'
    else:
        return 'Late'


def get_biomass_bins(df):
    """Create low/medium/high biomass bins."""
    quantiles = df['Dry_Total_g'].quantile([0, 1/3, 2/3, 1.0])
    bins = [quantiles.iloc[0], quantiles.iloc[1], quantiles.iloc[2], quantiles.iloc[3]]
    labels = ['Low biomass', 'Medium biomass', 'High biomass']
    df['biomass_bin'] = pd.cut(df['Dry_Total_g'], bins=bins, labels=labels, include_lowest=True)
    return df


# ============================================================
# TABLE GENERATION
# ============================================================

def table_17_error_by_biomass_range(df, oof_preds, oof_targets, oof_indices):
    """
    Table 17: Error by biomass range.
    Bins by Dry_Total_g quantiles -> Low/Medium/High.
    """
    print("\n" + "=" * 80)
    print("TABLE 17: Error by Biomass Range")
    print("=" * 80)
    
    # Assign biomass bins to the original dataframe
    df = get_biomass_bins(df)
    
    # Create index -> bin mapping
    index_to_bin = {idx: row['biomass_bin'] for idx, row in df.iterrows()}
    
    bin_names = ['Low biomass', 'Medium biomass', 'High biomass']
    rows = []
    
    for bin_name in bin_names:
        # Find which OOF samples belong to this bin
        mask = np.array([index_to_bin.get(idx, '') == bin_name for idx in oof_indices])
        
        if mask.sum() < 2:
            continue
        
        sub_preds = oof_preds[mask]
        sub_targets = oof_targets[mask]
        
        metrics = compute_subgroup_metrics(sub_targets, sub_preds, CFG.R2_WEIGHTS_VAL)
        
        rows.append({
            'Biomass range': bin_name,
            'Samples': f"{int(metrics['count'])}",
            'Target mean': f"{metrics['target_mean']:.2f}",
            'Weighted R2 ↑': f"{metrics['weighted_r2']:.4f}" if not np.isnan(metrics['weighted_r2']) else 'N/A',
            'RMSE ↓': f"{metrics['rmse']:.4f}" if not np.isnan(metrics['rmse']) else 'N/A',
            'MAE ↓': f"{metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else 'N/A',
            'Bias': f"{metrics['bias']:.4f}" if not np.isnan(metrics['bias']) else 'N/A',
        })
    
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS_DIR, 'table_17.csv'), index=False)
    print(result.to_string(index=False))
    return result


def table_18_error_by_temporal_period(df, oof_preds, oof_targets, oof_indices):
    """
    Table 18: Error by temporal period.
    Uses the same Early/Middle/Late periods as leave-one-period-out.
    """
    print("\n" + "=" * 80)
    print("TABLE 18: Error by Temporal Period")
    print("=" * 80)
    
    unique_dates = df['Sampling_Date'].unique()
    
    # Create index -> period mapping
    index_to_period = {}
    for idx, row in df.iterrows():
        index_to_period[idx] = date_to_period(row['Sampling_Date'], unique_dates)
    
    rows = []
    for period in PERIOD_NAMES:
        mask = np.array([index_to_period.get(idx, '') == period for idx in oof_indices])
        
        if mask.sum() < 2:
            continue
        
        sub_preds = oof_preds[mask]
        sub_targets = oof_targets[mask]
        
        metrics = compute_subgroup_metrics(sub_targets, sub_preds, CFG.R2_WEIGHTS_VAL)
        
        rows.append({
            'Period': period,
            'Samples': f"{int(metrics['count'])}",
            'Target mean': f"{metrics['target_mean']:.2f}",
            'Weighted R2 ↑': f"{metrics['weighted_r2']:.4f}" if not np.isnan(metrics['weighted_r2']) else 'N/A',
            'RMSE ↓': f"{metrics['rmse']:.4f}" if not np.isnan(metrics['rmse']) else 'N/A',
            'MAE ↓': f"{metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else 'N/A',
            'Bias': f"{metrics['bias']:.4f}" if not np.isnan(metrics['bias']) else 'N/A',
        })
    
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS_DIR, 'table_18.csv'), index=False)
    print(result.to_string(index=False))
    return result


def table_19_error_by_location(df, oof_preds, oof_targets, oof_indices):
    """
    Table 19: Error by location.
    Groups by State (Tas, NSW, Vic, WA).
    """
    print("\n" + "=" * 80)
    print("TABLE 19: Error by Location")
    print("=" * 80)
    
    # Create index -> location mapping
    index_to_location = {}
    for idx, row in df.iterrows():
        index_to_location[idx] = row['State']
    
    rows = []
    for loc in LOCATIONS:
        mask = np.array([index_to_location.get(idx, '') == loc for idx in oof_indices])
        
        if mask.sum() < 2:
            continue
        
        sub_preds = oof_preds[mask]
        sub_targets = oof_targets[mask]
        
        metrics = compute_subgroup_metrics(sub_targets, sub_preds, CFG.R2_WEIGHTS_VAL)
        
        rows.append({
            'Location': loc,
            'Samples': f"{int(metrics['count'])}",
            'Target mean': f"{metrics['target_mean']:.2f}",
            'Weighted R2 ↑': f"{metrics['weighted_r2']:.4f}" if not np.isnan(metrics['weighted_r2']) else 'N/A',
            'RMSE ↓': f"{metrics['rmse']:.4f}" if not np.isnan(metrics['rmse']) else 'N/A',
            'MAE ↓': f"{metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else 'N/A',
            'Bias': f"{metrics['bias']:.4f}" if not np.isnan(metrics['bias']) else 'N/A',
        })
    
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULTS_DIR, 'table_19.csv'), index=False)
    print(result.to_string(index=False))
    return result


# ============================================================
# MAIN
# ============================================================

def run_experiment_5():
    """
    Main entry point for Experiment 5 (Error Analysis).
    
    Strategy: Run date-location grouped CV across all 5 seeds,
    collect all OOF predictions, then compute metrics stratified by:
    - Biomass range (Table 17)
    - Temporal period (Table 18)
    - Location (Table 19)
    """
    print("=" * 80)
    print("EXPERIMENT 5: Error Analysis")
    print("=" * 80)
    
    # Load data
    df = get_df()
    print(f"Loaded {len(df)} images\n")
    
    # Collect OOF predictions for each seed and aggregate
    print("Collecting OOF predictions across seeds...")
    
    all_preds = []
    all_targets = []
    all_indices = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        result = collect_single_oof_predictions(df, seed)
        all_preds.append(result['preds'])
        all_targets.append(result['targets'])
        all_indices.append(result['indices'])
    
    # Concatenate across seeds (each sample appears in multiple seeds' OOF sets)
    oof_preds = np.concatenate(all_preds, axis=0)
    oof_targets = np.concatenate(all_targets, axis=0)
    oof_indices = np.concatenate(all_indices, axis=0)
    
    print(f"\nTotal OOF predictions collected: {len(oof_preds)}")
    print(f"(= {len(df)} images × {len(SEEDS)} seeds)")
    
    # Generate all 3 tables
    print("\n" + "=" * 80)
    table_17_error_by_biomass_range(df, oof_preds, oof_targets, oof_indices)
    
    print("\n" + "=" * 80)
    table_18_error_by_temporal_period(df, oof_preds, oof_targets, oof_indices)
    
    print("\n" + "=" * 80)
    table_19_error_by_location(df, oof_preds, oof_targets, oof_indices)
    
    print(f"\nAll tables saved to {RESULTS_DIR}/")
    
    # Save all OOF predictions for potential downstream use
    save_data = {
        'preds': oof_preds,
        'targets': oof_targets,
        'indices': oof_indices,
    }
    torch.save(save_data, os.path.join(RESULTS_DIR, 'oof_predictions.pt'))
    print(f"OOF predictions saved to {RESULTS_DIR}/oof_predictions.pt")


if __name__ == '__main__':
    run_experiment_5()