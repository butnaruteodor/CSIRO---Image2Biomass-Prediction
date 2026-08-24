"""
experiment_5.py — Error Analysis

Analyzes where the model fails by evaluating out-of-fold predictions
grouped by target, biomass range, temporal period, and location.
Uses date-location grouped CV (the best/primary protocol from Experiment 2).

Produces:
  1. Per-target error table (R2, RMSE, MAE, Bias for each of 5 targets)
  2. Error by Dry_Total_g biomass range (Low/Medium/High)
  3. Error by temporal period (Early/Middle/Late)
  4. Error by state (Tas/NSW/Vic/WA)
  5. Predicted vs true scatter plot (5 targets, colored by period)

Usage:
    python experiment_5.py           # After experiment_2.py has run

Output:
    results/experiment_5/
        table_per_target.csv
        table_biomass_range.csv
        table_temporal_period.csv
        table_location.csv
        pred_vs_true.png
"""

import os, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.cfg import CFG
from configs.deterministic import set_seed, seed_worker, get_generator
from dataset.preprocess_data import get_df, EmbeddingAugmentationDataset
from models.models import BiomassSimpleMLP
from utils.eval import global_weighted_r2_score, per_target_r2_score

# Reuse experiment_2's split functions
from experiment_2 import (
    get_date_location_grouped_splits,
    SEEDS, EMBED_DIR, FEATURE_DIM, N_AUG, DEVICE, TARGET_NAMES
)

RESULTS_DIR = 'results/experiment_5'
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SHORT = ['Green', 'Dead', 'Clover', 'GDM', 'Total']
PERIOD_NAMES = ['Early', 'Middle', 'Late']
LOCATIONS = ['Tas', 'NSW', 'Vic', 'WA']
R2_WEIGHTS = CFG.R2_WEIGHTS_VAL  # [0.1, 0.1, 0.1, 0.2, 0.5]

# Path to pre-trained fold models from experiment_2
FOLD_MODEL_DIR = 'results/experiment_2/date_location_grouped'


# ============================================================
# OOF PREDICTION COLLECTION (using pre-trained fold models)
# ============================================================

@torch.no_grad()
def predict_fold(model, loader):
    """Run inference on a validation loader."""
    model.eval()
    all_preds = []
    all_labels = []
    for feats, targets in tqdm(loader, desc='  predict', leave=False):
        feats = feats.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
        preds = torch.stack([p_green, p_dead, p_clover, p_gdm, p_total], dim=1).squeeze(-1)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def get_oof_for_seed(df, seed):
    """
    Load pre-trained fold models for a seed, run inference on validation sets,
    and assemble the full OOF (357, 5) array.
    
    Returns dict with 'preds', 'targets', 'indices' (all shape (357, 5) or (357,)).
    """
    set_seed(seed, deterministic=True)
    splits = get_date_location_grouped_splits(df, seed)
    
    n_total = len(df)
    n_targets = 5
    oof_preds = np.full((n_total, n_targets), np.nan)
    oof_targets = np.full((n_total, n_targets), np.nan)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        ckpt_path = os.path.join(FOLD_MODEL_DIR, f'fold_{fold_idx}_seed_{seed}.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Pre-trained model not found: {ckpt_path}\n"
                f"Run experiment_2.py first to generate fold models."
            )
        
        # Load model
        model = BiomassSimpleMLP(FEATURE_DIM).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()
        
        # Create validation dataset
        val_set = EmbeddingAugmentationDataset(val_idx, EMBED_DIR, n_aug=N_AUG, is_train=False)
        val_loader = DataLoader(
            val_set, batch_size=8, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # Predict
        preds, targets = predict_fold(model, val_loader)
        oof_preds[val_idx] = preds
        oof_targets[val_idx] = targets
        
        del model, val_loader, val_set
        gc.collect()
        torch.cuda.empty_cache()
    
    # Verify completeness
    missing = np.isnan(oof_preds).any(axis=1)
    if missing.any():
        raise RuntimeError(f"{missing.sum()}/{n_total} samples missing from OOF array!")
    
    return {
        'preds': oof_preds,       # (357, 5)
        'targets': oof_targets,   # (357, 5)
        'indices': np.arange(n_total),
    }


# ============================================================
# PER-SEED METRIC COMPUTATION
# ============================================================

def compute_per_target_metrics(y_true, y_pred):
    """
    Compute R2, RMSE, MAE, Bias for each of the 5 targets.
    
    Returns dict with keys 'r2', 'rmse', 'mae', 'bias', each shape (5,).
    """
    r2_dict = per_target_r2_score(y_true, y_pred)
    r2 = np.array([r2_dict[t] for t in TARGET_NAMES])  # (5,)
    
    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors**2, axis=0))   # (5,)
    mae = np.mean(np.abs(errors), axis=0)         # (5,)
    bias = np.mean(errors, axis=0)                # (5,)
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias}


def compute_subgroup_metrics(y_true, y_pred):
    """
    Compute weighted R2 + Dry_Total RMSE/MAE/Bias for a subgroup.
    
    Args:
        y_true: (N, 5)
        y_pred: (N, 5)
    
    Returns dict with 'weighted_r2', 'rmse_total', 'mae_total', 'bias_total', 'count'.
    """
    if len(y_true) < 2:
        return {'weighted_r2': np.nan, 'rmse_total': np.nan, 'mae_total': np.nan,
                'bias_total': np.nan, 'count': len(y_true)}
    
    # Weighted R2 (global, all 5 targets)
    weights_matrix = np.tile(R2_WEIGHTS, (y_true.shape[0], 1))
    weighted_sum = np.sum(weights_matrix * y_true)
    total_weight = np.sum(weights_matrix)
    y_bar_w = weighted_sum / total_weight
    ss_res = np.sum(weights_matrix * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights_matrix * (y_true - y_bar_w) ** 2)
    weighted_r2 = 1 - (ss_res / ss_tot)
    
    # Dry_Total (index 4) RMSE, MAE, Bias
    errors_total = y_pred[:, 4] - y_true[:, 4]
    rmse_total = np.sqrt(np.mean(errors_total**2))
    mae_total = np.mean(np.abs(errors_total))
    bias_total = np.mean(errors_total)
    
    return {
        'weighted_r2': weighted_r2,
        'rmse_total': rmse_total,
        'mae_total': mae_total,
        'bias_total': bias_total,
        'count': len(y_true),
    }


# ============================================================
# DATE-TO-PERIOD HELPER
# ============================================================

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


# ============================================================
# TABLE 1: Per-target error
# ============================================================

def table_per_target_error(seed_results):
    """
    For each target, report R2, RMSE, MAE, Bias as mean±std across seeds.
    Metrics computed on full per-seed OOF (357, 5).
    """
    print("\n" + "=" * 80)
    print("TABLE 1: Per-Target Error")
    print("=" * 80)
    
    # Collect per-target metrics for each seed
    n_seeds = len(seed_results)
    all_r2 = np.zeros((n_seeds, 5))
    all_rmse = np.zeros((n_seeds, 5))
    all_mae = np.zeros((n_seeds, 5))
    all_bias = np.zeros((n_seeds, 5))
    
    for i, sr in enumerate(seed_results):
        m = compute_per_target_metrics(sr['targets'], sr['preds'])
        all_r2[i] = m['r2']
        all_rmse[i] = m['rmse']
        all_mae[i] = m['mae']
        all_bias[i] = m['bias']
    
    rows = []
    for j, ts in enumerate(TARGET_SHORT):
        rows.append({
            'Target': ts,
            'R2': f"{np.mean(all_r2[:, j]):.4f} ± {np.std(all_r2[:, j], ddof=1):.4f}",
            'RMSE': f"{np.mean(all_rmse[:, j]):.2f} ± {np.std(all_rmse[:, j], ddof=1):.2f}",
            'MAE': f"{np.mean(all_mae[:, j]):.2f} ± {np.std(all_mae[:, j], ddof=1):.2f}",
            'Bias': f"{np.mean(all_bias[:, j]):.3f} ± {np.std(all_bias[:, j], ddof=1):.3f}",
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_per_target.csv'), index=False)
    print(df.to_string(index=False))
    return df


# ============================================================
# TABLE 2: Error by biomass range
# ============================================================

def table_error_by_biomass_range(df, seed_results):
    """
    Bin samples by Dry_Total_g tertiles → Low/Medium/High.
    For each bin, report per-seed metrics then mean±std across seeds.
    """
    print("\n" + "=" * 80)
    print("TABLE 2: Error by Biomass Range")
    print("=" * 80)
    
    # Create tertile bins
    quantiles = df['Dry_Total_g'].quantile([0, 1/3, 2/3, 1.0])
    bins = [quantiles.iloc[0], quantiles.iloc[1], quantiles.iloc[2], quantiles.iloc[3]]
    bin_labels = ['Low biomass', 'Medium biomass', 'High biomass']
    df['biomass_bin'] = pd.cut(df['Dry_Total_g'], bins=bins, labels=bin_labels, include_lowest=True)
    
    # Per-bin, collect metrics across seeds
    bin_metrics = {b: [] for b in bin_labels}
    
    for sr in seed_results:
        indices = sr['indices']
        for b in bin_labels:
            mask = df.iloc[indices]['biomass_bin'].values == b
            if mask.sum() < 2:
                bin_metrics[b].append({'weighted_r2': np.nan, 'rmse_total': np.nan,
                                       'mae_total': np.nan, 'bias_total': np.nan, 'count': 0})
            else:
                m = compute_subgroup_metrics(sr['targets'][mask], sr['preds'][mask])
                bin_metrics[b].append(m)
    
    rows = []
    for b in bin_labels:
        vals = bin_metrics[b]
        counts = [v['count'] for v in vals]
        r2s = [v['weighted_r2'] for v in vals if not np.isnan(v['weighted_r2'])]
        rmses = [v['rmse_total'] for v in vals if not np.isnan(v['rmse_total'])]
        maes = [v['mae_total'] for v in vals if not np.isnan(v['mae_total'])]
        biases = [v['bias_total'] for v in vals if not np.isnan(v['bias_total'])]
        
        rows.append({
            'Biomass range': b,
            'Samples': f"{int(np.mean(counts))}",
            'Weighted R2': f"{np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f}",
            'RMSE (Total)': f"{np.mean(rmses):.2f} ± {np.std(rmses, ddof=1):.2f}",
            'MAE (Total)': f"{np.mean(maes):.2f} ± {np.std(maes, ddof=1):.2f}",
            'Bias (Total)': f"{np.mean(biases):.3f} ± {np.std(biases, ddof=1):.3f}",
        })
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, 'table_biomass_range.csv'), index=False)
    print(df_out.to_string(index=False))
    return df_out


# ============================================================
# TABLE 3: Error by temporal period
# ============================================================

def table_error_by_temporal_period(df, seed_results):
    """
    Group by Early/Middle/Late period. Report per-seed metrics then mean±std.
    """
    print("\n" + "=" * 80)
    print("TABLE 3: Error by Temporal Period")
    print("=" * 80)
    
    unique_dates = df['Sampling_Date'].unique()
    
    # Per-period, collect metrics across seeds
    period_metrics = {p: [] for p in PERIOD_NAMES}
    
    for sr in seed_results:
        indices = sr['indices']
        for p in PERIOD_NAMES:
            mask = np.array([date_to_period(df.iloc[i]['Sampling_Date'], unique_dates) == p
                             for i in indices])
            if mask.sum() < 2:
                period_metrics[p].append({'weighted_r2': np.nan, 'rmse_total': np.nan,
                                          'mae_total': np.nan, 'bias_total': np.nan, 'count': 0})
            else:
                m = compute_subgroup_metrics(sr['targets'][mask], sr['preds'][mask])
                period_metrics[p].append(m)
    
    rows = []
    for p in PERIOD_NAMES:
        vals = period_metrics[p]
        counts = [v['count'] for v in vals]
        r2s = [v['weighted_r2'] for v in vals if not np.isnan(v['weighted_r2'])]
        rmses = [v['rmse_total'] for v in vals if not np.isnan(v['rmse_total'])]
        maes = [v['mae_total'] for v in vals if not np.isnan(v['mae_total'])]
        biases = [v['bias_total'] for v in vals if not np.isnan(v['bias_total'])]
        
        rows.append({
            'Period': p,
            'Samples': f"{int(np.mean(counts))}",
            'Weighted R2': f"{np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f}",
            'RMSE (Total)': f"{np.mean(rmses):.2f} ± {np.std(rmses, ddof=1):.2f}",
            'MAE (Total)': f"{np.mean(maes):.2f} ± {np.std(maes, ddof=1):.2f}",
            'Bias (Total)': f"{np.mean(biases):.3f} ± {np.std(biases, ddof=1):.3f}",
        })
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, 'table_temporal_period.csv'), index=False)
    print(df_out.to_string(index=False))
    return df_out


# ============================================================
# TABLE 4: Error by location (state)
# ============================================================

def table_error_by_location(df, seed_results):
    """
    Group by State (Tas/NSW/Vic/WA). Report per-seed metrics then mean±std.
    """
    print("\n" + "=" * 80)
    print("TABLE 4: Error by Location")
    print("=" * 80)
    
    # Per-location, collect metrics across seeds
    loc_metrics = {l: [] for l in LOCATIONS}
    
    for sr in seed_results:
        indices = sr['indices']
        for loc in LOCATIONS:
            mask = np.array([df.iloc[i]['State'] == loc for i in indices])
            if mask.sum() < 2:
                loc_metrics[loc].append({'weighted_r2': np.nan, 'rmse_total': np.nan,
                                         'mae_total': np.nan, 'bias_total': np.nan, 'count': 0})
            else:
                m = compute_subgroup_metrics(sr['targets'][mask], sr['preds'][mask])
                loc_metrics[loc].append(m)
    
    rows = []
    for loc in LOCATIONS:
        vals = loc_metrics[loc]
        counts = [v['count'] for v in vals]
        r2s = [v['weighted_r2'] for v in vals if not np.isnan(v['weighted_r2'])]
        rmses = [v['rmse_total'] for v in vals if not np.isnan(v['rmse_total'])]
        maes = [v['mae_total'] for v in vals if not np.isnan(v['mae_total'])]
        biases = [v['bias_total'] for v in vals if not np.isnan(v['bias_total'])]
        
        rows.append({
            'Location': loc,
            'Samples': f"{int(np.mean(counts))}",
            'Weighted R2': f"{np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f}",
            'RMSE (Total)': f"{np.mean(rmses):.2f} ± {np.std(rmses, ddof=1):.2f}",
            'MAE (Total)': f"{np.mean(maes):.2f} ± {np.std(maes, ddof=1):.2f}",
            'Bias (Total)': f"{np.mean(biases):.3f} ± {np.std(biases, ddof=1):.3f}",
        })
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, 'table_location.csv'), index=False)
    print(df_out.to_string(index=False))
    return df_out


# ============================================================
# FIGURE: Predicted vs True scatter (5 targets, colored by period)
# ============================================================

def plot_pred_vs_true(df, seed_results):
    """
    Generate 2×3 small multiples (or 1×5) of predicted vs true for each target.
    Points colored by Early/Middle/Late period. Dashed y=x line.
    Uses the first seed's OOF predictions for the figure.
    """
    print("\n" + "=" * 80)
    print("FIGURE: Predicted vs True Scatter")
    print("=" * 80)
    
    # Use first seed's OOF
    sr = seed_results[0]
    preds = sr['preds']   # (357, 5)
    targets = sr['targets']
    indices = sr['indices']
    
    # Assign period to each sample
    unique_dates = df['Sampling_Date'].unique()
    periods = np.array([date_to_period(df.iloc[i]['Sampling_Date'], unique_dates)
                        for i in indices])
    
    period_colors = {'Early': '#2196F3', 'Middle': '#FF9800', 'Late': '#F44336'}
    period_markers = {'Early': 'o', 'Middle': 's', 'Late': '^'}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for j, (ts, tn) in enumerate(zip(TARGET_SHORT, TARGET_NAMES)):
        ax = axes[j]
        
        for p in PERIOD_NAMES:
            mask = periods == p
            ax.scatter(targets[mask, j], preds[mask, j],
                       c=period_colors[p], marker=period_markers[p],
                       label=p, alpha=0.6, edgecolors='none', s=30)
        
        # y=x line
        all_vals = np.concatenate([targets[:, j], preds[:, j]])
        lims = [np.min(all_vals) - 5, np.max(all_vals) + 5]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel('True (g)', fontsize=11)
        ax.set_ylabel('Predicted (g)', fontsize=11)
        ax.set_title(ts, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add R2 annotation
        r2 = per_target_r2_score(targets, preds)[tn]
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        if j == 0:
            ax.legend(fontsize=9, loc='lower right')
    
    # Hide the last subplot (6th)
    axes[-1].set_visible(False)
    
    fig.suptitle('Predicted vs True Biomass by Target (OOF, date-location grouped CV)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    path = os.path.join(RESULTS_DIR, 'pred_vs_true.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================
# MAIN
# ============================================================

def run_experiment_5():
    """
    Main entry point for Experiment 5 (Error Analysis).
    
    Strategy:
    1. Load pre-trained fold models from experiment_2's date_location_grouped CV
    2. For each seed, assemble OOF (357, 5) predictions
    3. Compute per-seed metrics, report mean±std across seeds
    4. Generate 4 tables + 1 figure
    """
    print("=" * 80)
    print("EXPERIMENT 5: Error Analysis")
    print("=" * 80)
    
    # Load data
    df = get_df()
    print(f"Loaded {len(df)} images\n")
    
    # Verify pre-trained models exist
    expected = sum(1 for s in SEEDS for f in range(5))
    actual = len([f for f in os.listdir(FOLD_MODEL_DIR) if f.endswith('.pt')])
    print(f"Found {actual}/{expected} pre-trained fold models in {FOLD_MODEL_DIR}/")
    if actual < expected:
        print("WARNING: Some fold models missing. Run experiment_2.py first.")
    
    # Collect OOF predictions per seed
    print("\nCollecting OOF predictions from pre-trained fold models...")
    seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        sr = get_oof_for_seed(df, seed)
        seed_results.append(sr)
        # Compute and print overall weighted R2 for this seed
        w_r2 = global_weighted_r2_score(sr['targets'], sr['preds'])
        print(f"  OOF Weighted R2 = {w_r2:.4f}")
    
    # Compute across-seed overall metrics
    all_wr2 = [global_weighted_r2_score(sr['targets'], sr['preds']) for sr in seed_results]
    print(f"\nOverall OOF Weighted R2: {np.mean(all_wr2):.4f} ± {np.std(all_wr2, ddof=1):.4f}")
    
    # Generate all tables
    table_per_target_error(seed_results)
    table_error_by_biomass_range(df, seed_results)
    table_error_by_temporal_period(df, seed_results)
    table_error_by_location(df, seed_results)
    
    # Generate figure
    plot_pred_vs_true(df, seed_results)
    
    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    run_experiment_5()