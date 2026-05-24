"""
analyze_results.py — Comprehensive results analysis for Experiment 2

Loads full_results.pt from each experiment variant and generates:
  1. Enhanced Tables 8-11 with mean ± std for every metric
  2. Per-seed median stopping epoch tables
  3. Convergence statistics (best_epoch distribution)
  4. Per-state ground-truth statistics and visualizations
  5. Per-state × period location statistics

Usage:
    python analyze_results.py
"""

import os, sys, pickle, json, warnings, gc
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── numpy compatibility shim for files saved with numpy 2.x ──
import numpy as _np
from types import ModuleType as _MT
if not hasattr(_np, '_core'):
    _core_mod = _MT('numpy._core')
    for _attr in dir(_np.core):
        try:
            setattr(_core_mod, _attr, getattr(_np.core, _attr))
        except Exception:
            pass
    _core_mod.__file__ = getattr(_np.core, '__file__', None)
    sys.modules['numpy._core'] = _core_mod
    for _name in dir(_np.core):
        _obj = getattr(_np.core, _name)
        if isinstance(_obj, _MT):
            sys.modules[f'numpy._core.{_name}'] = _obj

import torch

from configs.cfg import CFG

def get_df():
    """Inline copy of get_df to avoid importing timm."""
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_wide = df_wide[['image_path'] + CFG.ALL_TARGET_COLS]
    aux_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    df_aux = df_long[aux_cols].drop_duplicates().reset_index(drop=True)
    df_wide = df_wide.merge(df_aux, on='image_path', how='left')
    df_wide['State_idx'],   _   = pd.factorize(df_wide['State'])
    df_wide['Species_idx'], _ = pd.factorize(df_wide['Species'])
    df_wide['Sampling_Date'] = pd.to_datetime(df_wide['Sampling_Date'])
    df_wide['day_of_year'] = df_wide['Sampling_Date'].dt.dayofyear
    df_wide['day_sin'] = np.sin(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['day_cos'] = np.cos(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['group'] = df_wide['State'].astype(str) + "_" + df_wide['Sampling_Date'].astype(str)
    df_wide['biomass_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False)
    df_wide['Weighted_g'] = sum(
        df_wide[col] * w for col, w in zip(CFG.ALL_TARGET_COLS, CFG.R2_WEIGHTS_VAL))
    return df_wide

# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_DIRS = [
    'results/experiment_2',
    'results/experiment_2_derive_dead',
    'results/experiment_2_derive_dead_clover',
]
OUTPUT_DIR = 'results/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_SHORT = ['Green', 'Dead', 'Clover', 'GDM', 'Total']
R2_WEIGHTS = CFG.R2_WEIGHTS_VAL  # [0.1, 0.1, 0.1, 0.2, 0.5]

SPLIT_DISPLAY_NAMES = {
    'random_stratified': 'Random stratified 5-fold CV',
    'date_grouped': 'Date-grouped 5-fold CV',
    'date_location_grouped': 'Date-location grouped 5-fold CV',
    'date_location_grouped_splits_weighted': 'Date-location grouped (weighted stratify)',
    'leave_one_period_out': 'Leave-one-period-out',
    'leave_one_state_out': 'Leave-one-state-out',
}

LOPO_PERIOD_NAMES = ['Early', 'Middle', 'Late']

# ============================================================
# LOADING HELPERS
# ============================================================

def load_results(path):
    """Load full_results.pt with numpy compatibility (using torch.load)."""
    return torch.load(path, map_location='cpu', weights_only=False)


def aggregate_fold_list(fold_results):
    """Aggregate fold-level results into mean±std for all metrics."""
    n = len(fold_results)
    if n == 0:
        return {}
    
    # Weighted R2
    w_r2_vals = np.array([r['weighted_r2'] for r in fold_results])
    
    # Per-target R2 (dicts)
    t_r2_vals = {tk: np.array([r['per_target_r2'][tk] for r in fold_results]) 
                 for tk in TARGET_NAMES}
    
    # Per-target arrays
    p_rmse = np.stack([r['per_rmse'] for r in fold_results])  # (n, 5)
    p_mae  = np.stack([r['per_mae'] for r in fold_results])
    p_bias = np.stack([r['per_bias'] for r in fold_results])
    
    # Best epoch
    epochs = np.array([r['best_epoch'] for r in fold_results])
    
    return {
        'weighted_r2_mean': np.mean(w_r2_vals),
        'weighted_r2_std':  np.std(w_r2_vals, ddof=1),
        'weighted_r2_min':  np.min(w_r2_vals),
        'weighted_r2_max':  np.max(w_r2_vals),
        'per_target_r2_mean': {tk: np.mean(t_r2_vals[tk]) for tk in TARGET_NAMES},
        'per_target_r2_std':  {tk: np.std(t_r2_vals[tk], ddof=1) for tk in TARGET_NAMES},
        'per_rmse_mean': np.mean(p_rmse, axis=0),  # (5,)
        'per_rmse_std':  np.std(p_rmse, axis=0, ddof=1),
        'per_mae_mean':  np.mean(p_mae, axis=0),
        'per_mae_std':   np.std(p_mae, axis=0, ddof=1),
        'per_bias_mean': np.mean(p_bias, axis=0),
        'per_bias_std':  np.std(p_bias, axis=0, ddof=1),
        'best_epoch_median': np.median(epochs),
        'best_epoch_mean':   np.mean(epochs),
        'best_epoch_std':    np.std(epochs, ddof=1),
        'best_epoch_min':    np.min(epochs),
        'best_epoch_max':    np.max(epochs),
        'n_results': n,
    }


def format_mean_std(mean, std, decimals=4):
    """Format as 'mean ± std'."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ============================================================
# TABLE 8: Protocol Comparison (enhanced)
# ============================================================

def generate_table_8(all_results, label):
    """Enhanced Table 8 with mean±std for each protocol."""
    protocol_order = [
        'random_stratified', 'date_grouped', 'date_location_grouped',
        'date_location_grouped_splits_weighted', 'leave_one_period_out',
        'leave_one_state_out'
    ]
    
    rows = []
    for protocol in protocol_order:
        if protocol not in all_results:
            continue
        fr_list = all_results[protocol]['fold_results']
        agg = aggregate_fold_list(fr_list)
        
        rows.append({
            'Protocol': SPLIT_DISPLAY_NAMES.get(protocol, protocol),
            'N_results': agg['n_results'],
            'Weighted R2': format_mean_std(agg['weighted_r2_mean'], agg['weighted_r2_std']),
            'R2 range': f"[{agg['weighted_r2_min']:.4f}, {agg['weighted_r2_max']:.4f}]",
            'RMSE Total': f"{agg['per_rmse_mean'][4]:.2f} ± {agg['per_rmse_std'][4]:.2f}",
            'MAE Total':  f"{agg['per_mae_mean'][4]:.2f} ± {agg['per_mae_std'][4]:.2f}",
            'Bias Total': f"{agg['per_bias_mean'][4]:.3f} ± {agg['per_bias_std'][4]:.3f}",
        })
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, f'{label}_table_8_enhanced.csv')
    df.to_csv(path, index=False)
    
    print(f"\n{'='*70}")
    print(f"TABLE 8 (enhanced) — {label}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    return df


# ============================================================
# TABLE 9: Per-target R2 × Protocol (enhanced)
# ============================================================

def generate_table_9(all_results, label):
    """Enhanced Table 9: per-target R2 mean±std per protocol."""
    protocol_order = [
        'random_stratified', 'date_grouped', 'date_location_grouped',
        'date_location_grouped_splits_weighted', 'leave_one_period_out',
        'leave_one_state_out'
    ]
    
    rows = []
    for protocol in protocol_order:
        if protocol not in all_results:
            continue
        fr_list = all_results[protocol]['fold_results']
        agg = aggregate_fold_list(fr_list)
        
        row = {'Protocol': SPLIT_DISPLAY_NAMES.get(protocol, protocol)}
        for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
            row[f'{ts} R2'] = format_mean_std(
                agg['per_target_r2_mean'][tk], agg['per_target_r2_std'][tk])
        row['Weighted R2'] = format_mean_std(
            agg['weighted_r2_mean'], agg['weighted_r2_std'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, f'{label}_table_9_enhanced.csv')
    df.to_csv(path, index=False)
    
    print(f"\n{'='*70}")
    print(f"TABLE 9 (enhanced) — {label}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    return df


# ============================================================
# TABLE 10: LOPO (enhanced with std)
# ============================================================

def generate_table_10(all_results, label):
    """Enhanced Table 10: LOPO with mean±std per period."""
    lopo = all_results.get('leave_one_period_out')
    if lopo is None:
        return None
    
    # Group fold results by period
    fr_by_period = {}
    for fr in lopo['fold_results']:
        period = LOPO_PERIOD_NAMES[fr['fold']]
        if period not in fr_by_period:
            fr_by_period[period] = []
        fr_by_period[period].append(fr)
    
    rows = []
    for period in LOPO_PERIOD_NAMES:
        if period not in fr_by_period:
            continue
        agg = aggregate_fold_list(fr_by_period[period])
        train_str = {'Early': 'Middle + Late', 'Middle': 'Early + Late', 'Late': 'Early + Middle'}[period]
        
        rows.append({
            'Held-out': period,
            'Training': train_str,
            'N': agg['n_results'],
            'Weighted R2': format_mean_std(agg['weighted_r2_mean'], agg['weighted_r2_std']),
            'RMSE (Total)': f"{agg['per_rmse_mean'][4]:.2f} ± {agg['per_rmse_std'][4]:.2f}",
            'MAE (Total)':  f"{agg['per_mae_mean'][4]:.2f} ± {agg['per_mae_std'][4]:.2f}",
            'Bias (Total)': f"{agg['per_bias_mean'][4]:.3f} ± {agg['per_bias_std'][4]:.3f}",
        })
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, f'{label}_table_10_enhanced.csv')
    df.to_csv(path, index=False)
    
    print(f"\n{'='*70}")
    print(f"TABLE 10 (enhanced) — {label}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    return df


# ============================================================
# TABLE 11: Per-target R2 × LOPO period (enhanced with std)
# ============================================================

def generate_table_11(all_results, label):
    """Enhanced Table 11: per-target R2 mean±std per LOPO period."""
    lopo = all_results.get('leave_one_period_out')
    if lopo is None:
        return None
    
    fr_by_period = {}
    for fr in lopo['fold_results']:
        period = LOPO_PERIOD_NAMES[fr['fold']]
        if period not in fr_by_period:
            fr_by_period[period] = []
        fr_by_period[period].append(fr)
    
    rows = []
    for period in LOPO_PERIOD_NAMES:
        if period not in fr_by_period:
            continue
        agg = aggregate_fold_list(fr_by_period[period])
        
        row = {'Held-out period': period}
        for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
            row[f'{ts} R2'] = format_mean_std(
                agg['per_target_r2_mean'][tk], agg['per_target_r2_std'][tk])
        row['Weighted R2'] = format_mean_std(
            agg['weighted_r2_mean'], agg['weighted_r2_std'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, f'{label}_table_11_enhanced.csv')
    df.to_csv(path, index=False)
    
    print(f"\n{'='*70}")
    print(f"TABLE 11 (enhanced) — {label}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    return df


# ============================================================
# STOPPING EPOCH ANALYSIS
# ============================================================

def generate_stopping_epoch_table(all_results, label):
    """Per-seed median stopping epoch + per-protocol summary."""
    protocol_order = [
        'random_stratified', 'date_grouped', 'date_location_grouped',
        'date_location_grouped_splits_weighted', 'leave_one_period_out',
        'leave_one_state_out'
    ]
    
    print(f"\n{'='*70}")
    print(f"STOPPING EPOCH ANALYSIS — {label}")
    print(f"{'='*70}")
    
    all_rows = []
    for protocol in protocol_order:
        if protocol not in all_results:
            continue
        fr_list = all_results[protocol]['fold_results']
        
        # Group by seed
        from collections import defaultdict
        by_seed = defaultdict(list)
        for fr in fr_list:
            by_seed[fr['seed']].append(fr['best_epoch'])
        
        seed_rows = []
        for seed in sorted(by_seed.keys()):
            epochs = by_seed[seed]
            median_epoch = np.median(epochs)
            seed_rows.append({
                'seed': seed,
                'median_epoch': median_epoch,
                'min_epoch': min(epochs),
                'max_epoch': max(epochs),
            })
            all_rows.append({
                'Protocol': SPLIT_DISPLAY_NAMES.get(protocol, protocol),
                'Seed': seed,
                'Median epoch': int(median_epoch),
                'Min': min(epochs),
                'Max': max(epochs),
            })
        
        # Per-protocol summary
        seed_medians = [sr['median_epoch'] for sr in seed_rows]
        print(f"  {SPLIT_DISPLAY_NAMES.get(protocol, protocol)}:")
        print(f"    Per-seed medians: {seed_medians}")
        print(f"    Across-seeds: mean={np.mean(seed_medians):.1f} ± {np.std(seed_medians, ddof=1):.1f}")
        
        # Also print all-folds stats
        all_epochs = [fr['best_epoch'] for fr in fr_list]
        print(f"    All folds:   min={min(all_epochs)}, max={max(all_epochs)}, "
              f"median={np.median(all_epochs):.0f}, mean={np.mean(all_epochs):.1f}±{np.std(all_epochs, ddof=1):.1f}")
    
    df = pd.DataFrame(all_rows)
    path = os.path.join(OUTPUT_DIR, f'{label}_stopping_epochs.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")
    return df


# ============================================================
# PER-TARGET ERROR BREAKDOWN (across all protocols)
# ============================================================

def generate_error_breakdown(all_results, label):
    """Which targets are hardest overall across all protocols."""
    print(f"\n{'='*70}")
    print(f"ERROR BREAKDOWN — {label}")
    print(f"{'='*70}")
    
    all_fr = []
    for protocol, data in all_results.items():
        for fr in data['fold_results']:
            fr['protocol'] = protocol
            all_fr.append(fr)
    
    agg = aggregate_fold_list(all_fr)
    
    print(f"  Overall across {agg['n_results']} folds (all protocols × seeds):")
    print(f"  Weighted R2: {agg['weighted_r2_mean']:.4f} ± {agg['weighted_r2_std']:.4f}")
    print()
    print(f"  {'Target':<12} {'R2 mean':<12} {'R2 std':<12} {'RMSE mean':<10} {'MAE mean':<10} {'Bias mean':<10}")
    print(f"  {'-'*66}")
    for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
        print(f"  {ts:<12} {agg['per_target_r2_mean'][tk]:<12.4f} {agg['per_target_r2_std'][tk]:<12.4f} "
              f"{agg['per_rmse_mean'][i]:<10.2f} {agg['per_mae_mean'][i]:<10.2f} {agg['per_bias_mean'][i]:<10.3f}")
    
    # Save
    rows = []
    for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
        rows.append({
            'Target': ts,
            'R2_mean': agg['per_target_r2_mean'][tk],
            'R2_std': agg['per_target_r2_std'][tk],
            'RMSE_mean': agg['per_rmse_mean'][i],
            'MAE_mean': agg['per_mae_mean'][i],
            'Bias_mean': agg['per_bias_mean'][i],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, f'{label}_error_breakdown.csv')
    df.to_csv(path, index=False)
    return df


# ============================================================
# PER-STATE GROUND-TRUTH STATISTICS
# ============================================================

def generate_per_state_stats(df_wide):
    """Ground-truth statistics per State for all 5 targets."""
    print(f"\n{'='*70}")
    print("PER-STATE GROUND-TRUTH STATISTICS")
    print(f"{'='*70}")
    
    # Summary stats
    state_groups = df_wide.groupby('State')
    
    print(f"\nStates: {sorted(df_wide['State'].unique())}")
    print(f"Total samples: {len(df_wide)}")
    
    # Count and nonzero count per state per target
    print(f"\n{'State':<8} {'N':<6} {'N_missions':<12}", end='')
    for ts in TARGET_SHORT:
        print(f'{ts + "_mean":<10} {ts + "_nonzero":<10} ', end='')
    print()
    print('-' * 80)
    
    rows = []
    for state in sorted(df_wide['State'].unique()):
        sub = df_wide[df_wide['State'] == state]
        n = len(sub)
        n_miss = sub['group'].nunique()
        row = {'State': state, 'N': n, 'N_missions': n_miss}
        print(f"{state:<8} {n:<6} {n_miss:<12}", end='')
        for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
            vals = sub[tk].values
            nonzero = np.sum(vals > 0)
            row[f'{ts}_mean'] = np.mean(vals)
            row[f'{ts}_std'] = np.std(vals, ddof=1)
            row[f'{ts}_nonzero'] = nonzero
            row[f'{ts}_nonzero_pct'] = nonzero / n * 100
            print(f"{np.mean(vals):<10.2f} {nonzero:<10} ", end='')
        print()
        rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'per_state_statistics.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")
    return df


# ============================================================
# PER-STATE VISUALIZATIONS
# ============================================================

def plot_per_state_distributions(df_wide):
    """Generate boxplot and violin plot for each target by State."""
    states = sorted(df_wide['State'].unique())
    
    # ── Boxplots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
        data = [df_wide[df_wide['State'] == s][tk].values for s in states]
        bp = axes[i].boxplot(data, labels=states, patch_artist=True, showmeans=True)
        # Color boxes
        for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(states)))):
            patch.set_facecolor(color)
        axes[i].set_title(f'{ts}', fontsize=12)
        axes[i].set_ylabel('grams')
        axes[i].tick_params(axis='x', rotation=45)
    axes[-1].set_visible(False)
    fig.suptitle('Target Distribution by State (Boxplot)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'per_state_boxplots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")
    
    # ── Violin plots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
        data = [df_wide[df_wide['State'] == s][tk].values for s in states]
        parts = axes[i].violinplot(data, positions=range(len(states)), 
                                    showmeans=True, showmedians=True)
        axes[i].set_xticks(range(len(states)))
        axes[i].set_xticklabels(states, rotation=45)
        axes[i].set_title(f'{ts}', fontsize=12)
        axes[i].set_ylabel('grams')
    axes[-1].set_visible(False)
    fig.suptitle('Target Density by State (Violin Plot)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'per_state_violins.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")
    
    # ── Correlation heatmaps per State ──
    n_states = len(states)
    n_cols = 3
    n_rows = int(np.ceil(n_states / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    for i, state in enumerate(states):
        corr = df_wide[df_wide['State'] == state][TARGET_NAMES].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                    ax=axes[i], cbar=i == 0,
                    xticklabels=TARGET_SHORT, yticklabels=TARGET_SHORT)
        axes[i].set_title(f'{state} — Target Correlations')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Per-State Target Correlation Matrices', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'per_state_correlations.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")
    
    # ── Zero-inflation bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    nonzero_pcts = []
    for state in states:
        sub = df_wide[df_wide['State'] == state]
        row = [np.sum(sub[tk] > 0) / len(sub) * 100 for tk in TARGET_NAMES]
        nonzero_pcts.append(row)
    nonzero_pcts = np.array(nonzero_pcts)  # (n_states, 5)
    
    x = np.arange(len(states))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, 5))
    for i, ts in enumerate(TARGET_SHORT):
        ax.bar(x + i * width, nonzero_pcts[:, i], width, label=ts, color=colors[i])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(states, rotation=45)
    ax.set_ylabel('% Non-zero samples')
    ax.set_title('Proportion of Non-zero Samples per Target by State')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'per_state_nonzero_pct.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================
# PER-LOCATION × PERIOD STATISTICS
# ============================================================

def plot_location_period_distributions(df_wide):
    """Statistics by location (State) and temporal period."""
    # Add period labels
    unique_dates = sorted(df_wide['Sampling_Date'].unique())
    n_dates = len(unique_dates)
    split1 = n_dates // 3
    split2 = 2 * n_dates // 3
    early_dates = set(unique_dates[:split1])
    middle_dates = set(unique_dates[split1:split2])
    late_dates = set(unique_dates[split2:])
    
    def get_period(date):
        if date in early_dates:
            return 'Early'
        elif date in middle_dates:
            return 'Middle'
        else:
            return 'Late'
    
    df_wide['period'] = df_wide['Sampling_Date'].map(get_period)
    
    # ── Mean target by State × Period heatmap ──
    for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
        pivot = df_wide.pivot_table(values=tk, index='State', columns='period', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'grams'})
        ax.set_title(f'Mean {ts} by State × Period')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'state_period_mean_{ts}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ── Sample count by State × Period ──
    count_pivot = df_wide.pivot_table(values='Dry_Total_g', index='State', 
                                       columns='period', aggfunc='count')
    fig, ax = plt.subplots(figsize=(6, max(4, len(count_pivot) * 0.5)))
    sns.heatmap(count_pivot, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                cbar_kws={'label': 'N samples'})
    ax.set_title('Sample Count by State × Period')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'state_period_sample_count.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("LOCATION × PERIOD STATISTICS")
    print(f"{'='*70}")
    print(f"\nSample counts:\n{count_pivot.to_string()}")
    
    return df_wide  # with period column added


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 2: COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 70)
    
    # ── Load ground-truth data ──
    print("\nLoading ground-truth data...")
    df_wide = get_df()
    print(f"Loaded {len(df_wide)} images, {df_wide['State'].nunique()} states")
    
    # ── Per-state ground-truth analysis ──
    generate_per_state_stats(df_wide)
    plot_per_state_distributions(df_wide)
    plot_location_period_distributions(df_wide)
    
    # ── Per-experiment results analysis ──
    for exp_dir in EXPERIMENT_DIRS:
        label = os.path.basename(exp_dir)
        pt_path = os.path.join(exp_dir, 'full_results.pt')
        
        if not os.path.exists(pt_path):
            print(f"\nWARNING: {pt_path} not found, skipping")
            continue
        
        print(f"\n{'#'*70}")
        print(f"Loading results: {exp_dir}")
        print(f"{'#'*70}")
        
        try:
            all_results = load_results(pt_path)
        except Exception as e:
            print(f"ERROR loading {pt_path}: {e}")
            continue
        
        print(f"  Protocols: {list(all_results.keys())}")
        for proto, data in all_results.items():
            n_folds = len(data['fold_results'])
            n_seeds = len(data['seed_results'])
            print(f"    {proto}: {n_folds} fold-results, {n_seeds} seeds")
        
        # Generate all tables
        generate_table_8(all_results, label)
        generate_table_9(all_results, label)
        generate_table_10(all_results, label)
        generate_table_11(all_results, label)
        generate_stopping_epoch_table(all_results, label)
        generate_error_breakdown(all_results, label)
    
    print(f"\n{'='*70}")
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()