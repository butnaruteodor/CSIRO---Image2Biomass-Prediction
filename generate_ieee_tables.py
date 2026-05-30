# -*- coding: utf-8 -*-
"""
generate_ieee_tables.py — Generate IEEE-ready tables from experiment 2 full_results.pt

Produces compact tables with mean ± std for all metrics.
Run after experiment_2.py has completed.

Usage:
    python generate_ieee_tables.py

Output:
    results/experiment_2/table_9_with_std.csv  (per-target R² mean ± std)
    results/experiment_2/table_8_with_std.csv  (protocol comparison)
"""

import os, sys, types, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── numpy compatibility shim for files saved with numpy 2.x ──
import numpy as _np
if not hasattr(_np, '_core'):
    _core_mod = types.ModuleType('numpy._core')
    for _attr in dir(_np.core):
        try:
            setattr(_core_mod, _attr, getattr(_np.core, _attr))
        except Exception:
            pass
    _core_mod.__file__ = getattr(_np.core, '__file__', None)
    sys.modules['numpy._core'] = _core_mod
    for _name in dir(_np.core):
        _obj = getattr(_np.core, _name)
        if isinstance(_obj, types.ModuleType):
            sys.modules[f'numpy._core.{_name}'] = _obj

import torch

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = 'results/experiment_2'
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_SHORT = ['Green', 'Dead', 'Clover', 'GDM', 'Total']

SPLIT_DISPLAY_NAMES = {
    'random_stratified': 'Random stratified 5-fold CV',
    'date_grouped': 'Date-grouped 5-fold CV',
    'date_location_grouped': 'Date-location grouped 5-fold CV',
    'date_location_grouped_splits_weighted': 'Date-location grouped (weighted stratify)',
    'leave_one_period_out': 'Leave-one-period-out',
    'leave_one_state_out': 'Leave-one-state-out',
}

PROTOCOL_ORDER = [
    'random_stratified',
    'date_grouped',
    'date_location_grouped',
    'date_location_grouped_splits_weighted',
    'leave_one_period_out',
    'leave_one_state_out',
]


def format_mean_std(mean, std, decimals=4):
    """Format as 'mean ± std' with given decimal places."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def aggregate_seed_results(seed_results):
    """Compute mean ± std across seeds for all per-target R² and weighted R²."""
    w_r2_vals = np.array([r['weighted_r2'] for r in seed_results])
    t_r2_vals = {tk: np.array([r['per_target_r2'][tk] for r in seed_results])
                 for tk in TARGET_NAMES}
    p_rmse = np.stack([r['per_rmse'] for r in seed_results])
    p_mae = np.stack([r['per_mae'] for r in seed_results])
    p_bias = np.stack([r['per_bias'] for r in seed_results])
    return {
        'weighted_r2_mean': np.mean(w_r2_vals),
        'weighted_r2_std': np.std(w_r2_vals, ddof=1),
        'per_target_r2_mean': {tk: np.mean(t_r2_vals[tk]) for tk in TARGET_NAMES},
        'per_target_r2_std': {tk: np.std(t_r2_vals[tk], ddof=1) for tk in TARGET_NAMES},
        'per_rmse_mean': np.mean(p_rmse, axis=0),
        'per_rmse_std': np.std(p_rmse, axis=0, ddof=1),
        'per_mae_mean': np.mean(p_mae, axis=0),
        'per_mae_std': np.std(p_mae, axis=0, ddof=1),
        'per_bias_mean': np.mean(p_bias, axis=0),
        'per_bias_std': np.std(p_bias, axis=0, ddof=1),
    }


def generate_table_9(all_results):
    """
    IEEE-compact Table 9: Per-target R² mean ± std per protocol.
    
    Columns: Validation protocol, Green R², Dead R², Clover R², GDM R², Total R², Weighted R²
    Each cell: value ± std
    """
    rows = []
    for protocol in PROTOCOL_ORDER:
        if protocol not in all_results:
            continue
        seed_results = all_results[protocol]['seed_results']
        agg = aggregate_seed_results(seed_results)
        row = {'Validation protocol': SPLIT_DISPLAY_NAMES.get(protocol, protocol)}
        for i, (tk, ts) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
            row[f'{ts} R²'] = format_mean_std(
                agg['per_target_r2_mean'][tk], agg['per_target_r2_std'][tk])
        row['Weighted R²'] = format_mean_std(
            agg['weighted_r2_mean'], agg['weighted_r2_std'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'table_9_with_std.csv')
    df.to_csv(path, index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 9: Per-Target R² (mean ± std across 5 seeds)")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved to {path}")
    return df


def generate_table_8(all_results):
    """
    IEEE-compact Table 8: Protocol comparison with weighted R², RMSE, MAE, Bias.
    """
    rows = []
    for protocol in PROTOCOL_ORDER:
        if protocol not in all_results:
            continue
        seed_results = all_results[protocol]['seed_results']
        agg = aggregate_seed_results(seed_results)
        rows.append({
            'Validation protocol': SPLIT_DISPLAY_NAMES.get(protocol, protocol),
            'Weighted R²': format_mean_std(agg['weighted_r2_mean'], agg['weighted_r2_std']),
            'RMSE (Total)': f"{agg['per_rmse_mean'][4]:.2f} ± {agg['per_rmse_std'][4]:.2f}",
            'MAE (Total)': f"{agg['per_mae_mean'][4]:.2f} ± {agg['per_mae_std'][4]:.2f}",
            'Bias (Total)': f"{agg['per_bias_mean'][4]:.3f} ± {agg['per_bias_std'][4]:.3f}",
        })
    
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'table_8_with_std.csv')
    df.to_csv(path, index=False)
    
    print("\n" + "=" * 80)
    print("TABLE 8: Protocol Comparison (mean ± std across 5 seeds)")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved to {path}")
    return df


def main():
    pt_path = os.path.join(RESULTS_DIR, 'full_results.pt')
    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found. Run experiment_2.py first.")
        sys.exit(1)
    
    print(f"Loading {pt_path}...")
    all_results = torch.load(pt_path, map_location='cpu', weights_only=False)
    print(f"Loaded {len(all_results)} protocols")
    
    generate_table_8(all_results)
    generate_table_9(all_results)


if __name__ == '__main__':
    main()