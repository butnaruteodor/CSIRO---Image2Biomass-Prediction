#!/usr/bin/env python3
"""
Generate Tables 8/9/10/11 from saved CV results.

Usage:
    python scripts/analysis/experiment_2_tables.py --results results/cv_date_location/full_results.pt --output results/tables
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CFG
from src.evaluation.metrics import global_weighted_r2_score, per_target_r2_score, per_target_rmse, per_target_mae, per_target_bias, TARGET_NAMES
from src.data.preprocessing import get_df

def _total_metric(fold_result, metric_key):
    """Compute total (Dry_Total_g) metric from fold results."""
    targets = np.array(fold_result["targets"])
    preds = np.array(fold_result["preds"])
    idx = TARGET_NAMES.index("Dry_Total_g")
    if metric_key == "per_rmse":
        return np.sqrt(np.mean((targets[:, idx] - preds[:, idx]) ** 2))
    elif metric_key == "per_mae":
        return np.mean(np.abs(targets[:, idx] - preds[:, idx]))
    elif metric_key == "per_bias":
        return np.mean(preds[:, idx] - targets[:, idx])
    return 0.0

def gen_table_8(all_results, out_dir):
    """Table 8: Cross-protocol comparison."""
    rows = []
    for protocol_key, protocol_data in all_results.items():
        if not isinstance(protocol_data, dict) or "fold_results_by_seed" not in protocol_data:
            continue
        wr2s = []
        for seed_key, seed_data in protocol_data["fold_results_by_seed"].items():
            wr2s.append(seed_data["oof_weighted_r2"])
        if wr2s:
            rows.append({
                "Validation Protocol": protocol_key,
                "Mean Weighted R2": f"{np.mean(wr2s):.4f}",
                "Std Weighted R2": f"{_std(wr2s):.4f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table_8.csv"), index=False)
    print(df.to_string(index=False))
    return df

def gen_table_9(seed_results, out_dir):
    """Table 9: Per-target R2 for best protocol."""
    rows = []
    for seed_key, seed_data in seed_results.items():
        fold_results = seed_data["fold_results"]
        all_preds = np.concatenate([np.array(fr["preds"]) for fr in fold_results])
        all_targets = np.concatenate([np.array(fr["targets"]) for fr in fold_results])
        per_target = per_target_r2_score(all_targets, all_preds)
        row = {"Seed": seed_key}
        for tn in TARGET_NAMES:
            row[tn] = f"{per_target[tn]:.4f}"
        row["Weighted R2"] = f"{global_weighted_r2_score(all_targets, all_preds):.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table_9.csv"), index=False)
    print(df.to_string(index=False))
    return df

def _std(values):
    """Sample std; 0.0 for a single value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def gen_tables_10_11(lopo_results, out_dir):
    """
    Tables 10/11: Leave-one-period-out.

    Table 10: per-period R2_w, RMSE/MAE/Bias for Dry_Total_g (mean over
    seeds), plus overall mean/std rows. Table 11: per-target R2 per period.
    """
    period_results = lopo_results["period_results"]

    rows = []
    for period in ["early", "middle", "late"]:
        runs = period_results[period]
        rows.append({
            "Held-out period": period.capitalize(),
            "Weighted R2": f"{np.mean([r['weighted_r2'] for r in runs]):.3f}",
            "RMSE (Dry_Total_g)": f"{np.mean([r['per_target_rmse']['Dry_Total_g'] for r in runs]):.2f}",
            "MAE (Dry_Total_g)": f"{np.mean([r['per_target_mae']['Dry_Total_g'] for r in runs]):.2f}",
            "Bias (Dry_Total_g)": f"{np.mean([r['per_target_bias']['Dry_Total_g'] for r in runs]):.3f}",
        })
    wr2s = [r["weighted_r2"] for p in period_results.values() for r in p]
    rmse = [r["per_target_rmse"]["Dry_Total_g"] for p in period_results.values() for r in p]
    mae = [r["per_target_mae"]["Dry_Total_g"] for p in period_results.values() for r in p]
    bias = [r["per_target_bias"]["Dry_Total_g"] for p in period_results.values() for r in p]
    rows.append({"Held-out period": "Mean", "Weighted R2": f"{np.mean(wr2s):.3f}",
                 "RMSE (Dry_Total_g)": f"{np.mean(rmse):.2f}",
                 "MAE (Dry_Total_g)": f"{np.mean(mae):.2f}",
                 "Bias (Dry_Total_g)": f"{np.mean(bias):.3f}"})
    rows.append({"Held-out period": "Std", "Weighted R2": f"{_std(wr2s):.3f}",
                 "RMSE (Dry_Total_g)": f"{_std(rmse):.2f}",
                 "MAE (Dry_Total_g)": f"{_std(mae):.2f}",
                 "Bias (Dry_Total_g)": f"{_std(bias):.3f}"})
    df10 = pd.DataFrame(rows)
    df10.to_csv(os.path.join(out_dir, "table_10.csv"), index=False)
    print(df10.to_string(index=False))

    rows = []
    for period in ["early", "middle", "late"]:
        runs = period_results[period]
        row = {"Held-out period": period.capitalize()}
        for tn in TARGET_NAMES:
            row[tn] = f"{np.mean([r['per_target_r2'][tn] for r in runs]):.3f}"
        rows.append(row)
    df11 = pd.DataFrame(rows)
    df11.to_csv(os.path.join(out_dir, "table_11.csv"), index=False)
    print()
    print(df11.to_string(index=False))
    return df10, df11

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to full_results.pt")
    parser.add_argument("--lopo-results", type=str, default=None,
                        help="Path to LOPO full_results.pt (enables Tables 10/11)")
    parser.add_argument("--output", type=str, default="results/tables")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    data = torch.load(args.results, map_location="cpu", weights_only=False)
    if "fold_results_by_seed" in data:
        gen_table_9(data["fold_results_by_seed"], args.output)
    if args.lopo_results:
        lopo_data = torch.load(args.lopo_results, map_location="cpu", weights_only=False)
        gen_tables_10_11(lopo_data, args.output)
    print(f"\nTables saved to {args.output}/")
