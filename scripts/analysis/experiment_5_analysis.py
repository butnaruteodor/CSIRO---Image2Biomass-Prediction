#!/usr/bin/env python3
"""
Error analysis from saved CV predictions.

Usage:
    python scripts/analysis/experiment_5_analysis.py --results results/cv_date_location/full_results.pt
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CFG
from src.evaluation.metrics import global_weighted_r2_score, per_target_r2_score, per_target_rmse, per_target_mae, per_target_bias, all_metrics, print_metrics_table, TARGET_NAMES
from src.data.preprocessing import get_df

RESULTS_DIR = "results/experiment_5"

def analyze(results_path):
    """Load CV results and produce error analysis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = torch.load(results_path, map_location="cpu", weights_only=False)
    seed_results = data.get("fold_results_by_seed", {})

    all_wr2 = []
    for seed_key, seed_data in seed_results.items():
        fold_results = seed_data["fold_results"]
        all_preds = np.concatenate([np.array(fr["preds"]) for fr in fold_results])
        all_targets = np.concatenate([np.array(fr["targets"]) for fr in fold_results])
        wr2 = global_weighted_r2_score(all_targets, all_preds)
        all_wr2.append(wr2)

    print(f"\nOOF Weighted R2: {np.mean(all_wr2):.4f} ± {np.std(all_wr2, ddof=1):.4f}")
    print(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to full_results.pt")
    args = parser.parse_args()
    analyze(args.results)
