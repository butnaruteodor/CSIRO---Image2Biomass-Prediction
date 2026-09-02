#!/usr/bin/env python3
"""
Select the final-model training duration per validation protocol.

For each protocol, the stopping epoch is the median of the best epochs
recorded across all cross-validation runs (5 folds x 5 seeds = 25 runs).
The final models are then retrained on the full public dataset using this
duration:

    python scripts/analysis/stopping_epochs.py --results results/cv_date_location/full_results.pt
    python scripts/train_model.py --head mlp --epochs <median>
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def best_epochs(results_path):
    """Return the list of best epochs over all runs of a CV result file."""
    data = torch.load(results_path, map_location="cpu", weights_only=False)
    epochs = []
    for seed_key, seed_data in data["fold_results_by_seed"].items():
        for fold in seed_data["fold_results"]:
            if fold.get("best_epoch", 0) > 0:
                epochs.append(int(fold["best_epoch"]))
    return epochs


def main():
    parser = argparse.ArgumentParser(description="Median stopping epoch from CV results")
    parser.add_argument("--results", type=str, nargs="+", required=True,
                        help="Path(s) to full_results.pt file(s)")
    parser.add_argument("--output", type=str, default="results/stopping_epochs.json")
    args = parser.parse_args()

    summary = {}
    for path in args.results:
        epochs = best_epochs(path)
        if not epochs:
            print(f"{path}: no best-epoch information found (Ridge runs record none)")
            continue
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        median = int(np.median(epochs))
        summary[name] = {
            "median_best_epoch": median,
            "n_runs": len(epochs),
            "min": int(np.min(epochs)),
            "max": int(np.max(epochs)),
        }
        print(f"{name}: median best epoch = {median} "
              f"({len(epochs)} runs, range {np.min(epochs)}-{np.max(epochs)})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")
    print("Retrain final models with: python scripts/train_model.py --epochs <median>")


if __name__ == "__main__":
    main()