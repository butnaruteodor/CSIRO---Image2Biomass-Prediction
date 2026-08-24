#!/usr/bin/env python3
"""
Local evaluation of trained models against test set ground truth.
Mimics Kaggle submission scoring with rich metrics.

Usage:
    python scripts/evaluate_local.py                          # Full eval
    python scripts/evaluate_local.py --fast                   # Small images for speed
    python scripts/evaluate_local.py --max-images 50          # Quick subset
    python scripts/evaluate_local.py --force-recompute        # Recompute embeddings
    python scripts/evaluate_local.py --skip-backbone          # Only eval cached

Output:
    - Per-seed and ensemble metrics printed to stdout
    - Metrics table with R2, RMSE, MAE, Bias for each of 5 targets
    - Results saved to results/local_evaluation/
"""
import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import InferenceConfig, CFG
from src.evaluation.evaluator import LocalEvaluator
from src.data.preprocessing import get_test_df
from src.evaluation.metrics import (
    all_metrics, print_metrics_table, global_weighted_r2_score,
    per_target_r2_score, per_target_rmse, per_target_mae, per_target_bias,
    TARGET_NAMES
)


def main():
    parser = argparse.ArgumentParser(description="Local evaluation of biomass models")
    parser.add_argument("--force-recompute-embeddings", action="store_true",
                        help="Force recompute test embeddings")
    parser.add_argument("--img-size", type=int, default=1008,
                        help="Image size for backbone (default: 1008)")
    parser.add_argument("--fast", action="store_true",
                        help="Use 224px images for fast approximate eval")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Only evaluate first N images")
    parser.add_argument("--skip-backbone", action="store_true",
                        help="Skip backbone inference, use cached embeddings only")
    args = parser.parse_args()

    config = InferenceConfig()
    if args.fast:
        config.img_size = 224
        print("FAST MODE: using 224px images (approximate)")
    else:
        config.img_size = args.img_size

    print("=" * 70)
    print("LOCAL EVALUATION - MIMICKING KAGGLE SCORING")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Backbone: {config.model_name}")
    print(f"Image size: {config.img_size}")
    print(f"Model dir: {config.model_dir}")
    print(f"Embed dir: {config.embed_dir}")
    print()

    # Quick check if seed models exist
    model_dir = config.model_dir
    seed_files = sorted([f for f in os.listdir(model_dir)
                         if f.startswith("seed_") and f.endswith("_final.pt")])
    if not seed_files:
        print(f"WARNING: No seed model files in {model_dir}!")
        print(f"Only ridge models found. Using those for evaluation.")
        ridge_files = sorted([f for f in os.listdir(model_dir)
                               if f.startswith("ridge_")])
        if not ridge_files:
            print(f"ERROR: No models found in {model_dir}!")
            print(f"Run: python experiments/train_submission.py")
            sys.exit(1)
        use_mlp = False
    else:
        use_mlp = True

    # Load test data (with ground truth)
    test_df = get_test_df()
    if args.max_images:
        test_df = test_df.iloc[:args.max_images]
        print(f"Limited to {args.max_images} images for quick test")
    print(f"Test samples: {len(test_df)}")
    print()

    # Get test embeddings (cached or compute)
    if args.skip_backbone:
        test_embeds, test_targets, _ = _load_or_fail()
    else:
        evaluator = LocalEvaluator(config)
        test_embeds, test_targets, _ = evaluator._get_test_embeddings(
            test_df, force_recompute=args.force_recompute_embeddings)
    
    if test_embeds is None:
        print("ERROR: Could not get test embeddings.")
        sys.exit(1)

    y_true = test_targets.numpy()
    X = test_embeds.numpy()

    # Evaluate all models
    seed_r2s = []
    all_preds = []
    
    import joblib
    import torch
    
    if use_mlp:
        print("\nEvaluating MLP seed models...")
        from src.models.heads import BiomassSimpleMLP
        with open(os.path.join(config.embed_dir, "metadata.json")) as f:
            import json
            _meta = json.load(f)
        FEATURE_DIM = _meta["embedding_dim"]
        
        for sf in seed_files:
            seed = int(sf.split("_")[1])
            model = BiomassSimpleMLP(FEATURE_DIM).to(config.device)
            state = torch.load(os.path.join(model_dir, sf),
                               map_location=config.device, weights_only=False)
            try:
                model.load_state_dict(state)
            except Exception:
                model_dict = model.state_dict()
                matched = {k: v for k, v in state.items() if k in model_dict}
                model.load_state_dict(matched, strict=False)
            model.eval()
            
            with torch.inference_mode():
                p_total, p_gdm, p_green, p_clover, p_dead = model(test_embeds.to(config.device))
                preds = torch.stack([
                    p_green.squeeze(), p_dead.squeeze(),
                    p_clover.squeeze(), p_gdm.squeeze(), p_total.squeeze()
                ], dim=1).cpu().numpy()
            
            r2 = global_weighted_r2_score(y_true, preds)
            seed_r2s.append(r2)
            all_preds.append(preds)
            print(f"  Seed {seed}: Weighted R2 = {r2:.4f}")
    else:
        print("\nEvaluating Ridge models...")
        for rf in ridge_files:
            model = joblib.load(os.path.join(model_dir, rf))
            preds = model.predict(X)
            # Order: [Green, Dead, Clover, GDM, Total]
            r2 = global_weighted_r2_score(y_true, preds)
            seed_r2s.append(r2)
            all_preds.append(preds)
            print(f"  {rf}: Weighted R2 = {r2:.4f}")

    # Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    r2_mean = float(np.mean(seed_r2s))
    r2_std = float(np.std(seed_r2s, ddof=1))
    print(f"\nModels: Weighted R2 = {r2_mean:.4f} +/- {r2_std:.4f}")
    print(f"  Per model: {[f'{x:.4f}' for x in seed_r2s]}")

    ensemble_preds = np.mean(all_preds, axis=0)
    ensemble_metrics = all_metrics(y_true, ensemble_preds)
    print_metrics_table(ensemble_metrics, "Ensemble Performance")

    output_dir = "results/local_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "model_type": "MLP" if use_mlp else "Ridge",
            "wr2_mean": r2_mean,
            "wr2_std": r2_std,
            "wr2_per_seed": [float(x) for x in seed_r2s],
            "ensemble": ensemble_metrics if isinstance(ensemble_metrics, dict) else {},
        }, f, indent=2)
    print(f"Results saved to {output_dir}/")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


def _load_or_fail():
    """Try to load cached embeddings, return None if not found."""
    import torch
    embed_path = "embeddings/test_clean_embeddings.pt"
    target_path = "embeddings/test_targets.pt"
    if os.path.exists(embed_path) and os.path.exists(target_path):
        return torch.load(embed_path), torch.load(target_path), None
    return None, None, None


if __name__ == "__main__":
    main()
