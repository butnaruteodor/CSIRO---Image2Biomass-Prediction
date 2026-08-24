"""
Local evaluator that mimics Kaggle submission scoring.
Loads trained models, runs inference on test set, computes all metrics.
"""
import os
import gc
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from src.config import InferenceConfig, CFG
from src.models.backbone import BackboneModel
from src.models.heads import BiomassSimpleMLP
from src.data.preprocessing import extract_test_embeddings, get_test_df
from src.evaluation.metrics import (
    global_weighted_r2_score, per_target_r2_score,
    per_target_rmse, per_target_mae, per_target_bias,
    all_metrics, print_metrics_table, TARGET_NAMES
)


class LocalEvaluator:
    """
    Local evaluation of trained models against test set ground truth.
    Mimics Kaggle's scoring pipeline exactly.
    """

    def __init__(self, config=None):
        self.config = config or InferenceConfig()
        self.device = self.config.device
        print(f"Device: {self.device}")

        meta_path = os.path.join(self.config.embed_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _meta = json.load(f)
            self.feature_dim = _meta["embedding_dim"]
        else:
            self.feature_dim = 2048
        print(f"Feature dimension: {self.feature_dim}")

    def evaluate(self, force_recompute_embeddings=False):
        """Run full evaluation pipeline."""
        # 1. Load test data with ground truth
        print("\n" + "=" * 70)
        print("LOADING TEST DATA")
        print("=" * 70)
        test_df = get_test_df()

        # 2. Load or compute test embeddings
        test_embeds, test_targets, _ = self._get_test_embeddings(
            test_df, force_recompute_embeddings)

        # 3. Load seed models and run inference
        print("\n" + "=" * 70)
        print("RUNNING INFERENCE")
        print("=" * 70)
        seed_metrics = []
        all_preds = []

        model_dir = self.config.model_dir
        if not os.path.exists(model_dir):
            print(f"WARNING: Model directory {model_dir} not found!")
            return None

        seed_files = sorted([f for f in os.listdir(model_dir)
                             if f.startswith("seed_") and f.endswith("_final.pt")])

        if not seed_files:
            print(f"WARNING: No seed model files found in {model_dir}!")
            return None

        print(f"Found {len(seed_files)} seed models")

        for sf in seed_files:
            seed = int(sf.split("_")[1])
            print(f"\n  Evaluating seed {seed}...")

            ckpt_path = os.path.join(model_dir, sf)
            model = BiomassSimpleMLP(self.feature_dim).to(self.device)
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    print(f"    Warning: state dict load issue: {e}")
                    model_dict = model.state_dict()
                    matched = {k: v for k, v in state.items() if k in model_dict}
                    if matched:
                        model.load_state_dict(matched, strict=False)
                        print(f"    Loaded {len(matched)}/{len(model_dict)} keys")

            model.eval()

            with torch.inference_mode():
                X = test_embeds.to(self.device)
                p_total, p_gdm, p_green, p_clover, p_dead = model(X)
                preds = torch.stack([
                    p_green.squeeze(), p_dead.squeeze(),
                    p_clover.squeeze(), p_gdm.squeeze(), p_total.squeeze()
                ], dim=1).cpu().numpy()

            targets = test_targets.numpy()
            met = all_metrics(targets, preds)
            seed_metrics.append(met)
            all_preds.append(preds)

            print(f"    Weighted R2 = {met['weighted_r2']:.4f}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # 4. Aggregate across seeds
        print("\n" + "=" * 70)
        print("FINAL RESULTS (mean +/- std across seeds)")
        print("=" * 70)

        wr2s = [m["weighted_r2"] for m in seed_metrics]
        print(f"\nWeighted R2: {np.mean(wr2s):.4f} +/- {np.std(wr2s, ddof=1):.4f}")

        print(f"\n{'Target':<15} {'R2_mean':>10} {'R2_std':>10} "
              f"{'RMSE':>10} {'MAE':>10} {'Bias':>10}")
        print("-" * 65)
        for name in TARGET_NAMES:
            r2s = [m["per_target_r2"][name] for m in seed_metrics]
            rmses = [m["per_target_rmse"][name] for m in seed_metrics]
            maes = [m["per_target_mae"][name] for m in seed_metrics]
            biases = [m["per_target_bias"][name] for m in seed_metrics]
            r2_mean = np.mean(r2s)
            r2_std = np.std(r2s, ddof=1)
            print(f"{name:<15} {r2_mean:>10.4f} {r2_std:>10.4f} "
                  f"{np.mean(rmses):>10.2f} {np.mean(maes):>10.2f} {np.mean(biases):>10.2f}")

        # Ensemble prediction (average across seeds)
        ensemble_preds = np.mean(all_preds, axis=0)
        ensemble_metrics = all_metrics(test_targets.numpy(), ensemble_preds)
        print(f"\nEnsemble (avg of {len(seed_files)} seeds): "
              f"Weighted R2 = {ensemble_metrics['weighted_r2']:.4f}")

        return {
            "per_seed": seed_metrics,
            "ensemble": ensemble_metrics,
            "wr2_per_seed": wr2s,
            "wr2_mean": np.mean(wr2s),
            "wr2_std": np.std(wr2s, ddof=1),
        }

    def _get_test_embeddings(self, test_df, force_recompute=False):
        """Load or compute test embeddings."""
        save_dir = self.config.embed_dir
        save_path = os.path.join(save_dir, "test_clean_embeddings.pt")

        if os.path.exists(save_path) and not force_recompute:
            print(f"Loading cached test embeddings from {save_path}")
            test_embeds = torch.load(save_path)
            test_targets = torch.load(os.path.join(save_dir, "test_targets.pt"))
            return test_embeds, test_targets, None

        # Need to compute test embeddings using backbone
        print("Computing test embeddings from scratch...")
        backbone_path = self.config.backbone_path
        
        if os.path.exists(backbone_path):
            print(f"Loading backbone from {backbone_path}...")
            backbone_model = BackboneModel(
                self.config.model_name, pretrained=False).to(self.device)
            state_dict = torch.load(backbone_path, map_location=self.device)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            backbone_model.backbone.load_state_dict(state_dict, strict=True)
        else:
            print(f"Backbone file not found at {backbone_path}, loading from timm pretrained...")
            import timm
            backbone_model = BackboneModel(
                self.config.model_name, pretrained=False).to(self.device)
            # Load pretrained weights from timm
            pretrained = timm.create_model(self.config.model_name, pretrained=True, num_classes=0)
            backbone_model.backbone.load_state_dict(pretrained.state_dict(), strict=True)
            del pretrained
        
        backbone_model.eval()

        if torch.cuda.device_count() > 1:
            backbone_model = torch.nn.DataParallel(backbone_model)

        test_embeds, test_targets, _ = extract_test_embeddings(
            backbone_model, test_df, save_dir=save_dir,
            img_size=self.config.img_size, device=self.device,
            batch_size=self.config.batch_size)

        del backbone_model
        gc.collect()
        torch.cuda.empty_cache()

        return test_embeds, test_targets, None
