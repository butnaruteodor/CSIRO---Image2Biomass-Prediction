"""
Ensemble predictor for combining model predictions.
"""
import numpy as np
import torch
from src.config import InferenceConfig


class EnsemblePredictor:
    """Runs MLP ensemble predictions on extracted features."""

    def __init__(self, mlp_models, config: InferenceConfig):
        self.mlp_models = mlp_models
        self.config = config
        self.device = config.device

        for m in self.mlp_models:
            m.to(self.device)
            m.eval()

    def predict_all(self, X_massive, N, K):
        """Predict on massive matrix and average TTA views."""
        mlp_preds_flat = self._predict_mlp(X_massive)
        mlp_final = self._reshape_and_avg(mlp_preds_flat, N, K)
        return mlp_final

    def _predict_mlp(self, X):
        """Run PyTorch MLP heads on numpy features."""
        X_tensor = torch.from_numpy(X).float().to(self.device)
        accum = {"total": 0, "green": 0, "gdm": 0, "clover": 0, "dead": 0}

        with torch.inference_mode():
            for model in self.mlp_models:
                p_total, p_gdm, p_green, p_clover, p_dead = model(X_tensor)
                accum["total"] += p_total.cpu().numpy().flatten()
                accum["green"] += p_green.cpu().numpy().flatten()
                accum["gdm"] += p_gdm.cpu().numpy().flatten()
                accum["clover"] += p_clover.cpu().numpy().flatten()
                accum["dead"] += p_dead.cpu().numpy().flatten()

        n_folds = len(self.mlp_models)
        return {k: v / n_folds for k, v in accum.items()}

    def _reshape_and_avg(self, preds_dict, N, K):
        """Average across TTA views."""
        result = {}
        for key in preds_dict:
            arr = preds_dict[key]
            arr_reshaped = arr.reshape(K, N)
            result[key] = arr_reshaped.mean(axis=0)
        return result
