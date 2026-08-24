"""
Unit tests for evaluation metrics.
"""
import unittest
import numpy as np

from src.evaluation.metrics import (
    global_weighted_r2_score,
    per_target_r2_score,
    per_target_rmse,
    per_target_mae,
    per_target_bias,
    all_metrics,
    TARGET_NAMES,
)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Simple test data
        np.random.seed(42)
        self.N = 100
        self.y_true = np.random.randn(self.N, 5) * 30 + 50  # realistic biomass values
        self.y_pred_perfect = self.y_true.copy()
        self.y_pred_noise = self.y_true + np.random.randn(self.N, 5) * 5

    def test_perfect_prediction_r2(self):
        """Perfect predictions should give R2 = 1.0."""
        r2 = global_weighted_r2_score(self.y_true, self.y_pred_perfect)
        self.assertAlmostEqual(r2, 1.0)

    def test_perfect_prediction_per_target_r2(self):
        """Perfect predictions should give R2 = 1.0 for each target."""
        r2s = per_target_r2_score(self.y_true, self.y_pred_perfect)
        for name in TARGET_NAMES:
            self.assertAlmostEqual(r2s[name], 1.0)

    def test_perfect_rmse_zero(self):
        """Perfect predictions should have RMSE = 0."""
        rmses = per_target_rmse(self.y_true, self.y_pred_perfect)
        for name in TARGET_NAMES:
            self.assertAlmostEqual(rmses[name], 0.0)

    def test_perfect_mae_zero(self):
        """Perfect predictions should have MAE = 0."""
        maes = per_target_mae(self.y_true, self.y_pred_perfect)
        for name in TARGET_NAMES:
            self.assertAlmostEqual(maes[name], 0.0)

    def test_perfect_bias_zero(self):
        """Perfect predictions should have Bias = 0."""
        biases = per_target_bias(self.y_true, self.y_pred_perfect)
        for name in TARGET_NAMES:
            self.assertAlmostEqual(biases[name], 0.0)

    def test_noisy_predictions_r2_less_than_one(self):
        """Noisy predictions should have R2 < 1."""
        r2 = global_weighted_r2_score(self.y_true, self.y_pred_noise)
        self.assertLess(r2, 1.0)
        self.assertGreater(r2, -1.0)

    def test_all_metrics_consistency(self):
        """all_metrics() should return expected structure."""
        metrics = all_metrics(self.y_true, self.y_pred_noise)
        self.assertIn("weighted_r2", metrics)
        self.assertIn("per_target_r2", metrics)
        self.assertIn("per_target_rmse", metrics)
        self.assertIn("per_target_mae", metrics)
        self.assertIn("per_target_bias", metrics)
        for name in TARGET_NAMES:
            self.assertIn(name, metrics["per_target_r2"])


if __name__ == "__main__":
    unittest.main()
