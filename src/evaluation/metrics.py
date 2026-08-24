"""
All evaluation metrics for biomass prediction.
Includes Kaggle-matching weighted R2, per-target metrics, RMSE, MAE, Bias.
"""
import numpy as np
from sklearn.metrics import r2_score
from src.config import CFG

TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']


def per_target_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Standard R2 score for each of the 5 targets individually.

    Args:
        y_true: (N, 5)
        y_pred: (N, 5)
    Returns:
        dict mapping target names to R2 scores
    """
    raw_scores = r2_score(y_true, y_pred, multioutput='raw_values')
    return {name: raw_scores[i] for i, name in enumerate(TARGET_NAMES)}


def global_weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Globally weighted R2 score matching Kaggle's metric exactly.
    y_true, y_pred: shape (N, 5) in order [Green, Dead, Clover, GDM, Total]
    weights: [0.1, 0.1, 0.1, 0.2, 0.5]
    """
    weights = CFG.R2_WEIGHTS_VAL
    weights_matrix = np.tile(weights, (y_true.shape[0], 1))

    # Weighted mean
    weighted_sum = np.sum(weights_matrix * y_true)
    total_weight = np.sum(weights_matrix)
    y_bar_w = weighted_sum / total_weight

    # SS_res and SS_tot
    ss_res = np.sum(weights_matrix * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights_matrix * (y_true - y_bar_w) ** 2)

    r2_w = 1 - (ss_res / ss_tot)
    return r2_w


def per_target_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE for each target."""
    result = {}
    for i, name in enumerate(TARGET_NAMES):
        result[name] = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
    return result


def per_target_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE for each target."""
    result = {}
    for i, name in enumerate(TARGET_NAMES):
        result[name] = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
    return result


def per_target_bias(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bias (mean error) for each target."""
    result = {}
    for i, name in enumerate(TARGET_NAMES):
        result[name] = np.mean(y_pred[:, i] - y_true[:, i])
    return result


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all metrics and return as a nested dict."""
    return {
        'weighted_r2': global_weighted_r2_score(y_true, y_pred),
        'per_target_r2': per_target_r2_score(y_true, y_pred),
        'per_target_rmse': per_target_rmse(y_true, y_pred),
        'per_target_mae': per_target_mae(y_true, y_pred),
        'per_target_bias': per_target_bias(y_true, y_pred),
    }


def print_metrics_table(metrics: dict, title: str = ""):
    """Pretty-print all metrics in a table."""
    if title:
        print(f"\n{'=' * 70}")
        print(f" {title}")
        print(f"{'=' * 70}")

    print(f"\nWeighted R²: {metrics['weighted_r2']:.4f}")

    print(f"\n{'Target':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Bias':>10}")
    print("-" * 55)
    for name in TARGET_NAMES:
        r2 = metrics['per_target_r2'][name]
        rmse = metrics['per_target_rmse'][name]
        mae = metrics['per_target_mae'][name]
        bias = metrics['per_target_bias'][name]
        print(f"{name:<15} {r2:>8.4f} {rmse:>10.2f} {mae:>10.2f} {bias:>10.2f}")
    print()
