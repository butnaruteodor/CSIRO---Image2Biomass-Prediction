# Evaluation

## Metrics

### Weighted R² (Primary Metric)

Matches the Kaggle competition metric exactly:

```python
def global_weighted_r2_score(y_true, y_pred):
    weights = [0.1, 0.1, 0.1, 0.2, 0.5]  # [Green, Dead, Clover, GDM, Total]
    ...
    return 1 - (ss_res / ss_tot)
```

This is a **weighted variant** of the standard R² where each target contributes according to its weight. Total biomass (weight 0.5) dominates.

### Per-Target Metrics

All computed per target (5 values):

| Metric | Description |
|--------|-------------|
| **R²** | Standard coefficient of determination |
| **RMSE** | Root mean squared error (g) |
| **MAE** | Mean absolute error (g) |
| **Bias** | Mean error (g), positive = overprediction |

### Local Evaluation

The `evaluate_local.py` script:
1. Loads cached test embeddings
2. Runs all 5 seed models
3. Computes per-seed weighted R²
4. Computes ensemble (average across seeds) metrics
5. Prints formatted results table
6. Saves to `results/local_evaluation/results.json`

```bash
python scripts/evaluate_local.py
```

Options:
- `--fast` — Use 224px images for quick approximation
- `--max-images N` — Only evaluate first N images
- `--force-recompute` — Recompute backbone embeddings
- `--skip-backbone` — Use cached embeddings only

## CV Analysis

```bash
# Summary statistics
python scripts/analysis/experiment_5_analysis.py --results results/cv_date_location/full_results.pt

# Generate tables
python scripts/analysis/experiment_2_tables.py --results results/cv_date_location/full_results.pt --output results/tables
```

## Unit Tests

```bash
venv/bin/python -m unittest tests/test_metrics.py
```

Tests cover:
- Perfect predictions → R² = 1, RMSE/MAE/Bias = 0
- Noisy predictions → R² < 1
- `all_metrics()` structural consistency