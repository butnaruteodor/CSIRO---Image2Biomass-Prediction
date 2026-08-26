# Inference

## Pipeline

The inference pipeline (`src/inference/pipeline.py`) orchestrates:

1. **Load test data** from `csiro-biomass/test.csv`
2. **Load backbone** (DINOv3 ViT-L/16)
3. **Extract embeddings** for all 5 TTA views
4. **Load 5 MLP seed models** from `results/submission_models/`
5. **Predict** on all TTA views, then average across views
6. **Average across seed models** (ensemble)
7. **Create submission CSV**

## Test-Time Augmentation (TTA)

5 TTA transforms applied to each image:

| View | Transform |
|------|-----------|
| 0 | Original (resize + normalize) |
| 1 | Horizontal flip |
| 2 | Vertical flip |
| 3 | Horizontal + vertical flip |
| 4 | Transpose |

Each view produces a 2048-dim embedding → predictions averaged across views for each image.

## Ensemble

5 independent seed models (seeds 13, 21, 42, 87, 101) are averaged:

```python
final_prediction = mean(predictions from seed_13, seed_21, ..., seed_101)
```

This reduces variance and typically improves weighted R² by 0.01–0.02 over single models.

## Submission Format

Output: `submission.csv`

```csv
sample_id,target
0,45.2
1,12.8
...
```

Long format matching the Kaggle competition requirements:
- 4025 rows (805 images × 5 targets)
- Targets: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g

## Usage

```bash
# Default inference
python scripts/run_inference.py

# Custom image size (lower = faster but less accurate)
python scripts/run_inference.py --img-size 768
```