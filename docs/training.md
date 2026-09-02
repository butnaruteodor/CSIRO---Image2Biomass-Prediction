# Training

## Loss Function

The loss is a weighted mean squared error following the weighting scheme of
the official competition metric (weights for Green, Dead, Clover, GDM, Total):

| Target | Loss | Weight |
|--------|------|--------|
| Dry_Total_g | MSE | 0.5 |
| GDM_g | MSE | 0.2 |
| Dry_Green_g | MSE | 0.1 |
| Dry_Clover_g | MSE | 0.1 |
| Dry_Dead_g | MSE | 0.1 |

The weights are defined once in `src/config.py` (`R2_WEIGHTS_TRAIN` /
`R2_WEIGHTS_VAL`) and shared by the loss and the evaluation metric.

## Optimizer

- **Algorithm**: AdamW
- **Learning rate**: 1×10⁻³
- **Weight decay**: 0.01
- **Schedule**: Cosine annealing (T_max = 80 epochs)
- **Gradient clipping**: norm = 1.0

## Embedding Augmentation

During training, each batch samples from 16 possible embeddings per image:
- **15 augmented copies** (spatial + photometric transforms applied during embedding extraction)
- **1 clean copy** (validation-style: resize + normalize only)

Random uniform selection per sample adds diversity without re-running the backbone.

## Cross-Validation

5 seeds × 5 folds per protocol:

```bash
python scripts/cross_validation.py --split date_location --head mlp --seeds 13 21 42 87 101
```

- **80 training epochs** per fold
- **Early stopping**: patience = 15 epochs based on validation weighted R²
- **Best model**: saved via epoch-level tracking
- **Checkpoint**: per-fold metrics and predictions saved to `full_results.pt`

## Full-Data Training

After CV, the stopping epoch per protocol is the median of the best epochs
over all 25 runs; the final models are then retrained on all 357 samples
with that duration:

```bash
python scripts/analysis/stopping_epochs.py --results results/cv_date_location/full_results.pt
python scripts/train_model.py --head mlp --epochs <median> --seeds 13 21 42 87 101
```

- No validation split; all data used for training
- Same hyperparameters as CV
- 5 independent seed models for per-seed submissions and ensembling

## Reproducibility

Seeds are managed via `src/deterministic.py`:

```python
set_seed(seed, deterministic=True)      # Seeds random, numpy, torch
seed_worker(worker_id)                   # DataLoader worker seeding
get_generator(seed)                       # PyTorch Generator for shuffling
```

When `deterministic=True`:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Note: Full determinism is not guaranteed on GPU due to CUDA convolution non-determinism.