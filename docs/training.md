# Training

## Loss Function

The loss combines MSE and L1 losses weighted by target importance:

| Target | Loss | Training Weight | Validation Weight |
|--------|------|-----------------|-------------------|
| Dry_Total_g | MSE | 1.0 | 0.5 |
| GDM_g | MSE | 1.0 | 0.2 |
| Dry_Green_g | MSE | 1.0 | 0.1 |
| Dry_Clover_g | L1 | 1.0 | 0.1 |
| Dry_Dead_g | L1 | 1.0 | 0.1 |

Reasons:
- **MSE** for Total, GDM, Green — smooth, non-sparse targets with meaningful variance
- **L1** for Clover, Dead — sparse targets where many values are near zero; L1 reduces sensitivity to outliers

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

After CV, train on all 357 samples:

```bash
python scripts/train_model.py --head mlp --seeds 13 21 42 87 101
```

- No validation split; all data used for training
- Same hyperparameters as CV
- 5 independent seed models for ensemble

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