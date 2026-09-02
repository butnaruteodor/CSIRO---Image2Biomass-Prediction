# CSIRO Pasture Biomass Prediction

> Estimating dry pasture biomass from aerial top-view images using a frozen DINOv3 ViT-L backbone with a shallow MLP head.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)]()

## Overview

This repository implements a two-stage pipeline for multivariate regression of pasture biomass from 2000×1000 px top-view images:

1. **Feature extraction**: A frozen DINOv3 ViT-Large/16 backbone produces 2048-dim embeddings per image (1024 per left/right crop)
2. **Regression head**: A shallow MLP (or Ridge) maps embeddings to 5 correlated target variables

The backbone runs once. The head trains in seconds, enabling rapid experimentation with different heads, augmentations, and cross-validation strategies.

## Quick Start

```bash
# 1. Setup (Python 3.10, CUDA 12.4 GPU recommended)
python3.10 -m venv venv
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
venv/bin/pip install -r requirements.txt

# 2. Data: place the competition data under csiro-biomass/ (train/, test/,
#    train.csv, test.csv) — only the small CSVs are versioned in this repo.

# 3. Extract embeddings (run once, then reuse for all experiments)
venv/bin/python scripts/extract_embeddings.py --mode train
venv/bin/python scripts/extract_embeddings.py --mode test

# 4. Cross-validation (repeat with --split random / date for the other protocols)
venv/bin/python scripts/cross_validation.py --split date_location --head mlp --seeds 13 21 42 87 101

# 5. Select the stopping epoch (median over the 25 CV runs) and retrain on all data
venv/bin/python scripts/analysis/stopping_epochs.py --results results/cv_date_location/full_results.pt
venv/bin/python scripts/train_model.py --head mlp --epochs <median> --seeds 13 21 42 87 101

# 6. Leave-One-Period-Out temporal analysis
venv/bin/python scripts/lopo_cv.py --head ridge

# 7. Local evaluation (needs test ground truth)
venv/bin/python scripts/evaluate_local.py

# 8. Generate Kaggle submissions (one per seed, or the full ensemble)
venv/bin/python scripts/run_inference.py --seeds 13
venv/bin/python scripts/run_inference.py
```

## Project Structure

```
├── README.md
├── requirements.txt           # Pinned dependencies (Python 3.10)
├── docs/                      # Detailed documentation
│   ├── overview.md            # Problem, dataset, approach
│   ├── setup.md               # Installation & environment
│   ├── data.md                # Data structure & preprocessing
│   ├── pipeline.md            # End-to-end workflow
│   ├── models.md              # Architecture details
│   ├── training.md            # Training procedures
│   ├── evaluation.md          # Metrics & evaluation
│   ├── inference.md           # Inference & submission
│   ├── glossary.md            # Domain terminology
│   └── api.md                 # Module reference
├── scripts/                   # Entry-point scripts
│   ├── extract_embeddings.py  # Frozen-backbone feature extraction (run once)
│   ├── cross_validation.py    # 5-fold CV per protocol and head
│   ├── train_model.py         # Final training on all data
│   ├── lopo_cv.py             # Leave-One-Period-Out temporal analysis
│   ├── evaluate_local.py      # Local evaluation (Kaggle-metric replica)
│   ├── run_inference.py       # Submission generation (per-seed or ensemble)
│   └── analysis/              # Table generation and error analysis
├── src/                       # Source package
│   ├── config.py              # Training/inference configuration
│   ├── deterministic.py       # Seeding and reproducibility helpers
│   ├── data/                  # Dataset, augmentations, splits (incl. LOPO)
│   ├── models/                # Backbone, MLP head, head factory
│   ├── training/              # Loss and training loops
│   ├── evaluation/            # Metrics and local evaluator
│   └── inference/             # Ensemble inference pipeline
├── tests/                     # Unit tests (metrics, loss, model head)
├── figures/                   # Figure-generation scripts (outputs gitignored)
├── csiro-biomass/             # Dataset directory (data not versioned; CSVs only)
├── embeddings/                # Precomputed features (regenerated locally)
└── results/                   # Experiment outputs (tables versioned, .pt ignored)
```

Large artifacts (dataset images, embedding tensors, model checkpoints, generated
figure files) are intentionally not versioned; every one of them is reproduced
by the scripts above.

## Key Results

| Metric | Value |
|--------|-------|
| Local OOF weighted R² (date-location grouped, MLP) | 0.794 ± 0.014 |
| Hidden-test weighted R² (MLP, per protocol) | 0.593 – 0.607 |
| Hidden-test weighted R² (Ridge) | 0.561 |
| Seeds | 13, 21, 42, 87, 101 |
| CV protocols | Random, date-grouped, date-location grouped, LOPO |

## Citation

```bibtex
@software{biomass_prediction,
  title = {CSIRO Pasture Biomass Prediction},
  year = {2025},
}
```

## Documentation

See the [docs/](docs/) directory for detailed documentation:

- **[Overview](docs/overview.md)** — Problem statement, dataset description, approach
- **[Setup](docs/setup.md)** — Installation, environment, GPU setup
- **[Data](docs/data.md)** — Data structure, preprocessing, CV splits
- **[Pipeline](docs/pipeline.md)** — End-to-end workflow
- **[Models](docs/models.md)** — Architecture details
- **[Training](docs/training.md)** — Training procedures, loss functions
- **[Evaluation](docs/evaluation.md)** — Metrics, local evaluation
- **[Inference](docs/inference.md)** — Inference, submission
- **[Glossary](docs/glossary.md)** — Domain terminology
- **[API Reference](docs/api.md)** — Module summaries

