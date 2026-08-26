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
# 1. Setup
python3.10 -m venv venv
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
venv/bin/pip install timm opencv-python-headless albumentations pandas tqdm matplotlib scikit-learn

# 2. Extract embeddings (run once)
venv/bin/python scripts/extract_embeddings.py --mode train
venv/bin/python scripts/extract_embeddings.py --mode test

# 3. Cross-validation
venv/bin/python scripts/cross_validation.py --split date_location --head mlp --seeds 13 21 42 87 101

# 4. Train final models on all data
venv/bin/python scripts/train_model.py --head mlp --seeds 13 21 42 87 101

# 5. Local evaluation (needs test ground truth)
venv/bin/python scripts/evaluate_local.py

# 6. Generate Kaggle submission
venv/bin/python scripts/run_inference.py --img-size 1008
```

## Project Structure

```
├── README.md
├── docs/                     # Detailed documentation
│   ├── overview.md           # Problem, dataset, approach
│   ├── setup.md              # Installation & environment
│   ├── data.md               # Data structure & preprocessing
│   ├── pipeline.md           # End-to-end workflow
│   ├── models.md             # Architecture details
│   ├── training.md           # Training procedures
│   ├── evaluation.md         # Metrics & evaluation
│   ├── inference.md          # Inference & submission
│   ├── glossary.md           # Domain terminology
│   └── api.md                # Module reference
├── scripts/                  # Entry-point scripts
├── src/                      # Source package
├── csiro-biomass/            # Dataset directory
├── embeddings/               # Precomputed features
├── results/                  # CV results, trained models
└── figures/                  # Publication figures
```

## Key Results

| Metric | Value |
|--------|-------|
| Weighted R² | [TBD] |
| Seeds | 13, 21, 42, 87, 101 |
| CV protocol | Date-location grouped |

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

