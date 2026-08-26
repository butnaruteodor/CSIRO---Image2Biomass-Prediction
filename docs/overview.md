# Overview

## Problem Statement

Estimate five correlated pasture biomass targets from top-view RGB images of pasture plots:

| Target | Description | Role |
|--------|-------------|------|
| **Dry_Total_g** | Total above-ground dry biomass (g) | Primary target |
| **GDM_g** | Green Dry Matter (Green + Clover) | Secondary target |
| **Dry_Green_g** | Green grass dry biomass | Component |
| **Dry_Dead_g** | Dead grass dry biomass (= Total − GDM) | Derived |
| **Dry_Clover_g** | Clover dry biomass (= GDM − Green) | Derived |

The targets satisfy: `Dry_Total = Dry_Green + Dry_Dead + Dry_Clover` and `GDM = Dry_Green + Dry_Clover`.

## Dataset

- **Source**: CSIRO pasture trial — Kaggle competition
- **Images**: 357 training + 805 test, each 2000×1000 px (two 1000² crops side by side)
- **Acquisition**: 2015 across 4 Australian states (NSW, Tas, Vic, WA) with 30 unique date×state missions
- **Metadata**: Species (15 classes), NDVI, canopy height

## Approach

**Two-stage pipeline** that decouples feature extraction from regression:

1. **Stage 1 — Backbone** (frozen, run once):
   - DINOv3 ViT-Large/16 via `timm`
   - Each image split into left/right 1008² crops, processed independently
   - Output: 2048-dim embedding per image (1024 per crop)
   - Augmented copies ×15 for training diversity

2. **Stage 2 — Regression Head** (trained iteratively):
   - MLP: 3-layer network with 4 independent heads (Total, GDM, Green, Clover)
   - Ridge: sklearn `MultiOutputRegressor` for faster baselines
   - Trains on precomputed embeddings in seconds

## Why Two-Stage?

- GPU requirement drops from 24GB+ to 4GB
- Experiment with different heads without re-running the backbone
- Fast cross-validation (5 seeds × 5 folds in minutes)
- Clean separation between representation learning and regression

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen backbone | DINOv3 is SSL-pretrained; fine-tuning on 357 samples risks overfitting |
| 4 heads + derive Dead | Enforces physical constraint (Total = Green + Dead + Clover) |
| Softplus activation | Guarantees non-negative biomass predictions |
| Date-location grouped CV | Most realistic evaluation: test on unseen missions |
| Embedding augmentation | Simulates lighting/view variation without re-running backbone |