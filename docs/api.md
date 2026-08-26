# API Reference

## Package: `src`

### `src.config`

| Class/Module | Description |
|---|---|
| `TrainingConfig` | Hyperparameters for training (paths, model, optimizer, seeds) |
| `InferenceConfig` | Configuration for inference pipeline |
| `CFG` | Legacy config class — backward compatible module-level constants |

Key parameters: `model_name`, `img_size`, `batch_size`, `epochs`, `lr`, `wd`, `seeds`, `all_target_cols`, `R2_WEIGHTS_VAL`

### `src.deterministic`

| Function | Description |
|---|---|
| `set_seed(seed, deterministic)` | Seeds random, numpy, torch, cuda |
| `seed_worker(worker_id)` | DataLoader worker init for reproducibility |
| `get_generator(seed)` | Creates seeded `torch.Generator` |

### `src.data`

| Module | Key Classes/Objects |
|---|---|
| `augmentations.py` | `get_val_transforms()`, `get_spatial_transforms()`, `get_photometric_transforms()`, `get_tta_transforms()` |
| `dataset.py` | `BiomassDatasetBase`, `EmbeddingAugmentationDataset`, `TestBiomassDataset`, `fast_slice_resize_image()` |
| `preprocessing.py` | `get_df()`, `check_splits()`, `get_random_stratified_splits()`, `get_date_grouped_splits()`, `get_date_location_grouped_splits()`, `load_embeddings_for_split()`, `extract_test_embeddings()`, `get_plant_neighbor_map()` |

### `src.models`

| Module | Key Classes/Objects |
|---|---|
| `backbone.py` | `BackboneModel` — Two-stream feature extractor |
| `heads.py` | `BiomassSimpleMLP` — 4-head MLP, `BiomassModelMLP` (legacy), `PerceiverResampler`, `ConvNeXtBlock`, `BiAttnBlock`, `BiomassMLPBlock` |
| `factory.py` | `HeadFactory.create(head_type, **kwargs)` — Creates MLP or Ridge heads |

### `src.training`

| Module | Key Functions |
|---|---|
| `loss.py` | `weighted_biomass_loss()`, `weighted_biomass_log_loss()`, `calculate_deltas()` |
| `trainer.py` | `train_epoch_mlp()`, `valid_epoch_mlp()` |

### `src.evaluation`

| Module | Key Functions |
|---|---|
| `metrics.py` | `global_weighted_r2_score()`, `per_target_r2_score()`, `per_target_rmse()`, `per_target_mae()`, `per_target_bias()`, `all_metrics()`, `print_metrics_table()` |
| `evaluator.py` | `LocalEvaluator` — Mimics Kaggle scoring |

### `src.inference`

| Module | Key Classes |
|---|---|
| `pipeline.py` | `InferencePipeline` — Full inference orchestrator |
| `ensemble.py` | `EnsemblePredictor` — MLP ensemble + TTA averaging |
| `feature_extractor.py` | `GlobalFeatureExtractor` — Backbone feature extraction with TTA |
| `model_loader.py` | `ModelLoader` — Loads backbone + seed models |
| `submission.py` | `SubmissionCreator` — Creates submission CSV |
| `transforms.py` | Re-exports from `src.data.augmentations`, `src.data.dataset` |

### `src.utils`

| Module | Key Functions |
|---|---|
| `helpers.py` | `compare_structure()`, `get_clean_timm_state_dict()`, `calculate_biomass_priors()`, `init_ratio_biases()`, `slerp()` |

## Scripts: `scripts/`

| Script | Description |
|---|---|
| `extract_embeddings.py` | Extract backbone features for train/test — run once |
| `cross_validation.py` | 5-fold CV with configurable split and head |
| `train_model.py` | Train on all training data, save models |
| `evaluate_local.py` | Evaluate trained models on test set (needs ground truth) |
| `run_inference.py` | Full inference → `submission.csv` |

### `scripts/analysis/`

| Script | Description |
|---|---|
| `experiment_2_tables.py` | Generate result tables from CV output |
| `experiment_5_analysis.py` | Error analysis from CV predictions |

## Figures: `figures/`

| Script | Output |
|---|---|
| `create_dataset_figure.py` | `figures/dataset_overview.pdf` — 3-panel dataset figure |
| `create_model_figure.py` | `figures/model_overview.pdf` — Architecture diagram |
| `generate_ieee_tables.py` | `results/experiment_2/table_*.csv` — IEEE-style tables |
| `plot_convergence.py` | `figures/convergence_curves.pdf` — Training curves |

## Tests: `tests/`

| File | Tests |
|---|---|
| `test_metrics.py` | Unit tests for all 5 metrics (perfect predictions, noise) |