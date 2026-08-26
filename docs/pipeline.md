# Pipeline

## End-to-End Workflow

```
                         ┌─────────────────┐
                         │ csiro-biomass/   │
                         │  train.csv       │
                         │  test.csv        │
                         │  train/ (357)    │
                         │  test/  (805)    │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Step 1: Extract Embeddings│
                    │ extract_embeddings.py     │
                    │  --mode train             │
                    │  --mode test              │
                    │                           │
                    │ Output: embeddings/       │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Step 2: Cross-Validation  │
                    │ cross_validation.py       │
                    │  --split date_location    │
                    │  --head mlp               │
                    │  --seeds 13 21 42 87 101  │
                    │                           │
                    │ Output: results/cv_*/     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Step 3: Train Final Models│
                    │ train_model.py            │
                    │  --head mlp               │
                    │  --seeds 13 21 42 87 101  │
                    │                           │
                    │ Output: results/          │
                    │  submission_models/       │
                    └─────────────┬─────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │ Evaluate Local  │  │ Analysis        │  │ Kaggle Submit   │
  │ evaluate_local  │  │ experiment_5_   │  │ run_inference   │
  │ .py             │  │ analysis.py     │  │ .py             │
  │                 │  │                 │  │                 │
  │ Needs test      │  │ Error analysis  │  │ Creates         │
  │ ground truth    │  │ by state/       │  │ submission.csv  │
  └─────────────────┘  │ species/biomass │  └─────────────────┘
                       └─────────────────┘
```

## Step-by-Step

### Step 1: Extract Embeddings

```bash
# Training: clean + 15 augmented copies
python scripts/extract_embeddings.py --mode train

# Test: 5 TTA views (original, HFlip, VFlip, HFlip+VFlip, Transpose)
python scripts/extract_embeddings.py --mode test
```

### Step 2: Cross-Validation

```bash
# Primary protocol: date-location grouped CV with MLP
python scripts/cross_validation.py --split date_location --head mlp --seeds 13 21 42 87 101

# Alternative: Ridge regression
python scripts/cross_validation.py --split date_location --head ridge
```

### Step 3: Train Final Models

```bash
# Train 5 MLP models on all data (one per seed)
python scripts/train_model.py --head mlp --seeds 13 21 42 87 101

# Ridge alternative
python scripts/train_model.py --head ridge --seeds 13 21 42 87 101
```

### Step 4: Evaluate

```bash
# Local evaluation (requires test ground truth)
python scripts/evaluate_local.py

# Analysis of CV results
python scripts/analysis/experiment_5_analysis.py --results results/cv_date_location/full_results.pt

# Generate tables
python scripts/analysis/experiment_2_tables.py --results results/cv_date_location/full_results.pt
```

### Step 5: Submission

```bash
python scripts/run_inference.py --img-size 1008
```

## Output Directory Structure

```
results/
├── cv_date_location/
│   └── full_results.pt          # All fold predictions + metrics
├── submission_models/
│   ├── seed_13_final.pt
│   ├── seed_21_final.pt
│   ├── seed_42_final.pt
│   ├── seed_87_final.pt
│   └── seed_101_final.pt
├── local_evaluation/
│   └── results.json
└── tables/                      # Generated analysis tables
```