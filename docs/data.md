# Data

## Raw Data Format

The dataset uses **long format** CSVs:

```csv
image_path,target_name,target
train/ID227847873.jpg,Dry_Green_g,47.9118
train/ID227847873.jpg,Dry_Dead_g,4.6794
```

Pivoted to wide format internally:

```csv
image_path,Dry_Green_g,Dry_Dead_g,Dry_Clover_g,GDM_g,Dry_Total_g
train/ID227847873.jpg,47.91,4.68,1.90,49.81,54.49
```

## Image Structure

Each 2000×1000 px image is split into two 1000×1000 square crops:

```
┌─────────────┬─────────────┐
│  Left crop  │ Right crop  │
│  1000×1000  │ 1000×1000   │
│             │             │
└─────────────┴─────────────┘
```

Both crops are resized to **1008×1008** before entering the backbone.

## Metadata

- **States**: Tas, NSW, WA, Vic (4 classes)
- **Species**: 15 pasture types (e.g., Ryegrass, Lucerne, Clover, mixed)
- **Dates**: January to November 2015
- **Missions**: 30 unique date×state combinations used as CV groups

## Preprocessing

The `get_df()` function in `src/data/preprocessing.py`:

1. Loads `train.csv` in long format
2. Pivots to wide format (one row per image, 5 target columns)
3. Merges metadata (State, Species, Date, NDVI, Height)
4. Creates cyclic temporal features (sin/cos of day-of-year)
5. Builds `group` column = `{State}_{Date}` for date-location CV
6. Quantizes `Dry_Total_g` into 10 bins for stratified splitting

## Cross-Validation Splits

Three strategies are available:

| Strategy | Splitter | Groups | Realism |
|----------|----------|--------|---------|
| Random | `StratifiedKFold` | None | Low (IID assumption) |
| Date-grouped | `StratifiedGroupKFold` | Date | Medium |
| **Date-location** | `StratifiedGroupKFold` | State × Date | **High** |

The **date-location grouped** protocol is the primary evaluation method. It ensures that all images from the same mission (same state + date) stay in the same fold, simulating the real scenario of predicting on entirely new field visits.

## Embeddings

Precomputed embeddings are stored in `embeddings/`:

| File | Shape | Content |
|------|-------|---------|
| `clean_embeddings.pt` | [357, 2048] | Validation-transformed train embeddings |
| `aug_embeddings.pt` | [357, 15, 2048] | 15 augmented copies per train image |
| `targets.pt` | [357, 5] | Training targets |
| `test_clean_embeddings.pt` | [805, 2048] | Test embeddings (TTA view 0) |
| `test_tta_embeddings.pt` | [4025, 2048] | All 5 TTA views stacked |
| `metadata.json` | — | Embedding parameters |