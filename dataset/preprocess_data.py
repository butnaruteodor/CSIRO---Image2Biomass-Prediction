from configs.cfg import CFG
import pandas as pd
import numpy as np
from datetime import datetime

def check_splits(splitter, df):
    fold_stats = []
    for fold, (tr_idx, val_idx) in enumerate(splitter):
        # Get the actual data for this fold
        train_fold = df.iloc[tr_idx]
        val_fold   = df.iloc[val_idx]
        
        # Calculate stats
        n_train = len(train_fold)
        n_val   = len(val_fold)
        ratio   = n_val / (n_train + n_val) * 100
        
        # Check "Hardness" (Mean target value)
        # If one fold has a mean of 100 and another 500, your folds are NOT balanced.
        val_mean_total_dry = val_fold['Dry_Total_g'].mean()
        val_mean_green_dry = val_fold['Dry_Green_g'].mean()
        val_mean_dead_dry = val_fold['Dry_Dead_g'].mean()
        val_mean_clover_dry = val_fold['Dry_Clover_g'].mean()
        val_mean_gdm = val_fold['GDM_g'].mean()

        val_mean_weighted = CFG.R2_WEIGHTS[4] * val_mean_total_dry + CFG.R2_WEIGHTS[0] * val_mean_green_dry + CFG.R2_WEIGHTS[1] * val_mean_dead_dry + CFG.R2_WEIGHTS[2] * val_mean_clover_dry + CFG.R2_WEIGHTS[3] * val_mean_gdm
        
        print(f"{fold+1:<5} | {n_train:<12} | {n_val:<10} | {ratio:<6.2f}% | Dry_Total_g:{val_mean_total_dry:<8.4f} | Dry_Green_g:{val_mean_green_dry:<8.4f} | Dry_Dead_g:{val_mean_dead_dry:<8.4f} | Dry_Clover_g:{val_mean_clover_dry:<8.4f} | GDM_g:{val_mean_gdm:<8.4f}  | Weighted_g:{val_mean_weighted:<8.4f}")
        
        fold_stats.append(n_val)

    # 3. Check Deviation
    mean_size = np.mean(fold_stats)
    max_dev = np.max(np.abs(fold_stats - mean_size)) / mean_size * 100
    print("-" * 65)
    print(f"Max deviation from ideal size: {max_dev:.2f}%")

def get_df():
    print("Loading data...")
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_wide = df_wide[['image_path'] + CFG.ALL_TARGET_COLS]
    print(f"{len(df_wide)} training images")

    # Aux task
    aux_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    df_aux = df_long[aux_cols].drop_duplicates().reset_index(drop=True)

    df_wide = df_wide.merge(df_aux, on='image_path', how='left')

    df_wide['State_idx'],   STATE_MAP   = pd.factorize(df_wide['State'])
    df_wide['Species_idx'], SPECIES_MAP = pd.factorize(df_wide['Species'])

    # 2. Convert Date to cyclical features (we'll predict these)
    df_wide['Sampling_Date'] = pd.to_datetime(df_wide['Sampling_Date'])
    df_wide['day_of_year'] = df_wide['Sampling_Date'].dt.dayofyear
    df_wide['day_sin'] = np.sin(2 * np.pi * df_wide['day_of_year'] / 365.25)
    df_wide['day_cos'] = np.cos(2 * np.pi * df_wide['day_of_year'] / 365.25)

    df_wide['group'] = df_wide['State'].astype(str) + "_" + df_wide['Sampling_Date'].astype(str)
    df_wide['biomass_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False)

    return df_wide