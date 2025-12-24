from configs.cfg import CFG
import pandas as pd
import numpy as np
from datetime import datetime

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