from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
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

    # df_wide['group'] = df_wide['State'].astype(str) + "_" + df_wide['Sampling_Date'].astype(str)
    df_wide['group'] = df_wide['Sampling_Date'].astype(str)
    df_wide['biomass_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False)

    return df_wide

def create_hard_extrapolation_split(df, target_col='Dry_Total_g', percentile=0.80):
    """
    Splits the dataframe based on target VALUE, not random chance.
    
    Train: The bottom 80% of biomass (Small/Medium plants)
    Test:  The top 20% of biomass (Large/Peak plants)
    
    Purpose: Tests if the model/TTA can predict values HIGHER than it has ever seen.
    """
    # 1. Determine the cutoff value
    cutoff_value = df[target_col].quantile(percentile)
    
    # 2. Strict Cut
    train_df = df[df[target_col] < cutoff_value].reset_index(drop=True)
    val_df   = df[df[target_col] >= cutoff_value].reset_index(drop=True)
    
    print(f"\n--- HARD EXTRAPOLATION SPLIT ---")
    print(f"Cutoff Value (80th percentile): {cutoff_value:.2f}")
    print(f"Train Set: {len(train_df)} images (Max Mass: {train_df[target_col].max():.2f})")
    print(f"Valid Set: {len(val_df)} images (Min Mass: {val_df[target_col].min():.2f})")
    print(f"The Validation set contains ONLY values unseen in Training.")
    
    return train_df, val_df

# Define your weights exactly as you listed them
WEIGHTS_MAP = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g':  0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g':       0.2,
    'Dry_Total_g': 0.5
}

def calculate_weighted_score(df):
    """Helper to create the column just like you did in check_splits"""
    # Start with 0
    weighted_col = np.zeros(len(df))
    for col, w in WEIGHTS_MAP.items():
        weighted_col += df[col] * w
    return weighted_col

def find_best_seed(df, n_seeds=2000):
    """
    Iterates through random seeds to find the one that minimizes the 
    statistical difference between folds for critical targets.
    """
    
    # 1. Prepare Data for Search
    # We calculate the weighted column purely for the balancing metric
    df_search = df.copy()
    df_search['Weighted_g'] = calculate_weighted_score(df_search)
    
    # We want to balance these specific columns. 
    # We prioritize 'Dry_Clover_g' because it is sparse/weird (your Fold 4 issue).
    # We prioritize 'Weighted_g' because it represents the overall difficulty.
    targets_to_balance = ['Weighted_g', 'Dry_Clover_g', 'Dry_Dead_g']
    
    best_seed = -1
    lowest_penalty = float('inf')
    
    print(f"🔍 Searching {n_seeds} seeds for the most balanced split...")
    
    for seed in tqdm(range(n_seeds)):
        # Initialize the splitter with the current seed
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # We collect the means of our targets for each fold
        fold_means = {t: [] for t in targets_to_balance}
        
        try:
            # Iterate through the folds
            for _, val_idx in sgkf.split(df_search, df_search['biomass_bin'], groups=df_search['group']):
                val_fold = df_search.iloc[val_idx]
                
                for t in targets_to_balance:
                    fold_means[t].append(val_fold[t].mean())
            
            # --- CALCULATE PENALTY ---
            # We use Coefficient of Variation (Std Dev / Mean). 
            # This normalizes the score so 'Total_g' (big numbers) doesn't overpower 'Clover' (small numbers).
            current_penalty = 0
            for t in targets_to_balance:
                means_array = np.array(fold_means[t])
                # If global mean is 0, avoid division by zero
                global_mean = df_search[t].mean() + 1e-6 
                
                # How much do the folds deviate from each other?
                cv = np.std(means_array) / global_mean
                current_penalty += cv
            
            # If this seed is more balanced than the previous best, save it
            if current_penalty < lowest_penalty:
                lowest_penalty = current_penalty
                best_seed = seed
                
        except ValueError:
            continue

    print("-" * 50)
    print(f"✅ Best Seed Found: {best_seed}")
    print(f"📉 Penalty Score: {lowest_penalty:.4f} (Lower is better)")
    
    return best_seed