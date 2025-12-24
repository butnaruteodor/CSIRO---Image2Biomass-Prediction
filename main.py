import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from configs.cfg import CFG
from torch.amp import GradScaler
from dataset.biomass_dataset import *
from utils.augs import *
from configs.deterministic import *
from models.models import *
from train.train import *

if __name__ == "__main__":
    set_seed(CFG.SEED,CFG.DETERMINISTIC)
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_wide = df_wide[['image_path'] + CFG.ALL_TARGET_COLS]
    aux_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    df_aux = df_long[aux_cols].drop_duplicates().reset_index(drop=True)
    df_wide = df_wide.merge(df_aux, on='image_path', how='left')
    df_wide['biomass_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False)
    df_wide['stratify_key'] = df_wide['Species'].astype(str) + "_" + df_wide['State'].astype(str)
    
    # Check for singletons! 
    # Stratified Split crashes if a group has only 1 member (can't split 1 item into 2 sets).
    # We filter out rare groups or assign them to a "misc" group for the split.
    counts = df_wide['stratify_key'].value_counts()
    singletons = counts[counts < 2].index
    
    # Fallback: If a group has only 1 sample, we treat it as part of a generic group 
    # just for the purpose of splitting, so the code doesn't crash.
    split_key = df_wide['stratify_key'].apply(lambda x: 'misc' if x in singletons else x)

    # --- C. PERFORM STRATIFIED SPLIT ---
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    # This returns the INDICES for the split
    train_idx, val_idx = next(splitter.split(df_wide, split_key))
    print(val_idx)
    # Create the actual sub-dataframes
    train_df = df_wide.iloc[train_idx].reset_index(drop=True)
    val_df = df_wide.iloc[val_idx].reset_index(drop=True)
    g=get_generator()
    train_dataset = BiomassDatasetBase(train_df, get_spatial_transforms(), get_photometric_transforms(), CFG.TRAIN_IMAGE_DIR)
    val_dataset = BiomassDatasetBase(val_df, None, get_val_transforms(), CFG.TRAIN_IMAGE_DIR)
    
    print(f"Data Split: {len(train_dataset)} Training | {len(val_dataset)} Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=CFG.NUM_WORKERS, worker_init_fn=seed_worker, generator=g)

    model = BiomassModelMLP(
            CFG.MODEL_NAME, 
            freeze_backbone=CFG.FREEZE_BACKBONE,
            checkpoint_path=CFG.CHECKPOINT_PATH
        )
    model = model.to(CFG.DEVICE)

    if CFG.FREEZE_BACKBONE:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        parameters = model.parameters()

    optimizer = torch.optim.AdamW(parameters, lr=CFG.LR, weight_decay=CFG.WD)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2, # Start from a very small LR
        end_factor=1.0,
        total_iters=CFG.WARMUP_EPOCHS
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.EPOCHS - CFG.WARMUP_EPOCHS
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[CFG.WARMUP_EPOCHS]
    )
    scaler = GradScaler()
    best_r2 = -np.inf
    patience = 0
    for epoch in range(1, CFG.EPOCHS+1):
        tr_loss = train_epoch_base(model, train_loader, optimizer, scheduler, CFG.DEVICE, scaler)
        val_loss, val_r2 = valid_epoch_base(model, val_loader, CFG.DEVICE)

        print(f'Epoch {epoch:02d} | '
                    f'TrainLoss {tr_loss:.5f} | '
                    f'ValLoss {val_loss:.5f} | '
                    f'ValR² {val_r2:.4f} {"(BEST)" if val_r2 > best_r2 else ""}')
        if val_r2 > best_r2:
            best_r2 = val_r2
            save_path = os.path.join(CFG.MODEL_DIR, f'best_model.pth')
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
            print(f'   → SAVED (R²: {best_r2:.4f})')
            patience = 0
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f'   → EARLY STOP (no improvement in {CFG.PATIENCE} epochs)')

    del model, train_loader, val_loader, optimizer, main_scheduler
    torch.cuda.empty_cache()