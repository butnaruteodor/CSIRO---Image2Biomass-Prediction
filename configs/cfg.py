import os, torch, numpy as np

class CFG:
    BASE_PATH       = 'csiro-biomass'
    TRAIN_CSV       = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    MODEL_DIR       = 'out'
    N_FOLDS         = 5

    # TIMM
    MODEL_NAME = 'convnext_base'
    PATIENCE     = 10
    
    # CLIP
    CLIP_NAME    = 'convnext_base_w'
    CLIP_FT_NAME = "laion2b_s13b_b82k_augreg"
    CLIP_PATIENCE = 5

    CHECKPOINT_PATH = None #'adapters/r8/lora_finetuned_convnext_base_r8.pt'
    FREEZE_BACKBONE = True

    IMG_SIZE     = 512 
    ALPHA_CLIP   = 0.1
    BATCH_SIZE   = 1
    GRAD_ACC     = 8    # effective batch = 8
    NUM_WORKERS  = 1
    EPOCHS       = 20
    LR           = 1e-3
    WD           = 0.01 #1e-2 convnext
    PATIENCE     = 10
    WARMUP_EPOCHS = 3
    WARMUP_HEAD_EPOCHS = 5

    DETERMINISTIC = True
    SEED = 694

    TARGET_COLS    = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS   = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS     = np.array([0.1, 0.1, 0.1, 0.2, 0.5])  # matches metric
    W_SPECIES = 0.25
    W_STATE = 0.25
    W_CONT = 0.5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')