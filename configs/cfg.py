import os, torch, numpy as np

class CFG:
    BASE_PATH       = 'csiro-biomass'
    TRAIN_CSV       = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    MODEL_DIR       = 'out'
    N_FOLDS         = 5
    LOG             = False

    # TIMM
    MODEL_NAME = 'vit_large_patch16_dinov3'
    PATIENCE     = 10
    
    # CLIP
    CLIP_NAME    = 'convnext_base_w'
    CLIP_FT_NAME = "laion2b_s13b_b82k_augreg"
    CLIP_PATIENCE = 10
    CLIP_EPOCHS = 50
    CLIP_WD = 0.01

    CHECKPOINT_PATH = None #'adapters/r8/lora_finetuned_convnext_base_r8.pt'
    FREEZE_BACKBONE = True

    IMG_SIZE     = 768 
    ALPHA_CLIP   = 0.1
    BATCH_SIZE   = 8
    GRAD_ACC     = 1    # effective batch = 8
    NUM_WORKERS  = 4
    EPOCHS       = 100
    LR           = 1e-3
    WD           = 0.01 #1e-2 convnext
    WARMUP_EPOCHS = 4
    WARMUP_HEAD_EPOCHS = 5

    DETERMINISTIC = True
    SEED = 204 # 3858 state+date 204 date

    TARGET_COLS    = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS   = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS     = np.array([0.1, 0.1, 0.1, 0.2, 0.5])  # matches metric
    W_SPECIES = 0.25
    W_STATE = 0.25
    W_CONT = 0.5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')