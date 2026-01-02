import wandb

from configs.cfg import CFG

LOG_KEYS = ['MODEL_NAME', 'IMG_SIZE', 'FREEZE_BACKBONE', 'LR', 'WD', 'BATCH_SIZE', 'GRAD_ACC', 'EPOCHS', 'SEED']

def init_logger(fold, group_name):
    config = {k: v for k, v in vars(CFG).items() if k in LOG_KEYS}

    if CFG.LOG:
        wandb.init(
                project="csiro-biomass",
                group=group_name,           # Group all folds under this name
                name=f"{group_name}_f{fold}", # Unique name for this fold
                config=config,              # Log config for each run
                reinit=True                 # Allow re-initialization
            )
        
def log(log_data, epoch):
    if CFG.LOG:
        wandb.log(log_data, step=epoch)

def finish_logger():
    if CFG.LOG:
        wandb.finish()