"""
train_embeddings.py
Trains BiomassSimpleMLP on precomputed embeddings from extract_features_organized.
Uses EmbeddingAugmentationDataset to load from embeddings/ directory.
"""

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from configs.cfg import CFG
from dataset.biomass_dataset import *
from utils.augs import *
from configs.deterministic import *
from models.models import BiomassSimpleMLP
from dataset.preprocess_data import get_df, EmbeddingAugmentationDataset
from utils.eval import *
import torch, numpy as np, os, gc, json
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EMBED_DIR = 'embeddings'

# Read feature dimension from metadata
with open(os.path.join(EMBED_DIR, 'metadata.json')) as f:
    _meta = json.load(f)
FEATURE_DIM = _meta['embedding_dim']
print(f"Embedding dimension: {FEATURE_DIM}")

# ============================================================
# TRAINING / VALIDATION EPOCHS (for precomputed features)
# ============================================================

def train_epoch_mlp(model, loader, optimizer, scaler):
    model.train()
    running = 0.0
    optimizer.zero_grad()
    for feats, targets in tqdm(loader, desc='train', leave=False):
        feats, targets = feats.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        with autocast('cuda', dtype=torch.bfloat16):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, targets)
        scaler.scale(loss).backward()
        running += loss.item() * feats.size(0)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return running / len(loader.dataset)

@torch.no_grad()
def valid_epoch_mlp(model, loader):
    model.eval()
    running_loss = 0.0
    preds = {'total':[], 'gdm':[], 'green':[], 'clover':[], 'dead':[]}
    all_labels = []
    for feats, targets in tqdm(loader, desc='valid', leave=False):
        feats, targets = feats.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        with autocast('cuda', dtype=torch.bfloat16):
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, targets)
        running_loss += loss.item() * feats.size(0)
        preds['total'].extend(p_total.cpu().float().numpy().ravel())
        preds['gdm'].extend(p_gdm.cpu().float().numpy().ravel())
        preds['green'].extend(p_green.cpu().float().numpy().ravel())
        preds['clover'].extend(p_clover.cpu().float().numpy().ravel())
        preds['dead'].extend(p_dead.cpu().float().numpy().ravel())
        all_labels.extend(targets.cpu().float().numpy())
    
    pred_total = np.array(preds['total'])
    pred_gdm   = np.array(preds['gdm'])
    pred_green = np.array(preds['green'])
    pred_clover = np.array(preds['clover'])
    pred_dead  = np.array(preds['dead'])
    true_labels = np.stack(all_labels)
    
    pred_all = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
    weighted_r2 = global_weighted_r2_score(true_labels, pred_all)
    per_target = per_target_r2_score(true_labels, pred_all)
    return running_loss / len(loader.dataset), weighted_r2, per_target

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    set_seed(CFG.SEED, CFG.DETERMINISTIC)
    df_wide = get_df()
    
    # --- Split logic (unchanged) ---
    fold_indices = {}
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=204)
    splitter = skf.split(X=df_wide, y=df_wide['biomass_bin'])
    for fold_id, (train_idx, val_idx) in enumerate(splitter):
        fold_indices[fold_id] = val_idx
        print(f"Fold {fold_id} captured: {len(val_idx)} images")
    
    train_folds = [1, 2, 3, 4]
    val_fold    = 0
    test_fold   = 4
    
    train_idx_final = np.concatenate([fold_indices[f] for f in train_folds])
    val_idx_final   = fold_indices[val_fold]
    test_idx_final  = fold_indices[test_fold]
    
    print(f"Train Size: {len(train_idx_final)}")
    print(f"Val Size:   {len(val_idx_final)}")
    
    # --- Create datasets from precomputed embeddings ---
    train_set = EmbeddingAugmentationDataset(
        train_idx_final, EMBED_DIR, n_aug=15, is_train=True
    )
    val_set = EmbeddingAugmentationDataset(
        val_idx_final, EMBED_DIR, n_aug=15, is_train=False
    )
    
    g = get_generator()
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                              num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False,
                              num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    
    # --- Build model ---
    model = BiomassSimpleMLP(FEATURE_DIM).to(DEVICE)
    
    EPOCHS = 80
    PATIENCE = 12
    
    # --- Optimizer + Scheduler (PDF spec) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scheduler = cosine
    scaler = torch.amp.GradScaler('cuda')
    
    # --- Training loop ---
    best_r2 = -np.inf
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch_mlp(model, train_loader, optimizer, scaler)
        val_loss, val_r2, per_target = valid_epoch_mlp(model, val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch:02d} | TrainLoss {train_loss:.5f} | ValLoss {val_loss:.5f} | "
              f"ValR² {val_r2:.4f} {'(BEST)' if val_r2 > best_r2 else ''} | "
              f"Green {per_target['Dry_Green']:.4f} | Dead {per_target['Dry_Dead']:.4f} | "
              f"Clover {per_target['Dry_Clover']:.4f} | GDM {per_target['GDM']:.4f} | "
              f"Total {per_target['Dry_Total']:.4f}")
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\nBest Val R²: {best_r2:.4f}")