import gc
# from catboost import CatBoostRegressor
from torch.amp import autocast
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import copy
from datetime import datetime
import optuna

from dataset.preprocess_data import get_plant_neighbor_map
from utils.eval import *
from utils.utils import *
from dataset.biomass_dataset import *
from utils.augs import *
from configs.deterministic import *
from models.models import *
from log.logging import *
from configs.cfg import *

def train_epoch_clip(model, loader, opt, scheduler, device, scaler, text_anchors):
    model.train()
    running = 0
    loss_fn = nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm(loader, desc='train', leave=False)):
        tiles = batch["pixel_values"].squeeze(0).to(device)
        indices = batch["index"].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            tile_features = model.encode_image(tiles)
            img_embedding = tile_features.mean(dim=0, keepdim=True)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            # Global Loss: Compare against train anchors
            logits = (img_embedding @ text_anchors.T) * logit_scale
            total_loss = loss_fn(logits, indices) 

        loss = total_loss / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        running += loss.item()
        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
    # scheduler.step()
    return running / len(loader)


def train_epoch_base(model, loader, opt, scheduler, device, scaler, deltas, epoch_num):
    model.train()
    running = 0.0
    opt.zero_grad()
    for i, (features, targets, n_feat, n_target) in enumerate(tqdm(loader, desc='train', leave=False)):
        features, targets = features.to(device), targets.to(device)
        n_feat, n_target = n_feat.to(device), n_target.to(device)
        # if np.random.rand() < 0.5: 
        #     alpha = 0.8
        #     lam = np.random.beta(alpha, alpha)
            
        #     # 2. Shuffle indices to find "Partner" for every sample
        #     batch_size = features.size(0)
        #     index = torch.randperm(batch_size).to(device)
            
        #     # 3. Create the Mixed Batch
        #     # We blend Feature A with Feature B
        #     features = slerp(lam, features, features[index])

        #     # 4. Create the Mixed Targets
        #     # We blend Label A with Label B (Works perfectly for regression!)
        #     targets = lam * targets + (1 - lam) * targets[index]

        epoch_target_weights = get_interpolated_weights(
            current_epoch=epoch_num,
            total_epochs=CFG.EPOCHS,  # or a shorter duration if you want faster transition
            start_weights=CFG.R2_WEIGHTS_TRAIN,
            end_weights=CFG.R2_WEIGHTS_VAL
            ).to(device)
        constraint_weight = get_constraint_weight(epoch_num, start_epoch=3,ramp_epochs=10,max_weight=1.0)
        with autocast('cuda',dtype=torch.bfloat16):
            (p_tot, p_gdm, p_green, p_clover, p_dead) = model(features)
            loss_reg = weighted_biomass_loss(p_tot, p_gdm, p_green, p_clover, p_dead, targets, deltas, constraint_weight, epoch_target_weights)
            # loss_reg = weighted_biomass_log_loss(p_tot, p_gdm, p_green, lab)
            total_loss = loss_reg
        
        loss = total_loss / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        # loss.backward()
        running += loss.item() * features.size(0) * CFG.GRAD_ACC

        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            # opt.step()
            opt.zero_grad()

    scheduler.step()
    return running / len(loader.dataset)

def train_epoch_base_w_clip(model, loader, opt, scheduler, device, scaler, text_anchors, use_clip=False):
    model.train()
    running = 0.0
    running_clip = 0.0
    opt.zero_grad()
    for i, (l, r, lab, idx) in enumerate(tqdm(loader, desc='train', leave=False)):
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        text_anchors = text_anchors.to(device)
        idx = idx.to(device)
        with autocast('cuda',dtype=torch.bfloat16):
            (p_tot, p_gdm, p_green), img_embeds = model(l, r)
            loss_reg = weighted_biomass_loss(p_tot, p_gdm, p_green, lab, use_huber=False)
            total_loss = loss_reg
            if use_clip:
                l_clip = global_clip_loss(img_embeds, text_anchors, idx, model.logit_scale)
                
                total_loss += (0.5 * l_clip)
        
        loss = total_loss / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        # loss.backward()
        running += loss.item() * l.size(0) * CFG.GRAD_ACC
        if use_clip:
            running_clip +=l_clip.item() * l.size(0) * CFG.GRAD_ACC
        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            # opt.step()
            opt.zero_grad()
    if use_clip:
        print(f"Clip LOSS: {running_clip}") 
    scheduler.step()
    return running / len(loader.dataset)

@torch.no_grad()
def valid_epoch_base(model, loader, device, deltas):
    model.eval()
    running_loss = 0.0
    preds = {'total':[], 'gdm':[], 'green':[]}
    all_labels = []

    for (features, targets) in tqdm(loader, desc='valid', leave=False):
        features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast('cuda',dtype=torch.bfloat16):
            (p_tot, p_gdm, p_green, p_clover, p_dead) = model(features)

            loss = weighted_biomass_loss(p_tot, p_gdm, p_green, p_clover, p_dead, targets, deltas, 0, CFG.R2_WEIGHTS_VAL)
        running_loss += loss.item() * features.size(0)

        preds['total'].extend(p_tot.cpu().float().numpy().ravel())
        preds['gdm'].extend(p_gdm.cpu().float().numpy().ravel())
        preds['green'].extend(p_green.cpu().float().numpy().ravel())
        all_labels.extend(targets.cpu().float().numpy())

    # Convert to numpy
    pred_total = np.array(preds['total'])
    pred_gdm   = np.array(preds['gdm'])
    pred_green = np.array(preds['green'])
    true_labels = np.stack(all_labels)  # (N, 5)

    # Compute derived
    pred_clover = np.clip(pred_gdm - pred_green, 0, None)
    pred_dead   = np.clip(pred_total - pred_gdm, 0, None)

    # Stack predictions in correct order
    pred_all = np.stack([
        pred_green,      # Dry_Green_g
        pred_dead,       # Dry_Dead_g
        pred_clover,     # Dry_Clover_g
        pred_gdm,        # GDM_g
        pred_total       # Dry_Total_g
    ], axis=1)

    # Compute weighted R²
    weighted_r2 = global_weighted_r2_score(true_labels, pred_all)
    per_target_r2 = per_target_r2_score(true_labels, pred_all)
    return running_loss / len(loader.dataset), weighted_r2, per_target_r2

def valid_epoch_clip(model, loader, text_anchors, device, val_index_offset=0):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='val', leave=False)):
            tiles = batch["pixel_values"].squeeze(0).to(device)
            local_index = batch["index"].to(device)
            target_index = local_index + val_index_offset
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                tile_features = model.encode_image(tiles)
                img_embedding = tile_features.mean(dim=0, keepdim=True)
                img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                
                logit_scale = model.logit_scale.exp()
                logits = (img_embedding @ text_anchors.T) * logit_scale
                
                loss = loss_fn(logits, target_index)
                total_loss += loss.item()
                
                # --- CALCULATE ACCURACY ---
                # Get the top 5 scores and their indices
                # top5_indices shape: [1, 5]
                _, top5_indices = logits.topk(5, dim=-1) 
                
                # Check Top 1
                if top5_indices[0, 0] == target_index:
                    correct_top1 += 1
                # Check Top 5 (Is target inside the list of 5?)
                if target_index in top5_indices[0]:
                    correct_top5 += 1
                
                total += 1
                
    acc_1 = (correct_top1 / total) * 100.0
    acc_5 = (correct_top5 / total) * 100.0
    avg_loss = total_loss / total
    
    return avg_loss, acc_1, acc_5

def precompute_epoch_anchors(model, dataset, tokenizer, device):
    """
    Generates FRESH random text for the entire dataset and encodes it.
    Returns: Tensor [Dataset_Size, Embed_Dim]
    """
    model.eval() # Text encoder is frozen/eval anyway
    print("Regenerating global text anchors for this epoch...")
    
    # 1. Ask Dataset to generate fresh strings (Triggering dropout/shuffle)
    # We manually iterate because DataLoader would load images too
    all_texts = []
    for i in range(len(dataset)):
        # We access the internal generation method directly
        row = dataset.df.iloc[i]
        # Force training=True to get the augmentations
        txt = dataset._generate_text_description(row, training=True) 
        all_texts.append(txt)
    
    # 2. Tokenize & Encode
    # Since N=357 is small, we can do this in one pass without OOM.
    with torch.no_grad():
        tokens = tokenizer(all_texts).to(device)
        anchors = model.encode_text(tokens)
        anchors = anchors / anchors.norm(dim=-1, keepdim=True)
        
    return anchors

def train_clip(tr_df, val_df):
    model, preprocess, tokenizer = get_lora_model()
    model = model.to(CFG.DEVICE)

    tr_set = BiomassDatasetClip(tr_df, train_aug, None, CFG.TRAIN_IMAGE_DIR, preprocess)
    val_set = BiomassDatasetClip(val_df, None, None, CFG.TRAIN_IMAGE_DIR, preprocess,is_train=False)

    g=get_generator()
    tr_loader  = DataLoader(tr_set,  batch_size=CFG.BATCH_SIZE, shuffle=True,
                                num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True, worker_init_fn=seed_worker,generator=g)
    val_loader = DataLoader(val_set, batch_size=CFG.BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker,generator=g)

    print("Pre-computing text anchors...")
    model.eval()
    # train_texts = tr_set.texts
    # train_text_tokens = tokenizer(train_texts).to(CFG.DEVICE)

    val_texts = val_set.texts
    val_text_tokens = tokenizer(val_texts).to(CFG.DEVICE)
    
    with torch.no_grad():
        # train_text_anchors = model.encode_text(train_text_tokens)
        # train_text_anchors = train_text_anchors / train_text_anchors.norm(dim=-1, keepdim=True)
        val_text_anchors = model.encode_text(val_text_tokens)
        val_text_anchors = val_text_anchors / val_text_anchors.norm(dim=-1, keepdim=True)
    
    print("Anchors computed. Starting Training...")
    # all_text_anchors = torch.cat((train_text_anchors,val_text_anchors),dim=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR,weight_decay=CFG.CLIP_WD)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = None

    best_model_weights=None
    best_val_loss = 100
    optimizer.zero_grad()
    patience = 0
    for epoch in range(CFG.CLIP_EPOCHS):
        train_anchors = precompute_epoch_anchors(model, tr_set, tokenizer, CFG.DEVICE)
        train_loss = train_epoch_clip(model,tr_loader,optimizer,scheduler,CFG.DEVICE,scaler,train_anchors)
        val_loss, acc_1, acc_5 = valid_epoch_clip(model,val_loader,val_text_anchors,CFG.DEVICE,0)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc 1: {acc_1:.2f}% | Val Acc 5: {acc_5:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"--> New Best Loss! Saving model...")
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= CFG.CLIP_PATIENCE:
                print(f'EARLY STOP (no improvement in {CFG.CLIP_PATIENCE} epochs)')
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    del optimizer, val_loader, tr_loader
    return model

def train_base(fold_dir, tr_df, val_df, model_id, model_state_dict=None, group_name=None, test_df=None):
    train_labels_tensor = torch.tensor(
        tr_df[["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]].values, 
        dtype=torch.float32
    )
    deltas = calculate_deltas(train_labels_tensor)

    train_data = torch.load(f"{fold_dir}/train.pt")
    val_data = torch.load(f"{fold_dir}/val.pt")

    VIEWS_PER_PLANT = 20 # <--- DOUBLE CHECK THIS
    neighbor_map = get_plant_neighbor_map(train_data['features'], views_per_plant=VIEWS_PER_PLANT)

    train_ds = PairedDataset(train_data['features'], train_data['targets'], neighbor_map)
    val_ds = torch.utils.data.TensorDataset(val_data['features'], val_data['targets']) 

    g = get_generator()
    tr_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False, worker_init_fn=seed_worker,generator=g) # Huge batch size!
    val_loader   = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False, worker_init_fn=seed_worker,generator=g)
    
    print("Building model...")
    model = BiomassOneLayer(2048)
    model = model.to(CFG.DEVICE)
    # model = nn.DataParallel(model)
    parameters = model.parameters()

    optimizer = torch.optim.AdamW(parameters, lr=CFG.LR, weight_decay=CFG.WD)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3, # Start from a very small LR
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

    init_logger(model_id, group_name)

    best_r2 = -np.inf
    patience = 0
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(1, CFG.EPOCHS+1):
        tr_loss = train_epoch_base(model, tr_loader, optimizer, scheduler, CFG.DEVICE, scaler, deltas, epoch)
        val_loss, val_r2, per_target_r2 = valid_epoch_base(model, val_loader, CFG.DEVICE, deltas)
        test_loss, test_val_r2=0,0

        if val_r2>best_r2:
            print(f'Epoch {epoch:02d} | '
                f'TrainLoss {tr_loss:.5f} | '
                f'ValLoss {val_loss:.5f} | '
                f'ValR² {val_r2:.4f} {"(BEST)" if val_r2 > best_r2 else ""} | '
                f'GreenR² {per_target_r2["Dry_Green"]:.5f} | '
                f'DeadR² {per_target_r2["Dry_Dead"]:.5f} | '
                f'CloverR² {per_target_r2["Dry_Clover"]:.5f} | '
                f'GDMR² {per_target_r2["GDM"]:.5f} | '
                f'TotalR² {per_target_r2["Dry_Total"]:.5f}')
        if test_df is not None:
            print(f'TestLoss {test_loss:.5f} | TestR2: {test_val_r2:.4f}')
        log_data = {"train_loss": tr_loss, "val_loss": val_loss, "val_r2": val_r2, "best_r2":best_r2, "test_loss": test_loss, "test_val_r2":test_val_r2, "r2_green": per_target_r2['Dry_Green'],
                    "r2_dead": per_target_r2['Dry_Dead'],"r2_clover": per_target_r2['Dry_Clover'],"r2_gdm": per_target_r2['GDM'],"r2_total": per_target_r2['Dry_Total']}

        if val_r2 > best_r2:
            best_r2 = val_r2
            save_path = os.path.join(CFG.MODEL_DIR, f'best_model_fold{model_id}.pth')
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
            print(f'SAVED (R²: {best_r2:.4f})')
            patience = 0
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f'EARLY STOP (no improvement in {CFG.PATIENCE} epochs)')
                for e in range(epoch + 1, CFG.EPOCHS + 1):
                    log(log_data, e)
                break

        log_data = {"train_loss": tr_loss, "val_loss": val_loss, "val_r2": val_r2, "best_r2":best_r2, "test_loss": test_loss, "test_val_r2":test_val_r2, "r2_green": per_target_r2['Dry_Green'],
                    "r2_dead": per_target_r2['Dry_Dead'],"r2_clover": per_target_r2['Dry_Clover'],"r2_gdm": per_target_r2['GDM'],"r2_total": per_target_r2['Dry_Total']}
        log(log_data, epoch)

    finish_logger()
    del optimizer,val_loader,tr_loader,model
    gc.collect()
    torch.cuda.empty_cache()
    return best_r2


def train_tree(fold_dir, fold, verbose=False):
    print(f"Loading data from {fold_dir}...")
    train_data = torch.load(os.path.join(fold_dir, "train.pt"), map_location='cpu')
    val_data   = torch.load(os.path.join(fold_dir, "val.pt"),   map_location='cpu')

    # 1. Step slicing to get Original Images only
    # Ensure your data is structured [Orig, Aug1, Aug2...] 
    # If not sure, this is dangerous!
    step_size = 20 
    X_train = train_data['features'][::step_size].numpy()
    y_train = train_data['targets'][::step_size].numpy()
    
    X_val   = val_data['features'].numpy()
    y_val   = val_data['targets'].numpy()

    # 2. Define Best Parameters for each Target
    # You can tweak these manually or use Optuna to find them
    MODEL_PARAMS = {
        'Total':  {'depth': 6, 'l2_leaf_reg': 3, 'learning_rate': 0.02, 'iterations': 3000},
        'Green':  {'depth': 6, 'l2_leaf_reg': 3, 'learning_rate': 0.02, 'iterations': 3000},
        'Clover': {'depth': 4, 'l2_leaf_reg': 7, 'learning_rate': 0.01, 'iterations': 2000}, # Sparse -> Regularize more!
        'GDM':    {'depth': 4, 'l2_leaf_reg': 7, 'learning_rate': 0.01, 'iterations': 2000}, # Sparse -> Regularize more!
    }

    targets_to_train = {
        'Green': 0,
        'Clover': 2,
        'GDM': 3,
        'Total': 4
    }
    
    # def objective(trial):
    #     # 1. Suggest params
    #     param = {
    #         'iterations': 2000,
    #         'depth': trial.suggest_int('depth', 3, 8),
    #         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
    #         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
    #         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    #         'task_type': 'CPU',
    #         # 'used_ram_limit': '4gb',
    #         'verbose': 0
    #     }
        
    #     # 2. Setup Data (Use only Fold 0 for speed)
    #     # ... load X_train, y_train (for Clover column only) ...
        
    #     model = CatBoostRegressor(**param)
    #     model.fit(X_train, y_train[:, col_idx], eval_set=(X_val, y_val[:, col_idx]), early_stopping_rounds=50)
        
    #     return model.get_best_score()['validation']['RMSE']

    val_preds = np.zeros_like(y_val)
    
    # --- 3. Train Loop ---
    for name, col_idx in targets_to_train.items():
        if verbose:
            print(f"  Training {name} model...")
            
        # Get specific params or default to generic ones
        params = MODEL_PARAMS.get(name, {'depth': 6, 'learning_rate': 0.01})
            
        model = CatBoostRegressor(
            loss_function='RMSE',
            eval_metric='RMSE',
            early_stopping_rounds=100,
            verbose=0,
            allow_writing_files=False,
            task_type="GPU", # Use GPU
            **params # Unpack the target-specific params
        )
        
        model.fit(
            X_train, y_train[:, col_idx],
            eval_set=(X_val, y_val[:, col_idx]),
            use_best_model=True
        )
        filename = f"catboost_{name.lower()}_fold{fold}.cbm"
        
        # 2. Define the full path
        # You can save this in the fold directory or your main models directory
        save_path = os.path.join(fold_dir, filename)
        
        # 3. Save the model
        model.save_model(save_path)
        
        # study = optuna.create_study(direction='minimize')
        # study.optimize(objective, n_trials=20)
        # print(study.best_params)

        # Predict
        val_preds[:, col_idx] = model.predict(X_val)
        
        best_rmse = model.get_best_score()['validation']['RMSE']
        print(f"  {name:<6} | RMSE: {best_rmse:.3f} | Params: D={params['depth']}, LR={params['learning_rate']}")

    # --- 4. Physics & Derivation (CORRECTED) ---
    
    # A. Correct Physics: Dead = Total - (Green + Clover + GDM)
    # Your previous code only subtracted GDM. You must subtract ALL living parts.
    val_preds[:, 1] = val_preds[:, 4] - val_preds[:, 3]
    
    # B. Sanity Check: If components > Total, scale them down?
    # Or just clamp Dead to 0 (which implicitly says Total was underestimated)
    val_preds[:, 1] = np.maximum(val_preds[:, 1], 0)
    
    # C. Global Clamp (Crucial for score)
    val_preds = np.maximum(val_preds, 0)
    
    final_score = global_weighted_r2_score(y_val, val_preds)
    print(f"Fold {fold} Result: Weighted R2 = {final_score:.5f}")
    
    return final_score

import joblib  # For saving the model
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_ridge(all_embeddings, all_targets, save_path="ridge_models.pkl"):
    print(f"Training Ridge on full dataset: {all_embeddings.shape}")
    
    target_map = {0: 'Green', 2: 'Clover', 3: 'GDM', 4: 'Total'}
    final_models = {}
    
    # Define Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('krr', KernelRidge(kernel='rbf'))
    ])

    # Parameters
    param_grid = {
        "krr__alpha": [0.001, 0.005, 0.01, 0.1, 1.0, 10.0],
        "krr__gamma": [0.0001, 0.005, 0.001, 0.01, 0.1, None] 
    }
    
    # DEFINE MULTIPLE SCORERS
    # We want to see R2, but we want to optimize (refit) based on RMSE.
    scorers = {
        'RMSE': 'neg_root_mean_squared_error',
        'R2': 'r2'
    }

    for col_idx, name in target_map.items():
        print(f"Fitting {name} with Scaled Kernel Ridge...")
        
        # refit='RMSE' means: "Find the params that give the best RMSE, and fit the final model using those."
        clf = GridSearchCV(
            pipe, 
            param_grid, 
            cv=5, 
            scoring=scorers, 
            refit='RMSE',
            n_jobs=-1  # Use all CPU cores for speed
        )
        
        clf.fit(all_embeddings, all_targets[:, col_idx])
        
        final_models[name] = clf.best_estimator_
        
        # --- EXTRACT SCORES ---
        # Get the index of the best parameters
        best_idx = clf.best_index_
        
        # Extract the specific scores for that index
        best_rmse = -clf.cv_results_['mean_test_RMSE'][best_idx]
        best_r2   = clf.cv_results_['mean_test_R2'][best_idx]
        
        print(f"  > Best RMSE: {best_rmse:.4f} | Best R2: {best_r2:.4f}")
        print(f"  > Params: {clf.best_params_}")

    print(f"Saving models to {save_path}...")
    joblib.dump(final_models, save_path)
    
    return final_models

from sklearn.neighbors import KNeighborsRegressor
import joblib

def train_knn(fold_dir, fold, verbose=False):
    if verbose:
        print(f"  🧠 Training KNN (Fold {fold})...")

    # 1. Load Data
    train_data = torch.load(os.path.join(fold_dir, "train.pt"), map_location='cpu')
    val_data   = torch.load(os.path.join(fold_dir, "val.pt"),   map_location='cpu')

    # Convert to Numpy
    # Step size 1 = Use ALL data for KNN (It needs dense data to work well)
    X_train = train_data['features'].numpy()
    y_train = train_data['targets'].numpy()
    
    X_val   = val_data['features'].numpy()
    y_val   = val_data['targets'].numpy()

    # 2. Select Targets: Green(0), Clover(2), GDM(3), Total(4)
    # We skip Dead(1) because we derive it.
    target_cols = [0, 2, 3, 4]
    y_train_sub = y_train[:, target_cols]

    # 3. Train KNN
    # metric='cosine' is crucial for Deep Learning embeddings
    # n_neighbors=30 is a robust balance for 2048-dim vectors
    model = KNeighborsRegressor(n_neighbors=40, weights='distance', metric='cosine', n_jobs=-1)
    model.fit(X_train, y_train_sub)

    # 4. Predict on Validation
    val_preds_raw = model.predict(X_val)
    
    # 5. Reconstruct the full 5-column format
    # Shape: (N_val, 5) -> [Green, Dead, Clover, GDM, Total]
    val_preds = np.zeros_like(y_val)
    
    val_preds[:, 0] = val_preds_raw[:, 0] # Green
    val_preds[:, 2] = val_preds_raw[:, 1] # Clover
    val_preds[:, 3] = val_preds_raw[:, 2] # GDM
    val_preds[:, 4] = val_preds_raw[:, 3] # Total
    
    # Derive Dead: Total - (Green + Clover + GDM)
    val_preds[:, 1] = val_preds[:, 4] - val_preds[:, 3]
    print(val_preds)
    
    # 6. Physics Clamp (Crucial)
    # val_preds = np.maximum(val_preds, 0)
    
    # 7. Score
    score = global_weighted_r2_score(y_val, val_preds)
    
    if verbose:
        print(f"  ✅ KNN Fold {fold} Score: {score:.5f}")

    # 8. SAVE Predictions and Model
    # Saving predictions is vital for your ensemble weight search later
    np.save(os.path.join(fold_dir, f"val_preds_knn_fold{fold}.npy"), val_preds)
    joblib.dump(model, os.path.join(fold_dir, f"knn_model_fold{fold}.pkl"))
    
    return score

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    running_loss = 0.0
    preds = {'total':[], 'gdm':[], 'green':[]}
    all_labels = []

    for l, r, lab in tqdm(loader, desc='valid', leave=False):
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with autocast('cuda',dtype=torch.bfloat16):
            (p_tot, p_gdm, p_green) = model(l, r)

            loss = weighted_biomass_loss(p_tot, p_gdm, p_green, lab)
        running_loss += loss.item() * l.size(0)

        preds['total'].extend(p_tot.cpu().float().numpy().ravel())
        preds['gdm'].extend(p_gdm.cpu().float().numpy().ravel())
        preds['green'].extend(p_green.cpu().float().numpy().ravel())
        all_labels.extend(lab.cpu().float().numpy())

    # Convert to numpy
    pred_total = np.array(preds['total'])
    pred_gdm   = np.array(preds['gdm'])
    pred_green = np.array(preds['green'])
    true_labels = np.stack(all_labels)  # (N, 5)

    # Compute derived
    pred_clover = np.clip(pred_gdm - pred_green, 0, None)
    pred_dead   = np.clip(pred_total - pred_gdm, 0, None)

    # Stack predictions in correct order
    pred_all = np.stack([
        pred_green,      # Dry_Green_g
        pred_dead,       # Dry_Dead_g
        pred_clover,     # Dry_Clover_g
        pred_gdm,        # GDM_g
        pred_total       # Dry_Total_g
    ], axis=1)

    return pred_all, true_labels

