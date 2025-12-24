from torch.amp import autocast
from utils.eval import *
import torch
from tqdm import tqdm

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
    scheduler.step()
    return running / len(loader)


def train_epoch_base(model, loader, opt, scheduler, device, scaler):
    model.train()
    running = 0.0

    opt.zero_grad()
    for i, (l, r, lab) in enumerate(tqdm(loader, desc='train', leave=False)):
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with autocast('cuda',dtype=torch.bfloat16):
            (p_tot, p_gdm, p_green) = model(l, r)
            loss_reg = weighted_biomass_loss(p_tot, p_gdm, p_green, lab, use_huber=False)
            total_loss = loss_reg
        
        loss = total_loss / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        # loss.backward()
        running += loss.item() * l.size(0) * CFG.GRAD_ACC

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
def valid_epoch_base(model, loader, device):
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

    # Compute weighted R²
    weighted_r2 = global_weighted_r2_score(true_labels, pred_all)
    return running_loss / len(loader.dataset), weighted_r2