"""
Training loop helpers for MLP on precomputed embeddings.
"""
import torch
from torch.amp import autocast
from tqdm import tqdm
import numpy as np

from src.config import CFG
from src.training.loss import weighted_biomass_loss
from src.evaluation.metrics import global_weighted_r2_score, per_target_r2_score


DEVICE = CFG.DEVICE


def _get_amp_context(device):
    """Return appropriate autocast context for the device."""
    if device.type == "cuda":
        return autocast("cuda", dtype=torch.bfloat16)
    return autocast("cpu")


def _get_scaler(device):
    """Return GradScaler for CUDA, None for CPU."""
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def train_epoch_mlp(model, loader, optimizer, scaler, grad_acc=1, device=DEVICE):
    """Train one epoch of the MLP head."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    amp_ctx = _get_amp_context(device)

    for i, (feats, targets) in enumerate(tqdm(loader, desc="train", leave=False)):
        feats = feats.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_ctx:
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(
                p_total, p_gdm, p_green, p_clover, p_dead, targets)

        loss = loss / grad_acc
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.item() * feats.size(0) * grad_acc

        if (i + 1) % grad_acc == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

    return running_loss / len(loader.dataset)


@torch.no_grad()
def valid_epoch_mlp(model, loader, device=DEVICE):
    """Validate one epoch of the MLP head."""
    model.eval()
    running_loss = 0.0
    preds_list = []
    labels_list = []
    amp_ctx = _get_amp_context(device)

    for feats, targets in tqdm(loader, desc="valid", leave=False):
        feats = feats.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_ctx:
            p_total, p_gdm, p_green, p_clover, p_dead = model(feats)
            loss = weighted_biomass_loss(
                p_total, p_gdm, p_green, p_clover, p_dead, targets)

        running_loss += loss.item() * feats.size(0)

        preds = torch.stack([
            p_green.squeeze(-1), p_dead.squeeze(-1),
            p_clover.squeeze(-1), p_gdm.squeeze(-1), p_total.squeeze(-1)
        ], dim=1)
        preds_list.append(preds.cpu().float().numpy())
        labels_list.append(targets.cpu().float().numpy())

    pred_all = np.concatenate(preds_list)
    true_all = np.concatenate(labels_list)

    weighted_r2 = global_weighted_r2_score(true_all, pred_all)
    per_target = per_target_r2_score(true_all, pred_all)

    return running_loss / len(loader.dataset), weighted_r2, per_target, pred_all, true_all
