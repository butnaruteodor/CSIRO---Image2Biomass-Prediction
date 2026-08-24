"""
Loss functions for biomass prediction.
Moved from utils/eval.py.
"""
import torch
import torch.nn as nn
from src.config import CFG


def weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, targets):
    """
    Weighted biomass loss combining MSE on direct targets (Total, GDM, Green)
    and L1 on derived targets (Clover, Dead).

    Args:
        p_total, p_gdm, p_green, p_clover, p_dead: [B, 1] predictions
        targets: [B, 5] ground truth [Green, Dead, Clover, GDM, Total]
    Returns:
        scalar loss
    """
    t_total = targets[:, 4:5]
    t_gdm = targets[:, 3:4]
    t_green = targets[:, 0:1]
    t_clover = targets[:, 2:3]
    t_dead = targets[:, 1:2]

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    loss_total = mse(p_total, t_total)
    loss_gdm = mse(p_gdm, t_gdm)
    loss_green = mse(p_green, t_green)
    loss_clover = l1(p_clover, t_clover)
    loss_dead = l1(p_dead, t_dead)

    weights = CFG.R2_WEIGHTS_TRAIN
    weighted_loss = (
        loss_green * weights[0] +
        loss_dead * weights[1] +
        loss_clover * weights[2] +
        loss_gdm * weights[3] +
        loss_total * weights[4]
    )
    return weighted_loss


def weighted_biomass_log_loss(p_total_log, p_gdm_log, p_green_log, labels_log, use_huber=False):
    """Log-Space loss with magnitude weighting."""
    loss_fn_log = nn.MSELoss(reduction='none')

    t_total_log = labels_log[:, 4]
    t_gdm_log = labels_log[:, 3]
    t_green_log = labels_log[:, 0]

    raw_loss_total = loss_fn_log(p_total_log.squeeze(), t_total_log)
    raw_loss_gdm = loss_fn_log(p_gdm_log.squeeze(), t_gdm_log)
    raw_loss_green = loss_fn_log(p_green_log.squeeze(), t_green_log)

    w_total = torch.expm1(t_total_log) + 1.0
    w_gdm = torch.expm1(t_gdm_log) + 1.0
    w_green = torch.expm1(t_green_log) + 1.0

    w_total = w_total / w_total.mean()
    w_gdm = w_gdm / w_gdm.mean()
    w_green = w_green / w_green.mean()

    loss_total = (raw_loss_total * w_total).mean()
    loss_gdm = (raw_loss_gdm * w_gdm).mean()
    loss_green = (raw_loss_green * w_green).mean()

    p_total_real = torch.expm1(p_total_log.squeeze())
    p_gdm_real = torch.expm1(p_gdm_log.squeeze())
    p_green_real = torch.expm1(p_green_log.squeeze())
    t_total_real = torch.expm1(t_total_log)
    t_gdm_real = torch.expm1(t_gdm_log)
    t_green_real = torch.expm1(t_green_log)

    p_clover_real = torch.clamp(p_gdm_real - p_green_real, min=0)
    p_dead_real = torch.clamp(p_total_real - p_gdm_real, min=0)
    t_clover_real = torch.clamp(t_gdm_real - t_green_real, min=0)
    t_dead_real = torch.clamp(t_total_real - t_gdm_real, min=0)

    loss_fn_linear = nn.L1Loss()
    loss_clover = loss_fn_linear(p_clover_real, t_clover_real)
    loss_dead = loss_fn_linear(p_dead_real, t_dead_real)

    weights = CFG.R2_WEIGHTS_TRAIN
    weighted_loss_sum = (
        loss_green * weights[0] +
        loss_dead * weights[1] +
        loss_clover * weights[2] +
        loss_gdm * weights[3] +
        loss_total * weights[4]
    )
    return weighted_loss_sum


@torch.no_grad()
def calculate_deltas(labels):
    """Calculate robust deltas for each target using MAD."""
    deltas = []
    target_names = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    print(f"{'Target':<12} | {'MAD':<8} | {'Proposed Delta':<14} | {'Strategy'}")
    print("-" * 55)

    for i in range(5):
        target_data = labels[:, i]
        median_val = torch.median(target_data)
        abs_dev = torch.abs(target_data - median_val)
        mad = torch.median(abs_dev)
        sigma_robust = 1.4826 * mad

        if i in [3, 4]:  # GDM, Total
            final_delta = sigma_robust * 3.0
            strategy = "MSE-ish"
        elif i == 2:  # Clover
            final_delta = sigma_robust * 1.0
            strategy = "MAE-ish"
        else:  # Green, Dead
            final_delta = sigma_robust * 2.0
            strategy = "Huber-ish"

        print(f"{target_names[i]:<12} | {mad:<8.3f} | {final_delta:<14.3f} | {strategy}")
        deltas.append(final_delta.item())

    print("-" * 55)
    return deltas
