"""
Loss functions for biomass prediction.
"""
import torch
import torch.nn as nn
from src.config import CFG


def weighted_biomass_loss(p_total, p_gdm, p_green, p_clover, p_dead, targets):
    """
    Weighted mean squared error following the weighting scheme of the
    official competition metric (weights w = [0.1, 0.1, 0.1, 0.2, 0.5] for
    Green, Dead, Clover, GDM, Total).

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

    weights = CFG.R2_WEIGHTS_TRAIN
    weighted_loss = (
        mse(p_green, t_green) * weights[0] +
        mse(p_dead, t_dead) * weights[1] +
        mse(p_clover, t_clover) * weights[2] +
        mse(p_gdm, t_gdm) * weights[3] +
        mse(p_total, t_total) * weights[4]
    )
    return weighted_loss


