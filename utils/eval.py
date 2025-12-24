import numpy as np
from configs.cfg import CFG
import torch.nn as nn
import torch

def global_weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes the globally weighted R² score as described in the evaluation.
    
    y_true, y_pred: shape (N, 5)
    weights: [0.1, 0.1, 0.1, 0.2, 0.5] (from CFG)
    """
    weights_matrix = np.tile(CFG.R2_WEIGHTS, (y_true.shape[0], 1))
    # y_bar_w = (sum(w_j * y_j)) / (sum(w_j))
    weighted_sum = np.sum(weights_matrix * y_true)
    total_weight = np.sum(weights_matrix)
    y_bar_w = weighted_sum / total_weight # This is a single scalar value
    # SS_res = sum(w_j * (y_j - y_pred_j)^2)
    ss_res = np.sum(weights_matrix * (y_true - y_pred) ** 2)
    # SS_tot = sum(w_j * (y_j - y_bar_w)^2)
    ss_tot = np.sum(weights_matrix * (y_true - y_bar_w) ** 2)
    # R²_w = 1 - (SS_res / SS_tot)
    r2_w = 1 - (ss_res / ss_tot)
    return r2_w

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: shape (N, 5)
    weights: [0.1, 0.1, 0.1, 0.2, 0.5]
    """
    weights = CFG.R2_WEIGHTS
    r2_scores = []
    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2

def weighted_biomass_loss(p_total, p_gdm, p_green, labels, use_huber=False):
    """
    Calculates the 5 individual MSE losses and returns their
    weighted sum, perfectly aligning with the R2 metric weights.
    """
    loss_fn = nn.HuberLoss(delta=15.0) if use_huber else nn.MSELoss()
    
    # 1. Calculate the 5 individual MSE losses
    loss_total = loss_fn(p_total.squeeze(), labels[:, 4]) # Corresponds to Dry_Total_g
    loss_gdm   = loss_fn(p_gdm.squeeze(),   labels[:, 3]) # Corresponds to GDM_g
    loss_green = loss_fn(p_green.squeeze(), labels[:, 0]) # Corresponds to Dry_Green_g

    # Calculate derived predictions
    p_clover = torch.clamp(p_gdm - p_green, min=0)
    p_dead   = torch.clamp(p_total - p_gdm, min=0)

    loss_clover = loss_fn(p_clover.squeeze(), labels[:, 2]) # Corresponds to Dry_Clover_g
    loss_dead   = loss_fn(p_dead.squeeze(),   labels[:, 1]) # Corresponds to Dry_Dead_g

    # 2. Get the weights
    weights = CFG.R2_WEIGHTS
    
    # 3. Apply the weights to their corresponding losses
    weighted_loss_sum = (
        loss_green  * weights[0] +
        loss_dead   * weights[1] +
        loss_clover * weights[2] +
        loss_gdm    * weights[3] +
        loss_total  * weights[4]
    )
    
    return weighted_loss_sum

def global_clip_loss(image_embeddings, all_text_anchors, global_indices, logit_scale):
    """
    Calculates CLIP loss against the entire dataset of text anchors.
    
    Args:
        image_embeddings: [Batch_Size, Dim] - The projected image features from the model.
        all_text_anchors: [Total_Dataset_Size, Dim] - Pre-computed embeddings for ALL texts.
        global_indices:   [Batch_Size] - The absolute index (0 to N) of each image in the dataset.
        logit_scale:      Scalar - The learnable temperature parameter.
    """
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    
    logits = (image_embeddings @ all_text_anchors.T) * logit_scale.exp()
    
    loss = nn.CrossEntropyLoss()(logits, global_indices)
    
    return loss