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

def calculate_deltas(labels):
    """
    Calculates robust deltas for each target column using MAD.
    Args:
        labels: Tensor of shape [N, 5] (Green, Dead, Clover, GDM, Total)
    Returns:
        List of 5 delta values.
    """
    deltas = []
    
    # 5 targets: Green, Dead, Clover, GDM, Total
    # Corresponding indices: 0, 1, 2, 3, 4
    target_names = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    
    print(f"{'Target':<12} | {'MAD':<8} | {'Proposed Delta':<14} | {'Strategy'}")
    print("-" * 55)

    for i in range(5):
        target_data = labels[:, i]
        
        # 1. Calculate Median
        median_val = torch.median(target_data)
        
        # 2. Calculate Absolute Deviations
        abs_dev = torch.abs(target_data - median_val)
        
        # 3. Calculate MAD (Median of Deviations)
        mad = torch.median(abs_dev)
        
        # 4. Calculate Robust Sigma (Approximate Standard Deviation)
        # 1.4826 is the scaling factor for normal distributions
        sigma_robust = 1.4826 * mad
        
        # 5. Determine Delta Strategy based on your CV/Stability
        # For 'Total' (Index 4) and 'GDM' (Index 3), we trust the data (MSE preference).
        # For 'Clover' (Index 2), we distrust outliers (MAE preference).
        
        if i in [3, 4]: # GDM, Total (Stable)
             # Relax the delta to allow more MSE behavior
            final_delta = sigma_robust * 3.0 # Covers 99% of normal data
            strategy = "MSE-ish"
        elif i == 2:    # Clover (Extreme Variance)
            # Tighten delta to clamp down on outliers earlier
            final_delta = sigma_robust * 1.0 # Only trust the core 50-60%
            strategy = "Robust"
        else:           # Green, Dead (Average)
            final_delta = sigma_robust * 2.0
            strategy = "Balanced"
            
        deltas.append(final_delta.item())
        print(f"{target_names[i]:<12} | {mad.item():.2f}     | {final_delta.item():.2f}{'':<10} | {strategy}")

    return deltas

class AdaptiveHuberLoss(nn.Module):
    def __init__(self, deltas):
        super().__init__()
        # Register deltas as a buffer so it moves to GPU automatically, 
        # but isn't a trainable parameter.
        self.register_buffer('deltas', torch.tensor(deltas))

    def forward(self, pred, target, index):
        """
        Calculates Huber loss with a specific delta for this target index.
        """
        delta = self.deltas[index]
        
        # Standard Huber Logic
        error = pred - target
        abs_error = torch.abs(error)
        
        # Quadratic part (MSE behavior)
        quadratic = torch.minimum(abs_error, delta)
        loss_quad = 0.5 * quadratic ** 2
        
        # Linear part (MAE behavior)
        linear = delta * (abs_error - 0.5 * delta)
        loss_lin = torch.where(abs_error <= delta, torch.tensor(0.0, device=pred.device), linear)
        
        return torch.mean(loss_quad + loss_lin)

def weighted_biomass_loss(p_total, p_gdm, p_green, labels, deltas):
    """
    Calculates the 5 individual MSE losses and returns their
    weighted sum, perfectly aligning with the R2 metric weights.
    """
    # loss_fn = nn.HuberLoss(delta=15.0) if use_huber else nn.MSELoss()
    loss_fn = AdaptiveHuberLoss(deltas)
    # loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
    
    # Calculate derived predictions
    p_clover = torch.clamp(p_gdm - p_green, min=0)
    p_dead   = torch.clamp(p_total - p_gdm, min=0)

    # 1. Calculate the 5 individual MSE losses
    loss_green = loss_fn(p_green.squeeze(), labels[:, 0],0) # Corresponds to Dry_Green_g
    loss_dead  = loss_fn(p_dead.squeeze(),  labels[:, 1],1) # Corresponds to Dry_Dead_g
    loss_clover = loss_fn(p_clover.squeeze(), labels[:, 2],2) # Corresponds to Dry_Clover_g
    loss_gdm   = loss_fn(p_gdm.squeeze(),   labels[:, 3],3) # Corresponds to GDM_g
    loss_total = loss_fn(p_total.squeeze(), labels[:, 4],4) # Corresponds to Dry_Total_g
    
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

def weighted_biomass_log_loss(p_total_log, p_gdm_log, p_green_log, labels_log, use_huber=False):
    """
    Log-Space Loss with 'Magnitude Weighting' to prevent overfitting on small plants.
    """
    # 1. Use reduction='none' so we get a vector of losses [Batch_Size], not a single number.
    loss_fn_log = nn.MSELoss(reduction='none') 
    
    # 2. Extract targets (Log Space)
    t_total_log = labels_log[:, 4]
    t_gdm_log   = labels_log[:, 3]
    t_green_log = labels_log[:, 0]

    # 3. Calculate Direct Losses (Element-wise)
    # Shape: [Batch_Size]
    raw_loss_total = loss_fn_log(p_total_log.squeeze(), t_total_log)
    raw_loss_gdm   = loss_fn_log(p_gdm_log.squeeze(),   t_gdm_log)
    raw_loss_green = loss_fn_log(p_green_log.squeeze(), t_green_log)

    # 4. CREATE WEIGHTS (The Key Fix)
    # We weight the loss by the physical size of the plant.
    # Convert log-target back to linear scale to get the "Mass Importance".
    # +1.0 ensures we don't multiply by zero for empty pots.
    w_total = torch.expm1(t_total_log) + 1.0
    w_gdm   = torch.expm1(t_gdm_log) + 1.0
    w_green = torch.expm1(t_green_log) + 1.0
    
    # Normalize weights so they average to 1.0 (keeps learning rate stable)
    w_total = w_total / w_total.mean()
    w_gdm   = w_gdm   / w_gdm.mean()
    w_green = w_green / w_green.mean()

    # Apply Weights
    loss_total = (raw_loss_total * w_total).mean()
    loss_gdm   = (raw_loss_gdm   * w_gdm).mean()
    loss_green = (raw_loss_green * w_green).mean()

    # --- Derived Targets (Clover/Dead) ---
    # These are calculated in Linear Space (L1), so they naturally care more 
    # about big errors. We generally don't need to re-weight these as aggressively.
    
    # Un-log predictions & targets
    p_total_real = torch.expm1(p_total_log.squeeze())
    p_gdm_real   = torch.expm1(p_gdm_log.squeeze())
    p_green_real = torch.expm1(p_green_log.squeeze())
    
    t_total_real = torch.expm1(t_total_log)
    t_gdm_real   = torch.expm1(t_gdm_log)
    t_green_real = torch.expm1(t_green_log)
    
    # Derived components
    p_clover_real = torch.clamp(p_gdm_real - p_green_real, min=0)
    p_dead_real   = torch.clamp(p_total_real - p_gdm_real, min=0)
    t_clover_real = torch.clamp(t_gdm_real - t_green_real, min=0)
    t_dead_real   = torch.clamp(t_total_real - t_gdm_real, min=0)

    # Use L1 Loss (Absolute Error) for these derived parts
    loss_fn_linear = nn.L1Loss()
    loss_clover = loss_fn_linear(p_clover_real, t_clover_real)
    loss_dead   = loss_fn_linear(p_dead_real,   t_dead_real)

    # 5. Final Combination
    weights = CFG.R2_WEIGHTS
    
    weighted_loss_sum = (
        loss_green  * weights[0] +
        loss_dead   * weights[1] +
        loss_clover * weights[2] +
        loss_gdm    * weights[3] +
        loss_total  * weights[4]
    )
    
    return weighted_loss_sum