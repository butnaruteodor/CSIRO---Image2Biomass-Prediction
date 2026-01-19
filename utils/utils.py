import torch
import open_clip
from peft import PeftModel
from collections import OrderedDict
from configs.cfg import CFG
import numpy as np

def compare_structure(path_a, path_b):
    print(f"--- Structure Comparison ---")
    print(f"File A: {path_a}")
    print(f"File B: {path_b}")
    print("-" * 30)

    # 1. Load Files
    try:
        raw_a = torch.load(path_a, map_location='cpu')
        raw_b = torch.load(path_b, map_location='cpu')
        
        # Unwrap state_dicts if nested
        state_a = raw_a.get('state_dict', raw_a).get('model', raw_a) if isinstance(raw_a, dict) else raw_a
        state_b = raw_b.get('state_dict', raw_b).get('model', raw_b) if isinstance(raw_b, dict) else raw_b
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Get Keys
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    
    # 3. Analyze Differences
    only_in_a = sorted(list(keys_a - keys_b))
    only_in_b = sorted(list(keys_b - keys_a))
    common = keys_a & keys_b

    print(f"Total Keys in A: {len(keys_a)}")
    print(f"Total Keys in B: {len(keys_b)}")
    print(f"Common Keys:     {len(common)}")
    print("-" * 30)

    if not only_in_a and not only_in_b:
        print("✅ STRUCTURE MATCH: Both files have the exact same layer names.")
    else:
        print("❌ STRUCTURE MISMATCH DETECTED\n")

    if only_in_a:
        print(f"Keys ONLY in File A (Method 1) [{len(only_in_a)}]:")
        for k in only_in_a[:10]: print(f"  + {k}")
        if len(only_in_a) > 10: print("  ... and more")
        print("")

    if only_in_b:
        print(f"Keys ONLY in File B (Method 2) [{len(only_in_b)}]:")
        for k in only_in_b[:10]: print(f"  + {k}")
        if len(only_in_b) > 10: print("  ... and more")
        print("")

# ==========================================
# USAGE
# ==========================================
# FILE_A = "out/pretrained_backbone.pth" 
# FILE_B = "adapters/second/lora_finetuned_convnext_base_second.pt" 

# compare_structure(FILE_A, FILE_B)

def get_clean_timm_state_dict(model):
    """
    Takes a trained OpenCLIP + LoRA model, merges weights, 
    cleans keys, and returns a timm-compatible state_dict.
    Does NOT save to disk.
    """
    # print("Merging LoRA weights in memory...")
    
    # 1. Merge LoRA weights back into the base visual model
    merged_visual_model = model.visual.merge_and_unload()
    
    # 2. Get the raw state dict
    raw_state_dict = merged_visual_model.state_dict()
    
    # print("Cleaning state dict keys...")
    clean_state_dict = OrderedDict()

    for key, value in raw_state_dict.items():
        # 1. Standardize keys (OpenCLIP -> timm)
        new_key = key.replace("trunk.", "")
        new_key = new_key.replace("visual.", "")
        new_key = new_key.replace("module.", "")
        
        # 2. Remove CLIP-specific Projection Layer
        # This is the "Structure Mismatch" fix
        if "head.proj" in new_key:
            continue  
            
        # 3. Handle Output Head (Optional but recommended)
        
        # 4. Move to CPU to save VRAM and decouple from GPU
        clean_state_dict[new_key] = value.cpu()

    # print(f"Conversion complete! {len(clean_state_dict)} keys ready for timm.")
    return clean_state_dict

def calculate_biomass_priors(labels):
    """
    Calculates the inverse-sigmoid bias values for the two ratio heads.
    
    Args:
        labels: Tensor [N, 5] corresponding to:
                [Green, Dead, Clover, GDM, Total]
    Returns:
        dict: {'gdm_bias': float, 'green_bias': float}
    """
    # Summing prevents division by zero on small plants and gives a weighted average
    total_mass = labels[:, 4].sum()
    gdm_mass   = labels[:, 3].sum()
    green_mass = labels[:, 0].sum()
    
    # 1. Ratio GDM = GDM / Total
    # Add epsilon 1e-6 to avoid numerical errors
    avg_gdm_ratio = (gdm_mass / (total_mass + 1e-6)).item()
    
    # 2. Ratio Green = Green / GDM
    avg_green_ratio = (green_mass / (gdm_mass + 1e-6)).item()
    
    # 3. Inverse Sigmoid Calculation: b = ln(p / (1-p))
    # We clip the ratio to [0.01, 0.99] to prevent math errors if data is skewed
    avg_gdm_ratio = np.clip(avg_gdm_ratio, 0.01, 0.99)
    avg_green_ratio = np.clip(avg_green_ratio, 0.01, 0.99)
    
    bias_gdm = np.log(avg_gdm_ratio / (1 - avg_gdm_ratio))
    bias_green = np.log(avg_green_ratio / (1 - avg_green_ratio))
    
    print(f"Calculated Priors -> GDM Ratio: {avg_gdm_ratio:.2f} (Bias: {bias_gdm:.2f})")
    print(f"Calculated Priors -> Green Ratio: {avg_green_ratio:.2f} (Bias: {bias_green:.2f})")
    
    return {'gdm_bias': bias_gdm, 'green_bias': bias_green}

def init_ratio_biases(model, priors):
    """
    Initializes the biases of the two ratio heads in the model.
    """
    with torch.no_grad():
        # --- Head 1: Ratio GDM ---
        # We assume head is Sequential(..., Linear, Sigmoid)
        # We need to find the last Linear layer to set the bias
        for layer in reversed(model.head_ratio_gdm):
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(0.1)
                # Optional: reduce weight noise so bias dominates at start
                layer.weight.normal_(0, 0.01) 
                break
        
        # --- Head 2: Ratio Green ---
        for layer in reversed(model.head_ratio_green):
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(0.1)
                layer.weight.normal_(0, 0.01)
                break
                
    print("Ratio heads initialized.")