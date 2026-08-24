"""
Utility / helper functions.
Moved from utils/utils.py.
"""
import torch
from collections import OrderedDict
import numpy as np


def compare_structure(path_a, path_b):
    """Compare two model checkpoint structures."""
    print(f"--- Structure Comparison ---")
    print(f"File A: {path_a}")
    print(f"File B: {path_b}")
    print("-" * 30)

    try:
        raw_a = torch.load(path_a, map_location='cpu')
        raw_b = torch.load(path_b, map_location='cpu')

        state_a = raw_a.get('state_dict', raw_a).get('model', raw_a) if isinstance(raw_a, dict) else raw_a
        state_b = raw_b.get('state_dict', raw_b).get('model', raw_b) if isinstance(raw_b, dict) else raw_b
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())

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
        print(f"Keys ONLY in File A [{len(only_in_a)}]:")
        for k in only_in_a[:10]:
            print(f"  + {k}")
        if len(only_in_a) > 10:
            print("  ... and more")
        print("")

    if only_in_b:
        print(f"Keys ONLY in File B [{len(only_in_b)}]:")
        for k in only_in_b[:10]:
            print(f"  + {k}")
        if len(only_in_b) > 10:
            print("  ... and more")
        print("")


def get_clean_timm_state_dict(model):
    """
    Takes a trained OpenCLIP + LoRA model, merges weights,
    cleans keys, and returns a timm-compatible state_dict.
    """
    merged_visual_model = model.visual.merge_and_unload()
    raw_state_dict = merged_visual_model.state_dict()
    clean_state_dict = OrderedDict()

    for key, value in raw_state_dict.items():
        new_key = key.replace("trunk.", "")
        new_key = new_key.replace("visual.", "")
        new_key = new_key.replace("module.", "")

        if "head.proj" in new_key:
            continue

        clean_state_dict[new_key] = value.cpu()

    return clean_state_dict


def calculate_biomass_priors(labels):
    """Calculate biomass priors (GDM ratio, Green ratio) from training labels."""
    total_mass = labels[:, 4].sum()
    gdm_mass = labels[:, 3].sum()
    green_mass = labels[:, 0].sum()

    avg_gdm_ratio = (gdm_mass / (total_mass + 1e-6)).item()
    avg_green_ratio = (green_mass / (gdm_mass + 1e-6)).item()

    avg_gdm_ratio = np.clip(avg_gdm_ratio, 0.01, 0.99)
    avg_green_ratio = np.clip(avg_green_ratio, 0.01, 0.99)

    bias_gdm = np.log(avg_gdm_ratio / (1 - avg_gdm_ratio))
    bias_green = np.log(avg_green_ratio / (1 - avg_green_ratio))

    print(f"Calculated Priors -> GDM Ratio: {avg_gdm_ratio:.2f} (Bias: {bias_gdm:.2f})")
    print(f"Calculated Priors -> Green Ratio: {avg_green_ratio:.2f} (Bias: {bias_green:.2f})")

    return {'gdm_bias': bias_gdm, 'green_bias': bias_green}


def init_ratio_biases(model, priors):
    """Initialize ratio head biases."""
    with torch.no_grad():
        for layer in reversed(model.head_ratio_gdm):
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(0.1)
                layer.weight.normal_(0, 0.01)
                break

        for layer in reversed(model.head_ratio_green):
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(0.1)
                layer.weight.normal_(0, 0.01)
                break
    print("Ratio heads initialized.")


def slerp(val, low, high):
    """Spherical Linear Interpolation."""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)

    dot = (low_norm * high_norm).sum(1).clamp(-1, 1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    mask = sin_omega > 1e-6

    scale0 = torch.zeros_like(omega)
    scale1 = torch.zeros_like(omega)

    scale0[mask] = torch.sin((1.0 - val) * omega[mask]) / sin_omega[mask]
    scale1[mask] = torch.sin(val * omega[mask]) / sin_omega[mask]

    scale0[~mask] = 1.0 - val
    scale1[~mask] = val

    scale0 = scale0.unsqueeze(1)
    scale1 = scale1.unsqueeze(1)

    res = scale0 * low + scale1 * high
    return res