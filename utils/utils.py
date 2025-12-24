import torch

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