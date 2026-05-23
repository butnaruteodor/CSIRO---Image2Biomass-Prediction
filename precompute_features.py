"""
precompute_features.py
Extracts DINOv3 ViT-L/16 embeddings for all images with compact single-file storage.
Uses CFG.IMG_SIZE (1008 = 63 × 16 for patch_size=16).
Run this ONCE before experiments.

Usage:
    python precompute_features.py

Output:
    embeddings/
        clean_embeddings.pt     # [N, 2048] — val transforms
        aug_embeddings.pt       # [N, 20, 2048] — augmented versions
        targets.pt              # [N, 5]
        image_ids.csv           # image_path → index mapping
        metadata.json           # Config info
"""

import torch
import timm
from dataset.preprocess_data import get_df, extract_features_organized
from configs.deterministic import set_seed
import warnings
warnings.filterwarnings('ignore')

def main():
    set_seed(42, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    df = get_df()
    print(f"Loaded {len(df)} images")
    
    # 2. Load backbone (DINOv3 ViT-L/16)
    model_name = 'vit_large_patch16_dinov3'
    print(f"Loading backbone: {model_name}")
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    backbone = backbone.to(device)
    backbone.eval()
    print(f"Backbone feature dim: {backbone.num_features}")
    
    # 3. Extract embeddings (uses CFG.IMG_SIZE for resize)
    n_aug = 20
    batch_size = 12  # Only 1 backbone forward per half, can fit large batches
    
    save_dir = extract_features_organized(
        df=df,
        model=backbone,
        save_dir='embeddings',
        n_aug=n_aug,
        batch_size=batch_size,
        device=device
    )
    
    # Verify sizes
    clean = torch.load(f'{save_dir}/clean_embeddings.pt')
    aug = torch.load(f'{save_dir}/aug_embeddings.pt')
    total_mb = (clean.numel() + aug.numel()) * 4 / 1e6
    print(f"\nDone! {clean.shape[0]} images × {1 + aug.shape[1]} versions = {clean.shape[0] * (1 + aug.shape[1])} embeddings ({total_mb:.1f} MB total)")

if __name__ == '__main__':
    main()