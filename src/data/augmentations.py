"""
Augmentation / transform helpers.
Moved from utils/augs.py + infer.ipynb TTA logic.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# ============================================================================
# Training transforms
# ============================================================================

def get_spatial_transforms(size=1008, seed=42):
    """Spatial transforms applied independently to both image halves."""
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], p=1.0, seed=seed)


def get_photometric_transforms(size=1008, seed=42):
    """Photometric transforms applied independently to each half."""
    return A.Compose([
        A.Resize(size, size),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0, seed=seed)


def get_val_transforms(size=1008, seed=42):
    """Validation transforms (resize + normalize only)."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0, seed=seed)


# ============================================================================
# TTA transforms for inference
# ============================================================================

def get_tta_transforms(img_size=1008):
    """
    Returns 5 TTA transforms: original, HFlip, VFlip.
    All include resize + normalize (no augmentations).
    """
    base = [
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    transforms = [
        A.Compose([A.Resize(img_size, img_size)] + base),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.Resize(img_size, img_size), A.Transpose(p=1.0)] + base),
    ]
    return transforms


def get_val_transform_for_inference(img_size=1008):
    """Simple val transform for inference (no TTA)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])