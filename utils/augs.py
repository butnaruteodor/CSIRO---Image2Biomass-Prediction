import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from configs.cfg import CFG

def get_spatial_transforms(seed=CFG.SEED, size=None,):
    """Spatial transforms applied IDENTICALLY to both image halves."""
    size = size or CFG.IMG_SIZE
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], p=1.0, seed=seed)

def get_photometric_transforms(seed=CFG.SEED, size=None):
    """Photometric transforms applied INDEPENDENTLY to each half."""
    size = size or CFG.IMG_SIZE
    return A.Compose([
        A.Resize(size, size),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0, seed=seed)

def get_val_transforms(seed=CFG.SEED, size=None):
    """Validation transforms (resize + normalize only)."""
    size = size or CFG.IMG_SIZE
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0, seed=seed)