"""
Dataset classes for biomass prediction.
Moved from dataset/biomass_dataset.py + infer.ipynb.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.augmentations import get_val_transforms


# ============================================================================
# Training Dataset (two-stream image input)
# ============================================================================
class BiomassDatasetBase(Dataset):
    """Base dataset for training with two-stream image input."""

    def __init__(self, df, transform, photometric_transform, img_dir,
                 multiplier=1, val_transform=None,
                 target_cols=None):
        self.df = df
        self.transform = transform
        self.ph_transform = photometric_transform
        self.val_transform = val_transform or get_val_transforms()
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.target_cols = target_cols or [
            'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        self.labels = df[self.target_cols].values.astype(np.float32)
        self.multiplier = multiplier

    def __len__(self):
        return len(self.df) * self.multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.df)
        path = os.path.join(self.img_dir, os.path.basename(self.paths[real_idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        if self.multiplier > 1 and idx // len(self.df) > 0:
            if self.transform:
                left = self.transform(image=left)['image']
                right = self.transform(image=right)['image']
            left = self.ph_transform(image=left)['image']
            right = self.ph_transform(image=right)['image']
        else:
            left = self.val_transform(image=left)['image']
            right = self.val_transform(image=right)['image']

        label = torch.from_numpy(self.labels[real_idx])
        return left, right, label


# ============================================================================
# Tiled extraction helpers
# ============================================================================
def fast_slice_resize_image(image, tile_size, target_size, mean, std):
    """
    1. Vectorized slice (numpy)
    2. Batch resize (OpenCV)
    3. Normalize (torch)
    """
    h, w, c = image.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)),
                       mode='constant', constant_values=0)
    n_rows = image.shape[0] // tile_size
    n_cols = image.shape[1] // tile_size
    tiles = image.reshape(n_rows, tile_size, n_cols, tile_size, c)
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, c)
    resized_tiles = []
    for i in range(tiles.shape[0]):
        resized = cv2.resize(tiles[i], (target_size, target_size),
                             interpolation=cv2.INTER_AREA)
        resized_tiles.append(resized)
    tiles = np.stack(resized_tiles)
    tiles = tiles.astype(np.float32) / 255.0
    mean_t = np.array(mean).reshape(1, 1, 1, 3)
    std_t = np.array(std).reshape(1, 1, 1, 3)
    tiles = (tiles - mean_t) / std_t
    return torch.from_numpy(tiles).permute(0, 3, 1, 2)


# ============================================================================
# Embedding-based dataset (for MLP on precomputed features)
# ============================================================================
class EmbeddingAugmentationDataset(Dataset):
    """
    Dataset that samples augmented embeddings on-the-fly.
    clean_embeddings.pt: [N, D] - val transforms
    aug_embeddings.pt:   [N, n_aug, D] - augmented versions
    """
    def __init__(self, indices, embed_dir, n_aug=20, is_train=True):
        self.indices = indices
        self.embed_dir = embed_dir
        self.n_aug = n_aug
        self.is_train = is_train
        self.targets = torch.load(os.path.join(embed_dir, 'targets.pt'))
        self.clean_embeddings = torch.load(os.path.join(embed_dir, 'clean_embeddings.pt'))
        aug_path = os.path.join(embed_dir, 'aug_embeddings.pt')
        if os.path.exists(aug_path):
            self.aug_embeddings = torch.load(aug_path)
        else:
            self.aug_embeddings = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if self.is_train and self.aug_embeddings is not None:
            aug_idx = np.random.randint(0, self.n_aug + 1)
            if aug_idx == 0:
                feat = self.clean_embeddings[real_idx]
            else:
                feat = self.aug_embeddings[real_idx, aug_idx - 1]
        else:
            feat = self.clean_embeddings[real_idx]
        target = self.targets[real_idx]
        return feat, target


# ============================================================================
# Test dataset for inference
# ============================================================================
class TestBiomassDataset(Dataset):
    """Dataset for inference on test images."""

    def __init__(self, test_df, transform, image_dir):
        self.image_paths = test_df['image_path'].values
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        full_path = os.path.join(self.image_dir, os.path.basename(img_path))
        image = cv2.imread(str(full_path))
        if image is None:
            print(f"Warning: Failed to load image: {full_path} -> black image")
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mid_point = width // 2
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]
        img_left_tensor = self.transform(image=img_left)['image']
        img_right_tensor = self.transform(image=img_right)['image']
        return img_left_tensor, img_right_tensor
