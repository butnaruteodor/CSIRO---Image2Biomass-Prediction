from torch.utils.data import Dataset
from configs.cfg import CFG
import os, cv2, numpy as np
import torch
from PIL import Image
import random
import pandas as pd
from utils.augs import *

class BiomassDatasetBase(Dataset):
    def __init__(self, df, transform, photometric_transform, img_dir, multiplier=1, val_transform=None):
        self.df        = df
        self.transform = transform
        self.ph_transform = photometric_transform
        self.val_transform = val_transform or get_val_transforms()
        self.img_dir   = img_dir
        self.paths     = df['image_path'].values
        self.labels    = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
        self.multiplier= multiplier

    def __len__(self): return len(self.df) * self.multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.df)
        is_train_copy = (idx // len(self.df)) > 0 or self.multiplier == 1
        # When multiplier=1: always val transforms
        # When multiplier>1: first pass though=val, rest=train
        path = os.path.join(self.img_dir, os.path.basename(self.paths[real_idx]))
        img  = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        mid = w // 2
        left  = img[:, :mid]
        right = img[:, mid:]
        
        if self.multiplier > 1 and idx // len(self.df) > 0:
            # Training copy: apply spatial then photometric
            if self.transform:
                left = self.transform(image=left)['image']
                right = self.transform(image=right)['image']
            left  = self.ph_transform(image=left)['image']
            right = self.ph_transform(image=right)['image']
        else:
            # Validation / first copy: val transforms only
            left = self.val_transform(image=left)['image']
            right = self.val_transform(image=right)['image']
            
        label = torch.from_numpy(self.labels[real_idx])
        return left, right, label
    
def fast_slice_resize_image(image, tile_size, target_size, mean, std):
    """
    1. Vectorized Slice (Numpy)
    2. Batch Resize (OpenCV is fast)
    3. Normalize (Torch)
    """
    h, w, c = image.shape
    
    # --- A. Pad (Vectorized) ---
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        
    # --- B. Slice (Vectorized Reshape) ---
    n_rows = image.shape[0] // tile_size
    n_cols = image.shape[1] // tile_size
    
    # Reshape to (Num_Tiles, tile_size, tile_size, 3)
    tiles = image.reshape(n_rows, tile_size, n_cols, tile_size, c)
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, c)
    
    # --- C. Batch Resize (The crucial memory fix) ---
    # We resize BEFORE converting to float to save memory, 
    # and we resize before stacking to keep the tensor small.
    
    resized_tiles = []
    # Loop is unavoidable for resize, but OpenCV is extremely fast (C++ backend)
    for i in range(tiles.shape[0]):
        # Resize 512x512 -> 256x256
        resized = cv2.resize(tiles[i], (target_size, target_size), interpolation=cv2.INTER_AREA)
        resized_tiles.append(resized)
    
    # Stack into one numpy array: (Num_Tiles, 256, 256, 3)
    batch_np = np.stack(resized_tiles)

    # --- D. Normalize (Vectorized Torch) ---
    # Convert to Tensor: (Num_Tiles, 3, 256, 256)
    batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    
    # Float conversion + CLIP Normalization
    batch_tensor = batch_tensor.float().div(255.0)
    batch_tensor = (batch_tensor - mean) / std
    
    return batch_tensor

class BiomassDatasetClip(Dataset):
    def __init__(self, df, transform, photometric_transform, img_dir, preprocess, tile_size=512, is_train=True):
        self.preprocess = preprocess
        self.img_dir   = img_dir
        self.df        = df
        self.transform = transform
        self.paths     = df['image_path'].values
        self.tile_size = tile_size
        self.labels    = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
        self.model_input_size = 256
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        
        self.texts = [self._generate_text_description(row,training=is_train) for _, row in df.iterrows()]

    def _generate_text_description(self, row, p=0.2, training=True):
        """
        Converts a dataframe row into a descriptive sentence.
        
        Args:
            p (float): Probability of DROPPING a specific measurement (0.0 to 1.0).
            training (bool): If False, all data is included (validation mode).
        """
        # 1. The Core Sentence (The "Anchor")
        # We rarely drop the Species, but we can optionally drop the State location
        intro = f"A photo of {row['Species']} vegetation"
        # 2. The Measurements List
        # We define them as a list of independent strings
        measurements = [
            f"Height {row['Height_Ave_cm']:.1f}cm",
            f"Green Mass {row['Dry_Green_g']:.1f}g",
            f"Dead Mass {row['Dry_Dead_g']:.1f}g",
            f"Clover {row['Dry_Clover_g']:.1f}g",
            f"NDVI {row['Pre_GSHH_NDVI']:.2f}",
            f"Total Dry Mass {row['Dry_Total_g']:.1f}g",
            f"GDM {row['GDM_g']:.1f}g"
        ]

        # 3. Apply Augmentation (Dropout & Shuffle)
        if training:
            # A. Filter: Keep items where random value > p
            kept_measurements = [m for m in measurements if random.random() > p]
            
            # B. Shuffle: Randomize the order so position doesn't matter
            # random.shuffle(kept_measurements)
        else:
            # Validation: Keep all, maintain fixed order
            kept_measurements = measurements

        # 4. Final Assembly
        # Only add the "Measurements:" prefix if we actually have data left
        if kept_measurements:
            measurements_str = ", ".join(kept_measurements)
            full_text = f"{intro} Measurements: {measurements_str}."
        else:
            full_text = intro

        return full_text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Albumentations
        if self.transform:
            img = self.transform(image=img)['image']

        pixel_values = fast_slice_resize_image(
            img, 
            tile_size=self.tile_size, 
            target_size=self.model_input_size,
            mean=self.mean,
            std=self.std
        )
        return {
            "pixel_values": pixel_values, # Shape: [Num_Tiles, 3, 512, 512]
            "text": self.texts[idx],
            "index": idx
        }

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, neighbor_map):
        self.features = features
        self.targets = targets
        self.neighbor_map = neighbor_map

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 1. Get Main Sample
        feat = self.features[idx]
        target = self.targets[idx]
        
        # 2. Get Its Pre-Calculated Neighbor
        n_idx = self.neighbor_map[idx]
        n_feat = self.features[n_idx]
        n_target = self.targets[n_idx]
        
        return feat, target, n_feat, n_target
    
def get_full_ridge_dataset(splitter, df_wide, fold_root_dir):
    """
    Constructs X_all (Mean Embeddings) and y_all for Ridge Regression.
    """
    N_total = len(df_wide)
    
    # Initialize placeholders
    # We will detect embedding dimension dynamically from the first file loaded
    X_all = None 
    y_all = np.zeros((N_total, 5))
    
    print(f"Building Ridge dataset from: {fold_root_dir}")

    # Convert generator to list so we can index neighbor folds
    splits = list(splitter)
    
    for current_fold_idx in range(len(splits)):
        # 1. Identify which images belong to this fold's validation set
        _, target_img_indices = splits[current_fold_idx]
        
        # 2. Find where these images are used as TRAINING data (the next fold)
        # Fold 1's validation images are inside Fold 2's training file
        neighbor_fold_idx = (current_fold_idx + 1) % len(splits)
        neighbor_train_indices, _ = splits[neighbor_fold_idx]
        
        # 3. Load that neighbor's training file (contains the augmentations we need)
        neighbor_path = os.path.join(fold_root_dir, f"fold_{neighbor_fold_idx + 1}", "train.pt")
        data = torch.load(neighbor_path, map_location='cpu')
        
        # --- FIRST RUN SETUP ---
        if X_all is None:
            # Detect embedding dimension (e.g., 512, 768, 1024)
            # data['features'] is (N_train * 20, Dim)
            emb_dim = data['features'].shape[1]
            X_all = np.zeros((N_total, emb_dim), dtype=np.float32)
            print(f"  > Detected Embedding Dimension: {emb_dim}")

        # 4. Filter: We need to grab only the rows corresponding to 'target_img_indices'
        # Reshape to (N_train, 20, Dim)
        n_train_imgs = len(neighbor_train_indices)
        features_reshaped = data['features'].view(n_train_imgs, 20, -1)
        targets_reshaped  = data['targets'].view(n_train_imgs, 20, 5)
        
        # Find position of our target images inside the neighbor file
        # This matches the Real Index (from df) to the Tensor Index
        mask = np.isin(neighbor_train_indices, target_img_indices)
        positions = np.where(mask)[0]
        
        # 5. Compute Mean & Fill X_all
        # Select the specific images -> (N_subset, 20, Dim)
        subset_feats = features_reshaped[positions]
        subset_targets = targets_reshaped[positions]
        
        # Calculate Mean Embedding
        # mean_feats = subset_feats.mean(dim=1).numpy()
        single_view_feats = subset_feats[:, 0, :].numpy()

        # # Targets are identical across augmentations, take 0th
        clean_targets = subset_targets[:, 0, :].numpy()
        
        
        # Map back to global X_all using the REAL dataframe indices
        # neighbor_train_indices[positions] gives the actual image IDs (e.g., 5, 120, 300...)
        real_indices = neighbor_train_indices[positions]
        
        X_all[real_indices] = single_view_feats
        y_all[real_indices] = clean_targets
        
        print(f"  > Fold {current_fold_idx+1}: Extracted {len(real_indices)} images from Fold {neighbor_fold_idx+1}")

    return X_all, y_all