from torch.utils.data import Dataset
from configs.cfg import CFG
import os, cv2, numpy as np
import torch
from PIL import Image
import random

class BiomassDatasetBase(Dataset):
    def __init__(self, df, transform, photometric_transform, img_dir):
        self.df        = df
        self.transform = transform
        self.ph_transform = photometric_transform
        self.img_dir   = img_dir
        self.paths     = df['image_path'].values
        self.labels    = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
        # train_idx = [0, 1, 2, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 27, 29, 
        #            30, 31, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53, 54, 55, 
        #            58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
        #            79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 
        #            102, 103, 105, 107, 108, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 
        #            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 
        #            141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 153, 156, 157, 160, 161, 162, 164, 
        #            165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 
        #            184, 186, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 
        #            204, 205, 206, 207, 210, 211, 212, 213, 214, 216, 217, 218, 219, 221, 222, 224, 225, 
        #            227, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245, 247, 
        #            248, 249, 251, 252, 254, 255, 259, 260, 261, 262, 264, 265, 267, 269, 270, 272, 273, 
        #            274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 286, 287, 288, 289, 290, 292, 293, 
        #            294, 295, 296, 297, 299, 300, 301, 303, 304, 306, 308, 309, 310, 313, 315, 316, 317, 
        #            319, 320, 323, 324, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 338, 339, 340, 
        #            341, 342, 343, 345, 346, 347, 348, 349, 351, 352, 353, 354, 355, 356]
        # val_idx = [3, 4, 6, 8, 19, 22, 25, 28, 32, 40, 41, 46, 49, 56, 57, 85, 95, 104, 106, 109, 112, 
        #         123, 145, 152, 154, 155, 158, 159, 163, 168, 180, 185, 187, 190, 208, 209, 215, 220, 
        #         223, 226, 228, 233, 242, 246, 250, 253, 256, 257, 258, 263, 266, 268, 271, 284, 285, 
        #         291, 298, 302, 305, 307, 311, 312, 314, 318, 321, 322, 325, 336, 337, 344, 350]
        # if len(self.paths)>100:
        #     lst = [self.paths[idx] for idx in range(len(train_idx))]
        # else:
        #     lst = [self.paths[idx] for idx in range(len(val_idx))]
        # print(lst)
        # if len(self.paths)>100:
        #     print("Train",{train_idx[idx]:self.paths[idx] for idx in range(len(train_idx))})
        # else:
        #     print("Val",{val_idx[idx]:self.paths[idx] for idx in range(len(val_idx))})

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img  = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        mid = w // 2
        left  = img[:, :mid]
        right = img[:, mid:]
        if self.transform:
            transformed = self.transform(image=left, image_right=right)
            left  = transformed['image']
            right = transformed['image_right']

        # 2. Apply PHOTOMETRIC transforms (independently)
        left  = self.ph_transform(image=left)['image']
        right = self.ph_transform(image=right)['image']

        label = torch.from_numpy(self.labels[idx])
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