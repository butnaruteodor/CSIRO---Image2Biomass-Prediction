from torch.utils.data import Dataset
from configs.cfg import CFG
import os, cv2, numpy as np
import torch
from PIL import Image

class BiomassDatasetBase(Dataset):
    def __init__(self, df, transform, photometric_transform, img_dir):
        self.df        = df
        self.transform = transform
        self.ph_transform = photometric_transform
        self.img_dir   = img_dir
        self.paths     = df['image_path'].values
        self.labels    = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)

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
    
def slice_image(image, tile_size=256):
    w, h = image.size
    tiles = []
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            box = (j, i, min(j + tile_size, w), min(i + tile_size, h))
            tile = image.crop(box)
            if tile.size != (tile_size, tile_size):
                new_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
           
                new_tile.paste(tile, (0, 0))
                tile = new_tile
            tiles.append(tile)
    return tiles

class BiomassDatasetClip(Dataset):
    def __init__(self, df, transform, photometric_transform, img_dir, preprocess, tokenizer, tile_size=512):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.img_dir   = img_dir
        self.df        = df
        self.transform = transform
        self.paths     = df['image_path'].values
        self.tile_size = tile_size
        self.labels    = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)

        self.texts = [self._generate_text_description(row) for _, row in df.iterrows()]

    def _generate_text_description(self, row):
        """
        Converts a dataframe row into a descriptive sentence.
        Adjust the template below to emphasize the features you care about most.
        """
        # Option A: Natural Language (Best for CLIP semantics)
        # We focus on the most visually distinct features: Species, State, and key measurements.
        # template = (
        #     f"A photo of {row['Species']} vegetation located in {row['State']}. "
        #     f"Measurements: Height {row['Height_Ave_cm']:.1f}cm, "
        #     f"Green Mass {row['Dry_Green_g']:.1f}g, "
        #     f"Dead Mass {row['Dry_Dead_g']:.1f}g, "
        #     f"Clover {row['Dry_Clover_g']:.1f}g, "
        #     f"NDVI {row['Pre_GSHH_NDVI']:.2f}."
        # )
        
        # Option B: Key-Value style (Sometimes works better for pure regression tasks)
        template = (
            f"Species: {row['Species']}, State: {row['State']}, "
            f"Height: {row['Height_Ave_cm']}, Green: {row['Dry_Green_g']}, "
            f"Dead: {row['Dry_Dead_g']}, Clover: {row['Dry_Clover_g']}, NDVI: {row['Pre_GSHH_NDVI']}"
        )
        
        return template

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Image Loading
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        
        # Use OpenCV for speed, but convert to PIL for OpenCLIP compatibility
        img = cv2.imread(path)
        if img is None:
            # Black placeholder if image missing (prevents crash)
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img) # slice_image expects PIL
        
        if self.transform:
            # pil_img = self.transform(pil_img)
            pil_img  = self.transform(image=pil_img)['image']

        # 2. Tiling
        tiles = slice_image(pil_img, tile_size=self.tile_size)
        
        # 3. Preprocess (Normalization + Tensor conversion)
        # Result: [Num_Tiles, 3, 256, 256]
        tile_tensors = [self.preprocess(tile) for tile in tiles]
        pixel_values = torch.stack(tile_tensors)
        
        return {
            "pixel_values": pixel_values,
            "text": self.texts[idx], # Returns the pre-computed string
            "index": idx
        }