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
    """
    Slices a numpy image into tiles. Pads the last tiles with black if they are too small.
    Args:
        image: Numpy array (H, W, 3)
        tile_size: Int, size of the tiles
    """
    h, w = image.shape[:2]
    tiles = []
    
    # Loop through height and width
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            
            # 1. Basic slicing (Numpy handles boundary checking by truncating)
            tile = image[i : i + tile_size, j : j + tile_size]
            
            # 2. Check size and Pad if necessary
            # (If we are at the edge, the tile might be smaller than tile_size)
            cur_h, cur_w = tile.shape[:2]
            
            if cur_h != tile_size or cur_w != tile_size:
                # Calculate how much to pad on bottom and right
                pad_bottom = tile_size - cur_h
                pad_right = tile_size - cur_w
                
                # copyMakeBorder is the OpenCV equivalent of creating a new canvas and pasting
                tile = cv2.copyMakeBorder(
                    tile, 
                    top=0, 
                    bottom=pad_bottom, 
                    left=0, 
                    right=pad_right, 
                    borderType=cv2.BORDER_CONSTANT, 
                    value=(0, 0, 0) # Black padding
                )
            
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
        # 1. Image Loading (Native OpenCV)
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        
        img = cv2.imread(path)
        if img is None:
            # Create a black placeholder numpy array
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            # OpenCV loads as BGR, convert to RGB for consistency
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Albumentations Transform
        # Pass the numpy array directly. Albumentations returns a dict.
        if self.transform:
            # Ensure your transform is an Albumentations Compose pipeline
            transformed = self.transform(image=img)
            img = transformed['image']

        # 3. Tiling (Using the new Numpy function)
        tiles = slice_image(img, tile_size=self.tile_size)
        
        # 4. Preprocess (Bridge to OpenCLIP)
        # The OpenCLIP 'preprocess' function expects a PIL Image. 
        # We convert tiles briefly to PIL here just for that compatibility.
        # Result: [Num_Tiles, 3, 256, 256]
        tile_tensors = [self.preprocess(Image.fromarray(tile)) for tile in tiles]
        
        pixel_values = torch.stack(tile_tensors)
        
        return {
            "pixel_values": pixel_values,
            "text": self.texts[idx],
            "index": idx
        }