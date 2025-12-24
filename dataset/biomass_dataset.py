from torch.utils.data import Dataset
from configs.cfg import CFG
import os, cv2, numpy as np
import torch

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