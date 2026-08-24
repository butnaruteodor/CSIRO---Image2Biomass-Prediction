"""
Feature extraction for inference.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import InferenceConfig
from src.data.dataset import TestBiomassDataset


class GlobalFeatureExtractor:
    """Extracts features from test images using backbone + TTA."""

    def __init__(self, backbone, config: InferenceConfig):
        self.backbone = backbone
        self.config = config
        self.device = config.device

    def get_all_embeddings(self, test_df, tta_transforms):
        """
        Extract features for all TTA views.
        Returns (X_massive, N, K) where:
            X_massive: (N*K, D) stacked features
            N: number of images
            K: number of TTA views
        """
        N = len(test_df)
        K = len(tta_transforms)
        all_folds_features = []

        print(f"Extracting features ({K} TTA views)...")

        for i, transform in enumerate(tta_transforms):
            print(f"  View {i+1}/{K}...")
            dataset = TestBiomassDataset(
                test_df, transform, self.config.test_image_dir)
            loader = DataLoader(
                dataset, batch_size=self.config.batch_size,
                shuffle=False, num_workers=self.config.num_workers,
                pin_memory=True)
            view_feats = self._extract_one_view(loader)
            all_folds_features.append(view_feats)

        X_massive = np.vstack(all_folds_features)
        print(f"  Matrix shape: {X_massive.shape}")
        return X_massive, N, K

    def _extract_one_view(self, loader):
        feats_list = []
        with torch.inference_mode():
            for img_left, img_right in tqdm(loader, leave=False):
                img_left = img_left.to(self.device)
                img_right = img_right.to(self.device)
                with torch.amp.autocast("cuda") if torch.cuda.is_available() else torch.amp.autocast("cpu"):
                    batch_feats = self.backbone(img_left, img_right)
                feats_list.append(batch_feats.cpu().numpy())
        return np.vstack(feats_list)
