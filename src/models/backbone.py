"""
Backbone model for two-stream feature extraction.
Single source of truth (merged from models/models.py and infer.ipynb).
"""
import torch
import torch.nn as nn
import timm
import os
from src.config import CFG


class BackboneModel(nn.Module):
    """
    Two-stream feature extractor.
    Returns concatenated left+right features (no regression heads).
    """

    def __init__(self, model_name: str, pretrained: bool = False,
                 checkpoint_path: str = None, is_linear: bool = False):
        super().__init__()
        self.is_linear = is_linear

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0)
        print(f"{model_name} parameters: "
              f"{sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading backbone checkpoint: {checkpoint_path}")
            weights = torch.load(checkpoint_path, map_location='cpu')
            self.backbone.load_state_dict(weights, strict=True)

        self.nf = self.backbone.num_features

    @property
    def feature_dim(self):
        return self.nf * 2

    def forward(self, img_left, img_right, aux_cont=None):
        """
        Args:
            img_left: [B, C, H, W]
            img_right: [B, C, H, W]
        Returns:
            image_features: [B, feature_dim]
        """
        fl = self.backbone(img_left)
        fr = self.backbone(img_right)
        image_features = torch.cat([fl, fr], dim=1)
        return image_features

    def freeze_backbone(self):
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def load_pretrained(self):
        """Load pretrained timm weights."""
        try:
            state_dict = timm.create_model(
                self.backbone.default_cfg['architecture'],
                pretrained=True, num_classes=0).state_dict()
            self.backbone.load_state_dict(state_dict, strict=True)
            print("Pretrained weights loaded (CPU)")
        except Exception as e:
            print(f"Warning: Pretrained load failed: {e}")
