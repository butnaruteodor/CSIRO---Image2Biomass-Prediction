"""
MLP regression heads for biomass prediction.
Single source of truth (merged from models/models.py and infer.ipynb).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiomassSimpleMLP(nn.Module):
    """
    MLP regression head that predicts 5 biomass targets from concatenated
    left+right features.

    Architecture (five independent branches):
        - One branch per biomass component: Total, GDM, Green, Clover, Dead
        - Each branch: Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear -> Softplus
        - Softplus guarantees non-negative predictions
    """

    def __init__(self, image_feature_dim: int):
        super().__init__()
        self.head_total = self._create_head(image_feature_dim)
        self.head_gdm = self._create_head(image_feature_dim)
        self.head_clover = self._create_head(image_feature_dim)
        self.head_green = self._create_head(image_feature_dim)
        self.head_dead = self._create_head(image_feature_dim)

    @staticmethod
    def _create_head(feature_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(self, feats):
        """
        Args:
            feats: [B, feature_dim] concatenated left+right features
        Returns:
            (p_total, p_gdm, p_green, p_clover, p_dead) each [B, 1]
        """
        p_total = F.softplus(self.head_total(feats))
        p_gdm = F.softplus(self.head_gdm(feats))
        p_clover = F.softplus(self.head_clover(feats))
        p_green = F.softplus(self.head_green(feats))
        p_dead = F.softplus(self.head_dead(feats))
        return p_total, p_gdm, p_green, p_clover, p_dead

