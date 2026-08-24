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

    Architecture:
        - 4 independent heads (Total, GDM, Green, Clover)
        - Dead derived as Total - GDM
        - Each head: Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear
        - Output clamped via Softplus for non-negativity
    """

    def __init__(self, image_feature_dim: int):
        super().__init__()
        self.head_total = self._create_head(image_feature_dim)
        self.head_gdm = self._create_head(image_feature_dim)
        self.head_clover = self._create_head(image_feature_dim)
        self.head_green = self._create_head(image_feature_dim)

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
        p_dead = p_total - p_gdm
        return p_total, p_gdm, p_green, p_clover, p_dead


# Legacy model with backbone + heads combined
class BiomassModelMLP(nn.Module):
    """Full model with backbone + regression heads (legacy)."""

    def __init__(self, model_name, freeze_backbone=False,
                 checkpoint_path=None, model_state_dict=None, is_linear=False):
        super().__init__()
        self.is_linear = is_linear
        from src.models.backbone import BackboneModel
        self.backbone = BackboneModel(model_name, pretrained=False,
                                      checkpoint_path=checkpoint_path)
        self.backbone.load_pretrained()

        if model_state_dict:
            print("Loading pretrained clip model")
            self.backbone.backbone.load_state_dict(model_state_dict, strict=True)

        image_feature_dim = self.backbone.feature_dim

        self.head_total = self._create_head(image_feature_dim)
        self.head_gdm = self._create_head(image_feature_dim)
        self.head_clover = self._create_head(image_feature_dim)
        self.head_green = self._create_head(image_feature_dim)

        if freeze_backbone:
            self.backbone.freeze_backbone()

    def _create_head(self, feature_dim):
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(self, left, right):
        features = self.backbone(left, right)
        p_total = F.softplus(self.head_total(features))
        p_gdm = F.softplus(self.head_gdm(features))
        p_clover = F.softplus(self.head_clover(features))
        p_green = F.softplus(self.head_green(features))
        p_dead = p_total - p_gdm
        return (p_total, p_gdm, p_green, p_clover, p_dead)


# ============================================================================
# Additional architectures (from models/models.py)
# ============================================================================
class LatentResampler(nn.Module):
    """Perceiver-style resampler for token aggregation."""

    def __init__(self, input_dim, num_latents=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.latents = nn.Parameter(torch.randn(1, num_latents, input_dim) * 0.02)
        self.q = nn.Linear(input_dim, input_dim)
        self.kv = nn.Linear(input_dim, input_dim * 2)
        self.proj = nn.Linear(input_dim, input_dim)
        self.norm_latents = nn.LayerNorm(input_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.norm_post = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, N, C = x.shape
        latents = self.latents.expand(B, -1, -1)
        q_in = self.norm_latents(latents)
        kv_in = self.norm_input(x)
        q = self.q(q_in).reshape(B, self.latents.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv_in).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        latents_out = (attn @ v).transpose(1, 2).reshape(B, self.latents.shape[1], C)
        latents_out = self.proj(latents_out)
        latents = latents + latents_out
        latents = latents + self.mlp(self.norm_post(latents))
        return latents


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block."""

    def __init__(self, dim, kernel_size=7, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class BiAttnBlock(nn.Module):
    """Bidirectional attention block."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = (q @ k.transpose(-2, -1)) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class BiomassMLPBlock(nn.Module):
    """Simple MLP block with LayerNorm."""

    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return shortcut + x
