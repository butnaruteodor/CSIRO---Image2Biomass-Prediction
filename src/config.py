"""
Unified configuration dataclasses for training and inference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import torch
import numpy as np


# ============================================================================
# Training Configuration  (superset of original CFG)
# ============================================================================
@dataclass
class TrainingConfig:
    # Paths
    base_path: str = 'csiro-biomass'
    train_csv: str = field(init=False)
    train_image_dir: str = field(init=False)
    model_dir: str = 'out'
    embed_dir: str = 'embeddings'

    def __post_init__(self):
        self.train_csv = os.path.join(self.base_path, 'train.csv')
        self.train_image_dir = os.path.join(self.base_path, 'train')

    # Folds / seeds
    n_folds: int = 5
    seeds: tuple = (13, 21, 42, 87, 101)

    # Backbone
    model_name: str = 'vit_large_patch16_dinov3'
    img_size: int = 1008
    freeze_backbone: bool = True
    checkpoint_path: Optional[str] = None

    # MLP training
    batch_size: int = 8
    grad_acc: int = 1
    epochs: int = 80
    lr: float = 1e-3
    wd: float = 0.01
    warmup_epochs: int = 5
    patience: int = 15
    n_aug: int = 15

    # Loss weights
    r2_weights_val: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.2, 0.5]))
    r2_weights_train: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0]))

    # Targets
    target_cols: tuple = ('Dry_Total_g', 'GDM_g', 'Dry_Green_g')
    derived_cols: tuple = ('Dry_Clover_g', 'Dry_Dead_g')
    all_target_cols: tuple = ('Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g')

    # Determinism
    deterministic: bool = True
    seed: int = 3858

    # Device
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    @property
    def feature_dim_half(self):
        """Half the feature dimension (single backbone output)."""
        return 1024  # vit_large_patch16_dinov3

    @property
    def feature_dim(self):
        """Concatenated left+right feature dimension."""
        return self.feature_dim_half * 2  # 2048


# ============================================================================
# Inference Configuration
# ============================================================================
@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline (matches Kaggle submission)."""
    # Paths
    base_path: Path = Path('csiro-biomass')
    test_csv: Path = field(init=False)
    test_image_dir: Path = field(init=False)
    submission_file: str = 'submission.csv'
    embed_dir: str = 'embeddings'
    model_dir: str = 'results/submission_models'
    backbone_path: Path = Path('vit_large_patch16_dinov3.pth')

    def __post_init__(self):
        self.test_csv = self.base_path / 'test.csv'
        self.test_image_dir = self.base_path / 'test'

    # Model settings (must match training)
    model_name: str = 'vit_large_patch16_dinov3'
    img_size: int = 1008 

    # Device
    device: torch.device = field(default_factory=lambda: torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ))

    # Inference
    batch_size: int = 8
    num_workers: int = 1
    n_folds: int = 5
    n_tta: int = 3  # original + HFlip + VFlip

    # Ensemble weights (MLP-only for clean baseline)
    w_mlp: float = 1.0

    # Targets (must match training)
    all_target_cols: tuple = ('Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g')

    @property
    def feature_dim(self):
        return 2048


# ============================================================================
# Backward-compatible CFG class (re-exports from TrainingConfig)
# ============================================================================
class CFG:
    """Legacy config class — delegates to TrainingConfig for new code."""
    BASE_PATH = 'csiro-biomass'
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    MODEL_DIR = 'out'
    N_FOLDS = 5
    LOG = False

    # TIMM
    MODEL_NAME = 'vit_large_patch16_dinov3'
    PATIENCE = 20

    # CLIP (legacy)
    CLIP_NAME = 'convnext_base_w'
    CLIP_FT_NAME = "laion2b_s13b_b82k_augreg"
    CLIP_PATIENCE = 10
    CLIP_EPOCHS = 50
    CLIP_WD = 0.01

    CHECKPOINT_PATH = None
    FREEZE_BACKBONE = True

    IMG_SIZE = 1008
    ALPHA_CLIP = 0.1
    BATCH_SIZE = 8
    GRAD_ACC = 1
    NUM_WORKERS = 4
    EPOCHS = 60
    LR = 1e-3
    WD = 0.01
    WARMUP_EPOCHS = 4
    WARMUP_HEAD_EPOCHS = 5

    DETERMINISTIC = True
    SEED = 3858

    TARGET_COLS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    R2_WEIGHTS_VAL = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    R2_WEIGHTS_TRAIN = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    W_SPECIES = 0.25
    W_STATE = 0.25
    W_CONT = 0.5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')