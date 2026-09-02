"""
Model loading utilities for inference.
"""
import os
import torch
import torch.nn as nn
from src.config import InferenceConfig
from src.models.backbone import BackboneModel
from src.models.heads import BiomassSimpleMLP


class ModelLoader:
    """Class for loading trained models for inference."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device

    def load_backbone(self, backbone_path=None) -> nn.Module:
        """Load backbone model."""
        bp = backbone_path or self.config.backbone_path
        print(f"Loading backbone from {bp}")

        if not os.path.exists(bp):
            raise FileNotFoundError(f"Backbone not found: {bp}")

        model = BackboneModel(self.config.model_name, pretrained=False)
        state_dict = torch.load(bp, map_location=self.device)

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.backbone.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        print(f"  Loaded {bp}")
        return model

    def load_fold_models(self, model_dir=None, seeds=None) -> list:
        """Load seed MLP models, optionally restricted to `seeds`."""
        md = model_dir or self.config.model_dir
        print(f"Loading MLP models from {md}")

        models = []
        seed_files = sorted([f for f in os.listdir(md)
                             if f.startswith("seed_") and f.endswith("_final.pt")])

        for sf in seed_files:
            seed = int(sf.split("_")[1])
            if seeds is not None and seed not in seeds:
                continue
            ckpt_path = os.path.join(md, sf)
            model = BiomassSimpleMLP(self.config.feature_dim)
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                try:
                    model.load_state_dict(state)
                except Exception:
                    model_dict = model.state_dict()
                    matched = {k: v for k, v in state.items() if k in model_dict}
                    if matched:
                        model.load_state_dict(matched, strict=False)

            model.to(self.device)
            model.eval()
            models.append(model)
            print(f"  Loaded {sf}")

        return models
