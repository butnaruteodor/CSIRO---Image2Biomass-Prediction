"""
Orchestrator for the full inference pipeline.
"""
import os
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

from src.config import InferenceConfig
from src.inference.model_loader import ModelLoader
from src.inference.feature_extractor import GlobalFeatureExtractor
from src.inference.ensemble import EnsemblePredictor
from src.inference.submission import SubmissionCreator
from src.data.augmentations import get_tta_transforms


class InferencePipeline:
    """Orchestrates the entire inference pipeline."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.submission_creator = SubmissionCreator(config)

    def run(self) -> None:
        """Execute the full inference pipeline."""
        print(f"\n{'=' * 70}")
        print(f"Starting Inference Pipeline")
        print(f"{'=' * 70}")

        # 1. Load test data
        test_df_long, test_df_unique = self._load_test_data()

        # 2. Load backbone
        backbone = self.model_loader.load_backbone()
        feat_extractor = GlobalFeatureExtractor(backbone, self.config)
        del backbone
        torch.cuda.empty_cache()

        # 3. Load MLP models
        mlp_models = self.model_loader.load_fold_models()

        # 4. TTA inference
        tta_transforms = get_tta_transforms(self.config.img_size)
        X_massive, N, K = feat_extractor.get_all_embeddings(test_df_unique, tta_transforms)

        predictor = EnsemblePredictor(mlp_models, self.config)
        final_preds = predictor.predict_all(X_massive, N, K)

        # 5. Create submission
        self.submission_creator.create(final_preds, test_df_long, test_df_unique)

        # 6. Cleanup
        del X_massive, mlp_models, predictor
        gc.collect()
        torch.cuda.empty_cache()

    def _load_test_data(self):
        """Load test CSV data."""
        print(f"Loading test data: {self.config.test_csv}")

        if not os.path.exists(self.config.test_csv):
            raise FileNotFoundError(f"test.csv not found: {self.config.test_csv}")

        test_df_long = pd.read_csv(self.config.test_csv)
        test_df_unique = test_df_long.drop_duplicates(
            subset=["image_path"]
        ).reset_index(drop=True)

        print(f"  Long format: {len(test_df_long)} rows")
        print(f"  Unique images: {len(test_df_unique)} images")

        return test_df_long, test_df_unique
