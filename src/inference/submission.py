"""
Submission CSV creation.
"""
import pandas as pd
import numpy as np
from src.config import InferenceConfig


class SubmissionCreator:
    """Creates Kaggle submission CSV from predictions."""

    def __init__(self, config: InferenceConfig):
        self.config = config

    def create(self, predictions, test_df_long, test_df_unique):
        """
        Create and save submission CSV.

        Args:
            predictions: dict with keys total, gdm, green, clover, dead
            test_df_long: Original test.csv (long format)
            test_df_unique: DataFrame of unique images
        """
        print("Creating submission CSV...")

        preds_wide = pd.DataFrame({
            "image_path": test_df_unique["image_path"],
            "Dry_Green_g": predictions["green"],
            "Dry_Dead_g": predictions["dead"],
            "Dry_Clover_g": predictions["clover"],
            "GDM_g": predictions["gdm"],
            "Dry_Total_g": predictions["total"],
        })

        preds_long = preds_wide.melt(
            id_vars=["image_path"],
            value_vars=list(self.config.all_target_cols),
            var_name="target_name",
            value_name="target",
        )

        submission = pd.merge(
            test_df_long[["sample_id", "image_path", "target_name"]],
            preds_long,
            on=["image_path", "target_name"],
            how="left",
        )

        submission = submission[["sample_id", "target"]]
        submission.to_csv(self.config.submission_file, index=False)

        print(f"Submission saved: {self.config.submission_file}")
        print(f"  Shape: {submission.shape}")
        return submission
