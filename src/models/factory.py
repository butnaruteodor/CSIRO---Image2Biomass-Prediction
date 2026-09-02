"""
Head factory for easy switching between model heads.

Usage:
    factory = HeadFactory()
    model = factory.create("mlp", feature_dim=2048)
    model = factory.create("ridge")
"""
from sklearn.multioutput import MultiOutputRegressor
from src.models.heads import BiomassSimpleMLP


class HeadFactory:
    """Factory for creating model heads on precomputed embeddings."""

    @staticmethod
    def create(head_type: str, **kwargs):
        """
        Create a model head.

        Args:
            head_type: 'mlp' | 'ridge'
            **kwargs: type-specific arguments

        MLP kwargs:
            feature_dim (int): Input feature dimension (default: 2048)
            device: torch device

        Ridge kwargs:
            alphas (tuple): Candidate regularization strengths for RidgeCV
            device: ignored, sklearn runs on CPU
        """
        if head_type == "mlp":
            return _create_mlp(**kwargs)
        elif head_type == "ridge":
            return _create_ridge(**kwargs)
        else:
            raise ValueError(f"Unknown head type: {head_type}. Choose 'mlp' or 'ridge'.")


def _create_mlp(feature_dim=2048, device=None, **_):
    """Create an MLP head."""
    import torch
    model = BiomassSimpleMLP(feature_dim)
    if device is not None:
        model = model.to(device)
    return model


def _create_ridge(alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), **_):
    """
    Create a Ridge regression head (sklearn).

    One RidgeCV regressor is fitted per target via MultiOutputRegressor;
    the regularization parameter is selected from `alphas` by internal
    cross-validation.
    """
    from sklearn.linear_model import RidgeCV
    return MultiOutputRegressor(RidgeCV(alphas=list(alphas)))