# Models

## Architecture Overview

```
┌────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│ Input      │     │ Backbone (frozen)    │     │ MLP Heads                │
│ 2000×1000  │     │                      │     │                          │
│            │     │ Left crop 1008²      │     │ ┌────────────────────┐   │
│ ┌────┬────┐│ ──► │ → DINOv3             │     │ │ Total Head         │──►│
│ │ L  │ R  ││     │ → [CLS] 1024-d       │ ──► │ │ GDM Head           │──►│
│ └────┴────┘│     │                      │     │ │ Green Head         │──►│
│            │     │ Right crop 1008²     │     │ │ Clover Head        │──►│
│            │     │ → DINOv3             │     │ │ Dead Head          │──►│
│            │     │ → [CLS] 1024-d       │     │ └────────────────────┘   │
└────────────┘     └──────────────────────┘     │                          │
                         │                       └──────────────────────────┘
                    ┌────▼────┐
                    │ Concat  │
                    │ [2048]  │
                    └─────────┘
```

## Backbone (`src/models/backbone.py`)

`BackboneModel` wraps `timm.create_model()` with two-stream processing:

```python
model = BackboneModel(model_name="vit_large_patch16_dinov3", pretrained=True)
features = model(left_crop, right_crop)  # → [B, 2048]
```

- **Architecture**: DINOv3 ViT-Large/16 (303M params)
- **Output per crop**: 1024-dim [CLS] token
- **Frozen**: Backbone weights are not updated during training
- **Two-stream**: Same weights applied independently to left and right crops

## MLP Head (`src/models/heads.py`)

`BiomassSimpleMLP` has 5 independent regression branches (one per biomass component):

```python
model = BiomassSimpleMLP(feature_dim=2048)
p_total, p_gdm, p_green, p_clover, p_dead = model(features)
```

Each branch:
```
Linear(2048 → 1024) → GELU → Dropout(0.3) → Linear(1024 → 512) → GELU → Dropout(0.3) → Linear(512 → 1) → Softplus
```

Key properties:
- **Five independent branches**: Total, GDM, Green, Clover, Dead — each target gets its own head
- **Softplus activation**: Ensures non-negative predictions on every branch

## Head Factory (`src/models/factory.py`)

`HeadFactory` provides a unified interface:

```python
# MLP head
model = HeadFactory.create("mlp", feature_dim=2048, device=device)

# Ridge head (sklearn)
model = HeadFactory.create("ridge")
```

## Ridge Head

Uses `sklearn.multioutput.MultiOutputRegressor(RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]))`: one RidgeCV regressor per target, with the regularization strength selected by internal cross-validation. Faster to train but typically lower performance than MLP.

## Legacy Models

This module previously contained experimental architectures (Perceiver-style resamplers, ConvNeXt blocks, a combined backbone+MLP model). They were removed as unused; the git history retains them.