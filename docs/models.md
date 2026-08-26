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
│            │     │ → DINOv3             │     │ └────────────────────┘   │
│            │     │ → [CLS] 1024-d       │     │                          │
└────────────┘     └──────────────────────┘     │ Dead = Total − GDM       │
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

`BiomassSimpleMLP` has 4 parallel regression heads:

```python
model = BiomassSimpleMLP(feature_dim=2048)
p_total, p_gdm, p_green, p_clover, p_dead = model(features)
```

Each head:
```
Linear(2048 → 1024) → GELU → Dropout(0.3) → Linear(1024 → 512) → GELU → Dropout(0.3) → Linear(512 → 1) → Softplus
```

Key properties:
- **Softplus activation**: Ensures non-negative predictions
- **Derived Dead**: `p_dead = p_total − p_gdm` enforces the physical constraint
- **4 heads instead of 5**: Dead is derived, reducing parameters and enforcing consistency

## Head Factory (`src/models/factory.py`)

`HeadFactory` provides a unified interface:

```python
# MLP head
model = HeadFactory.create("mlp", feature_dim=2048, device=device)

# Ridge head (sklearn)
model = HeadFactory.create("ridge", alpha=1.0)
```

## Ridge Head

Uses `sklearn.multioutput.MultiOutputRegressor(Ridge(alpha=1.0))`. Faster to train but typically lower performance than MLP. Does not enforce the composition constraint.

## Legacy Models

- `BiomassModelMLP` — Combined backbone + MLP (useful for end-to-end fine-tuning)
- `PerceiverResampler` — Alternative attention-based pooling (not used in primary pipeline)