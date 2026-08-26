# Glossary

| Term | Definition |
|------|------------|
| **GDM** | Green Dry Matter — biomass of photosynthetically active vegetation (Green grass + Clover) |
| **Dry Total** | Total above-ground dry biomass (Green + Dead + Clover) |
| **DINOv3** | Self-supervised Vision Transformer pretrained with DINOv3 algorithm (no labels needed) |
| **ViT-L/16** | Vision Transformer Large with 16×16 patch size — 303M parameters |
| **TTA** | Test-Time Augmentation — apply multiple transforms at inference, average predictions |
| **OOF** | Out-Of-Fold — predictions on validation folds during cross-validation |
| **Weighted R²** | Competition metric: weighted R² with target importance weights |
| **Date-location** | CV grouping: images from same State × Date kept together |
| **Embedding** | 2048-dim feature vector output by the backbone |
| **Two-stream** | Processing left and right image crops independently with shared weights |
| **Softplus** | Smooth ReLU variant: `ln(1 + e^x)` — ensures non-negative outputs |
| **Compositional constraint** | `Total = Green + Dead + Clover`; enforced via derivations |
| **LOPO** | Leave-One-Period-Out — cross-validation leaving out one time period |
| **Autocast** | PyTorch automatic mixed precision (FP16/BF16 for speed) |
| **timm** | PyTorch Image Models library (`timm.create_model`) |