# Setup

## Requirements

- **Python**: 3.10+
- **CUDA**: 12.4+ (NVIDIA driver ≥ 555)
- **GPU**: ≥4 GB VRAM (tested on RTX 3050 4GB)
- **OS**: Linux (tested on Ubuntu 20.04)

## Create Virtual Environment

```bash
python3.10 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel
```

## Install Dependencies

### PyTorch with CUDA

```bash
venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Other Packages

```bash
venv/bin/pip install timm opencv-python-headless albumentations pandas tqdm matplotlib scikit-learn
```

### Verify CUDA

```bash
venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
Device: NVIDIA GeForce RTX 3050 Laptop GPU
```

## Package Versions

| Package | Purpose |
|---------|---------|
| torch 2.6.0+cu124 | Deep learning framework |
| torchvision 0.21.0+cu124 | Image operations |
| timm 1.0.28 | Vision backbones (DINOv3 ViT-L/16) |
| opencv-python-headless | Image I/O |
| albumentations 2.0.8 | Data augmentation |
| pandas 2.3.3 | Data handling |
| numpy 2.2.6 | Numerical computation |
| scikit-learn | Ridge, metrics, CV splits |
| tqdm | Progress bars |
| matplotlib | Figure generation |

## Data

Place the Kaggle competition data in `csiro-biomass/`:

```
csiro-biomass/
├── train.csv          # Training labels (long format)
├── test.csv           # Test labels (long format)
├── train/             # Training images (357 JPEGs)
└── test/              # Test images (805 JPEGs)
```