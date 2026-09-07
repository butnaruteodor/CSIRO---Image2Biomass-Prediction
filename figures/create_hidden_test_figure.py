#!/usr/bin/env python3
"""
create_hidden_test_figure.py — Qualitative visualization of model behavior on
the hidden Kaggle test set.

POST-HOC VISUALIZATION ONLY:
The hidden test data is used exclusively to visualize already-final model
predictions. No model parameters, hyperparameters, stopping epochs, or
validation decisions were (or may be) based on this data.

Pipeline (reuses the existing codebase inference path):
  1. Ground truth for the 805 hidden test images: csiro-biomass/test.csv
     (via src.data.preprocessing.get_test_df), aligned to the cached
     embedding order in embeddings/test_image_ids.csv.
  2. Features: cached DINOv3 test embeddings with the 5 TTA views used for
     Kaggle submissions (embeddings/test_tta_embeddings.pt, image-major
     (N, K) layout — verified against ground-truth weighted R^2).
  3. Head: final MLP checkpoint from models/seed_101_final.pt.
  4. Predictions are averaged over TTA views (matches EnsemblePredictor).

Selection:
  - Good            : smallest |error| (pred - true)
  - Overprediction  : largest positive error
  - Underprediction : largest negative error
  Among the top candidates for each category, the image whose pasture
  appearance (mean greenness) is most distinct from the already-picked
  images is preferred, to illustrate visual variety.

Output:
  figures/model_action_hidden.png           (300 DPI, 1x3 panels)
  figures/model_action_hidden_summary.txt   (image IDs, true/pred/error)

Usage:
    python figures/create_hidden_test_figure.py [--topk 10]
"""
import os, sys

# ── Auto-detect venv and re-execute if needed ───────────────────────
_venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'venv', 'bin', 'python3')
if sys.executable != _venv_python and os.path.exists(_venv_python):
    try:
        import torch  # noqa
        import matplotlib  # noqa
        import cv2  # noqa
    except ImportError:
        os.execv(_venv_python, [_venv_python] + sys.argv)

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from matplotlib.patches import Rectangle

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / 'figures'

# Inference-path constants (match src/config.py InferenceConfig)
EMBED_DIR = BASE_DIR / 'embeddings'
MODEL_DIR = BASE_DIR / 'models'
MODEL_FILE = 'seed_101_final.pt'
FEATURE_DIM = 2048            # embeddings/metadata.json: embedding_dim
TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TOTAL_IDX = TARGET_COLS.index('Dry_Total_g')
DISPLAY_WIDTH = 800           # px, image display width (2000x1000 -> 800x400)

ERROR_TYPES = [
    ('Good',            '#2ca02c'),   # green
    ('Overprediction',  '#ff7f0e'),   # orange
    ('Underprediction', '#d62728'),   # red
]

# ══════════════════════════════════════════════════════════════════════════
# Data / inference (reuses existing codebase components)
# ══════════════════════════════════════════════════════════════════════════

def load_ground_truth():
    """Hidden test ground truth aligned to the cached embedding order."""
    from src.data.preprocessing import get_test_df

    ids = pd.read_csv(EMBED_DIR / 'test_image_ids.csv')
    tdf = get_test_df()
    df = ids.merge(tdf, on='image_path', how='left', validate='one_to_one')
    assert df['Dry_Total_g'].notna().all(), 'Missing ground truth for some images'
    return df


def load_model(device):
    """Load the final MLP checkpoint used for Kaggle submissions."""
    from src.models.heads import BiomassSimpleMLP

    ckpt = MODEL_DIR / MODEL_FILE
    model = BiomassSimpleMLP(FEATURE_DIM)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f'Loaded MLP head: {ckpt}')
    return model


def run_inference(model, device):
    """
    Run the final MLP on the cached TTA test embeddings and average views.

    Returns (N, 5) predictions in TARGET_COLS order.
    """
    ids = pd.read_csv(EMBED_DIR / 'test_image_ids.csv')
    N = len(ids)
    X = torch.load(EMBED_DIR / 'test_tta_embeddings.pt', map_location='cpu').float()
    K = X.shape[0] // N
    assert X.shape[0] == K * N, f'Unexpected TTA layout: {X.shape} for N={N}'

    with torch.inference_mode():
        p_total, p_gdm, p_green, p_clover, p_dead = model(X.to(device))
        preds_flat = torch.cat([p_green, p_dead, p_clover, p_gdm, p_total],
                               dim=1).cpu().numpy()

    # Cached file is image-major: (N, K, 5) -> average over the K views.
    # (Verified: this ordering reproduces the hidden-test weighted R^2
    #  reported in the README; the view-major reading does not.)
    preds = preds_flat.reshape(N, K, len(TARGET_COLS)).mean(axis=1)
    print(f'Inference done: {N} images x {K} TTA views -> {preds.shape}')
    return preds


# ══════════════════════════════════════════════════════════════════════════
# Image selection
# ══════════════════════════════════════════════════════════════════════════

def image_greenness(image_path, image_dir):
    """Mean (G - R) over a downscaled RGB image — simple pasture-greenness proxy."""
    img = cv2.imread(str(image_dir / os.path.basename(image_path)))
    if img is None:
        return 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 100), interpolation=cv2.INTER_AREA)
    return float((img[:, :, 1].astype(np.float32) - img[:, :, 0]).mean())


def select_examples(df, preds, topk=10):
    """
    Pick (good, over, under) rows maximizing error extremes, preferring
    visually distinct pastures (greenness spread) among top candidates.
    """
    true_t = df['Dry_Total_g'].values
    pred_t = preds[:, TOTAL_IDX]
    err = pred_t - true_t

    df = df.copy().reset_index(drop=True)
    df['pred_total'] = pred_t
    df['error'] = err
    df['rel_error_pct'] = 100.0 * err / np.maximum(true_t, 1e-6)

    candidates = {
        'Good':            np.argsort(np.abs(err))[:topk],
        'Overprediction':  np.argsort(err)[::-1][:topk],   # largest positive
        'Underprediction': np.argsort(err)[:topk],         # largest negative
    }

    image_dir = BASE_DIR / 'csiro-biomass' / 'test'
    greenness = {int(idx): image_greenness(df.at[idx, 'image_path'], image_dir)
                 for idxs in candidates.values() for idx in idxs}

    picked = {}
    for name, idxs in candidates.items():
        idxs = [int(i) for i in idxs]
        if not picked:
            # First pick: median-greenness candidate (avoid extreme outliers)
            best = min(idxs, key=lambda i: abs(greenness[i] -
                        np.median([greenness[j] for j in idxs])))
        else:
            # Later picks: maximize min greenness distance to picked set
            best = max(idxs, key=lambda i: min(abs(greenness[i] - greenness[p])
                                               for p in picked.values()))
        picked[name] = int(best)
    return df, picked

# ══════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════

def load_display_image(image_path, image_dir, width=DISPLAY_WIDTH):
    """Load a 2000x1000 test image and resize to the display width."""
    img = cv2.imread(str(image_dir / os.path.basename(image_path)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    new_h = int(round(h * width / w))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def create_figure(df, picked, out_path):
    image_dir = BASE_DIR / 'csiro-biomass' / 'test'
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2))

    for ax, (name, color) in zip(axes, ERROR_TYPES):
        idx = picked[name]
        row = df.iloc[idx]
        img = load_display_image(row['image_path'], image_dir)

        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        # Colored border marking the error type
        h, w = img.shape[:2]
        ax.add_patch(Rectangle((-1, -1), w + 2, h + 2, fill=False,
                               edgecolor=color, linewidth=8,
                               clip_on=False, zorder=5))

        # Text panel: true vs predicted
        box = (
            f"True  (Dry Total): {row['Dry_Total_g']:8.1f} g\n"
            f"Pred (Dry Total): {row['pred_total']:8.1f} g\n"
            f"Error: {row['error']:+.1f} g  ({row['rel_error_pct']:+.1f}%)"
        )
        ax.text(0.5, -0.05, box, transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                          edgecolor=color, linewidth=1.6))

        img_id = os.path.basename(row['image_path']).replace('.jpg', '')
        ax.set_title(f'{name}\n{img_id}', fontsize=14, fontweight='bold',
                     color=color, pad=10)

    fig.suptitle('Model Behavior on Hidden Test Data (post-hoc visualization)',
                 fontsize=15, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.26, wspace=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Figure saved: {out_path}')


def write_summary(df, picked, wr2, out_path):
    lines = [
        '=' * 72,
        'MODEL BEHAVIOR ON HIDDEN TEST DATA — QUALITATIVE SUMMARY',
        '=' * 72,
        'NOTE: Hidden test data used ONLY for post-hoc visualization of the',
        '      final model. No model parameters or validation decisions were',
        '      based on this data.',
        '',
        f'Model checkpoint : models/{MODEL_FILE} (final MLP, Kaggle submission head)',
        f'Features         : frozen DINOv3 ViT-L embeddings + 5-view TTA',
        f'                   (embeddings/test_tta_embeddings.pt, averaged over views)',
        f'Target           : Dry_Total_g (g per quadrat)',
        f'Hidden test size : {len(df)} images',
        f'Weighted R2      : {wr2:.4f} (all 5 targets, official metric weights)',
        '',
        f'{"Type":<18} {"Image ID":<14} {"True (g)":>9} {"Pred (g)":>9} '
        f'{"Error (g)":>10} {"Rel. err":>9}',
        '-' * 72,
    ]
    for name, _ in ERROR_TYPES:
        row = df.iloc[picked[name]]
        img_id = os.path.basename(row['image_path']).replace('.jpg', '')
        lines.append(
            f'{name:<18} {img_id:<14} {row["Dry_Total_g"]:>9.1f} '
            f'{row["pred_total"]:>9.1f} {row["error"]:>+10.1f} '
            f'{row["rel_error_pct"]:>+8.1f}%')
    lines += ['=' * 72, '']

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Summary saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Hidden-test qualitative figure')
    parser.add_argument('--topk', type=int, default=10,
                        help='Candidate pool per error category (default: 10)')
    args = parser.parse_args()

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(BASE_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = load_ground_truth()
    model = load_model(device)
    preds = run_inference(model, device)

    # Sanity metric (same as scripts/evaluate_local.py) — expected ~0.6
    from src.evaluation.metrics import global_weighted_r2_score
    y = df[TARGET_COLS].values.astype(np.float32)
    wr2 = global_weighted_r2_score(y, preds)
    print(f'Hidden-test weighted R2 (final MLP + TTA): {wr2:.4f}')

    df, picked = select_examples(df, preds, topk=args.topk)
    create_figure(df, picked, FIGURES_DIR / 'model_action_hidden.png')
    write_summary(df, picked, wr2, FIGURES_DIR / 'model_action_hidden_summary.txt')


if __name__ == '__main__':
    main()
