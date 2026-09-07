#!/usr/bin/env python3
"""
create_hidden_test_slide_assets.py — Beamer slide assets for the
"Model in Action – Qualitative Examples" frame.

POST-HOC VISUALIZATION ONLY: hidden test data is used exclusively to
visualize predictions of the already-final model.

For each of the three selected hidden-test images (Good / Overprediction /
Underprediction — same selection as create_hidden_test_figure.py) this
script saves:

  figures/slide/example_<tag>.png   the quadrat image (2:1 aspect), border
                                    in the category color, 300 DPI
  figures/slide/bar_<tag>.png       compact grouped horizontal bar chart,
                                    True vs Pred for ALL 5 targets
                                    (Green, Dead, Clover, GDM, Total)

Sizing: the Beamer frame displays the bars at width=\\linewidth inside a
0.31\\textwidth column with height=2.2cm (~2:1 aspect), so the bar charts
use a matching wide, short layout with slide-sized fonts.

Usage:
    python figures/create_hidden_test_slide_assets.py
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

import os, sys, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the inference + selection logic from the figure script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_hidden_test_figure import (
    BASE_DIR, TARGET_COLS, load_ground_truth, load_model, run_inference,
    select_examples, load_display_image,
)

OUT_DIR = BASE_DIR / 'figures' / 'slide'

# Display order (top -> bottom in the horizontal bar chart)
BAR_TARGETS = [('Total', 'Dry_Total_g'), ('GDM', 'GDM_g'),
               ('Green', 'Dry_Green_g'), ('Clover', 'Dry_Clover_g'),
               ('Dead', 'Dry_Dead_g')]

COLOR_TRUE = '#4C72B0'   # steel blue (True bars)
ERROR_COLORS = {
    'Good':            '#2ca02c',
    'Overprediction':  '#ff7f0e',
    'Underprediction': '#d62728',
}
TAG = {'Good': 'good', 'Overprediction': 'over', 'Underprediction': 'under'}


# ════════════════════════════════════════════════════════════════════
# Asset builders
# ════════════════════════════════════════════════════════════════════

def save_example_image(row, color, out_path):
    """Quadrat image (2:1) with a colored border, 300 DPI."""
    img = load_display_image(row['image_path'],
                             BASE_DIR / 'csiro-biomass' / 'test')
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 300, h / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(6)
    fig.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)

def _darken(hex_color, factor=0.35):
    """Darken a hex color (factor = fraction of brightness removed)."""
    h = hex_color.lstrip('#')
    r, g, b = (int(round(int(h[i:i + 2], 16) * (1 - factor))) for i in (0, 2, 4))
    return f'#{r:02x}{g:02x}{b:02x}'


def save_bar_chart(row, color, out_path):
    """
    Compact grouped horizontal bars: True vs Pred for all 5 targets.

    Sized for a 0.31\textwidth Beamer column (~3.4-4.3 cm wide, 2.2 cm
    tall): a short figure with large, bold, dark labels so the numbers
    stay legible after LaTeX downscaling. X tick numbers are dropped
    (every bar carries its own value label). Pairs with nearly equal
    values get nudged apart vertically to avoid collisions.
    """
    names = [n for n, _ in BAR_TARGETS]
    true_v = [row[c] for _, c in BAR_TARGETS]
    pred_v = [row['pred_' + c] for _, c in BAR_TARGETS]

    y = np.arange(len(names))[::-1]          # first name at the top
    bh = 0.30

    fig, ax = plt.subplots(figsize=(2.5, 1.55))
    ax.barh(y + bh / 2, true_v, height=bh, color=COLOR_TRUE, label='True')
    ax.barh(y - bh / 2, pred_v, height=bh, color=color, label='Predicted')

    xmax = max(max(true_v), max(pred_v)) * 1.28
    off = xmax * 0.012

    # Value labels: bold, darkened series colors for high contrast.
    # When the two bars of a pair have nearly equal values, their labels
    # cannot sit side by side at the font size used, so the predicted
    # label is staggered horizontally just beyond the true label.
    c_true_txt = _darken(COLOR_TRUE)
    c_pred_txt = _darken(color)
    for yi_t, yi_p, vt, vp in zip(y + bh / 2, y - bh / 2, true_v, pred_v):
        ax.text(vt + off, yi_t, f'{vt:.1f}', va='center', ha='left',
                fontsize=8.5, fontweight='bold', color=c_true_txt, zorder=6)
        s_p = f'{vp:.1f}'
        if abs(vt - vp) < 0.14 * xmax:
            xp = (max(vt, vp) + off
                  + xmax * (0.035 + 0.027 * len(s_p) + 0.015))
        else:
            xp = vp + off
        ax.text(xp, yi_p, s_p, va='center', ha='left',
                fontsize=8.5, fontweight='bold', color=c_pred_txt, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])                        # values are printed on bars
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.margins(y=0.04)

    # Legend below the plot so it can never collide with the bottom bars;
    # the unit travels as the legend title instead of an xlabel.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2,
              frameon=False, fontsize=8, title='g per quadrat',
              title_fontsize=8, handlelength=1.1, handleheight=0.8,
              handletextpad=0.4, labelspacing=0.3, columnspacing=1.0)

    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=300, facecolor='white',
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    sys.path.insert(0, str(BASE_DIR))
    os.chdir(BASE_DIR)
    OUT_DIR.mkdir(exist_ok=True)

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = load_ground_truth()
    model = load_model(device)
    preds = run_inference(model, device)

    # Per-target prediction columns (pred_<target>) for the bar charts
    for i, col in enumerate(TARGET_COLS):
        df['pred_' + col] = preds[:, i]

    df, picked = select_examples(df, preds, topk=10)

    print('\nSlide assets (values for the Beamer frame):')
    print(f'{"Type":<18} {"Image ID":<14} {"Error (g)":>10} {"Rel. err":>9}')
    print('-' * 55)
    for name in ERROR_COLORS:
        idx = picked[name]
        row = df.iloc[idx]
        tag = TAG[name]
        color = ERROR_COLORS[name]

        save_example_image(row, color, OUT_DIR / f'example_{tag}.png')
        save_bar_chart(row, color, OUT_DIR / f'bar_{tag}.png')

        img_id = os.path.basename(row['image_path']).replace('.jpg', '')
        print(f'{name:<18} {img_id:<14} {row["error"]:>+10.1f} '
              f'{row["rel_error_pct"]:>+8.1f}%')

    print(f'\nAssets saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
