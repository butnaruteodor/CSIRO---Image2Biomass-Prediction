#!/usr/bin/env python3
"""Model overview figure: Input → Frozen Backbone → Multi-branch MLP."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from PIL import Image

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})

BASE = '/home/teo/repos/CSIRO---Image2Biomass-Prediction'
OUTPUT = os.path.join(BASE, 'figures', 'model_overview.pdf')
OUTPUT_PNG = os.path.join(BASE, 'figures', 'model_overview.png')
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

GREEN = '#C8F7D5'
GRAY  = '#E0E0E0'
COLS  = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']

fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.0))
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 6.0)
ax.axis('off')

def rb(x, y, w, h, c, a=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",
                                facecolor=c,edgecolor='black',linewidth=0.4,alpha=a))

def tx(x, y, s, sz=3.5, w='normal', c='black', ha='center', style='normal'):
    ax.text(x, y, s, fontsize=sz, color=c, ha=ha, va='center', fontweight=w,
            fontstyle=style, clip_on=False)

def ar(x1, y1, x2, y2, lw=0.6):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='#444', lw=lw,
                                shrinkA=4, shrinkB=4), clip_on=False)

# Load sample image for square thumbnails
img_path = os.path.join(BASE, 'csiro-biomass', 'train', 'ID227847873.jpg')
sample = np.array(Image.open(img_path))
h, w = sample.shape[0], sample.shape[1]
# Both halves are square (1000×1000 each from a 2000×1000 image)
left_img = sample[:, :w//2, :]
right_img = sample[:, w//2:, :]

CY = 3.2

# ═══ 1. INPUT ═════════════════════════════════════════════════════════
ix, iw, ih = 0.15, 1.6, 2.6
iy = CY - ih/2
rb(ix, iy, iw, ih, GREEN)
tx(ix+iw/2, iy+ih-0.2, 'Input Image', sz=4, w='bold')
tx(ix+iw/2, iy+ih-0.45, '(2000×1000 px)', sz=3)

# Square thumbnails stacked vertically
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

thumb_size = 0.55  # square
# Center x for thumbnails
thumb_cx = ix + iw/2

# Left crop (top)
yl = iy + 1.6
l_sq = np.array(Image.fromarray(left_img).resize((int(thumb_size*50), int(thumb_size*50))))
im_l = OffsetImage(l_sq, zoom=0.3)
ab_l = AnnotationBbox(im_l, (thumb_cx, yl), frameon=True, pad=0.02,
                       box_alignment=(0.5, 0.5), clip_on=False)
ax.add_artist(ab_l)
tx(ix+0.35, yl, 'L', sz=3, w='bold', ha='left')
tx(ix+iw-0.1, yl, '1008²', sz=2.5, ha='right')

# Right crop (bottom)
yr = iy + 0.9
r_sq = np.array(Image.fromarray(right_img).resize((int(thumb_size*50), int(thumb_size*50))))
im_r = OffsetImage(r_sq, zoom=0.3)
ab_r = AnnotationBbox(im_r, (thumb_cx, yr), frameon=True, pad=0.02,
                       box_alignment=(0.5, 0.5), clip_on=False)
ax.add_artist(ab_r)
tx(ix+0.35, yr, 'R', sz=3, w='bold', ha='left')
tx(ix+iw-0.1, yr, '1008²', sz=2.5, ha='right')

# ═══ Arrow: Input → Backbone ═════════════════════════════════════════
ar(ix+iw-0.1, CY, ix+iw+0.6, CY)

# ═══ 2. BACKBONE ══════════════════════════════════════════════════════
bx, bw, bh = 2.25, 2.6, 1.5
by = CY - bh/2
rb(bx, by, bw, bh, GRAY)
# Frozen indicator: dashed border
# ax.add_patch(FancyBboxPatch((bx+0.1,by+0.1),bw-0.2,bh-0.2,
#                              boxstyle="round,pad=0.03",facecolor='none',
#                              edgecolor='#999',linewidth=0.6,linestyle='--'))

tx(bx+bw/2, by+bh-0.2, 'Feature Extractor', sz=4, w='bold')
tx(bx+bw/2, by+bh-0.45, 'DINOv3 ViT-L/16', sz=3.5)
# tx(bx+bw/2, by+bh-0.70, '(frozen)', sz=3)

# Simple two-path representation: two horizontal lines through the backbone
# representing L and R being processed by the same weights
ly = CY + 0.3
lry = CY - 0.3
# ax.plot([bx, bx+bw], [ly, ly], color='#444', lw=0.5, clip_on=False)
# ax.plot([bx, bx+bw], [lry, lry], color='#444', lw=0.5, clip_on=False)
# # Arrow heads at the right end
# ar(bx+bw-0.15, ly, bx+bw, ly, lw=0.5)
# ar(bx+bw-0.15, lry, bx+bw, lry, lw=0.5)
# Label "shared" in the middle
tx(bx+bw/2, CY, 'Shared', sz=3, c='#555')

# Output label below
tx(bx+bw/2, by+0.2, 'Concat → [2048]', sz=3, w='bold')

# ═══ Arrow → Heads ════════════════════════════════════════════════════
sx = bx + bw + 0.6
ar(bx+bw-0.1, by+0.75, sx, by+0.75)

# Vertical stem going to heads
n = 5
hh = 0.25
hg = 0.15
th = n*hh + (n-1)*hg
hb = CY - th/2

ax.plot([sx-0.2, sx-0.2], [by-0.1, hb+th-0.1], color='#444', lw=0.6, clip_on=False)

# ═══ 3. MLP HEADS ═════════════════════════════════════════════════════
hx, hw = sx + 0.2, 1.6
lbls = ['Dead (0.1)', 'Clover (0.1)', 'Green (0.1)', 'GDM (0.2)', 'Total (0.5)']

for i in range(n):
    hy = hb + i*(hh+hg)
    rb(hx, hy, hw, hh, COLS[i], 0.8)
    tx(hx+hw/2, hy+hh/2, 'MLP', sz=3, w='bold', c='#222')
    # Target name outside to the right
    tx(hx+hw+0.06, hy+hh/2, lbls[i], sz=3, c='#444', ha='left')
    # Horizontal connector from stem
    ax.plot([sx, hx], [hy+hh/2, hy+hh/2], color='#444', lw=0.4, clip_on=False)
    ax.annotate('', xy=(hx, hy+hh/2), xytext=(hx-0.03, hy+hh/2),
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.4), clip_on=False)

tx(hx+hw/2, hb+th+0.2, 'MLP Heads', sz=4, w='bold')

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(OUTPUT, format='pdf')
fig.savefig(OUTPUT_PNG, format='png', dpi=600)
print(f'Saved: {OUTPUT}')
print(f'Saved: {OUTPUT_PNG}')
plt.close(fig)