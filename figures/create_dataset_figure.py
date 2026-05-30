#!/usr/bin/env python3
"""
Generate a three-panel dataset overview figure for IEEE paper:
(a) Sample top-view pasture image with biomass values
(b) Target biomass distributions (boxplots)
(c) Temporal sampling structure (monthly grouped)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from datetime import datetime
import os
from PIL import Image
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Paths ──────────────────────────────────────────────────────────────
BASE = '/home/teo/repos/CSIRO---Image2Biomass-Prediction'
CSV = os.path.join(BASE, 'csiro-biomass', 'train.csv')
TRAIN_DIR = os.path.join(BASE, 'csiro-biomass', 'train')
OUTPUT = os.path.join(BASE, 'figures', 'dataset_overview.pdf')
OUTPUT_PNG = os.path.join(BASE, 'figures', 'dataset_overview.png')
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df['Date'] = pd.to_datetime(df['Sampling_Date'], format='%Y/%m/%d')

# Per-image target table (one row per image)
img_targets = df.pivot_table(
    index='image_path', columns='target_name', values='target', aggfunc='first'
).reset_index()
img_meta = df[['image_path', 'State', 'Date']].drop_duplicates(subset='image_path')
img_data = img_meta.merge(img_targets, on='image_path')

# ── Panel (a): Sample image with biomass values ──────────────────────
best_img_path = os.path.join(TRAIN_DIR, 'ID227847873.jpg')  # Vic, 2015/9/29
sample_img = np.array(Image.open(best_img_path))
sample_img_rot = np.rot90(sample_img, k=-1)

# Get biomass values for this image
img_row = img_data[img_data['image_path'] == 'train/ID227847873.jpg'].iloc[0]
biomass_text = (
    f"Dry Total: {img_row['Dry_Total_g']:.1f}g\n"
    f"Dry Green: {img_row['Dry_Green_g']:.1f}g\n"
    f"Dry Dead:  {img_row['Dry_Dead_g']:.1f}g\n"
    f"Dry Clover: {img_row['Dry_Clover_g']:.1f}g\n"
    f"GDM:       {img_row['GDM_g']:.1f}g"
)

# ── Panel (b): Target distributions ──────────────────────────────────
target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
target_labels = [
    'Dry\nGreen',
    'Dry\nDead',
    'Dry\nClover',
    'GDM',
    'Dry\nTotal',
]

# ── Panel (c): Temporal structure (monthly bins) ─────────────────────
# LOPO boundaries — sample-balanced split (same logic as experiment_2.py)
date_order = sorted(df['Date'].unique())
date_order = [pd.Timestamp(d) for d in date_order]
n_dates = len(date_order)
total_samples = df['image_path'].nunique()
target = total_samples // 3

period_dates = {'Early': [], 'Middle': [], 'Late': []}
period_names = ['Early', 'Middle', 'Late']
cumulative = 0
current_period_idx = 0

for i, d in enumerate(date_order):
    count = df[df['Date'] == d]['image_path'].nunique()
    cumulative += count
    period_dates[period_names[current_period_idx]].append(d)
    
    if cumulative >= target and current_period_idx < 2:
        remaining_dates = n_dates - i - 1
        remaining_periods = 2 - current_period_idx
        if remaining_dates >= remaining_periods:
            cumulative = 0
            current_period_idx += 1

early_dates = set(period_dates['Early'])
middle_dates = set(period_dates['Middle'])
late_dates = set(period_dates['Late'])

early_last = max(early_dates)
middle_last = max(middle_dates)

print(f"Period 1 (Early): {min(early_dates).strftime('%Y-%m-%d')} to {early_last.strftime('%Y-%m-%d')} ({len(early_dates)} dates, "
      f"{df[df['Date'].isin(early_dates)]['image_path'].nunique()} samples)")
print(f"Period 2 (Middle): {min(middle_dates).strftime('%Y-%m-%d')} to {middle_last.strftime('%Y-%m-%d')} ({len(middle_dates)} dates, "
      f"{df[df['Date'].isin(middle_dates)]['image_path'].nunique()} samples)")
print(f"Period 3 (Late): {min(late_dates).strftime('%Y-%m-%d')} to {max(late_dates).strftime('%Y-%m-%d')} ({len(late_dates)} dates, "
      f"{df[df['Date'].isin(late_dates)]['image_path'].nunique()} samples)")
# Verify no date leakage
all_split_dates = set()
for dlist in period_dates.values():
    for d in dlist:
        assert d not in all_split_dates, f"DUPLICATE DATE {d}!"
        all_split_dates.add(d)
assert len(all_split_dates) == len(date_order), "Not all dates assigned!"

state_colors = {'NSW': '#E24A33', 'Tas': '#348ABD', 'Vic': '#988ED5', 'WA': '#F5A623'}
state_labels_short = {'NSW': 'NSW', 'Tas': 'Tas', 'Vic': 'Vic', 'WA': 'WA'}

# Group by month for cleaner bars
counts_per_date = df[['image_path', 'Date', 'State']].drop_duplicates(subset=['image_path', 'State'])
counts_per_date['Month'] = counts_per_date['Date'].dt.to_period('M')
monthly_counts = counts_per_date.groupby(['Month', 'State']).size().unstack(fill_value=0)
for s in ['NSW', 'Tas', 'Vic', 'WA']:
    if s not in monthly_counts.columns:
        monthly_counts[s] = 0
monthly_counts = monthly_counts[['NSW', 'Tas', 'Vic', 'WA']]

# Convert month periods to datetime for plotting (use mid-month)
month_dates = [p.to_timestamp() + pd.DateOffset(days=14) for p in monthly_counts.index]

# ── Build the figure ──────────────────────────────────────────────────
fig = plt.figure(figsize=(7.16, 2.8))

gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.3, 1.7], wspace=0.28,
                       left=0.05, right=0.97, bottom=0.18, top=0.92)

# ── Panel (a): Image with biomass overlay ────────────────────────────
ax_a = fig.add_subplot(gs[0])
ax_a.imshow(sample_img_rot)
ax_a.set_title('(a) Example pasture image', fontweight='bold', pad=3)
ax_a.axis('off')

# Add biomass values as text overlay in bottom-left
ax_a.text(0.05, 0.05, biomass_text, transform=ax_a.transAxes,
          fontsize=5.5, color='white', fontfamily='monospace',
          verticalalignment='bottom', horizontalalignment='left',
          path_effects=[pe.withStroke(linewidth=2, foreground='black')])

# ── Panel (b): Target distributions ──────────────────────────────────
ax_b = fig.add_subplot(gs[1])

plot_data = [img_data[col].values for col in target_cols]

bp = ax_b.boxplot(plot_data, patch_artist=True, widths=0.55,
                  medianprops=dict(color='black', linewidth=1.2),
                  whiskerprops=dict(linewidth=0.8),
                  capprops=dict(linewidth=0.8),
                  flierprops=dict(markersize=3, marker='o', markerfacecolor='gray',
                                  markeredgecolor='gray', alpha=0.4),
                  showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='darkred',
                                 markeredgecolor='darkred', markersize=4))

box_colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.8)

ax_b.set_xticklabels(target_labels, fontsize=6.5)
ax_b.set_ylabel('Dry biomass (g)', fontsize=8)
ax_b.set_title('(b) Target distributions', fontweight='bold', pad=3)
ax_b.set_ylim(bottom=-3)
ax_b.yaxis.set_major_locator(mticker.MultipleLocator(50))
ax_b.yaxis.set_minor_locator(mticker.MultipleLocator(25))
ax_b.grid(axis='y', alpha=0.3, linewidth=0.4)

ax_b.text(0.02, 0.98, f'n = {len(img_data)}', transform=ax_b.transAxes,
          va='top', ha='left', fontsize=7, fontstyle='italic',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.5))

# ── Panel (c): Monthly grouped bars ──────────────────────────────────
ax_c = fig.add_subplot(gs[2])

bar_width = 20  # days
x_pos = month_dates

bottom = np.zeros(len(x_pos))
for state in ['NSW', 'Tas', 'Vic', 'WA']:
    vals = monthly_counts[state].values
    ax_c.bar(x_pos, vals, width=bar_width, bottom=bottom,
             color=state_colors[state], label=state_labels_short[state],
             edgecolor='white', linewidth=0.3, alpha=0.85, zorder=3)
    bottom += vals

# LOPO boundary lines
boundary_early = early_last + pd.Timedelta(days=1)
boundary_middle = middle_last + pd.Timedelta(days=1)

ax_c.axvline(x=boundary_early, color='#333333', linewidth=0.9,
             linestyle='--', alpha=0.7, zorder=4)
ax_c.axvline(x=boundary_middle, color='#333333', linewidth=0.9,
             linestyle='--', alpha=0.7, zorder=4)

# Period labels — Middle at its actual midpoint, Early/Late symmetric around it
ax_c.set_ylim(bottom=0)
ax_c.autoscale()
ylim = ax_c.get_ylim()
ann_y = ylim[1] * 0.97

x_min = pd.to_datetime('2015/1/1') - pd.Timedelta(days=10)
x_max = pd.to_datetime('2015/12/31')

# Middle at its original position (midpoint of middle period dates)
middle_pos = min(middle_dates) + (max(middle_dates) - min(middle_dates)) / 2
# Offset: distance from middle to nearest axis edge, scaled back slightly
offset = min((middle_pos - x_min).days, (x_max - middle_pos).days) * 0.7
early_pos = middle_pos - pd.Timedelta(days=offset)
late_pos = middle_pos + pd.Timedelta(days=offset)

period_positions = {
    'Early': early_pos,
    'Middle': middle_pos,
    'Late': late_pos,
}
for period, pos in period_positions.items():
    ax_c.text(pos, ann_y, period, ha='center', va='top', fontsize=7,
              fontstyle='italic', color='#555555')

ax_c.set_xlim(pd.to_datetime('2015/1/1') - pd.Timedelta(days=10),
              pd.to_datetime('2015/12/31'))
ax_c.set_ylabel('Number of images', fontsize=8)
ax_c.set_title('(c) Acquisition timeline', fontweight='bold', pad=3)

ax_c.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax_c.tick_params(axis='x', pad=2)
ax_c.grid(axis='y', alpha=0.3, linewidth=0.4)

legend = ax_c.legend(loc='upper left', ncol=1, framealpha=0.9,
                     edgecolor='#cccccc', columnspacing=0.8, handletextpad=0.4,
                     markerscale=0.9, fontsize=6.5, labelspacing=0.5)
legend.get_frame().set_linewidth(0.5)

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(OUTPUT, format='pdf')
fig.savefig(OUTPUT_PNG, format='png', dpi=600)
print(f'Figure saved to {OUTPUT}')
print(f'Preview saved to {OUTPUT_PNG}')
plt.close(fig)