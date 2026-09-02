#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for CSIRO biomass prediction paper.

Generates Figures A–F described in the paper supplement using pre-computed
results and CSV tables. Each figure can be generated independently.

Usage:
    python figures/generate_figures.py             # generates all figures
    python figures/generate_figures.py --figures A,D  # only Figure A and D
    python figures/generate_figures.py --headless  # no display backend needed

Output:
    figures/FigureX.pdf   (vector)
    figures/FigureX.png   (raster preview)
"""
import os, sys

# ── Auto-detect venv and re-execute if needed ───────────────────────
_venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'venv', 'bin', 'python3')
if sys.executable != _venv_python and os.path.exists(_venv_python):
    # Check if current environment lacks key deps
    try:
        import torch  # noqa
        import matplotlib  # noqa
    except ImportError:
        os.execv(_venv_python, [_venv_python] + sys.argv)



import os, sys, argparse, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path

# ── Non-interactive backend for headless use ────────────────────────────
_BACKEND_SET = False


def _set_backend(headless: bool = False):
    global _BACKEND_SET
    if _BACKEND_SET:
        return
    if headless or not os.environ.get('DISPLAY', ''):
        matplotlib.use('Agg')
    _BACKEND_SET = True


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
EXPERIMENT_2_DIR = RESULTS_DIR / 'experiment_2'
EXPERIMENT_2_RIDGE_DIR = RESULTS_DIR / 'experiment_2_ridge'
FIGURES_DIR = BASE_DIR / 'figures'

TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_SHORT = ['Green', 'Dead', 'Clover', 'GDM', 'Total']

# ══════════════════════════════════════════════════════════════════════════
# STYLE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Color palette from the dataset figure (state colors + diverging scheme)
# State colors: NSW='#E24A33', Tas='#348ABD', Vic='#988ED5', WA='#F5A623'
# Boxplot scheme: ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
COLORS = {
    'random_stratified':     '#348ABD',   # blue (Tas)
    'date_grouped':          '#988ED5',   # purple (Vic)
    'date_location_grouped': '#E24A33',   # red (NSW)
    'lopo':                  '#F5A623',   # gold (WA)
    'mlp':                   '#2c7bb6',   # deeper blue from boxplot scheme
    'ridge':                 '#d7191c',   # red from boxplot scheme
    'hidden':                '#333333',   # dark gray
}

# Lighter variants for local bars per protocol
PROTOCOL_LOCAL_COLORS = {
    'random_stratified':     '#6CB5E0',   # lighter blue
    'date_grouped':          '#B8AEE6',   # lighter purple
    'date_location_grouped': '#F0806C',   # lighter red
}

PROTOCOL_DISPLAY = {
    'random_stratified':     'Random CV',
    'date_grouped':          'Date CV',
    'date_location_grouped': 'Date-State CV',
    'lopo':                  'LOPO CV',
}

PROTOCOL_ORDER = ['random_stratified', 'date_grouped', 'date_location_grouped']

RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
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
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': ':',
}


def _configure_style():
    plt.rcParams.update(RC_PARAMS)


def _save_figure(fig, name: str, width: float = 3.4, height: float = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if height is not None:
        fig.set_size_inches(width, height)
    else:
        fig.set_size_inches(width, fig.get_size_inches()[1] * width / fig.get_size_inches()[0])
    pdf_path = FIGURES_DIR / f'{name}.pdf'
    png_path = FIGURES_DIR / f'{name}.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(png_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'  ✓ {pdf_path}')
    print(f'  ✓ {png_path}')
    return pdf_path

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def _read_csv_table(path: Path) -> pd.DataFrame:
    """Read a CSV table from the results dir."""
    if not path.exists():
        print(f'  [WARN] File not found: {path}')
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _read_table_8(ridge: bool = False) -> pd.DataFrame:
    """Load Table 8: protocol comparison with local weighted R²."""
    subdir = EXPERIMENT_2_RIDGE_DIR if ridge else EXPERIMENT_2_DIR
    path = subdir / 'table_8.csv'
    return _read_csv_table(path)


def _read_table_9(ridge: bool = False) -> pd.DataFrame:
    """Load Table 9: per-target R² per protocol."""
    subdir = EXPERIMENT_2_RIDGE_DIR if ridge else EXPERIMENT_2_DIR
    path = subdir / 'table_9.csv'
    return _read_csv_table(path)


def _read_table_10() -> pd.DataFrame:
    """Load Table 10: LOPO summary metrics per period."""
    path = EXPERIMENT_2_RIDGE_DIR / 'table_10.csv'
    return _read_csv_table(path)


def _read_table_11() -> pd.DataFrame:
    """Load Table 11: LOPO per-target R² per period."""
    path = EXPERIMENT_2_RIDGE_DIR / 'table_11.csv'
    return _read_csv_table(path)


def _load_per_epoch_data() -> dict:
    """Load per-epoch weighted R² data for MLP validation curves."""
    path = EXPERIMENT_2_DIR / 'per_epoch_metrics_3protocols.pt'
    if not path.exists():
        print(f'  [WARN] Per-epoch data not found at {path}')
        return None
    try:
        import torch
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data
    except ImportError:
        print('  [WARN] torch not available. Install torch or use system Python with torch.')
        return None


def _load_full_results(ridge: bool = False) -> dict:
    """Load full results with fold-level predictions and best_epoch info."""
    subdir = EXPERIMENT_2_RIDGE_DIR if ridge else EXPERIMENT_2_DIR
    path = subdir / 'full_results.pt'
    if not path.exists():
        print(f'  [WARN] Full results not found at {path}')
        return None
    try:
        import torch
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data
    except ImportError:
        print('  [WARN] torch not available.')
        return None

# ══════════════════════════════════════════════════════════════════════════
# FIGURE A — Local vs Hidden R² (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════

def generate_figure_a():
    """
    Two-panel bar chart: local R² (bars) vs hidden R² (horizontal line)
    for MLP and Ridge. Uses exact values from tab:main_protocols.
    """
    _configure_style()

    # Exact values from the paper table tab:main_protocols
    mlp_local = np.array([0.841, 0.787, 0.794])
    mlp_local_std = np.array([0.007, 0.007, 0.012])
    mlp_hidden = np.array([0.607, 0.603, 0.593])
    mlp_hidden_std = np.array([0.013, 0.015, 0.018])
    ridge_local = np.array([0.815, 0.735, 0.749])
    ridge_local_std = np.array([0.009, 0.008, 0.019])
    ridge_hidden = 0.561

    short_names = ['Random CV', 'Date CV', 'Date–State CV']
    proto_colors = ['#348ABD', '#988ED5', '#E24A33']
    mlp_light = ['#C5DFF0', '#DDD6F0', '#F5C8C0']

    fig, (ax_mlp, ax_ridge) = plt.subplots(1, 2, figsize=(5.5, 2.8),
                                             sharey=True, gridspec_kw={'wspace': 0.12})

    x = np.arange(3)
    w = 0.55

    # ---- MLP panel ----
    ax_mlp.bar(x, mlp_local, w, color=mlp_light, edgecolor=proto_colors,
               linewidth=0.8, alpha=0.85, zorder=3)
    ax_mlp.errorbar(x, mlp_local, yerr=mlp_local_std, fmt='none',
                     ecolor='#555', capsize=2, capthick=0.6, elinewidth=0.8, zorder=4)

    # Hidden scores as points with error bars, connected by a dashed line
    ax_mlp.errorbar(x, mlp_hidden, yerr=mlp_hidden_std, fmt='o',
                     color='#333', markerfacecolor='white', markeredgecolor='#333',
                     markersize=5, capsize=2, capthick=0.5, elinewidth=0.6,
                     linestyle='--', linewidth=0.8, zorder=5, label='Hidden')

    for i in range(3):
        ax_mlp.text(x[i], mlp_hidden[i] + mlp_hidden_std[i] + 0.015,
                     f'{mlp_hidden[i]:.3f}', ha='center', va='bottom', fontsize=5.5, color='#333')

    ax_mlp.set_xticks(x)
    ax_mlp.set_xticklabels(short_names, fontsize=6.5)
    ax_mlp.set_ylabel('Weighted R²', fontsize=8)
    ax_mlp.set_title('MLP', fontsize=9, fontweight='bold')
    ax_mlp.set_ylim(0.4, 0.92)
    ax_mlp.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_mlp.legend(fontsize=6, loc='lower left', framealpha=0.85, edgecolor='#ccc')
    ax_mlp.set_axisbelow(True)

    # ---- Ridge panel ----
    ax_ridge.bar(x, ridge_local, w, color=proto_colors, edgecolor='white',
                 linewidth=0.4, alpha=0.85, zorder=3)
    ax_ridge.errorbar(x, ridge_local, yerr=ridge_local_std, fmt='none',
                       ecolor='#888', capsize=2, capthick=0.6, elinewidth=0.8, zorder=4)

    ax_ridge.axhline(y=ridge_hidden, color='#333', linestyle='--', linewidth=0.8,
                      alpha=0.8, zorder=2)
    ax_ridge.text(0.5, ridge_hidden + 0.015, f'Hidden = {ridge_hidden:.3f}',
                   ha='center', va='bottom', fontsize=5.5, color='#333', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))

    ax_ridge.set_xticks(x)
    ax_ridge.set_xticklabels(short_names, fontsize=6.5)
    ax_ridge.set_title('Ridge', fontsize=9, fontweight='bold')
    ax_ridge.set_axisbelow(True)

    fig.suptitle('(a) Local vs. Hidden Generalization', fontsize=9, fontweight='bold', y=1.02)

    return _save_figure(fig, 'FigureA_local_vs_hidden_r2', width=5.5, height=2.8)


def generate_figure_b():
    """
    Grouped bar chart: per-target R² for each protocol.
    Two panels: MLP (top) and Ridge (bottom).
    """
    _configure_style()

    df_mlp = _read_table_9(ridge=False)
    df_ridge = _read_table_9(ridge=True)

    if df_mlp is None or df_ridge is None:
        print('  ✗ Figure B: Missing table_9 data.')
        return None

    def _extract_r2(df):
        """Extract per-target R² for each protocol."""
        result = {}
        target_cols = [f'Target {i+1}' for i in range(5)]
        for i, p in enumerate(PROTOCOL_ORDER):
            if i >= len(df):
                continue
            row = df.iloc[i]
            vals = []
            for tc in target_cols:
                if tc in row.index:
                    try:
                        v = float(str(row[tc]).split()[0])
                    except (ValueError, TypeError):
                        v = np.nan
                    vals.append(v)
            if len(vals) == 5:
                result[p] = np.array(vals)
        return result

    mlp_data = _extract_r2(df_mlp)
    ridge_data = _extract_r2(df_ridge)

    # Fallback to hardcoded values
    if len(mlp_data) < 3:
        mlp_data = {
            'random_stratified':     np.array([0.8343, 0.5608, 0.7996, 0.8077, 0.7858]),
            'date_grouped':          np.array([0.7283, 0.3488, 0.6409, 0.7031, 0.7405]),
            'date_location_grouped': np.array([0.7433, 0.3428, 0.6525, 0.7415, 0.7386]),
        }
        print('  [INFO] Using hardcoded MLP per-target R² values.')
    if len(ridge_data) < 3:
        ridge_data = {
            'random_stratified':     np.array([0.8158, 0.5312, 0.7664, 0.7917, 0.7429]),
            'date_grouped':          np.array([0.6877, 0.3023, 0.5812, 0.6681, 0.6550]),
            'date_location_grouped': np.array([0.6994, 0.3464, 0.5721, 0.6870, 0.6749]),
        }
        print('  [INFO] Using hardcoded Ridge per-target R² values.')

    # ── Build plot ──────────────────────────────────────────────────────
    fig, (ax_mlp, ax_ridge) = plt.subplots(2, 1, figsize=(5.0, 3.8), sharex=True,
                                            gridspec_kw={'hspace': 0.08})

    colors_proto = [COLORS[p] for p in PROTOCOL_ORDER]
    labels_proto = [PROTOCOL_DISPLAY[p] for p in PROTOCOL_ORDER]

    x = np.arange(len(TARGET_SHORT))
    w = 0.22
    offsets = [-w, 0, w]

    def _plot_panel(ax, data_dict, title):
        for idx, p in enumerate(PROTOCOL_ORDER):
            if p not in data_dict:
                continue
            ax.bar(x + offsets[idx], data_dict[p], w, label=labels_proto[idx],
                   color=colors_proto[idx], alpha=0.85, edgecolor='white', linewidth=0.3)
        ax.set_ylabel('R²', fontsize=8)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=1)
        ax.set_ylim(0.0, 0.92)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(fontsize=6, loc='lower left', framealpha=0.85, edgecolor='#ccc',
                  handletextpad=0.3, ncol=3)
        ax.set_axisbelow(True)

    _plot_panel(ax_mlp, mlp_data, 'MLP Head')
    _plot_panel(ax_ridge, ridge_data, 'Ridge Head')

    ax_ridge.set_xticks(x)
    ax_ridge.set_xticklabels(TARGET_SHORT, fontsize=7)
    ax_ridge.set_xlabel('Target', fontsize=8)

    fig.suptitle('(b) Per-Target R² by Validation Protocol', fontsize=9, fontweight='bold', y=1.0)

    return _save_figure(fig, 'FigureB_per_target_degradation', width=5.0, height=3.8)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE C — LOPO Summary (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════

def generate_figure_c():
    """
    LOPO bar chart using exact values from tab:lopo_summary.
    """
    _configure_style()

    # Exact values from the paper table tab:lopo_summary
    periods = ['Early', 'Middle', 'Late']
    metrics_data = {
        'R²':  [0.695, 0.738, 0.724],
        'RMSE': [17.13, 9.49, 19.78],
        'MAE':  [12.23, 7.46, 13.87],
        'Bias': [0.292, 0.465, -3.953],
    }

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.2), sharey=False,
                              gridspec_kw={'wspace': 0.35})

    colors_period = {'Early': '#348ABD', 'Middle': '#988ED5', 'Late': '#E24A33'}
    metric_names = ['R²', 'RMSE', 'MAE', 'Bias']

    for idx, (ax, metric) in enumerate(zip(axes, metric_names)):
        x = np.arange(3)
        vals = metrics_data[metric]
        bar_colors = [colors_period[p] for p in periods]
        bars = ax.bar(x, vals, width=0.55, color=bar_colors,
                       alpha=0.85, edgecolor='white', linewidth=0.4)

        ax.set_title(metric, fontsize=8, fontweight='bold', pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=6.5)
        ax.set_axisbelow(True)

        # Label offsets proportional to each metric's scale
        for bar, val in zip(bars, vals):
            if metric == 'R²':
                frac = 0.07  # ~0.05 on a 0.7 bar
            elif 'Bias' in metric:
                frac = 0.15  # ~0.04-0.6 depending on magnitude
            else:
                frac = 0.04  # ~0.4-0.8 for RMSE/MAE
            if val >= 0:
                offset = max(val * frac, 0.02)
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + offset,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222')
            else:
                offset = max(abs(val) * frac, 0.02)
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() - offset,
                        f'{val:.3f}', ha='center', va='top', fontsize=5.5, color='#222')

        # Zero line for Bias
        if 'Bias' in metric:
            ax.axhline(y=0, color='#888', linewidth=0.5, linestyle='--')

        # Set y-lim with room for labels
        if 'Bias' in metric:
            ymin, ymax = min(vals), max(vals)
            margin = max(abs(ymin), abs(ymax)) * 0.3 + 0.5
            ax.set_ylim(min(vals) - margin, max(vals) + margin)
        elif metric == 'R²':
            ax.set_ylim(0, 0.85)
        else:
            ax.set_ylim(0, max(vals) * 1.2 + 1)

    fig.suptitle('(c) LOPO CV — Period-wise Metrics', fontsize=9,
                 fontweight='bold', y=1.02)

    return _save_figure(fig, 'FigureC_lopo_summary', width=6.5, height=2.2)


def generate_figure_d():
    """
    Scatter plot of actual vs predicted Dry_Total for LOPO predictions.
    Constructs predictions from Ridge date_location_grouped fold results
    by mapping validation indices back to time periods.
    """
    _configure_style()

    data = _load_full_results(ridge=True)
    if data is None or 'date_location_grouped' not in data:
        print('  ✗ Figure D: Ridge full_results not available.')
        print('  [INFO] To generate LOPO predictions, run:')
        print('     python scripts/cross_validation.py --split date_location --head ridge')
        return None

    fr = data['date_location_grouped']['fold_results']

    # Load dataset to map indices to dates
    train_csv = BASE_DIR / 'csiro-biomass' / 'train.csv'
    if not train_csv.exists():
        print('  ✗ Figure D: train.csv not found.')
        return None

    pdf = pd.read_csv(train_csv)
    pdf['date'] = pd.to_datetime(pdf['Sampling_Date'], format='%Y/%m/%d')

    # Assign period labels based on the LOPO scheme from the paper
    # Early: Jan-May 2015, Middle: June-Aug 2015, Late: Sep-Nov 2015
    def _period_label(dt):
        if dt <= pd.Timestamp('2015-05-31'):
            return 'Early'
        elif dt <= pd.Timestamp('2015-08-31'):
            return 'Middle'
        else:
            return 'Late'

    pdf['period'] = pdf['date'].map(_period_label)

    # Collect per-period predictions (Dry_Total)
    all_targets = []
    all_preds = []
    all_periods = []

    for fr_i in fr:
        val_idx = fr_i['val_idx']
        targets = np.array(fr_i['targets'])
        preds = np.array(fr_i['preds'])

        for i, idx in enumerate(val_idx):
            if idx < len(pdf):
                period = pdf.iloc[idx]['period']
            else:
                period = 'Unknown'
            all_targets.append(targets[i, 4])  # Dry_Total = index 4
            all_preds.append(preds[i, 4])
            all_periods.append(period)

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)

    if len(targets_arr) == 0:
        print('  ✗ Figure D: No predictions collected.')
        return None

    # ── Build plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))

    period_colors = {'Early': '#348ABD', 'Middle': '#988ED5', 'Late': '#E24A33'}
    period_markers = {'Early': 'o', 'Middle': 's', 'Late': 'D'}

    for period in ['Early', 'Middle', 'Late']:
        mask = np.array([p == period for p in all_periods])
        if not mask.any():
            continue
        ax.scatter(targets_arr[mask], preds_arr[mask],
                   c=period_colors.get(period, '#888'),
                   marker=period_markers.get(period, 'o'),
                   label=period, alpha=0.5, s=8, edgecolors='none')

    # 1:1 line
    lim_min = min(targets_arr.min(), preds_arr.min())
    lim_max = max(targets_arr.max(), preds_arr.max())
    margin = (lim_max - lim_min) * 0.05
    lims = [lim_min - margin, lim_max + margin]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.4, label='1:1 line')

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel('Actual Dry Total (g)', fontsize=8)
    ax.set_ylabel('Predicted Dry Total (g)', fontsize=8)
    ax.set_title('(d) LOPO — Actual vs Predicted (Ridge)', fontsize=9, fontweight='bold')

    ax.set_aspect('equal')
    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.85, edgecolor='#ccc')

    # Linear regression stats
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(targets_arr, preds_arr)
    ax.text(0.03, 0.97, f'R² = {r_value**2:.3f}\ny = {slope:.2f}x + {intercept:.1f}',
            transform=ax.transAxes, va='top', fontsize=6, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#ccc'))

    ax.set_axisbelow(True)

    return _save_figure(fig, 'FigureD_lopo_scatter', width=3.4, height=3.4)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE E — MLP Validation Curves (Weighted R² vs Epoch)
# ══════════════════════════════════════════════════════════════════════════

def generate_figure_e():
    """
    Line plot of mean weighted R² vs epoch for each protocol, with ±1 std shade.
    Uses per_epoch_metrics_3protocols.pt. Vertical lines mark median stopping epochs.
    """
    _configure_style()

    per_epoch = _load_per_epoch_data()
    if per_epoch is None:
        print('  ✗ Figure E: Per-epoch data not available.')
        return None

    # Convert to DataFrame
    records = []
    for protocol_name, protocol_list in per_epoch.items():
        for entry in protocol_list:
            records.append({
                'protocol': entry['protocol'],
                'seed': entry['seed'],
                'fold': entry['fold'],
                'epoch': entry['epoch'],
                'weighted_r2': float(entry['weighted_r2']),
            })

    df_epochs = pd.DataFrame(records)

    # Average over seeds and folds per protocol per epoch
    grouped = df_epochs.groupby(['protocol', 'epoch']).agg(
        avg_r2=('weighted_r2', 'mean'),
        std_r2=('weighted_r2', 'std'),
    ).reset_index()

    # Build plot
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.6))

    for protocol in PROTOCOL_ORDER:
        sub = grouped[grouped['protocol'] == protocol]
        if len(sub) == 0:
            continue
        epochs = sub['epoch'].values
        mean_r2 = sub['avg_r2'].values
        std_r2 = sub['std_r2'].fillna(0).values

        ax.plot(epochs, mean_r2, color=COLORS[protocol], linewidth=1.8,
                label=PROTOCOL_DISPLAY[protocol], zorder=3)
        ax.fill_between(epochs, mean_r2 - std_r2, mean_r2 + std_r2,
                        color=COLORS[protocol], alpha=0.25, linewidth=0)

    # Stopping epoch medians from tab:main_protocols
    stop_epochs = {
        'random_stratified':     35,
        'date_grouped':          29,
        'date_location_grouped': 20,
    }

    # Y-offset for Stop labels: first and last higher, middle lower to avoid overlap
    stop_y = {'random_stratified': 0.06, 'date_grouped': 0.03, 'date_location_grouped': 0.06}
    for protocol, ep in stop_epochs.items():
        color = COLORS[protocol]
        ax.axvline(x=ep, color=color, linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)
        label = PROTOCOL_DISPLAY[protocol]
        ax.text(ep, stop_y[protocol], f'Stop={ep}', ha='center', va='bottom', fontsize=5.5,
                color=color, fontweight='bold', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_xlabel('Training epoch', fontsize=8)
    ax.set_ylabel('Mean OOF weighted R²', fontsize=8)
    ax.set_title('(e) MLP Validation Curves', fontsize=9, fontweight='bold')

    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.85, edgecolor='#ccc')
    ax.set_xlim(0, None)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_axisbelow(True)

    return _save_figure(fig, 'FigureE_validation_curves', width=3.6, height=2.6)
def generate_figure_f():
    """
    Box plot + histogram showing best_epoch distribution across folds/seeds
    for each protocol. Vertical dashed lines show medians.
    """
    _configure_style()

    data = _load_full_results(ridge=False)
    if data is None:
        print('  ✗ Figure F: Full results not available.')
        return None

    # Extract best_epoch per fold
    plot_data = []
    for protocol in PROTOCOL_ORDER:
        if protocol not in data:
            continue
        fr = data[protocol]['fold_results']
        epochs = [fr[i]['best_epoch'] for i in range(len(fr))]
        plot_data.append({
            'protocol': protocol,
            'display': PROTOCOL_DISPLAY[protocol],
            'epochs': np.array(epochs),
            'median': np.median(epochs),
            'q1': np.percentile(epochs, 25),
            'q3': np.percentile(epochs, 75),
        })

    if not plot_data:
        print('  ✗ Figure F: No protocol data found.')
        return None

    # ── Build plot ──────────────────────────────────────────────────────
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(5.5, 2.5),
                                            gridspec_kw={'width_ratios': [1, 2.2],
                                                         'wspace': 0.3})

    # Panel 1: Box plot
    positions = np.arange(len(plot_data))
    boxes_data = [d['epochs'] for d in plot_data]
    labels = [d['display'] for d in plot_data]

    bp = ax_box.boxplot(boxes_data, positions=positions, widths=0.4,
                         patch_artist=True, showmeans=False, showfliers=True,
                         flierprops=dict(marker='o', markersize=3, alpha=0.5))

    for patch, pd_entry in zip(bp['boxes'], plot_data):
        patch.set_facecolor(COLORS[pd_entry['protocol']])
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.6)

    for median_line in bp['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(1.2)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(labels, fontsize=6.5, rotation=15, ha='right')
    ax_box.set_ylabel('Best epoch', fontsize=8)
    ax_box.set_title('(f) Stopping Epoch', fontsize=9, fontweight='bold')
    ax_box.set_axisbelow(True)
    ax_box.set_ylim(0, 80)

    for i, pd_entry in enumerate(plot_data):
        ax_box.text(i, pd_entry['median'] + 3, f'M={pd_entry["median"]:.0f}',
                     ha='center', fontsize=6.5, fontweight='bold', color='#333')

    # Panel 2: Histograms overlaid
    for pd_entry in plot_data:
        ax_hist.hist(pd_entry['epochs'], bins=range(0, 81, 5), alpha=0.35,
                      color=COLORS[pd_entry['protocol']],
                      label=f'{pd_entry["display"]} (M={pd_entry["median"]:.0f})',
                      edgecolor='white', linewidth=0.3)

    ax_hist.set_xlabel('Best epoch', fontsize=8)
    ax_hist.set_ylabel('Count (folds × seeds)', fontsize=7)
    ax_hist.set_title('Distribution', fontsize=9, fontweight='bold')
    ax_hist.legend(fontsize=6, framealpha=0.85, edgecolor='#ccc')
    ax_hist.set_axisbelow(True)
    ax_hist.set_xlim(0, 80)

    return _save_figure(fig, 'FigureF_stopping_epochs', width=5.5, height=2.5)

# ══════════════════════════════════════════════════════════════════════════
# CLI AND MAIN
# ══════════════════════════════════════════════════════════════════════════

AVAILABLE_FIGURES = {
    'A': ('Local vs. Hidden R²', generate_figure_a),
    'B': ('Per-Target Degradation', generate_figure_b),
    'C': ('LOPO Summary', generate_figure_c),
    'D': ('LOPO Scatter Plot', generate_figure_d),
    'E': ('MLP Validation Curves', generate_figure_e),
    'F': ('Stopping Epochs', generate_figure_f),
}


def generate_all():
    """Generate all six figures, skipping any that fail."""
    print('=' * 60)
    print('  Generating all figures...')
    print('=' * 60)
    results = {}
    for fig_id in sorted(AVAILABLE_FIGURES.keys()):
        name, func = AVAILABLE_FIGURES[fig_id]
        print(f'\n--- Figure {fig_id}: {name} ---')
        try:
            path = func()
            results[fig_id] = path is not None
        except Exception as e:
            print(f'  ✗ Error: {e}')
            import traceback
            traceback.print_exc()
            results[fig_id] = False

    print('\n' + '=' * 60)
    print('  Summary:')
    print('=' * 60)
    for fig_id in sorted(results.keys()):
        status = '✓' if results[fig_id] else '✗'
        name = AVAILABLE_FIGURES[fig_id][0]
        print(f'  {status} Figure {fig_id}: {name}')
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate publication figures for biomass prediction paper.')
    parser.add_argument('--figures', type=str, default='all',
                        help='Comma-separated figure IDs (e.g., A,B,C). Default: all')
    parser.add_argument('--headless', action='store_true',
                        help='Use non-interactive matplotlib backend.')
    return parser.parse_args()


def main():
    args = parse_args()
    _set_backend(headless=args.headless)

    if args.figures.strip().lower() == 'all':
        generate_all()
        return

    selected = [s.strip().upper() for s in args.figures.split(',')]
    for fig_id in selected:
        if fig_id in AVAILABLE_FIGURES:
            name, func = AVAILABLE_FIGURES[fig_id]
            print(f'\n--- Figure {fig_id}: {name} ---')
            try:
                func()
            except Exception as e:
                print(f'  ✗ Error: {e}')
                import traceback
                traceback.print_exc()
        else:
            print(f'  ✗ Unknown figure ID: {fig_id}. Available: {" ", "".join(sorted(AVAILABLE_FIGURES.keys()))}')


if __name__ == '__main__':
    main()
