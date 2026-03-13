"""
SCT Theory — Publication-Quality Plotting Defaults.

Provides standardized plotting functions and styles for all SCT figures,
ensuring consistency across publications, talks, and internal reports.

Uses SciencePlots for journal-standard formatting (PRL, PRD, JHEP compatible).

IMPORTANT: Always `import scienceplots` before using plt.style.use('science').

Usage:
    from sct_tools.plotting import init_style, plot_form_factors, save_figure

    init_style()  # sets publication defaults
    fig, ax = plot_form_factors(x_range=(0.01, 100))
    save_figure(fig, "form_factors_combined", fmt="pdf")
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import scienceplots to register styles
try:
    import scienceplots  # noqa: F401
    _SCIENCEPLOTS_OK = True
except ImportError:
    _SCIENCEPLOTS_OK = False
    warnings.warn(
        "SciencePlots not available. Install: pip install SciencePlots. "
        "Falling back to matplotlib defaults.",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_FIGURES_DIR = _PROJECT_DIR / "analysis" / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SCT color palette
# ---------------------------------------------------------------------------
SCT_COLORS = {
    'scalar': '#2196F3',      # Blue — spin 0
    'dirac': '#F44336',       # Red — spin 1/2
    'vector': '#4CAF50',      # Green — spin 1
    'total': '#000000',       # Black — SM total
    'prediction': '#FF9800',  # Orange — SCT predictions
    'data': '#9C27B0',        # Purple — experimental data
    'theory_band': '#E3F2FD', # Light blue — theory uncertainty band
    'reference': '#757575',   # Gray — reference/comparison
}


# ---------------------------------------------------------------------------
# Style initialization
# ---------------------------------------------------------------------------
def init_style(style='science', usetex=False, font_size=10, dpi=300):
    """Initialize publication-quality plotting style.

    Parameters:
        style: matplotlib style name (default: 'science' from SciencePlots)
        usetex: use LaTeX for text rendering (requires LaTeX installation)
        font_size: base font size in points
        dpi: figure resolution for raster formats
    """
    if _SCIENCEPLOTS_OK and style in plt.style.available:
        # Use science + high-vis for good contrast
        styles_to_use = [style]
        if 'high-vis' in plt.style.available:
            styles_to_use.append('high-vis')
        plt.style.use(styles_to_use)
    else:
        # Fallback: clean matplotlib defaults
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })

    # Override with SCT-specific settings
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size + 1,
        'axes.titlesize': font_size + 2,
        'legend.fontsize': font_size - 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'text.usetex': usetex,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
    })

    if usetex:
        plt.rcParams.update({
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
            'font.family': 'serif',
        })


# ---------------------------------------------------------------------------
# Figure creation helpers
# ---------------------------------------------------------------------------
def create_figure(nrows=1, ncols=1, figsize=None, squeeze=True):
    """Create a figure with SCT defaults.

    Parameters:
        nrows, ncols: subplot grid
        figsize: (width, height) in inches. Default: single-column (3.4")
            or double-column (7") based on ncols.
        squeeze: if True, return axes directly when nrows=ncols=1

    Returns:
        (fig, axes) tuple
    """
    if figsize is None:
        if ncols == 1:
            figsize = (3.4, 2.8)  # Single-column (PRL/PRD)
        else:
            figsize = (7.0, 2.8)  # Double-column

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=squeeze)
    return fig, axes


def save_figure(fig, name, fmt="pdf", directory=None):
    """Save figure in publication-quality format.

    Parameters:
        fig: matplotlib Figure
        name: filename (without extension)
        fmt: format ('pdf', 'svg', 'png', 'eps')
        directory: save directory (default: analysis/figures/)
    """
    if directory is None:
        directory = _FIGURES_DIR
    else:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

    filepath = directory / f"{name}.{fmt}"
    fig.savefig(filepath, format=fmt, bbox_inches='tight')
    print(f"Figure saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# SCT-specific plot templates
# ---------------------------------------------------------------------------
def plot_form_factors(x_range=(0.01, 100), n_points=500, xi=0.0,
                     show_total=True, figsize=None):
    """Plot all form factors h_C and h_R for all spins.

    Parameters:
        x_range: (x_min, x_max) range
        n_points: number of evaluation points
        xi: scalar coupling constant
        show_total: overlay SM total F_1, F_2
        figsize: figure size

    Returns:
        (fig, (ax_C, ax_R)) tuple
    """
    from . import form_factors as ff

    init_style()
    fig, (ax_C, ax_R) = create_figure(1, 2, figsize=figsize or (7.0, 3.0))

    x = np.linspace(x_range[0], x_range[1], n_points)

    # h_C (Weyl tensor form factors)
    ax_C.plot(x, [ff.hC_scalar_fast(xi_) for xi_ in x],
              color=SCT_COLORS['scalar'], label=r'$h_C^{(0)}$ (scalar)')
    ax_C.plot(x, [ff.hC_dirac_fast(xi_) for xi_ in x],
              color=SCT_COLORS['dirac'], label=r'$h_C^{(1/2)}$ (Dirac)')
    ax_C.plot(x, [ff.hC_vector_fast(xi_) for xi_ in x],
              color=SCT_COLORS['vector'], label=r'$h_C^{(1)}$ (vector)')

    if show_total:
        ax_C.plot(x, [ff.F1_total(xi_, xi=xi) for xi_ in x],
                  color=SCT_COLORS['total'], ls='--', lw=2,
                  label=r'$F_1^{\mathrm{SM}}$ (total)')

    ax_C.set_xlabel(r'$x = -\nabla^2/\Lambda^2$')
    ax_C.set_ylabel(r'$h_C(x)$')
    ax_C.set_title('Weyl form factors')
    ax_C.legend(fontsize=7)
    ax_C.axhline(y=0, color='gray', lw=0.5, ls=':')

    # h_R (Ricci scalar form factors)
    ax_R.plot(x, [ff.hR_scalar_fast(xi_, xi=xi) for xi_ in x],
              color=SCT_COLORS['scalar'], label=r'$h_R^{(0)}$ (scalar)')
    ax_R.plot(x, [ff.hR_dirac_fast(xi_) for xi_ in x],
              color=SCT_COLORS['dirac'], label=r'$h_R^{(1/2)}$ (Dirac)')
    ax_R.plot(x, [ff.hR_vector_fast(xi_) for xi_ in x],
              color=SCT_COLORS['vector'], label=r'$h_R^{(1)}$ (vector)')

    if show_total:
        ax_R.plot(x, [ff.F2_total(xi_, xi=xi) for xi_ in x],
                  color=SCT_COLORS['total'], ls='--', lw=2,
                  label=r'$F_2^{\mathrm{SM}}$ (total)')

    ax_R.set_xlabel(r'$x = -\nabla^2/\Lambda^2$')
    ax_R.set_ylabel(r'$h_R(x)$')
    ax_R.set_title('Ricci form factors')
    ax_R.legend(fontsize=7)
    ax_R.axhline(y=0, color='gray', lw=0.5, ls=':')

    fig.tight_layout()
    return fig, (ax_C, ax_R)


def plot_spectral_dimension(x_range=(0.01, 100), n_points=200,
                            figsize=None):
    """Plot spectral dimension d_S(l) for SCT.

    Returns:
        (fig, ax) tuple
    """
    init_style()
    fig, ax = create_figure(figsize=figsize or (3.4, 2.8))

    # Placeholder — will be filled by NT-3
    x = np.linspace(x_range[0], x_range[1], n_points)
    # d_S = 4 at large scales, flows to 2 at UV
    d_S = 4 - 2 * np.exp(-x / 10)

    ax.plot(x, d_S, color=SCT_COLORS['prediction'], lw=2)
    ax.axhline(y=4, color='gray', ls=':', lw=0.5, label=r'$d_S = 4$ (IR)')
    ax.axhline(y=2, color='gray', ls='--', lw=0.5, label=r'$d_S = 2$ (UV)')
    ax.set_xlabel(r'$\ell / \ell_P$')
    ax.set_ylabel(r'$d_S(\ell)$')
    ax.set_title('Spectral dimension flow')
    ax.set_xscale('log')
    ax.legend()

    return fig, ax


def plot_residuals(x_data, y_data, y_model, yerr=None, figsize=None,
                   label=""):
    """Plot residuals (data - model) / error.

    Standard diagnostic plot for theory-data comparison.

    Returns:
        (fig, ax) tuple
    """
    init_style()
    fig, ax = create_figure(figsize=figsize or (3.4, 2.0))

    residuals = np.array(y_data) - np.array(y_model)
    if yerr is not None:
        residuals = residuals / np.array(yerr)
        ylabel = r'$(y_{\mathrm{data}} - y_{\mathrm{model}}) / \sigma$'
    else:
        ylabel = r'$y_{\mathrm{data}} - y_{\mathrm{model}}$'

    ax.errorbar(x_data, residuals, yerr=1 if yerr is not None else None,
                fmt='o', ms=3, color=SCT_COLORS['data'], label=label)
    ax.axhline(y=0, color='gray', lw=1)
    if yerr is not None:
        ax.fill_between(ax.get_xlim(), -2, 2, alpha=0.1, color='green',
                        label=r'$\pm 2\sigma$')
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax


def plot_comparison_table(labels, sct_values, competitor_values,
                          data_values, data_errors=None,
                          title="Theory comparison", figsize=None):
    """Bar chart comparing SCT predictions vs competitors vs data.

    Parameters:
        labels: list of observable names
        sct_values: SCT predictions
        competitor_values: dict of {theory_name: [values]}
        data_values: measured values
        data_errors: measurement uncertainties

    Returns:
        (fig, ax) tuple
    """
    init_style()
    n = len(labels)
    fig, ax = create_figure(figsize=figsize or (7.0, 3.0))

    x = np.arange(n)
    width = 0.25
    offset = 0

    # SCT
    ax.bar(x + offset * width, sct_values, width, label='SCT',
           color=SCT_COLORS['prediction'], alpha=0.8)
    offset += 1

    # Competitors
    colors = ['#2196F3', '#4CAF50', '#9C27B0']
    for i, (name, vals) in enumerate(competitor_values.items()):
        ax.bar(x + offset * width, vals, width, label=name,
               color=colors[i % len(colors)], alpha=0.6)
        offset += 1

    # Data
    if data_errors is not None:
        ax.errorbar(x + offset * width, data_values, yerr=data_errors,
                    fmt='ko', ms=5, capsize=3, label='Data')
    else:
        ax.bar(x + offset * width, data_values, width, label='Data',
               color='black', alpha=0.5)

    ax.set_xticks(x + width * (offset - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def annotate_theory_version(ax, version="SCT v0.6", date=None):
    """Add theory version watermark to plot."""
    if date is None:
        from datetime import date as dt
        date = dt.today().isoformat()
    ax.text(0.02, 0.02, f"{version} [{date}]",
            transform=ax.transAxes, fontsize=6,
            color='gray', alpha=0.5, va='bottom')


def annotate_prediction(ax, x, y, text, fontsize=8):
    """Add an annotation arrow pointing to a prediction."""
    ax.annotate(text, xy=(x, y), xytext=(x * 1.5, y * 1.2),
                fontsize=fontsize,
                arrowprops=dict(arrowstyle='->', color=SCT_COLORS['prediction']),
                color=SCT_COLORS['prediction'])
