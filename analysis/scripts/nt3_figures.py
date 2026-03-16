# ruff: noqa: E402, I001
"""
NT-3: Publication figures for the spectral dimension of SCT.

Generates three figures:
  1. d_S(sigma) vs sigma for SCT (4 methods) + comparison (GR, Stelle, AS)
  2. P(sigma) for SCT (ML, ASZ, propagator) vs GR heat kernel (log-log)
  3. d_S(UV) comparison bar chart: SCT vs other QG approaches

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sct_tools.plotting import init_style, save_figure, SCT_COLORS, create_figure

from scripts.nt3_spectral_dimension import (
    compute_ds,
    compute_P_heat_kernel,
    compute_P_propagator,
    compute_P_asz,
    compute_P_mittag_leffler,
    ds_stelle,
    ds_asymptotic_safety,
    ds_horava_lifshitz,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "nt3"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Figure 1: d_S(sigma) for SCT + comparisons
# ===================================================================

def figure_1_spectral_dimension():
    """d_S(sigma) vs sigma for SCT (4 methods) and comparison QG approaches."""
    init_style()

    sigma_array = np.logspace(-2, 3, 80)

    # SCT methods
    ds_ml = np.array([compute_ds(float(s), 'mittag_leffler') for s in sigma_array])
    ds_prop = np.array([compute_ds(float(s), 'propagator') for s in sigma_array])
    ds_asz = np.array([compute_ds(float(s), 'asz') for s in sigma_array])

    # Comparison
    ds_GR = np.full_like(sigma_array, 4.0)
    ds_st = np.array([ds_stelle(s) for s in sigma_array])
    ds_as = np.array([ds_asymptotic_safety(s) for s in sigma_array])
    ds_hl = np.array([ds_horava_lifshitz(s) for s in sigma_array])

    fig, ax = create_figure(figsize=(7.0, 4.5))

    # SCT curves
    ax.plot(sigma_array, ds_ml, '-', color=SCT_COLORS['total'], linewidth=2.0,
            label='SCT (ML poles)', zorder=10)
    ax.plot(sigma_array, ds_asz, '--', color=SCT_COLORS['prediction'], linewidth=1.5,
            label='SCT (ASZ/fakeon)')
    ax.plot(sigma_array, ds_prop, ':', color=SCT_COLORS['scalar'], linewidth=1.5,
            label='SCT (propagator)')

    # Comparison curves
    ax.plot(sigma_array, ds_GR, '-', color=SCT_COLORS['reference'], linewidth=1.0,
            alpha=0.6, label='GR ($d_S=4$)')
    ax.plot(sigma_array, ds_st, '--', color='#E91E63', linewidth=1.2,
            alpha=0.7, label='Stelle')
    ax.plot(sigma_array, ds_as, '-.', color='#00BCD4', linewidth=1.2,
            alpha=0.7, label='Asympt. Safety')
    ax.plot(sigma_array, ds_hl, ':', color='#795548', linewidth=1.2,
            alpha=0.7, label=r'Ho\v{r}ava-Lifshitz')

    # Reference lines
    ax.axhline(y=4.0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axhline(y=2.0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axhline(y=0.0, color='gray', linewidth=0.5, alpha=0.3)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma$ [$\Lambda^{-2}$]')
    ax.set_ylabel(r'$d_S(\sigma)$')
    ax.set_title('Spectral dimension: SCT vs other QG approaches')
    ax.set_ylim(-1.0, 5.5)
    ax.set_xlim(1e-2, 1e3)
    ax.legend(loc='center right', fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    out = save_figure(fig, "nt3_spectral_dimension", fmt="pdf",
                      directory=str(FIGURES_DIR))
    plt.close(fig)
    print(f"Figure 1 saved: {out}")
    return out


# ===================================================================
# Figure 2: P(sigma) log-log plot
# ===================================================================

def figure_2_return_probability():
    """P(sigma) for SCT (ML, ASZ, propagator) vs GR heat kernel."""
    init_style()

    sigma_array = np.logspace(-1.5, 2.5, 60)

    P_hk = np.array([compute_P_heat_kernel(s) for s in sigma_array])
    P_ml = np.array([compute_P_mittag_leffler(float(s)) for s in sigma_array])
    P_prop = np.array([compute_P_propagator(float(s)) for s in sigma_array])
    P_asz = np.array([compute_P_asz(float(s)) for s in sigma_array])

    fig, ax = create_figure(figsize=(7.0, 4.5))

    ax.loglog(sigma_array, P_hk, '-', color=SCT_COLORS['reference'],
              linewidth=1.2, label='GR heat kernel', alpha=0.7)
    ax.loglog(sigma_array, np.abs(P_ml), '-', color=SCT_COLORS['total'],
              linewidth=2.0, label='SCT (ML poles)')
    ax.loglog(sigma_array, P_prop, ':', color=SCT_COLORS['scalar'],
              linewidth=1.5, label='SCT (propagator)')
    ax.loglog(sigma_array, P_asz, '--', color=SCT_COLORS['prediction'],
              linewidth=1.5, label='SCT (ASZ/fakeon)')

    # Reference slope: sigma^{-2} (d_S = 4)
    s_ref = np.logspace(-1, 2.5, 50)
    P_ref = 0.5e-2 * s_ref**(-2)
    ax.loglog(s_ref, P_ref, '--', color='gray', linewidth=0.8, alpha=0.4,
              label=r'$\propto \sigma^{-2}$ ($d_S=4$)')

    ax.set_xlabel(r'$\sigma$ [$\Lambda^{-2}$]')
    ax.set_ylabel(r'$P(\sigma)$')
    ax.set_title('Return probability: SCT methods vs GR')
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out = save_figure(fig, "nt3_return_probability", fmt="pdf",
                      directory=str(FIGURES_DIR))
    plt.close(fig)
    print(f"Figure 2 saved: {out}")
    return out


# ===================================================================
# Figure 3: UV bar chart
# ===================================================================

def figure_3_uv_comparison():
    """d_S(UV) comparison bar chart for different QG approaches."""
    init_style()

    approaches = [
        'GR', 'Stelle', 'AS', 'HL', 'CDT',
        'SCT\n(ML)', 'SCT\n(ASZ)', 'SCT\n(prop)',
    ]
    uv_values = [
        4.0,    # GR
        2.0,    # Stelle
        2.0,    # AS
        2.0,    # HL (z=3)
        1.80,   # CDT (Monte Carlo)
        4.0,    # SCT ML
        0.0,    # SCT ASZ
        2.0,    # SCT propagator
    ]

    colors = [
        SCT_COLORS['reference'],  # GR
        '#E91E63',                # Stelle
        '#00BCD4',                # AS
        '#795548',                # HL
        '#FF5722',                # CDT
        SCT_COLORS['total'],      # SCT ML
        SCT_COLORS['prediction'], # SCT ASZ
        SCT_COLORS['scalar'],     # SCT prop
    ]

    fig, ax = create_figure(figsize=(7.0, 4.0))

    bars = ax.bar(approaches, uv_values, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)

    # Annotate values
    for bar, val in zip(bars, uv_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=2.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5,
               label=r'$d_S = 2$ (``universality'')')
    ax.axhline(y=4.0, color='gray', linewidth=0.8, linestyle='-', alpha=0.3,
               label=r'$d_S = 4$ (classical)')

    ax.set_ylabel(r'$d_S(\mathrm{UV})$')
    ax.set_title('UV spectral dimension across quantum gravity approaches')
    ax.set_ylim(-0.5, 5.5)
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    out = save_figure(fig, "nt3_uv_comparison", fmt="pdf",
                      directory=str(FIGURES_DIR))
    plt.close(fig)
    print(f"Figure 3 saved: {out}")
    return out


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NT-3 Figures: Generating publication-quality plots")
    print("=" * 70)

    f1 = figure_1_spectral_dimension()
    f2 = figure_2_return_probability()
    f3 = figure_3_uv_comparison()

    print("\n" + "=" * 70)
    print("All 3 figures generated successfully.")
    print("=" * 70)
