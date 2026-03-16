# ruff: noqa: E402, I001
"""
SS Scalar Sector -- Publication Figures.

Generates 3 figures as PDF in analysis/figures/ss/:
  1. ss_zero_trajectories.pdf   -- Scalar zeros in the complex plane as xi varies
  2. ss_spectral_function.pdf   -- Im[G_s(s)] for several xi values
  3. ss_scalar_vs_tensor.pdf    -- Comparison of scalar and tensor ghost structure

Execute:  python analysis/scripts/ss_figures.py

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sct_tools.plotting import SCT_COLORS

import mpmath as mp

from scripts.ss_scalar_sector import (
    scalar_Pi_s,
    scalar_local_c2,
    scalar_spectral_function,
    find_scalar_zeros,
    scalar_residues,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = ANALYSIS_DIR / "figures" / "ss"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ANALYSIS_DIR / "results" / "ss"
RESULTS_FILE = RESULTS_DIR / "ss_scalar_sector_results.json"

# Try to set SciencePlots style, fall back gracefully
try:
    plt.style.use(["science", "ieee"])
except Exception:
    pass

# Disable LaTeX rendering if not available
try:
    import shutil
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False
except Exception:
    plt.rcParams["text.usetex"] = False

DPS = 30  # Lower dps for figure generation speed


# ===================================================================
# Figure 1: Zero trajectory plot
# ===================================================================

def figure_zero_trajectories():
    """Plot scalar zeros in the complex plane as xi varies."""
    print("Generating: ss_zero_trajectories.pdf")

    fig, ax = plt.subplots(figsize=(5, 4))

    # xi values to survey
    xi_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, 1)

    for xi in xi_values:
        c_s = float(scalar_local_c2(xi, dps=DPS))
        if c_s < 1e-10:
            continue

        zeros = find_scalar_zeros(mp.mpf(xi), R_max=80, dps=DPS)
        if not zeros:
            continue

        z_re = [e["z_re"] for e in zeros]
        z_im = [e["z_im"] for e in zeros]

        color = cmap(norm(xi))
        ax.scatter(z_re, z_im, c=[color] * len(z_re), s=20,
                   label=f"$\\xi={xi:.2f}$" if xi in [0.0, 0.25, 1.0] else None,
                   zorder=3)

    ax.axhline(y=0, color='gray', lw=0.5, ls=':')
    ax.axvline(x=0, color='gray', lw=0.5, ls=':')

    ax.set_xlabel(r"$\mathrm{Re}(z)$")
    ax.set_ylabel(r"$\mathrm{Im}(z)$")
    ax.set_title(r"Scalar ghost zeros $\Pi_s(z_n, \xi) = 0$")
    ax.legend(fontsize=7, loc="upper left")

    # Add colorbar for xi
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=r"$\xi$")

    fig.tight_layout()
    path = FIG_DIR / "ss_zero_trajectories.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===================================================================
# Figure 2: Spectral function
# ===================================================================

def figure_spectral_function():
    """Plot Im[G_s(s)] for several xi values."""
    print("Generating: ss_spectral_function.pdf")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    xi_values = [0.0, 0.1, 0.25, 1.0]
    colors = [SCT_COLORS['scalar'], SCT_COLORS['dirac'],
              SCT_COLORS['vector'], SCT_COLORS['prediction']]
    labels = [r"$\xi = 0$", r"$\xi = 0.1$", r"$\xi = 0.25$", r"$\xi = 1$"]

    s_values = np.linspace(0.05, 15, 200)

    for xi, color, label in zip(xi_values, colors, labels):
        im_vals = []
        for s in s_values:
            im_g = scalar_spectral_function(float(s), mp.mpf(xi), dps=DPS)
            im_vals.append(float(im_g))

        ax.plot(s_values, im_vals, color=color, lw=1.2, label=label)

    ax.axhline(y=0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel(r"$s = k^2/\Lambda^2$")
    ax.set_ylabel(r"$\mathrm{Im}[G_s(s + i\varepsilon)]$")
    ax.set_title(r"Scalar spectral function")
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = FIG_DIR / "ss_spectral_function.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===================================================================
# Figure 3: Scalar vs tensor ghost comparison
# ===================================================================

def figure_scalar_vs_tensor():
    """Compare scalar and tensor propagator denominators on the real axis."""
    print("Generating: ss_scalar_vs_tensor.pdf")

    from scripts.mr1_lorentzian import Pi_TT_complex

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # --- Left panel: Euclidean (positive real) axis ---
    z_values = np.linspace(0.01, 20, 300)
    pi_tt_vals = []
    pi_s_0 = []
    pi_s_025 = []
    pi_s_1 = []

    for z in z_values:
        z_mp = mp.mpc(float(z))
        pi_tt_vals.append(float(mp.re(Pi_TT_complex(z_mp, dps=DPS))))
        pi_s_0.append(float(mp.re(scalar_Pi_s(z_mp, 0.0, dps=DPS))))
        pi_s_025.append(float(mp.re(scalar_Pi_s(z_mp, 0.25, dps=DPS))))
        pi_s_1.append(float(mp.re(scalar_Pi_s(z_mp, 1.0, dps=DPS))))

    ax1.plot(z_values, pi_tt_vals, 'k-', lw=1.5, label=r'$\Pi_{TT}(z)$')
    ax1.plot(z_values, pi_s_0, color=SCT_COLORS['scalar'], lw=1.2,
             label=r'$\Pi_s(z, \xi=0)$')
    ax1.plot(z_values, pi_s_025, color=SCT_COLORS['vector'], lw=1.2, ls='--',
             label=r'$\Pi_s(z, \xi=1/4)$')
    ax1.plot(z_values, pi_s_1, color=SCT_COLORS['prediction'], lw=1.2, ls='-.',
             label=r'$\Pi_s(z, \xi=1)$')

    ax1.axhline(y=0, color='gray', lw=0.5, ls=':')
    ax1.axhline(y=1, color='gray', lw=0.3, ls=':')
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$\Pi(z)$')
    ax1.set_title('Euclidean axis')
    ax1.legend(fontsize=6, loc='lower left')
    ax1.set_ylim(-5, 2)

    # --- Right panel: c_s vs xi ---
    xi_range = np.linspace(0, 1.5, 200)
    c_s_vals = [6 * (xi - 1.0 / 6) ** 2 for xi in xi_range]
    c_tt = 13.0 / 60.0  # TT coefficient

    ax2.plot(xi_range, c_s_vals, color=SCT_COLORS['scalar'], lw=1.5,
             label=r'$c_2^{(s)}(\xi) = 6(\xi - 1/6)^2$')
    ax2.axhline(y=c_tt, color='k', lw=1.0, ls='--',
                label=r'$c_2^{(TT)} = 13/60$')
    ax2.axvline(x=1.0 / 6, color='gray', lw=0.5, ls=':',
                label=r'$\xi = 1/6$ (conformal)')

    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'$c_2$')
    ax2.set_title(r'Local coefficients')
    ax2.legend(fontsize=6)
    ax2.set_ylim(-0.1, 5)

    fig.tight_layout()
    path = FIG_DIR / "ss_scalar_vs_tensor.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("SS Scalar Sector -- Publication Figures")
    print("=" * 60)

    paths = []
    paths.append(figure_zero_trajectories())
    paths.append(figure_spectral_function())
    paths.append(figure_scalar_vs_tensor())

    print(f"\nAll {len(paths)} figures generated successfully.")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
