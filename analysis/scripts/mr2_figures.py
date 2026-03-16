# ruff: noqa: E402, I001
"""
MR-2 Unitarity and Stability -- Publication Figures.

Generates 4 figures as PDF in analysis/figures/mr2/:
  1. spectral_function.pdf        -- Delta-function spectral structure
  2. complex_zeros_extended.pdf   -- All 8 zeros in the complex z-plane
  3. Pi_TT_complex.pdf            -- 2D color map of |Pi_TT(z)| in complex plane
  4. sum_rule_convergence.pdf     -- Partial sum convergence toward -6/83

Execute:  python analysis/scripts/mr2_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
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
from matplotlib.patches import Circle

from sct_tools.plotting import SCT_COLORS

from scripts.mr1_lorentzian import Pi_TT_complex

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = ANALYSIS_DIR / "figures" / "mr2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPS = 30  # Lower precision for plotting speed

# Try to set SciencePlots style, fall back gracefully
try:
    plt.style.use(["science", "ieee"])
except Exception:
    pass

# Always disable usetex and set clean defaults
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Reference data (from D/D-R consensus)
# ---------------------------------------------------------------------------
# All known zeros within |z| <= 100
ZEROS = [
    {"re": -1.28070, "im": 0.0, "type": "B", "label": "$z_L$", "R_abs": 0.538},
    {"re": 2.41484, "im": 0.0, "type": "A", "label": "$z_0$", "R_abs": 0.493},
    {"re": 6.0511, "im": 33.290, "type": "C", "label": None, "R_abs": 0.00856},
    {"re": 6.0511, "im": -33.290, "type": "C", "label": None, "R_abs": 0.00856},
    {"re": 7.1436, "im": 58.931, "type": "C", "label": None, "R_abs": 0.00488},
    {"re": 7.1436, "im": -58.931, "type": "C", "label": None, "R_abs": 0.00488},
    {"re": 7.8417, "im": 84.274, "type": "C", "label": None, "R_abs": 0.00342},
    {"re": 7.8417, "im": -84.274, "type": "C", "label": None, "R_abs": 0.00342},
]

# Partial sums (graviton +1, then adding ghost residues in order of |z|)
# From D-R report: sum converges toward -6/83 ~ -0.0723
PARTIAL_SUMS = [
    (0.0, 1.0, "graviton"),           # graviton at z=0, R=+1
    (1.28, 1.0 - 0.538, "+$z_L$"),     # + Lorentzian ghost
    (2.41, 1.0 - 0.538 - 0.493, "+$z_0$"),  # + Euclidean ghost
    (33.84, -0.0309 - 2 * 0.00101, "+C#1"),
    (59.36, -0.0329 - 2 * 0.000425, "+C#2"),
    (84.64, -0.0338 - 2 * 0.000238, "+C#3"),
]


# ===================================================================
# Figure 1: Spectral function (delta-function structure)
# ===================================================================

def figure_spectral_function():
    """Plot the delta-function spectral structure rho_TT(sigma)."""
    print("  Generating spectral_function.pdf ...")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Graviton at sigma = 0
    ax.annotate(
        "", xy=(0, 1.0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=SCT_COLORS['vector'], lw=2.0),
    )
    ax.text(0.05, 0.85, "graviton\n$R=+1$", fontsize=8, color=SCT_COLORS['vector'],
            ha='left', va='top')

    # Lorentzian ghost at sigma = 1.2807 Lambda^2
    sigma_L = 1.2807
    ax.annotate(
        "", xy=(sigma_L, -0.538), xytext=(sigma_L, 0),
        arrowprops=dict(arrowstyle="-|>", color=SCT_COLORS['dirac'], lw=2.0),
    )
    ax.text(sigma_L + 0.1, -0.45, f"ghost ($z_L$)\n$R_L = -0.538$",
            fontsize=8, color=SCT_COLORS['dirac'], ha='left', va='top')

    # Euclidean ghost note (spacelike, not on this axis)
    ax.text(3.5, 0.5,
            "Euclidean ghost at $z_0 = 2.41$\n(spacelike, $k^2 < 0$,\nnot in physical spectrum)",
            fontsize=7, color=SCT_COLORS['reference'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#cccccc'))

    # Zero line
    ax.axhline(0, color='gray', lw=0.5, ls='-')

    # Styling
    ax.set_xlabel(r"$\sigma / \Lambda^2$")
    ax.set_ylabel(r"$\rho_{\mathrm{TT}}(\sigma)$  [arb. units]")
    ax.set_title("SCT spectral function (delta-function structure)")
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.8, 1.3)
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0])

    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectral_function.pdf")
    plt.close(fig)
    print("    -> spectral_function.pdf")


# ===================================================================
# Figure 2: Complex zeros (extended to |z| <= 100)
# ===================================================================

def figure_complex_zeros_extended():
    """All 8 zeros in the complex z-plane with markers by type."""
    print("  Generating complex_zeros_extended.pdf ...")

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Type-specific markers and colors
    markers = {"A": "o", "B": "s", "C": "D"}
    colors = {"A": SCT_COLORS['prediction'], "B": SCT_COLORS['dirac'], "C": SCT_COLORS['vector']}
    labels_done = set()

    for z in ZEROS:
        t = z["type"]
        label = None
        if t not in labels_done:
            label_map = {"A": "Type A (Euclidean real)", "B": "Type B (Lorentzian real)",
                         "C": "Type C (complex pair)"}
            label = label_map[t]
            labels_done.add(t)

        ax.plot(z["re"], z["im"], marker=markers[t], color=colors[t],
                markersize=8 if t != "C" else 6, label=label,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)

        if z["label"]:
            offset = (10, 10) if z["im"] >= 0 else (10, -15)
            ax.annotate(z["label"], (z["re"], z["im"]),
                        textcoords="offset points", xytext=offset,
                        fontsize=9, color=colors[t])

    # Graviton at origin
    ax.plot(0, 0, marker="*", color=SCT_COLORS['total'], markersize=12,
            label="graviton ($z=0$)", zorder=6,
            markeredgecolor='black', markeredgewidth=0.5)

    # Circles showing MR-1 range and full range
    circle_20 = Circle((0, 0), 20, fill=False, ls='--', color='gray', lw=0.8)
    circle_100 = Circle((0, 0), 100, fill=False, ls=':', color='gray', lw=0.8)
    ax.add_patch(circle_20)
    ax.add_patch(circle_100)
    ax.text(14, -14, "$|z|=20$\n(MR-1)", fontsize=7, color='gray', ha='center')
    ax.text(70, -70, "$|z|=100$", fontsize=7, color='gray', ha='center')

    # Real axis
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)

    ax.set_xlabel("Re($z$)")
    ax.set_ylabel("Im($z$)")
    ax.set_title("Zeros of $\\Pi_{TT}(z)$ in the complex plane ($|z| \\leq 100$)")
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "complex_zeros_extended.pdf")
    plt.close(fig)
    print("    -> complex_zeros_extended.pdf")


# ===================================================================
# Figure 3: |Pi_TT(z)| in the complex plane
# ===================================================================

def figure_Pi_TT_complex():
    """2D color map of |Pi_TT(z)| in the complex plane."""
    print("  Generating Pi_TT_complex.pdf ...")
    mp.mp.dps = DPS

    # Grid: real from -5 to 10, imag from -40 to 40
    n_re, n_im = 200, 200
    re_vals = np.linspace(-5, 10, n_re)
    im_vals = np.linspace(-40, 40, n_im)
    RE, IM = np.meshgrid(re_vals, im_vals)
    Z_abs = np.zeros_like(RE)

    for i in range(n_im):
        for j in range(n_re):
            z = mp.mpc(re_vals[j], im_vals[i])
            val = Pi_TT_complex(z, dps=DPS)
            Z_abs[i, j] = float(abs(val))

    fig, ax = plt.subplots(figsize=(6, 5))

    # Log scale color map
    Z_log = np.log10(Z_abs + 1e-20)

    im = ax.pcolormesh(RE, IM, Z_log, cmap='viridis', shading='auto',
                       vmin=-2, vmax=2)
    cbar = fig.colorbar(im, ax=ax, label=r"$\log_{10}|\Pi_{TT}(z)|$")

    # Mark zeros
    for z in ZEROS:
        if abs(z["im"]) <= 40 and -5 <= z["re"] <= 10:
            ax.plot(z["re"], z["im"], 'w+', markersize=10, markeredgewidth=2)

    ax.plot(0, 0, 'w*', markersize=10)

    ax.set_xlabel("Re($z$)")
    ax.set_ylabel("Im($z$)")
    ax.set_title("$|\\Pi_{TT}(z)|$ in the complex plane")
    ax.axhline(0, color='white', lw=0.3, alpha=0.5)
    ax.axvline(0, color='white', lw=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Pi_TT_complex.pdf")
    plt.close(fig)
    print("    -> Pi_TT_complex.pdf")


# ===================================================================
# Figure 4: Sum rule convergence
# ===================================================================

def figure_sum_rule_convergence():
    """Plot showing how the partial sum of residues converges to -6/83."""
    print("  Generating sum_rule_convergence.pdf ...")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Data: |z| of each zero added, cumulative sum of Re(residues)
    z_abs_vals = [p[0] for p in PARTIAL_SUMS]
    cum_sums = [p[1] for p in PARTIAL_SUMS]
    labels = [p[2] for p in PARTIAL_SUMS]

    ax.plot(z_abs_vals, cum_sums, 'o-', color=SCT_COLORS['prediction'],
            markersize=6, lw=1.5, label="partial sum of residues")

    # Target line: -6/83
    target = -6.0 / 83
    ax.axhline(target, color=SCT_COLORS['dirac'], ls='--', lw=1.0,
               label=f"$-6/83 = {target:.4f}$")

    # Zero line
    ax.axhline(0, color='gray', ls='-', lw=0.3)

    # Label key points
    for i, (x, y, lbl) in enumerate(PARTIAL_SUMS):
        if i < 3:  # Only label first few for clarity
            offset_y = 0.05 if i == 0 else -0.04
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(8, 10 if i == 0 else -12),
                        fontsize=7, color=SCT_COLORS['reference'])

    ax.set_xlabel("$|z|$ of included zeros")
    ax.set_ylabel("$1 + \\sum_i R_i$")
    ax.set_title("Residue sum rule convergence")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-5, 100)
    ax.set_ylim(-0.12, 1.1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sum_rule_convergence.pdf")
    plt.close(fig)
    print("    -> sum_rule_convergence.pdf")


# ===================================================================
# Main
# ===================================================================

def main():
    print("MR-2 Unitarity and Stability -- Generating Figures")
    print("=" * 55)
    print(f"Output directory: {FIG_DIR}")
    print()

    figure_spectral_function()
    figure_complex_zeros_extended()
    figure_Pi_TT_complex()
    figure_sum_rule_convergence()

    print()
    print("All figures generated successfully.")
    print(f"Files in: {FIG_DIR}")


if __name__ == "__main__":
    main()
