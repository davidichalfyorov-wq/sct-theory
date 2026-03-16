# ruff: noqa: E402, I001
"""
MR-1 Lorentzian Continuation -- Publication Figures.

Generates 4 figures as PDF in analysis/figures/mr1/:
  1. phi_lorentzian.pdf    -- phi(-x) vs x with bounds envelope
  2. Pi_TT_lorentzian.pdf  -- Pi_TT on Lorentzian axis with ghost pole
  3. complex_zeros_map.pdf -- Complex-plane plot of all zeros
  4. stelle_comparison.pdf -- SCT vs Stelle comparison bar chart

Execute:  python analysis/scripts/mr1_figures.py
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

from sct_tools.plotting import SCT_COLORS

from scripts.mr1_lorentzian import (
    Pi_TT_lorentzian,
    phi_lorentzian_closed,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = ANALYSIS_DIR / "figures" / "mr1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPS = 30  # Lower precision for plotting speed

# Try to set SciencePlots style, fall back gracefully
try:
    plt.style.use(["science", "ieee"])
except Exception:
    pass

# Always disable usetex (requires system LaTeX) and set clean defaults
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


# ===================================================================
# Figure 1: phi(-x) with bounds envelope
# ===================================================================

def figure_phi_lorentzian():
    """phi(-x) vs x with lower bound 1 and upper bound e^{x/4}."""
    print("  Generating phi_lorentzian.pdf ...")
    mp.mp.dps = DPS

    x_vals = np.linspace(0.01, 20, 300)
    phi_vals = np.array([float(phi_lorentzian_closed(mp.mpf(x), dps=DPS)) for x in x_vals])
    upper = np.exp(x_vals / 4)
    lower = np.ones_like(x_vals)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.fill_between(x_vals, lower, upper, alpha=0.12, color=SCT_COLORS["theory_band"],
                    label=r"Bounds: $1 \leq \varphi(-x) \leq e^{x/4}$")
    ax.plot(x_vals, phi_vals, color=SCT_COLORS["total"], lw=2,
            label=r"$\varphi(-x) = e^{x/4}\sqrt{\pi/x}\,\mathrm{erf}(\sqrt{x}/2)$")
    ax.plot(x_vals, upper, "--", color=SCT_COLORS["reference"], lw=1, alpha=0.7)
    ax.plot(x_vals, lower, "--", color=SCT_COLORS["reference"], lw=1, alpha=0.7)

    ax.set_xlabel(r"$x = k^2 / \Lambda^2$")
    ax.set_ylabel(r"$\varphi(-x)$")
    ax.set_title("Lorentzian Master Function")
    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.legend(loc="upper left", frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    path = FIG_DIR / "phi_lorentzian.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Figure 2: Pi_TT on Lorentzian axis with ghost pole
# ===================================================================

def figure_Pi_TT_lorentzian():
    """Pi_TT(z_L) on the positive Lorentzian axis."""
    print("  Generating Pi_TT_lorentzian.pdf ...")
    mp.mp.dps = DPS

    z_vals = np.linspace(0.01, 5, 400)
    pi_vals = np.array([float(Pi_TT_lorentzian(mp.mpf(z), dps=DPS)) for z in z_vals])

    z_L = 1.2807022780634851
    z_0_E = 2.41484  # Euclidean ghost (on negative real Lorentzian axis, shown for reference)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(z_vals, pi_vals, color=SCT_COLORS["total"], lw=2,
            label=r"$\Pi_{\mathrm{TT}}^{\mathrm{Lor}}(z_L)$")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(z_L, color=SCT_COLORS["prediction"], lw=1.5, ls=":",
               label=fr"$z_L = {z_L:.4f}$ (ghost)")
    ax.plot(z_L, 0, "o", color=SCT_COLORS["prediction"], ms=8, zorder=5)

    # Annotations
    ax.annotate(r"$R_L = -0.538$", xy=(z_L, 0), xytext=(z_L + 0.6, 0.3),
                fontsize=9, color=SCT_COLORS["prediction"],
                arrowprops=dict(arrowstyle="->", color=SCT_COLORS["prediction"]))
    ax.annotate(r"$\Pi_{\mathrm{TT}}(0) = 1$", xy=(0.2, 1), xytext=(0.5, 0.85),
                fontsize=9, color=SCT_COLORS["reference"])

    ax.set_xlabel(r"$z_L = k^2 / \Lambda^2$ (Lorentzian)")
    ax.set_ylabel(r"$\Pi_{\mathrm{TT}}(z_L)$")
    ax.set_title("TT Propagator Denominator (Lorentzian Axis)")
    ax.set_xlim(0, 5)
    ax.set_ylim(-8, 1.2)
    ax.legend(loc="lower left", frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    path = FIG_DIR / "Pi_TT_lorentzian.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Figure 3: Complex zero map
# ===================================================================

def figure_complex_zeros_map():
    """Complex-plane plot of all Pi_TT and Pi_scalar zeros."""
    print("  Generating complex_zeros_map.pdf ...")

    # Zero data from catalogue
    # Pi_TT zeros (in Euclidean frame z_E)
    tt_zeros = [
        {"re": -1.2807, "im": 0, "label": r"$z_B$", "type": "B"},
        {"re": 2.4148, "im": 0, "label": r"$z_A$", "type": "A"},
    ]
    # Pi_scalar zeros (xi=0)
    sc_zeros = [
        {"re": -2.076, "im": 3.184, "label": r"$z_{C}$", "type": "C"},
        {"re": -2.076, "im": -3.184, "label": r"$z_{C}^*$", "type": "C"},
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Plot circles for |z| = 5, 10, 15, 20
    for R in [5, 10, 15, 20]:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(theta), R * np.sin(theta), "--", color="gray", lw=0.5, alpha=0.4)
        ax.text(R * 0.72, R * 0.72, f"|z|={R}", fontsize=7, color="gray", alpha=0.6)

    # Pi_TT zeros
    for z in tt_zeros:
        color = SCT_COLORS["prediction"] if z["type"] == "B" else SCT_COLORS["dirac"]
        marker = "s" if z["type"] == "A" else "D"
        ax.plot(z["re"], z["im"], marker, color=color, ms=10, zorder=5,
                label=f"$\\Pi_{{TT}}$ Type {z['type']}")
        ax.annotate(z["label"], xy=(z["re"], z["im"]),
                    xytext=(z["re"] + 0.5, z["im"] + 0.8),
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    # Pi_scalar zeros
    for i, z in enumerate(sc_zeros):
        label_kw = {"label": r"$\Pi_s(\xi\!=\!0)$ Type C"} if i == 0 else {}
        ax.plot(z["re"], z["im"], "^", color=SCT_COLORS["vector"], ms=10, zorder=5,
                **label_kw)
        ax.annotate(z["label"], xy=(z["re"], z["im"]),
                    xytext=(z["re"] - 2.5, z["im"] + 0.5 * (1 if z["im"] > 0 else -1)),
                    fontsize=10, color=SCT_COLORS["vector"],
                    arrowprops=dict(arrowstyle="->", color=SCT_COLORS["vector"], lw=0.8))

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.set_xlabel(r"Re$(z)$  [Euclidean frame]")
    ax.set_ylabel(r"Im$(z)$")
    ax.set_title("Complex Zero Map: Propagator Denominators")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=True, fancybox=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    path = FIG_DIR / "complex_zeros_map.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Figure 4: Stelle comparison
# ===================================================================

def figure_stelle_comparison():
    """Bar chart comparing SCT vs Stelle gravity ghost properties."""
    print("  Generating stelle_comparison.pdf ...")

    categories = [r"$z_0$ (ghost location)", r"$|R_2|$ (residue)", r"$m_{\rm ghost}/\Lambda$"]
    sct_vals = [2.415, 0.493, np.sqrt(2.415)]
    stelle_vals = [60 / 13, 1.0, np.sqrt(60 / 13)]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars1 = ax.bar(x - width / 2, stelle_vals, width,
                   label="Stelle gravity", color=SCT_COLORS["reference"], alpha=0.7)
    bars2 = ax.bar(x + width / 2, sct_vals, width,
                   label="SCT (nonlocal)", color=SCT_COLORS["prediction"], alpha=0.9)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8,
                color=SCT_COLORS["reference"])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8,
                color=SCT_COLORS["prediction"])

    # Suppression arrows
    for i in range(len(categories)):
        pct = (1 - sct_vals[i] / stelle_vals[i]) * 100
        mid_y = (stelle_vals[i] + sct_vals[i]) / 2
        ax.annotate(f"{pct:.0f}%", xy=(x[i] + 0.25, mid_y),
                    fontsize=8, color=SCT_COLORS["total"], ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("SCT vs Stelle Gravity: Ghost Properties")
    ax.legend(loc="upper left", frameon=True, fancybox=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(stelle_vals) + 1)

    path = FIG_DIR / "stelle_comparison.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("MR-1 LORENTZIAN CONTINUATION -- PUBLICATION FIGURES")
    print("=" * 60)

    paths = []
    paths.append(figure_phi_lorentzian())
    paths.append(figure_Pi_TT_lorentzian())
    paths.append(figure_complex_zeros_map())
    paths.append(figure_stelle_comparison())

    print("\n" + "=" * 60)
    print(f"Generated {len(paths)} figures in {FIG_DIR}")
    for p in paths:
        print(f"  {p.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
