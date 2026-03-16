# ruff: noqa: E402, I001
"""
KK-D: Publication-quality figures for the Kubo-Kugo Resolution.

Generates:
  1. Ghost threshold energy spectrum (bar chart)
  2. Fakeon vs Feynman amplitudes near the ghost threshold
  3. Two-pole dominance pie chart

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.kk_kugo_resolution import (
    RL_LORENTZIAN,
    TYPE_C_ZEROS,
    Z0_EUCLIDEAN,
    ZL_LORENTZIAN,
    _build_ghost_catalogue,
    fakeon_vs_feynman_amplitude,
    ghost_threshold_energies,
    two_pole_dominance,
)
from scripts.mr1_lorentzian import Pi_TT_complex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "kk"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Try SciencePlots
try:
    plt.style.use(["science", "no-latex"])
    HAS_SCIENCE = True
except Exception:
    HAS_SCIENCE = False

# SCT colors
SCT_BLUE = "#2166ac"
SCT_RED = "#b2182b"
SCT_GREEN = "#1b7837"
SCT_ORANGE = "#e66101"
SCT_PURPLE = "#762a83"
SCT_GRAY = "#636363"


def figure_ghost_threshold_spectrum() -> Path:
    """
    Figure 1: Ghost threshold energy spectrum.

    Horizontal bar chart showing E_th/Lambda for each ghost pole,
    color-coded by type (A=spacelike, B=timelike, C=Lee-Wick).
    """
    result = ghost_threshold_energies(dps=30)
    thresholds = result["thresholds"]

    labels = []
    energies = []
    colors = []
    hatches = []

    for t in thresholds:
        labels.append(t["label"])
        energies.append(t["E_threshold_over_Lambda"])

        if t["type"] == "B":
            colors.append(SCT_RED)
            hatches.append("")
        elif t["type"] == "A":
            colors.append(SCT_BLUE)
            hatches.append("//")
        else:
            colors.append(SCT_GRAY)
            hatches.append("")

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, energies, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.6)

    # Add hatching for spacelike
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$E_{\mathrm{threshold}} / \Lambda$", fontsize=12)
    ax.set_title("Ghost Pair Production Thresholds", fontsize=13)

    # Mark Lambda scale
    ax.axvline(x=2.0, color="black", linestyle="--", alpha=0.3,
               label=r"$2\Lambda$")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SCT_RED, edgecolor="black", label="Timelike (physical)"),
        Patch(facecolor=SCT_BLUE, edgecolor="black", hatch="//",
              label="Spacelike (no production)"),
        Patch(facecolor=SCT_GRAY, edgecolor="black",
              label="Lee-Wick (complex)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()

    out = FIGURES_DIR / "kk_ghost_threshold_spectrum.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def figure_fakeon_vs_feynman() -> Path:
    """
    Figure 2: Fakeon vs Feynman imaginary part near ghost threshold.

    Plots Im[G(s)] for both prescriptions as a function of s/m^2_ghost,
    showing the delta-function-like peak for Feynman and the zero for fakeon.
    """
    mp.mp.dps = 30

    # Compute Im[G] for a range of s near the ghost mass
    m2_ghost = float(abs(ZL_LORENTZIAN))
    s_ratios = np.linspace(0.3, 2.0, 200)
    s_values = s_ratios * m2_ghost

    eps = 1e-6  # small but finite for numerical plot

    feynman_im = []
    fakeon_im = []

    for s in s_values:
        z_plus = mp.mpc(-s, eps)
        z_minus = mp.mpc(-s, -eps)

        G_plus = 1 / (z_plus * Pi_TT_complex(z_plus, dps=30))
        G_minus = 1 / (z_minus * Pi_TT_complex(z_minus, dps=30))

        # Feynman: Im[G(s + i*eps)]
        feynman_im.append(float(mp.im(G_plus)))

        # Fakeon (PV): Im[(G+ + G-)/2]
        G_PV = (G_plus + G_minus) / 2
        fakeon_im.append(float(mp.im(G_PV)))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(s_ratios, feynman_im, color=SCT_RED, linewidth=1.5,
            label=r"Feynman: $\mathrm{Im}[G_F]$")
    ax.plot(s_ratios, fakeon_im, color=SCT_GREEN, linewidth=1.5,
            linestyle="--", label=r"Fakeon (PV): $\mathrm{Im}[G_{PV}]$")

    ax.axvline(x=1.0, color="black", linestyle=":", alpha=0.5,
               label=r"$s = m_{\mathrm{ghost}}^2$")
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.2)

    ax.set_xlabel(r"$s / m_{\mathrm{ghost}}^2$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Im}[G(s)]$", fontsize=12)
    ax.set_title("Fakeon vs Feynman Prescription Near Ghost Pole", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()

    out = FIGURES_DIR / "kk_fakeon_vs_feynman.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def figure_two_pole_dominance() -> Path:
    """
    Figure 3: Two-pole dominance pie chart.

    Shows the fraction of total residue weight carried by z_L, z_0,
    and all Type C poles combined.
    """
    result = two_pole_dominance(dps=30)

    z_L_pct = result["z_L_fraction"] * 100
    z_0_pct = result["z_0_fraction"] * 100
    lw_pct = 100 - z_L_pct - z_0_pct

    sizes = [z_L_pct, z_0_pct, lw_pct]
    labels = [
        f"$z_L$ (timelike)\n{z_L_pct:.1f}%",
        f"$z_0$ (spacelike)\n{z_0_pct:.1f}%",
        f"Type C (Lee-Wick)\n{lw_pct:.1f}%",
    ]
    colors = [SCT_RED, SCT_BLUE, SCT_GRAY]
    explode = (0.05, 0.05, 0.0)

    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="", startangle=90, textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "black", "linewidth": 0.5},
    )

    ax.set_title(
        "Residue Weight Distribution\n"
        f"(Two-pole dominance: {result['two_pole_dominance_percent']:.1f}%)",
        fontsize=13,
    )

    plt.tight_layout()

    out = FIGURES_DIR / "kk_two_pole_dominance.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def generate_all_figures() -> list[Path]:
    """Generate all KK figures."""
    print("Generating KK figures...")
    figs = []
    figs.append(figure_ghost_threshold_spectrum())
    figs.append(figure_fakeon_vs_feynman())
    figs.append(figure_two_pole_dominance())
    print(f"Generated {len(figs)} figures in {FIGURES_DIR}")
    return figs


if __name__ == "__main__":
    generate_all_figures()
