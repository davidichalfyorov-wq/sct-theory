# ruff: noqa: E402, I001
"""
MR-5: Publication figures for all-orders finiteness analysis.

Generates:
    Fig 1: D(L) for GR vs Stelle vs SCT (background field)
    Fig 2: L_break vs Lambda/M_Pl
    Fig 3: Perturbative reliability: |Gamma^(L)|/|Gamma^(1)| vs L

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Path setup
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.mr5_finiteness import (  # noqa: E402
    M_PL_EV,
    loop_break_scale,
    perturbative_reliability,
    power_counting_background_field,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr5"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Try SciencePlots style
try:
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (7, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

# SCT color palette
SCT_COLORS = {
    "sct": "#2E86AB",
    "gr": "#A23B72",
    "stelle": "#F18F01",
    "modesto": "#2CA58D",
    "highlight": "#E63946",
    "neutral": "#6C757D",
}


def fig1_power_counting(save: bool = True) -> plt.Figure:
    """Fig 1: Superficial degree of divergence D(L) for GR, Stelle, SCT.

    Shows that GR and SCT (naive) have D growing linearly with L,
    while Stelle has D constant. SCT background field gives D=0 at L=1
    (verified) with L>=2 open.
    """
    L_values = list(range(1, 11))

    D_GR = []
    D_Stelle = []
    D_SCT_naive = []
    D_SCT_BF = []

    for L in L_values:
        pc = power_counting_background_field(L)
        D_GR.append(pc["D_GR"])
        D_Stelle.append(pc["D_Stelle"])
        D_SCT_naive.append(pc["D_naive_SCT"])
        D_SCT_BF.append(pc["D_background_field"])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(L_values, D_GR, "o-", color=SCT_COLORS["gr"],
            label="GR: $D = 2L$", linewidth=2, markersize=6)
    ax.plot(L_values, D_Stelle, "s--", color=SCT_COLORS["stelle"],
            label=r"Stelle: $D = 2$", linewidth=2, markersize=6)
    ax.plot(L_values, D_SCT_naive, "^:", color=SCT_COLORS["neutral"],
            label=r"SCT naive: $D = 2L$", linewidth=1.5, markersize=5,
            alpha=0.6)

    # SCT background field: D=0 at L=1 (verified), open at L>=2
    ax.plot(1, 0, "D", color=SCT_COLORS["sct"], markersize=12, zorder=5,
            label="SCT BF: $D=0$ (L=1, verified)")
    for L in range(2, 11):
        ax.plot(L, 0, "D", color=SCT_COLORS["sct"], markersize=8,
                alpha=0.3, zorder=4)
    # Add question marks for open values
    for L in range(2, 7):
        ax.annotate("?", (L, 0.5), ha="center", va="center",
                     fontsize=12, color=SCT_COLORS["sct"], fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    ax.set_xlabel("Loop order $L$")
    ax.set_ylabel("Superficial degree of divergence $D$")
    ax.set_title("Power counting: GR vs Stelle vs SCT\n(self-energy, $E_{\\rm ext}=2$)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-2, 22)
    ax.set_xticks(L_values)

    if save:
        path = FIGURES_DIR / "mr5_power_counting.pdf"
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def fig2_L_break_vs_Lambda(save: bool = True) -> plt.Figure:
    """Fig 2: L_break vs Lambda/M_Pl.

    Shows how the perturbative breakdown loop order depends on the
    cutoff scale. Marked scales: PPN, electroweak, GUT, Planck.
    """
    # Generate L_break for a range of Lambda values
    log_ratios = np.linspace(-30, 0, 100)
    ratios = 10.0**log_ratios
    L_breaks = []

    for r in ratios:
        Lambda_eV = float(r * float(M_PL_EV))
        try:
            lb = loop_break_scale(Lambda_eV, dps=30)
            L_breaks.append(lb["L_break_refined"])
        except Exception:
            L_breaks.append(float("nan"))

    L_breaks = np.array(L_breaks)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(log_ratios, np.log10(L_breaks), "-", color=SCT_COLORS["sct"],
            linewidth=2.5, label=r"SCT: $L_{\rm break}$")

    # GR line at L=2
    ax.axhline(y=np.log10(2), color=SCT_COLORS["gr"], linewidth=2,
               linestyle="--", label="GR: $L_{\\rm break} = 2$")

    # Mark key scales (skip PPN -- off-chart at log_r ~ -30, log_Lb ~ 62)
    marked_scales = [
        ("EW", 246e9, -50, -20),
        ("GUT", 1e25, 20, 15),
        ("Planck", float(M_PL_EV), -55, 15),
    ]
    for label, Lambda_eV, dx, dy in marked_scales:
        r = Lambda_eV / float(M_PL_EV)
        log_r = np.log10(r) if r > 0 else -30
        lb = loop_break_scale(Lambda_eV, dps=30)
        log_Lb = min(np.log10(lb["L_break_refined"]), 34)
        ax.plot(log_r, log_Lb, "o", color=SCT_COLORS["highlight"],
                markersize=8, zorder=5)
        # Format L_break in scientific notation for large numbers
        Lb_val = lb["L_break_refined"]
        if Lb_val > 1e6:
            Lb_str = f"$10^{{{np.log10(Lb_val):.0f}}}$"
        else:
            Lb_str = f"${Lb_val:.0f}$"
        ax.annotate(f"{label}\n$L_{{\\rm break}}\\approx${Lb_str}",
                     (log_r, log_Lb), textcoords="offset points",
                     xytext=(dx, dy), ha="center",
                     fontsize=9, color=SCT_COLORS["highlight"],
                     arrowprops={"arrowstyle": "->", "color": SCT_COLORS["highlight"],
                                 "lw": 0.8})

    ax.set_xlabel(r"$\log_{10}(\Lambda / M_{\rm Pl})$")
    ax.set_ylabel(r"$\log_{10}(L_{\rm break})$")
    ax.set_title(r"Perturbative breakdown scale: SCT vs GR")
    ax.legend(loc="upper right")
    ax.set_xlim(-32, 1)
    ax.set_ylim(-0.5, 35)

    # Shaded region: perturbatively reliable
    ax.fill_between(log_ratios, np.log10(L_breaks),
                     np.full_like(log_ratios, -0.5),
                     alpha=0.1, color=SCT_COLORS["sct"])

    if save:
        path = FIGURES_DIR / "mr5_L_break.pdf"
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def fig3_perturbative_reliability(save: bool = True) -> plt.Figure:
    """Fig 3: |Gamma^(L)|/|Gamma^(1)| vs L at Planck and GUT scales.

    Shows the perturbative reliability of the loop expansion.
    The optimal truncation is where the ratio reaches its minimum.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    scales = [
        ("Planck", float(M_PL_EV), SCT_COLORS["sct"]),
        ("GUT", 1e25, SCT_COLORS["stelle"]),
        ("Electroweak", 246e9, SCT_COLORS["modesto"]),
    ]

    for label, Lambda_eV, color in scales:
        pr = perturbative_reliability(15, Lambda_eV, dps=30)
        L_vals = [row["L"] for row in pr["loop_data"]]
        ratios = [row["ratio_to_one_loop"] for row in pr["loop_data"]]
        log_ratios = [np.log10(r) if r > 0 else -300 for r in ratios]

        # Clip for plotting
        log_ratios = [max(lr, -50) for lr in log_ratios]

        ax.plot(L_vals, log_ratios, "o-", color=color, linewidth=2,
                markersize=5, label=f"{label} ($\\varepsilon={pr['epsilon']:.1e}$)")

        # Mark optimal truncation
        opt_L = pr["optimal_truncation_L"]
        opt_idx = opt_L - 1
        if opt_idx < len(log_ratios):
            ax.plot(opt_L, log_ratios[opt_idx], "s", color=color,
                    markersize=10, zorder=5)

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.annotate(r"$|\Gamma^{(L)}| = |\Gamma^{(1)}|$", (14, 0.3),
                 fontsize=10, color="black")

    ax.set_xlabel("Loop order $L$")
    ax.set_ylabel(r"$\log_{10}\left(|\Gamma^{(L)}| / |\Gamma^{(1)}|\right)$")
    ax.set_title("Perturbative reliability of the loop expansion")
    ax.legend(loc="lower right")
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(-50, 20)
    ax.set_xticks(range(1, 16))

    if save:
        path = FIGURES_DIR / "mr5_perturbative_reliability.pdf"
        fig.savefig(path)
        print(f"Saved: {path}")

    return fig


def generate_all_figures():
    """Generate all MR-5 figures."""
    print("Generating MR-5 figures...")
    fig1_power_counting()
    fig2_L_break_vs_Lambda()
    fig3_perturbative_reliability()
    print("All MR-5 figures generated.")
    plt.close("all")


if __name__ == "__main__":
    generate_all_figures()
