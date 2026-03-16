# ruff: noqa: E402, I001
"""
MR-4 Figures: Two-loop effective action analysis for SCT.

Generates:
    1. Power-counting D vs loop order L for GR, Stelle, SCT
    2. Two-loop correction magnitude vs Lambda/M_Pl
    3. Spectral function moments f_n vs n

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

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import SCT_COLORS, create_figure, init_style, save_figure

from scripts.mr4_two_loop import (
    correct_power_counting,
    spectral_function_moments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr4"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig1_power_counting():
    """Fig 1: Superficial degree of divergence D vs loop order L.

    Compares GR, Stelle gravity, and SCT (naive power counting)
    for the graviton self-energy (E_ext = 2).
    """
    init_style()
    fig, ax = create_figure(figsize=(3.4, 2.8))

    L_values = np.arange(1, 6)
    D_GR = []
    D_Stelle = []
    D_SCT_naive = []

    for L in L_values:
        pc = correct_power_counting(int(L), 2)
        D_GR.append(pc["D_GR"])
        D_Stelle.append(pc["D_Stelle"])
        D_SCT_naive.append(pc["D_SCT_naive"])

    ax.plot(L_values, D_GR, "o-", color=SCT_COLORS["dirac"],
            label="GR ($G \\sim 1/k^2$)", linewidth=1.5, markersize=5)
    ax.plot(L_values, D_Stelle, "s--", color=SCT_COLORS["vector"],
            label="Stelle ($G \\sim 1/k^4$)", linewidth=1.5, markersize=5)
    ax.plot(L_values, D_SCT_naive, "^:", color=SCT_COLORS["total"],
            label="SCT naive ($G \\sim 1/k^2$)", linewidth=1.5, markersize=5)

    # Mark the MR-7 verified point: SCT at L=1 is D=0 (heat kernel)
    ax.plot(1, 0, "*", color=SCT_COLORS["prediction"], markersize=12, zorder=5,
            label="SCT heat kernel (MR-7)")

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.axhline(y=4, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Loop order $L$")
    ax.set_ylabel("$\\mathcal{D}$ (self-energy, $E_{\\mathrm{ext}} = 2$)")
    ax.set_title("Power counting: GR vs Stelle vs SCT")
    ax.set_xticks(L_values)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-1, 12)

    save_figure(fig, "mr4_power_counting", directory=FIGURES_DIR)
    plt.close(fig)
    print("  Saved: mr4_power_counting.pdf")


def fig2_two_loop_magnitude():
    """Fig 2: Two-loop correction magnitude vs Lambda/M_Pl.

    Shows the loop expansion parameter epsilon = (Lambda/M_Pl)^2 / (8*pi)
    and the two-loop suppression factor epsilon^2.
    """
    init_style()
    fig, ax = create_figure(figsize=(3.4, 2.8))

    # Lambda values from PPN-1 bound to Planck scale
    log_Lambda = np.linspace(-3, 27, 200)  # in eV
    Lambda_values = 10.0**log_Lambda

    M_Pl = 2.435e27  # eV
    ratios = Lambda_values / M_Pl
    epsilon_1 = ratios**2 / (8 * np.pi)
    epsilon_2 = epsilon_1**2

    ax.semilogy(np.log10(ratios), epsilon_1, "-", color=SCT_COLORS["dirac"],
                label=r"$\epsilon_1 = (\Lambda/M_{\mathrm{Pl}})^2/(8\pi)$",
                linewidth=1.5)
    ax.semilogy(np.log10(ratios), epsilon_2, "--", color=SCT_COLORS["total"],
                label=r"$\epsilon_2 = \epsilon_1^2$",
                linewidth=1.5)

    # Mark physical scales
    scales = {
        r"PPN-1": 2.38e-3 / M_Pl,
        r"EW": 246e9 / M_Pl,
        r"GUT": 1e25 / M_Pl,
    }
    for label, ratio in scales.items():
        ax.axvline(np.log10(ratio), color="gray", linestyle=":", alpha=0.4,
                   linewidth=0.5)
        ax.annotate(label, (np.log10(ratio), 1e-2), fontsize=6,
                    rotation=90, ha="right", va="bottom")

    ax.axhline(1.0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.annotate("perturbativity bound", (0.5, 1.2), fontsize=6,
                ha="center", va="bottom", color="gray",
                transform=ax.get_yaxis_transform())

    ax.set_xlabel(r"$\log_{10}(\Lambda/M_{\mathrm{Pl}})$")
    ax.set_ylabel("Loop expansion parameter")
    ax.set_title("Two-loop corrections vs cutoff scale")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(1e-130, 1e2)
    ax.set_xlim(-31, 1)

    save_figure(fig, "mr4_two_loop_magnitude", directory=FIGURES_DIR)
    plt.close(fig)
    print("  Saved: mr4_two_loop_magnitude.pdf")


def fig3_spectral_moments():
    """Fig 3: Spectral function moments f_n vs n.

    f_n = Gamma(n/2) for psi(u) = e^{-u}.
    Shows the factorial growth of the coefficients in the
    heat kernel expansion of the spectral action.
    """
    init_style()
    fig, (ax1, ax2) = create_figure(nrows=1, ncols=2, figsize=(7.0, 2.8))

    n_values = np.arange(2, 16, 2)  # n = 2, 4, 6, ..., 14
    f_values = []
    for n in n_values:
        f_values.append(float(spectral_function_moments(int(n))))

    # Linear scale
    ax1.bar(n_values, f_values, width=1.5, color=SCT_COLORS["scalar"],
            edgecolor="black", linewidth=0.5, alpha=0.8)
    ax1.set_xlabel("Heat kernel order $n$")
    ax1.set_ylabel("$f_n = \\Gamma(n/2)$")
    ax1.set_title("Spectral function moments")
    ax1.set_xticks(n_values)

    # Mark key moments
    for n, f in zip(n_values[:4], f_values[:4]):
        ax1.annotate(f"$f_{{{n}}}={f:.0f}$", (n, f), fontsize=6,
                     ha="center", va="bottom", textcoords="offset points",
                     xytext=(0, 3))

    # Log scale
    ax2.semilogy(n_values, f_values, "o-", color=SCT_COLORS["total"],
                 linewidth=1.5, markersize=5)

    # Overlay (n/2 - 1)! for comparison
    factorial_values = [float(mp.factorial(int(n) // 2 - 1)) for n in n_values]
    ax2.semilogy(n_values, factorial_values, "s--", color=SCT_COLORS["dirac"],
                 linewidth=1.0, markersize=4, alpha=0.7,
                 label=r"$(n/2-1)!$")

    ax2.set_xlabel("Heat kernel order $n$")
    ax2.set_ylabel("$f_n$")
    ax2.set_title("Factorial growth (log scale)")
    ax2.set_xticks(n_values)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, "mr4_spectral_moments", directory=FIGURES_DIR)
    plt.close(fig)
    print("  Saved: mr4_spectral_moments.pdf")


def generate_all_figures():
    """Generate all MR-4 figures."""
    print("Generating MR-4 figures...")
    fig1_power_counting()
    fig2_two_loop_magnitude()
    fig3_spectral_moments()
    print("All MR-4 figures generated.")


if __name__ == "__main__":
    generate_all_figures()
