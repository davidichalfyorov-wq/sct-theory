# ruff: noqa: E402, I001
# -*- coding: utf-8 -*-
"""
Russian-language figures for Paper 4 (field equations).

Figure 1: nt4a_newtonian_potential_ru.pdf
Figure 2: ppn1_exclusion_ru.pdf
Figure 3: nt4b/nt4b_effective_masses_ru.pdf
Figure 4: mt2/mt2_suppression_ru.pdf

Author: David Alfyorov
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
})

from sct_tools.plotting import SCT_COLORS

# Imports for figure 1
from scripts.nt4a_newtonian import sample_potential_curve

# Imports for figure 2
from scripts.ppn1_parameters import lower_bound_Lambda

# Imports for figure 3
from scripts.nt4b_nonlinear import (
    ALPHA_C,
    scalar_mode_coefficient,
)

# Imports for figure 4
from scripts.mt2_cosmology import (
    LAMBDA_PPN,
    M_PL_EV,
    suppression_factor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
FIGURES_NT4B = FIGURES_DIR / "nt4b"
FIGURES_MT2 = FIGURES_DIR / "mt2"

# Colors (consistent with English versions)
C_TT = "#2196F3"
C_S = "#FF5722"
C_S2 = "#4CAF50"
C_S3 = "#9C27B0"
C_GRAY = "#757575"


def fig1_newtonian_potential_ru():
    """Figure 1: NT-4a local Yukawa approximation (Russian)."""
    print("Generating Figure 1: nt4a_newtonian_potential_ru.pdf")

    xi_values = [0.0, 1 / 6, 1.0]
    radii = [10 ** exponent for exponent in (-1, 0, 1, 2, 3, 4, 5, 6)]

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for xi in xi_values:
        samples = sample_potential_curve(radii, Lambda=1.0, xi=xi)
        ax.plot(
            [s["r"] for s in samples],
            [s["ratio"] for s in samples],
            marker="o",
            label=rf"$\xi = {float(xi):.3g}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$r / \ell_P$")
    ax.set_ylabel(r"$V(r) / V_{\mathrm{Newton}}(r)$")
    ax.set_title("NT-4a: \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u043e\u0435 \u044e\u043a\u0430\u0432\u0441\u043a\u043e\u0435 \u043f\u0440\u0438\u0431\u043b\u0438\u0436\u0435\u043d\u0438\u0435")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "nt4a_newtonian_potential_ru.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: nt4a_newtonian_potential_ru.pdf")


def fig2_ppn_exclusion_ru():
    """Figure 2: PPN exclusion plot (Russian)."""
    print("Generating Figure 2: ppn1_exclusion_ru.pdf")

    xi = 0.0
    experiments = ["cassini", "messenger", "eot-wash"]
    bounds = {}
    for exp in experiments:
        result = lower_bound_Lambda(exp, xi=xi)
        if result["Lambda_min_eV"] is not None:
            bounds[result["experiment"]] = result["Lambda_min_eV"]

    # |gamma-1| curve
    rL_vals = np.logspace(-1, 3, 200)
    gamma_minus_1 = []
    for rL in rL_vals:
        m2_ov_L = math.sqrt(60 / 13)
        m0_ov_L = math.sqrt(6)
        g_m_1 = abs(
            2 / 3 * math.exp(-m2_ov_L * rL)
            - 2 / 3 * math.exp(-m0_ov_L * rL)
        )
        gamma_minus_1.append(max(g_m_1, 1e-300))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel
    ax1.semilogy(rL_vals, gamma_minus_1,
                 color=SCT_COLORS["total"], linewidth=1.5)
    ax1.axhline(y=2.3e-5, color=SCT_COLORS["data"], linestyle="--",
                alpha=0.7, label="\u0413\u0440\u0430\u043d\u0438\u0446\u0430 \u041a\u0430\u0441\u0441\u0438\u043d\u0438")
    ax1.set_xlabel(r"$r \Lambda$")
    ax1.set_ylabel(r"$|\gamma - 1|$")
    ax1.set_xscale("log")
    ax1.set_xlim(0.1, 1000)
    ax1.set_ylim(1e-15, 2)
    ax1.set_title("\u041e\u0442\u043a\u043b\u043e\u043d\u0435\u043d\u0438\u0435 \u041f\u041f\u041d " + r"$\gamma$" + " (\u0443\u0440\u043e\u0432\u0435\u043d\u044c 2)")
    ax1.legend(fontsize=8)

    # Right panel
    colors = {
        "Cassini": SCT_COLORS["scalar"],
        "MESSENGER": SCT_COLORS["vector"],
        "Eot-Wash": SCT_COLORS["dirac"],
    }
    y_pos = 0
    for label, lam_min in sorted(bounds.items(), key=lambda x: x[1]):
        short = label.split("(")[0].strip()
        color = colors.get(short, SCT_COLORS["reference"])
        ax2.barh(y_pos, math.log10(lam_min) - (-20), left=-20,
                 height=0.6, color=color, alpha=0.4)
        ax2.axvline(x=math.log10(lam_min), color=color, linestyle="--",
                    linewidth=1.2)
        ax2.text(math.log10(lam_min) + 0.3, y_pos,
                 f"{short}\n" + r"$\Lambda > $" + f"{lam_min:.1e} \u044d\u0412",
                 fontsize=7, va="center")
        y_pos += 1

    ax2.set_xlabel(r"$\log_{10}(\Lambda\,/\,$" + "\u044d\u0412)")
    ax2.set_xlim(-20, 2)
    ax2.set_yticks([])
    ax2.set_title("\u041d\u0438\u0436\u043d\u0438\u0435 \u0433\u0440\u0430\u043d\u0438\u0446\u044b \u043d\u0430 " + r"$\Lambda$")
    ax2.axvspan(-20, max(math.log10(v) for v in bounds.values()),
                alpha=0.05, color="red", label="\u0418\u0441\u043a\u043b\u044e\u0447\u0435\u043d\u043e")

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "ppn1_exclusion_ru.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ppn1_exclusion_ru.pdf")


def fig3_effective_masses_ru():
    """Figure 3: Effective masses and Newtonian potential (Russian)."""
    print("Generating Figure 3: nt4b/nt4b_effective_masses_ru.pdf")
    FIGURES_NT4B.mkdir(parents=True, exist_ok=True)

    mp.mp.dps = 50
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    xi_arr = np.linspace(-0.3, 1.5, 300)
    m2_sq = float(mp.mpf(60) / 13)
    m0_sq_vals = []
    for xi in xi_arr:
        s = float(scalar_mode_coefficient(xi))
        m0_sq_vals.append(1.0 / s if s > 1e-10 else np.nan)

    ax1.axhline(y=m2_sq, color=C_TT, linewidth=1.5,
                label=f"$m_2^2/\\Lambda^2 = 60/13 = {m2_sq:.3f}$")
    ax1.plot(xi_arr, m0_sq_vals, color=C_S, linewidth=1.5,
             label="$m_0^2/\\Lambda^2 = 1/[6(\\xi-1/6)^2]$")
    ax1.axvline(x=1.0 / 6, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$m^2 / \Lambda^2$")
    ax1.set_title("\u042d\u0444\u0444\u0435\u043a\u0442\u0438\u0432\u043d\u044b\u0435 \u043c\u0430\u0441\u0441\u043e\u0432\u044b\u0435 \u043c\u0430\u0441\u0448\u0442\u0430\u0431\u044b")
    ax1.set_xlim(-0.3, 1.5)
    ax1.set_ylim(0, 30)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    r_arr = np.linspace(0.01, 10, 500)
    m2_l = float(mp.sqrt(mp.mpf(60) / 13))
    m0_l = float(mp.sqrt(6))
    V_ratio = 1 - (4.0 / 3) * np.exp(-m2_l * r_arr) + (1.0 / 3) * np.exp(-m0_l * r_arr)

    ax2.plot(r_arr, V_ratio, color=C_TT, linewidth=1.5, label="$V(r)/V_N(r)$")
    ax2.axhline(y=1, color="gray", linewidth=0.5, linestyle=":")
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax2.plot(0, 0, "o", color=C_S, markersize=5,
             label="$V(0) = 0$ (\u043a\u043e\u043d\u0435\u0447\u043d\u043e)")
    ax2.set_xlabel(r"$r \cdot \Lambda$")
    ax2.set_ylabel("$V(r) / V_N(r)$")
    ax2.set_title("\u041c\u043e\u0434\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u044f \u043d\u044c\u044e\u0442\u043e\u043d\u043e\u0432\u0441\u043a\u043e\u0433\u043e \u043f\u043e\u0442\u0435\u043d\u0446\u0438\u0430\u043b\u0430 " + r"($\xi=0$)")
    ax2.set_xlim(0, 10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(FIGURES_NT4B / "nt4b_effective_masses_ru.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: nt4b/nt4b_effective_masses_ru.pdf")


def fig4_mt2_suppression_ru():
    """Figure 4: MT-2 suppression factor (Russian)."""
    print("Generating Figure 4: mt2/mt2_suppression_ru.pdf")
    FIGURES_MT2.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    Lambda_configs = [
        ("$\\Lambda = 2{,}38 \\times 10^{-3}$ \u044d\u0412 (\u041f\u041f\u041d)", LAMBDA_PPN, "#d62728"),
        ("$\\Lambda = 1$ \u044d\u0412", mp.mpf("1"), "#ff7f0e"),
        ("$\\Lambda = 1$ \u0413\u044d\u0412", mp.mpf("1e9"), "#2ca02c"),
        ("$\\Lambda = 10^{6}$ \u0413\u044d\u0412", mp.mpf("1e15"), "#1f77b4"),
        ("$\\Lambda = M_{\\rm Pl}$", M_PL_EV, "#9467bd"),
    ]

    z_plot = np.logspace(-2, np.log10(1200), 200)
    z_plot = np.concatenate([[0], z_plot])

    for label, lam_val, color in Lambda_configs:
        log_sup = []
        for z_val in z_plot:
            s = suppression_factor(z_val, lam_val, xi=0)
            if s["log10_suppression"] is not None:
                log_sup.append(s["log10_suppression"])
            else:
                log_sup.append(-200)
        ax.plot(z_plot[1:], log_sup[1:], label=label, color=color, lw=1.8)

    ax.set_xlabel("\u041a\u0440\u0430\u0441\u043d\u043e\u0435 \u0441\u043c\u0435\u0449\u0435\u043d\u0438\u0435 $z$", fontsize=13)
    ax.set_ylabel(r"$\log_{10}(\delta H^2 / H^2)$", fontsize=13)
    ax.set_title(
        "\u0424\u0430\u043a\u0442\u043e\u0440 \u043f\u043e\u0434\u0430\u0432\u043b\u0435\u043d\u0438\u044f \u0421\u041a\u0422: "
        + r"$\delta H^2/H^2 = \beta_R \cdot (H/\Lambda)^2$"
        + "\n"
        + "($\\xi = 0$, \u0444\u043e\u043d FLRW)",
        fontsize=12,
    )
    ax.set_xscale("log")
    ax.legend(fontsize=10, loc="upper left")
    ax.axhline(y=np.log10(0.18), color="black", ls="--", lw=1.2)
    ax.text(1.5, np.log10(0.18) + 1.5,
            "\u0422\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f \u0434\u043b\u044f $H_0$-\u043d\u0430\u043f\u0440\u044f\u0436\u0451\u043d\u043d\u043e\u0441\u0442\u0438: $\\delta H^2/H^2 \\sim 0.18$",
            fontsize=10, ha="left")
    ax.set_xlim(0.01, 1200)
    ax.set_ylim(-130, 5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_MT2 / "mt2_suppression_ru.pdf"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: mt2/mt2_suppression_ru.pdf")


if __name__ == "__main__":
    fig1_newtonian_potential_ru()
    fig2_ppn_exclusion_ru()
    fig3_effective_masses_ru()
    fig4_mt2_suppression_ru()
    print("\nAll 4 Russian figures generated successfully.")
