# ruff: noqa: E402, I001
"""
Publication figures for the CJ bridge formula paper.

Generates 6 PDF figures in papers/drafts/figures/:
  1. fig_cj_ratio.pdf       -- Central result: measured/predicted ratio vs N
  2. fig_cj_nscaling.pdf    -- N-scaling: log-log fit and residuals
  3. fig_cj_epsilon.pdf     -- Epsilon independence of normalized CJ
  4. fig_cj_diagnostics.pdf -- 2x2 diagnostic panel (spacetimes, polarization, etc.)
  5. fig_cj_derivation.pdf  -- Derivation chain flowchart
  6. fig_cj_validity.pdf    -- Validity range: ratio vs xi = eps/N^{1/4}

Usage:
    python analysis/scripts/generate_cj_bridge_figures.py
"""

from __future__ import annotations

import sys
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import init_style, SCT_COLORS

# Output directory
FIGURES_DIR = PROJECT_ROOT / "papers" / "drafts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style initialization ────────────────────────────────────────────
init_style()


# =====================================================================
#  Verified numerical data (hardcoded from production runs)
# =====================================================================

# Raw CJ data: eps=3, T=1, exact pp-wave predicate
N_DATA = np.array([500, 1000, 2000, 3000, 5000, 8000, 10000, 15000])
CJ_MEAN = np.array([0.00759, 0.01379, 0.02345, 0.03511,
                     0.05673, 0.08650, 0.11254, 0.15495])
SEEDS = np.array([20, 20, 20, 13, 8, 5, 3, 2])

# Physical parameters for the measurement
EPS = 3.0
T = 1.0
E_SQUARED = EPS**2 / 2  # = 4.5  (geometric: E_ij E^ij = eps^2/2 for pp-wave)

# Analytical formula (parameter-free):
#   CJ_pred = C_0 * N^{8/9} * E^2 * T^4
# where C_0 = 32*pi^2 / (3 * 9! * 45)
# The factor 32 = 4 * 8 arises from:
#   - factor 4 = 2^2 from two-leg structure of Y = log2(p_down * p_up + 1)
#   - 8*pi^2/(3*9!*45) is the combinatorial skeleton
PREFACTOR = 32 * np.pi**2 / (3 * factorial(9) * 45)
# = 6.4469e-6  (NO free parameters)

# Predicted CJ for each N
CJ_PRED = PREFACTOR * N_DATA**(8/9) * E_SQUARED * T**4

# Approximate standard errors (relative scatter ~ 10% per seed)
CJ_SE = CJ_MEAN * 0.10 / np.sqrt(SEEDS)

# Epsilon independence (stratification test, N=2000, T=1):
EPS_VALUES = np.array([1, 2, 3, 4, 5, 6, 8])
CJ_OVER_E2 = np.array([0.00688, 0.00539, 0.00486, 0.00472,
                        0.00473, 0.00472, 0.00470])

# Kottler/dS comparison (N=5000)
SPACETIME_LABELS = ["pp-wave", "Schwarzschild", "Kottler", "de Sitter"]
COHEN_D = [4.40, 4.40, 3.44, 3.81]  # Cohen's d vs CRN null
CJ_SPACETIME = np.array([0.05673, 0.01122, 0.01310, 0.00178])
E2_SPACETIME = np.array([4.50, 0.96, 1.043, 0.083])

# SJ entropy decorrelation
CORR_CJ_ENTROPY = -0.074

# N-scaling: theory prediction
ALPHA_THEORY = 8 / 9  # ~ 0.889

# Polarization ratio (cross/plus)
POL_RATIO = 3.0

# Validity range parameter xi = eps / N^{1/4}
XI_DATA = EPS / N_DATA**(1/4)

# ── Colour palette ──────────────────────────────────────────────────
C_MAIN = SCT_COLORS["total"]        # black
C_PRED = SCT_COLORS["prediction"]   # orange
C_DATA = SCT_COLORS["data"]         # purple
C_THEORY = SCT_COLORS["scalar"]     # blue
C_GREEN = SCT_COLORS["vector"]      # green
C_RED = SCT_COLORS["dirac"]         # red
C_BAND = SCT_COLORS["theory_band"]  # light blue
C_REF = SCT_COLORS["reference"]     # gray


# =====================================================================
#  Figure 1: Central result — measured / predicted ratio
# =====================================================================
def fig_cj_ratio():
    """CJ_measured / CJ_predicted vs N with uncertainty band."""
    ratio = CJ_MEAN / CJ_PRED
    ratio_err = CJ_SE / CJ_PRED

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # +-4% band
    ax.axhspan(0.96, 1.04, color="#e0e0e0", alpha=0.5, zorder=0,
               label=r"$\pm 4\%$")
    # Reference line
    ax.axhline(1.0, color=C_MAIN, ls="--", lw=0.8, zorder=1)

    # Data points
    ax.errorbar(N_DATA, ratio, yerr=ratio_err,
                fmt="o", ms=5, color=C_DATA, ecolor=C_DATA,
                capsize=2.5, capthick=0.8, lw=1.0, zorder=3,
                label=r"$\varepsilon = 3$, $T = 1$")

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\mathrm{CJ}_{\mathrm{meas}} \,/\, \mathrm{CJ}_{\mathrm{pred}}$")
    ax.set_xlim(0, 16000)
    ax.set_ylim(0.88, 1.12)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, framealpha=0.9)

    fig.tight_layout(pad=0.3)
    outpath = FIGURES_DIR / "fig_cj_ratio.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/6] {outpath.name}  (ratio range: {ratio.min():.3f} -- {ratio.max():.3f})")


# =====================================================================
#  Figure 2: N-scaling — log-log fit and residuals
# =====================================================================
def fig_cj_nscaling():
    """Two-panel: log-log CJ vs N (top) and residuals (bottom)."""
    log_N = np.log(N_DATA)
    log_CJ = np.log(CJ_MEAN)

    # Linear fit in log-log space: log(CJ) = intercept + alpha * log(N)
    coeffs = np.polyfit(log_N, log_CJ, 1)
    alpha_fit = coeffs[0]
    intercept = coeffs[1]

    # Uncertainty on alpha from residual scatter
    log_CJ_fit = intercept + alpha_fit * log_N
    res_log = log_CJ - log_CJ_fit
    n_pts = len(log_N)
    s_res = np.sqrt(np.sum(res_log**2) / (n_pts - 2))
    ss_xx = np.sum((log_N - log_N.mean())**2)
    alpha_err = s_res / np.sqrt(ss_xx)

    # Smooth curves for plotting
    N_fine = np.linspace(400, 17000, 200)

    # Fit line
    CJ_fit_fine = np.exp(intercept) * N_fine**alpha_fit

    # Theory line (8/9 exponent) anchored at data midpoint for visual comparison
    mid_idx = len(N_DATA) // 2
    A_theory = CJ_MEAN[mid_idx] / N_DATA[mid_idx]**(ALPHA_THEORY)
    CJ_theory_fine = A_theory * N_fine**ALPHA_THEORY

    # Residuals: CJ_measured / CJ_fit - 1
    CJ_fit_at_data = np.exp(intercept) * N_DATA**alpha_fit
    residuals = (CJ_MEAN / CJ_fit_at_data - 1.0) * 100  # percent

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # ── Top panel: log-log ───────────────────────────────────────────
    ax1.loglog(N_DATA, CJ_MEAN, "o", ms=5, color=C_DATA, zorder=3,
               label="Data")
    ax1.loglog(N_fine, CJ_fit_fine,
               "-", color=C_PRED, lw=1.2, zorder=2,
               label=rf"Fit: $\alpha = {alpha_fit:.3f} \pm {alpha_err:.3f}$")
    ax1.loglog(N_fine, CJ_theory_fine,
               "--", color=C_REF, lw=0.9, zorder=1,
               label=rf"$\alpha = 8/9 \approx {ALPHA_THEORY:.3f}$")

    ax1.set_ylabel(r"$\langle \mathrm{CJ} \rangle$")
    ax1.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9)
    ax1.tick_params(axis="x", which="both", labelbottom=False)

    # ── Bottom panel: residuals ──────────────────────────────────────
    ax2.axhline(0, color=C_MAIN, ls="--", lw=0.7)
    ax2.fill_between([400, 17000], -4, 4, color="#e0e0e0", alpha=0.4, zorder=0)
    ax2.plot(N_DATA, residuals, "s", ms=4, color=C_PRED, zorder=3)
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"Residual [\%]")
    ax2.set_ylim(-10, 10)
    ax2.set_xscale("log")

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(hspace=0.06)
    outpath = FIGURES_DIR / "fig_cj_nscaling.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/6] {outpath.name}  (alpha_fit = {alpha_fit:.4f} +/- {alpha_err:.4f})")


# =====================================================================
#  Figure 3: Epsilon independence
# =====================================================================
def fig_cj_epsilon():
    r"""CJ / E^2 vs epsilon, showing convergence for eps >= 3."""
    # Normalize by CJ_over_E2 at eps=3 (index 2) for clarity
    cj_normalized = CJ_OVER_E2 / CJ_OVER_E2[2]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # Plateau band (mean +/- 1% for eps >= 3)
    plateau_vals = cj_normalized[2:]
    plateau_mean = np.mean(plateau_vals)
    ax.axhspan(plateau_mean - 0.02, plateau_mean + 0.02,
               color=C_BAND, alpha=0.5, zorder=0,
               label=rf"Plateau $\pm 2\%$")

    # Data points with line
    ax.plot(EPS_VALUES, cj_normalized, "o-", ms=5, color=C_DATA,
            lw=1.0, zorder=3, label=r"$\mathrm{CJ}/\mathcal{E}^2$ (norm.)")

    # Convergence threshold
    ax.axvline(3, color=C_REF, ls=":", lw=0.8, alpha=0.7)
    ax.annotate(r"$\varepsilon \geq 3$", xy=(3, 1.38),
                fontsize=7, color=C_REF, ha="left",
                xytext=(3.3, 1.38))

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\mathrm{CJ} / \mathcal{E}^2$ (normalized to $\varepsilon{=}3$)")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.90, 1.50)
    ax.legend(loc="upper right", fontsize=6, frameon=True, framealpha=0.9)

    fig.tight_layout(pad=0.3)
    outpath = FIGURES_DIR / "fig_cj_epsilon.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/6] {outpath.name}")


# =====================================================================
#  Figure 4: Diagnostics — 2x2 panel
# =====================================================================
def fig_cj_diagnostics():
    """2x2 diagnostic panel:
    (a) Spacetime Cohen's d comparison
    (b) CJ/E^2 convergence across epsilon (raw values)
    (c) Polarization ratio cross/plus
    (d) SJ entropy decorrelation
    """
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.2))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lbl in zip(axes.flat, panel_labels):
        ax.text(0.04, 0.94, lbl, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")

    # ── (a) Spacetime comparison: Cohen's d ──────────────────────────
    ax = axes[0, 0]
    colors_bar = [C_THEORY, C_DATA, C_GREEN, C_RED]
    x_pos = np.arange(len(SPACETIME_LABELS))
    ax.bar(x_pos, COHEN_D, width=0.6, color=colors_bar, alpha=0.85,
           edgecolor="k", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SPACETIME_LABELS, fontsize=5.5, rotation=20, ha="right")
    ax.set_ylabel(r"Cohen's $d$", fontsize=7)
    ax.axhline(0.8, color=C_REF, ls=":", lw=0.6, alpha=0.7)
    ax.text(3.45, 0.95, "large", fontsize=5.5, color=C_REF, ha="right",
            style="italic")
    ax.set_ylim(0, 5.5)

    # ── (b) CJ/E^2 vs epsilon (raw values) ──────────────────────────
    ax = axes[0, 1]
    ax.plot(EPS_VALUES, CJ_OVER_E2 * 1e3, "o-", ms=4, color=C_DATA, lw=0.9)
    # Plateau line
    plateau = np.mean(CJ_OVER_E2[2:]) * 1e3
    ax.axhline(plateau, color=C_PRED, ls="--", lw=0.8,
               label=rf"plateau $= {plateau:.2f} \times 10^{{-3}}$")
    ax.set_xlabel(r"$\varepsilon$", fontsize=7)
    ax.set_ylabel(r"$\mathrm{CJ} / \mathcal{E}^2$ [$\times 10^{-3}$]", fontsize=7)
    ax.legend(fontsize=5.5, loc="upper right", frameon=True, framealpha=0.9)
    ax.set_xlim(0.5, 8.5)

    # ── (c) Polarization ratio ───────────────────────────────────────
    ax = axes[1, 0]
    bar_labels = [r"$+$", r"$\times$"]
    bar_vals = [1.0, POL_RATIO]
    bar_colors = [C_THEORY, C_PRED]
    ax.bar([0, 1], bar_vals, width=0.5, color=bar_colors, alpha=0.85,
           edgecolor="k", linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Relative CJ", fontsize=7)
    ax.text(0.96, 0.92, rf"$\times / + = {POL_RATIO:.1f}$",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor=C_REF,
                      boxstyle="round,pad=0.3", alpha=0.9))
    ax.set_ylim(0, 4.0)

    # ── (d) SJ entropy decorrelation ─────────────────────────────────
    ax = axes[1, 1]
    # Summary statistic (no per-seed scatter data available)
    ax.text(0.50, 0.58,
            rf"$r(\mathrm{{CJ}},\,\Delta S) = {CORR_CJ_ENTROPY:.3f}$",
            transform=ax.transAxes, fontsize=11, ha="center", va="center",
            bbox=dict(facecolor=C_BAND, edgecolor=C_THEORY,
                      boxstyle="round,pad=0.5", alpha=0.8))
    ax.text(0.50, 0.28, "CJ independent\nof entanglement entropy",
            transform=ax.transAxes, fontsize=7, ha="center", va="center",
            color=C_REF, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove spines for the text-only panel
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.5)
    outpath = FIGURES_DIR / "fig_cj_diagnostics.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/6] {outpath.name}")


# =====================================================================
#  Figure 5: Derivation chain flowchart
# =====================================================================
def fig_cj_derivation():
    """Derivation chain: BD -> BD^2 -> vacuum avg -> E^2 -> CJ.

    Shows the factorization of the CJ observable into combinatorial
    and geometric factors, with conditions A (N-scaling) and B (eps saturation).
    """
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.set_xlim(-0.3, 10.7)
    ax.set_ylim(-0.9, 2.0)
    ax.axis("off")

    # Box definitions: (x_center, y_center, width, height, label, colour)
    boxes = [
        (0.5,  0.5, 1.4, 0.8, r"$\hat{B}_D$",                      C_THEORY),
        (2.5,  0.5, 1.5, 0.8, r"$\hat{B}_D^2$",                    C_THEORY),
        (4.8,  0.5, 1.8, 0.8, r"$\langle\cdot\rangle_{\rm vac}$",  "#E0E0E0"),
        (7.0,  0.5, 1.4, 0.8, r"$\mathcal{E}^2$",                  C_PRED),
        (9.3,  0.5, 1.6, 0.8, r"$\langle\mathrm{CJ}\rangle$",     C_GREEN),
    ]

    for (x, y, w, h, text, color) in boxes:
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="k",
                             linewidth=0.8, alpha=0.25, zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=2)

    # Arrows between consecutive boxes
    arrow_x = [
        (1.2,  1.75),   # BD -> BD^2
        (3.25, 3.9),    # BD^2 -> vac
        (5.7,  6.3),    # vac -> E^2
        (7.7,  8.5),    # E^2 -> CJ
    ]
    for (x1, x2) in arrow_x:
        arrow = FancyArrowPatch((x1, 0.5), (x2, 0.5),
                                arrowstyle="-|>", mutation_scale=10,
                                lw=1.0, color=C_MAIN, zorder=3)
        ax.add_patch(arrow)

    # Step annotations above each arrow
    annots = [
        (1.50, 1.10, r"$c_4 = 4/\sqrt{6}$"),
        (3.55, 1.10, r"$d{=}4$, Wick"),
        (6.00, 1.10, r"$\times\, 1/9!$"),
        (8.10, 1.10, r"$\times\, \pi^2/45$"),
    ]
    for (x, y, text) in annots:
        ax.text(x, y, text, ha="center", va="bottom", fontsize=6.5,
                color=C_REF)

    # Conditions banner
    ax.text(5.2, -0.50,
            r"Conditions: (A) $N^{8/9}$ link scaling  "
            r"(B) $\varepsilon \geq 3$ saturation",
            ha="center", va="top", fontsize=6, color=C_REF, style="italic")

    fig.tight_layout(pad=0.2)
    outpath = FIGURES_DIR / "fig_cj_derivation.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5/6] {outpath.name}")


# =====================================================================
#  Figure 6: Validity range — ratio vs xi = eps / N^{1/4}
# =====================================================================
def fig_cj_validity():
    r"""Ratio CJ_meas / CJ_pred vs xi = eps / N^{1/4}."""
    ratio = CJ_MEAN / CJ_PRED
    ratio_err = CJ_SE / CJ_PRED

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # Validity band xi in [0.25, 0.55]
    ax.axvspan(0.25, 0.55, color="#C8E6C9", alpha=0.35, zorder=0,
               label="Sweet spot")

    # Reference line
    ax.axhline(1.0, color=C_MAIN, ls="--", lw=0.7, zorder=1)

    # Data with error bars
    ax.errorbar(XI_DATA, ratio, yerr=ratio_err,
                fmt="o", ms=5, color=C_DATA, ecolor=C_DATA,
                capsize=2.5, capthick=0.8, lw=1.0, zorder=3)

    # Annotate selected points with N values
    label_set = {500, 2000, 5000, 10000, 15000}
    for i, n in enumerate(N_DATA):
        if n in label_set:
            dy = 8 if ratio[i] >= 1.0 else -12
            ax.annotate(rf"$N\!={n}$", (XI_DATA[i], ratio[i]),
                        textcoords="offset points", xytext=(6, dy),
                        fontsize=5.5, color=C_REF)

    ax.set_xlabel(r"$\xi = \varepsilon / N^{1/4}$")
    ax.set_ylabel(r"$\mathrm{CJ}_{\mathrm{meas}} \,/\, \mathrm{CJ}_{\mathrm{pred}}$")
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(0.88, 1.12)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True, framealpha=0.9)

    fig.tight_layout(pad=0.3)
    outpath = FIGURES_DIR / "fig_cj_validity.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6/6] {outpath.name}")


# =====================================================================
#  Main entry point
# =====================================================================
def main():
    print("Generating CJ bridge formula figures...")
    print(f"  Output: {FIGURES_DIR}")
    print(f"  Analytical prefactor C_0 = 32*pi^2/(3*9!*45) = {PREFACTOR:.6e}")
    print(f"  E^2 = {E_SQUARED},  T = {T}")
    print()

    fig_cj_ratio()
    fig_cj_nscaling()
    fig_cj_epsilon()
    fig_cj_diagnostics()
    fig_cj_derivation()
    fig_cj_validity()

    print()
    print("All 6 figures generated successfully.")


if __name__ == "__main__":
    main()
