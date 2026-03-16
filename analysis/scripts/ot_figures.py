# ruff: noqa: E402, I001
"""
OT Optical Theorem -- Publication Figures.

Generates 4 figures as PDF in analysis/figures/ot/:
  1. ot_spectral_positivity.pdf  -- Im[G_dressed(s)] vs s, positivity at all points
  2. ot_optical_theorem.pdf      -- Im[T(s)] vs s, non-negativity verification
  3. ot_fakeon_vs_feynman.pdf    -- Fakeon vs Feynman prescription comparison
  4. ot_stelle_comparison.pdf    -- SCT width vs Stelle width comparison

Execute:  python analysis/scripts/ot_figures.py
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

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = ANALYSIS_DIR / "figures" / "ot"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ANALYSIS_DIR / "results" / "ot"
RESULTS_FILE = RESULTS_DIR / "ot_optical_theorem_results.json"

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
# Load results
# ---------------------------------------------------------------------------
def load_results() -> dict:
    """Load the OT optical theorem results JSON."""
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ===================================================================
# Figure 1: Spectral positivity -- Im[G_dressed(s)] vs s
# ===================================================================

def figure_spectral_positivity(data: dict):
    """Plot Im[G_dressed(s)] vs s showing positivity at all test points."""
    print("  Generating ot_spectral_positivity.pdf ...")

    step4 = data["step4_spectral_positivity"]
    test_points = step4["test_points"]

    s_vals = [pt["s"] for pt in test_points]
    im_g_vals = [pt["Im_G_dressed"] for pt in test_points]
    near_ghost = [pt["near_ghost_pole"] for pt in test_points]

    # Ghost mass location
    z_L = 1.2807022780634851

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot Im[G_dressed] on log-log scale
    # Separate near-ghost point for distinct marker
    s_regular = [s for s, ng in zip(s_vals, near_ghost) if not ng]
    im_g_regular = [v for v, ng in zip(im_g_vals, near_ghost) if not ng]
    s_ghost = [s for s, ng in zip(s_vals, near_ghost) if ng]
    im_g_ghost = [v for v, ng in zip(im_g_vals, near_ghost) if ng]

    ax.semilogy(s_regular, im_g_regular, "o-", color=SCT_COLORS["prediction"],
                markersize=6, lw=1.5, label=r"Im[$G_{\mathrm{dressed}}^{\mathrm{FK}}(s)$]")

    if s_ghost:
        ax.semilogy(s_ghost, im_g_ghost, "D", color=SCT_COLORS["dirac"],
                    markersize=9, zorder=6,
                    label=r"Near ghost pole ($s \approx m_2^2$)")

    # Ghost pole vertical line
    ax.axvline(z_L, color=SCT_COLORS["dirac"], ls=":", lw=1.0, alpha=0.6,
               label=fr"$m_2^2/\Lambda^2 = {z_L:.4f}$")

    # Positivity region shading
    ax.fill_between([min(s_vals) * 0.5, max(s_vals) * 2],
                    [1e-40, 1e-40], [1e-2, 1e-2],
                    alpha=0.03, color=SCT_COLORS["vector"],
                    label=None)

    # Annotations
    ax.annotate("Im[$G$] > 0 everywhere\n(unitarity satisfied)",
                xy=(5, 2.5e-10), fontsize=9, color=SCT_COLORS["vector"],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                          edgecolor=SCT_COLORS["vector"], alpha=0.8))

    ax.annotate("Peak at ghost pole\n(fakeon: PV prescription)",
                xy=(z_L, im_g_ghost[0] if im_g_ghost else 1e26),
                xytext=(3.5, 1e22),
                fontsize=8, color=SCT_COLORS["dirac"],
                arrowprops=dict(arrowstyle="->", color=SCT_COLORS["dirac"],
                                lw=0.8))

    ax.set_xlabel(r"$s / \Lambda^2$")
    ax.set_ylabel(r"Im[$G_{\mathrm{dressed}}(s)$]  (arb. units)")
    ax.set_title("Spectral Positivity of the Dressed Graviton Propagator")
    ax.set_xlim(0.05, 150)
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False)
    ax.grid(True, alpha=0.3)

    path = FIG_DIR / "ot_spectral_positivity.pdf"
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Figure 2: Optical theorem -- Im[T(s)] vs s
# ===================================================================

def figure_optical_theorem(data: dict):
    """Plot Im[T(s)] vs s showing non-negativity (optical theorem)."""
    print("  Generating ot_optical_theorem.pdf ...")

    step6 = data["step6_optical_theorem"]
    checks = step6["checks"]

    s_vals = np.array([pt["s"] for pt in checks])
    im_t_lhs = np.array([pt["Im_T_lhs"] for pt in checks])
    im_t_rhs = np.array([pt["Im_T_rhs"] for pt in checks])

    # Ghost mass for reference
    z_L = 1.2807022780634851

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(6, 5), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # Main panel: Im[T] from LHS and RHS on log scale
    ax_main.semilogy(s_vals, im_t_lhs, "o-", color=SCT_COLORS["prediction"],
                     markersize=5, lw=1.5,
                     label=r"LHS: Im[$T(s)$] = Im[$\Sigma$]/$|s\Pi_{TT}|^2$")
    ax_main.semilogy(s_vals, im_t_rhs, "x--", color=SCT_COLORS["scalar"],
                     markersize=7, lw=1.0,
                     label=r"RHS: $\sum_{\mathrm{SM}} |M|^2 d\Phi$")

    # Ghost pole
    ax_main.axvline(z_L, color=SCT_COLORS["dirac"], ls=":", lw=1.0, alpha=0.6)
    ax_main.annotate(r"$m_2^2/\Lambda^2$", xy=(z_L, 1e-18),
                     xytext=(z_L * 3, 1e-18), fontsize=8,
                     color=SCT_COLORS["dirac"],
                     arrowprops=dict(arrowstyle="->",
                                     color=SCT_COLORS["dirac"], lw=0.8))

    # Non-negativity band
    ax_main.annotate("Im[$T(s)$] $\\geq 0$  for all $s > 0$",
                     xy=(20, 1e-15), fontsize=9, color=SCT_COLORS["vector"],
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                               edgecolor=SCT_COLORS["vector"], alpha=0.8))

    ax_main.set_ylabel(r"Im[$T(s)$]  (arb. units)")
    ax_main.set_title("One-Loop Optical Theorem Verification")
    ax_main.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xscale("log")

    # Residual panel: relative difference (LHS - RHS) / LHS
    # All should be zero to machine precision
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(
            im_t_lhs > 0,
            np.abs(im_t_lhs - im_t_rhs) / im_t_lhs,
            0.0,
        )

    ax_resid.semilogy(s_vals, rel_diff + 1e-50, "s", color=SCT_COLORS["total"],
                      markersize=5)
    ax_resid.axhline(1e-15, color=SCT_COLORS["reference"], ls="--", lw=0.8,
                     label=r"$10^{-15}$ (machine precision)")
    ax_resid.set_xlabel(r"$s / \Lambda^2$")
    ax_resid.set_ylabel(r"$|\Delta|$ / Im[$T$]")
    ax_resid.set_ylim(1e-50, 1e0)
    ax_resid.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False)
    ax_resid.grid(True, alpha=0.3)

    path = FIG_DIR / "ot_optical_theorem.pdf"
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Figure 3: Fakeon vs Feynman prescription comparison
# ===================================================================

def figure_fakeon_vs_feynman(data: dict):
    """Compare fakeon vs Feynman prescription for ghost contribution."""
    print("  Generating ot_fakeon_vs_feynman.pdf ...")

    step5 = data["step5_fakeon_vs_feynman"]
    points = step5["comparison_points"]

    s_vals = np.array([pt["s_over_Lambda2"] for pt in points])
    im_matter = np.array([pt["Im_Sigma_matter"] for pt in points])
    im_total_fey = np.array([pt["Im_Sigma_total_Feynman"] for pt in points])
    im_total_fk = np.array([pt["Im_Sigma_total_Fakeon"] for pt in points])

    # Ghost threshold: s = 4*m_ghost^2 = 4*z_L*Lambda^2
    z_L = 1.2807022780634851
    ghost_threshold = 4 * z_L

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 5.5), height_ratios=[2, 1],
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # Top panel: Im[Sigma] from matter and total (both prescriptions)
    ax_top.semilogy(s_vals, im_matter, "o-", color=SCT_COLORS["vector"],
                    markersize=5, lw=1.5,
                    label=r"Im[$\Sigma_{\mathrm{matter}}$] (SM loops)")
    ax_top.semilogy(s_vals, im_total_fk, "D--", color=SCT_COLORS["prediction"],
                    markersize=6, lw=1.2,
                    label=r"Im[$\Sigma_{\mathrm{FK}}$] (fakeon)")
    ax_top.semilogy(s_vals, np.abs(im_total_fey), "s:", color=SCT_COLORS["dirac"],
                    markersize=5, lw=1.2,
                    label=r"|Im[$\Sigma_{\mathrm{Fey}}$]| (Feynman)")

    # Ghost threshold line
    ax_top.axvline(ghost_threshold, color=SCT_COLORS["reference"], ls="--",
                   lw=0.8, alpha=0.7)
    ax_top.annotate(r"$s = 4m_{\mathrm{ghost}}^2$",
                    xy=(ghost_threshold, im_matter[0]),
                    xytext=(ghost_threshold * 2, im_matter[0] * 3),
                    fontsize=8, color=SCT_COLORS["reference"],
                    arrowprops=dict(arrowstyle="->",
                                    color=SCT_COLORS["reference"], lw=0.8))

    # Annotation boxes
    ax_top.annotate("Below threshold:\nidentical",
                    xy=(0.5, 3e-8), fontsize=7, color=SCT_COLORS["reference"],
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5",
                              edgecolor="#CCCCCC", alpha=0.8))

    ax_top.annotate("Above threshold:\nghost subtracted (Feynman)\nghost excluded (fakeon)",
                    xy=(60, 2e-4), fontsize=7, color=SCT_COLORS["dirac"],
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE",
                              edgecolor=SCT_COLORS["dirac"], alpha=0.8))

    ax_top.set_ylabel(r"Im[$\Sigma(s)$]  (arb. units)")
    ax_top.set_title("Fakeon vs Feynman: Ghost Contribution to Absorptive Part")
    ax_top.legend(loc="lower right", fontsize=7, frameon=True, fancybox=False)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_xscale("log")

    # Bottom panel: ghost-to-matter ratio
    ghost_ratio = np.array([pt["ghost_over_matter_ratio"] for pt in points])
    asymptotic = step5["ghost_to_matter_ratio_asymptotic"]

    ax_bot.plot(s_vals, ghost_ratio * 100, "o-", color=SCT_COLORS["dirac"],
                markersize=5, lw=1.5,
                label=r"|Im[$\Sigma_{\mathrm{ghost}}$]| / Im[$\Sigma_{\mathrm{matter}}$]")
    ax_bot.axhline(asymptotic * 100, color=SCT_COLORS["reference"], ls="--",
                   lw=0.8,
                   label=f"Asymptotic: {asymptotic * 100:.1f}%")

    # Ghost threshold
    ax_bot.axvline(ghost_threshold, color=SCT_COLORS["reference"], ls="--",
                   lw=0.8, alpha=0.7)

    ax_bot.set_xlabel(r"$s / \Lambda^2$")
    ax_bot.set_ylabel("Ghost/Matter (%)")
    ax_bot.set_ylim(-0.5, 8)
    ax_bot.legend(loc="upper left", fontsize=7, frameon=True, fancybox=False)
    ax_bot.grid(True, alpha=0.3)

    path = FIG_DIR / "ot_fakeon_vs_feynman.pdf"
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Figure 4: SCT vs Stelle width comparison
# ===================================================================

def figure_stelle_comparison(data: dict):
    """Bar chart comparing SCT vs Stelle gravity ghost properties and widths."""
    print("  Generating ot_stelle_comparison.pdf ...")

    step9 = data["step9_stelle_comparison"]
    stelle = step9["stelle_ghost"]
    sct = step9["sct_ghost"]
    comp = step9["comparison"]

    fig, (ax_props, ax_width) = plt.subplots(
        1, 2, figsize=(7.5, 3.8),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # --- Left panel: Ghost properties comparison ---
    categories = [
        r"$|z_{\mathrm{ghost}}|$",
        r"$|R|$ (residue)",
        r"$m_{\mathrm{ghost}} / \Lambda$",
    ]
    stelle_vals = [
        abs(stelle["z"]),
        abs(stelle["R"]),
        stelle["m_over_Lambda"],
    ]
    sct_vals = [
        abs(sct["z"]),
        abs(sct["R"]),
        sct["m_over_Lambda"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars_stelle = ax_props.bar(
        x - width / 2, stelle_vals, width,
        label="Stelle gravity", color=SCT_COLORS["reference"], alpha=0.7,
    )
    bars_sct = ax_props.bar(
        x + width / 2, sct_vals, width,
        label="SCT (nonlocal)", color=SCT_COLORS["prediction"], alpha=0.9,
    )

    # Value labels on bars
    for bar in bars_stelle:
        ax_props.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7,
            color=SCT_COLORS["reference"],
        )
    for bar in bars_sct:
        ax_props.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7,
            color=SCT_COLORS["prediction"],
        )

    # Suppression percentages
    for i in range(len(categories)):
        pct = (1 - sct_vals[i] / stelle_vals[i]) * 100
        mid_y = (stelle_vals[i] + sct_vals[i]) / 2
        ax_props.annotate(
            f"$-${pct:.0f}%", xy=(x[i] + 0.3, mid_y),
            fontsize=8, color=SCT_COLORS["total"], ha="left",
        )

    ax_props.set_xticks(x)
    ax_props.set_xticklabels(categories, fontsize=8)
    ax_props.set_ylabel("Value")
    ax_props.set_title("Ghost Properties")
    ax_props.legend(loc="upper left", fontsize=7, frameon=True, fancybox=False)
    ax_props.grid(True, axis="y", alpha=0.3)
    ax_props.set_ylim(0, max(stelle_vals) + 1)

    # --- Right panel: Width ratio ---
    width_ratio = comp["width_ratio_SCT_Stelle"]
    residue_ratio = comp["residue_ratio_abs"]
    mass_ratio = comp["mass_ratio_SCT_Stelle"]

    ratios = [mass_ratio, residue_ratio, width_ratio]
    ratio_labels = [
        r"$m_{\mathrm{SCT}} / m_{\mathrm{Stelle}}$",
        r"$|R_{\mathrm{SCT}}| / |R_{\mathrm{Stelle}}|$",
        r"$\Gamma_{\mathrm{SCT}} / \Gamma_{\mathrm{Stelle}}$",
    ]
    bar_colors = [
        SCT_COLORS["scalar"],
        SCT_COLORS["vector"],
        SCT_COLORS["prediction"],
    ]

    y_pos = np.arange(len(ratios))
    bars = ax_width.barh(y_pos, ratios, 0.5, color=bar_colors, alpha=0.85)

    # Value labels
    for bar, val in zip(bars, ratios):
        ax_width.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", ha="left", va="center", fontsize=8,
            color=SCT_COLORS["total"],
        )

    # Reference line at 1.0 (Stelle = Stelle)
    ax_width.axvline(1.0, color=SCT_COLORS["reference"], ls="--", lw=0.8,
                     alpha=0.7, label="Stelle = 1")

    ax_width.set_yticks(y_pos)
    ax_width.set_yticklabels(ratio_labels, fontsize=8)
    ax_width.set_xlabel("Ratio (SCT / Stelle)")
    ax_width.set_title("SCT Suppression vs Stelle")
    ax_width.set_xlim(0, 1.2)
    ax_width.grid(True, axis="x", alpha=0.3)

    # Unitarity status box
    status_text = (
        "Stelle (Feynman): VIOLATED\n"
        "Stelle (fakeon): satisfied\n"
        "SCT (fakeon): SATISFIED"
    )
    ax_width.text(
        0.55, 0.05, status_text, fontsize=6.5, transform=ax_width.transAxes,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                  edgecolor=SCT_COLORS["prediction"], alpha=0.9),
    )

    path = FIG_DIR / "ot_stelle_comparison.pdf"
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")
    return path


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("OT OPTICAL THEOREM -- PUBLICATION FIGURES")
    print("=" * 60)

    data = load_results()
    print(f"Loaded results from: {RESULTS_FILE}")
    print(f"Output directory: {FIG_DIR}")
    print()

    paths = []
    paths.append(figure_spectral_positivity(data))
    paths.append(figure_optical_theorem(data))
    paths.append(figure_fakeon_vs_feynman(data))
    paths.append(figure_stelle_comparison(data))

    print()
    print("=" * 60)
    print(f"Generated {len(paths)} figures in {FIG_DIR}")
    for p in paths:
        print(f"  {p.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
