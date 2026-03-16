# ruff: noqa: E402, I001
"""
MR-3 Figures: Causality analysis visualizations.

Generates publication-quality figures for the MR-3 causality analysis:
  1. Retarded propagator vs spacetime coordinates (light-cone structure)
  2. Kramers-Kronig validation (Im[Pi_TT] on real axis)
  3. Front velocity vs frequency
  4. Macrocausality: acausal decay vs distance
  5. Stelle vs SCT potential comparison

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

from scripts.mr3_causality import (
    RL_LORENTZIAN,
    Z0_EUCLIDEAN,
    ZL_LORENTZIAN,
    compare_stelle_sct,
    front_velocity,
    kramers_kronig_check,
    macrocausality_bound,
    retarded_propagator_fakeon_1d,
    retarded_propagator_massive_1d,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr3"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# SciencePlots style if available (with no-latex fallback)
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
except (ImportError, OSError):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "text.usetex": False,
    })

# SCT color scheme
SCT_COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#059669",
    "warning": "#D97706",
    "neutral": "#6B7280",
    "dark": "#1F2937",
}


def fig1_retarded_propagator():
    """
    Figure 1: Retarded vs fakeon propagator in position space.

    Shows the light-cone structure: G_ret vanishes for spacelike separations,
    while G_FK (fakeon) has a nonzero tail violating microcausality.
    """
    m_ghost = float(mp.sqrt(abs(ZL_LORENTZIAN)))  # ~ 1.132
    m2 = float(mp.sqrt(abs(Z0_EUCLIDEAN)))  # ~ 1.554

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): G_ret and G_FK vs time at fixed r
    r_fixed = 0.5
    t_values = np.linspace(-4, 6, 500)
    g_ret = [retarded_propagator_massive_1d(t, r_fixed, m_ghost) for t in t_values]
    g_fk = [retarded_propagator_fakeon_1d(t, r_fixed, m_ghost) for t in t_values]

    ax = axes[0]
    ax.plot(t_values, g_ret, color=SCT_COLORS["primary"], linewidth=1.5,
            label=r"$G_{\mathrm{ret}}$ (causal)")
    ax.plot(t_values, g_fk, color=SCT_COLORS["secondary"], linewidth=1.5,
            linestyle="--", label=r"$G_{\mathrm{FK}}$ (fakeon)")
    ax.axvline(x=r_fixed, color=SCT_COLORS["neutral"], linestyle=":",
               linewidth=0.8, alpha=0.7, label=f"Light cone ($t = r = {r_fixed}$)")
    ax.axvline(x=-r_fixed, color=SCT_COLORS["neutral"], linestyle=":",
               linewidth=0.8, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.3)
    ax.set_xlabel(r"$t\Lambda$")
    ax.set_ylabel(r"$G(t, r)$")
    ax.set_title(f"(a) Propagator vs time ($r\\Lambda = {r_fixed}$, $m = \\sqrt{{|z_L|}}\\Lambda$)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-4, 6)

    # Panel (b): G_FK for t < 0 (acausal region)
    t_neg = np.linspace(-5, -0.1, 200)
    g_fk_neg = [retarded_propagator_fakeon_1d(t, r_fixed, m_ghost) for t in t_neg]

    ax = axes[1]
    ax.plot(t_neg, g_fk_neg, color=SCT_COLORS["secondary"], linewidth=1.5)
    ax.axhline(y=0, color="black", linewidth=0.3)
    ax.fill_between(t_neg, 0, g_fk_neg, alpha=0.15, color=SCT_COLORS["secondary"])
    ax.set_xlabel(r"$t\Lambda$")
    ax.set_ylabel(r"$G_{\mathrm{FK}}(t < 0, r)$")
    ax.set_title("(b) Acausal fakeon tail ($t < 0$)")
    ax.annotate("Microcausality\nviolation", xy=(-2, max(g_fk_neg) * 0.5),
                fontsize=8, ha="center", color=SCT_COLORS["secondary"])

    plt.tight_layout()
    out = FIGURES_DIR / "mr3_retarded_propagator.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def fig2_kramers_kronig():
    """
    Figure 2: Kramers-Kronig validation.

    Shows Im[Pi_TT(z)] = 0 on the real axis (both Euclidean and Lorentzian).
    """
    kk = kramers_kronig_check(omega_values=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0,
                                             2.0, 5.0, 10.0, 30.0, 50.0, 100.0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Im[Pi_TT] on Euclidean axis
    ax = axes[0]
    omegas = [e["omega"] for e in kk["euclidean_axis"]]
    im_vals = [abs(e["Im_Pi_TT"]) for e in kk["euclidean_axis"]]
    re_vals = [e["Re_Pi_TT"] for e in kk["euclidean_axis"]]
    ax.semilogy(omegas, [max(v, 1e-25) for v in im_vals], "o-",
                color=SCT_COLORS["secondary"], markersize=4,
                label=r"$|\mathrm{Im}\,\Pi_{\mathrm{TT}}|$")
    ax.axhline(y=1e-20, color=SCT_COLORS["neutral"], linestyle="--",
               linewidth=0.8, label=r"$10^{-20}$ threshold")
    ax.set_xlabel(r"$\omega/\Lambda$")
    ax.set_ylabel(r"$|\mathrm{Im}\,\Pi_{\mathrm{TT}}(z)|$")
    ax.set_title("(a) Euclidean axis: $z = \\omega^2/\\Lambda^2 > 0$")
    ax.legend(fontsize=8)
    ax.set_ylim(1e-25, 1)

    # Panel (b): Re[Pi_TT] on Euclidean axis
    ax = axes[1]
    ax.plot(omegas, re_vals, "s-", color=SCT_COLORS["primary"], markersize=4)
    ax.axhline(y=0, color="black", linewidth=0.3)
    ax.axhline(y=1, color=SCT_COLORS["neutral"], linestyle=":",
               linewidth=0.8, label="GR limit ($\\Pi = 1$)")
    ax.set_xlabel(r"$\omega/\Lambda$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Pi_{\mathrm{TT}}(z)$")
    ax.set_title("(b) Real part on Euclidean axis")
    ax.legend(fontsize=8)
    ax.set_xscale("log")

    plt.tight_layout()
    out = FIGURES_DIR / "mr3_kramers_kronig.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def fig3_front_velocity():
    """
    Figure 3: Phase velocity vs frequency.

    Shows v_phase -> 1 (speed of light) at all frequencies.
    The front velocity v_front = lim_{omega->inf} omega/k = c.
    """
    fv = front_velocity(Lambda=1.0, omega_max=20.0, n_points=30, dps=30)

    fig, ax = plt.subplots(figsize=(6, 4))

    omegas = [d["omega"] for d in fv["v_phase_data"]]
    v_ph = [d["v_phase"] for d in fv["v_phase_data"]]

    ax.plot(omegas, v_ph, "o-", color=SCT_COLORS["primary"], markersize=3,
            linewidth=1.5, label=r"$v_{\mathrm{phase}}(\omega) = \omega / \mathrm{Re}[k(\omega)]$")
    ax.axhline(y=1.0, color=SCT_COLORS["secondary"], linestyle="--",
               linewidth=1.0, label="$c = 1$ (speed of light)")
    ax.set_xlabel(r"$\omega / \Lambda$")
    ax.set_ylabel(r"$v_{\mathrm{phase}}$")
    ax.set_title("Front velocity analysis")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.5)

    # Add annotation
    ax.annotate(f"$v_{{\\mathrm{{front}}}} = {fv['v_front_estimate']:.4f}$",
                xy=(0.7, 0.15), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = FIGURES_DIR / "mr3_front_velocity.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def fig4_macrocausality():
    """
    Figure 4: Acausal contribution decay vs distance.

    Shows the exponential decay of the fakeon (PV) contribution
    as a function of distance from the source.
    """
    macro = macrocausality_bound(r_min=0.1, r_max=50.0)

    fig, ax = plt.subplots(figsize=(6, 4))

    r_vals = [d["r_Lambda"] for d in macro["decay_data"]]
    fk_ratio = [d["fakeon_ratio"] for d in macro["decay_data"]]
    yk_ratio = [d["yukawa_ratio"] for d in macro["decay_data"]]
    total = [d["total_deviation"] for d in macro["decay_data"]]

    ax.semilogy(r_vals, fk_ratio, "-", color=SCT_COLORS["secondary"],
                linewidth=1.5, label=r"Fakeon ($z_L$): $|R_L| e^{-m_L r}$")
    ax.semilogy(r_vals, yk_ratio, "-", color=SCT_COLORS["primary"],
                linewidth=1.5, label=r"Yukawa ($z_0$): $|R_0| e^{-m_2 r}$")
    ax.semilogy(r_vals, total, "--", color=SCT_COLORS["dark"],
                linewidth=1.0, label="Total deviation")

    # Mark threshold distances
    thresholds = {1e-3: "1 ppm", 1e-6: "$10^{-6}$", 1e-10: "$10^{-10}$"}
    for thresh, label in thresholds.items():
        ax.axhline(y=thresh, color=SCT_COLORS["neutral"], linestyle=":",
                   linewidth=0.5, alpha=0.5)
        ax.annotate(label, xy=(45, thresh * 1.5), fontsize=7,
                    color=SCT_COLORS["neutral"])

    ax.set_xlabel(r"$r \Lambda$")
    ax.set_ylabel("Deviation from Newton")
    ax.set_title("Macrocausality: acausal decay vs distance")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(1e-25, 1)

    plt.tight_layout()
    out = FIGURES_DIR / "mr3_macrocausality.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def fig5_stelle_comparison():
    """
    Figure 5: Stelle vs SCT modified Newtonian potential.
    """
    comp = compare_stelle_sct(
        r_values=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0,
                  5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        xi=0.0,
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    r_vals = [p["r_Lambda"] for p in comp["potentials"]]
    v_stelle = [p["V_Stelle_over_V_N"] for p in comp["potentials"]]
    v_sct = [p["V_SCT_local_over_V_N"] for p in comp["potentials"]]

    ax.plot(r_vals, v_stelle, "s-", color=SCT_COLORS["secondary"],
            markersize=3, linewidth=1.5, label="Stelle (Class II)")
    ax.plot(r_vals, v_sct, "o-", color=SCT_COLORS["primary"],
            markersize=3, linewidth=1.5, label="SCT (Class III)")
    ax.axhline(y=1.0, color=SCT_COLORS["neutral"], linestyle="--",
               linewidth=0.8, label="Newton (GR)")

    ax.set_xlabel(r"$r \Lambda$")
    ax.set_ylabel(r"$V(r) / V_N(r)$")
    ax.set_title(r"Modified Newtonian potential ($\xi = 0$)")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 1.2)

    plt.tight_layout()
    out = FIGURES_DIR / "mr3_stelle_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def generate_all_figures():
    """Generate all MR-3 figures."""
    print("Generating MR-3 figures...")
    paths = []
    paths.append(fig1_retarded_propagator())
    paths.append(fig2_kramers_kronig())
    paths.append(fig3_front_velocity())
    paths.append(fig4_macrocausality())
    paths.append(fig5_stelle_comparison())
    print(f"\n{len(paths)} figures generated in {FIGURES_DIR}")
    return paths


if __name__ == "__main__":
    generate_all_figures()
