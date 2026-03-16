# ruff: noqa: E402, I001
"""
MR-7 Figures: Publication-quality plots for graviton scattering in SCT.

Generates:
  Fig 1: Propagator modification Pi_TT(z) for SCT vs Stelle vs GR
  Fig 2: One-loop correction ratio |delta M|/|M_tree| vs sqrt(s)/Lambda
  Fig 3: Ward identity violation (machine-precision zero)

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

from scripts.mr1_lorentzian import Pi_TT_complex
from scripts.mr7_scattering import (
    LOCAL_C2,
    ward_identity_tree_level,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr7"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Try to use publication style
try:
    from sct_tools.plotting import SCT_COLORS, init_style
    init_style()
    _HAS_STYLE = True
except Exception:
    _HAS_STYLE = False
    SCT_COLORS = {
        "total": "#1f77b4",
        "prediction": "#d62728",
        "reference": "#2ca02c",
        "scalar": "#9467bd",
        "data": "#ff7f0e",
    }


def fig1_propagator_comparison():
    """
    Fig 1: Pi_TT(z) for SCT, Stelle, and GR.

    Shows how the propagator denominator evolves with z = k^2/Lambda^2.
    GR is flat (Pi = 1), Stelle is linear, SCT is nonlocal.
    """
    mp.mp.dps = 30

    z_values = np.linspace(0.01, 15.0, 500)
    pi_sct = []
    pi_stelle = []
    pi_gr = []

    for z in z_values:
        pi_sct.append(float(mp.re(Pi_TT_complex(mp.mpf(z), dps=30))))
        pi_stelle.append(1.0 + float(LOCAL_C2) * z)
        pi_gr.append(1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_values, pi_gr, "--", color="gray", lw=1.5, label=r"GR ($\Pi = 1$)")
    ax.plot(z_values, pi_stelle, "-.", color=SCT_COLORS["reference"], lw=1.5,
            label=r"Stelle ($1 + c_2 z$)")
    ax.plot(z_values, pi_sct, "-", color=SCT_COLORS["total"], lw=2,
            label=r"SCT ($\Pi_{\mathrm{TT}}(z)$)")

    ax.axhline(y=0, color="black", lw=0.5, ls=":")
    ax.axhline(y=-83.0/6, color=SCT_COLORS["prediction"], lw=0.8, ls=":",
               label=r"$\Pi_{\mathrm{TT}}(\infty) = -83/6$", alpha=0.7)

    # Mark ghost pole
    ax.axvline(x=2.4148, color=SCT_COLORS["data"], lw=0.8, ls="--", alpha=0.7)
    ax.annotate(r"$z_0 = 2.415$", xy=(2.4148, 0), xytext=(3.5, 2),
                arrowprops=dict(arrowstyle="->", color=SCT_COLORS["data"]),
                fontsize=10, color=SCT_COLORS["data"])

    ax.set_xlabel(r"$z = k^2/\Lambda^2$", fontsize=12)
    ax.set_ylabel(r"$\Pi_{\mathrm{TT}}(z)$", fontsize=12)
    ax.set_title("Graviton Propagator Denominator: SCT vs Stelle vs GR", fontsize=13)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(0, 15)
    ax.set_ylim(-15, 5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = FIGURES_DIR / "mr7_fig1_propagator_comparison.pdf"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Fig 1 saved: {outpath}")
    return outpath


def fig2_one_loop_correction():
    """
    Fig 2: One-loop correction ratio vs sqrt(s)/Lambda.

    Shows |delta M|^2 / |M_tree|^2 for various Lambda/M_Pl ratios.
    """
    sqrt_s_over_Lambda = np.linspace(0.01, 3.0, 200)
    z_values = sqrt_s_over_Lambda**2  # z = s/Lambda^2

    fig, ax = plt.subplots(figsize=(8, 5))

    lambda_ratios = [1e-3, 1e-5, 1e-10, 1e-17]
    colors = [SCT_COLORS["prediction"], SCT_COLORS["total"],
              SCT_COLORS["reference"], SCT_COLORS["scalar"]]

    for lr, color in zip(lambda_ratios, colors):
        # delta ~ kappa^2 * s / (16*pi^2) + (Lambda/M_Pl)^2
        # kappa^2 * s = 16*pi*G * s = 16*pi * z * Lambda^2 / M_Pl^2 = 16*pi * z * lr^2
        delta = 16 * np.pi * z_values * lr**2 / (16 * np.pi**2) + lr**2
        delta = np.clip(delta, 1e-50, None)  # avoid log(0)
        ax.semilogy(sqrt_s_over_Lambda, delta, "-", color=color, lw=1.5,
                     label=rf"$\Lambda/M_{{\mathrm{{Pl}}}} = 10^{{{int(np.log10(lr))}}}$")

    ax.axhline(y=1, color="black", lw=1, ls="--", alpha=0.5, label="Perturbativity bound")
    ax.set_xlabel(r"$\sqrt{s}/\Lambda$", fontsize=12)
    ax.set_ylabel(r"$|\delta\mathcal{M}|^2 / |\mathcal{M}_{\mathrm{tree}}|^2$", fontsize=12)
    ax.set_title("One-Loop Graviton Scattering Correction", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 3)
    ax.set_ylim(1e-40, 1e2)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    outpath = FIGURES_DIR / "mr7_fig2_one_loop_correction.pdf"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Fig 2 saved: {outpath}")
    return outpath


def fig3_ward_identity():
    """
    Fig 3: Ward identity violation vs k^2.

    Should be machine-precision zero for all momenta.
    """
    # Generate momenta with varying k^2
    k_magnitudes = np.logspace(-1, 3, 50)
    violations_P2 = []
    violations_P0s = []

    for km in k_magnitudes:
        k = np.array([km, km * 0.3, km * 0.4, km * 0.5])
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        k2 = float(k @ eta @ k)

        if abs(k2) < 1e-20:
            violations_P2.append(0.0)
            violations_P0s.append(0.0)
            continue

        eps = np.zeros((4, 4))
        result = ward_identity_tree_level(k, eps, dps=30)
        violations_P2.append(result["ward_P2_max_violation"])
        violations_P0s.append(result["ward_P0s_max_violation"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(k_magnitudes, violations_P2, "o-", color=SCT_COLORS["total"],
                ms=3, lw=1, label=r"$k^\mu P^{(2)}_{\mu\nu,\rho\sigma}$")
    ax.semilogy(k_magnitudes, violations_P0s, "s-", color=SCT_COLORS["prediction"],
                ms=3, lw=1, label=r"$k^\mu P^{(0-s)}_{\mu\nu,\rho\sigma}$")

    ax.axhline(y=1e-10, color="gray", ls="--", alpha=0.5, label=r"$10^{-10}$ threshold")
    ax.set_xlabel(r"$|k|$", fontsize=12)
    ax.set_ylabel("Ward identity violation (max element)", fontsize=12)
    ax.set_title("Ward Identity Check: Tree-Level Graviton Amplitude", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(k_magnitudes[0], k_magnitudes[-1])
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    outpath = FIGURES_DIR / "mr7_fig3_ward_identity.pdf"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Fig 3 saved: {outpath}")
    return outpath


def generate_all_figures():
    """Generate all MR-7 figures."""
    print("\n--- MR-7 Figure Generation ---")
    paths = []
    paths.append(fig1_propagator_comparison())
    paths.append(fig2_one_loop_correction())
    paths.append(fig3_ward_identity())
    print(f"\n  Total: {len(paths)} figures generated in {FIGURES_DIR}")
    return paths


# ===================================================================
# Self-test (CQ3)
# ===================================================================

def self_test() -> bool:
    """Self-test: verify all figures can be generated."""
    print("\n=== MR-7 FIGURES SELF-TEST ===")
    try:
        paths = generate_all_figures()
        all_exist = all(p.exists() for p in paths)
        if all_exist:
            print("  [PASS] All figures generated successfully")
            return True
        else:
            print("  [FAIL] Some figures missing")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MR-7 Figures")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        ok = self_test()
        import sys as _sys
        _sys.exit(0 if ok else 1)
    else:
        generate_all_figures()
