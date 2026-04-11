# ruff: noqa: E402, I001
"""
MR-9 COMPARISON: SCT vs IDG vs Stelle vs Modesto vs Hayward.

Systematic comparison of black hole singularity resolution across
different modified gravity frameworks. All results are for the
LINEARIZED regime unless otherwise noted.

COMPARISON AXES:
1. Propagator UV behavior
2. Source smearing (effective energy density)
3. Core type (de Sitter, bounce, singular)
4. Kretschner scaling at r→0
5. Mass function scaling m(r) at r→0
6. Geodesic completeness
7. Ghost structure
8. Entire function order
9. Energy condition violation
10. Free parameters

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
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

from scripts.nt4a_propagator import (
    scalar_local_mass,
    spin2_local_mass,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr9"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr9"


# ===========================================================================
# FRAMEWORK DEFINITIONS
# ===========================================================================

@dataclass
class GravityFramework:
    """Description of a modified gravity framework for BH singularity."""
    name: str
    action_type: str
    propagator_uv: str
    entire_order: str
    source_smearing: str
    core_type: str
    K_scaling: str
    mass_function: str
    geodesic_complete: str
    ghost_structure: str
    nec_violation: str
    free_params: str
    singularity_status: str
    key_reference: str
    notes: str = ""


def build_frameworks() -> dict[str, GravityFramework]:
    """Build the comparison table of gravity frameworks."""

    sct = GravityFramework(
        name="SCT (Spectral Causal Theory)",
        action_type="S = R + α_C C² F₁(□/Λ²) + α_R R² F₂(□/Λ²)",
        propagator_uv="Π_TT → const ≈ -13.83 (φ~2/z, order 1)",
        entire_order="1 (φ = e^{-z/4}√(π/z)·erfi(√z/2))",
        source_smearing="NO (linearized: integral diverges as 1/r⁴)",
        core_type="NO de Sitter core (linearized). Mass: m(r) ~ a₁r (linear)",
        K_scaling="K ~ 1/r⁴ (softened from 1/r⁶, linearized)",
        mass_function="m(r) ~ M·(4m₂/3 - m₀/3)·r (LINEAR, not cubic)",
        geodesic_complete="INDETERMINATE (linearized: incomplete, nonlinear: unknown)",
        ghost_structure="1 real ghost (z_L=1.28), 3 Lee-Wick pairs, fakeon Rx",
        nec_violation="NEC SATISFIED everywhere (linearized). SEC status: open.",
        free_params="0 (α_C=13/120, Λ from experiment). ξ from NCG (1/6).",
        singularity_status="SOFTENED (linearized). K-T: Schw 1/r branch forbidden (Case 2), but milder singularity NOT excluded. Fakeon caveat.",
        key_reference="Alfyorov 2026 (DOI:10.5281/zenodo.19098042)",
        notes=(
            "Linearized analysis gives K~1/r⁴ (not resolved). "
            "Koshelev-Tokareva theorem (order 1 ≥ 1/2) implies singularity "
            "forbidden at nonperturbative level. Gap G1 (OP-01) blocks "
            "explicit nonlinear solution."
        ),
    )

    idg = GravityFramework(
        name="IDG (Infinite Derivative Gravity)",
        action_type="S = R + R F₁(□) R + C F₃(□) C, F_i ~ exp(-□/M²)",
        propagator_uv="Π ~ exp(k²/M²) → ∞ exponentially",
        entire_order="1 (standard exponential exp(-□/M²))",
        source_smearing="YES (Gaussian: ρ_eff ~ exp(-r²M²/4)/(πM²)^{3/2})",
        core_type="de Sitter core: A(r) ≈ 1 - r²/l² with l² ~ M²/(M_P² M⁴)",
        K_scaling="K → FINITE at r=0 (regular de Sitter core)",
        mass_function="m(r) ~ r³ (CUBIC, de Sitter)",
        geodesic_complete="YES (proven: Conroy-Mazumdar 2017)",
        ghost_structure="NO ghosts (exp has no zeros → no extra poles)",
        nec_violation="NEC violated in nonlocality region (r < 1/M)",
        free_params="1 (nonlocality scale M)",
        singularity_status="RESOLVED (model-specific nonlinear proof: Koshelev-Marto-Mazumdar 2018). Also Case 2 in K-T taxonomy.",
        key_reference="Biswas-Gerwick-Koivisto-Mazumdar 2012, arXiv:1110.5249; Koshelev-Marto-Mazumdar 2018, arXiv:1803.00309",
        notes=(
            "Ghost-free by construction (exp has no zeros). "
            "de Sitter core with Λ_eff ~ M·(M/M_P)². "
            "Also K-T Case 2 (order 1 < 3/2). Resolution proven by SEPARATE model-specific argument, not by general K-T taxonomy."
        ),
    )

    stelle = GravityFramework(
        name="Stelle Gravity (Quadratic, local)",
        action_type="S = R + α C² + β R² (4-derivative, local)",
        propagator_uv="Π ~ const (polynomial, degree 2)",
        entire_order="N/A (polynomial, not entire)",
        source_smearing="PARTIAL (improved UV, still divergent)",
        core_type="NO de Sitter core",
        K_scaling="K ~ 1/r⁴ (softened from 1/r⁶)",
        mass_function="m(r) ~ a₁r (linear, same as SCT Yukawa)",
        geodesic_complete="NO (finite proper time)",
        ghost_structure="1 massive ghost (Weyl), 1 massive scalar (R²)",
        nec_violation="SEC violated, NEC marginal",
        free_params="2 (α, β → masses m₂, m₀)",
        singularity_status="NOT RESOLVED (K still diverges at r=0)",
        key_reference="Stelle 1977, Phys. Rev. D 16, 953",
        notes=(
            "SCT's Yukawa approximation reduces to Stelle at tree level. "
            "The form factor F₁(□/Λ²) = 1 + O(□) reproduces Stelle as first term. "
            "SCT adds infinitely many higher derivatives through the entire function."
        ),
    )

    modesto = GravityFramework(
        name="Modesto (Super-renormalizable NLG)",
        action_type="S = R + R exp(-□^N/M^{2N}) R + ..., N ≥ 1",
        propagator_uv="Π ~ exp(k^{2N}/M^{2N}) → ∞ (order N ≥ 1)",
        entire_order="N (adjustable, typically N=1 or 2)",
        source_smearing="YES (Gaussian-type for N=1, sharper for N>1)",
        core_type="de Sitter core",
        K_scaling="K → FINITE (regular)",
        mass_function="m(r) ~ r³ (de Sitter) or r^{2N+1}",
        geodesic_complete="YES (for N ≥ 1)",
        ghost_structure="NO ghosts (entire exp has no zeros)",
        nec_violation="NEC violated in core region",
        free_params="2 (M, N)",
        singularity_status="RESOLVED",
        key_reference="Modesto 2012, Phys. Rev. D 86, 044005, arXiv:1107.2403",
        notes=(
            "SCT can be viewed as a specific realization with N=1 (order 1), "
            "but with a DIFFERENT entire function (erfi-based vs pure exp). "
            "The difference: SCT has ghost poles (fakeon Rx), Modesto doesn't."
        ),
    )

    hayward = GravityFramework(
        name="Hayward (Phenomenological regular BH)",
        action_type="No action principle. Metric ansatz: f = 1 - 2Mr²/(r³+2ML²)",
        propagator_uv="N/A (metric model, no QFT)",
        entire_order="N/A",
        source_smearing="YES (effective: ρ ~ L²M/(r³+2ML²)²)",
        core_type="de Sitter core: f ≈ 1 - r²/L² near r=0",
        K_scaling="K → 96/L⁴ (FINITE, pure de Sitter)",
        mass_function="m(r) ~ Mr³/(r³+2ML²) → r³ for small r",
        geodesic_complete="YES (de Sitter core is geodesically complete)",
        ghost_structure="N/A (no propagator)",
        nec_violation="NEC violated (weak energy condition also violated)",
        free_params="1 (regularization length L)",
        singularity_status="RESOLVED (by construction)",
        key_reference="Hayward 2006, Phys. Rev. Lett. 96, 031103",
        notes=(
            "Phenomenological model, not derived from an action principle. "
            "Serves as a TARGET: if SCT resolves the singularity, the "
            "resulting geometry should resemble Hayward/Bardeen with L ~ 1/Λ."
        ),
    )

    leewick = GravityFramework(
        name="Lee-Wick Gravity (6-derivative)",
        action_type="S = R + α□R + βC□C (6-derivative, complex poles)",
        propagator_uv="Π ~ k⁴ (polynomial, improved by 2 powers)",
        entire_order="N/A (polynomial truncation)",
        source_smearing="YES (oscillating Yukawa, multi-horizon)",
        core_type="Regular multi-horizon (oscillating potential)",
        K_scaling="K → FINITE (from complex pole pairs)",
        mass_function="m(r) ~ r³ (de Sitter-like core)",
        geodesic_complete="YES (regular interior)",
        ghost_structure="Complex conjugate ghost pairs (LW prescription)",
        nec_violation="NEC violated, oscillating corrections",
        free_params="2 (α, β → complex mass parameters)",
        singularity_status="RESOLVED (multi-horizon regular BH)",
        key_reference="Burzilla et al. 2023, arXiv:2308.12810",
        notes=(
            "Lee-Wick gravity is a TRUNCATION of the full nonlocal theory. "
            "SCT has 3 Lee-Wick pairs among 8 propagator zeros: similar mechanism. "
            "The fakeon prescription (Anselmi-Piva) handles the LW ghost pairs."
        ),
    )

    return {
        "SCT": sct,
        "IDG": idg,
        "Stelle": stelle,
        "Modesto": modesto,
        "Hayward": hayward,
        "Lee-Wick": leewick,
    }


# ===========================================================================
# NUMERICAL COMPARISON
# ===========================================================================

def numerical_comparison(*, Lambda=1.0, xi=0.0, dps=30):
    """Compute numerical values for the comparison table.

    For each framework, compute V(r)/V_N(r) and K(r) at selected radii.
    """
    mp.mp.dps = dps
    m2 = float(spin2_local_mass(mp.mpf(Lambda)))
    m0_mp = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))
    m0 = float(m0_mp) if m0_mp is not None else None

    import scipy.special

    r_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    GM = 0.1  # in natural units

    table = []
    for r in r_values:
        row = {"r_Lambda": r}

        # SCT Yukawa
        h_sct = 1 - 4/3 * np.exp(-m2 * r)
        if m0 is not None:
            h_sct += 1/3 * np.exp(-m0 * r)
        row["V_ratio_SCT"] = h_sct

        # IDG (erf)
        row["V_ratio_IDG"] = float(scipy.special.erf(r * Lambda / 2))

        # Stelle (same as SCT Yukawa at tree level for matching masses)
        row["V_ratio_Stelle"] = h_sct  # identical in Yukawa approximation

        # Schwarzschild
        row["V_ratio_GR"] = 1.0

        # Kretschner (approximate, using f = 1 - 2GM·h/r)
        r_s = 2 * GM
        if r > 0.001:
            # SCT
            f_sct = 1 - r_s * h_sct / r
            K_GR = 48 * GM ** 2 / r ** 6

            # Rough Kretschner ratio
            if abs(h_sct) > 1e-20:
                row["K_ratio_SCT_over_GR"] = h_sct ** 2  # leading order
            else:
                row["K_ratio_SCT_over_GR"] = 0.0

            # IDG
            erf_val = float(scipy.special.erf(r * Lambda / 2))
            row["K_ratio_IDG_over_GR"] = erf_val ** 2 if erf_val > 0 else 0.0
        else:
            row["K_ratio_SCT_over_GR"] = None
            row["K_ratio_IDG_over_GR"] = None

        table.append(row)

    return table


# ===========================================================================
# DISCRIMINATING TESTS
# ===========================================================================

def discriminating_properties():
    """Identify properties that DISTINGUISH SCT from other frameworks.

    These are the key axes along which SCT differs from IDG, Stelle, etc.
    """
    return {
        "SCT_vs_IDG": {
            "ghost_structure": (
                "SCT: 1 real ghost + 3 LW pairs (fakeon Rx). "
                "IDG: NO ghosts (exp has no zeros). "
                "DISCRIMINATING: ghost spectrum is measurable in principle "
                "via scattering cross-sections at E ~ Λ."
            ),
            "source_smearing": (
                "SCT (linearized): NO source smearing (Π_TT → const). "
                "IDG: Gaussian smearing (Π ~ exp). "
                "DISCRIMINATING: different mass function scaling at r→0 "
                "(linear vs cubic)."
            ),
            "parameters": (
                "SCT: 0 free parameters (α_C = 13/120 from SM, Λ from data). "
                "IDG: 1 free parameter (nonlocality scale M). "
                "DISCRIMINATING: SCT is predictive, IDG has freedom."
            ),
            "origin": (
                "SCT: derived from spectral action principle (NCG). "
                "IDG: postulated action with ghost-free condition. "
                "DISCRIMINATING: SCT has deeper theoretical motivation."
            ),
        },
        "SCT_vs_Stelle": {
            "UV_behavior": (
                "SCT: Π_TT has infinitely many terms (entire function). "
                "Stelle: Π = 1 + c₂z (truncation to first term). "
                "K-T theorem: SCT (order 1) may resolve singularity "
                "nonperturbatively; Stelle (order 0) cannot."
            ),
            "renormalizability": (
                "SCT: UV-finite in D²-quantization (MR-5 CHIRAL-Q). "
                "Stelle: renormalizable but NOT finite."
            ),
        },
        "SCT_vs_Hayward": {
            "action_principle": (
                "SCT: derived from spectral action Tr f(D²/Λ²). "
                "Hayward: phenomenological metric ansatz, no action. "
                "DISCRIMINATING: SCT is a QFT framework, Hayward is classical."
            ),
            "predictivity": (
                "SCT: predicts L ~ 1/Λ (if nonlinear analysis succeeds). "
                "Hayward: L is a free parameter."
            ),
        },
    }


# ===========================================================================
# FIGURES
# ===========================================================================

def generate_comparison_figures(*, Lambda=1.0, xi=0.0, dps=30):
    """Generate comparison figures."""
    init_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    m2 = float(spin2_local_mass(mp.mpf(Lambda)))
    m0_mp = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))
    m0 = float(m0_mp) if m0_mp is not None else None

    import scipy.special

    # ---- Figure: V(r)/V_N comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    r_vals = np.logspace(-2, 1.5, 200)

    # Left: Potential ratios
    ax = axes[0]

    # GR
    ax.semilogx(r_vals, np.ones_like(r_vals), 'k--', linewidth=0.5,
                  label='GR (Schwarzschild)', alpha=0.5)

    # SCT Yukawa
    h_sct = np.array([1 - 4/3 * np.exp(-m2 * r) + (1/3 * np.exp(-m0 * r) if m0 else 0)
                       for r in r_vals])
    ax.semilogx(r_vals, h_sct, color=SCT_COLORS['prediction'],
                  linewidth=2.0, label='SCT')

    # IDG
    h_idg = np.array([scipy.special.erf(r * Lambda / 2) for r in r_vals])
    ax.semilogx(r_vals, h_idg, color=SCT_COLORS['reference'],
                  linewidth=1.5, linestyle='-.', label='IDG')

    # Hayward-like (approximate with r³/(r³+l³))
    l_hayward = 1.0 / m2  # use r_NL as regularization scale
    h_hayward = np.array([r ** 3 / (r ** 3 + l_hayward ** 3) for r in r_vals])
    ax.semilogx(r_vals, h_hayward, color='green',
                  linewidth=1.0, linestyle=':', label='Hayward-like')

    ax.axhline(y=0, color='gray', linewidth=0.3)
    ax.set_xlabel(r'$r \cdot \Lambda$')
    ax.set_ylabel(r'$V(r)/V_N(r)$')
    ax.set_title('Modified Potential')
    ax.legend(fontsize=7)
    ax.set_ylim(-0.5, 1.3)

    # Right: Mass function scaling
    ax2 = axes[1]

    # m(r)/M for each framework
    ax2.loglog(r_vals, np.abs(h_sct), color=SCT_COLORS['prediction'],
                linewidth=2.0, label=r'SCT: $m \sim r$')
    ax2.loglog(r_vals, h_idg, color=SCT_COLORS['reference'],
                linewidth=1.5, linestyle='-.', label=r'IDG: $m \sim r^3$')
    ax2.loglog(r_vals, h_hayward, color='green',
                linewidth=1.0, linestyle=':', label=r'Hayward: $m \sim r^3$')

    # Reference lines
    a1 = 4/3 * m2 - (1/3 * m0 if m0 else 0)
    ax2.loglog(r_vals, a1 * r_vals, 'k--', linewidth=0.3, alpha=0.4)
    ax2.annotate(r'$\sim r$', xy=(0.05, 0.15), fontsize=7, color='gray')
    ax2.loglog(r_vals, 0.5 * r_vals ** 3, 'k:', linewidth=0.3, alpha=0.4)
    ax2.annotate(r'$\sim r^3$', xy=(0.3, 0.01), fontsize=7, color='gray')

    ax2.set_xlabel(r'$r \cdot \Lambda$')
    ax2.set_ylabel(r'$m(r)/M$')
    ax2.set_title('Mass Function')
    ax2.legend(fontsize=7)
    ax2.set_ylim(1e-6, 2)

    fig.tight_layout()
    save_figure(fig, "mr9_framework_comparison", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)

    print(f"Comparison figure saved to {FIGURES_DIR}")
    return fig


# ===========================================================================
# FULL ANALYSIS
# ===========================================================================

def run_full_comparison(*, Lambda=1.0, xi=0.0, dps=30, verbose=True):
    """Run the complete comparison analysis."""
    if verbose:
        print("=" * 72)
        print("MR-9: Framework Comparison for BH Singularity Resolution")
        print("=" * 72)

    frameworks = build_frameworks()
    numerical = numerical_comparison(Lambda=Lambda, xi=xi, dps=dps)
    discriminating = discriminating_properties()

    if verbose:
        for name, fw in frameworks.items():
            print(f"\n--- {name} ---")
            print(f"  Singularity: {fw.singularity_status}")
            print(f"  K scaling: {fw.K_scaling}")
            print(f"  Core: {fw.core_type}")
            print(f"  Geodesic: {fw.geodesic_complete}")
            print(f"  Ghosts: {fw.ghost_structure}")

    report = {
        "frameworks": {k: v.__dict__ for k, v in frameworks.items()},
        "numerical_table": numerical,
        "discriminating_properties": discriminating,
        "summary": (
            "SCT occupies a unique position: derived from the spectral action "
            "(0 free parameters beyond Λ), with an order-1 entire function "
            "that is ABOVE the Nicolini threshold (≥1/2) for singularity "
            "resolution, but BELOW the exponential (order 1 = borderline). "
            "The linearized analysis gives Stelle-like behavior (K~1/r⁴), "
            "but the Koshelev-Tokareva nonperturbative theorem predicts "
            "that the singularity is forbidden. Resolving this requires "
            "the full nonlinear solution (blocked by OP-01, Gap G1)."
        ),
    }

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MR-9: Framework comparison for BH singularity.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = run_full_comparison(Lambda=1.0, xi=args.xi)

    output_path = args.output or RESULTS_DIR / "mr9_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to: {output_path}")

    if args.figures:
        generate_comparison_figures(Lambda=1.0, xi=args.xi)

    return report


if __name__ == "__main__":
    main()
