"""
Exact effective metric f(r) for the SCT linearized solution.

Computes and compares:
  1. GR Schwarzschild:  f(r) = 1 - 2M/r
  2. SCT local approx:  V(r)/V_N = 1 - (4/3)e^{-m2_L r} + (1/3)e^{-m0_L r}
     with m2_L = sqrt(60/13) Lambda, m0_L = sqrt(6) Lambda
  3. SCT exact (Stelle/Feynman): uses exact masses and residues from Pi_TT zeros
  4. SCT exact (fakeon):         PV integral, ghost Yukawa removed

Key finding: the LOCAL approximation overestimates both masses by >30%.
  Exact: z2 = 2.4149, m2/Lambda = 1.5540
  Local: z2_local = 60/13 = 4.615, m2_local/Lambda = 2.148

The scalar sector Pi_s(z, xi=0) has NO zero => no scalar Yukawa at xi=0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.form_factors import F1_total, F2_total
from sct_tools.plotting import create_figure, init_style, save_figure

# Color palette for this figure
C_BLUE = "#2196F3"
C_RED = "#F44336"
C_GREEN = "#4CAF50"
C_ORANGE = "#FF9800"

PROJECT_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "gap_g1"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"

# ===========================================================================
# Physical constants
# ===========================================================================
ALPHA_C = 13.0 / 120.0
LOCAL_C2 = 2 * ALPHA_C  # = 13/60


def _alpha_C_of_x(x: float) -> float:
    return 16 * np.pi**2 * F1_total(x)


def _alpha_R_of_x(x: float, xi: float = 0.0) -> float:
    return 16 * np.pi**2 * F2_total(x, xi=xi)


def _F1_shape(z: float) -> float:
    if z < 1e-15:
        return 1.0
    return _alpha_C_of_x(z) / ALPHA_C


def Pi_TT(z: float) -> float:
    """Spin-2 propagator denominator, normalized Pi_TT(0) = 1."""
    return 1.0 + LOCAL_C2 * z * _F1_shape(z)


def Pi_s(z: float, xi: float = 0.0) -> float:
    """Scalar propagator denominator, normalized Pi_s(0) = 1."""
    coeff = 6.0 * (xi - 1.0 / 6) ** 2
    if abs(coeff) < 1e-40:
        return 1.0
    aR0 = 2.0 * (xi - 1.0 / 6) ** 2
    return 1.0 + coeff * z * _alpha_R_of_x(z, xi=xi) / aR0


# ===========================================================================
# Exact TT zero and derivative (high-precision via mpmath)
# ===========================================================================
def compute_exact_tt_zero(dps: int = 50) -> dict:
    """Find the exact first positive zero of Pi_TT and its derivative."""
    mp.mp.dps = dps

    # Import complex form factors for mpmath precision
    sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))
    from nt2_entire_function import F1_total_complex

    _alpha_c = mp.mpf(13) / 120
    _c2 = 2 * _alpha_c

    def F1_shape_mp(z):
        val = F1_total_complex(z, xi=0.0, dps=dps)
        val0 = F1_total_complex(0, xi=0.0, dps=dps)
        return val / val0

    def Pi_TT_mp(z):
        z_mp = mp.mpf(z)
        return mp.re(1 + _c2 * z_mp * F1_shape_mp(z_mp))

    z2 = mp.findroot(Pi_TT_mp, mp.mpf("2.41"))

    # Derivative via central difference (stable at 50 digits)
    h = mp.mpf("1e-12")
    dPi = (Pi_TT_mp(z2 + h) - Pi_TT_mp(z2 - h)) / (2 * h)

    m2 = mp.sqrt(z2)

    return {
        "z2": float(z2),
        "z2_mp": mp.nstr(z2, 30),
        "m2_over_Lambda": float(m2),
        "m2_over_Lambda_mp": mp.nstr(m2, 30),
        "dPi_TT_at_z2": float(dPi),
        "dPi_TT_at_z2_mp": mp.nstr(dPi, 25),
    }


# ===========================================================================
# Potential via partial fractions (Stelle/Feynman prescription)
# ===========================================================================
def V_ratio_local(r: np.ndarray, Lambda: float = 1.0) -> np.ndarray:
    """Local (Stelle 1977) Yukawa approximation: V(r)/V_N(r).

    Uses local masses from the z -> 0 expansion of Pi_TT, Pi_s.
    """
    m2 = np.sqrt(60.0 / 13) * Lambda
    m0 = np.sqrt(6.0) * Lambda
    return 1.0 - (4.0 / 3) * np.exp(-m2 * r) + (1.0 / 3) * np.exp(-m0 * r)


def V_ratio_exact_stelle(
    r: np.ndarray, z2: float, dPi_z2: float, Lambda: float = 1.0
) -> np.ndarray:
    """Exact Yukawa (Stelle/Feynman) with corrected mass and residue.

    Uses the exact zero z2 of Pi_TT and the exact residue.
    The scalar sector Pi_s(z, xi=0) has NO zero => only spin-2 Yukawa.
    """
    m2 = np.sqrt(z2) * Lambda
    # Residue for V(r)/V_N: spin-2 contribution is -|R2| (ghost = repulsive)
    # R2 = 4/(3 * z2 * Pi'(z2)) where Pi' < 0 => R2 < 0
    R2 = 4.0 / (3.0 * z2 * dPi_z2)
    # No scalar Yukawa at xi=0 (Pi_s has no zero)
    return 1.0 + R2 * np.exp(-m2 * r)


def V_ratio_exact_PV(
    r_arr: np.ndarray, Lambda: float = 1.0, xi: float = 0.0,
    kmax: float = 150.0, eps: float = 1e-6,
) -> np.ndarray:
    """Exact V(r)/V_N via principal-value Fourier integral (fakeon).

    V/V_N = (2/pi) * P.V. int_0^inf dk sin(kr)/k * [4/(3 Pi_TT) - 1/(3 Pi_s)]
    """
    z2 = brentq(Pi_TT, 2.0, 3.0, xtol=1e-14)
    k_pole = np.sqrt(z2) * Lambda

    def integrand(k, r_val):
        if k < 1e-15:
            return 0.0
        z = (k / Lambda) ** 2
        K = 4.0 / (3.0 * Pi_TT(z)) - 1.0 / (3.0 * Pi_s(z, xi=xi))
        return np.sin(k * r_val) / k * K

    results = np.empty_like(r_arr)
    for i, r_val in enumerate(r_arr):
        I1, _ = quad(
            integrand, 0, k_pole - eps, args=(r_val,),
            limit=1000, epsabs=1e-12, epsrel=1e-10,
        )
        I2, _ = quad(
            integrand, k_pole + eps, kmax * Lambda, args=(r_val,),
            limit=2000, epsabs=1e-12, epsrel=1e-10,
        )
        results[i] = 2.0 / np.pi * (I1 + I2)
    return results


# ===========================================================================
# Effective metric f(r) = 1 - 2M/r * V(r)/V_N(r)
# ===========================================================================
def f_GR(r: np.ndarray, M: float = 1.0) -> np.ndarray:
    """Schwarzschild metric function."""
    return 1.0 - 2 * M / r


def f_SCT(r: np.ndarray, V_ratio: np.ndarray, M: float = 1.0) -> np.ndarray:
    """SCT effective metric f(r) = 1 - 2M/r * V(r)/V_N(r)."""
    return 1.0 - 2 * M / r * V_ratio


# ===========================================================================
# Main computation
# ===========================================================================
def main():
    print("=" * 70)
    print("GAP-G1: Exact effective metric f(r) for SCT")
    print("=" * 70)

    # --- Step 1: Exact TT zero ---
    print("\n--- Computing exact TT zero (mpmath 50 digits) ---")
    tt_data = compute_exact_tt_zero(dps=50)
    z2 = tt_data["z2"]
    m2_exact = tt_data["m2_over_Lambda"]
    dPi = tt_data["dPi_TT_at_z2"]
    print(f"  z2_exact      = {tt_data['z2_mp']}")
    print(f"  m2/Lambda     = {tt_data['m2_over_Lambda_mp']}")
    print(f"  Pi'_TT(z2)    = {tt_data['dPi_TT_at_z2_mp']}")

    # Local comparison
    z2_local = 60.0 / 13
    m2_local = np.sqrt(z2_local)
    m0_local = np.sqrt(6.0)
    print(f"\n  z2_local      = {z2_local:.10f}")
    print(f"  m2_local/Lam  = {m2_local:.10f}")
    print(f"  m0_local/Lam  = {m0_local:.10f}")
    print(f"  m2 shift      = {(m2_exact - m2_local)/m2_local*100:+.2f}%")

    # Residues
    R2_exact = 4.0 / (3.0 * z2 * dPi)
    R2_local = -4.0 / 3.0  # Stelle local
    R0_local = 1.0 / 3.0
    print(f"\n  R2_exact (Stelle) = {R2_exact:.10f}")
    print(f"  R2_local (Stelle) = {R2_local:.10f}")
    print(f"  R0_local (Stelle) = {R0_local:.10f}")
    print(f"  Scalar Pi_s(z, xi=0): NO zero => no scalar Yukawa")

    # --- Step 2: Compute potential ratios ---
    print("\n--- Computing V(r)/V_N at tabulated r values ---")
    r_fine = np.linspace(0.2, 20.0, 500)

    V_loc = V_ratio_local(r_fine)
    V_stelle = V_ratio_exact_stelle(r_fine, z2, dPi)

    # For the PV (fakeon): compute at selected points (expensive)
    r_pv_points = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    print("  Computing PV integral (fakeon) at selected r values...")
    V_pv = V_ratio_exact_PV(r_pv_points, kmax=100, eps=1e-6)

    # --- Step 3: Table ---
    print("\n  r/Lambda   V_local    V_exact_Stelle  V_PV(fakeon)  diff(Stelle-local)")
    print("  " + "-" * 75)
    for r_val in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        idx = np.argmin(np.abs(r_fine - r_val))
        vl = V_loc[idx]
        vs = V_stelle[idx]
        pv_idx = np.argmin(np.abs(r_pv_points - r_val))
        vpv = V_pv[pv_idx] if abs(r_pv_points[pv_idx] - r_val) < 0.1 else float("nan")
        print(f"  {r_val:5.1f}      {vl:10.6f}   {vs:12.6f}     {vpv:12.6f}    {vs - vl:+.4e}")

    # --- Step 4: Generate figure ---
    print("\n--- Generating figure ---")
    init_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: V(r)/V_N comparison (the key physics plot) ----
    r_fine2 = np.linspace(0.1, 8.0, 800)
    V_loc2 = V_ratio_local(r_fine2)
    V_stelle2 = V_ratio_exact_stelle(r_fine2, z2, dPi)

    fig1, ax1 = create_figure(figsize=(7, 5))
    ax1.plot(r_fine2, np.ones_like(r_fine2), ":", color="gray", lw=0.8)
    ax1.plot(r_fine2, V_loc2, "--", color=C_BLUE, lw=1.8,
             label=(rf"Local approx: $m_2={m2_local:.3f}\Lambda$, "
                    rf"$m_0={m0_local:.3f}\Lambda$"))
    ax1.plot(r_fine2, V_stelle2, "-", color=C_RED, lw=2.0,
             label=(rf"Exact (Stelle): $m_2={m2_exact:.4f}\Lambda$, "
                    r"no scalar pole"))
    ax1.axhline(0, color="gray", lw=0.5, ls=":")
    ax1.set_xlabel(r"$r \cdot \Lambda$")
    ax1.set_ylabel(r"$V(r) / V_{\rm N}(r)$")
    ax1.set_title("SCT modified Newtonian potential: exact vs local")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_xlim(0, 8)
    ax1.set_ylim(-0.2, 1.5)

    # Inset: zoom into difference
    ax_in = ax1.inset_axes([0.35, 0.15, 0.60, 0.35])
    diff_V = V_stelle2 - V_loc2
    ax_in.plot(r_fine2, diff_V, "-", color=C_RED, lw=1.5)
    ax_in.axhline(0, color="gray", lw=0.5, ls=":")
    ax_in.set_xlabel(r"$r \cdot \Lambda$", fontsize=7)
    ax_in.set_ylabel(r"$\Delta(V/V_{\rm N})$", fontsize=7)
    ax_in.set_title("Exact $-$ Local", fontsize=7)
    ax_in.set_xlim(0, 6)
    ax_in.tick_params(labelsize=6)

    fig1.tight_layout()
    save_figure(fig1, "mr9_effective_potential", fmt="pdf", directory=FIGURES_DIR)
    save_figure(fig1, "mr9_effective_potential", fmt="png", directory=FIGURES_DIR)
    print(f"  Saved to {FIGURES_DIR / 'mr9_effective_potential.pdf'}")

    # ---- Figure 2: f(r) for near-Planckian BH ----
    # Use M = 0.5/Lambda so r_h = 1/Lambda (SCT corrections visible)
    M_vals = [0.5, 1.0, 5.0]
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, M_ov_L in zip(axes2, M_vals):
        r_plot = np.linspace(0.05, 6 * M_ov_L + 2, 800)
        f_gr_vals = f_GR(r_plot, M=M_ov_L)
        f_loc_vals = f_SCT(r_plot, V_ratio_local(r_plot), M=M_ov_L)
        f_ex_vals = f_SCT(r_plot, V_ratio_exact_stelle(r_plot, z2, dPi), M=M_ov_L)

        ax.plot(r_plot, f_gr_vals, "-", color="black", lw=1.2, label="GR")
        ax.plot(r_plot, f_loc_vals, "--", color=C_BLUE, lw=1.5, label="SCT local")
        ax.plot(r_plot, f_ex_vals, "-", color=C_RED, lw=2.0, label="SCT exact")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel(r"$r / \Lambda^{-1}$")
        ax.set_title(rf"$M = {M_ov_L}\,\Lambda^{{-1}}$")
        ax.set_xlim(r_plot[0], r_plot[-1])
        ax.set_ylim(-2.5, 1.2)
        if M_ov_L == M_vals[0]:
            ax.set_ylabel(r"$f(r)$")
            ax.legend(fontsize=7, loc="lower right")

    fig2.suptitle(
        r"Effective metric $f(r) = 1 - \frac{2M}{r}\,\frac{V(r)}{V_{\rm N}(r)}$"
        r" (Stelle prescription)",
        fontsize=11,
    )
    fig2.tight_layout()
    save_figure(fig2, "gap_g1_f_metric_comparison", fmt="pdf", directory=FIGURES_DIR)
    save_figure(fig2, "gap_g1_f_metric_comparison", fmt="png", directory=FIGURES_DIR)
    print(f"  Saved to {FIGURES_DIR / 'gap_g1_f_metric_comparison.pdf'}")

    # --- Step 5: Third figure — V(r)/V_N with PV dots ---
    fig3, ax3 = create_figure(figsize=(6, 4.5))

    ax3.plot(r_fine, V_loc, "--", color=C_BLUE, lw=1.5,
             label=rf"Local ($m_2={m2_local:.3f}\Lambda$, $m_0={m0_local:.3f}\Lambda$)")
    ax3.plot(r_fine, V_ratio_exact_stelle(r_fine, z2, dPi), "-", color=C_RED, lw=2.0,
             label=rf"Exact Stelle ($m_2={m2_exact:.4f}\Lambda$, no scalar pole)")
    ax3.plot(r_pv_points, V_pv, "o", color=C_GREEN, ms=5,
             label="Exact PV (fakeon)")
    ax3.axhline(1, color="gray", lw=0.5, ls=":")
    ax3.set_xlabel(r"$r \cdot \Lambda$")
    ax3.set_ylabel(r"$V(r) / V_{\rm N}(r)$")
    ax3.set_title("SCT modified Newtonian potential")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 15)
    ax3.set_ylim(-0.5, 2.5)
    fig3.tight_layout()

    save_figure(fig3, "gap_g1_potential_ratio", fmt="pdf", directory=FIGURES_DIR)
    save_figure(fig3, "gap_g1_potential_ratio", fmt="png", directory=FIGURES_DIR)
    print(f"  Saved to {FIGURES_DIR / 'gap_g1_potential_ratio.pdf'}")

    # --- Step 6: Save JSON ---
    results = {
        "description": "Exact effective potential for SCT linearized solution",
        "TT_sector": {
            "z2_exact": z2,
            "z2_exact_30digit": tt_data["z2_mp"],
            "m2_over_Lambda_exact": m2_exact,
            "m2_over_Lambda_exact_30digit": tt_data["m2_over_Lambda_mp"],
            "dPi_TT_at_z2": dPi,
            "R2_Stelle_residue": R2_exact,
            "z2_local": z2_local,
            "m2_local_over_Lambda": float(m2_local),
            "m2_shift_percent": (m2_exact - m2_local) / m2_local * 100,
        },
        "scalar_sector_xi0": {
            "has_zero": False,
            "Pi_s_monotonic": True,
            "Pi_s_at_z2": float(Pi_s(z2, xi=0.0)),
            "note": "Pi_s(z, xi=0) > 1 for all z > 0; no scalar Yukawa pole",
        },
        "local_approximation": {
            "m2_over_Lambda": float(m2_local),
            "m0_over_Lambda": float(m0_local),
            "R2": -4.0 / 3,
            "R0": 1.0 / 3,
            "formula": "V/V_N = 1 - (4/3)exp(-m2*r) + (1/3)exp(-m0*r)",
        },
        "exact_stelle": {
            "m2_over_Lambda": m2_exact,
            "R2": R2_exact,
            "no_scalar_yukawa": True,
            "formula": "V/V_N = 1 + R2*exp(-m2*r), R2 = 4/(3*z2*Pi'(z2))",
        },
        "fakeon_note": (
            "With the fakeon (Anselmi) prescription, the spin-2 ghost does not "
            "contribute a Yukawa term. The potential is computed via PV Fourier "
            "integral. At xi=0 the scalar has no pole either, so V/V_N -> 1 "
            "with only power-law (non-Yukawa) corrections from the nonlocal "
            "form factors."
        ),
        "V_ratio_table": {},
    }

    # Table of V/V_N values
    for r_val in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        idx = np.argmin(np.abs(r_fine - r_val))
        entry = {
            "V_local": float(V_loc[idx]),
            "V_exact_stelle": float(V_ratio_exact_stelle(np.array([r_val]), z2, dPi)[0]),
        }
        pv_idx = np.argmin(np.abs(r_pv_points - r_val))
        if abs(r_pv_points[pv_idx] - r_val) < 0.15:
            entry["V_PV_fakeon"] = float(V_pv[pv_idx])
        results["V_ratio_table"][f"r={r_val:.1f}"] = entry

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "effective_potential_exact.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved results to {output_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Exact ghost mass:  m2 = {m2_exact:.6f} Lambda")
    print(f"  Local ghost mass:  m2 = {m2_local:.6f} Lambda")
    print(f"  Mass shift:        {(m2_exact - m2_local)/m2_local*100:+.1f}%")
    print(f"  Exact residue R2:  {R2_exact:.8f}")
    print(f"  Local residue R2:  {R2_local:.8f}")
    print(f"  Residue shift:     {(R2_exact - R2_local)/abs(R2_local)*100:+.1f}%")
    print(f"  Scalar pole (xi=0): NONE (Pi_s monotonically increasing)")
    print(f"  Qualitative change: local formula has BOTH Yukawa terms;")
    print(f"                      exact formula has ONLY spin-2 Yukawa (at xi=0)")


if __name__ == "__main__":
    main()
