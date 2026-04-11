"""
PPN-1 correction: exact nonlocal fakeon mass vs local (Stelle) approximation.

Quantifies how the Eot-Wash bound on Lambda changes when using
the exact Pi_TT zero z0 = 2.4148 instead of the local approximation
z_Stelle = 60/13 = 4.6154 for the spin-2 effective mass.

Key finding: the exact pole has
  - smaller mass: m2 = 1.554*Lambda (vs 2.148*Lambda)
  - NEGATIVE residue: R_pole = -1.191 (vs +60/13 = +4.615)
  - Weaker Yukawa: |alpha| = 0.657 (vs 4/3 = 1.333)
  - |alpha| < 1 => Eot-Wash threshold not reached!
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mpmath as mp

mp.mp.dps = 80

from scripts.ppn1_parameters import (
    _Pi_TT,
    _Pi_scalar,
    _find_tt_zero,
    _Pi_TT_prime_at_z0,
)

# Physical constants
HBAR_C_EV_M = mp.mpf("1.97326980459e-7")
M_TO_EV_INV = 1 / HBAR_C_EV_M
AU_M = mp.mpf("1.495978707e11")
AU_EV_INV = AU_M * M_TO_EV_INV
c2 = mp.mpf(13) / 60


def main():
    print("=" * 70)
    print("PPN-1 CORRECTION: Local vs Exact Fakeon Mass")
    print("=" * 70)

    # ---- LOCAL (Stelle) ----
    z0_local = 1 / c2  # 60/13
    m2_over_L_local = mp.sqrt(z0_local)
    print(f"\nLOCAL APPROXIMATION:")
    print(f"  z_pole = 60/13 = {float(z0_local):.10f}")
    print(f"  m2/Lambda = sqrt(60/13) = {mp.nstr(m2_over_L_local, 15)}")

    # ---- EXACT ----
    z0_exact = _find_tt_zero(dps=80)
    m2_over_L_exact = mp.sqrt(z0_exact)
    print(f"\nEXACT Pi_TT ZERO:")
    print(f"  z0 = {mp.nstr(z0_exact, 25)}")
    print(f"  m2/Lambda = sqrt(z0) = {mp.nstr(m2_over_L_exact, 15)}")
    print(f"  Ratio exact/local = {mp.nstr(m2_over_L_exact / m2_over_L_local, 10)}")

    # ---- RESIDUE ----
    Pi_prime = _Pi_TT_prime_at_z0(z0_exact, dps=80)
    R_pole_exact = 1 / Pi_prime
    R_pole_local = 1 / c2
    print(f"\nRESIDUE:")
    print(f"  Pi_TT'(z0) = {mp.nstr(Pi_prime, 20)}")
    print(f"  R_pole (exact)  = {mp.nstr(R_pole_exact, 15)}")
    print(f"  R_pole (local)  = {mp.nstr(R_pole_local, 15)}")

    # ---- YUKAWA AMPLITUDE ----
    # V/V_N = 1 - alpha_2 * exp(-m2*r) + alpha_0 * exp(-m0*r)
    # where alpha_2 = -(4/3)*R_pole/z0 from the Euclidean prescription
    alpha_2_local = mp.mpf(4) / 3  # 1.3333
    alpha_2_exact = -(mp.mpf(4) / 3) * R_pole_exact / z0_exact
    print(f"\nYUKAWA COEFFICIENT (spin-2):")
    print(f"  alpha_2 (local) = {mp.nstr(alpha_2_local, 15)}")
    print(f"  alpha_2 (exact) = {mp.nstr(alpha_2_exact, 15)}")
    print(f"  |alpha_exact|/|alpha_local| = {mp.nstr(abs(alpha_2_exact) / abs(alpha_2_local), 10)}")

    # ---- SCALAR CHECK ----
    Pi_s_100 = mp.re(_Pi_scalar(100, xi=0.0, dps=30))
    m0_over_L = mp.sqrt(mp.mpf(6))
    print(f"\nSCALAR SECTOR (xi=0):")
    print(f"  Pi_s(z=100) = {float(Pi_s_100):.4f} (monotonically increasing, no pole)")
    print(f"  m0/Lambda = sqrt(6) = {mp.nstr(m0_over_L, 15)} (unchanged)")

    # ---- EOT-WASH BOUND ----
    lambda_max_m = mp.mpf("38.6e-6")
    lambda_max_eV_inv = lambda_max_m * M_TO_EV_INV
    m2_min = 1 / lambda_max_eV_inv

    Lambda_min_local = m2_min / m2_over_L_local
    Lambda_min_exact = m2_min / m2_over_L_exact

    Lambda_local_meV = float(Lambda_min_local) * 1e3
    Lambda_exact_meV = float(Lambda_min_exact) * 1e3
    inv_L_local_um = float(1 / (Lambda_min_local * M_TO_EV_INV)) * 1e6
    inv_L_exact_um = float(1 / (Lambda_min_exact * M_TO_EV_INV)) * 1e6
    change_pct = (Lambda_exact_meV - Lambda_local_meV) / Lambda_local_meV * 100

    print(f"\nEOT-WASH BOUND:")
    print(f"  lambda_max = 38.6 um")
    print(f"  m2_min = {float(m2_min):.6e} eV")
    print(f"  LOCAL:  Lambda > {Lambda_local_meV:.4f} meV  (1/Lambda = {inv_L_local_um:.2f} um)")
    print(f"  EXACT:  Lambda > {Lambda_exact_meV:.4f} meV  (1/Lambda = {inv_L_exact_um:.2f} um)")
    print(f"  CHANGE: +{change_pct:.2f}%")

    print(f"\n  AMPLITUDE EFFECT:")
    print(f"  |alpha_2| (local) = {float(abs(alpha_2_local)):.6f}")
    print(f"  |alpha_2| (exact) = {float(abs(alpha_2_exact)):.6f}")
    if abs(alpha_2_exact) < 1:
        print(f"  *** |alpha_exact| < 1: Eot-Wash threshold NOT reached! ***")
        print(f"  The exact SCT Yukawa may pass Eot-Wash entirely.")

    # ---- CASSINI CHECK ----
    m2_test = m2_over_L_local * mp.mpf("2.38e-3")
    m2r_AU = m2_test * AU_EV_INV
    print(f"\nCASSINI SANITY:")
    print(f"  m2*r_AU = {float(m2r_AU):.4e}")
    print(f"  exp(-m2*r_AU) ~ e^(-{float(m2r_AU):.1e}) => effectively zero")

    # ---- SUMMARY ----
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"1. Local mass: m2 = {mp.nstr(m2_over_L_local, 8)}*Lambda")
    print(f"2. Exact mass: m2 = {mp.nstr(m2_over_L_exact, 8)}*Lambda")
    print(f"3. Mass ratio: {mp.nstr(m2_over_L_exact / m2_over_L_local, 6)}")
    print(f"4. |alpha| ratio: {float(abs(alpha_2_exact) / abs(alpha_2_local)):.4f}")
    print(f"5. Range-only bound: {Lambda_exact_meV:.3f} meV (+{change_pct:.1f}%)")
    print(f"6. Exact |alpha_2| = {float(abs(alpha_2_exact)):.3f} < 1")
    print(f"   => Eot-Wash |alpha|<1 threshold not reached")
    print(f"   => Strongest lab constraint may be REMOVED")
    print(f"7. Fakeon prescription for positive-z pole: Level 1 DEFERRED")

    # ---- SAVE JSON ----
    results = {
        "task": "PPN-1 correction with exact fakeon mass",
        "local_approximation": {
            "z_pole": float(z0_local),
            "m2_over_Lambda": float(m2_over_L_local),
            "alpha_2_Yukawa": float(alpha_2_local),
            "Lambda_min_meV": Lambda_local_meV,
            "inv_Lambda_um": inv_L_local_um,
        },
        "exact_nonlocal": {
            "z0_PiTT_zero": float(z0_exact),
            "Pi_TT_prime_z0": float(Pi_prime),
            "R_pole": float(R_pole_exact),
            "m2_over_Lambda": float(m2_over_L_exact),
            "alpha_2_Yukawa": float(alpha_2_exact),
            "abs_alpha_2": float(abs(alpha_2_exact)),
            "Lambda_min_meV_range_only": Lambda_exact_meV,
            "inv_Lambda_um": inv_L_exact_um,
        },
        "comparison": {
            "mass_ratio_exact_over_local": float(m2_over_L_exact / m2_over_L_local),
            "alpha_ratio": float(abs(alpha_2_exact) / abs(alpha_2_local)),
            "Lambda_change_percent": change_pct,
            "exact_alpha_below_1": bool(abs(alpha_2_exact) < 1),
        },
        "eot_wash": {
            "lambda_max_um": 38.6,
            "m2_min_eV": float(m2_min),
            "local_bound_meV": Lambda_local_meV,
            "exact_bound_range_only_meV": Lambda_exact_meV,
            "note": (
                "Exact |alpha_2| = 0.657 < 1, so Eot-Wash |alpha|<1 threshold "
                "not reached. Exact SCT may pass Eot-Wash entirely."
            ),
        },
        "scalar_sector": {
            "m0_over_Lambda_xi0": float(m0_over_L),
            "scalar_has_pole": False,
            "scalar_coefficient": "1/3 (unchanged)",
        },
        "cassini": {
            "note": "exp(-m2*r_AU) ~ e^{-10^14}, not constraining",
        },
        "verdict": {
            "status": "SIGNIFICANT CORRECTION",
            "summary": (
                "The exact (nonlocal) Pi_TT zero at z0=2.415 gives a smaller "
                "effective mass m2=1.554*Lambda (vs 2.148*Lambda local) and "
                "crucially a WEAKER Yukawa amplitude |alpha|=0.66 (vs 4/3=1.33). "
                "The range-only bound increases by 38% to 3.29 meV. However, "
                "|alpha|<1 means the Eot-Wash threshold is not reached, "
                "potentially removing the strongest laboratory constraint."
            ),
            "caveats": [
                "Euclidean prescription for positive-z pole may differ from fakeon",
                "Level 1 (exact nonlocal) potential is not a simple Yukawa",
                "Full resolution requires Level 1 numerical implementation (DEFERRED)",
            ],
        },
    }

    out_dir = Path(__file__).resolve().parent.parent.parent / "analysis" / "results" / "gap_g1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppn_correction_exact_m2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
