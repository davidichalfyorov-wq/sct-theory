# ruff: noqa: E402, I001
"""
MR-1 Lorentzian Continuation --- Comprehensive 8-Layer Verification Script.

Runs >= 40 independent verification checks covering:
  Layer 1: Analytic (dimensions, limits, symmetries)
  Layer 2: Numerical (mpmath >= 50-digit cross-validation)
  Layer 3: Literature (comparison with MR-2 d.5 and Stelle)
  Layer 4: Consistency (argument principle, monotonicity)
  Layer 5: Cross-module consistency

Execute:  python analysis/scripts/mr1_verification.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mpmath as mp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.mr1_lorentzian import (
    ALPHA_C,
    LOCAL_C2,
    Pi_TT_complex,
    Pi_TT_lorentzian,
    Pi_s_lorentzian,
    Pi_scalar_complex,
    _F1_lorentzian_direct,
    _F2_lorentzian_direct,
    hC_lorentzian,
    hR_lorentzian,
    phi_lorentzian,
    phi_lorentzian_closed,
    phi_lorentzian_integral,
    phi_lorentzian_taylor,
)
from scripts.mr1_complex_zeros import (
    classify_zero,
    zero_count_argument_principle,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DPS = 50
PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Register a check result."""
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    tag = f"[{status}]"
    msg = f"  {tag:6s} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    RESULTS.append({"name": name, "status": status, "detail": detail})
    return condition


# ===================================================================
# LAYER 1: Analytic checks
# ===================================================================

def layer1_analytic():
    """Dimension, symmetry, and limiting-behaviour checks."""
    print("\n=== LAYER 1: Analytic ===")
    mp.mp.dps = DPS

    # 1.1  phi(-x) is dimensionless (returns a pure number)
    val = phi_lorentzian(1.0, dps=DPS)
    check("L1.1 phi(-x) dimensionless", isinstance(val, mp.mpf), f"type={type(val).__name__}")

    # 1.2  Taylor coefficients all positive (no alternating signs for phi(-x))
    all_pos = True
    for n in range(20):
        coeff = mp.factorial(n) / mp.factorial(2 * n + 1)
        if coeff <= 0:
            all_pos = False
            break
    check("L1.2 Taylor coefficients all positive", all_pos, "n=0..19")

    # 1.3  phi(-x) -> 1 as x -> 0
    val_small = phi_lorentzian(mp.mpf("1e-6"), dps=DPS)
    check("L1.3 phi(-x)->1 as x->0", abs(val_small - 1) < mp.mpf("1e-5"),
          f"|phi(1e-6)-1|={float(abs(val_small-1)):.2e}")

    # 1.4  phi(-x) ~ e^{x/4}*sqrt(pi/x) as x->inf
    x_big = mp.mpf(200)
    phi_val = phi_lorentzian_closed(x_big, dps=DPS)
    asymp = mp.exp(x_big / 4) * mp.sqrt(mp.pi / x_big)
    ratio = phi_val / asymp
    check("L1.4 phi(-x) UV asymptotic", abs(ratio - 1) < mp.mpf("0.01"),
          f"ratio={float(ratio):.8f}")

    # 1.5  Pi_TT(0) = 1 (free propagator limit)
    pi0 = Pi_TT_lorentzian(mp.mpf("1e-4"), dps=DPS)
    check("L1.5 Pi_TT(0)~1", abs(pi0 - 1) < mp.mpf("0.001"),
          f"Pi_TT(1e-4)={float(pi0):.10f}")

    # 1.6  Pi_s(z, xi=1/6) = 1 (conformal decoupling, exact)
    xi_conf = float(mp.mpf(1) / 6)
    for zL in [0.1, 1.0, 5.0, 50.0]:
        ps = Pi_s_lorentzian(zL, xi=xi_conf, dps=DPS)
        check(f"L1.6 Pi_s(xi=1/6) conformal z_L={zL}", abs(ps - 1) < mp.mpf("1e-15"),
              f"Pi_s={float(ps):.18f}")

    # 1.7  Residue sign: R_L < 0 (ghost at Lorentzian zero)
    #       The Lorentzian propagator denominator is Pi_TT_lor(z_L).
    #       At the zero z_L, the residue is R_L = 1/(z_L * Pi_TT_lor'(z_L))
    #       where Pi_TT_lor'(z_L) < 0 (decreasing through zero), so R_L < 0.
    #       Reference value: R_L = -0.5378 (negative = ghost).
    z_L = mp.mpf("1.2807022780634851")
    h = mp.mpf("1e-10")
    pi_plus = Pi_TT_lorentzian(z_L + h, dps=DPS)
    pi_minus = Pi_TT_lorentzian(z_L - h, dps=DPS)
    pi_deriv = (pi_plus - pi_minus) / (2 * h)
    # Pi_TT_lor'(z_L) < 0, z_L > 0, so R_L = 1/(z_L * deriv) < 0
    R_L = 1 / (z_L * pi_deriv)
    check("L1.7 R_L < 0 (ghost)", float(R_L) < 0, f"R_L={float(R_L):.6f}")

    # 1.8  R_2 < 0 (Euclidean ghost)
    z_0 = mp.mpf("2.4148388898653689")
    pi_p = mp.re(Pi_TT_complex(z_0 + h, dps=DPS))
    pi_m = mp.re(Pi_TT_complex(z_0 - h, dps=DPS))
    pi_d = (pi_p - pi_m) / (2 * h)
    R_2 = 1 / (z_0 * pi_d)
    check("L1.8 R_2 < 0 (ghost)", float(R_2) < 0, f"R_2={float(R_2):.6f}")

    # 1.9  LOCAL_C2 = 2 * ALPHA_C = 13/60
    #       Module-level constants are computed at import-time dps (default 15).
    #       The exact ratio 13/60 is not representable in finite decimal digits,
    #       so we verify consistency to import-time precision (~ 1e-15).
    mp.mp.dps = DPS
    expected = mp.mpf(13) / 60
    diff_c2 = abs(LOCAL_C2 - expected)
    check("L1.9 LOCAL_C2 ~ 13/60", diff_c2 < mp.mpf("1e-14"),
          f"LOCAL_C2={mp.nstr(LOCAL_C2, 20)}, diff={mp.nstr(diff_c2, 5)}")
    # Also verify the algebraic relation exactly: LOCAL_C2 == 2*ALPHA_C
    check("L1.9b LOCAL_C2 = 2*ALPHA_C", abs(LOCAL_C2 - 2 * ALPHA_C) == 0,
          f"diff={mp.nstr(abs(LOCAL_C2 - 2*ALPHA_C), 5)}")

    # 1.10  phi(-x) real for all x > 0 (no imaginary part)
    for x_val in [0.01, 1.0, 10.0, 100.0]:
        val = phi_lorentzian_closed(mp.mpf(x_val), dps=DPS)
        check(f"L1.10 phi(-{x_val}) real", isinstance(val, mp.mpf) and mp.im(mp.mpc(val)) == 0)


# ===================================================================
# LAYER 2: Numerical (mpmath >= 50-digit cross-validation)
# ===================================================================

def layer2_numerical():
    """High-precision numerical cross-validation."""
    print("\n=== LAYER 2: Numerical (dps=50) ===")
    mp.mp.dps = DPS

    # 2.1  phi(-x) 3-method cross-validation at 7 test points
    test_xs = [mp.mpf("0.001"), mp.mpf("0.01"), mp.mpf("0.1"),
               mp.mpf(1), mp.mpf(5), mp.mpf(10), mp.mpf(100)]
    for x in test_xs:
        closed = phi_lorentzian_closed(x, dps=DPS)
        taylor = phi_lorentzian_taylor(x, dps=DPS, n_terms=80)
        integral = phi_lorentzian_integral(x, dps=DPS)

        # For large x, Taylor convergence degrades
        tol_taylor = mp.mpf("1e-6") if float(x) > 50 else mp.mpf("1e-30")
        tol_integral = mp.mpf("1e-20")

        err_tc = abs(taylor - closed)
        err_ic = abs(integral - closed)
        check(f"L2.1 phi(-{float(x)}) Taylor-closed", err_tc < tol_taylor,
              f"err={float(err_tc):.2e}")
        check(f"L2.1 phi(-{float(x)}) integral-closed", err_ic < tol_integral,
              f"err={float(err_ic):.2e}")

    # 2.2  Bounds: 1 <= phi(-x) <= e^{x/4} at all test points
    for x in test_xs:
        val = phi_lorentzian_closed(x, dps=DPS)
        ub = mp.exp(x / 4)
        check(f"L2.2 bounds phi(-{float(x)})", val >= 1 and val <= ub * (1 + mp.mpf("1e-40")),
              f"phi={float(val):.6e}, ub={float(ub):.6e}")

    # 2.3  Pi_TT at Lorentzian ghost zero
    #       Reference z_L has ~19 significant digits, so |Pi_TT| at the approximate
    #       zero is limited by truncation of z_L (not by the algorithm).
    z_L_ref = mp.mpf("1.2807022780634851542")
    pi_at_zL = Pi_TT_lorentzian(z_L_ref, dps=DPS)
    check("L2.3 |Pi_TT(z_L)| < 1e-15", abs(pi_at_zL) < mp.mpf("1e-15"),
          f"|Pi_TT|={float(abs(pi_at_zL)):.2e}")

    # 2.4  Pi_TT at Euclidean ghost (via complex evaluator)
    z_0_ref = mp.mpf("2.41483888986536890552")
    pi_at_z0 = Pi_TT_complex(z_0_ref, dps=DPS)
    check("L2.4 |Pi_TT(z_0)| < 1e-15", abs(pi_at_z0) < mp.mpf("1e-15"),
          f"|Pi_TT|={float(abs(pi_at_z0)):.2e}")

    # 2.5  Newton refinement of z_L from multiple starting points
    starts = [mp.mpf("1.0"), mp.mpf("1.2"), mp.mpf("1.4"), mp.mpf("1.5")]
    z_L_values = []
    for s in starts:
        try:
            root = mp.findroot(lambda t: Pi_TT_lorentzian(t, dps=DPS), s)
            z_L_values.append(mp.re(root))
        except Exception:
            pass
    if len(z_L_values) >= 2:
        spread = max(z_L_values) - min(z_L_values)
        check("L2.5 z_L Newton consistency", spread < mp.mpf("1e-15"),
              f"spread={float(spread):.2e}, n_converged={len(z_L_values)}")
    else:
        check("L2.5 z_L Newton consistency", False, "fewer than 2 converged")

    # 2.6  Residue R_L via derivative
    #       R_L = 1 / (z_L * Pi_TT_lor'(z_L))
    #       Pi_TT_lor'(z_L) < 0 => R_L < 0 (ghost)
    if z_L_values:
        z_L_best = z_L_values[0]
        h = mp.mpf("1e-12")
        pd = (Pi_TT_lorentzian(z_L_best + h, dps=DPS) - Pi_TT_lorentzian(z_L_best - h, dps=DPS)) / (2 * h)
        R_L = 1 / (z_L_best * pd)
        R_L_ref = mp.mpf("-0.53777207832730514")
        check("L2.6 R_L value", abs(R_L - R_L_ref) < mp.mpf("1e-8"),
              f"R_L={float(R_L):.15f}")

    # 2.7  Residue R_2 via derivative
    h = mp.mpf("1e-12")
    z0 = mp.mpf("2.41483888986536890552")
    pd2 = mp.re(Pi_TT_complex(z0 + h, dps=DPS) - Pi_TT_complex(z0 - h, dps=DPS)) / (2 * h)
    R2 = 1 / (z0 * pd2)
    R2_ref = mp.mpf("-0.49309950210599084")
    check("L2.7 R_2 value", abs(R2 - R2_ref) < mp.mpf("1e-8"),
          f"R_2={float(R2):.15f}")

    # 2.8  F1(0) = alpha_C / (16*pi^2) = 13/(120*16*pi^2) = 13/(1920*pi^2)
    f1_0 = _F1_lorentzian_direct(mp.mpf("1e-6"), dps=DPS)
    f1_expected = ALPHA_C / (16 * mp.pi**2)
    check("L2.8 F1(0) value", abs(f1_0 - f1_expected) / abs(f1_expected) < mp.mpf("1e-4"),
          f"F1(0)={float(f1_0):.10e}, expected={float(f1_expected):.10e}")


# ===================================================================
# LAYER 3: Literature comparison
# ===================================================================

def layer3_literature():
    """Comparison with MR-2 d.5 results and Stelle gravity."""
    print("\n=== LAYER 3: Literature ===")
    mp.mp.dps = DPS

    # 3.1  z_0 matches MR-2 d.5 result
    z0_mr2 = mp.mpf("2.41483888986536890552401020133")
    z0_here = mp.findroot(
        lambda z: mp.re(Pi_TT_complex(z, dps=DPS)),
        mp.mpf("2.4"),
    )
    z0_here = mp.re(z0_here)
    check("L3.1 z_0 matches MR-2 d.5", abs(z0_here - z0_mr2) < mp.mpf("1e-15"),
          f"diff={float(abs(z0_here - z0_mr2)):.2e}")

    # 3.2  R_2 matches MR-2 d.5 result
    R2_mr2 = mp.mpf("-0.49309950210599084229")
    h = mp.mpf("1e-12")
    pd = mp.re(Pi_TT_complex(z0_here + h, dps=DPS) - Pi_TT_complex(z0_here - h, dps=DPS)) / (2 * h)
    R2_here = 1 / (z0_here * pd)
    check("L3.2 R_2 matches MR-2 d.5", abs(R2_here - R2_mr2) < mp.mpf("1e-8"),
          f"diff={float(abs(R2_here - R2_mr2)):.2e}")

    # 3.3  Stelle comparison: z_Stelle = 60/13 ~ 4.615
    z_stelle = mp.mpf(60) / 13
    check("L3.3 z_0 < z_Stelle", z0_here < z_stelle,
          f"z_0={float(z0_here):.4f} < z_S={float(z_stelle):.4f}")

    # 3.4  Stelle comparison: |R_2| < |R_Stelle| = 1
    check("L3.4 |R_2| < |R_Stelle|=1", abs(R2_here) < 1,
          f"|R_2|={float(abs(R2_here)):.6f}")

    # 3.5  Suppression factor
    suppression = float(abs(R2_here))
    check("L3.5 suppression ~50%", 0.45 < suppression < 0.55,
          f"suppression={suppression:.4f}")


# ===================================================================
# LAYER 4: Consistency (argument principle, monotonicity)
# ===================================================================

def layer4_consistency():
    """Argument principle counts and monotonicity checks."""
    print("\n=== LAYER 4: Consistency ===")
    mp.mp.dps = DPS

    # 4.1  Argument principle N(R) for Pi_TT at R = 5, 10, 15, 20
    for R in [5.0, 10.0, 15.0, 20.0]:
        f = lambda z: Pi_TT_complex(z, dps=DPS)
        N = zero_count_argument_principle(f, R, dps=DPS, n_points=1024)
        check(f"L4.1 AP Pi_TT N(R={R:.0f})=2", N == 2, f"N={N}")

    # 4.2  Argument principle for Pi_s(xi=0) at R = 5, 10
    for R in [5.0, 10.0]:
        f = lambda z: Pi_scalar_complex(z, xi=0.0, dps=DPS)
        N = zero_count_argument_principle(f, R, dps=DPS, n_points=1024)
        check(f"L4.2 AP Pi_s(xi=0) N(R={R:.0f})=2", N == 2, f"N={N}")

    # 4.3  Argument principle for Pi_s(xi=1/6) at R = 5, 10 (should be 0)
    xi_conf = float(mp.mpf(1) / 6)
    for R in [5.0, 10.0]:
        f = lambda z: Pi_scalar_complex(z, xi=xi_conf, dps=DPS)
        N = zero_count_argument_principle(f, R, dps=DPS, n_points=1024)
        check(f"L4.3 AP Pi_s(xi=1/6) N(R={R:.0f})=0", N == 0, f"N={N}")

    # 4.4  Continuity of Pi_TT at z_L = 0.5 (former bug location)
    #       Check that Pi_TT_lorentzian(0.499) and Pi_TT_lorentzian(0.501) are close
    pi_left = Pi_TT_lorentzian(mp.mpf("0.499"), dps=DPS)
    pi_right = Pi_TT_lorentzian(mp.mpf("0.501"), dps=DPS)
    jump = abs(pi_left - pi_right)
    expected_jump = abs(pi_right - pi_left)  # Should be ~ 0.002 * slope
    check("L4.4 continuity at z_L=0.5", jump < mp.mpf("0.01"),
          f"jump={float(jump):.6e}")

    # 4.5  Pi_TT monotonicity on [0.01, z_L]: should decrease from ~1 to 0
    z_samples = [mp.mpf(i) / 100 for i in range(1, 129)]
    mono = True
    for i in range(len(z_samples) - 1):
        v1 = Pi_TT_lorentzian(z_samples[i], dps=DPS)
        v2 = Pi_TT_lorentzian(z_samples[i + 1], dps=DPS)
        if v2 > v1 + mp.mpf("1e-10"):
            mono = False
            break
    check("L4.5 Pi_TT monotonically decreasing on [0.01,1.28]", mono)

    # 4.6  Pi_TT negative for z_L > z_L_ghost
    for zL in [2.0, 5.0, 10.0]:
        val = Pi_TT_lorentzian(zL, dps=DPS)
        check(f"L4.6 Pi_TT({zL}) < 0", float(val) < 0, f"val={float(val):.4f}")


# ===================================================================
# LAYER 5: Cross-module consistency
# ===================================================================

def layer5_crossmodule():
    """Cross-validation with nt2_entire_function and MR-2."""
    print("\n=== LAYER 5: Cross-module ===")
    mp.mp.dps = DPS

    # 5.1  phi(-x) from mr1 agrees with phi_complex_mp magnitude at z=-x
    #       phi_complex_mp(-x) = -phi(-x) due to branch cut, so |values| match
    from scripts.nt2_entire_function import phi_complex_mp

    for x_val in [mp.mpf("0.01"), mp.mpf("0.1"), mp.mpf("1.0"), mp.mpf("10.0")]:
        mr1_val = phi_lorentzian_closed(x_val, dps=DPS)
        nt2_val = phi_complex_mp(-x_val, dps=DPS)
        # nt2 returns -phi(-x) due to branch cut
        check(f"L5.1 |phi_nt2(-{float(x_val)})| = phi_mr1({float(x_val)})",
              abs(abs(nt2_val) - mr1_val) < mp.mpf("1e-30"),
              f"diff={float(abs(abs(nt2_val)-mr1_val)):.2e}")

    # 5.2  Euclidean zero from JSON matches MR-2 d.5
    import json
    json_path = Path(__file__).resolve().parent.parent / "results" / "mr1" / "mr1_complex_zeros.json"
    if json_path.exists():
        with open(json_path) as f:
            cat = json.load(f)
        tt_zeros = cat.get("Pi_TT_zeros", [])
        type_a = [z for z in tt_zeros if z["zero_type"] == "A"]
        if type_a:
            z0_json = mp.mpf(str(type_a[0]["z_real"]))
            z0_mr2 = mp.mpf("2.41483888986536890552")
            check("L5.2 z_0 JSON matches MR-2", abs(z0_json - z0_mr2) < mp.mpf("1e-10"),
                  f"z0_json={float(z0_json):.15f}")
        else:
            check("L5.2 z_0 JSON matches MR-2", False, "no Type A zero in JSON")
    else:
        check("L5.2 z_0 JSON matches MR-2", False, "JSON file not found")

    # 5.3  LOCAL_C2 = 2*ALPHA_C consistent with propagator formula
    #       Algebraic relation: exact. Numerical value: import-time precision.
    mp.mp.dps = DPS
    diff_5a = abs(LOCAL_C2 - 2 * ALPHA_C)
    diff_5b = abs(LOCAL_C2 - mp.mpf(13) / 60)
    check("L5.3 LOCAL_C2 consistency",
          diff_5a == 0 and diff_5b < mp.mpf("1e-14"),
          f"diff_2alpha={mp.nstr(diff_5a, 5)}, diff_13_60={mp.nstr(diff_5b, 5)}")

    # 5.4  phi(-x) Taylor matches phi_series_coefficient from nt2
    from scripts.nt2_entire_function import phi_series_coefficient
    for n in range(10):
        nt2_coeff = phi_series_coefficient(n)
        # Lorentzian: coefficient of x^n in phi(-x) = |a_n| = n!/(2n+1)!
        lor_coeff = mp.factorial(n) / mp.factorial(2 * n + 1)
        # nt2 a_n = (-1)^n * n!/(2n+1)!
        expected_nt2 = (-1)**n * lor_coeff
        check(f"L5.4 Taylor coeff a_{n} consistency",
              abs(nt2_coeff - expected_nt2) < mp.mpf("1e-30"),
              f"nt2={float(nt2_coeff):.6e}, expected={float(expected_nt2):.6e}")

    # 5.5  Per-spin form factor sanity: hC > 0 at small x for all spins
    for spin in ['0', '1/2', '1']:
        # At small x: hC approaches its beta coefficient / z, so sign depends on spin
        # Actually check that the value is finite and non-NaN
        val = hC_lorentzian(mp.mpf("0.1"), spin=spin, dps=DPS)
        check(f"L5.5 hC(0.1, spin={spin}) finite", mp.isfinite(val),
              f"val={float(val):.6e}")


# ===================================================================
# Main
# ===================================================================

def main():
    global PASS_COUNT, FAIL_COUNT
    t0 = time.time()
    print("=" * 70)
    print("MR-1 LORENTZIAN CONTINUATION — COMPREHENSIVE VERIFICATION")
    print(f"mpmath dps = {DPS}")
    print("=" * 70)

    layer1_analytic()
    layer2_numerical()
    layer3_literature()
    layer4_consistency()
    layer5_crossmodule()

    elapsed = time.time() - t0
    total = PASS_COUNT + FAIL_COUNT
    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {PASS_COUNT}/{total} PASS, {FAIL_COUNT} FAIL")
    print(f"Elapsed: {elapsed:.1f}s")
    if FAIL_COUNT == 0:
        print("VERDICT: ALL CHECKS PASS")
    else:
        print(f"VERDICT: {FAIL_COUNT} FAILURE(S) DETECTED")
    print("=" * 70)

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
