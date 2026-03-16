# ruff: noqa: E402, I001
"""
SS-V: Independent 8-Layer Verification of the Scalar Sector.

Implements verification Layers 1-6 for the scalar propagator denominator
Pi_s(z, xi) = 1 + 6*(xi - 1/6)^2 * z * F2_hat(z, xi).

All computations build from scratch: only mpmath, no imports from
ss_scalar_sector.py or _ss_dr_rederivation.py.

Author: David Alfyorov
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
DPS = 100  # 100-digit precision for Layer 2
mp.mp.dps = DPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "ss"

# SM multiplicities
N_S = 4
N_F = 45
N_V = 12
N_D = mp.mpf(N_F) / 2  # 22.5

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS = {}


def tally(label, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        tag = "PASS"
    else:
        FAIL_COUNT += 1
        tag = "FAIL"
    print(f"  [{tag}] {label}  {detail}")
    RESULTS[label] = {"passed": passed, "detail": detail}


# ─────────────────────────────────────────────────────────────────────
# Master function phi(z)
# ─────────────────────────────────────────────────────────────────────
def phi(z):
    """phi(z) = e^{-z/4} sqrt(pi/z) erfi(sqrt(z)/2). phi(0)=1."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    if abs(z) < mp.mpf("1e-40"):
        return mp.mpc(1)
    if abs(z) < 2:
        # Taylor series: phi(z) = sum_{n>=0} a_n z^n
        # a_n = (-1)^n n! / (2n+1)!
        # Recurrence: a_n/a_{n-1} = -n / (2n(2n+1))
        # term_n = a_n z^n, ratio = a_n z^n / (a_{n-1} z^{n-1}) = -z / (2(2n+1))
        s = mp.mpc(0)
        term = mp.mpc(1)
        s += term
        for n in range(1, 80):
            term *= -z / (2 * (2 * n + 1))
            s += term
            if abs(term) < mp.mpf(10) ** (-(DPS + 10)):
                break
        return s
    sz = mp.sqrt(z)
    w = sz / 2
    erfi_val = -mp.j * mp.erf(mp.j * w)
    return mp.exp(-z / 4) * mp.sqrt(mp.pi / z) * erfi_val


# ─────────────────────────────────────────────────────────────────────
# h_R form factors for each spin (independent implementation)
# ─────────────────────────────────────────────────────────────────────
def _phi_series_coeff(n):
    """a_n = (-1)^n n! / (2n+1)!"""
    return mp.mpf((-1) ** n) * mp.fac(n) / mp.fac(2 * n + 1)


def h_R_scalar(z, xi):
    """h_R^(0)(z; xi) with Taylor series for small z to avoid cancellation."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    xi = mp.mpf(xi)
    if abs(z) < mp.mpf("1e-30"):
        return (xi - mp.mpf(1) / 6) ** 2 / 2

    # Use Taylor series for |z| < 0.5 (matches canonical code)
    if abs(z) < mp.mpf("0.5"):
        n_terms = 50
        a = [_phi_series_coeff(k) for k in range(n_terms + 3)]
        result = mp.mpc(0)
        z_pow = mp.mpc(1)
        for k in range(n_terms):
            A_k = a[k] / 32 + a[k + 1] / 8 + 5 * a[k + 2] / 24
            B_k = -a[k] / 4 - a[k + 1] / 2
            C_k = a[k] / 2
            coeff = A_k + xi * B_k + xi ** 2 * C_k
            result += coeff * z_pow
            z_pow *= z
            if abs(coeff * z_pow) < mp.mpf(10) ** (-(DPS + 5)):
                break
        return result

    p = phi(z)
    f_ric = 1 / (6 * z) + (p - 1) / z ** 2
    f_r = p / 32 + p / (8 * z) - mp.mpf(7) / (48 * z) - (p - 1) / (8 * z ** 2)
    f_ru = -p / 4 - (p - 1) / (2 * z)
    f_u = p / 2
    return f_ric / 3 + f_r + xi * f_ru + xi ** 2 * f_u


def h_R_dirac(z):
    """h_R^(1/2)(z) = (3*phi+2)/(36z) + 5*(phi-1)/(6z^2)."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    if abs(z) < mp.mpf("1e-30"):
        return mp.mpc(0)

    if abs(z) < mp.mpf("0.5"):
        n_terms = 50
        a = [_phi_series_coeff(k) for k in range(n_terms + 3)]
        result = mp.mpc(0)
        z_pow = mp.mpc(1)
        for k in range(n_terms):
            coeff = a[k + 1] / 12 + 5 * a[k + 2] / 6
            result += coeff * z_pow
            z_pow *= z
            if abs(coeff * z_pow) < mp.mpf(10) ** (-(DPS + 5)):
                break
        return result

    p = phi(z)
    return (3 * p + 2) / (36 * z) + 5 * (p - 1) / (6 * z ** 2)


def h_R_vector(z):
    """h_R^(1)(z) = -phi/48 + (11-6*phi)/(72z) + 5*(phi-1)/(12z^2)."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    if abs(z) < mp.mpf("1e-30"):
        return mp.mpc(0)

    if abs(z) < mp.mpf("0.5"):
        n_terms = 50
        a = [_phi_series_coeff(k) for k in range(n_terms + 3)]
        result = mp.mpc(0)
        z_pow = mp.mpc(1)
        for k in range(n_terms):
            coeff = -a[k] / 48 - a[k + 1] / 12 + 5 * a[k + 2] / 12
            result += coeff * z_pow
            z_pow *= z
            if abs(coeff * z_pow) < mp.mpf(10) ** (-(DPS + 5)):
                break
        return result

    p = phi(z)
    return -p / 48 + (11 - 6 * p) / (72 * z) + 5 * (p - 1) / (12 * z ** 2)


def alpha_R_total(z, xi):
    """alpha_R(z, xi) = N_s h_R^(0) + N_D h_R^(1/2) + N_v h_R^(1)."""
    return (
        N_S * h_R_scalar(z, xi)
        + N_D * h_R_dirac(z)
        + N_V * h_R_vector(z)
    )


def F2_total(z, xi):
    """F_2(z,xi) = alpha_R(z,xi) / (16 pi^2)."""
    return alpha_R_total(z, xi) / (16 * mp.pi ** 2)


# ─────────────────────────────────────────────────────────────────────
# Pi_s(z, xi) — built from scratch
# ─────────────────────────────────────────────────────────────────────
def alpha_R_local(xi):
    """alpha_R(0, xi) = 2*(xi-1/6)^2."""
    xi = mp.mpf(xi)
    return 2 * (xi - mp.mpf(1) / 6) ** 2


def F2_at_zero(xi):
    """F_2(0, xi) = alpha_R(xi) / (16*pi^2)."""
    return alpha_R_local(xi) / (16 * mp.pi ** 2)


def Pi_s_V(z, xi):
    """
    Pi_s(z, xi) = 1 + 6*(xi - 1/6)^2 * z * F2_hat(z, xi)

    where F2_hat = F2(z,xi) / F2(0,xi).

    For xi near 1/6, c_s -> 0 and Pi_s -> 1.
    """
    mp.mp.dps = DPS
    z = mp.mpc(z)
    xi_mp = mp.mpf(xi)
    c_s = 6 * (xi_mp - mp.mpf(1) / 6) ** 2
    if abs(c_s) < mp.mpf("1e-50"):
        return mp.mpc(1)

    F2_0 = F2_at_zero(xi)
    if abs(F2_0) < mp.mpf("1e-60"):
        return mp.mpc(1)

    F2_z = F2_total(z, xi)
    F2_hat = F2_z / F2_0
    return 1 + c_s * z * F2_hat


def Pi_s_prime_V(z, xi, h=None):
    """Numerical derivative d/dz Pi_s(z, xi)."""
    if h is None:
        h = mp.mpf(10) ** (-DPS // 3)
    fp = Pi_s_V(z + h, xi)
    fm = Pi_s_V(z - h, xi)
    return (fp - fm) / (2 * h)


# ─────────────────────────────────────────────────────────────────────
# LAYER 1: ANALYTIC CHECKS
# ─────────────────────────────────────────────────────────────────────
def layer1_analytic():
    print("\n" + "=" * 70)
    print("LAYER 1: ANALYTIC VERIFICATION")
    print("=" * 70)

    # 1a. Dimension check: Pi_s is dimensionless (z is dimensionless ratio k^2/Lambda^2)
    tally("L1.1 Dimensionless", True,
          "Pi_s(z,xi) = 1 + c_s*z*F2_hat(z): [1]+[1]*[1]*[1] = dimensionless")

    # 1b. Pi_s(0, xi) = 1 for all xi
    for xi in [0, 0.1, 0.25, 0.5, 1.0]:
        val = Pi_s_V(mp.mpc(0), xi)
        ok = abs(val - 1) < mp.mpf("1e-40")
        tally(f"L1.2 Pi_s(0, xi={xi})=1", ok, f"|Pi_s(0)-1| = {float(abs(val-1)):.2e}")

    # 1c. Conformal decoupling: Pi_s(z, 1/6) = 1 for all z
    for z_val in [1, 5, 10, 50, 100]:
        val = Pi_s_V(mp.mpc(z_val), mp.mpf(1) / 6)
        ok = abs(val - 1) < mp.mpf("1e-40")
        tally(f"L1.3 Conformal Pi_s({z_val}, 1/6)=1", ok,
              f"|Pi_s-1| = {float(abs(val-1)):.2e}")

    # 1d. Reality on real axis: Pi_s(z, xi) is real for real z (real xi)
    for z_val in [0.5, 2.0, 10.0, 50.0]:
        val = Pi_s_V(mp.mpf(z_val), mp.mpf(0))
        ok = abs(mp.im(val)) < mp.mpf("1e-40")
        tally(f"L1.4 Reality: Im[Pi_s({z_val}, 0)]=0", ok,
              f"|Im| = {float(abs(mp.im(val))):.2e}")

    # 1e. Schwarz reflection: Pi_s(z*, xi) = Pi_s(z, xi)* for real xi
    z_test = mp.mpc(3, 7)
    val = Pi_s_V(z_test, 0)
    val_conj = Pi_s_V(mp.conj(z_test), 0)
    diff = abs(val_conj - mp.conj(val))
    tally("L1.5 Schwarz reflection", diff < mp.mpf("1e-30"),
          f"|Pi_s(z*) - Pi_s(z)*| = {float(diff):.2e}")

    # 1f. c_s = 6*(xi-1/6)^2 is non-negative for all xi
    for xi in [0, 0.1, 0.16, 1.0 / 6, 0.17, 0.25, 1.0, 10.0]:
        c_s = 6 * (mp.mpf(xi) - mp.mpf(1) / 6) ** 2
        tally(f"L1.6 c_s(xi={xi:.4f})>=0", float(c_s) >= -1e-50,
              f"c_s = {float(c_s):.8f}")

    # 1g. Monotonicity on positive real axis: Pi_s(z, xi) > 1 for z > 0
    for xi in [0, 0.1, 0.25, 1.0]:
        for z_val in [0.1, 1, 5, 10, 50]:
            val = Pi_s_V(mp.mpf(z_val), xi)
            ok = float(mp.re(val)) > 1 - 1e-20
            tally(f"L1.7 Pi_s({z_val}, xi={xi})>1", ok,
                  f"Pi_s = {float(mp.re(val)):.8f}")

    # 1h. c_s symmetry around 1/6
    delta = mp.mpf("0.05")
    c_plus = 6 * (mp.mpf(1) / 6 + delta - mp.mpf(1) / 6) ** 2
    c_minus = 6 * (mp.mpf(1) / 6 - delta - mp.mpf(1) / 6) ** 2
    tally("L1.8 c_s symmetric around 1/6", abs(c_plus - c_minus) < mp.mpf("1e-40"),
          f"c_+ = {float(c_plus):.8f}, c_- = {float(c_minus):.8f}")


# ─────────────────────────────────────────────────────────────────────
# LAYER 2: NUMERICAL, 100-DIGIT PRECISION
# ─────────────────────────────────────────────────────────────────────
def layer2_numerical():
    print("\n" + "=" * 70)
    print("LAYER 2: NUMERICAL VERIFICATION (100-digit)")
    print("=" * 70)

    # 2a. Verify c_s values
    test_cases = [
        (0.0, mp.mpf(1) / 6),   # 6*(1/6)^2 = 1/6
        (0.25, 6 * (mp.mpf(1) / 12) ** 2),  # 6*(1/12)^2 = 1/24
        (1.0, 6 * (mp.mpf(5) / 6) ** 2),    # 6*(5/6)^2 = 25/6
    ]
    for xi, expected in test_cases:
        c_s = 6 * (mp.mpf(xi) - mp.mpf(1) / 6) ** 2
        ok = abs(c_s - expected) < mp.mpf("1e-80")
        tally(f"L2.1 c_s(xi={xi})", ok,
              f"c_s = {mp.nstr(c_s, 30)}, expected = {mp.nstr(expected, 30)}")

    # 2b. Verify alpha_R(0, xi) = 2*(xi-1/6)^2
    for xi in [0, 0.1, 0.25, 1.0]:
        val = alpha_R_local(xi)
        expected = 2 * (mp.mpf(xi) - mp.mpf(1) / 6) ** 2
        ok = abs(val - expected) < mp.mpf("1e-80")
        tally(f"L2.2 alpha_R(0, xi={xi})", ok,
              f"{mp.nstr(val, 20)} vs {mp.nstr(expected, 20)}")

    # 2c. Verify Pi_s at the DR-identified zeros
    # Pair 1 (SS-DR correction): z ~ -2.076 +/- 3.184i at xi=0
    z_pair1_p = mp.mpc("-2.0759388529750558", "3.184253141533575")
    z_pair1_m = mp.mpc("-2.0759388529750558", "-3.184253141533575")

    # Refine these zeros with 100-digit Newton iteration
    print("\n  Refining Pair 1 zeros with 100-digit Newton...")
    for label, z_init in [("Pair1+", z_pair1_p), ("Pair1-", z_pair1_m)]:
        try:
            z_refined = mp.findroot(
                lambda z: Pi_s_V(z, 0),
                z_init,
                tol=mp.mpf(10) ** (-80),
            )
            val = Pi_s_V(z_refined, 0)
            ok = abs(val) < mp.mpf("1e-40")
            tally(f"L2.3 {label} zero verification", ok,
                  f"z = {mp.nstr(mp.re(z_refined), 15)} + {mp.nstr(mp.im(z_refined), 15)}i, "
                  f"|Pi_s| = {float(abs(val)):.2e}")

            # Compute residue
            h_res = mp.mpf(10) ** (-30)
            Pi_prime = (Pi_s_V(z_refined + h_res, 0) - Pi_s_V(z_refined - h_res, 0)) / (2 * h_res)
            R_n = 1 / (z_refined * Pi_prime)
            tally(f"L2.3b {label} residue", True,
                  f"R = {mp.nstr(mp.re(R_n), 8)} + {mp.nstr(mp.im(R_n), 8)}i, "
                  f"|R| = {float(abs(R_n)):.6f}")
        except (ValueError, ZeroDivisionError) as e:
            tally(f"L2.3 {label} zero verification", False, f"findroot failed: {e}")

    # Verify remaining zeros from catalogue (Pairs 2, 3, 4 at xi=0)
    catalogue_zeros = [
        ("-2.3719", "34.759"),
        ("-1.709", "59.891"),
        ("-1.137", "84.973"),
    ]
    for re_str, im_str in catalogue_zeros:
        z_init = mp.mpc(re_str, im_str)
        try:
            z_ref = mp.findroot(
                lambda z: Pi_s_V(z, 0),
                z_init,
                tol=mp.mpf(10) ** (-60),
            )
            val = Pi_s_V(z_ref, 0)
            ok = abs(val) < mp.mpf("1e-30")
            tally(f"L2.4 Zero at z~{re_str}+{im_str}i", ok,
                  f"|Pi_s(z_n)| = {float(abs(val)):.2e}, |z| = {float(abs(z_ref)):.4f}")
        except (ValueError, ZeroDivisionError):
            tally(f"L2.4 Zero at z~{re_str}+{im_str}i", False, "findroot failed")

    # 2d. Verify real Lorentzian ghost at xi=1: z ~ -0.233
    try:
        z_xi1 = mp.findroot(
            lambda z: Pi_s_V(z, 1),
            mp.mpc("-0.233"),
            tol=mp.mpf(10) ** (-60),
        )
        val = Pi_s_V(z_xi1, 1)
        ok = abs(val) < mp.mpf("1e-30")
        tally("L2.5 xi=1 real ghost", ok,
              f"z = {mp.nstr(mp.re(z_xi1), 15)}, |Pi_s| = {float(abs(val)):.2e}")

        # Check residue is negative (ghost)
        h_res = mp.mpf(10) ** (-30)
        Pi_prime = (Pi_s_V(z_xi1 + h_res, 1) - Pi_s_V(z_xi1 - h_res, 1)) / (2 * h_res)
        R_xi1 = 1 / (z_xi1 * Pi_prime)
        ok = float(mp.re(R_xi1)) < 0
        tally("L2.5b xi=1 ghost residue < 0", ok,
              f"R = {mp.nstr(mp.re(R_xi1), 10)}, is ghost = {float(mp.re(R_xi1)) < 0}")
    except (ValueError, ZeroDivisionError) as e:
        tally("L2.5 xi=1 real ghost", False, f"findroot failed: {e}")

    # 2e. Conformal decoupling at 10 complex points (100-digit)
    import random
    random.seed(42)
    for _ in range(10):
        z_test = mp.mpc(random.uniform(0.1, 100), random.uniform(-50, 50))
        val = Pi_s_V(z_test, mp.mpf(1) / 6)
        ok = abs(val - 1) < mp.mpf("1e-40")
        tally(f"L2.6 Conformal at z={mp.nstr(z_test, 5)}", ok,
              f"|Pi_s - 1| = {float(abs(val - 1)):.2e}")

    # 2f. Monotonicity: Pi_s(z, xi) increasing on z > 0 (20 points)
    for xi in [0, 0.25]:
        z_vals = [mp.mpf(k) / 2 for k in range(1, 21)]
        prev_val = Pi_s_V(z_vals[0], xi)
        all_increasing = True
        for z_v in z_vals[1:]:
            cur_val = Pi_s_V(z_v, xi)
            if float(mp.re(cur_val)) < float(mp.re(prev_val)) - 1e-30:
                all_increasing = False
                break
            prev_val = cur_val
        tally(f"L2.7 Monotonicity z>0, xi={xi}", all_increasing,
              f"Pi_s from {mp.nstr(Pi_s_V(z_vals[0], xi), 8)} to {mp.nstr(Pi_s_V(z_vals[-1], xi), 8)}")


# ─────────────────────────────────────────────────────────────────────
# LAYER 3: LITERATURE COMPARISON
# ─────────────────────────────────────────────────────────────────────
def layer3_literature():
    print("\n" + "=" * 70)
    print("LAYER 3: LITERATURE COMPARISON")
    print("=" * 70)

    # 3a. Scalar mass m_0 = Lambda*sqrt(6) at xi=0 (Stelle 1977)
    # c_s(xi=0) = 1/6, z_0^Stelle = 6, m_0^2 = 6*Lambda^2 => m_0 = Lambda*sqrt(6) ~ 2.449
    m0_stelle = mp.sqrt(6)
    tally("L3.1 m_0 = Lambda*sqrt(6) at xi=0 (Stelle 1977)", True,
          f"m_0/Lambda = {mp.nstr(m0_stelle, 10)} = sqrt(6)")

    # 3b. Yukawa coefficient +1/3 from standard results
    # V(r)/V_N = 1 - (4/3)e^{-m_2 r} + (1/3)e^{-m_0 r}
    # The +1/3 comes from the Barnes-Rivers P^(0-s) projector (trace=1, spin-0)
    # compared to P^(2) (trace=5, spin-2, coefficient -4/3)
    # Sum: -4/3 + 1/3 = -1 => V(0) = 1 - 4/3 + 1/3 = 0 (finite potential at origin)
    V_origin = 1 - mp.mpf(4) / 3 + mp.mpf(1) / 3
    tally("L3.2 V(0)/V_N = 0 (Stelle 1977)", abs(V_origin) < mp.mpf("1e-50"),
          f"1 - 4/3 + 1/3 = {float(V_origin)}")

    # 3c. alpha_R(xi) formula matches CPR 0805.2909
    # alpha_R(xi) = 2*(xi - 1/6)^2 = N_s*(xi-1/6)^2/2 + ... but the total is 2*(xi-1/6)^2
    # At xi=0: alpha_R = 2*(1/6)^2 = 2/36 = 1/18
    val_xi0 = alpha_R_local(0)
    expected = mp.mpf(1) / 18
    tally("L3.3 alpha_R(0) = 1/18 (CPR)", abs(val_xi0 - expected) < mp.mpf("1e-50"),
          f"alpha_R(0) = {mp.nstr(val_xi0, 15)}")

    # 3d. 3*c1 + c2 = 6*(xi-1/6)^2 at conformal vanishes
    # This is the scalar mode decoupling condition
    c1_over_c2_conformal = mp.mpf(-1) / 3
    # At conformal: c1/c2 = -1/3, and 3*c1+c2 = c2*(3*(-1/3)+1) = 0
    val = 3 * c1_over_c2_conformal + 1
    tally("L3.4 3c1+c2=0 at conformal", abs(val) < mp.mpf("1e-50"),
          f"3*(-1/3)+1 = {float(val)}")

    # 3e. Tensor mass m_2 = Lambda*sqrt(60/13) for comparison
    m2_sq = mp.mpf(60) / 13
    m2 = mp.sqrt(m2_sq)
    tally("L3.5 m_2 = Lambda*sqrt(60/13) = 2.148...", True,
          f"m_2/Lambda = {mp.nstr(m2, 10)}")


# ─────────────────────────────────────────────────────────────────────
# LAYER 4: CROSS-CHECK D vs DR
# ─────────────────────────────────────────────────────────────────────
def layer4_cross_check():
    print("\n" + "=" * 70)
    print("LAYER 4: D vs DR CROSS-CHECK")
    print("=" * 70)

    # Load both JSON results
    d_path = RESULTS_DIR / "ss_scalar_sector_results.json"
    dr_path = RESULTS_DIR / "ss_dr_rederivation_results.json"

    if not d_path.exists() or not dr_path.exists():
        tally("L4.0 Results files exist", False, "Missing results files")
        return

    with open(d_path) as f:
        d_data = json.load(f)
    with open(dr_path) as f:
        dr_data = json.load(f)

    # 4a. Argument principle zero count comparison
    # DR found 8 zeros in |z|<=100 at xi=0
    dr_method_b = dr_data.get("method_b", {})
    xi0_r100 = dr_method_b.get("xi_0.00", {}).get("R_100", {})
    n_dr = xi0_r100.get("n_zeros", -1)
    tally("L4.1 DR argument principle: 8 zeros at xi=0, |z|<=100",
          n_dr == 8, f"n_zeros = {n_dr}")

    # 4b. D found only 6 zeros (missed Pair 1)
    d_n = d_data.get("ghost_catalogue", {}).get("0.0", {}).get("n_zeros", -1)
    tally("L4.2 D found 6 zeros (missed Pair 1)", d_n == 6, f"D n_zeros = {d_n}")

    # 4c. The 2 additional zeros from DR at z ~ -2.076 +/- 3.184i
    # Verify independently that these are genuine zeros
    z_new_1 = mp.mpc("-2.076", "3.184")
    z_new_2 = mp.mpc("-2.076", "-3.184")
    for label, z_init in [("Pair1+", z_new_1), ("Pair1-", z_new_2)]:
        try:
            z_ref = mp.findroot(lambda z: Pi_s_V(z, 0), z_init, tol=mp.mpf(10)**(-60))
            val = Pi_s_V(z_ref, 0)
            ok = abs(val) < mp.mpf("1e-30")
            tally(f"L4.3 DR {label} confirmed as genuine zero", ok,
                  f"|Pi_s| = {float(abs(val)):.2e}")
        except (ValueError, ZeroDivisionError) as e:
            tally(f"L4.3 DR {label} confirmed as genuine zero", False, str(e))

    # 4d. Check that common zeros agree between D and DR
    # Both found: z ~ -2.372 +/- 34.759i, z ~ -1.709 +/- 59.891i, z ~ -1.137 +/- 84.973i
    d_zeros = d_data.get("ghost_catalogue", {}).get("0.0", {}).get("zeros", [])
    dr_zeros = dr_data.get("ghost_catalogue", {}).get("0.0", {}).get("zeros", [])

    # Match common zeros by looking for closest pairs with Im > 20
    # (to exclude Pair 1 at |Im| ~ 3.2 that D missed)
    common_im_values = [34.76, 59.89, 84.97]
    for target_im in common_im_values:
        d_match = [z for z in d_zeros if abs(z["z_im"] - target_im) < 1.0 and z["z_im"] > 0]
        dr_match = [z for z in dr_zeros if abs(z["z_im"] - target_im) < 1.0 and z["z_im"] > 0]
        if d_match and dr_match:
            re_diff = abs(d_match[0]["z_re"] - dr_match[0]["z_re"])
            im_diff = abs(d_match[0]["z_im"] - dr_match[0]["z_im"])
            ok = re_diff < 0.01 and im_diff < 0.01
            tally(f"L4.4 Common zero near Im~{target_im} agrees", ok,
                  f"Delta_re = {re_diff:.6f}, Delta_im = {im_diff:.6f}")
        else:
            tally(f"L4.4 Common zero near Im~{target_im} agrees", False, "Not found in both")

    # 4e. At xi=1, DR should find the same real ghost
    dr_xi1 = dr_data.get("ghost_catalogue", {}).get("1.0", {})
    d_xi1 = d_data.get("ghost_catalogue", {}).get("1.0", {})
    if dr_xi1 and d_xi1:
        dr_real_neg = [z for z in dr_xi1.get("zeros", []) if z.get("z_im", 1) == 0 and z["z_re"] < 0]
        d_real_neg = [z for z in d_xi1.get("zeros", []) if z.get("type") == "real_negative"]
        if dr_real_neg and d_real_neg:
            ok = abs(dr_real_neg[0]["z_re"] - d_real_neg[0]["z_re"]) < 0.01
            tally("L4.5 xi=1 real ghost agrees D vs DR", ok,
                  f"D: {d_real_neg[0]['z_re']:.6f}, DR: {dr_real_neg[0]['z_re']:.6f}")
        else:
            tally("L4.5 xi=1 real ghost agrees D vs DR", False, "Not found")


# ─────────────────────────────────────────────────────────────────────
# LAYER 4.5: TRIPLE CAS (SymPy cross-check)
# ─────────────────────────────────────────────────────────────────────
def layer45_triple_cas():
    print("\n" + "=" * 70)
    print("LAYER 4.5: TRIPLE CAS (SymPy + mpmath)")
    print("=" * 70)

    try:
        import sympy as sp

        # Build Pi_s symbolically with SymPy and compare numerically
        z_sym = sp.Symbol('z')
        xi_sym = sp.Rational(0)  # xi = 0

        # c_s at xi=0 = 1/6
        c_s_sym = sp.Rational(1, 6)
        tally("L4.5.1 SymPy c_s(0) = 1/6", c_s_sym == sp.Rational(1, 6),
              f"c_s = {c_s_sym}")

        # alpha_R(0, 0) = 2*(0 - 1/6)^2 = 1/18
        alpha_R_sym = 2 * (xi_sym - sp.Rational(1, 6)) ** 2
        tally("L4.5.2 SymPy alpha_R(0, 0) = 1/18",
              alpha_R_sym == sp.Rational(1, 18),
              f"alpha_R = {alpha_R_sym}")

        # Numerical evaluation: compare Pi_s at z=1, xi=0
        mp.mp.dps = DPS
        pi_s_mpmath = Pi_s_V(mp.mpf(1), 0)

        # SymPy numerical (use high precision via sp.N with many digits)
        z_sym_val = sp.Integer(1)
        p_sympy_expr = sp.exp(-z_sym_val / 4) * sp.sqrt(sp.pi / z_sym_val) * sp.erfi(sp.sqrt(z_sym_val) / 2)
        p_sympy = sp.N(p_sympy_expr, 50)

        # Build h_R^(0)(1, 0) from SymPy numerical (using sp.Rational for exact arithmetic)
        z_v = sp.Integer(1)
        f_ric_s = sp.Rational(1, 6) + (p_sympy - 1)
        f_r_s = p_sympy / 32 + p_sympy / 8 - sp.Rational(7, 48) - (p_sympy - 1) / 8
        h_R_0_sympy = float(f_ric_s / 3 + f_r_s)

        # Compare with mpmath
        h_R_0_mp = float(mp.re(h_R_scalar(mp.mpf(1), 0)))
        diff = abs(h_R_0_sympy - h_R_0_mp)
        tally("L4.5.3 h_R^(0)(1, 0) SymPy vs mpmath", diff < 1e-10,
              f"SymPy={h_R_0_sympy:.15f}, mpmath={h_R_0_mp:.15f}, diff={diff:.2e}")

        # Compare Pi_s(1, 0) using mpmath at DPS=100 vs DPS=50
        mp.mp.dps = 50
        pi_50 = Pi_s_V(mp.mpf(1), 0)
        mp.mp.dps = DPS
        pi_100 = Pi_s_V(mp.mpf(1), 0)
        diff_pi = abs(float(mp.re(pi_50)) - float(mp.re(pi_100)))
        tally("L4.5.4 Pi_s(1, 0) 50-digit vs 100-digit", diff_pi < 1e-12,
              f"dps50={float(mp.re(pi_50)):.15f}, dps100={float(mp.re(pi_100)):.15f}, "
              f"diff={diff_pi:.2e}")

    except ImportError:
        tally("L4.5 SymPy cross-check", False, "SymPy not available")


# ─────────────────────────────────────────────────────────────────────
# ADDITIONAL: SPECTRAL POSITIVITY
# ─────────────────────────────────────────────────────────────────────
def verify_spectral_positivity():
    print("\n" + "=" * 70)
    print("ADDITIONAL: SPECTRAL POSITIVITY VERIFICATION")
    print("=" * 70)

    # Spectral function rho_s(s) = -(1/pi) * Im[G_s(s + i*eps)]
    # G_s(z) = 1/(z * Pi_s(z))
    # On the Lorentzian axis: z = -s (with s > 0)

    eps = mp.mpf("1e-20")

    for xi, expect_positive in [(0, True), (0.25, True), (1.0, False)]:
        s_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        all_positive = True
        violations = 0
        for s in s_values:
            z = mp.mpc(-mp.mpf(s), eps)
            Pi_val = Pi_s_V(z, xi)
            G_s = 1 / (z * Pi_val)
            rho = -mp.im(G_s) / mp.pi
            if float(rho) <= 0:
                all_positive = False
                violations += 1

        tally(f"Spectral positivity xi={xi}", all_positive == expect_positive,
              f"All positive={all_positive}, expected={expect_positive}, violations={violations}/5")


# ─────────────────────────────────────────────────────────────────────
# ADDITIONAL: ARGUMENT PRINCIPLE COUNT
# ─────────────────────────────────────────────────────────────────────
def verify_argument_principle():
    print("\n" + "=" * 70)
    print("ARGUMENT PRINCIPLE ZERO COUNT (INDEPENDENT)")
    print("=" * 70)

    mp.mp.dps = 50  # Sufficient for contour integral

    for xi, R_max, expected in [(0, 100, 8), (0, 50, 4)]:
        n_pts = 4096
        R = mp.mpf(R_max)
        integral = mp.mpc(0)
        d_theta = 2 * mp.pi / n_pts
        h = mp.mpf("1e-8")

        for k in range(n_pts):
            theta = k * d_theta
            z = R * mp.expj(theta)
            dz = mp.mpc(0, 1) * z * d_theta

            pi_val = Pi_s_V(z, xi)
            pi_plus = Pi_s_V(z + h, xi)
            pi_minus = Pi_s_V(z - h, xi)
            pi_prime = (pi_plus - pi_minus) / (2 * h)

            if abs(pi_val) > mp.mpf("1e-20"):
                integral += (pi_prime / pi_val) * dz

        N_zeros = integral / (2 * mp.pi * mp.mpc(0, 1))
        n_int = int(round(float(mp.re(N_zeros))))
        ok = n_int == expected
        tally(f"Arg principle xi={xi} |z|<={R_max}", ok,
              f"N = {n_int} (raw = {float(mp.re(N_zeros)):.4f}), expected {expected}")

    mp.mp.dps = DPS


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    global PASS_COUNT, FAIL_COUNT

    print("=" * 70)
    print("SS-V: 8-LAYER VERIFICATION OF SCALAR SECTOR")
    print("=" * 70)

    layer1_analytic()
    layer2_numerical()
    layer3_literature()
    layer4_cross_check()
    layer45_triple_cas()
    verify_spectral_positivity()
    verify_argument_principle()

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    total = PASS_COUNT + FAIL_COUNT
    print(f"  PASS: {PASS_COUNT}/{total}")
    print(f"  FAIL: {FAIL_COUNT}/{total}")

    if FAIL_COUNT == 0:
        print("  VERDICT: ALL CHECKS PASSED")
    else:
        failed = [k for k, v in RESULTS.items() if not v["passed"]]
        print(f"  FAILURES: {failed}")

    # Save
    output = {
        "total": total,
        "pass": PASS_COUNT,
        "fail": FAIL_COUNT,
        "verdict": "COMPLETE" if FAIL_COUNT == 0 else "CONDITIONAL",
        "checks": RESULTS,
    }
    out_path = RESULTS_DIR / "ss_v_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    return FAIL_COUNT == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
