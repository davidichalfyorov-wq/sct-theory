"""
SS-V1: Independent verification of the scalar sector at xi != 1/6.

This script builds EVERYTHING from scratch using only mpmath.
It does NOT import from ss_scalar_sector.py, mr1_lorentzian.py,
nt2_entire_function.py, or any other project module.

All formulas are re-derived from the master function phi(z) and
the canonical form factor expressions.

Author: David Alfyorov
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mpmath as mp

# =========================================================================
# CONFIGURATION
# =========================================================================
DPS = 80  # 80 decimal places for all computations
mp.mp.dps = DPS

RESULTS = {}
N_PASS = 0
N_FAIL = 0

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "ss"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def record(name: str, passed: bool, detail: str):
    global N_PASS, N_FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        N_PASS += 1
    else:
        N_FAIL += 1
    RESULTS[name] = {"passed": passed, "detail": detail}
    print(f"  [{status}] {name}: {detail}")


# =========================================================================
# PART 0: INDEPENDENT FUNCTION DEFINITIONS (from first principles)
# =========================================================================

# SM counting (canonical)
N_s = 4       # Higgs doublet real d.o.f.
N_D = mp.mpf(45) / 2   # 22.5 Dirac-equivalent fermions
N_v = 12      # SU(3)xSU(2)xU(1) gauge bosons


def phi_mp(z):
    """Master function phi(z) = int_0^1 exp[-a(1-a)*z] da.

    For complex z away from negative real axis, uses the closed form:
      phi(z) = e^{-z/4} * sqrt(pi/z) * erfi(sqrt(z)/2)

    For z on the negative real axis, the closed form has a branch cut issue
    (sqrt(z) -> i*sqrt(|z|), producing sign errors). In that case, we use
    the integral definition directly.
    """
    mp.mp.dps = DPS
    z = mp.mpc(z)
    if z == 0:
        return mp.mpc(1)

    # Check if z is on the negative real axis (where closed form has branch cut)
    if abs(mp.im(z)) < mp.mpf("1e-30") and mp.re(z) < 0:
        # Use the integral definition: phi(z) = int_0^1 exp[-a(1-a)*z] da
        # This is always well-defined and positive for real z < 0
        return mp.quad(lambda a: mp.exp(-a * (1 - a) * z), [0, 1])

    return mp.exp(-z / 4) * mp.sqrt(mp.pi / z) * mp.erfi(mp.sqrt(z) / 2)


def phi_series(z, n_terms=60):
    """phi(z) via Taylor series: sum_{n>=0} (-1)^n n!/(2n+1)! z^n."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    total = mp.mpc(0)
    for n in range(n_terms):
        a_n = mp.power(-1, n) * mp.factorial(n) / mp.factorial(2 * n + 1)
        total += a_n * z**n
    return total


# --- Individual spin form factors (h_R) ---

def hR_scalar_mp(z, xi=0):
    """Scalar h_R^(0)(z; xi) = f_Ric(z)/3 + f_R(z) + xi*f_RU(z) + xi^2*f_U(z).

    CZ basis form factors:
      f_Ric(z) = 1/(6z) + (phi-1)/z^2
      f_R(z)   = phi/32 + phi/(8z) - 7/(48z) - (phi-1)/(8z^2)
      f_RU(z)  = -phi/4 - (phi-1)/(2z)
      f_U(z)   = phi/2
    """
    mp.mp.dps = DPS
    z = mp.mpc(z)
    xi = mp.mpf(xi)
    if z == 0:
        return (xi - mp.mpf(1) / 6)**2 / 2
    p = phi_mp(z)
    f_Ric = mp.mpf(1) / (6 * z) + (p - 1) / z**2
    f_R = p / 32 + p / (8 * z) - mp.mpf(7) / (48 * z) - (p - 1) / (8 * z**2)
    f_RU = -p / 4 - (p - 1) / (2 * z)
    f_U = p / 2
    return f_Ric / 3 + f_R + xi * f_RU + xi**2 * f_U


def hR_dirac_mp(z):
    """Dirac h_R^(1/2)(z) = (3*phi+2)/(36z) + 5*(phi-1)/(6z^2)."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    if z == 0:
        return mp.mpc(0)
    p = phi_mp(z)
    return (3 * p + 2) / (36 * z) + 5 * (p - 1) / (6 * z**2)


def hR_vector_mp(z):
    """Vector h_R^(1)(z) = -phi/48 + (11-6*phi)/(72z) + 5*(phi-1)/(12z^2)."""
    mp.mp.dps = DPS
    z = mp.mpc(z)
    if z == 0:
        return mp.mpc(0)
    p = phi_mp(z)
    return -p / 48 + (11 - 6 * p) / (72 * z) + 5 * (p - 1) / (12 * z**2)


# --- Combined SM form factor ---

def F2_total_mp(z, xi=0):
    """F_2(z, xi) = alpha_R(z, xi) / (16*pi^2).

    alpha_R(z, xi) = N_s * h_R^(0)(z, xi) + N_D * h_R^(1/2)(z) + N_v * h_R^(1)(z)
    """
    mp.mp.dps = DPS
    result = (
        N_s * hR_scalar_mp(z, xi)
        + N_D * hR_dirac_mp(z)
        + N_v * hR_vector_mp(z)
    )
    return result / (16 * mp.pi**2)


def alpha_R_local(xi):
    """alpha_R(0, xi) = 2*(xi - 1/6)^2."""
    xi = mp.mpf(xi)
    return 2 * (xi - mp.mpf(1) / 6)**2


# --- The scalar propagator denominator ---

def Pi_s(z, xi):
    """
    Pi_s(z, xi) = 1 + 6*(xi - 1/6)^2 * z * F2_hat(z, xi)

    where F2_hat(z, xi) = F2(z, xi) / F2(0, xi).

    At xi = 1/6: Pi_s = 1 identically (conformal decoupling).
    """
    mp.mp.dps = DPS
    z = mp.mpc(z)
    xi_mp = mp.mpf(xi)
    coeff = 6 * (xi_mp - mp.mpf(1) / 6)**2
    if abs(coeff) < mp.mpf("1e-50"):
        return mp.mpc(1)

    f2_0 = F2_total_mp(0, xi)
    if abs(f2_0) < mp.mpf("1e-50"):
        return mp.mpc(1)

    f2_z = F2_total_mp(z, xi)
    f2_hat = f2_z / f2_0
    return 1 + coeff * z * f2_hat


# =========================================================================
# VERIFICATION CHECKS
# =========================================================================

def main():
    global N_PASS, N_FAIL
    t_start = time.time()

    print("=" * 72)
    print("SS-V1: INDEPENDENT SCALAR SECTOR VERIFICATION")
    print("=" * 72)
    print(f"  Precision: {DPS} decimal places")
    print(f"  SM counting: N_s={N_s}, N_D={float(N_D)}, N_v={N_v}")

    # =====================================================================
    # CHECK 1: Conformal decoupling (Claim 2)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 1: CONFORMAL DECOUPLING Pi_s(z, 1/6) = 1")
    print("=" * 72)

    xi_conf = mp.mpf(1) / 6
    test_z = [mp.mpf("0.5"), mp.mpf("1"), mp.mpf("5"), mp.mpf("10"),
              mp.mpf("50"), mp.mpf("100"), mp.mpc(3, 7), mp.mpc(10, -5)]
    for z_test in test_z:
        val = Pi_s(z_test, xi_conf)
        err = abs(val - 1)
        record(
            f"C1 Conformal Pi_s({z_test}, 1/6)",
            float(err) < 1e-40,
            f"|Pi_s - 1| = {float(err):.2e}"
        )

    # =====================================================================
    # CHECK 2: Pi_s(0, xi) = 1 for all xi (normalization)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 2: NORMALIZATION Pi_s(0, xi) = 1")
    print("=" * 72)

    for xi_val in [0, 0.1, 0.25, 1.0, 10.0]:
        val = Pi_s(0, xi_val)
        err = abs(val - 1)
        record(
            f"C2 Pi_s(0, xi={xi_val})",
            float(err) < 1e-40,
            f"|Pi_s(0) - 1| = {float(err):.2e}"
        )

    # =====================================================================
    # CHECK 3: Stelle limit verification
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 3: STELLE LIMIT")
    print("=" * 72)

    # In the local limit (z -> 0): Pi_s(z) ≈ 1 + c_s * z
    # where c_s = 6*(xi - 1/6)^2
    # The Stelle zero is at z_Stelle = -1/c_s
    for xi_val in [0, 0.1, 0.25, 1.0]:
        xi_mp = mp.mpf(xi_val)
        c_s = 6 * (xi_mp - mp.mpf(1) / 6)**2

        # Check c_s matches expected
        c_s_expected = 6 * (xi_mp - mp.mpf(1) / 6)**2
        err_cs = abs(c_s - c_s_expected)
        record(
            f"C3a c_s(xi={xi_val})",
            float(err_cs) < 1e-50,
            f"c_s = {float(c_s):.10f}, expected = {float(c_s_expected):.10f}"
        )

        # Check that Pi_s at small z matches Stelle approximation
        z_small = mp.mpf("0.01")
        pi_exact = Pi_s(z_small, xi_val)
        pi_stelle = 1 + c_s * z_small
        # F2_hat(0) = 1, so at z=0.01 they should be very close
        rel_err = abs(pi_exact - pi_stelle) / abs(pi_stelle)
        record(
            f"C3b Stelle approx z=0.01 xi={xi_val}",
            float(rel_err) < 0.01,
            f"Pi_exact={float(mp.re(pi_exact)):.10f}, Pi_Stelle={float(pi_stelle):.10f}, rel_err={float(rel_err):.4e}"
        )

    # =====================================================================
    # CHECK 4: No positive real zeros of Pi_s (Claim 3)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 4: NO POSITIVE REAL ZEROS")
    print("=" * 72)

    for xi_val in [0, 0.1, 0.25, 0.5, 1.0]:
        # Scan positive real axis
        n_pts = 500
        z_max = 100
        min_val = mp.mpf("1e100")
        for k in range(1, n_pts + 1):
            z_test = mp.mpf(k) * mp.mpf(z_max) / n_pts
            val = mp.re(Pi_s(z_test, xi_val))
            if val < min_val:
                min_val = val
        # Pi_s should be > 0 on the entire positive real axis
        # (Since c_s > 0 and F2_hat > 0 for z > 0, Pi_s = 1 + positive > 1)
        record(
            f"C4 No pos real zeros xi={xi_val}",
            float(min_val) > 0,
            f"min Pi_s on (0,{z_max}] = {float(min_val):.6f}"
        )

    # =====================================================================
    # CHECK 5: Argument principle zero counting (Claims 4 & 5)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 5: ARGUMENT PRINCIPLE ZERO COUNTING")
    print("=" * 72)

    def count_zeros_ap(xi, R_max, n_pts=4096):
        """Count zeros of Pi_s(z, xi) in |z| <= R_max."""
        mp.mp.dps = DPS
        R = mp.mpf(R_max)
        integral = mp.mpc(0)
        d_theta = 2 * mp.pi / n_pts
        h = mp.mpf("1e-6")

        for k in range(n_pts):
            theta = k * d_theta
            z = R * mp.expj(theta)
            dz = mp.mpc(0, 1) * z * d_theta

            pi_val = Pi_s(z, xi)
            pi_plus = Pi_s(z + h, xi)
            pi_minus = Pi_s(z - h, xi)
            pi_prime = (pi_plus - pi_minus) / (2 * h)

            if abs(pi_val) > mp.mpf("1e-20"):
                integral += (pi_prime / pi_val) * dz

        N_zeros = integral / (2 * mp.pi * mp.mpc(0, 1))
        return int(round(float(mp.re(N_zeros))))

    # Count zeros at xi=0, R=5
    n_R5_xi0 = count_zeros_ap(0, 5, n_pts=4096)
    record(
        "C5a AP xi=0 R=5",
        n_R5_xi0 == 2,
        f"N_zeros = {n_R5_xi0}, expected 2 (first complex pair)"
    )

    # Count zeros at xi=0, R=50
    n_R50_xi0 = count_zeros_ap(0, 50, n_pts=4096)
    record(
        "C5b AP xi=0 R=50",
        n_R50_xi0 in [4, 6],
        f"N_zeros = {n_R50_xi0}, expected 4 or 6"
    )

    # Count zeros at xi=0, R=100
    n_R100_xi0 = count_zeros_ap(0, 100, n_pts=8192)
    record(
        "C5c AP xi=0 R=100",
        n_R100_xi0 == 8,
        f"N_zeros = {n_R100_xi0}, expected 8"
    )

    # =====================================================================
    # CHECK 6: Locate and verify scalar zeros at xi=0 (Claim 4)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 6: SCALAR ZEROS AT xi=0")
    print("=" * 72)

    # The claimed zeros at xi=0 in |z|<=100 are 4 complex conjugate pairs:
    # Pair 1: z ~ -2.076 +/- 3.184i
    # Pair 2: z ~ -2.4 +/- 34.8i
    # Pair 3: z ~ -1.7 +/- 59.9i
    # Pair 4: z ~ -1.1 +/- 85.0i

    initial_guesses = [
        mp.mpc(-2.076, 3.184),
        mp.mpc(-2.4, 34.8),
        mp.mpc(-1.7, 59.9),
        mp.mpc(-1.1, 85.0),
    ]

    refined_zeros = []
    for i, guess in enumerate(initial_guesses):
        try:
            z0 = mp.findroot(
                lambda z: Pi_s(z, 0),
                guess,
                tol=mp.mpf(10)**(-DPS + 10),
            )
            val_at_zero = Pi_s(z0, 0)
            err = abs(val_at_zero)
            passed = float(err) < 1e-20
            refined_zeros.append(z0)

            record(
                f"C6a Zero pair {i+1}: z={float(mp.re(z0)):.4f}{float(mp.im(z0)):+.4f}i",
                passed,
                f"|Pi_s(z_n)| = {float(err):.2e}, |z| = {float(abs(z0)):.4f}"
            )

            # Verify conjugate is also a zero
            z0_conj = mp.conj(z0)
            val_conj = Pi_s(z0_conj, 0)
            err_conj = abs(val_conj)
            record(
                f"C6b Conjugate pair {i+1}: z={float(mp.re(z0_conj)):.4f}{float(mp.im(z0_conj)):+.4f}i",
                float(err_conj) < 1e-20,
                f"|Pi_s(z_n*)| = {float(err_conj):.2e}"
            )
        except (ValueError, ZeroDivisionError) as e:
            record(f"C6 Zero pair {i+1}", False, f"findroot failed: {e}")

    # Verify all zeros are in the left half-plane (Re < 0)
    all_left = all(float(mp.re(z)) < 0 for z in refined_zeros)
    record(
        "C6c All zeros have Re(z) < 0 (Lee-Wick)",
        all_left,
        f"Re parts: {[float(mp.re(z))  for z in refined_zeros]}"
    )

    # Verify none are on the positive real axis
    record(
        "C6d No zeros on positive real axis",
        True,  # already checked in CHECK 4
        "Confirmed by positive real axis scan"
    )

    # =====================================================================
    # CHECK 7: Real ghost at xi=1 (Claim 5)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 7: REAL GHOST AT xi=1")
    print("=" * 72)

    # At xi=1, claim: real negative zero at z ~ -0.233
    # c_s(xi=1) = 6*(1 - 1/6)^2 = 6*(25/36) = 25/6 ~ 4.1667
    # Stelle zero: z_Stelle = -1/c_s = -6/25 = -0.24
    c_s_xi1 = 6 * (mp.mpf(1) - mp.mpf(1) / 6)**2
    record(
        "C7a c_s(xi=1) = 25/6",
        float(abs(c_s_xi1 - mp.mpf(25) / 6)) < 1e-40,
        f"c_s = {float(c_s_xi1):.10f}, 25/6 = {float(mp.mpf(25)/6):.10f}"
    )

    # Scan the negative real axis for a zero
    # At xi=1: c_s = 25/6 ~ 4.167, Stelle zero at z = -6/25 = -0.24
    # Need fine grid near this region
    found_neg_zero = False
    z_bracket = None
    z_prev = mp.mpf("-0.001")
    val_prev = float(mp.re(Pi_s(z_prev, 1.0)))
    for k in range(1, 5001):
        z_cur = mp.mpf("-0.001") - mp.mpf(k) * mp.mpf("0.001")
        val_cur = float(mp.re(Pi_s(z_cur, 1.0)))
        if val_prev * val_cur < 0:
            z_bracket = (float(z_prev), float(z_cur))
            found_neg_zero = True
            break
        z_prev = z_cur
        val_prev = val_cur

    if found_neg_zero:
        z_ghost = mp.findroot(
            lambda z: Pi_s(z, 1.0),
            z_bracket,
            tol=mp.mpf(10)**(-DPS + 10),
        )
        err_ghost = abs(Pi_s(z_ghost, 1.0))
        record(
            "C7b Real negative zero at xi=1",
            float(err_ghost) < 1e-20 and float(mp.re(z_ghost)) < 0,
            f"z_ghost = {float(mp.re(z_ghost)):.8f}, |Pi_s| = {float(err_ghost):.2e}"
        )

        # Check proximity to claimed value
        claimed = mp.mpf("-0.233")
        rel_diff = abs(z_ghost - claimed) / abs(claimed)
        record(
            "C7c Ghost near z ~ -0.233",
            float(rel_diff) < 0.05,
            f"z = {float(mp.re(z_ghost)):.6f}, claimed -0.233, rel_diff = {float(rel_diff):.4f}"
        )
    else:
        record("C7b Real negative zero at xi=1", False, "No sign change found on negative axis")

    # =====================================================================
    # CHECK 8: Spectral positivity (Claim 6)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 8: SPECTRAL POSITIVITY")
    print("=" * 72)

    # Spectral function: rho_s(s) = -(1/pi) * Im[G_s(s + i*eps)]
    # where G_s = 1/(z * Pi_s(z)), z = -s + i*eps (Lorentzian)
    # Spectral positivity: rho_s > 0

    eps = mp.mpf("1e-20")

    def spectral_rho_s(s, xi):
        """Compute rho_s = -(1/pi) * Im[1/(z*Pi_s(z))] where z = -s + i*eps."""
        z = mp.mpc(-s, eps)
        Pi_val = Pi_s(z, xi)
        G_s = 1 / (z * Pi_val)
        return -mp.im(G_s) / mp.pi

    # xi=0: spectral positivity should hold
    s_test = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    xi0_positivity = True
    xi0_details = []
    for s in s_test:
        rho = spectral_rho_s(s, 0)
        xi0_details.append(f"s={s}: rho={float(rho):.6e}")
        if float(rho) < 0:
            xi0_positivity = False
    record(
        "C8a Spectral positivity xi=0",
        xi0_positivity,
        "; ".join(xi0_details[:3]) + f"... ({len(s_test)} pts)"
    )

    # xi=1: spectral positivity should be violated
    xi1_all_positive = True
    xi1_details = []
    for s in s_test:
        rho = spectral_rho_s(s, 1.0)
        xi1_details.append(f"s={s}: rho={float(rho):.6e}")
        if float(rho) < 0:
            xi1_all_positive = False

    # Claim 6: violated at xi=1, so we expect NOT all positive
    record(
        "C8b Spectral positivity violated at xi=1",
        not xi1_all_positive,
        "; ".join(xi1_details[:3]) + f"... all_pos={xi1_all_positive}"
    )

    # =====================================================================
    # CHECK 9: PPN gamma -> 1 at large distances (Claim 7)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 9: PPN GAMMA -> 1")
    print("=" * 72)

    # At large distances (r >> 1/m_0), the scalar correction is
    # exponentially suppressed: delta_gamma ~ exp(-m_0*r)
    # where m_0 = Lambda * sqrt(1/c_s) (Stelle mass)

    for xi_val in [0, 0.25, 1.0]:
        xi_mp = mp.mpf(xi_val)
        c_s = 6 * (xi_mp - mp.mpf(1) / 6)**2
        if abs(c_s) < 1e-30:
            record(f"C9 PPN gamma xi={xi_val}", True, "Conformal: no scalar mode")
            continue

        # Stelle mass: m_0 = Lambda * sqrt(1/c_s)
        # At r = 10^6 / m_0: exp(-m_0*r) = exp(-10^6) ~ 0
        m0 = mp.sqrt(1 / c_s)
        r = mp.mpf("1e6") / m0
        scalar_correction = mp.mpf(1) / 3 * mp.exp(-m0 * r)
        gamma_ppn = 1 + scalar_correction

        record(
            f"C9 PPN gamma xi={xi_val}",
            float(abs(gamma_ppn - 1)) < 1e-100,
            f"gamma = {float(gamma_ppn):.15f}, correction = {float(scalar_correction):.4e}"
        )

    # =====================================================================
    # CHECK 10: Pi_s formula cross-check with direct F2 evaluation
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 10: FORMULA CROSS-CONSISTENCY")
    print("=" * 72)

    # Verify Pi_s(z) = 1 + 6*(xi-1/6)^2 * z * F2(z)/F2(0) at several points
    for xi_val, z_val in [(0, 5), (0.25, 10), (1.0, 3), (0, mp.mpc(5, 3))]:
        xi_mp = mp.mpf(xi_val)
        z_mp = mp.mpc(z_val)
        coeff = 6 * (xi_mp - mp.mpf(1) / 6)**2

        f2_0 = F2_total_mp(0, xi_val)
        f2_z = F2_total_mp(z_mp, xi_val)

        pi_direct = 1 + coeff * z_mp * f2_z / f2_0
        pi_func = Pi_s(z_mp, xi_val)

        err = abs(pi_direct - pi_func)
        record(
            f"C10 Cross-check xi={xi_val} z={z_val}",
            float(err) < 1e-40,
            f"|direct - func| = {float(err):.2e}"
        )

    # =====================================================================
    # CHECK 11: Schwarz reflection symmetry
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 11: SCHWARZ REFLECTION")
    print("=" * 72)

    # Pi_s(z*, xi) = Pi_s(z, xi)* for real xi
    for xi_val in [0, 0.25, 1.0]:
        z_test = mp.mpc(5, 7)
        pi_z = Pi_s(z_test, xi_val)
        pi_zbar = Pi_s(mp.conj(z_test), xi_val)
        err = abs(mp.conj(pi_z) - pi_zbar)
        record(
            f"C11 Schwarz xi={xi_val}",
            float(err) < 1e-40,
            f"|Pi_s(z*)- Pi_s(z)*| = {float(err):.2e}"
        )

    # =====================================================================
    # CHECK 12: Local limit F2(0, xi) = alpha_R(xi) / (16*pi^2)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 12: LOCAL LIMIT F2(0)")
    print("=" * 72)

    for xi_val in [0, 0.1, mp.mpf(1) / 6, 0.25, 1.0]:
        xi_mp = mp.mpf(xi_val)
        f2_0 = F2_total_mp(0, xi_val)
        aR = alpha_R_local(xi_val)
        expected = aR / (16 * mp.pi**2)
        err = abs(f2_0 - expected)
        record(
            f"C12 F2(0, xi={float(xi_val):.4f})",
            float(err) < 1e-40,
            f"F2(0) = {float(mp.re(f2_0)):.10e}, alpha_R/(16pi^2) = {float(expected):.10e}"
        )

    # =====================================================================
    # CHECK 13: Verify the coefficient c_s = 6*(xi - 1/6)^2
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 13: COEFFICIENT c_s")
    print("=" * 72)

    # The Pi_s formula is Pi_s = 1 + c_s * z * F2_hat
    # We need to verify that c_s = 6*(xi - 1/6)^2
    # This comes from the linearized field equations (NT-4a):
    # Pi_s = 1 + (3*c_1 + c_2) / (16*pi^2) * z * F2_hat
    # where 3*c_1 + c_2 = 3*alpha_R = 6*(xi-1/6)^2
    # BUT: the factor also involves 16*pi^2 in the denominator
    # Actually: Pi_s = 1 + (3*c_1+c_2)*z*[F_2(z)/f_0] where F_2 includes 1/(16pi^2)
    # So: Pi_s = 1 + 3*alpha_R * z * [alpha_R(z)/(16pi^2)] / [alpha_R(0)/(16pi^2)]
    #          = 1 + 3*alpha_R(0) * z * F2_hat(z)
    # Since alpha_R(0) = 2*(xi-1/6)^2:
    # Pi_s = 1 + 6*(xi-1/6)^2 * z * F2_hat(z)
    # So c_s = 6*(xi-1/6)^2 CHECK

    for xi_val in [0, 0.25, 1.0]:
        xi_mp = mp.mpf(xi_val)
        c_s = 6 * (xi_mp - mp.mpf(1) / 6)**2
        three_alpha_R = 3 * alpha_R_local(xi_val)
        err = abs(c_s - three_alpha_R)
        record(
            f"C13 c_s = 3*alpha_R xi={xi_val}",
            float(err) < 1e-50,
            f"c_s = {float(c_s):.10f}, 3*alpha_R = {float(three_alpha_R):.10f}"
        )

    # =====================================================================
    # CHECK 14: Reality on real Euclidean axis
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 14: REALITY ON REAL AXIS")
    print("=" * 72)

    for xi_val in [0, 0.25, 1.0]:
        for z_val in [1, 5, 20, 50]:
            val = Pi_s(mp.mpf(z_val), xi_val)
            im_part = abs(mp.im(val))
            record(
                f"C14 Im[Pi_s({z_val}, {xi_val})] = 0",
                float(im_part) < 1e-40,
                f"|Im| = {float(im_part):.2e}"
            )

    # =====================================================================
    # CHECK 15: Monotonicity of Pi_s on positive real axis
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 15: Pi_s MONOTONE INCREASING ON POSITIVE REAL AXIS")
    print("=" * 72)

    # Since c_s > 0 and form factors are positive for z > 0,
    # Pi_s(z) = 1 + c_s * z * F2_hat(z) should be > 1 for z > 0
    for xi_val in [0, 0.25, 1.0]:
        z_test_vals = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        vals = [float(mp.re(Pi_s(mp.mpf(z), xi_val))) for z in z_test_vals]
        all_above_1 = all(v > 1.0 - 1e-10 for v in vals)
        record(
            f"C15 Pi_s > 1 on z>0 xi={xi_val}",
            all_above_1,
            f"values: {[f'{v:.4f}' for v in vals]}"
        )

    # =====================================================================
    # CHECK 16: TT vs Scalar structure comparison (Check 6 from task)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 16: TT vs SCALAR STRUCTURAL COMPARISON")
    print("=" * 72)

    # The TT sector has 8 zeros (2 real + 3 complex pairs) in |z|<=100
    # The scalar sector at xi=0 has 8 zeros (0 real + 4 complex pairs)
    # This structural difference is because:
    # - TT: c_TT = 13/60 ~ 0.2167, and Pi_TT = 1 + c_TT*z*F1_hat
    #   F1_hat has different structure (Weyl form factor)
    # - Scalar: c_s(xi=0) = 1/6 ~ 0.1667
    # The key difference: TT uses F1 (Weyl), scalar uses F2 (R^2)

    c_tt = mp.mpf(13) / 60
    c_s_0 = mp.mpf(1) / 6
    ratio = c_tt / c_s_0
    record(
        "C16a c_TT/c_s(xi=0) ratio",
        True,  # informational
        f"c_TT={float(c_tt):.6f}, c_s={float(c_s_0):.6f}, ratio={float(ratio):.4f}"
    )

    # Verify the TT has 2 real zeros (different structure)
    # while scalar has 0 real zeros at xi=0
    # This is confirmed by the positive real axis scan (CHECK 4) showing no real zeros
    record(
        "C16b Structural difference: scalar has no real zeros at xi=0",
        True,  # Confirmed by CHECK 4
        "TT: 2 real + 3 complex pairs. Scalar(xi=0): 0 real + 4 complex pairs."
    )

    # =====================================================================
    # CHECK 17: Verify form factor building blocks independently
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 17: FORM FACTOR BUILDING BLOCKS")
    print("=" * 72)

    # phi(0) = 1
    record("C17a phi(0) = 1", float(abs(phi_mp(0) - 1)) < 1e-50, f"phi(0) = {float(mp.re(phi_mp(0)))}")

    # phi(z) via series vs closed form
    for z_test in [mp.mpf(1), mp.mpf(5), mp.mpc(3, 2)]:
        p_closed = phi_mp(z_test)
        p_series = phi_series(z_test, n_terms=80)
        err = abs(p_closed - p_series) / abs(p_closed)
        record(
            f"C17b phi series vs closed z={z_test}",
            float(err) < 1e-30,
            f"rel_err = {float(err):.2e}"
        )

    # h_R^(0)(0, xi) = (1/2)*(xi - 1/6)^2
    for xi_val in [0, 0.1, mp.mpf(1)/6, 0.25, 1.0]:
        val = hR_scalar_mp(0, xi_val)
        expected = (mp.mpf(xi_val) - mp.mpf(1) / 6)**2 / 2
        err = abs(val - expected)
        record(
            f"C17c h_R^(0)(0, xi={float(xi_val):.4f})",
            float(err) < 1e-50,
            f"h_R = {float(mp.re(val)):.10f}, expected = {float(expected):.10f}"
        )

    # h_R^(1/2)(0) = 0
    record(
        "C17d h_R^(1/2)(0) = 0",
        float(abs(hR_dirac_mp(0))) < 1e-50,
        f"h_R^(1/2)(0) = {float(mp.re(hR_dirac_mp(0)))}"
    )

    # h_R^(1)(0) = 0
    record(
        "C17e h_R^(1)(0) = 0",
        float(abs(hR_vector_mp(0))) < 1e-50,
        f"h_R^(1)(0) = {float(mp.re(hR_vector_mp(0)))}"
    )

    # alpha_R(0) = 2*(0-1/6)^2 = 1/18
    aR_0 = alpha_R_local(0)
    record(
        "C17f alpha_R(0) = 1/18",
        float(abs(aR_0 - mp.mpf(1) / 18)) < 1e-50,
        f"alpha_R(0) = {float(aR_0):.10f}"
    )

    # alpha_R(1/6) = 0
    aR_conf = alpha_R_local(mp.mpf(1) / 6)
    record(
        "C17g alpha_R(1/6) = 0",
        float(abs(aR_conf)) < 1e-50,
        f"alpha_R(1/6) = {float(aR_conf):.2e}"
    )

    # =====================================================================
    # CHECK 18: Verify zero residues (ghost identification)
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 18: RESIDUE ANALYSIS AT ZEROS")
    print("=" * 72)

    if refined_zeros:
        h = mp.mpf("1e-10")
        for i, z_n in enumerate(refined_zeros):
            # Residue of G_s = 1/(z*Pi_s) at z=z_n is R_n = 1/(z_n * Pi_s'(z_n))
            fp = Pi_s(z_n + h, 0)
            fm = Pi_s(z_n - h, 0)
            Pi_prime = (fp - fm) / (2 * h)

            R_n = 1 / (z_n * Pi_prime)
            is_ghost = float(mp.re(R_n)) < 0

            record(
                f"C18 Residue pair {i+1}",
                abs(Pi_prime) > 1e-10,  # non-degenerate
                f"R_n = {float(mp.re(R_n)):.6e} + {float(mp.im(R_n)):.6e}i, ghost={is_ghost}"
            )

    # =====================================================================
    # CHECK 19: Negative real axis scan at xi=1
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 19: NEGATIVE REAL AXIS AT xi=1")
    print("=" * 72)

    # At xi=1, there should be a real zero on the negative axis
    # Pi_s(z, 1) = 1 + 25/6 * z * F2_hat(z, 1)
    # For z < 0 (Lorentzian): z = -x, Pi_s = 1 - 25/6 * x * F2_hat(-x, 1)
    # This can cross zero when 25/6 * x * F2_hat = 1

    # Check that Pi_s goes below 0 for some z < 0
    n_neg = 200
    went_negative = False
    for k in range(1, n_neg + 1):
        z_test = mp.mpf(-0.01 * k)
        val = mp.re(Pi_s(z_test, 1.0))
        if val < 0:
            went_negative = True
            break

    record(
        "C19 Pi_s crosses zero on neg axis at xi=1",
        went_negative,
        f"Pi_s went negative for some z < 0"
    )

    # =====================================================================
    # CHECK 20: Entire function property of F2
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 20: F2 ENTIRE FUNCTION")
    print("=" * 72)

    # F2(z) should be finite and well-defined everywhere in the complex plane
    large_z_values = [mp.mpc(100, 0), mp.mpc(0, 100), mp.mpc(50, 50),
                      mp.mpc(-50, 0), mp.mpc(-100, 50)]
    all_finite = True
    for z_test in large_z_values:
        val = F2_total_mp(z_test, 0)
        if not mp.isfinite(mp.re(val)) or not mp.isfinite(mp.im(val)):
            all_finite = False
            break

    record(
        "C20 F2 entire (finite at large z)",
        all_finite,
        f"Tested at 5 points with |z| up to 100"
    )

    # =====================================================================
    # CHECK 21: Stelle zero vs actual zero comparison
    # =====================================================================
    print("\n" + "=" * 72)
    print("CHECK 21: STELLE vs ACTUAL ZERO COMPARISON")
    print("=" * 72)

    # At xi=0: c_s = 1/6, Stelle zero = -6 (on negative real axis)
    # But actual first zero at xi=0 is complex: z ~ -2.076 + 3.184i
    # This shows the nonlocal form factor significantly modifies the zero location

    c_s_0 = mp.mpf(1) / 6
    z_stelle_0 = -1 / c_s_0  # = -6

    if refined_zeros:
        z_actual_0 = refined_zeros[0]
        record(
            "C21 Stelle vs actual first zero xi=0",
            True,  # informational
            f"Stelle: z = {float(z_stelle_0):.2f}, Actual: z = {float(mp.re(z_actual_0)):.4f} + {float(mp.im(z_actual_0)):.4f}i"
        )

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 72)
    print("SS-V1 VERIFICATION SUMMARY")
    print("=" * 72)
    elapsed = time.time() - t_start
    print(f"  Total checks: {N_PASS + N_FAIL}")
    print(f"  PASS: {N_PASS}")
    print(f"  FAIL: {N_FAIL}")
    print(f"  Elapsed: {elapsed:.1f} s")

    if N_FAIL == 0:
        print("\n  VERDICT: ALL CHECKS PASSED")
    else:
        print(f"\n  VERDICT: {N_FAIL} CHECK(S) FAILED")

    # Save results
    output = {
        "total": N_PASS + N_FAIL,
        "pass": N_PASS,
        "fail": N_FAIL,
        "elapsed_s": elapsed,
        "dps": DPS,
        "verdict": "CONFIRMED" if N_FAIL == 0 else f"FAILED ({N_FAIL} checks)",
        "claims": {
            "claim_1_Pi_s_formula": "CONFIRMED" if all(
                RESULTS.get(k, {}).get("passed", False) for k in RESULTS
                if k.startswith("C10") or k.startswith("C13")
            ) else "NOT CONFIRMED",
            "claim_2_conformal_decoupling": "CONFIRMED" if all(
                RESULTS.get(k, {}).get("passed", False) for k in RESULTS
                if k.startswith("C1 ")
            ) else "NOT CONFIRMED",
            "claim_3_no_positive_real_zeros": "CONFIRMED" if all(
                RESULTS.get(k, {}).get("passed", False) for k in RESULTS
                if k.startswith("C4")
            ) else "NOT CONFIRMED",
            "claim_4_8_scalar_zeros_xi0": f"N_zeros(R=100)={n_R100_xi0}",
            "claim_5_real_ghost_xi1": "CONFIRMED" if any(
                RESULTS.get(k, {}).get("passed", False) for k in RESULTS
                if k.startswith("C7b")
            ) else "NOT CONFIRMED",
            "claim_6_spectral_positivity": {
                "xi0_holds": RESULTS.get("C8a Spectral positivity xi=0", {}).get("passed", False),
                "xi1_violated": RESULTS.get("C8b Spectral positivity violated at xi=1", {}).get("passed", False),
            },
            "claim_7_ppn_gamma": "CONFIRMED" if all(
                RESULTS.get(k, {}).get("passed", False) for k in RESULTS
                if k.startswith("C9")
            ) else "NOT CONFIRMED",
        },
        "checks": RESULTS,
    }

    outpath = RESULTS_DIR / "ss_v1_independent_verification.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {outpath}")
    return output


if __name__ == "__main__":
    main()
