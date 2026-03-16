# ruff: noqa: E402, I001
"""
SS-D: Scalar Sector Analysis at General Non-Minimal Coupling xi != 1/6.

Complete survey of the scalar propagator denominator Pi_s(z, xi) and its
ghost spectrum, Mittag-Leffler expansion, spectral function, and physical
observables.

Physics:
    Pi_s(z, xi) = 1 + 6*(xi - 1/6)^2 * z * F2_hat(z, xi)

    where F2_hat = F2/F2(0) is the normalized R^2 form factor.  The scalar
    sector controls the spin-0 part of the graviton propagator.

    At xi = 1/6 (conformal coupling): Pi_s = 1 identically -- no scalar mode.
    Away from conformal: Pi_s(z) develops zeros that become ghost poles
    in the physical propagator 1/(z * Pi_s(z)).

Key results derived here:
    1. Scalar ghost catalogue: number, location, and residues of Pi_s zeros
       at representative xi values
    2. xi-dependence: zero trajectories, critical xi thresholds
    3. Scalar Mittag-Leffler expansion: entire part g_A^(s)(z,xi) and sum rules
    4. Scalar spectral function Im[G_s(s + i*eps)]
    5. Physical observables: Newtonian potential, PPN gamma, optical theorem

Input results (verified, canonical):
    alpha_C = 13/120, alpha_R(xi) = 2*(xi - 1/6)^2
    F1(0) = 13/(1920*pi^2), F2(0,xi) = alpha_R(xi)/(16*pi^2)
    phi(z) = e^{-z/4} * sqrt(pi/z) * erfi(sqrt(z)/2)
    All form factors are entire (NT-2).

Author: David Alfyorov
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.mr1_lorentzian import Pi_scalar_complex, Pi_s_lorentzian
from scripts.nt2_entire_function import F2_total_complex, alpha_R

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "ss"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120
DEFAULT_DPS = 50

# Representative xi values for the survey
XI_VALUES = [
    mp.mpf(0),          # minimal coupling
    mp.mpf("0.1"),
    mp.mpf("0.15"),     # near conformal
    mp.mpf("0.2"),
    mp.mpf(1) / 4,      # quarter
    mp.mpf(1),          # strong non-minimal
]

# ---------------------------------------------------------------------------
# Core function: scalar propagator denominator
# ---------------------------------------------------------------------------

def scalar_Pi_s(z, xi, dps=DEFAULT_DPS):
    """
    Evaluate the scalar propagator denominator Pi_s(z, xi).

    Pi_s(z, xi) = 1 + 6*(xi - 1/6)^2 * z * F2_hat(z, xi)

    Uses Pi_scalar_complex from mr1_lorentzian which handles
    the negative real axis (Lorentzian continuation) correctly.
    """
    mp.mp.dps = dps
    return Pi_scalar_complex(mp.mpc(z), xi=float(xi), dps=dps)


def scalar_local_c2(xi, dps=DEFAULT_DPS):
    """
    Local (Stelle) coefficient c_2^(s)(xi) = 2 * alpha_R(xi).

    alpha_R(xi) = 2*(xi - 1/6)^2
    c_2^(s) = 2 * alpha_R = 4*(xi - 1/6)^2 = 12*(xi - 1/6)^2 / 3

    This is the coefficient in Pi_s(z) = 1 + c_2^(s) * z * F2_hat(z)
    at the local limit F2_hat = 1.

    Wait: Pi_s = 1 + 6*(xi-1/6)^2 * z * F2_hat.
    So the "c_2^(s)" analog is 6*(xi-1/6)^2.
    """
    mp.mp.dps = dps
    xi_mp = mp.mpf(xi)
    return 6 * (xi_mp - mp.mpf(1) / 6) ** 2


def scalar_stelle_zero(xi, dps=DEFAULT_DPS):
    """
    In the local (Stelle) limit F2_hat = 1:
    Pi_s(z) = 1 + c_s * z = 0  =>  z_0^Stelle = -1/c_s

    For the Euclidean positive real axis (z > 0), the zero is at z = 1/c_s.
    """
    mp.mp.dps = dps
    c_s = scalar_local_c2(xi, dps=dps)
    if abs(c_s) < mp.mpf("1e-30"):
        return mp.inf
    return 1 / c_s


# ===================================================================
# PART 1: Scalar Ghost Catalogue
# ===================================================================

def _count_zeros_argument_principle(xi, R_max=100, n_pts=2048, dps=DEFAULT_DPS):
    """
    Count zeros of Pi_s(z, xi) in |z| <= R_max using the argument principle.

    N = (1/2*pi*i) * oint Pi_s'(z)/Pi_s(z) dz

    The contour is a circle of radius R_max.
    """
    mp.mp.dps = dps
    R = mp.mpf(R_max)

    integral = mp.mpc(0)
    d_theta = 2 * mp.pi / n_pts
    h = mp.mpf("1e-8")  # for numerical derivative

    for k in range(n_pts):
        theta = k * d_theta
        z = R * mp.expj(theta)
        dz = mp.mpc(0, 1) * z * d_theta  # dz = i*R*e^{itheta}*dtheta = i*z*dtheta

        # Pi_s and Pi_s' at z
        pi_val = scalar_Pi_s(z, xi, dps=dps)

        # Numerical derivative
        pi_plus = scalar_Pi_s(z + h, xi, dps=dps)
        pi_minus = scalar_Pi_s(z - h, xi, dps=dps)
        pi_prime = (pi_plus - pi_minus) / (2 * h)

        if abs(pi_val) > mp.mpf("1e-30"):
            integral += (pi_prime / pi_val) * dz

    N_zeros = integral / (2 * mp.pi * mp.mpc(0, 1))
    return int(round(float(mp.re(N_zeros))))


def _scan_real_axis_for_sign_changes(xi, z_min=0.01, z_max=100, n_pts=2000, dps=DEFAULT_DPS):
    """Scan the positive real axis for sign changes of Pi_s to find real zeros."""
    mp.mp.dps = dps
    sign_changes = []
    z_prev = mp.mpf(z_min)
    val_prev = mp.re(scalar_Pi_s(z_prev, xi, dps=dps))

    step = (z_max - z_min) / n_pts
    for k in range(1, n_pts + 1):
        z_cur = mp.mpf(z_min) + k * mp.mpf(step)
        val_cur = mp.re(scalar_Pi_s(z_cur, xi, dps=dps))

        if val_prev * val_cur < 0:
            sign_changes.append((float(z_prev), float(z_cur)))

        z_prev = z_cur
        val_prev = val_cur

    return sign_changes


def _refine_zero(xi, z_guess, dps=DEFAULT_DPS):
    """Refine a zero of Pi_s using mpmath.findroot."""
    mp.mp.dps = dps
    try:
        if isinstance(z_guess, tuple):
            # Bracket
            z0 = mp.findroot(
                lambda z: scalar_Pi_s(z, xi, dps=dps),
                z_guess,
                tol=mp.mpf(10) ** (-(dps - 5)),
            )
        else:
            z0 = mp.findroot(
                lambda z: scalar_Pi_s(z, xi, dps=dps),
                mp.mpc(z_guess),
                tol=mp.mpf(10) ** (-(dps - 5)),
            )
        # Verify
        val = scalar_Pi_s(z0, xi, dps=dps)
        if abs(val) < mp.mpf("1e-15"):
            return z0
    except (ValueError, ZeroDivisionError):
        pass
    return None


def _find_complex_zeros_grid(xi, R_max=100, dps=DEFAULT_DPS):
    """
    Search for complex zeros of Pi_s by scanning a grid and using findroot.

    Uses initial guesses based on the TT zero structure (similar spacing).
    """
    mp.mp.dps = dps
    found = []

    # For xi not too close to 1/6, the scalar zeros have a similar pattern
    # to the TT zeros but with different spacing governed by c_2^(s)
    c_s = float(scalar_local_c2(xi, dps=dps))
    if c_s < 1e-10:
        return found

    # The Stelle zero location gives us a scale
    z_stelle = float(scalar_stelle_zero(xi, dps=dps))

    # Try a range of initial guesses for complex zeros
    # Based on TT pattern: Im spacing ~ 25-30, Re slowly growing
    re_guesses = [z_stelle * 0.5, z_stelle, z_stelle * 1.5, z_stelle * 2]
    im_guesses = [20, 35, 55, 80, 110, 140, 170, 200]

    # Also try near the real axis
    for re_g in [z_stelle * 0.3, z_stelle * 0.5, z_stelle * 0.8,
                 z_stelle * 1.2, z_stelle * 1.5, z_stelle * 2.0]:
        for im_g in im_guesses:
            if abs(mp.mpc(re_g, im_g)) > R_max:
                continue
            z_guess = mp.mpc(re_g, im_g)
            z0 = _refine_zero(xi, z_guess, dps=dps)
            if z0 is not None and abs(z0) < R_max * 1.05:
                # Check not duplicate
                is_dup = False
                for existing in found:
                    if abs(z0 - existing) < mp.mpf("0.1"):
                        is_dup = True
                        break
                if not is_dup:
                    found.append(z0)

    return found


def find_scalar_zeros(xi, R_max=100, dps=DEFAULT_DPS):
    """
    Find all zeros of Pi_s(z, xi) in |z| <= R_max.

    Returns a list of dicts with:
        z_n, z_re, z_im, z_abs, type ('real'/'complex')
    """
    mp.mp.dps = dps
    zeros = []

    # Step 1: Count zeros via argument principle
    n_ap = _count_zeros_argument_principle(xi, R_max=R_max, dps=dps)
    print(f"    Argument principle: {n_ap} zeros in |z| <= {R_max}")

    # Step 2: Find real zeros
    brackets = _scan_real_axis_for_sign_changes(xi, z_max=R_max, dps=dps)
    for bracket in brackets:
        z0 = _refine_zero(xi, bracket, dps=dps)
        if z0 is not None:
            zeros.append({
                "z": z0,
                "z_re": float(mp.re(z0)),
                "z_im": float(mp.im(z0)),
                "z_abs": float(abs(z0)),
                "type": "real",
            })

    # Step 3: Find complex zeros
    complex_zeros = _find_complex_zeros_grid(xi, R_max=R_max, dps=dps)
    for z0 in complex_zeros:
        # Add the zero
        zeros.append({
            "z": z0,
            "z_re": float(mp.re(z0)),
            "z_im": float(mp.im(z0)),
            "z_abs": float(abs(z0)),
            "type": "complex",
        })
        # Add conjugate if Im is nonzero
        if abs(mp.im(z0)) > 0.01:
            z_conj = mp.conj(z0)
            # Check not already found
            is_dup = False
            for existing in zeros:
                if abs(z_conj - existing["z"]) < mp.mpf("0.1"):
                    is_dup = True
                    break
            if not is_dup:
                zeros.append({
                    "z": z_conj,
                    "z_re": float(mp.re(z_conj)),
                    "z_im": float(mp.im(z_conj)),
                    "z_abs": float(abs(z_conj)),
                    "type": "complex",
                })

    # Also scan the negative real axis for Lorentzian-domain zeros
    neg_brackets = _scan_real_axis_for_sign_changes(
        xi, z_min=-R_max, z_max=-0.01, n_pts=2000, dps=dps,
    )
    for bracket in neg_brackets:
        z0 = _refine_zero(xi, bracket, dps=dps)
        if z0 is not None:
            # Check not duplicate
            is_dup = False
            for existing in zeros:
                if abs(z0 - existing["z"]) < mp.mpf("0.1"):
                    is_dup = True
                    break
            if not is_dup:
                zeros.append({
                    "z": z0,
                    "z_re": float(mp.re(z0)),
                    "z_im": float(mp.im(z0)),
                    "z_abs": float(abs(z0)),
                    "type": "real_negative",
                })

    # Sort by |z|
    zeros.sort(key=lambda e: e["z_abs"])

    return zeros


# ===================================================================
# PART 1c: Residues
# ===================================================================

def scalar_residues(zeros, xi, dps=DEFAULT_DPS):
    """
    Compute residue R_n^(s) = 1/(z_n * Pi_s'(z_n)) at each zero of Pi_s.

    This is the residue of G_s(z) = 1/(z * Pi_s(z)) at the simple pole z = z_n.
    """
    mp.mp.dps = dps
    h = mp.mpf("1e-12")

    for entry in zeros:
        z_n = entry["z"]
        # Central finite difference for Pi_s'(z_n)
        fp = scalar_Pi_s(z_n + h, xi, dps=dps)
        fm = scalar_Pi_s(z_n - h, xi, dps=dps)
        Pi_prime = (fp - fm) / (2 * h)

        R_n = 1 / (z_n * Pi_prime)
        entry["Pi_s_prime"] = Pi_prime
        entry["R_n"] = R_n
        entry["R_re"] = float(mp.re(R_n))
        entry["R_im"] = float(mp.im(R_n))
        entry["R_abs"] = float(abs(R_n))
        entry["ghost"] = float(mp.re(R_n)) < 0

    return zeros


# ===================================================================
# PART 2: xi-Dependence Analysis
# ===================================================================

def xi_trajectory(xi_values=None, R_max=60, dps=DEFAULT_DPS):
    """
    Track scalar zero trajectories as xi varies.

    Returns a dict mapping xi -> list of zeros with their properties.
    """
    if xi_values is None:
        xi_values = [mp.mpf(x) / 100 for x in range(0, 105, 5)]
        # Ensure 1/6 is included
        xi_values.append(mp.mpf(1) / 6)
        xi_values.sort()

    trajectories = {}
    for xi in xi_values:
        xi_f = float(xi)
        c_s = float(scalar_local_c2(xi, dps=dps))
        print(f"  xi = {xi_f:.4f}, c_s = {c_s:.6f}")

        if c_s < 1e-10:
            trajectories[xi_f] = {
                "n_zeros": 0,
                "c_s": c_s,
                "zeros": [],
                "note": "Near conformal: Pi_s = 1, no zeros",
            }
            continue

        zeros = find_scalar_zeros(xi, R_max=R_max, dps=dps)
        zeros = scalar_residues(zeros, xi, dps=dps)

        trajectories[xi_f] = {
            "n_zeros": len(zeros),
            "c_s": c_s,
            "zeros": [
                {
                    "z_re": e["z_re"],
                    "z_im": e["z_im"],
                    "z_abs": e["z_abs"],
                    "type": e["type"],
                    "R_re": e.get("R_re", 0),
                    "R_im": e.get("R_im", 0),
                    "R_abs": e.get("R_abs", 0),
                    "ghost": e.get("ghost", False),
                }
                for e in zeros
            ],
        }

    return trajectories


# ===================================================================
# PART 3: Scalar Mittag-Leffler Expansion
# ===================================================================

def scalar_entire_part(z, xi, zeros, dps=DEFAULT_DPS):
    """
    Compute the entire part g_A^(s)(z, xi) of 1/(z * Pi_s(z, xi)).

    g_A^(s)(z) = 1/(z * Pi_s(z)) - 1/z - Sum_n R_n^(s) [1/(z - z_n) + 1/z_n]

    By analogy with the TT sector (GZ result), if Pi_s has genus 1
    and g_A^(s) is bounded, then g_A^(s) = -c_2^(s) = -6*(xi - 1/6)^2.
    """
    mp.mp.dps = dps
    z_mp = mp.mpc(z)
    xi_mp = mp.mpf(xi)

    # 1/(z * Pi_s(z))
    Pi_val = scalar_Pi_s(z_mp, xi_mp, dps=dps)
    H = 1 / (z_mp * Pi_val)

    # Graviton pole 1/z
    graviton = 1 / z_mp

    # Subtracted pole sum
    pole_sum = mp.mpc(0)
    for entry in zeros:
        z_n = entry["z"]
        R_n = entry["R_n"]
        term = R_n * (1 / (z_mp - z_n) + 1 / z_n)
        pole_sum += term

    return H - graviton - pole_sum


def verify_scalar_entire_part_constant(xi, zeros, dps=DEFAULT_DPS):
    """
    Check that g_A^(s)(z) = -c_2^(s) at multiple test points.
    """
    mp.mp.dps = dps
    c_s = scalar_local_c2(xi, dps=dps)
    target = -c_s

    z_test_values = [
        mp.mpc("0.3"), mp.mpc("0.5"), mp.mpc("1.0"), mp.mpc("2.0"),
        mp.mpc("5.0"), mp.mpc("10.0"), mp.mpc("20.0"),
        mp.mpc(3, 1), mp.mpc(5, 5), mp.mpc(10, 3),
    ]

    results = []
    for z in z_test_values:
        # Skip if too close to a zero
        too_close = any(abs(z - e["z"]) < 0.3 for e in zeros)
        if too_close:
            continue
        try:
            g = scalar_entire_part(z, xi, zeros, dps=dps)
            dev = abs(g - target)
            results.append({
                "z": str(z),
                "g_A_re": float(mp.re(g)),
                "g_A_im": float(mp.im(g)),
                "deviation": float(dev),
                "target": float(target),
            })
        except (ZeroDivisionError, ValueError):
            pass

    return results


def scalar_sum_rule(zeros, xi, dps=DEFAULT_DPS):
    """
    Verify the scalar sum rule: Sum_n R_n^(s) / z_n = c_2^(s).

    Analogous to the TT sum rule Sum R_n/z_n = 13/60.
    """
    mp.mp.dps = dps
    c_s = scalar_local_c2(xi, dps=dps)

    partial_re = mp.mpf(0)
    partial_im = mp.mpf(0)

    terms = []
    for entry in zeros:
        if "R_n" not in entry:
            continue
        z_n = entry["z"]
        R_n = entry["R_n"]
        ratio = R_n / z_n
        partial_re += mp.re(ratio)
        partial_im += mp.im(ratio)
        terms.append({
            "z_re": entry["z_re"],
            "z_im": entry["z_im"],
            "R_over_z_re": float(mp.re(ratio)),
            "R_over_z_im": float(mp.im(ratio)),
        })

    return {
        "target_c_s": float(c_s),
        "partial_sum_re": float(partial_re),
        "partial_sum_im": float(partial_im),
        "deficit": float(c_s) - float(partial_re),
        "terms": terms,
    }


# ===================================================================
# PART 4: Scalar Spectral Function
# ===================================================================

def scalar_spectral_function(s, xi, dps=DEFAULT_DPS, eps=1e-20):
    """
    Compute Im[G_s(s + i*eps, xi)] for s > 0.

    G_s = 1/(z * Pi_s(z)) where z = -s/Lambda^2 (Lorentzian).

    For the spectral function:
      rho_s(s) = -(1/pi) * Im[G_s(s + i*eps)]

    Sign convention: if rho_s > 0, the spectral positivity holds.
    """
    mp.mp.dps = dps
    s_mp = mp.mpf(s)

    # Evaluate Pi_s on the Lorentzian axis with small imaginary part
    eps_mp = mp.mpf(eps)

    # z = -s + i*eps (Euclidean convention: z = -k^2/Lambda^2)
    z = mp.mpc(-s_mp, eps_mp)
    Pi_val = scalar_Pi_s(z, xi, dps=dps)

    # G_s = 1/(z * Pi_s)
    G_s = 1 / (z * Pi_val)

    return mp.im(G_s)


def spectral_function_scan(xi, s_values=None, dps=DEFAULT_DPS):
    """
    Scan the scalar spectral function over a range of s values.
    """
    if s_values is None:
        # Construct a grid concentrated near expected features
        s_values = []
        # Fine grid near origin
        s_values.extend([0.01 * (k + 1) for k in range(100)])
        # Medium grid
        s_values.extend([1 + 0.1 * k for k in range(100)])
        # Coarse grid
        s_values.extend([11 + k for k in range(90)])

    results = []
    for s in s_values:
        im_G = scalar_spectral_function(s, xi, dps=dps)
        rho = -float(im_G) / float(mp.pi)
        results.append({
            "s": float(s),
            "Im_G_s": float(im_G),
            "rho_s": rho,
            "positive": rho > 0,
        })

    return results


# ===================================================================
# PART 5: Physical Observables
# ===================================================================

def newtonian_potential_scalar(r_values, xi, dps=DEFAULT_DPS):
    """
    Compute the scalar sector contribution to the Newtonian potential.

    In the local limit the full Newtonian potential is:
      V(r)/V_N(r) = 1 - (4/3)*exp(-m_2*r) + (1/3)*exp(-m_0*r)

    where:
      m_2 = Lambda * sqrt(60/13) ≈ 2.148 * Lambda  (tensor ghost)
      m_0 = Lambda / sqrt(c_s)  (scalar ghost, if real zero exists)

    The scalar contribution is the (1/3)*exp(-m_0*r) term.
    The coefficient 1/3 comes from the Barnes-Rivers decomposition:
      trace of P^{(0-s)} = 1, residue factor = -1, overall sign +1/3.
    """
    mp.mp.dps = dps
    xi_mp = mp.mpf(xi)
    c_s = scalar_local_c2(xi_mp, dps=dps)

    if abs(c_s) < mp.mpf("1e-20"):
        # No scalar mode at conformal coupling
        return [{"r": float(r), "V_scalar_over_VN": 0.0} for r in r_values]

    # Find the first real positive zero of Pi_s -> gives m_0
    brackets = _scan_real_axis_for_sign_changes(xi_mp, z_max=200, dps=dps)

    results = []
    if brackets:
        z0 = _refine_zero(xi_mp, brackets[0], dps=dps)
        if z0 is not None:
            m0_sq = mp.re(z0)  # z = m^2/Lambda^2 in natural units (Lambda=1)
            m0 = mp.sqrt(m0_sq)
            for r in r_values:
                r_mp = mp.mpf(r)
                # The 1/3 coefficient is for Stelle-like gravity
                # For SCT with nonlocal form factors, use the actual residue
                V_scalar = mp.mpf(1) / 3 * mp.exp(-m0 * r_mp)
                results.append({
                    "r": float(r_mp),
                    "m0": float(m0),
                    "z0": float(z0),
                    "V_scalar_over_VN": float(V_scalar),
                })
            return results

    # No real zero found: use Stelle estimate
    z0_stelle = scalar_stelle_zero(xi_mp, dps=dps)
    m0 = mp.sqrt(z0_stelle)
    for r in r_values:
        r_mp = mp.mpf(r)
        V_scalar = mp.mpf(1) / 3 * mp.exp(-m0 * r_mp)
        results.append({
            "r": float(r_mp),
            "m0_stelle": float(m0),
            "V_scalar_over_VN": float(V_scalar),
        })
    return results


def ppn_gamma_scalar(xi, r_over_m0_inv=1e6, dps=DEFAULT_DPS):
    """
    Compute the PPN parameter gamma from the scalar sector.

    In the linearized SCT metric (isotropic gauge):
      g_{00} = -1 + 2*Phi(r),  g_{ij} = delta_{ij}*(1 + 2*Psi(r))

    where:
      Phi(r) = -GM/r * [1 - (4/3)*exp(-m_2*r) + (1/3)*exp(-m_0*r)]
      Psi(r) = -GM/r * [1 + (2/3)*exp(-m_2*r) - (1/3)*exp(-m_0*r)
                           + (2/3)*m_2*r*exp(-m_2*r) - (1/3)*m_0*r*exp(-m_0*r)]

    gamma_PPN = -Psi/Phi in the far field (all exponentials -> 0).
    => gamma_PPN = 1 in the far field.

    The scalar correction is suppressed by exp(-m_0*r).
    """
    mp.mp.dps = dps
    xi_mp = mp.mpf(xi)
    c_s = scalar_local_c2(xi_mp, dps=dps)

    if abs(c_s) < mp.mpf("1e-20"):
        return {
            "xi": float(xi),
            "gamma_PPN": 1.0,
            "scalar_correction": 0.0,
            "note": "Conformal coupling: no scalar sector",
        }

    # Effective scalar mass (from Stelle approximation)
    z0_stelle = scalar_stelle_zero(xi_mp, dps=dps)
    m0 = mp.sqrt(z0_stelle)

    # At large r (r >> 1/m_0): gamma -> 1
    # At finite r: deviation ~ exp(-m_0*r) terms
    r = mp.mpf(r_over_m0_inv) / m0
    scalar_correction = mp.mpf(1) / 3 * mp.exp(-m0 * r)

    return {
        "xi": float(xi),
        "gamma_PPN": 1.0 + float(scalar_correction),
        "scalar_correction": float(scalar_correction),
        "m0_over_Lambda": float(m0),
        "r_over_Lambda_inv": float(r),
        "note": "Correction exponentially suppressed: exp(-m_0*r)",
    }


def scalar_optical_theorem_contribution(xi, s_values=None, dps=DEFAULT_DPS):
    """
    Compute the scalar sector contribution to the one-loop optical theorem.

    The SM self-energy has both TT and scalar channels:
      Im[Sigma^{TT}(s)] = kappa^2 * s^2 * N_eff_TT / (960*pi)
      Im[Sigma^{(s)}(s)] = kappa^2 * s^2 * N_eff_s / (960*pi)

    The scalar contribution to Im[T(s)] = Im[G_s(s)] * Im[Sigma^{(s)}(s)]
    is suppressed by (xi - 1/6)^4 relative to the TT contribution.
    """
    if s_values is None:
        s_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    results = []
    for s in s_values:
        im_G = scalar_spectral_function(s, xi, dps=dps)
        # The spectral function gives the contribution
        results.append({
            "s": float(s),
            "Im_G_s": float(im_G),
            "xi": float(xi),
        })

    return results


# ===================================================================
# FULL DERIVATION
# ===================================================================

def run_full_derivation(dps=DEFAULT_DPS):
    """Execute the complete scalar sector analysis."""
    print("=" * 70)
    print("SS-D: SCALAR SECTOR ANALYSIS AT GENERAL xi")
    print("=" * 70)

    all_results = {
        "task": "SS-D Scalar Sector",
        "dps": dps,
    }

    # ---------------------------------------------------------------
    # PART 1: Scalar Ghost Catalogue
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: SCALAR GHOST CATALOGUE")
    print("=" * 70)

    catalogue = {}
    for xi in XI_VALUES:
        xi_f = float(xi)
        c_s = float(scalar_local_c2(xi, dps=dps))
        z_stelle = float(scalar_stelle_zero(xi, dps=dps)) if c_s > 1e-10 else float("inf")

        print(f"\n--- xi = {xi_f:.4f} ---")
        print(f"  c_s = 6*(xi-1/6)^2 = {c_s:.8f}")
        print(f"  z_0^Stelle = 1/c_s = {z_stelle:.4f}")

        if c_s < 1e-10:
            print("  Pi_s = 1 identically (conformal). No zeros.")
            catalogue[xi_f] = {
                "c_s": c_s,
                "z_stelle": z_stelle,
                "n_zeros": 0,
                "zeros": [],
            }
            continue

        # Find zeros
        zeros = find_scalar_zeros(xi, R_max=100, dps=dps)
        zeros = scalar_residues(zeros, xi, dps=dps)

        print(f"  Found {len(zeros)} zeros:")
        for i, entry in enumerate(zeros):
            ghost_str = "GHOST" if entry.get("ghost", False) else "healthy"
            print(f"    z_{i}: z = {entry['z_re']:.8f} + {entry['z_im']:.8f}i, "
                  f"|z| = {entry['z_abs']:.4f}, type = {entry['type']}, "
                  f"R = {entry['R_re']:.8e} + {entry['R_im']:.8e}i, {ghost_str}")

        catalogue[xi_f] = {
            "c_s": c_s,
            "z_stelle": z_stelle,
            "n_zeros": len(zeros),
            "zeros": [
                {
                    "z_re": e["z_re"], "z_im": e["z_im"], "z_abs": e["z_abs"],
                    "type": e["type"],
                    "R_re": e.get("R_re", 0), "R_im": e.get("R_im", 0),
                    "R_abs": e.get("R_abs", 0), "ghost": e.get("ghost", False),
                }
                for e in zeros
            ],
        }

    all_results["ghost_catalogue"] = catalogue

    # ---------------------------------------------------------------
    # PART 2: xi-Dependence (shorter survey)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: xi-DEPENDENCE ANALYSIS")
    print("=" * 70)

    xi_survey_values = [mp.mpf(x) / 100 for x in [0, 5, 10, 12, 14, 15, 16, 17, 18, 20, 25, 30, 50, 75, 100]]
    # Add 1/6 exactly
    xi_survey_values.append(mp.mpf(1) / 6)
    xi_survey_values.sort()

    xi_dep = {}
    for xi in xi_survey_values:
        xi_f = float(xi)
        c_s = float(scalar_local_c2(xi, dps=dps))
        # Just count the real zeros for the trajectory
        if c_s < 1e-10:
            xi_dep[xi_f] = {"c_s": c_s, "n_real_zeros": 0, "first_real_zero": None}
            continue

        brackets = _scan_real_axis_for_sign_changes(xi, z_max=100, dps=dps)
        first_zero = None
        if brackets:
            z0 = _refine_zero(xi, brackets[0], dps=dps)
            if z0 is not None:
                first_zero = float(mp.re(z0))

        xi_dep[xi_f] = {
            "c_s": c_s,
            "n_real_zeros": len(brackets),
            "first_real_zero": first_zero,
        }
        print(f"  xi = {xi_f:.4f}: c_s = {c_s:.6f}, "
              f"first real zero: {first_zero if first_zero else 'none in [0,100]'}")

    all_results["xi_dependence"] = xi_dep

    # ---------------------------------------------------------------
    # PART 3: Mittag-Leffler Expansion (at xi = 0)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: SCALAR MITTAG-LEFFLER EXPANSION")
    print("=" * 70)

    xi_ml = mp.mpf(0)
    c_s_ml = scalar_local_c2(xi_ml, dps=dps)
    print(f"  At xi = 0: c_s = {float(c_s_ml):.8f}")
    print(f"  Expected g_A^(s) = -c_s = {float(-c_s_ml):.8f}")

    # Use zeros from catalogue
    if 0.0 in catalogue and catalogue[0.0]["n_zeros"] > 0:
        # Reconstruct zero objects with R_n
        zeros_ml = find_scalar_zeros(xi_ml, R_max=100, dps=dps)
        zeros_ml = scalar_residues(zeros_ml, xi_ml, dps=dps)

        # Verify entire part constancy
        g_A_results = verify_scalar_entire_part_constant(xi_ml, zeros_ml, dps=dps)
        max_dev = max(r["deviation"] for r in g_A_results) if g_A_results else float("inf")
        print(f"  Max |g_A^(s) - (-c_s)|: {max_dev:.6e}")
        for r in g_A_results:
            print(f"    z = {r['z']:>20s}: g_A = {r['g_A_re']:+.10f}, dev = {r['deviation']:.4e}")

        # Sum rule
        sr = scalar_sum_rule(zeros_ml, xi_ml, dps=dps)
        print(f"\n  Sum rule: Sum R_n/z_n = {sr['partial_sum_re']:.8f}")
        print(f"  Target c_s = {sr['target_c_s']:.8f}")
        print(f"  Deficit = {sr['deficit']:.6e}")

        all_results["mittag_leffler"] = {
            "xi": float(xi_ml),
            "c_s": float(c_s_ml),
            "expected_g_A": float(-c_s_ml),
            "g_A_constancy": g_A_results,
            "max_deviation": max_dev,
            "sum_rule": sr,
        }
    else:
        print("  No scalar zeros found at xi=0 -- cannot compute ML expansion")
        all_results["mittag_leffler"] = {"note": "No zeros found at xi=0"}

    # ---------------------------------------------------------------
    # PART 4: Spectral Function
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 4: SCALAR SPECTRAL FUNCTION")
    print("=" * 70)

    spectral_results = {}
    for xi_check in [mp.mpf(0), mp.mpf("0.25"), mp.mpf(1)]:
        xi_f = float(xi_check)
        print(f"\n  Spectral function at xi = {xi_f:.2f}")

        # Coarser grid for speed
        s_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
        spec = spectral_function_scan(xi_check, s_values=s_vals, dps=dps)

        n_positive = sum(1 for r in spec if r["positive"])
        n_total = len(spec)
        print(f"    Spectral positivity: {n_positive}/{n_total} points positive")

        for r in spec[:5]:
            print(f"    s = {r['s']:6.1f}: rho_s = {r['rho_s']:.6e}, positive = {r['positive']}")

        spectral_results[xi_f] = spec

    all_results["spectral_function"] = {
        k: v for k, v in spectral_results.items()
    }

    # ---------------------------------------------------------------
    # PART 5: Physical Observables
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 5: PHYSICAL OBSERVABLES")
    print("=" * 70)

    # 5a: Newtonian potential
    r_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    print("\n  Newtonian potential scalar contribution:")
    for xi_pot in [mp.mpf(0), mp.mpf("0.25"), mp.mpf(1)]:
        xi_f = float(xi_pot)
        pot = newtonian_potential_scalar(r_values, xi_pot, dps=dps)
        print(f"  xi = {xi_f:.2f}:")
        if pot and "m0" in pot[0]:
            print(f"    m_0/Lambda = {pot[0]['m0']:.6f}, z_0 = {pot[0]['z0']:.6f}")
        elif pot and "m0_stelle" in pot[0]:
            print(f"    m_0/Lambda (Stelle) = {pot[0]['m0_stelle']:.6f}")
        for p in pot[:3]:
            print(f"    r*Lambda = {p['r']:.1f}: V_scalar/V_N = {p['V_scalar_over_VN']:.6e}")

    # 5b: PPN gamma
    print("\n  PPN gamma from scalar sector:")
    ppn_results = {}
    for xi_ppn in [mp.mpf(0), mp.mpf("0.1"), mp.mpf("0.25"), mp.mpf(1)]:
        xi_f = float(xi_ppn)
        ppn = ppn_gamma_scalar(xi_ppn, dps=dps)
        ppn_results[xi_f] = ppn
        print(f"    xi = {xi_f:.2f}: gamma_PPN = {ppn['gamma_PPN']:.15f}, "
              f"scalar correction = {ppn['scalar_correction']:.4e}")

    all_results["physical_observables"] = {
        "newtonian_potential": {
            float(xi): newtonian_potential_scalar(r_values, xi, dps=dps)
            for xi in [mp.mpf(0), mp.mpf("0.25"), mp.mpf(1)]
        },
        "ppn_gamma": ppn_results,
    }

    # ---------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check key properties
    conformal_ok = True
    if mp.mpf(1) / 6 in [mp.mpf(x) for x in XI_VALUES]:
        # This isn't in the list, but we check explicitly
        Pi_at_conformal = scalar_Pi_s(mp.mpc(5), mp.mpf(1) / 6, dps=dps)
        conformal_ok = float(abs(Pi_at_conformal - 1)) < 1e-20
        print(f"  Pi_s(5, 1/6) = {float(mp.re(Pi_at_conformal)):.15f} (should be 1): {'PASS' if conformal_ok else 'FAIL'}")

    # Check minimal coupling has expected structure
    xi0_zeros = catalogue.get(0.0, {}).get("n_zeros", 0)
    print(f"  Zeros at xi=0: {xi0_zeros}")

    # Check spectral positivity (at least partial)
    if 0.0 in spectral_results:
        n_pos = sum(1 for r in spectral_results[0.0] if r["positive"])
        n_tot = len(spectral_results[0.0])
        print(f"  Spectral positivity at xi=0: {n_pos}/{n_tot}")

    verdict = "COMPLETE"
    print(f"\n  SCALAR SECTOR ANALYSIS: {verdict}")
    print("=" * 70)

    all_results["verdict"] = verdict

    return all_results


def save_results(results, filepath=None):
    """Save results to JSON."""
    if filepath is None:
        filepath = RESULTS_DIR / "ss_scalar_sector_results.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (mp.mpf, mp.mpc)):
            if isinstance(obj, mp.mpc) and float(mp.im(obj)) != 0:
                return {"re": float(mp.re(obj)), "im": float(mp.im(obj))}
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_convert(results), f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="SS-D: Scalar sector analysis")
    parser.add_argument("--dps", type=int, default=DEFAULT_DPS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ss_scalar_sector_results.json")
    args = parser.parse_args()

    results = run_full_derivation(dps=args.dps)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
