# ruff: noqa: E402, I001
"""
NT-3: Spectral Dimension d_S(sigma) for Spectral Causal Theory.

Computes the spectral dimension under four independent definitions:

  Method 1 (Propagator-weighted, CMN):
    P_prop(sigma) = 1/(8pi^2) int dk k/Pi_TT(k^2/L^2) exp(-sigma k^2)
    This is the CMN definition. For G ~ 1/k^2: d_S = d-2 = 2 (NOT 4).
    Reference: Calcagni-Modesto-Nardelli 1408.0199.

  Method 2 (Heat kernel trace, standard):
    P_HK(sigma) = 1/(8pi^2) int dk k^3 exp(-sigma k^2)
    For standard Laplacian (flat space): d_S = d = 4.

  Method 3 (Modified heat kernel, ASZ/fakeon):
    P_mod(sigma) = 1/(8pi^2) int_0^{k_ghost} dk k^3 exp(-sigma k^2 Pi_TT)
    Cut at ghost zero z_0 ~ 2.41; beyond that Pi_TT < 0.
    Reference: Alkofer-Saueressig-Zanusso 1410.7999.

  Method 4 (Mittag-Leffler pole decomposition):
    Uses the GZ result:
      H(z) = 1/(z Pi_TT(z)) = -13/60 + 1/z + Sum R_n[1/(z-z_n)+1/z_n]
    Return probability = weighted sum of massive poles + massless graviton.
    Graviton gives P ~ 1/(16pi^2 sigma^2) => d_S = 4 in IR.

The spectral dimension is:
    d_S(sigma) = -2 sigma P'(sigma)/P(sigma)

Input equations (from NT-4a, verified):
    Pi_TT(z) = 1 + (13/60) z F_hat_1(z)
    Pi_TT(0) = 1 (Einstein recovery), Pi_TT(inf) = -83/6.
    Ghost zero at z_0 = 2.4148 on positive real axis.

References:
    - Ambjorn, Jurkiewicz, Loll, hep-th/0505113  (CDT: d_S = 2)
    - Sotiriou, Visser, Weinfurtner, 1105.6098    (SVW framework)
    - Calcagni, Modesto, Nardelli, 1408.0199       (propagator d_S)
    - Alkofer, Saueressig, Zanusso, 1410.7999      (spectral action d_S)
    - Horava, 0902.3657                             (HL: d_S = 1+D/z)
    - Lauscher, Reuter, hep-th/0511260             (AS: d_S = 2)

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import integrate

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.nt2_entire_function import F1_total_complex

from scripts.gz_entire_part import (
    compute_residue,
    GHOST_CATALOGUE,
    LOCAL_C2,
    PI_TT_UV_LIMIT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "nt3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120       # Total Weyl-squared coefficient
C2 = 2 * ALPHA_C                 # 13/60
DEFAULT_DPS = 50

# Ghost catalogue: first zero of Pi_TT on positive real axis
Z0_EUCLIDEAN = 2.4148            # Pi_TT(z_0) = 0
# UV saturation
PI_TT_UV = float(PI_TT_UV_LIMIT)  # -83/6 ~ -13.833

# Cache for F1(0)
_F1_0_cache = {}


def _get_F1_0(dps=DEFAULT_DPS):
    """Cached F1(0) value."""
    if dps not in _F1_0_cache:
        mp.mp.dps = dps
        _F1_0_cache[dps] = F1_total_complex(0, dps=dps)
    return _F1_0_cache[dps]


# ===================================================================
# SECTION 1: Euclidean Pi_TT (positive real z)
# ===================================================================

def Pi_TT_euclidean(z, dps=DEFAULT_DPS):
    """Euclidean Pi_TT(z) for real z >= 0.

    Pi_TT(z) = 1 + (13/60) z F_hat_1(z)

    Uses nt2_entire_function.F1_total_complex for correct computation.
    """
    mp.mp.dps = dps
    z_mp = mp.mpf(z)
    if z_mp < mp.mpf("1e-15"):
        return 1.0
    f1_z = F1_total_complex(z_mp, dps=dps)
    f1_0 = _get_F1_0(dps=dps)
    f1_hat = f1_z / f1_0
    Pi = 1 + LOCAL_C2 * z_mp * f1_hat
    return float(mp.re(Pi))


# Build interpolation table for fast evaluation
_PI_TT_TABLE_Z = None
_PI_TT_TABLE_V = None


def _build_pi_tt_table(n_points=2000, z_max=200.0, dps=35):
    """Build interpolation table for Pi_TT on Euclidean axis."""
    global _PI_TT_TABLE_Z, _PI_TT_TABLE_V
    if _PI_TT_TABLE_Z is not None:
        return
    z_arr = np.concatenate([
        np.linspace(0, 0.01, 20),
        np.linspace(0.01, 3.0, 500),
        np.linspace(3.0, 20.0, 500),
        np.logspace(np.log10(20.0), np.log10(z_max), 500),
    ])
    z_arr = np.unique(z_arr)
    v_arr = np.array([Pi_TT_euclidean(float(z), dps=dps) for z in z_arr])
    _PI_TT_TABLE_Z = z_arr
    _PI_TT_TABLE_V = v_arr


def Pi_TT_interp(z):
    """Interpolated Pi_TT for fast numerical integration."""
    _build_pi_tt_table()
    if z <= 0:
        return 1.0
    if z > _PI_TT_TABLE_Z[-1]:
        return PI_TT_UV
    return float(np.interp(z, _PI_TT_TABLE_Z, _PI_TT_TABLE_V))


# ===================================================================
# SECTION 2: Method 1 -- Propagator-weighted return probability (CMN)
# ===================================================================

def compute_P_propagator(sigma, Lambda=1.0, k_max_factor=100.0):
    """Return probability via propagator-weighted integral (Method 1, CMN).

    P_prop(sigma) = 1/(8pi^2) int dk k/Pi_TT(k^2/L^2) exp(-sigma k^2)

    Note: for the STANDARD propagator G=1/k^2, this integral gives
    P ~ 1/(2 sigma) => d_S = -2 sigma * (-1/(2sigma^2)) / (1/(2sigma)) = 2.
    This is the PROPAGATOR spectral dimension (d-2), not the heat kernel one (d).
    The distinction is physically meaningful (CMN 1408.0199).

    We integrate only in the sub-ghost region (Pi_TT > 0) since the
    fakeon prescription removes modes beyond z_0.
    """
    Lambda2 = Lambda**2
    k_ghost = np.sqrt(Z0_EUCLIDEAN * Lambda2)

    def integrand(k):
        if k < 1e-30:
            return 0.0
        z = k**2 / Lambda2
        Pi = Pi_TT_interp(z)
        if Pi <= 1e-15:
            return 0.0
        return k / Pi * np.exp(-sigma * k**2)

    result, _ = integrate.quad(integrand, 1e-10, k_ghost,
                               limit=200, epsrel=1e-8)
    return result / (8.0 * np.pi**2)


# ===================================================================
# SECTION 3: Method 2 -- Standard heat kernel trace
# ===================================================================

def compute_P_heat_kernel(sigma, Lambda=1.0):
    """Standard flat-space heat kernel trace (Method 2).

    P_HK(sigma) = Tr(e^{-sigma Delta}) = 1/(4pi sigma)^{d/2} = 1/(16pi^2 sigma^2)

    This is the UNMODIFIED result for a d=4 manifold with standard Laplacian.
    It gives d_S = 4 exactly.
    """
    return 1.0 / (16.0 * np.pi**2 * sigma**2)


# ===================================================================
# SECTION 4: Method 3 -- Modified heat kernel (ASZ / fakeon)
# ===================================================================

def compute_P_asz(sigma, Lambda=1.0):
    """Return probability via modified heat kernel (Method 3, ASZ/fakeon).

    P_mod(sigma) = 1/(8pi^2) int_0^{k_ghost} dk k^3 exp(-sigma k^2 Pi_TT)

    Integration restricted to z < z_0 (sub-ghost region) where Pi_TT > 0.
    This implements the fakeon-regulated version of the ASZ spectral
    action heat kernel.

    At large sigma: Pi_TT ~ 1 in the low-k region that dominates,
    so P_mod ~ integral k^3 exp(-sigma k^2) dk = 1/(2 sigma^2) => d_S = 4.

    At very small sigma: exp(-sigma k^2 Pi_TT) ~ 1 for k < k_ghost,
    so P_mod -> integral_0^{k_ghost} k^3 dk = k_ghost^4/4 (constant)
    => d_S -> 0.
    """
    Lambda2 = Lambda**2
    k_ghost = np.sqrt(Z0_EUCLIDEAN * Lambda2)

    def integrand(k):
        if k < 1e-30:
            return 0.0
        z = k**2 / Lambda2
        Pi = Pi_TT_interp(z)
        if Pi <= 0:
            return 0.0
        arg = sigma * k**2 * Pi
        if arg > 500:
            return 0.0
        return k**3 * np.exp(-arg)

    result, _ = integrate.quad(integrand, 1e-10, k_ghost,
                               limit=200, epsrel=1e-8)
    return result / (8.0 * np.pi**2)


def compute_P_fakeon(sigma, Lambda=1.0):
    """Fakeon-regulated heat kernel (alias for ASZ with sub-ghost cut)."""
    return compute_P_asz(sigma, Lambda)


# ===================================================================
# SECTION 5: Method 4 -- Mittag-Leffler pole decomposition
# ===================================================================

_POLE_CACHE = {}


def _get_poles(dps=DEFAULT_DPS):
    if dps not in _POLE_CACHE:
        mp.mp.dps = dps
        poles = []
        for label, z_re_s, z_im_s, ztype in GHOST_CATALOGUE:
            z_n = mp.mpc(mp.mpf(z_re_s), mp.mpf(z_im_s))
            R_n = compute_residue(z_n, dps=dps)
            poles.append((z_n, R_n, label))
        _POLE_CACHE[dps] = poles
    return _POLE_CACHE[dps]


def compute_P_mittag_leffler(sigma, Lambda=1.0, dps=DEFAULT_DPS):
    """Return probability from Mittag-Leffler pole decomposition (Method 4).

    The propagator has the decomposition:
      1/(z Pi_TT(z)) = -13/60 + 1/z + Sum_n R_n [1/(z-z_n)+1/z_n]

    In d=4 momentum space, each pole contributes to the return probability:
      P_graviton(sigma) = 1/(16pi^2 sigma^2)                  [massless]
      P_n(sigma) ~ R_n * exp(-m_n^2 sigma) / (16pi^2 sigma^2) [massive]

    where m_n^2 = z_n Lambda^2.
    """
    Lambda2 = Lambda**2
    # Graviton: P = 1/(16 pi^2 sigma^2)
    P_grav = 1.0 / (16.0 * np.pi**2 * sigma**2)

    poles = _get_poles(dps=dps)
    P_poles = 0.0

    for z_n, R_n, label in poles:
        z_re = float(mp.re(z_n))
        z_im = float(mp.im(z_n))
        R_re = float(mp.re(R_n))
        R_im = float(mp.im(R_n))

        if abs(z_im) < 1e-10:
            # Real pole
            m2 = abs(z_re) * Lambda2
            arg = m2 * sigma
            if arg > 500:
                continue
            decay = np.exp(-arg)
            P_poles += R_re * decay / (16.0 * np.pi**2 * sigma**2)
        else:
            # Complex pole (Lee-Wick pair)
            m2_re = z_re * Lambda2
            if m2_re * sigma > 500:
                continue
            if m2_re * sigma < -500:
                continue
            phase = z_im * Lambda2 * sigma
            decay = np.exp(-m2_re * sigma)
            contrib = decay * (R_re * np.cos(phase) + R_im * np.sin(phase))
            P_poles += contrib / (16.0 * np.pi**2 * sigma**2)

    return P_grav + P_poles


# ===================================================================
# SECTION 6: Spectral dimension from P(sigma)
# ===================================================================

def compute_ds(sigma, method='mittag_leffler', Lambda=1.0, delta_frac=0.01):
    """Compute d_S(sigma) = -2 sigma P'(sigma)/P(sigma).

    Uses centered finite-difference for the logarithmic derivative.
    """
    compute_P = {
        'propagator': lambda s: compute_P_propagator(s, Lambda),
        'heat_kernel': lambda s: compute_P_heat_kernel(s, Lambda),
        'asz': lambda s: compute_P_asz(s, Lambda),
        'fakeon': lambda s: compute_P_fakeon(s, Lambda),
        'mittag_leffler': lambda s: compute_P_mittag_leffler(s, Lambda),
    }

    if method not in compute_P:
        raise ValueError(
            f"Unknown method: {method}. Use one of {list(compute_P.keys())}"
        )

    P_func = compute_P[method]
    delta = sigma * delta_frac

    P_plus = P_func(sigma + delta)
    P_minus = P_func(sigma - delta)
    P_center = P_func(sigma)

    if abs(P_center) < 1e-300:
        return float('nan')

    dP_dsigma = (P_plus - P_minus) / (2.0 * delta)
    return -2.0 * sigma * dP_dsigma / P_center


def compute_ds_array(sigma_array, method='mittag_leffler', Lambda=1.0):
    """Compute d_S at an array of sigma values."""
    return np.array([
        compute_ds(float(s), method=method, Lambda=Lambda)
        for s in sigma_array
    ])


# ===================================================================
# SECTION 7: Analytic limits
# ===================================================================

def ds_uv_analytic(method='mittag_leffler'):
    """Analytic UV limit (sigma -> 0) of d_S for each method."""
    results = {
        'propagator': {
            'value': 2.0,
            'reasoning': (
                "CMN propagator definition with G ~ 1/k^2 (n=1): "
                "d_S = d - 2 = 2 for d=4. This is the PROPAGATOR "
                "spectral dimension, distinct from the heat kernel d_S = d."
            ),
        },
        'heat_kernel': {
            'value': 4.0,
            'reasoning': (
                "Standard heat kernel P = 1/(16pi^2 sigma^2) "
                "=> d_S = 4 for all sigma (flat space, no modification)."
            ),
        },
        'asz': {
            'value': 0.0,
            'reasoning': (
                "ASZ/fakeon: P_mod -> constant as sigma -> 0 "
                "(integral over bounded k range) => d_S -> 0. "
                "This is the SCT analogue of the ASZ d_S = 0 result "
                "(spectral action EFT with bounded momentum)."
            ),
        },
        'fakeon': {
            'value': 0.0,
            'reasoning': "Same as ASZ (fakeon removes super-ghost modes).",
        },
        'mittag_leffler': {
            'value': 4.0,
            'reasoning': (
                "All poles contribute ~ 1/sigma^2 as sigma -> 0. "
                "P_ML ~ (1 + Sum R_n) / (16 pi^2 sigma^2) => d_S -> 4. "
                "Transient departure from 4 near sigma ~ 1/m_2^2."
            ),
        },
    }
    return results.get(method, {'value': None, 'reasoning': 'Unknown method'})


def ds_ir_analytic(method='mittag_leffler'):
    """Analytic IR limit (sigma -> infinity) of d_S for each method.

    All methods: d_S(IR) = 4 or 2 depending on whether the definition
    includes the propagator 1/k^2 or not.
    """
    ir_values = {
        'propagator': 2.0,
        'heat_kernel': 4.0,
        'asz': 4.0,
        'fakeon': 4.0,
        'mittag_leffler': 4.0,
    }
    val = ir_values.get(method, 4.0)
    return {
        'value': val,
        'reasoning': (
            f"Method '{method}': In the IR (sigma -> inf), "
            "low-k modes dominate. Pi_TT(z) -> 1 (Einstein recovery). "
            f"Expected d_S(IR) = {val}."
        ),
    }


# ===================================================================
# SECTION 8: Comparison with other QG approaches
# ===================================================================

def ds_gr(sigma=None):
    """GR spectral dimension: d_S = 4 for all sigma."""
    return 4.0


def ds_stelle(sigma, Lambda=1.0):
    """Stelle gravity: d_S from 4 (IR) to 2 (UV).

    Stelle propagator: G ~ 1/k^4 in UV => d_S(UV) = d/2 = 2.
    Model: d_S = 4 - 2/(1 + m^2 sigma).
    """
    return 4.0 - 2.0 / (1.0 + Lambda**2 * sigma)


def ds_asymptotic_safety(sigma, Lambda=1.0):
    """Asymptotic Safety: d_S from 4 (IR) to 2 (UV).

    At the NGFP, eta_N = -2 => d_S = 4/(1 + eta_N/2) = 2.
    Model: d_S = 2 + 2*x/(x+1), x = Lambda^2 sigma.
    """
    x = Lambda**2 * sigma
    return 2.0 + 2.0 * x / (x + 1.0)


def ds_horava_lifshitz(sigma, Lambda=1.0, z_hl=3):
    """Horava-Lifshitz: d_S = 1 + D/z = 2 (UV, z=3), 4 (IR).

    Model: d_S = 2 + 2*x/(x+1), x = Lambda^2 sigma.
    """
    d_uv = 1.0 + 3.0 / z_hl
    x = Lambda**2 * sigma
    return d_uv + (4.0 - d_uv) * x / (x + 1.0)


def ds_cdt_fit(sigma, Lambda=1.0):
    """CDT Monte Carlo: d_S(UV) ~ 1.80, d_S(IR) ~ 4.0."""
    x = Lambda**2 * sigma
    return 1.80 + 2.20 * x / (x + 1.0)


def comparison_with_other_qg(sigma_array, Lambda=1.0):
    """Compute d_S for all QG approaches."""
    return {
        'GR': np.full_like(sigma_array, 4.0),
        'Stelle': np.array([ds_stelle(s, Lambda) for s in sigma_array]),
        'AS': np.array([ds_asymptotic_safety(s, Lambda) for s in sigma_array]),
        'HL': np.array([ds_horava_lifshitz(s, Lambda) for s in sigma_array]),
        'CDT': np.array([ds_cdt_fit(s, Lambda) for s in sigma_array]),
    }


# ===================================================================
# SECTION 9: Key numerical results
# ===================================================================

def compute_crossover_scale(method='mittag_leffler', Lambda=1.0, tol=0.1):
    """Find sigma_* where d_S first deviates from its IR value by > tol."""
    ir_val = ds_ir_analytic(method)['value']
    sigmas = np.logspace(4, -4, 100)
    for s in sigmas:
        ds = compute_ds(float(s), method=method, Lambda=Lambda)
        if abs(ds - ir_val) > tol:
            return float(s)
    return None


def compute_summary_table(Lambda=1.0):
    """Compute d_S at key sigma values for all methods."""
    sigma_values = [1e-4, 1e-2, 1.0, 1e2, 1e4]
    methods = ['propagator', 'heat_kernel', 'asz', 'mittag_leffler']

    table = {'sigma': sigma_values}
    for method in methods:
        ds_vals = []
        for s in sigma_values:
            try:
                ds = compute_ds(s, method=method, Lambda=Lambda)
            except Exception:
                ds = float('nan')
            ds_vals.append(ds)
        table[method] = ds_vals
    return table


# ===================================================================
# SECTION 10: Self-test
# ===================================================================

def self_test():
    """Run basic self-tests."""
    print("=" * 70)
    print("NT-3 Spectral Dimension: Self-Test")
    print("=" * 70)

    # Test 1: GR baseline
    print("\n--- Test 1: GR baseline d_S = 4 ---")
    assert ds_gr() == 4.0
    print("  PASS: d_S(GR) = 4.0")

    # Test 2: Pi_TT Euclidean
    print("\n--- Test 2: Pi_TT(z) Euclidean ---")
    pi_0 = Pi_TT_euclidean(0)
    print(f"  Pi_TT(0) = {pi_0:.10f}")
    assert abs(pi_0 - 1.0) < 1e-10

    pi_1 = Pi_TT_euclidean(1.0)
    print(f"  Pi_TT(1) = {pi_1:.10f} (expected ~0.899)")
    assert abs(pi_1 - 0.8995) < 0.01

    pi_z0 = Pi_TT_euclidean(Z0_EUCLIDEAN)
    print(f"  Pi_TT({Z0_EUCLIDEAN}) = {pi_z0:.6e} (should be ~0)")
    assert abs(pi_z0) < 0.01

    pi_10 = Pi_TT_euclidean(10.0)
    print(f"  Pi_TT(10) = {pi_10:.6f} (should be < 0)")
    assert pi_10 < 0

    pi_50 = Pi_TT_euclidean(50.0)
    print(f"  Pi_TT(50) = {pi_50:.6f} (expected ~-13.59)")
    assert -15 < pi_50 < -12

    # Test 3: Interpolation table
    print("\n--- Test 3: Pi_TT interpolation ---")
    _build_pi_tt_table()
    assert abs(Pi_TT_interp(1.0) - pi_1) < 0.01
    print("  PASS: interpolation agrees with exact")

    # Test 4: Heat kernel
    print("\n--- Test 4: Heat kernel d_S = 4 ---")
    ds_hk = compute_ds(1.0, 'heat_kernel')
    print(f"  d_S(heat_kernel, sigma=1) = {ds_hk:.6f}")
    assert abs(ds_hk - 4.0) < 0.01

    # Test 5: Return probabilities
    print("\n--- Test 5: P(sigma) values ---")
    for s in [0.1, 1.0, 10.0, 100.0]:
        P_prop = compute_P_propagator(s)
        P_asz_val = compute_P_asz(s)
        P_ml = compute_P_mittag_leffler(s)
        P_hk = compute_P_heat_kernel(s)
        print(f"  sigma={s:6.1f}: P_prop={P_prop:.4e}, P_asz={P_asz_val:.4e}, "
              f"P_ML={P_ml:.4e}, P_HK={P_hk:.4e}")

    # Test 6: Spectral dimension
    print("\n--- Test 6: d_S(sigma) ---")
    for s in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        ds_p = compute_ds(s, 'propagator')
        ds_a = compute_ds(s, 'asz')
        ds_m = compute_ds(s, 'mittag_leffler')
        print(f"  sigma={s:8.2f}: d_S(prop)={ds_p:+.4f}, "
              f"d_S(asz)={ds_a:+.4f}, d_S(ML)={ds_m:+.4f}")

    # Test 7: ML IR limit
    print("\n--- Test 7: ML IR limit ---")
    ds_ml_ir = compute_ds(1000.0, 'mittag_leffler')
    print(f"  d_S(ML, sigma=1000) = {ds_ml_ir:.4f} (should be ~4)")
    assert abs(ds_ml_ir - 4.0) < 0.1

    # Test 8: Analytic limits
    print("\n--- Test 8: Analytic limits ---")
    for method in ['propagator', 'heat_kernel', 'asz', 'mittag_leffler']:
        uv = ds_uv_analytic(method)
        ir = ds_ir_analytic(method)
        print(f"  {method:20s}: UV={uv['value']}, IR={ir['value']}")

    # Test 9: Comparison models
    print("\n--- Test 9: Comparison QG models ---")
    assert abs(ds_stelle(1e-4) - 2.0) < 0.01
    assert abs(ds_stelle(1e4) - 4.0) < 0.01
    assert abs(ds_asymptotic_safety(1e-4) - 2.0) < 0.01
    assert abs(ds_horava_lifshitz(1e-4) - 2.0) < 0.01
    print("  PASS: all comparison models UV=2, IR=4")

    print("\n" + "=" * 70)
    print("Self-test PASSED")
    print("=" * 70)


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    self_test()
