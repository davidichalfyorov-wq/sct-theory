"""
Boundary Seeley-DeWitt analysis for causal diamond in pp-wave spacetime.
========================================================================

CONTEXT: For pp-wave spacetimes with quadratic profile H = eps*(x^2-y^2),
ALL bulk Seeley-DeWitt coefficients a_k = 0 for k >= 1 (VSI spacetime).
Yet the causal set commutator [H,M] = (L^TL - LL^T)/2 gives nonzero signal
(Cohen d = +0.59, p ~ 10^{-6}).

QUESTION: Do boundary Seeley-DeWitt terms provide an explanation?

APPROACH:
  1. Check applicability of standard Gilkey formulas to null boundary
  2. Compute null geometry (expansion, shear) of the diamond boundary
  3. Compute diamond volume and area corrections
  4. Assess whether boundary effects explain the commutator signal

RESULTS:
  - Standard Gilkey formulas DO NOT APPLY (null boundary, conical tips, corner)
  - Null shear sigma_AB = O(eps) is NONZERO on the diamond boundary
  - Diamond area and volume corrections are O(eps^2) for scalar invariants
  - The causal set commutator detects O(eps) TENSOR effects, not just scalars
  - Boundary deformation provides a QUALITATIVE explanation for the signal
  - Quantitative prediction requires a null-boundary heat kernel theory (open problem)

REFERENCES:
  Vassilevich (2003) hep-th/0306138 — heat kernel review
  Branson-Gilkey (1990) — boundary heat kernel coefficients
  Dowker (2014) arXiv:1310.0581 — heat kernel with conical singularities
  Fursaev (2006) hep-th/0602134 — conical contributions to spectral geometry
  Gibbons-Solodukhin (2007) arXiv:0703098 — causal diamond volume
  Solodukhin (2011) Living Reviews — entanglement entropy and heat kernel
  Coley-Hervik-Pelavas (2009) arXiv:0901.0791 — VSI spacetimes
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import (
    symbols, sqrt, simplify, series, Rational, pi, integrate,
    cos, sin, cosh, sinh, diff, log, tanh, Function,
)


# =====================================================================
# Coordinates and parameters
# =====================================================================
u, eps, U = symbols('u epsilon U', positive=True)
se = sqrt(eps)


# =====================================================================
# PART 1: Null geodesics in pp-wave
# =====================================================================
def null_geodesics():
    """Exact null geodesics from origin in pp-wave H = eps*(x^2-y^2).

    Geodesic equations: x'' = -(1/2) dH/dx = -eps*x  => x(u) = sin (focusing)
                        y'' = -(1/2) dH/dy = +eps*y  => y(u) = sinh (defocusing)

    x(u) = (x0d/sqrt(eps)) * sin(sqrt(eps)*u)      [focusing in x]
    y(u) = (y0d/sqrt(eps)) * sinh(sqrt(eps)*u)      [defocusing in y]
    v(u) = -(x0d^2 + y0d^2)/2 * u                   [constant, exact]
    """
    x0d, y0d = symbols('xdot0 ydot0', real=True)

    x_u = (x0d / se) * sin(se * u)    # focusing: x'' = -eps*x
    y_u = (y0d / se) * sinh(se * u)   # defocusing: y'' = +eps*y

    # v-equation: dv/du = -(1/2)(H + xdot^2 + ydot^2)  [Brinkmann null geodesic]
    xdot = diff(x_u, u)
    ydot = diff(y_u, u)
    H_val = eps * (x_u**2 - y_u**2)
    v_dot = -Rational(1, 2) * (xdot**2 + ydot**2 + H_val)
    v_dot_simp = simplify(v_dot)
    # Uses cos^2 + sin^2 = 1 and cosh^2 - sinh^2 = 1
    # => v_dot = -(x0d^2 + y0d^2)/2 = CONSTANT (exact for all eps)

    return {
        'x': x_u,
        'y': y_u,
        'v_dot': v_dot_simp,
        'v_dot_exact': -(x0d**2 + y0d**2) / 2,
    }


# =====================================================================
# PART 2: Null expansion and shear
# =====================================================================
def null_expansion_and_shear():
    """Compute null expansion theta and shear sigma on the diamond boundary.

    theta = sqrt(eps) * [coth(sqrt(eps)*u) + cot(sqrt(eps)*u)]
    theta_flat = 2/u

    sigma_AB = diag(theta_x - theta/2, theta_y - theta/2)  [2x2 traceless]
    |sigma|^2 = (theta_x - theta_y)^2 / 2
    """
    theta_x = se * cosh(se * u) / sinh(se * u)  # se * coth(se*u)
    theta_y = se * cos(se * u) / sin(se * u)     # se * cot(se*u)
    theta = theta_x + theta_y
    theta_flat = 2 / u

    delta_theta = theta - theta_flat
    sigma_sq = (theta_x - theta_y)**2 / 2

    # Perturbative expansions
    delta_theta_pert = series(delta_theta, eps, 0, n=3)
    sigma_sq_pert = series(sigma_sq, eps, 0, n=3)
    diff_theta_pert = series(theta_x - theta_y, eps, 0, n=3)

    # Verify Raychaudhuri equation: dtheta/du + theta^2/2 + sigma^2 = 0
    dtheta_du = diff(theta, u)
    raychaudhuri_check = simplify(dtheta_du + theta**2 / 2 + sigma_sq)

    return {
        'theta': theta,
        'theta_flat': theta_flat,
        'delta_theta': delta_theta,
        'delta_theta_pert': delta_theta_pert,
        'theta_x': theta_x,
        'theta_y': theta_y,
        'sigma_sq': sigma_sq,
        'sigma_sq_pert': sigma_sq_pert,
        'diff_theta_pert': diff_theta_pert,
        'raychaudhuri_check': raychaudhuri_check,
    }


# =====================================================================
# PART 3: Boundary area and diamond volume
# =====================================================================
def boundary_geometry():
    """Compute cross-section area, boundary area, and diamond volume.

    Key results:
      A(u)/A_flat(u) = sinh(se*u)*sin(se*u)/(eps*u^2) = 1 - eps^2*u^4/90 + O(eps^4)
      S_ppwave/S_flat = 1 - eps^2*U^4/210 + O(eps^4)
      V_ppwave/V_flat = 1 - (5/15552)*eps^2*U^4 + O(eps^4)
    """
    # Cross-sectional area at parameter u
    A_ppwave = sinh(se * u) * sin(se * u) / eps
    A_flat = u**2
    area_ratio = A_ppwave / A_flat

    # Perturbative expansion
    area_ratio_pert = series(area_ratio, eps, 0, n=4)

    # Boundary 3-area (half-diamond, 0 to U)
    S_ppwave = integrate(A_ppwave, (u, 0, U))
    S_flat_val = integrate(A_flat, (u, 0, U))
    delta_S_over_S = series((S_ppwave - S_flat_val) / S_flat_val, eps, 0, n=4)

    # Diamond 4-volume (with v-weight u*(U-u), by symmetry 2 * half)
    V_ppwave = 2 * integrate(A_ppwave * u * (U - u), (u, 0, U / 2))
    V_flat_val = 2 * integrate(A_flat * u * (U - u), (u, 0, U / 2))
    delta_V_over_V = series((V_ppwave - V_flat_val) / V_flat_val, eps, 0, n=4)

    # Van Vleck-Morette determinant (transverse)
    Delta_ppwave = eps / (sinh(se * u) * sin(se * u))
    Delta_flat = 1 / u**2
    Delta_ratio_pert = series(Delta_ppwave / Delta_flat, eps, 0, n=4)

    return {
        'area_ratio_pert': area_ratio_pert,
        'delta_S_over_S': delta_S_over_S,
        'delta_V_over_V': delta_V_over_V,
        'Delta_ratio_pert': Delta_ratio_pert,
    }


# =====================================================================
# PART 4: Null second fundamental form invariants
# =====================================================================
def null_extrinsic_invariants():
    """Compute changes in null extrinsic curvature invariants.

    B_{AB}B^{AB} = theta_x^2 + theta_y^2
    delta(theta^2) = -8*eps^2*u^2/45 + O(eps^3)
    delta(B_{AB}B^{AB}) = +2*eps^2*u^2/15 + O(eps^3)
    """
    theta_x = se * cosh(se * u) / sinh(se * u)
    theta_y = se * cos(se * u) / sin(se * u)
    theta = theta_x + theta_y
    theta_flat = 2 / u

    delta_theta_sq = series(theta**2 - theta_flat**2, eps, 0, n=3)
    delta_BB = series(
        (theta_x**2 + theta_y**2) - 2 / u**2, eps, 0, n=3
    )

    return {
        'delta_theta_sq': delta_theta_sq,
        'delta_BB': delta_BB,
    }


# =====================================================================
# PART 5: Numerical evaluation
# =====================================================================
def numerical_evaluation(eps_val: float, U_val: float) -> dict:
    """Evaluate boundary quantities numerically for given eps and U.

    Parameters
    ----------
    eps_val : float
        Amplitude of pp-wave (H = eps*(x^2-y^2))
    U_val : float
        Half-size of causal diamond (retarded time extent)

    Returns
    -------
    dict with numerical results
    """
    se_num = np.sqrt(eps_val)
    u_eq = U_val / 2  # equator

    # Cross-section area ratio at equator
    if eps_val > 0:
        area_ratio_eq = (
            np.sinh(se_num * u_eq) * np.sin(se_num * u_eq)
            / (eps_val * u_eq**2)
        )
    else:
        area_ratio_eq = 1.0

    # Null expansion
    if eps_val > 0 and u_eq > 0:
        theta_x = se_num / np.tanh(se_num * u_eq)
        theta_y = se_num / np.tan(se_num * u_eq)
        theta_ppwave = theta_x + theta_y
        theta_flat = 2.0 / u_eq
        delta_theta = theta_ppwave - theta_flat
        sigma_sq = 0.5 * (theta_x - theta_y)**2
    else:
        delta_theta = 0.0
        sigma_sq = 0.0
        theta_x = theta_y = theta_ppwave = theta_flat = 0.0

    # Perturbative predictions
    delta_V_pert = -(5.0 / 15552) * eps_val**2 * U_val**4
    delta_S_pert = -eps_val**2 * U_val**4 / 210

    # Caustic check: y-direction caustic at u = pi/sqrt(eps)
    if eps_val > 0:
        u_caustic = np.pi / se_num
        caustic_in_diamond = u_caustic < U_val
    else:
        u_caustic = np.inf
        caustic_in_diamond = False

    return {
        'eps': eps_val,
        'U': U_val,
        'area_ratio_equator': area_ratio_eq,
        'delta_theta': delta_theta,
        'sigma_sq': sigma_sq,
        'theta_x_minus_theta_y': theta_x - theta_y if eps_val > 0 else 0.0,
        'delta_V_V_pert': delta_V_pert,
        'delta_S_S_pert': delta_S_pert,
        'u_caustic': u_caustic,
        'caustic_in_diamond': caustic_in_diamond,
    }


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print('=' * 72)
    print('BOUNDARY SEELEY-DEWITT ANALYSIS: PP-WAVE CAUSAL DIAMOND')
    print('=' * 72)

    # ---- Part 1: Gilkey applicability ----
    print('\n' + '=' * 72)
    print('PART 1: APPLICABILITY OF STANDARD GILKEY FORMULAS')
    print('=' * 72)
    print("""
The causal diamond boundary is the union of two NULL hypersurfaces
(past and future light cones).

Three independent obstructions to the standard Gilkey-Branson formulas:

  1. NULL BOUNDARY: The induced metric on a null surface is degenerate
     (rank d-2). Standard formulas require nondegenerate induced metric.
     [Vassilevich 2003, Sec. 4; Branson-Gilkey 1990]

  2. CONICAL SINGULARITIES: The diamond tips are conical singularities
     where all null generators converge to a point.
     [Cheeger 1983; Fursaev 2006 hep-th/0602134]

  3. CORNER AT EQUATOR: The two null sheets meet at a codimension-2
     surface, requiring additional spectral contributions.
     [Dowker 2014 arXiv:1310.0581]

  CONCLUSION: Standard Gilkey formulas DO NOT APPLY.
""")

    # ---- Part 2: Null geometry ----
    print('=' * 72)
    print('PART 2: NULL GEOMETRY OF THE DIAMOND BOUNDARY')
    print('=' * 72)

    geo = null_geodesics()
    print(f'\nNull geodesics from origin:')
    print(f'  v_dot = {geo["v_dot"]}  (EXACT for all eps)')

    nul = null_expansion_and_shear()
    print(f'\nNull expansion:')
    print(f'  delta_theta = {nul["delta_theta_pert"]}')
    print(f'\nNull shear:')
    print(f'  theta_x - theta_y = {nul["diff_theta_pert"]}')
    print(f'  |sigma|^2 = {nul["sigma_sq_pert"]}')
    print(f'\nRaychaudhuri check: {nul["raychaudhuri_check"]}')
    assert nul['raychaudhuri_check'] == 0, 'Raychaudhuri equation FAILED!'
    print('  PASS: Raychaudhuri equation satisfied exactly.')

    # ---- Part 3: Boundary geometry ----
    print('\n' + '=' * 72)
    print('PART 3: GEOMETRIC MODIFICATIONS OF THE DIAMOND')
    print('=' * 72)

    bnd = boundary_geometry()
    print(f'\nCross-section area ratio:')
    print(f'  A_ppwave/A_flat = {bnd["area_ratio_pert"]}')
    print(f'\nBoundary 3-area correction:')
    print(f'  delta_S/S = {bnd["delta_S_over_S"]}')
    print(f'\nDiamond 4-volume correction:')
    print(f'  delta_V/V = {bnd["delta_V_over_V"]}')
    print(f'\nVan Vleck-Morette determinant ratio:')
    print(f'  Delta_ppwave/Delta_flat = {bnd["Delta_ratio_pert"]}')

    # ---- Part 4: Null extrinsic invariants ----
    print('\n' + '=' * 72)
    print('PART 4: NULL EXTRINSIC CURVATURE INVARIANTS')
    print('=' * 72)

    inv = null_extrinsic_invariants()
    print(f'\n  delta(theta^2)    = {inv["delta_theta_sq"]}')
    print(f'  delta(B_AB B^AB)  = {inv["delta_BB"]}')

    # ---- Part 5: Numerical evaluation ----
    print('\n' + '=' * 72)
    print('PART 5: NUMERICAL EVALUATION')
    print('=' * 72)

    print('\n  eps=2.0, U=1.0 (typical FND-1 experiment):')
    res = numerical_evaluation(2.0, 1.0)
    for k, v in res.items():
        print(f'    {k}: {v}')

    print('\n  Scan over eps values (U=1):')
    for e in [0.1, 0.5, 1.0, 2.0, 5.0]:
        r = numerical_evaluation(e, 1.0)
        print(
            f'    eps={e:.1f}: area_ratio={r["area_ratio_equator"]:.6f}, '
            f'|sigma|^2={r["sigma_sq"]:.6f}, '
            f'caustic={r["caustic_in_diamond"]}'
        )

    # ---- Part 6: Conclusions ----
    print('\n' + '=' * 72)
    print('CONCLUSIONS')
    print('=' * 72)
    print("""
Q1: Can we apply Gilkey formulas to the causal diamond boundary?
    NO. Three obstructions: null boundary, conical tips, equatorial corner.
    No published "null boundary Seeley-DeWitt coefficient" formalism exists.

Q2: Is the extrinsic curvature (null version) nonzero?
    YES. The null shear sigma_AB = diag(+eps*u/3, -eps*u/3) is O(eps).
    Scalar invariants (|sigma|^2, delta(theta^2)) are O(eps^2).
    Raychaudhuri equation verified exactly.

Q3: Do the boundary invariants differ between flat and pp-wave?
    YES. All boundary geometric quantities are modified:
      - Cross-section area: 1 - eps^2*u^4/90 + O(eps^4)
      - Boundary area:      1 - eps^2*U^4/210 + O(eps^4)
      - Diamond volume:     1 - (5/15552)*eps^2*U^4 + O(eps^4)
    The O(eps) correction cancels (traceless/Ricci-flat property).

Q4: Do boundary terms explain the commutator signal?
    PARTIALLY. The boundary IS modified, and the modification is nonzero.
    The causal set commutator probes O(eps) TENSOR structure (the shear),
    not just O(eps^2) scalar invariants. This explains why [H,M] detects
    geometry that is invisible to all polynomial curvature scalars (VSI).

    However, a QUANTITATIVE prediction requires either:
    (a) A null-boundary heat kernel theory (open mathematical problem), or
    (b) Direct numerical verification of eps-scaling at multiple values.

KEY PHYSICAL INSIGHT:
    The pp-wave Riemann tensor R^x_{uxu} = -eps, R^y_{uyu} = +eps
    is NONZERO but has VANISHING scalar invariants (VSI).
    It creates anisotropic null shear on the diamond boundary.
    The causal set commutator is a nonlocal scalar functional sensitive to
    this O(eps) anisotropy (null shear) of the boundary geometry,
    going beyond what any scalar Seeley-DeWitt coefficient can capture.

    This is the fundamental reason why the commutator signal is nonzero
    on a VSI spacetime: the causal structure encodes tensor information
    that is lost in the scalar heat kernel trace.
""")
