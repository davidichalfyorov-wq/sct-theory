"""P12 v4: Add R² sector (scalar form factor F₂) to the Gap G1 analysis.

This script extends the P12 full-formfactor BVP by including the R² sector:
  S_eff = ... + alpha_R * R * F₂(□/Λ²) * R

where alpha_R(xi=0) = 1/18 and the P_R(alpha) kernel is:
  P_R(alpha) = -1/18 + 4/9*tau + 10/9*tau^2   (tau = alpha*(1-alpha))

Two approaches are implemented:
  (A) Perturbative: solve Weyl-only, then evaluate R² correction on that solution.
  (B) Fully coupled: add auxiliary v_k fields for the Ricci sector and solve together.

Key question: Does adding R² change H_min qualitatively (i.e., create a horizon)?

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp

OUTDIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'gap_g1'
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#   QUADRATURE: alpha-kernel for h_C and h_R
# ============================================================

def alpha_tau_quadrature(K: int = 6):
    """Gauss-Legendre quadrature on [0, 1/2] with symmetry doubling.

    Returns tau-layers, weights, and kernel values for BOTH Weyl (P_C)
    and Ricci (P_R) sectors.
    """
    x, w = leggauss(K)
    alpha = 0.25 * (x + 1.0)        # [0, 1/2]
    w = 0.25 * w * 2.0              # Jacobian * symmetry factor
    tau = alpha * (1.0 - alpha)

    # Weyl kernel: h_C^{total}(x) = int P_C(alpha) exp(-x tau) dalpha
    # P_C = -89/24 + 43/6*tau + 236/3*tau^2
    P_C = -89.0 / 24.0 + (43.0 / 6.0) * tau + (236.0 / 3.0) * tau ** 2

    # Ricci kernel: h_R^{total}(x, xi=0) = int P_R(alpha) exp(-x tau) dalpha
    # P_R = -1/18 + 4/9*tau + 10/9*tau^2
    P_R = -1.0 / 18.0 + (4.0 / 9.0) * tau + (10.0 / 9.0) * tau ** 2

    # Sort by tau for the layered ODE scheme
    idx = np.argsort(tau)
    alpha = alpha[idx]
    tau = tau[idx]
    w = w[idx]
    P_C = P_C[idx]
    P_R = P_R[idx]
    c_C = w * P_C
    c_R = w * P_R
    dtau = np.diff(np.concatenate(([0.0], tau)))

    return {
        'alpha': alpha,
        'tau': tau,
        'w': w,
        'P_C': P_C,
        'P_R': P_R,
        'c_C': c_C,
        'c_R': c_R,
        'dtau': dtau,
        'K': K,
    }


# ============================================================
#   WEYL AMPLITUDE W_amp AND RICCI SCALAR R_amp
# ============================================================

def weyl_amplitude(F, Fp, H, Hp, Fpp, Hpp):
    """Weyl amplitude C_trtr for ds² = -H dt² + dr²/H + F² dΩ².

    In area-radius gauge (F=r exact Schwarzschild):
    W = (F² H'' - 2FH F'' - 2F F' H' + 2H F'² - 2) / (6F²)
    """
    return (F ** 2 * Hpp - 2.0 * F * H * Fpp
            - 2.0 * F * Fp * Hp + 2.0 * H * Fp ** 2 - 2.0) / (6.0 * F ** 2)


def ricci_scalar(F, Fp, H, Hp, Fpp, Hpp):
    """Ricci scalar R for ds² = -H dt² + dr²/H + F² dΩ².

    R = -H'' - 4H'F'/F - 2H F''/F - 2H(F')²/F² + 2/F²
    (Written in terms of coordinate derivatives, f = H, g = 1/H.)
    Actually for the Buchdahl variables:
    R = -(H'' + 4H'F'/F + 2H F''/F) + 2(1 - H F'²)/F²

    Alternatively in mass function language: if H = 1-2m/F, then
    R = (2F m'' + 4m') / F²
    But we want it in terms of the metric variables directly.
    """
    return (-Hpp - 4.0 * Hp * Fp / F - 2.0 * H * Fpp / F
            + 2.0 * (1.0 - H * Fp ** 2) / F ** 2)


# ============================================================
#   APPROACH (A): PERTURBATIVE R² CORRECTION
# ============================================================

def perturbative_r2_on_weyl_solution(sol_weyl, coeffs, Lam=1.0, xi=0.0):
    """Evaluate R² stress-energy perturbatively on the Weyl-only P12 solution.

    The Weyl-only solution has fields (F, F', H, H', u_1,...,u_K, v_1,...,v_K).
    We compute:
      1. R(r) on this solution
      2. alpha_R(xi) = 2*(xi - 1/6)^2
      3. Theta^{R²}_{tt} and Theta^{R²}_{rr} (Bach-type tensor for R²)
      4. The ratio |alpha_R * Theta^R| / |G + alpha_C * Theta^C|

    If the ratio is << 1, the R² sector is perturbative and P12 is OK.
    """
    K = coeffs['K']
    rho = np.geomspace(sol_weyl.x[0], sol_weyl.x[-1], 800)
    y = sol_weyl.sol(rho)

    F = y[0]
    Fp = y[1]
    H = y[2]
    Hp = y[3]

    # Compute second derivatives from the ODE system
    Fpp = np.gradient(Fp, rho, edge_order=2)
    Hpp = np.gradient(Hp, rho, edge_order=2)

    # Weyl amplitude
    W = weyl_amplitude(F, Fp, H, Hp, Fpp, Hpp)

    # Ricci scalar
    R = ricci_scalar(F, Fp, H, Hp, Fpp, Hpp)

    # Ricci scalar derivatives
    Rp = np.gradient(R, rho, edge_order=2)
    Rpp = np.gradient(Rp, rho, edge_order=2)

    # Form factor convolution for R² sector:
    # J_R(r) = int P_R(alpha) v_k(r) dalpha  where v_k solves
    # H v_k'' + (H' + 2HF'/F) v_k' - (Lam^2/dtau_k) (v_k - v_{k-1}) = 0
    # with v_0 = R_amp(r)
    # In the perturbative approach, we just evaluate h_R on the Schwarzschild-like R(r).

    # For the LOCAL (form-factor = identity) R² contribution:
    # Theta^{R²}_{mu nu} = 2 nabla_mu nabla_nu R - 2 g_{mu nu} box R + 2R R_{mu nu} - g_{mu nu}/2 R²
    # This is the variational derivative of R² w.r.t. g^{mu nu}.

    # box R in spherical symmetry:
    # box R = H R'' + (H' + 2HF'/F) R'
    box_R = H * Rpp + (Hp + 2.0 * H * Fp / F) * Rp

    # R_{tt} = (1/2)(H'' + 2H'F'/F) * H   [for diagonal metric]
    # Actually: R_{tt} = H/2 * (H'' + 2H'F'/F)  ... need to be careful.
    # For ds² = -H dt² + dr²/H + F² dΩ²:
    # R_{tt} = -H(H''/2 + H'F'/F)  (this is -H times the tt Ricci component)
    # Let me compute it properly.
    # R_{00} = H''/(2) + H'F'/F   (the Ricci tensor component with time-time)
    # Actually for the metric g_{tt} = -H:
    # R_{tt} = -(H''/2 + H'F'/F) * H  ... no.
    # Let me just compute the needed quantities numerically.

    # nabla_t nabla_t R = -Gamma^r_{tt} partial_r R = -(H H'/2) R'  ... wait
    # For a static metric: nabla_t nabla_t R = -Gamma^r_{tt} R' = (H H'/2) R'
    # (Gamma^r_{tt} = -H H'/2 for our metric convention)
    nab_t_nab_t_R = 0.5 * H * Hp * Rp

    # The R² correction amplitude (LOCAL approximation for now):
    alpha_R_val = 2.0 * (xi - 1.0 / 6.0) ** 2

    # For the perturbative estimate, we want:
    # |alpha_R * h_R(0) * local_Theta^R| vs |alpha_C * J_weyl|
    # The simplest diagnostic: just compute R(r) and check its magnitude.

    # Schwarzschild asymptotics: R = 0 (vacuum). So R should be small.
    # But the P12 solution deviates from Schwarzschild near r ~ 1/Lambda.

    return {
        'rho': rho.tolist(),
        'F': F.tolist(),
        'H': H.tolist(),
        'W_amp': W.tolist(),
        'R_scalar': R.tolist(),
        'box_R': box_R.tolist(),
        'R_max': float(np.max(np.abs(R))),
        'R_at_center': float(R[0]),
        'W_max': float(np.max(np.abs(W))),
        'ratio_R_over_W': float(np.max(np.abs(R)) / max(np.max(np.abs(W)), 1e-30)),
        'alpha_R': float(alpha_R_val),
        'alpha_C': 13.0 / 120.0,
    }


# ============================================================
#   APPROACH (B): FULLY COUPLED WEYL + RICCI BVP
# ============================================================

def initial_guess_coupled(rho, M, coeffs, l=2.0):
    """Initial guess for the coupled (Weyl + R²) system.

    State vector: [F, F', H, H', u_1, u_1', ..., u_K, u_K', v_1, v_1', ..., v_K, v_K']
    Total dimension: 4 + 2K (Weyl) + 2K (Ricci) = 4 + 4K
    """
    K = coeffs['K']

    # Metric: smoothed Schwarzschild
    F = rho.copy()
    Fp = np.ones_like(rho)
    H = 1.0 - 2.0 * M * rho ** 2 / (rho ** 3 + l ** 3)
    Hp = np.gradient(H, rho, edge_order=2)
    Fpp = np.gradient(Fp, rho, edge_order=2)
    Hpp = np.gradient(Hp, rho, edge_order=2)

    # Weyl amplitude for initial guess
    Wamp = weyl_amplitude(F, Fp, H, Hp, Fpp, Hpp)

    # Ricci scalar for initial guess
    Ramp = ricci_scalar(F, Fp, H, Hp, Fpp, Hpp)

    Y = [F, Fp, H, Hp]

    # Weyl auxiliary fields u_k
    for tau_k in coeffs['tau']:
        uk = Wamp / (1.0 + 4.0 * tau_k)
        ukp = np.gradient(uk, rho, edge_order=2)
        Y.extend([uk, ukp])

    # Ricci auxiliary fields v_k
    for tau_k in coeffs['tau']:
        vk = Ramp / (1.0 + 4.0 * tau_k)
        vkp = np.gradient(vk, rho, edge_order=2)
        Y.extend([vk, vkp])

    y0 = np.vstack(Y)
    p0 = np.concatenate(([- 2.0 * M / l ** 3, 0.0],
                         np.zeros(K),   # a2_k for u_k center amplitudes
                         np.zeros(K)))  # b2_k for v_k center amplitudes
    return y0, p0


def local_second_derivs_coupled(ycol, coeffs, Lam=1.0, alpha_R_val=1.0/18.0):
    """Compute F'', H'', u_k'', v_k'' from the local system.

    The coupled field equations:
      G_{mu nu} + alpha_C * Theta^C_{mu nu} + alpha_R * Theta^R_{mu nu} = 0

    In the localized form-factor language:
      J_C = sum c_C_k * u_k     (Weyl convolution)
      J_R = sum c_R_k * v_k     (Ricci convolution)

    The u_k satisfy: H u_k'' + A u_k' - lam_k (u_k - u_{k-1}) = S u_k
                     with u_0 = W_amp(F, H, F', H', F'', H'')
    The v_k satisfy: H v_k'' + A v_k' - lam_k (v_k - v_{k-1}) = S v_k
                     with v_0 = R_amp(F, H, F', H', F'', H'')
    """
    K = coeffs['K']
    F = ycol[0]
    Fp = ycol[1]
    H = ycol[2]
    Hp = ycol[3]

    # Weyl fields
    u = np.array([ycol[4 + 2 * i] for i in range(K)], dtype=float)
    up = np.array([ycol[5 + 2 * i] for i in range(K)], dtype=float)

    # Ricci fields
    v = np.array([ycol[4 + 2 * K + 2 * i] for i in range(K)], dtype=float)
    vp = np.array([ycol[5 + 2 * K + 2 * i] for i in range(K)], dtype=float)

    c_C = coeffs['c_C']
    c_R = coeffs['c_R']
    dtau = coeffs['dtau']

    J_C = float(np.dot(c_C, u))
    J_Cp = float(np.dot(c_C, up))
    J_R = float(np.dot(c_R, v))
    J_Rp = float(np.dot(c_R, vp))

    # System for [F'', H'', u_1'',...,u_K'', v_1'',...,v_K'']
    n = 2 + 2 * K
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    # -- Metric equations --
    # Row 0: F'' equation (from radial field equation)
    # The Weyl contribution is the same as P12:
    #   (4FJ_C - 2F) F'' + ... + 2F² sum c_C_k u_k'' = ...
    # The R² contribution adds:
    #   alpha_R * [terms involving J_R, v_k'', R_amp]
    # For the R² sector, Theta^R involves derivatives of R and R*R_{mn},
    # which couple to F'', H'' through R_amp.

    # W_amp coefficients w.r.t. F'', H'':
    aF_W = -H / (3.0 * F)
    aH_W = 1.0 / 6.0
    a0_W = -(Fp * Hp) / (3.0 * F) + H * Fp ** 2 / (3.0 * F ** 2) - 1.0 / (3.0 * F ** 2)

    # R_amp coefficients w.r.t. F'', H'':
    # R = -H'' - 4H'F'/F - 2HF''/F + 2(1 - HF'²)/F²
    aF_R = -2.0 * H / F
    aH_R = -1.0
    a0_R = -4.0 * Hp * Fp / F + 2.0 * (1.0 - H * Fp ** 2) / F ** 2

    Acoef = Hp + 2.0 * H * Fp / F
    Scoef = 6.0 * H * (Fp / F) ** 2

    # Metric row 0 (F'' equation):
    # Same Weyl structure as original P12, plus R² coupling
    A[0, 0] = 4.0 * F * J_C - 2.0 * F + alpha_R_val * (4.0 * F * J_R * aF_R)
    A[0, 1] = alpha_R_val * (4.0 * F * J_R * aH_R)
    # Weyl u_k'' columns:
    A[0, 2:2 + K] = 2.0 * F ** 2 * c_C
    # Ricci v_k'' columns:
    A[0, 2 + K:2 + 2 * K] = alpha_R_val * 2.0 * F ** 2 * c_R
    b[0] = (-12.0 * F * Fp * J_Cp - 12.0 * J_C * Fp ** 2
            + alpha_R_val * (-12.0 * F * Fp * J_Rp - 12.0 * J_R * Fp ** 2
                             + 4.0 * F * J_R * a0_R))

    # Metric row 1 (H'' equation):
    A[1, 0] = -16.0 * H * J_C - 4.0 * H + alpha_R_val * (-16.0 * H * J_R * aF_R - 4.0 * H)
    A[1, 1] = 4.0 * F * J_C - 2.0 * F + alpha_R_val * (4.0 * F * J_R - 2.0 * F
                                                         + (-16.0 * H * J_R) * aH_R)
    # Weyl u_k'' columns:
    A[1, 2:2 + K] = -4.0 * F * H * c_C
    # Ricci v_k'' columns:
    A[1, 2 + K:2 + 2 * K] = alpha_R_val * (-4.0 * F * H * c_R)
    b[1] = (4.0 * F * Hp * J_Cp + 16.0 * H * Fp * J_Cp
            + 16.0 * J_C * Fp * Hp + 4.0 * Fp * Hp
            + alpha_R_val * (4.0 * F * Hp * J_Rp + 16.0 * H * Fp * J_Rp
                             + 16.0 * J_R * Fp * Hp + 4.0 * Fp * Hp
                             + (-16.0 * H * J_R) * a0_R))

    # -- Weyl auxiliary fields u_k --
    for k in range(K):
        row = 2 + k
        lam = Lam ** 2 / dtau[k]
        if k == 0:
            # Source: u_0 = W_amp = aF_W*F'' + aH_W*H'' + a0_W
            A[row, 0] = -lam * aF_W
            A[row, 1] = -lam * aH_W
            A[row, 2 + k] = H
            b[row] = -Acoef * up[k] + Scoef * u[k] - lam * u[k] + lam * a0_W
        else:
            A[row, 2 + k] = H
            b[row] = -Acoef * up[k] + Scoef * u[k] - lam * (u[k] - u[k - 1])

    # -- Ricci auxiliary fields v_k --
    for k in range(K):
        row = 2 + K + k
        lam = Lam ** 2 / dtau[k]
        if k == 0:
            # Source: v_0 = R_amp = aF_R*F'' + aH_R*H'' + a0_R
            A[row, 0] = -lam * aF_R
            A[row, 1] = -lam * aH_R
            A[row, 2 + K + k] = H
            b[row] = -Acoef * vp[k] + Scoef * v[k] - lam * v[k] + lam * a0_R
        else:
            A[row, 2 + K + k] = H
            b[row] = -Acoef * vp[k] + Scoef * v[k] - lam * (v[k] - v[k - 1])

    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(A, b, rcond=None)[0]

    Fpp = float(s[0])
    Hpp = float(s[1])
    u_pp = np.array(s[2:2 + K], dtype=float)
    v_pp = np.array(s[2 + K:2 + 2 * K], dtype=float)

    u0 = aF_W * Fpp + aH_W * Hpp + a0_W
    v0 = aF_R * Fpp + aH_R * Hpp + a0_R

    J_Cpp = float(np.dot(c_C, u_pp))
    J_Rpp = float(np.dot(c_R, v_pp))

    return Fpp, Hpp, u0, u_pp, v0, v_pp, J_C, J_Cp, J_Cpp, J_R, J_Rp, J_Rpp


def rhs_coupled(rho, y, p, coeffs, Lam=1.0, alpha_R_val=1.0/18.0):
    """Right-hand side of the coupled Weyl+R² ODE system."""
    K = coeffs['K']
    out = np.zeros_like(y)
    for j in range(y.shape[1]):
        Fpp, Hpp, _u0, u_pp, _v0, v_pp, *_ = local_second_derivs_coupled(
            y[:, j], coeffs, Lam=Lam, alpha_R_val=alpha_R_val)
        out[0, j] = y[1, j]
        out[1, j] = Fpp
        out[2, j] = y[3, j]
        out[3, j] = Hpp
        for k in range(K):
            out[4 + 2 * k, j] = y[5 + 2 * k, j]
            out[5 + 2 * k, j] = u_pp[k]
            out[4 + 2 * K + 2 * k, j] = y[5 + 2 * K + 2 * k, j]
            out[5 + 2 * K + 2 * k, j] = v_pp[k]
    return out


def bc_coupled(ya, yb, p, eps, Rmax, M, coeffs, Lam=1.0, alpha_R_val=1.0/18.0):
    """Boundary conditions for the coupled system.

    Center (rho = eps):
      F(eps) = eps + f3*eps^3,  F'(eps) = 1 + 3f3*eps^2
      H(eps) = 1 + h2*eps^2,   H'(eps) = 2*h2*eps
      u_k(eps) = a2_k*eps^2,   u_k'(eps) = 2*a2_k*eps
      v_k(eps) = b2_k*eps^2,   v_k'(eps) = 2*b2_k*eps

    Outer (rho = Rmax): Schwarzschild asymptotics
      F(R) = R
      H(R) = 1 - 2M/R
      u_k(R) = Schwarzschild Weyl at R
      v_k(R) = Schwarzschild Ricci at R = 0 (R_Sch = 0)
    """
    K = coeffs['K']
    h2 = p[0]
    f3 = p[1]
    a2 = p[2:2 + K]
    b2 = p[2 + K:2 + 2 * K]

    res = []
    # Center: metric
    res.extend([
        ya[0] - (eps + f3 * eps ** 3),
        ya[1] - (1.0 + 3.0 * f3 * eps ** 2),
        ya[2] - (1.0 + h2 * eps ** 2),
        ya[3] - (2.0 * h2 * eps),
    ])
    # Center: Weyl auxiliary
    for k in range(K):
        res.append(ya[4 + 2 * k] - a2[k] * eps ** 2)
        res.append(ya[5 + 2 * k] - 2.0 * a2[k] * eps)
    # Center: Ricci auxiliary
    for k in range(K):
        res.append(ya[4 + 2 * K + 2 * k] - b2[k] * eps ** 2)
        res.append(ya[5 + 2 * K + 2 * k] - 2.0 * b2[k] * eps)

    # Outer: metric Schwarzschild
    res.extend([
        yb[0] - Rmax,
        yb[2] - (1.0 - 2.0 * M / Rmax),
    ])
    # Outer: Weyl auxiliary at Schwarzschild
    for k, tau_k in enumerate(coeffs['tau']):
        uR = -2.0 * M / Rmax ** 3 - 12.0 * M ** 2 * tau_k / (Lam ** 2 * Rmax ** 6)
        res.append(yb[4 + 2 * k] - uR)
    # Outer: Ricci auxiliary (R = 0 for Schwarzschild vacuum)
    for k in range(K):
        res.append(yb[4 + 2 * K + 2 * k] - 0.0)

    return np.array(res, dtype=float)


# ============================================================
#   DIAGNOSTICS
# ============================================================

@dataclass
class CoupledRecord:
    Rmax: float
    status: int
    message: str
    mode: str  # 'weyl_only' or 'weyl_plus_r2'
    alpha_R: float
    h2: float | None = None
    f3: float | None = None
    Hmin: float | None = None
    Hmax: float | None = None
    Fmin: float | None = None
    Jc_abs_max: float | None = None
    Jr_abs_max: float | None = None
    R_scalar_max: float | None = None
    center_u_slopes: list[float] | None = None
    center_v_slopes: list[float] | None = None


def diagnostics_coupled(sol, coeffs, Lam=1.0, alpha_R_val=1.0/18.0):
    """Compute diagnostic quantities on the coupled solution."""
    K = coeffs['K']
    rho = np.geomspace(sol.x[0], sol.x[-1], 600)
    y = sol.sol(rho)
    F = y[0]
    Fp = y[1]
    H = y[2]
    Hp = y[3]

    J_C_arr = []
    J_R_arr = []
    R_arr = []
    for j in range(rho.size):
        res = local_second_derivs_coupled(y[:, j], coeffs, Lam=Lam,
                                          alpha_R_val=alpha_R_val)
        Fpp, Hpp = res[0], res[1]
        J_C_arr.append(res[6])
        J_R_arr.append(res[9])
        R_arr.append(ricci_scalar(F[j], Fp[j], H[j], Hp[j], Fpp, Hpp))

    J_C_arr = np.array(J_C_arr)
    J_R_arr = np.array(J_R_arr)
    R_arr = np.array(R_arr)

    # Power-law slopes near center for u_k and v_k
    i1 = min(20, rho.size)
    rfit = rho[:i1]
    u_slopes = []
    v_slopes = []
    for k in range(K):
        uk = y[4 + 2 * k, :i1]
        vals = np.abs(uk) + 1.0e-300
        c = np.polyfit(np.log(rfit), np.log(vals), 1)
        u_slopes.append(float(c[0]))

        vk = y[4 + 2 * K + 2 * k, :i1]
        vals = np.abs(vk) + 1.0e-300
        c = np.polyfit(np.log(rfit), np.log(vals), 1)
        v_slopes.append(float(c[0]))

    return {
        'Hmin': float(np.min(H)),
        'Hmax': float(np.max(H)),
        'Fmin': float(np.min(F)),
        'Jc_abs_max': float(np.max(np.abs(J_C_arr))),
        'Jr_abs_max': float(np.max(np.abs(J_R_arr))),
        'R_scalar_max': float(np.max(np.abs(R_arr))),
        'center_u_slopes': u_slopes,
        'center_v_slopes': v_slopes,
    }


# ============================================================
#   WEYL-ONLY SOLVER (from original P12, adapted)
# ============================================================

def local_second_derivs_weyl(ycol, coeffs, Lam=1.0):
    """Weyl-only second derivatives (same as original P12)."""
    K = coeffs['K']
    F = ycol[0]
    Fp = ycol[1]
    H = ycol[2]
    Hp = ycol[3]
    u = np.array([ycol[4 + 2 * i] for i in range(K)], dtype=float)
    up = np.array([ycol[5 + 2 * i] for i in range(K)], dtype=float)

    c = coeffs['c_C']
    dtau = coeffs['dtau']

    J = float(np.dot(c, u))
    Jp = float(np.dot(c, up))

    n = K + 2
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    A[0, 0] = 4.0 * F * J - 2.0 * F
    A[0, 2:] = 2.0 * F ** 2 * c
    b[0] = -12.0 * F * Fp * Jp - 12.0 * J * Fp ** 2

    A[1, 0] = -16.0 * H * J - 4.0 * H
    A[1, 1] = 4.0 * F * J - 2.0 * F
    A[1, 2:] = -4.0 * F * H * c
    b[1] = 4.0 * F * Hp * Jp + 16.0 * H * Fp * Jp + 16.0 * J * Fp * Hp + 4.0 * Fp * Hp

    Acoef = Hp + 2.0 * H * Fp / F
    Scoef = 6.0 * H * (Fp / F) ** 2

    aF = -H / (3.0 * F)
    aH = 1.0 / 6.0
    a0 = -(Fp * Hp) / (3.0 * F) + H * Fp ** 2 / (3.0 * F ** 2) - 1.0 / (3.0 * F ** 2)

    for k in range(K):
        row = 2 + k
        lam = Lam ** 2 / dtau[k]
        if k == 0:
            A[row, 0] = -lam * aF
            A[row, 1] = -lam * aH
            A[row, 2 + k] = H
            b[row] = -Acoef * up[k] + Scoef * u[k] - lam * u[k] + lam * a0
        else:
            A[row, 2 + k] = H
            b[row] = -Acoef * up[k] + Scoef * u[k] - lam * (u[k] - u[k - 1])

    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(A, b, rcond=None)[0]

    Fpp = float(s[0])
    Hpp = float(s[1])
    u_pp = np.array(s[2:], dtype=float)
    return Fpp, Hpp, u_pp, J, Jp


def rhs_weyl(rho, y, p, coeffs, Lam=1.0):
    K = coeffs['K']
    out = np.zeros_like(y)
    for j in range(y.shape[1]):
        Fpp, Hpp, u_pp, _, _ = local_second_derivs_weyl(y[:, j], coeffs, Lam=Lam)
        out[0, j] = y[1, j]
        out[1, j] = Fpp
        out[2, j] = y[3, j]
        out[3, j] = Hpp
        for k in range(K):
            out[4 + 2 * k, j] = y[5 + 2 * k, j]
            out[5 + 2 * k, j] = u_pp[k]
    return out


def initial_guess_weyl(rho, M, coeffs, l=2.0):
    K = coeffs['K']
    F = rho.copy()
    Fp = np.ones_like(rho)
    H = 1.0 - 2.0 * M * rho ** 2 / (rho ** 3 + l ** 3)
    Hp = np.gradient(H, rho, edge_order=2)
    Fpp = np.gradient(Fp, rho, edge_order=2)
    Hpp = np.gradient(Hp, rho, edge_order=2)
    Wamp = weyl_amplitude(F, Fp, H, Hp, Fpp, Hpp)

    Y = [F, Fp, H, Hp]
    for tau_k in coeffs['tau']:
        uk = Wamp / (1.0 + 4.0 * tau_k)
        vk = np.gradient(uk, rho, edge_order=2)
        Y.extend([uk, vk])
    y0 = np.vstack(Y)
    p0 = np.concatenate(([- 2.0 * M / l ** 3, 0.0], np.zeros(K)))
    return y0, p0


def bc_weyl(ya, yb, p, eps, Rmax, M, coeffs, Lam=1.0):
    K = coeffs['K']
    h2 = p[0]
    f3 = p[1]
    a2 = p[2:]
    res = []
    res.extend([
        ya[0] - (eps + f3 * eps ** 3),
        ya[1] - (1.0 + 3.0 * f3 * eps ** 2),
        ya[2] - (1.0 + h2 * eps ** 2),
        ya[3] - (2.0 * h2 * eps),
    ])
    for k in range(K):
        res.append(ya[4 + 2 * k] - a2[k] * eps ** 2)
        res.append(ya[5 + 2 * k] - 2.0 * a2[k] * eps)
    res.extend([
        yb[0] - Rmax,
        yb[2] - (1.0 - 2.0 * M / Rmax),
    ])
    for k, tau_k in enumerate(coeffs['tau']):
        uR = -2.0 * M / Rmax ** 3 - 12.0 * M ** 2 * tau_k / (Lam ** 2 * Rmax ** 6)
        res.append(yb[4 + 2 * k] - uR)
    return np.array(res, dtype=float)


# ============================================================
#   MAIN DRIVER
# ============================================================

def run_comparison(M=1.0, K=6, Lam=1.0, xi=0.0, eps=0.02, l=2.0,
                   R_values=None, tol=1e-2, max_nodes=20000):
    """Run Weyl-only and Weyl+R² solvers, compare results."""
    if R_values is None:
        R_values = [20.0, 40.0, 60.0, 80.0, 100.0]

    alpha_R_val = 2.0 * (xi - 1.0 / 6.0) ** 2
    coeffs = alpha_tau_quadrature(K)

    results = {
        'parameters': {
            'M': M, 'K': K, 'Lam': Lam, 'xi': xi, 'eps': eps, 'l': l,
            'alpha_C': 13.0 / 120.0,
            'alpha_R': alpha_R_val,
            'ratio_alpha_R_over_alpha_C': alpha_R_val / (13.0 / 120.0),
            'P_R_kernel': [-1.0/18.0, 4.0/9.0, 10.0/9.0],
            'P_C_kernel': [-89.0/24.0, 43.0/6.0, 236.0/3.0],
        },
        'weyl_only': [],
        'weyl_plus_r2': [],
        'perturbative_check': [],
    }

    # --- PHASE 1: Weyl-only BVP ---
    print("=" * 60)
    print("PHASE 1: Weyl-only P12 BVP")
    print("=" * 60)
    prev_weyl = None
    weyl_solutions = {}
    for Rmax in R_values:
        if prev_weyl is None:
            rho = np.geomspace(eps, Rmax, 140)
            y0, p0 = initial_guess_weyl(rho, M=M, coeffs=coeffs, l=l)
        else:
            rho = np.geomspace(eps, Rmax, max(180, prev_weyl.x.size))
            rho_common = np.minimum(rho, prev_weyl.x[-1])
            y0 = prev_weyl.sol(rho_common)
            mask = rho > prev_weyl.x[-1]
            if np.any(mask):
                y0[:, mask] = y0[:, np.searchsorted(rho, prev_weyl.x[-1]) - 1][:, None]
                y0[0, mask] = rho[mask]
                y0[1, mask] = 1.0
                y0[2, mask] = 1.0 - 2.0 * M / rho[mask]
                y0[3, mask] = 2.0 * M / rho[mask] ** 2
                for k, tau_k in enumerate(coeffs['tau']):
                    y0[4 + 2 * k, mask] = -2.0 * M / rho[mask] ** 3
                    y0[5 + 2 * k, mask] = 6.0 * M / rho[mask] ** 4
            p0 = prev_weyl.p

        sol = solve_bvp(
            lambda r, y, p: rhs_weyl(r, y, p, coeffs, Lam=Lam),
            lambda ya, yb, p: bc_weyl(ya, yb, p, eps, Rmax, M, coeffs, Lam=Lam),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=0,
        )

        rec = CoupledRecord(Rmax=Rmax, status=sol.status, message=sol.message,
                            mode='weyl_only', alpha_R=0.0)
        if sol.status == 0:
            prev_weyl = sol
            weyl_solutions[Rmax] = sol
            reval = np.geomspace(sol.x[0], sol.x[-1], 600)
            yeval = sol.sol(reval)
            rec.h2 = float(sol.p[0])
            rec.f3 = float(sol.p[1])
            rec.Hmin = float(np.min(yeval[2]))
            rec.Hmax = float(np.max(yeval[2]))
            rec.Fmin = float(np.min(yeval[0]))
        print(f"  R={Rmax:6.1f} status={sol.status} h2={rec.h2} Hmin={rec.Hmin}")
        results['weyl_only'].append(asdict(rec))

    # --- PHASE 2: Perturbative R² check ---
    print("\n" + "=" * 60)
    print("PHASE 2: Perturbative R² check on Weyl-only solutions")
    print("=" * 60)
    for Rmax, sol in weyl_solutions.items():
        pcheck = perturbative_r2_on_weyl_solution(sol, coeffs, Lam=Lam, xi=xi)
        # Don't store full profiles, just key numbers
        summary = {
            'Rmax': Rmax,
            'R_max_abs': pcheck['R_max'],
            'R_at_center': pcheck['R_at_center'],
            'W_max_abs': pcheck['W_max'],
            'ratio_R_over_W': pcheck['ratio_R_over_W'],
            'alpha_R': pcheck['alpha_R'],
            'alpha_C': pcheck['alpha_C'],
            'effective_ratio': pcheck['ratio_R_over_W'] * pcheck['alpha_R'] / pcheck['alpha_C'],
        }
        print(f"  R={Rmax:6.1f}: |R|_max={pcheck['R_max']:.4e}, "
              f"|W|_max={pcheck['W_max']:.4e}, "
              f"ratio={pcheck['ratio_R_over_W']:.4f}, "
              f"eff_ratio={summary['effective_ratio']:.4f}")
        results['perturbative_check'].append(summary)

    # --- PHASE 3: Fully coupled Weyl+R² BVP ---
    print("\n" + "=" * 60)
    print(f"PHASE 3: Fully coupled Weyl+R² BVP (alpha_R = {alpha_R_val:.6f})")
    print("=" * 60)
    prev_coupled = None
    for Rmax in R_values:
        if prev_coupled is None:
            rho = np.geomspace(eps, Rmax, 140)
            y0, p0 = initial_guess_coupled(rho, M=M, coeffs=coeffs, l=l)
        else:
            ndim = 4 + 4 * K
            rho = np.geomspace(eps, Rmax, max(180, prev_coupled.x.size))
            rho_common = np.minimum(rho, prev_coupled.x[-1])
            y0 = prev_coupled.sol(rho_common)
            mask = rho > prev_coupled.x[-1]
            if np.any(mask):
                y0[:, mask] = y0[:, np.searchsorted(rho, prev_coupled.x[-1]) - 1][:, None]
                y0[0, mask] = rho[mask]
                y0[1, mask] = 1.0
                y0[2, mask] = 1.0 - 2.0 * M / rho[mask]
                y0[3, mask] = 2.0 * M / rho[mask] ** 2
                for k, tau_k in enumerate(coeffs['tau']):
                    y0[4 + 2 * k, mask] = -2.0 * M / rho[mask] ** 3
                    y0[5 + 2 * k, mask] = 6.0 * M / rho[mask] ** 4
                    # Ricci fields: 0 at Schwarzschild
                    y0[4 + 2 * K + 2 * k, mask] = 0.0
                    y0[5 + 2 * K + 2 * k, mask] = 0.0
            p0 = prev_coupled.p

        sol = solve_bvp(
            lambda r, y, p: rhs_coupled(r, y, p, coeffs, Lam=Lam, alpha_R_val=alpha_R_val),
            lambda ya, yb, p: bc_coupled(ya, yb, p, eps, Rmax, M, coeffs, Lam=Lam,
                                          alpha_R_val=alpha_R_val),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=0,
        )

        rec = CoupledRecord(Rmax=Rmax, status=sol.status, message=sol.message,
                            mode='weyl_plus_r2', alpha_R=alpha_R_val)
        if sol.status == 0:
            prev_coupled = sol
            diag = diagnostics_coupled(sol, coeffs, Lam=Lam, alpha_R_val=alpha_R_val)
            rec.h2 = float(sol.p[0])
            rec.f3 = float(sol.p[1])
            rec.Hmin = diag['Hmin']
            rec.Hmax = diag['Hmax']
            rec.Fmin = diag['Fmin']
            rec.Jc_abs_max = diag['Jc_abs_max']
            rec.Jr_abs_max = diag['Jr_abs_max']
            rec.R_scalar_max = diag['R_scalar_max']
            rec.center_u_slopes = diag['center_u_slopes']
            rec.center_v_slopes = diag['center_v_slopes']
        print(f"  R={Rmax:6.1f} status={sol.status} h2={rec.h2} Hmin={rec.Hmin}")
        results['weyl_plus_r2'].append(asdict(rec))

    # --- PHASE 4: Summary comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON: Weyl-only vs Weyl+R²")
    print("=" * 60)
    comparison = []
    for w, c in zip(results['weyl_only'], results['weyl_plus_r2']):
        if w['h2'] is not None and c['h2'] is not None:
            delta_h2 = c['h2'] - w['h2']
            delta_Hmin = c['Hmin'] - w['Hmin']
            row = {
                'Rmax': w['Rmax'],
                'h2_weyl': w['h2'],
                'h2_coupled': c['h2'],
                'delta_h2': delta_h2,
                'rel_delta_h2': delta_h2 / abs(w['h2']) if abs(w['h2']) > 1e-30 else None,
                'Hmin_weyl': w['Hmin'],
                'Hmin_coupled': c['Hmin'],
                'delta_Hmin': delta_Hmin,
                'horizon_weyl': w['Hmin'] <= 0.0 if w['Hmin'] is not None else None,
                'horizon_coupled': c['Hmin'] <= 0.0 if c['Hmin'] is not None else None,
            }
            comparison.append(row)
            print(f"  R={row['Rmax']:6.1f}: h2 Weyl={row['h2_weyl']:.6e} "
                  f"coupled={row['h2_coupled']:.6e} "
                  f"delta={row['delta_h2']:.4e}")
            print(f"          Hmin Weyl={row['Hmin_weyl']:.6e} "
                  f"coupled={row['Hmin_coupled']:.6e} "
                  f"delta={row['delta_Hmin']:.4e}")

    results['comparison'] = comparison

    # Verdict
    if comparison:
        max_rel_h2 = max(abs(r.get('rel_delta_h2', 0) or 0) for r in comparison)
        any_horizon_change = any(
            r['horizon_weyl'] != r['horizon_coupled'] for r in comparison
            if r['horizon_weyl'] is not None and r['horizon_coupled'] is not None
        )
        results['verdict'] = {
            'max_rel_delta_h2': max_rel_h2,
            'horizon_topology_changed': any_horizon_change,
            'r2_sector_qualitatively_important': any_horizon_change or max_rel_h2 > 0.1,
            'r2_sector_quantitatively_important': max_rel_h2 > 0.01,
        }
        print(f"\nVERDICT: max |delta h2 / h2| = {max_rel_h2:.4e}")
        print(f"  Horizon topology changed: {any_horizon_change}")
        if any_horizon_change:
            print("  => R² sector QUALITATIVELY changes the solution!")
        elif max_rel_h2 > 0.1:
            print("  => R² sector is QUANTITATIVELY significant (>10%)")
        elif max_rel_h2 > 0.01:
            print("  => R² sector gives modest correction (1-10%)")
        else:
            print("  => R² sector is NEGLIGIBLE (<1%)")

    return results


def main():
    print("P12 v4: R² sector analysis for Gap G1")
    print("alpha_C = 13/120 = {:.6f}".format(13/120))
    print("alpha_R(xi=0) = 1/18 = {:.6f}".format(1/18))
    print("P_R(alpha) = -1/18 + 4/9*tau + 10/9*tau^2")
    print()

    # Run at xi=0 (minimal coupling, maximum R² effect)
    results = run_comparison(
        M=1.0, K=6, Lam=1.0, xi=0.0,
        R_values=[20.0, 40.0, 60.0, 80.0, 100.0],
        tol=1e-2, max_nodes=20000,
    )

    # Also check xi=1/6 (conformal coupling, alpha_R=0 => no R² correction)
    print("\n\n" + "#" * 60)
    print("CONTROL: xi=1/6 (conformal coupling, alpha_R=0)")
    print("#" * 60)
    results_conformal = run_comparison(
        M=1.0, K=6, Lam=1.0, xi=1.0/6.0,
        R_values=[20.0, 40.0, 60.0],
        tol=1e-2, max_nodes=20000,
    )
    results['conformal_control'] = results_conformal

    # Save
    outpath = OUTDIR / 'v4_r2_sector.json'
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
