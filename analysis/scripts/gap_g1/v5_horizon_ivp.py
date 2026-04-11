"""
v5_horizon_ivp.py — Horizon-to-center IVP with fakeon-selected BCs.

CRITICAL QUESTION: Can the coupled P12 system connect a HORIZON (H=0)
to a REGULAR CENTER (H→1, m→0) when fakeon BCs are imposed?

Previous results:
  Stage 1 (fixed bg):  n_eff ≈ 0.55 (= 3 - sqrt(6)), J/W → 0
  Stage 2 (coupled):   H_end < 0, n_eff wild — horizon crossing mishandled

This script fixes the horizon degeneracy via Frobenius analysis and runs
four tests:
  Test 1: Single auxiliary field u(r) on fixed Schwarzschild — Frobenius IVP
  Test 2: Full K-layer P12 system on fixed Schwarzschild
  Test 3: Coupled (m, u_k) system with linearized backreaction
  Test 4: BVP horizon ↔ center (strictest test)

Parameters: M = 1, Lam = 1, K = 6.

David Alfyorov, SCT Theory project, 2026-04-06
"""

from __future__ import annotations
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp, solve_bvp

OUTDIR = Path(__file__).resolve().parent.parent.parent / "results" / "gap_g1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Physical parameters ─────────────────────────────────────────────
M_BH   = 1.0
LAM    = 1.0
ALPHA_C = 13.0 / 120.0
K_LAYERS = 6
R_H    = 2.0 * M_BH   # horizon radius
KAPPA  = 1.0 / (4 * M_BH)  # surface gravity = 1/(4M)
SQRT6  = np.sqrt(6.0)
M2SQ   = 60.0 / 13.0 * LAM**2  # effective spin-2 mass squared = Lam^2 * 60/13


# ── Alpha-kernel quadrature ─────────────────────────────────────────
def build_layers(K: int = K_LAYERS) -> dict:
    """Build tau-layers from the symmetric P(alpha) kernel."""
    x, w = leggauss(K)
    alpha = 0.25 * (x + 1.0)        # [0, 1/2]
    w_q = 0.25 * w * 2.0            # Jacobian x symmetry
    tau = alpha * (1.0 - alpha)
    P = -89.0/24.0 + (43.0/6.0)*tau + (236.0/3.0)*tau**2

    idx = np.argsort(tau)
    tau, w_q, P = tau[idx], w_q[idx], P[idx]
    c = w_q * P
    dtau = np.diff(np.concatenate(([0.0], tau)))

    integral = float(np.dot(w_q, P))
    print(f"  P(alpha) quadrature: integral = {integral:.8f} (13/120 = {13/120:.8f})")
    return dict(tau=tau, c=c, dtau=dtau, P=P, w=w_q, K=K)


# ── Schwarzschild background functions ──────────────────────────────
def sch_H(r):   return 1.0 - R_H / r
def sch_Hp(r):  return R_H / r**2
def sch_Hpp(r): return -2 * R_H / r**3
def sch_W(r):   return -R_H / (2 * r**3)   # = -2M/r^3, Weyl amplitude
def sch_Wp(r):  return 3 * R_H / (2 * r**4)  # = 6M/r^4


# ═══════════════════════════════════════════════════════════════════
#  TEST 1: Single auxiliary field — Frobenius IVP
# ═══════════════════════════════════════════════════════════════════

def test1_single_field():
    """
    Single scalar auxiliary u(r) on fixed Schwarzschild, with effective
    mass m2sq = 60/13 * Lam^2.

    ODE: H u'' + (H' + 2H/r) u' + (m2sq - 6H/r^2) u = 0

    At horizon (H=0): this is a regular singular point.
    Frobenius analysis:
      Let rho = r - r_h. Then H ~ kappa * rho + O(rho^2).
      The indicial equation gives s=0 (regular) and s depends on details.

      The regular solution (fakeon-selected): u = u_h + u_h' rho + ...
      At H=0: the constraint gives u_h' in terms of u_h:
        H'(r_h) u'(r_h) + m2sq u(r_h) = 0
        => u'(r_h) = -m2sq * u(r_h) / H'(r_h)
    """
    print("\n" + "="*70)
    print("TEST 1: Single auxiliary field on Schwarzschild — Frobenius IVP")
    print("="*70)

    m2sq = M2SQ
    print(f"  m2sq = {m2sq:.6f}, sqrt(m2sq) = {np.sqrt(m2sq):.6f}")
    print(f"  r_h = {R_H:.4f}, kappa = {KAPPA:.4f}")

    # Frobenius initial data at r = r_h - delta
    delta = 1e-4
    r_start = R_H - delta

    # Value at horizon: Schwarzschild Weyl amplitude
    u_h = sch_W(R_H)  # = -1/4
    print(f"  u(r_h) = W(r_h) = {u_h:.6f}")

    # Constraint at H=0 (from the ODE with H=0):
    # H'(r_h) u'(r_h) + (m2sq - 0) u(r_h) = 0
    # Wait: the full coefficient of u is (m2sq - 6H/r^2). At H=0: just m2sq.
    # And the u' coefficient is (H' + 2H/r). At H=0: just H'.
    # So: H'_h u'(r_h) + m2sq u(r_h) = 0
    # => u'(r_h) = -m2sq u_h / H'_h

    Hp_h = sch_Hp(R_H)  # = 1/2 for M=1
    up_h = -m2sq * u_h / Hp_h
    print(f"  u'(r_h) = -m2sq * u_h / H'_h = {up_h:.6f}")
    print(f"  (H'_h = {Hp_h:.6f})")

    # Second derivative at horizon (from expanding ODE to next order):
    # H u'' ~ kappa * rho * u''_h gives O(rho), but we also have
    # O(1) terms from H'' u' and from d/dr of the other terms.
    # For the Frobenius: u ~ u_h + up_h * rho + (1/2) u''_h * rho^2
    # The ODE at O(rho^0):
    #   H'_h * up_h + m2sq * u_h = 0  (already satisfied)
    # At O(rho^1):
    #   H'_h * u''_h + (H''_h + 2H'_h/r_h) * up_h + (m2sq - 6H'_h/r_h^2) * u_h
    #   + kappa * u''_h [from H * u'' at next order] = 0
    # Wait, need to be more careful.

    # Actually, let's just use the constraint at H=0 to get u'(r_h),
    # then take a Taylor step to r_start = r_h - delta.
    # u(r_start) = u_h + up_h * (-delta) + O(delta^2)
    # u'(r_start) = up_h + u''_h * (-delta) + ...

    # For u''_h: evaluate the ODE at r slightly above r_h where H != 0
    r_eps = R_H + 1e-6
    H_eps = sch_H(r_eps)
    Hp_eps = sch_Hp(r_eps)
    u_eps = u_h + up_h * 1e-6
    up_eps = up_h

    coeff_up = Hp_eps + 2.0 * H_eps / r_eps
    coeff_u = m2sq - 6.0 * H_eps / r_eps**2
    upp_eps = -(coeff_up * up_eps + coeff_u * u_eps) / H_eps
    print(f"  u''(r_h) ~ {upp_eps:.6f}")

    # Initial data at r_start (inside horizon)
    u0 = u_h - up_h * delta + 0.5 * upp_eps * delta**2
    up0 = up_h - upp_eps * delta

    # Also compute for H < 0 region
    H_start = sch_H(r_start)
    print(f"  H(r_start) = {H_start:.6e} (should be < 0)")
    print(f"  u(r_start) = {u0:.6f}, u'(r_start) = {up0:.6f}")

    # ODE for r < r_h (H < 0 but the equation is still well-defined)
    def single_ode(r, y):
        u, up = y
        if r < 1e-14:
            return [0.0, 0.0]
        H = sch_H(r)
        Hp = sch_Hp(r)
        coeff_up = Hp + 2.0 * H / r
        coeff_u = m2sq - 6.0 * H / r**2
        if abs(H) < 1e-15:
            return [up, 0.0]
        upp = -(coeff_up * up + coeff_u * u) / H
        return [up, upp]

    # Integrate inward: r_start → r_end
    r_end = 0.01 * R_H  # r = 0.02
    n_pts = 20000
    r_eval = np.linspace(r_start, r_end, n_pts)

    sol = solve_ivp(single_ode, [r_start, r_end], [u0, up0],
                    method='DOP853', t_eval=r_eval,
                    rtol=1e-12, atol=1e-14, max_step=delta)

    r = sol.t
    u = sol.y[0]
    up = sol.y[1]

    print(f"\n  Integration: {sol.status} ({sol.message}), {len(r)} points")
    print(f"  r range: [{r[-1]:.4e}, {r[0]:.4e}]")

    # Analyze: power-law behavior u ~ r^s near center
    # On Schwarzschild: W = -2M/r^3, so the "trivial" solution has u ~ r^{-3}.
    # The Frobenius exponents of the homogeneous eq at r=0:
    # s(s-1) + s - 6 = 0 => s^2 - 6 = 0 => s = ±sqrt(6)
    # So u_hom = A r^{+sqrt6} + B r^{-sqrt6}
    # The particular solution (Schwarzschild Weyl) gives u_part ~ r^{-3}.
    # The general solution: u = u_part + A r^{sqrt6} + B r^{-sqrt6}

    # Compute effective exponent
    ln_r = np.log(np.maximum(r, 1e-300))
    ln_u = np.log(np.maximum(np.abs(u), 1e-300))

    # Power-law fit in inner region
    inner = r < 0.2 * R_H
    if np.sum(inner) > 20:
        p = np.polyfit(ln_r[inner], ln_u[inner], 1)
        s_eff = p[0]
    else:
        s_eff = np.nan

    # n_eff for E^2 ~ r^{-6+2n}: if u ~ r^s, then u^2 ~ r^{2s},
    # and n_eff = (2s + 6)/2 = s + 3
    n_eff_inner = s_eff + 3

    print(f"\n  Inner region u ~ r^s:")
    print(f"    s_eff = {s_eff:.4f}")
    print(f"    n_eff = s_eff + 3 = {n_eff_inner:.4f}")
    print(f"    Expected: s = -3 (Schwarzschild), s = -sqrt(6) ≈ {-SQRT6:.4f} (singular hom)")
    print(f"    If n_eff ≈ 0: Schwarzschild singularity preserved")
    print(f"    If n_eff ≈ {3 - SQRT6:.4f}: subdominant Frobenius (singular)")
    print(f"    If n_eff = 3: regular center!")

    # Profile table
    print(f"\n  {'r/r_h':>8s} {'r':>10s} {'u':>14s} {'u_Sch':>14s} {'u/W_Sch':>10s} {'H(r)':>10s}")
    print("  " + "-"*70)
    for frac in [0.99, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        r_t = R_H * frac
        if r_t < r[-1] or r_t > r[0]:
            continue
        idx = np.argmin(np.abs(r - r_t))
        W_sch = sch_W(r[idx])
        ratio = u[idx] / W_sch if abs(W_sch) > 1e-300 else np.nan
        print(f"  {frac:8.4f} {r[idx]:10.4e} {u[idx]:14.6e} {W_sch:14.6e} {ratio:10.6f} {sch_H(r[idx]):10.6f}")

    return dict(
        test="single_field_frobenius",
        s_eff=float(s_eff),
        n_eff=float(n_eff_inner),
        u_end=float(u[-1]),
        r_end=float(r[-1]),
        H_end=float(sch_H(r[-1])),
        status=sol.status,
        n_points=len(r),
    )


# ═══════════════════════════════════════════════════════════════════
#  TEST 2: Full K-layer P12 on fixed Schwarzschild
# ═══════════════════════════════════════════════════════════════════

def test2_full_layers():
    """
    K-layer localized system on fixed Schwarzschild.

    For each k: H u_k'' + A u_k' - S u_k = lam_k (u_k - u_{k-1})
    where u_{-1} = W (bare Weyl), lam_k = Lam^2 / dtau_k.

    At H(r_h) = 0: the constraint gives
      u_k'(r_h) = [lam_k (u_k(r_h) - u_{k-1}(r_h)) + S_h u_k(r_h)] / A_h
    where A_h = H'(r_h) + 0 = H'(r_h), S_h = 0.
    So: u_k'(r_h) = lam_k (u_k(r_h) - u_{k-1}(r_h)) / H'(r_h)
    """
    print("\n" + "="*70)
    print("TEST 2: Full K-layer P12 system on Schwarzschild")
    print("="*70)

    layers = build_layers(K_LAYERS)
    K = layers['K']
    tau = layers['tau']
    c = layers['c']
    dtau = layers['dtau']

    # Frobenius initial data at r_h - delta
    delta = 1e-4
    r_start = R_H - delta
    Hp_h = sch_Hp(R_H)
    W_h = sch_W(R_H)  # = -1/4

    # u_k at horizon: stationary approximation
    u_h = np.zeros(K)
    v_h = np.zeros(K)

    for k in range(K):
        # Stationary: u_k = W / (1 + 4 tau_k) for the uncoupled limit
        u_h[k] = W_h / (1.0 + 4.0 * tau[k])

    # Compute v_k(r_h) from the constraint at H=0:
    # H'_h * v_k = lam_k * (u_k - u_{k-1})
    # where u_{-1} = W_h for k=0, u_{k-1} for k>0
    for k in range(K):
        lam_k = LAM**2 / dtau[k]
        if k == 0:
            u_prev = W_h
        else:
            u_prev = u_h[k-1]
        v_h[k] = lam_k * (u_h[k] - u_prev) / Hp_h

    print(f"  IC at r_h = {R_H}:")
    for k in range(K):
        print(f"    u_{k}(r_h) = {u_h[k]:.6e}, v_{k}(r_h) = {v_h[k]:.6e}, tau={tau[k]:.6f}")

    # Taylor step to r_start
    y0 = np.zeros(2*K)
    for k in range(K):
        y0[2*k] = u_h[k] - v_h[k] * delta  # u_k(r_h - delta)
        y0[2*k+1] = v_h[k]  # first-order approx

    def layers_rhs(r, y):
        """RHS for the K-layer system on fixed Schwarzschild."""
        if r < 1e-14:
            return np.zeros(2*K)

        H = sch_H(r)
        Hp = sch_Hp(r)
        W = sch_W(r)

        A_coeff = Hp + 2.0 * H / r
        S_coeff = 6.0 * H / r**2

        dy = np.zeros(2*K)
        for k in range(K):
            uk = y[2*k]
            vk = y[2*k+1]
            lam_k = LAM**2 / dtau[k]

            if k == 0:
                u_prev = W
            else:
                u_prev = y[2*(k-1)]

            # H u'' + A u' - S u = lam_k (u - u_prev)
            # u'' = [lam_k(u - u_prev) - A u' + S u] / H
            if abs(H) < 1e-14:
                # Constraint: u' = [lam_k(u - u_prev) + S u] / A
                # u'' indeterminate; use L'Hopital
                upp = 0.0
            else:
                upp = (lam_k * (uk - u_prev) - A_coeff * vk + S_coeff * uk) / H

            dy[2*k] = vk
            dy[2*k+1] = upp

        return dy

    # Integrate inward
    r_end = 0.01 * R_H
    n_pts = 20000
    r_eval = np.linspace(r_start, r_end, n_pts)

    sol = solve_ivp(layers_rhs, [r_start, r_end], y0,
                    method='DOP853', t_eval=r_eval,
                    rtol=1e-11, atol=1e-13, max_step=delta)

    r = sol.t
    print(f"\n  Integration: status={sol.status}, {len(r)} points")

    # Compute J(r) = sum c_k u_k
    J = np.zeros(len(r))
    for k in range(K):
        J += c[k] * sol.y[2*k]

    W_arr = np.array([sch_W(ri) for ri in r])
    ratio = J / W_arr

    # Power-law fit for J near center
    inner = r < 0.2 * R_H
    if np.sum(inner) > 20:
        ln_r = np.log(r[inner])
        ln_J = np.log(np.maximum(np.abs(J[inner]), 1e-300))
        p = np.polyfit(ln_r, ln_J, 1)
        s_J = p[0]
    else:
        s_J = np.nan

    n_eff_J = s_J + 3  # since E^2 ~ J^2 ~ r^{2s}, n_eff = (2s+6)/2 = s+3

    print(f"\n  J(r) power law near center: J ~ r^{s_J:.4f}")
    print(f"  n_eff = {n_eff_J:.4f}")
    print(f"  J/W at r_end: {ratio[-1]:.6f}")

    # Profile
    print(f"\n  {'r/r_h':>8s} {'r':>10s} {'J':>14s} {'W_Sch':>14s} {'J/W':>10s}")
    print("  " + "-"*60)
    for frac in [0.99, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        r_t = R_H * frac
        if r_t < r[-1] or r_t > r[0]:
            continue
        idx = np.argmin(np.abs(r - r_t))
        print(f"  {frac:8.4f} {r[idx]:10.4e} {J[idx]:14.6e} {W_arr[idx]:14.6e} {ratio[idx]:10.6f}")

    return dict(
        test="full_layers_fixed_bg",
        K=K,
        s_J=float(s_J),
        n_eff_J=float(n_eff_J),
        J_end=float(J[-1]),
        JW_ratio_end=float(ratio[-1]),
        r_end=float(r[-1]),
        status=sol.status,
        n_points=len(r),
    )


# ═══════════════════════════════════════════════════════════════════
#  TEST 3: Coupled mass-function IVP with backreaction
# ═══════════════════════════════════════════════════════════════════

def test3_coupled_ivp():
    """
    Coupled system: (m, m', u_0, v_0, ..., u_{K-1}, v_{K-1}).

    The metric: H(r) = 1 - 2m(r)/r.
    The SCT field equation determines m'(r) including form-factor backreaction.

    Backreaction model (from NT-4b modified Einstein equation):
    In the mass-function formulation, the SCT correction to the tt-component gives:
      m'(r) = alpha_C * r^2 * [contribution from u_k dressing]

    The simplest self-consistent coupling:
    Define W_bare = (4r m' - 6m)/(6r^3) (Weyl from current metric, no m'' term)
    J = sum(c_k u_k) (dressed Weyl from auxiliary fields)

    The modified field equation gives:
      m'' = alpha_C * Lam^2 * 6r * (J - W_bare) + [GR terms]
    where the GR term for vacuum is m'' = 0 (Schwarzschild).

    BUT: this model has issues (self-consistency forces W=J algebraically).

    BETTER: treat the backreaction PERTURBATIVELY.
    m = M + alpha_C * delta_m, where delta_m satisfies:
      delta_m'' = f(r, u_k, delta_m, ...)
    with delta_m(r_h) = 0 (horizon stays at r_h to leading order).

    For the coupled system we instead solve the FULL self-consistent
    problem: m'' from requiring the EOM, with u_k evolving on the
    CURRENT metric H = 1 - 2m/r.
    """
    print("\n" + "="*70)
    print("TEST 3: Coupled mass-function IVP with backreaction")
    print("="*70)

    layers = build_layers(K_LAYERS)
    K = layers['K']
    tau = layers['tau']
    c = layers['c']
    dtau = layers['dtau']

    # State vector: y = [m, m', u_0, v_0, ..., u_{K-1}, v_{K-1}]
    # Size: 2 + 2K

    # IC at r_h - delta
    delta = 1e-4
    r_start = R_H - delta

    # At horizon: m(r_h) = M, m'(r_h) = 0 (Schwarzschild)
    m0 = M_BH
    mp0 = 0.0

    # u_k at horizon
    W_h = sch_W(R_H)
    Hp_h = sch_Hp(R_H)

    y0 = np.zeros(2 + 2*K)
    y0[0] = m0
    y0[1] = mp0 - 0.0 * delta  # m'(r_start) ~ 0 (Schwarzschild)

    for k in range(K):
        u_init = W_h / (1.0 + 4.0 * tau[k])
        lam_k = LAM**2 / dtau[k]
        if k == 0:
            u_prev = W_h
        else:
            u_prev = W_h / (1.0 + 4.0 * tau[k-1])
        v_init = lam_k * (u_init - u_prev) / Hp_h

        # Taylor step to r_start
        y0[2 + 2*k] = u_init - v_init * delta
        y0[3 + 2*k] = v_init

    print(f"  IC: m = {m0:.4f}, m' = {mp0:.6f}")
    print(f"  r_start = {r_start:.6f}, H(r_start) = {sch_H(r_start):.6e}")

    def coupled_rhs(r, y):
        if r < 1e-14:
            return np.zeros_like(y)

        m  = y[0]
        mp = y[1]

        # Current metric
        H  = 1.0 - 2.0 * m / r
        Hp = 2.0 * m / r**2 - 2.0 * mp / r

        # Bare Weyl (without m'' contribution)
        W_bare = (4.0 * r * mp - 6.0 * m) / (6.0 * r**3)

        # Extract u_k, v_k
        u = np.array([y[2 + 2*k] for k in range(K)])
        v = np.array([y[3 + 2*k] for k in range(K)])

        # J = dressed Weyl
        J = float(np.dot(c, u))

        # Backreaction: m'' from SCT field equation
        # The self-consistency: W_full = J means
        #   (-r^2 m'' + 4r m' - 6m)/(6r^3) = J
        #   => m'' = (4r m' - 6m - 6r^3 J) / r^2
        #
        # This is the exact self-consistency condition.
        mpp = (4.0 * r * mp - 6.0 * m - 6.0 * r**3 * J) / r**2

        # Full Weyl amplitude
        W_full = (-r**2 * mpp + 4 * r * mp - 6 * m) / (6 * r**3)
        # By construction W_full = J

        # Coefficients for u_k equation
        A_coeff = Hp + 2.0 * H / r
        S_coeff = 6.0 * H / r**2

        dy = np.zeros_like(y)
        dy[0] = mp
        dy[1] = mpp

        for k in range(K):
            uk = u[k]
            vk = v[k]
            lam_k = LAM**2 / dtau[k]

            if k == 0:
                u_prev = W_full  # source: current (self-consistent) Weyl
            else:
                u_prev = u[k-1]

            if abs(H) < 1e-13:
                upp = 0.0
            else:
                upp = (lam_k * (uk - u_prev) - A_coeff * vk + S_coeff * uk) / H

            dy[2 + 2*k] = vk
            dy[3 + 2*k] = upp

        return dy

    # Integrate inward with event detection for blow-up
    r_end = 0.01 * R_H
    n_pts = 20000
    r_eval = np.linspace(r_start, r_end, n_pts)

    def blowup_event(r, y):
        """Stop if m or u_k blow up."""
        return 1e6 - np.max(np.abs(y))
    blowup_event.terminal = True
    blowup_event.direction = -1

    sol = solve_ivp(coupled_rhs, [r_start, r_end], y0,
                    method='DOP853', t_eval=r_eval,
                    rtol=1e-10, atol=1e-12,
                    max_step=delta/2,
                    events=blowup_event)

    r = sol.t
    m_arr = sol.y[0]
    mp_arr = sol.y[1]
    H_arr = 1.0 - 2.0 * m_arr / r

    # J profile
    J_arr = np.zeros(len(r))
    for k in range(K):
        J_arr += c[k] * sol.y[2 + 2*k]

    # n_eff
    ln_r = np.log(np.maximum(r, 1e-300))
    J_sq = J_arr**2
    ln_J2 = np.log(np.maximum(np.abs(J_sq), 1e-300))
    slope = np.gradient(ln_J2, ln_r)
    n_eff = (slope + 6) / 2

    print(f"\n  Integration: status={sol.status}, {len(r)} points")
    print(f"\n  {'r/r_h':>8s} {'m(r)':>10s} {'m_prime':>12s} {'H(r)':>10s} {'J':>14s} {'n_eff':>8s}")
    print("  " + "-"*70)
    for frac in [0.99, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        r_t = R_H * frac
        if r_t < r[-1] or r_t > r[0]:
            continue
        idx = np.argmin(np.abs(r - r_t))
        print(f"  {frac:8.4f} {m_arr[idx]:10.6f} {mp_arr[idx]:12.4e} {H_arr[idx]:10.6f} "
              f"{J_arr[idx]:14.6e} {n_eff[idx]:8.3f}")

    # Key diagnostics
    print(f"\n  m(r_end)/M = {m_arr[-1]/M_BH:.6f}")
    print(f"  H(r_end) = {H_arr[-1]:.6f}")
    print(f"  {'REGULAR' if H_arr[-1] > 0.5 else 'NOT REGULAR'} center" +
          f" (H → 1 needed)")

    # Power-law fit for m near center
    inner = r < 0.2 * R_H
    if np.sum(inner) > 20:
        ln_m = np.log(np.maximum(np.abs(m_arr[inner]), 1e-300))
        p_m = np.polyfit(ln_r[inner], ln_m, 1)
        n_mass = p_m[0]
        print(f"  m(r) ~ r^{n_mass:.3f} near center")
        print(f"    n=0: Schwarzschild (m=const)")
        print(f"    n=1: SCT linear")
        print(f"    n=3: regular center")
    else:
        n_mass = np.nan

    return dict(
        test="coupled_ivp",
        K=K,
        m_end=float(m_arr[-1]),
        H_end=float(H_arr[-1]),
        J_end=float(J_arr[-1]),
        n_mass=float(n_mass),
        r_end=float(r[-1]),
        status=sol.status,
        n_points=len(r),
    )


# ═══════════════════════════════════════════════════════════════════
#  TEST 4: BVP horizon ↔ regular center
# ═══════════════════════════════════════════════════════════════════

def test4_bvp_horizon_center():
    """
    BVP connecting horizon (H=0) to regular center (H=1, m=0).

    Left boundary (r = eps, center):
      m(eps) = a3 * eps^3  (regular: m ~ r^3 from fakeon beta=0)
      m'(eps) = 3 * a3 * eps^2
      u_k(eps) = b_k * eps^2  (regular: u ~ r^2 from Frobenius +sqrt(6))

    Right boundary (r = r_h, horizon):
      m(r_h) = r_h / 2 = M
      H(r_h) = 1 - 2m(r_h)/r_h = 0 (automatic from m(r_h) = M)

    The constraint at H=0 fixes u_k'(r_h) in terms of u_k(r_h).

    Free parameters: a3 (center mass coefficient), b_k (center u_k values)
    """
    print("\n" + "="*70)
    print("TEST 4: BVP horizon to regular center")
    print("="*70)

    layers = build_layers(K_LAYERS)
    K = layers['K']
    tau = layers['tau']
    c = layers['c']
    dtau = layers['dtau']

    eps = 0.02
    r_max = R_H - 1e-4  # just inside horizon

    print(f"  Domain: [{eps}, {r_max}]")
    print(f"  K = {K} layers")

    # State: y = [m, m', u_0, v_0, ..., u_{K-1}, v_{K-1}]
    n_vars = 2 + 2*K

    def bvp_rhs(r, y, p):
        out = np.zeros_like(y)
        for j in range(y.shape[1]):
            rj = r[j]
            if rj < 1e-14:
                continue

            m  = y[0, j]
            mp = y[1, j]

            H  = 1.0 - 2.0 * m / rj
            Hp = 2.0 * m / rj**2 - 2.0 * mp / rj

            # J from u_k
            J = sum(c[k] * y[2 + 2*k, j] for k in range(K))

            # Self-consistency: m'' from W = J
            mpp = (4.0 * rj * mp - 6.0 * m - 6.0 * rj**3 * J) / rj**2

            W = (-rj**2 * mpp + 4 * rj * mp - 6 * m) / (6 * rj**3)

            A_coeff = Hp + 2.0 * H / rj
            S_coeff = 6.0 * H / rj**2

            out[0, j] = mp
            out[1, j] = mpp

            for k in range(K):
                uk = y[2 + 2*k, j]
                vk = y[3 + 2*k, j]
                lam_k = LAM**2 / dtau[k]
                u_prev = W if k == 0 else y[2 + 2*(k-1), j]

                if abs(H) < 1e-12:
                    upp = 0.0
                else:
                    upp = (lam_k * (uk - u_prev) - A_coeff * vk + S_coeff * uk) / H

                out[2 + 2*k, j] = vk
                out[3 + 2*k, j] = upp

        return out

    def bvp_bc(ya, yb, p):
        """
        ya = state at r = eps (center)
        yb = state at r = r_max (horizon)
        p = free parameters: [a3, b_0, b_1, ..., b_{K-1}]
        """
        a3 = p[0]
        b = p[1:]  # K free parameters

        res = []

        # Center BCs: m(eps) ~ a3 * eps^3, m'(eps) ~ 3 a3 eps^2
        res.append(ya[0] - a3 * eps**3)
        res.append(ya[1] - 3.0 * a3 * eps**2)

        # Center: u_k(eps) ~ b_k * eps^2 (regular, s = +sqrt(6) ≈ 2.449)
        # Actually for exact Frobenius: u ~ r^{sqrt(6)}, so:
        # u_k(eps) = b_k * eps^{sqrt(6)}
        # v_k(eps) = b_k * sqrt(6) * eps^{sqrt(6)-1}
        s_reg = SQRT6
        for k in range(K):
            res.append(ya[2 + 2*k] - b[k] * eps**s_reg)
            res.append(ya[3 + 2*k] - b[k] * s_reg * eps**(s_reg - 1))

        # Horizon BC: m(r_max) = M (so H = 0)
        res.append(yb[0] - M_BH)

        # Horizon: u_k constraint (H=0 algebraic relation)
        # H'_h v_k = lam_k (u_k - u_prev)
        Hp_h = sch_Hp(R_H)
        for k in range(K):
            lam_k = LAM**2 / dtau[k]
            if k == 0:
                # u_prev = W at horizon. W = (-r^2 m'' + 4r m' - 6m)/(6r^3)
                # At horizon m = M, and W = -2M/r_h^3 for Schwarzschild.
                # For the BVP solution, W at horizon is determined by the solution.
                # Use the algebraic relation: W = J at horizon (self-consistency)
                # So u_prev = J = sum(c_k u_k) evaluated at yb
                J_h = sum(c[kk] * yb[2 + 2*kk] for kk in range(K))
                u_prev = J_h
            else:
                u_prev = yb[2 + 2*(k-1)]
            # Constraint: Hp_h * v_k = lam_k * (u_k - u_prev)
            res.append(Hp_h * yb[3 + 2*k] - lam_k * (yb[2 + 2*k] - u_prev))

        return np.array(res, dtype=float)

    # Total BCs needed: 2K+2 (state) + K+1 (parameters) must match
    # BCs provided: 2 (center m) + 2K (center u_k) + 1 (horizon m) + K (horizon constraint) = 3+3K
    # State dimension: 2+2K. With K+1 free params, we need (2+2K) + (K+1) = 3K+3 BCs.
    n_params = K + 1  # a3, b_0, ..., b_{K-1}

    # Initial mesh and guess
    n_mesh = 200
    r_mesh = np.linspace(eps, r_max, n_mesh)

    y_guess = np.zeros((n_vars, n_mesh))
    for j, rj in enumerate(r_mesh):
        frac = (rj - eps) / (r_max - eps)
        # Interpolate: m goes from 0 to M
        y_guess[0, j] = M_BH * frac**3  # m ~ r^3 near center, M at horizon
        y_guess[1, j] = 3.0 * M_BH * frac**2 / (r_max - eps)

        W_guess = sch_W(max(rj, 0.1))
        for k in range(K):
            y_guess[2 + 2*k, j] = W_guess / (1.0 + 4.0 * tau[k])
            y_guess[3 + 2*k, j] = sch_Wp(max(rj, 0.1)) / (1.0 + 4.0 * tau[k])

    p_guess = np.zeros(n_params)
    p_guess[0] = M_BH / R_H**3  # a3 ~ M/r_h^3
    for k in range(K):
        p_guess[1 + k] = sch_W(1.0) / (1.0 + 4.0 * tau[k])

    print(f"  n_vars = {n_vars}, n_params = {n_params}")
    print(f"  Initial guess: a3 = {p_guess[0]:.6e}")

    # Try multiple tolerances
    results = {}
    for tol in [1e-2, 1e-3, 1e-4]:
        label = f"tol_{tol}"
        print(f"\n  Trying tol = {tol} ...")

        try:
            sol = solve_bvp(
                bvp_rhs,
                lambda ya, yb, p: bvp_bc(ya, yb, p),
                r_mesh,
                y_guess,
                p=p_guess,
                tol=tol,
                max_nodes=30000,
                verbose=0,
            )

            if sol.status == 0:
                m_sol = sol.sol(r_mesh)[0]
                H_sol = 1.0 - 2.0 * m_sol / r_mesh

                print(f"    CONVERGED! a3 = {sol.p[0]:.6e}")
                print(f"    m(eps) = {m_sol[0]:.6e} (should be ~ 0)")
                print(f"    m(r_max) = {m_sol[-1]:.6f} (should be {M_BH})")
                print(f"    H(eps) = {H_sol[0]:.6f} (should be ~ 1)")
                print(f"    H(r_max) = {H_sol[-1]:.6e} (should be ~ 0)")

                results[label] = {
                    "converged": True,
                    "a3": float(sol.p[0]),
                    "m_center": float(m_sol[0]),
                    "H_center": float(H_sol[0]),
                    "n_nodes": int(sol.x.size),
                }
                # Use converged solution as guess for next tol
                r_mesh = sol.x
                y_guess = sol.y
                p_guess = sol.p
            else:
                print(f"    FAILED: {sol.message}")
                results[label] = {"converged": False, "message": sol.message}

        except Exception as e:
            print(f"    ERROR: {e}")
            results[label] = {"converged": False, "error": str(e)}

    return dict(test="bvp_horizon_center", K=K, results=results)


# ═══════════════════════════════════════════════════════════════════
#  TEST 5: Connection matrix verification
# ═══════════════════════════════════════════════════════════════════

def test5_connection_matrix():
    """
    Compute the connection matrix M relating center and horizon bases.

    At center (r → 0): two independent solutions
      u_1 ~ r^{+sqrt(6)}  (regular, fakeon-selected)
      u_2 ~ r^{-sqrt(6)}  (singular, fakeon-killed)

    At horizon (r → r_h): two independent solutions
      v_1 ~ 1 + c_1 ρ + ...  (regular at horizon)
      v_2 ~ v_1 ln|ρ| + ... (log-divergent, fakeon-killed here too)

    Connection: (v_1, v_2) = M (u_1, u_2)

    If M_12 ≠ 0: regularity at center does NOT imply regularity at horizon.
    The fakeon must select at BOTH ends.

    KEY TEST: If we impose u_2 = 0 (regular center) and v_2 = 0 (regular horizon),
    is there a consistent solution? Only if M_11 M_22 - M_12 M_21 ≠ 0 AND
    the double constraint is satisfiable.

    Actually: u = α u_1 (center regular) should decompose as
    u = α(M_11 v_1 + M_21 v_2) at the horizon.
    For horizon regularity: M_21 = 0. If M_21 ≠ 0: double regularity impossible
    without fine-tuning.
    """
    print("\n" + "="*70)
    print("TEST 5: Connection matrix — center ↔ horizon")
    print("="*70)

    m2sq_values = [M2SQ, 1.0, 4.0, 10.0]
    eps_center = 1e-4
    delta_hor = 1e-4
    r_match = R_H - delta_hor  # match point (just inside horizon)

    results = {}

    for m2sq in m2sq_values:
        label = f"m2sq_{m2sq:.4f}"
        print(f"\n  m2sq = {m2sq:.4f} (Lam = {np.sqrt(m2sq * 13/60):.4f})")

        # ODE on Schwarzschild background
        def ode(r, y):
            u, up = y
            if r < 1e-15:
                return [0.0, 0.0]
            H = sch_H(r)
            Hp = sch_Hp(r)
            coeff_up = Hp + 2.0 * H / r
            coeff_u = m2sq - 6.0 * H / r**2
            if abs(H) < 1e-15:
                return [up, 0.0]
            upp = -(coeff_up * up + coeff_u * u) / H
            return [up, upp]

        # Integrate u_1 = r^{+sqrt6} from center to match point
        y0_u1 = [eps_center**SQRT6, SQRT6 * eps_center**(SQRT6 - 1)]
        sol_u1 = solve_ivp(ode, [eps_center, r_match], y0_u1,
                           method='DOP853', rtol=1e-13, atol=1e-15,
                           max_step=0.005)

        # Integrate u_2 = r^{-sqrt6} from center
        y0_u2 = [eps_center**(-SQRT6), -SQRT6 * eps_center**(-SQRT6 - 1)]
        sol_u2 = solve_ivp(ode, [eps_center, r_match], y0_u2,
                           method='DOP853', rtol=1e-13, atol=1e-15,
                           max_step=0.005)

        u1_end = sol_u1.y[0, -1]
        u1p_end = sol_u1.y[1, -1]
        u2_end = sol_u2.y[0, -1]
        u2p_end = sol_u2.y[1, -1]

        # Horizon Frobenius basis
        rho = -delta_hor  # r_match - R_H < 0 (inside horizon)
        c1 = -2 * m2sq
        v1 = 1 + c1 * rho
        v1p = c1
        ln_rho = np.log(abs(rho))
        v2 = v1 * ln_rho
        v2p = c1 * ln_rho + v1 / rho

        # Extraction: u_i = M_i1 v_1 + M_i2 v_2 at match point
        det_v = v1 * v2p - v2 * v1p
        M11 = (v2p * u1_end - v2 * u1p_end) / det_v
        M12 = (-v1p * u1_end + v1 * u1p_end) / det_v
        M21 = (v2p * u2_end - v2 * u2p_end) / det_v
        M22 = (-v1p * u2_end + v1 * u2p_end) / det_v

        det_M = M11 * M22 - M12 * M21
        ratio12 = abs(M12 / M11) if abs(M11) > 1e-300 else float('inf')
        ratio21 = abs(M21 / M22) if abs(M22) > 1e-300 else float('inf')

        print(f"    M = [[{M11:+.6e}, {M12:+.6e}],")
        print(f"         [{M21:+.6e}, {M22:+.6e}]]")
        print(f"    det(M) = {det_M:.6e}")
        print(f"    |M12/M11| = {ratio12:.6e}")
        print(f"    |M21/M22| = {ratio21:.6e}")

        # Interpretation:
        # u_1 (regular center) = M11 v_1 + M12 v_2
        # If M12 != 0: regular center generates log-divergence at horizon
        # The fakeon must kill BOTH u_2 (center) and v_2 (horizon)
        # This requires: the COMBINED system selects M12 = 0 effectively
        # OR: the BH interpretation fails
        if abs(M12) > 1e-10 * abs(M11):
            verdict = "M12 != 0: regular center GENERATES horizon log-divergence"
        else:
            verdict = "M12 ~ 0: regular center is compatible with smooth horizon"

        print(f"    VERDICT: {verdict}")

        results[label] = {
            "m2sq": float(m2sq),
            "M11": float(M11), "M12": float(M12),
            "M21": float(M21), "M22": float(M22),
            "det_M": float(det_M),
            "ratio_M12_M11": float(ratio12),
            "ratio_M21_M22": float(ratio21),
            "verdict": verdict,
        }

    return dict(test="connection_matrix", results=results)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("v5 HORIZON IVP: Can SCT connect horizon to regular center?")
    print(f"Parameters: M = {M_BH}, Lam = {LAM}, K = {K_LAYERS}")
    print(f"m2 = sqrt(60/13)*Lam = {np.sqrt(60/13)*LAM:.4f}")
    print(f"m2*r_h = {np.sqrt(60/13)*LAM*R_H:.4f}")
    print("=" * 70)

    all_results = {
        "parameters": {
            "M": M_BH,
            "Lam": LAM,
            "alpha_C": ALPHA_C,
            "K": K_LAYERS,
            "r_h": R_H,
            "m2sq": M2SQ,
            "sqrt6": float(SQRT6),
        }
    }

    t0 = time.time()

    # Test 1: Single field
    try:
        r1 = test1_single_field()
        all_results["test1_single_field"] = r1
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")
        traceback.print_exc()
        all_results["test1_single_field"] = {"error": str(e)}

    # Test 2: Full layers
    try:
        r2 = test2_full_layers()
        all_results["test2_full_layers"] = r2
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")
        traceback.print_exc()
        all_results["test2_full_layers"] = {"error": str(e)}

    # Test 3: Coupled IVP
    try:
        r3 = test3_coupled_ivp()
        all_results["test3_coupled_ivp"] = r3
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        traceback.print_exc()
        all_results["test3_coupled_ivp"] = {"error": str(e)}

    # Test 4: BVP
    try:
        r4 = test4_bvp_horizon_center()
        all_results["test4_bvp"] = r4
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")
        traceback.print_exc()
        all_results["test4_bvp"] = {"error": str(e)}

    # Test 5: Connection matrix
    try:
        r5 = test5_connection_matrix()
        all_results["test5_connection"] = r5
    except Exception as e:
        print(f"  TEST 5 FAILED: {e}")
        traceback.print_exc()
        all_results["test5_connection"] = {"error": str(e)}

    elapsed = time.time() - t0

    # ── Final verdict ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    verdicts = []

    # Test 1
    t1 = all_results.get("test1_single_field", {})
    if "error" not in t1:
        n1 = t1.get("n_eff", np.nan)
        v1 = f"Single field n_eff = {n1:.4f}"
        if abs(n1 - (3 - SQRT6)) < 0.1:
            v1 += " (= 3-sqrt6, singular Frobenius)"
        elif n1 > 2.5:
            v1 += " (REGULAR!)"
        verdicts.append(v1)

    # Test 2
    t2 = all_results.get("test2_full_layers", {})
    if "error" not in t2:
        n2 = t2.get("n_eff_J", np.nan)
        v2 = f"Full layers n_eff = {n2:.4f}"
        verdicts.append(v2)

    # Test 3
    t3 = all_results.get("test3_coupled_ivp", {})
    if "error" not in t3:
        H3 = t3.get("H_end", np.nan)
        m3 = t3.get("m_end", np.nan)
        v3 = f"Coupled: H(r_end) = {H3:.4f}, m(r_end)/M = {m3/M_BH:.4f}"
        if H3 > 0.5:
            v3 += " → REGULAR CENTER REACHED"
        elif H3 < 0:
            v3 += " → H < 0 (SINGULAR)"
        else:
            v3 += " → intermediate"
        verdicts.append(v3)

    # Test 4
    t4 = all_results.get("test4_bvp", {})
    if "error" not in t4:
        bvp_results = t4.get("results", {})
        any_converged = any(v.get("converged", False) for v in bvp_results.values())
        if any_converged:
            verdicts.append("BVP horizon-center: CONVERGED — connection exists!")
        else:
            verdicts.append("BVP horizon-center: FAILED to converge — connection obstructed")

    # Test 5
    t5 = all_results.get("test5_connection", {})
    if "error" not in t5:
        conn = t5.get("results", {})
        phys = conn.get(f"m2sq_{M2SQ:.4f}", {})
        ratio = phys.get("ratio_M12_M11", np.nan)
        v5 = f"Connection matrix |M12/M11| = {ratio:.4e}"
        if ratio > 0.01:
            v5 += " — regular center GENERATES log at horizon (M12 != 0)"
        else:
            v5 += " — compatible"
        verdicts.append(v5)

    for v in verdicts:
        print(f"  {v}")

    # Overall conclusion
    print("\n  CONCLUSION:")
    t5_phys = all_results.get("test5_connection", {}).get("results", {}).get(f"m2sq_{M2SQ:.4f}", {})
    M12_ratio = t5_phys.get("ratio_M12_M11", np.nan)
    if M12_ratio > 0.01:
        print("  The connection matrix M12 != 0 proves that imposing regularity")
        print("  at the center (fakeon beta=0) GENERATES a log-divergent mode")
        print("  at the horizon. The SINGLE auxiliary field equation cannot")
        print("  simultaneously be regular at both center and horizon.")
        print()
        print("  PHYSICAL MEANING:")
        print("  - For the BH to exist with a regular interior, the FULL coupled")
        print("    system (including metric backreaction) would need to cancel")
        print("    the M12 obstruction. This is a NONLINEAR effect.")
        print("  - If the coupled system also fails (Test 3-4): horizonless")
        print("    soliton (gravastar/star) is the only regular solution.")
        print("  - The fakeon prescription alone cannot save BH regularity —")
        print("    it requires the dynamics of the full nonlinear system.")
    else:
        print("  The connection matrix allows simultaneous regularity at")
        print("  both center and horizon. BH interpretation is viable.")

    all_results["verdicts"] = verdicts
    all_results["elapsed_seconds"] = elapsed

    # Save
    outfile = OUTDIR / "v5_horizon_ivp.json"
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
