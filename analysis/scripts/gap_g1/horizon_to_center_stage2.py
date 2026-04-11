"""
Stage 2: Coupled backreaction IVP — horizon to center.

Full system: metric H(r) + localized Weyl fields u_k(r), integrated INWARD
from the horizon. The u_k feed back into H through the effective field equations.

KEY QUESTION: Does backreaction push toward regular center (n=3) or reinforce
the singular branch (n_eff ≈ 0.55 from Stage 1)?

Physics: SCT action S = (1/16piG) int sqrt(-g) [R + alpha_C C^2 F_1(box/Lam^2)] d^4x
In spherical symmetry with f = 1 - 2m(r)/r:
  - Weyl amplitude W = (-r^2 m'' + 4r m' - 6m) / (6r^3)  [= E_{rr}/something]
  - Localized fields: u_k satisfy diffusion equation with source W
  - Backreaction: J = sum(c_k u_k) modifies m'(r)

The coupled first-order system (state = [m, m', H, H', u_0, v_0, ..., u_{K-1}, v_{K-1}]):
  dm/dr = m'
  dm'/dr = m''  (from constraint + u_k coupling)
  du_k/dr = v_k
  dv_k/dr = u_k''  (from localized equation)

David Alfyorov, SCT Theory project, 2026-04-06
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
import json
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent.parent / "results" / "gap_g1"


def build_layers(K=8):
    x, w = leggauss(K)
    alpha = 0.25*(x + 1.0)
    w_q = 0.25*w*2.0
    tau = alpha*(1.0 - alpha)
    P = -89.0/24.0 + (43.0/6.0)*tau + (236.0/3.0)*tau**2
    idx = np.argsort(tau)
    tau, w_q, P = tau[idx], w_q[idx], P[idx]
    c = w_q * P
    dtau = np.diff(np.concatenate(([0.0], tau)))
    print(f"  P(alpha) integral = {np.dot(w_q, P):.8f} (should be {13/120:.8f})")
    return dict(tau=tau, c=c, dtau=dtau, K=K)


def coupled_rhs(r, y, layers, Lam, alpha_C):
    """
    Full coupled RHS for the one-function system.

    State: y = [m, m', u_0, v_0, ..., u_{K-1}, v_{K-1}]
    where m(r) = mass function, f = 1 - 2m/r.

    The Weyl amplitude on this background:
      W(r) = (-r^2 m'' + 4r m' - 6m) / (6 r^3)

    Localized equation for u_k:
      H u_k'' + A_coeff u_k' - S_coeff u_k = lam_k (u_k - u_{k-1})
    where H = 1 - 2m/r, A_coeff = H' + 2H/r, S_coeff = 6H/r^2

    Backreaction: the effective field equation gives m'' in terms of u_k.
    In vacuum SCT, the trace of the modified Einstein equation gives:

      m''(r) = m''_GR + alpha_C * correction_from_J

    For the GR part in vacuum: m'' = 0 (Schwarzschild has m = const).
    The SCT correction comes from the variation of alpha_C C^2 F_1(box/Lam^2).

    Simplified backreaction model:
      m'' = alpha_C * 6 * r * (J - W) * Lam^2
    where J - W represents the difference between form-factor-dressed and bare Weyl.
    This is the leading correction: when J = W (no dressing), m'' = 0 (Schwarzschild).
    """
    K = layers['K']
    c = layers['c']
    dtau = layers['dtau']

    if r < 1e-15:
        return np.zeros_like(y)

    m  = y[0]
    mp = y[1]  # m'

    # Metric
    H  = 1.0 - 2.0*m/r
    Hp = 2.0*m/r**2 - 2.0*mp/r

    # Weyl amplitude from CURRENT m(r)
    # W = (-r^2 m'' + 4r m' - 6m) / (6r^3)
    # But we don't know m'' yet — it's part of what we're solving for.
    # Use the PREVIOUS step's estimate: compute W from current m, m'.
    # For the zeroth iteration, use Schwarzschild-like: m'' ≈ 0.
    # This makes W = (4r m' - 6m) / (6r^3)
    W_partial = (4.0*r*mp - 6.0*m) / (6.0*r**3)

    # Extract u_k, v_k
    u = np.array([y[2 + 2*k] for k in range(K)])
    v = np.array([y[3 + 2*k] for k in range(K)])

    # J = sum(c_k * u_k)
    J = np.dot(c, u)
    Jp = np.dot(c, v)

    # Backreaction: m'' from SCT correction
    # The key formula: when form factors dress the Weyl tensor,
    # the effective stress-energy generates m''.
    # Leading-order backreaction:
    #   delta_m'' ~ alpha_C * Lam^2 * r * (correction terms)
    #
    # More carefully: the SCT modification to Gtt gives
    #   m'(r) = m'_GR + alpha_C * (...) involving J, J', J''
    # Taking derivative: m'' = alpha_C * (...)
    #
    # For a tractable model, use the simplest gauge-invariant coupling:
    #   m''_SCT = alpha_C * 6 * [(J - W_partial) * (-r^2) + 4r * (J_derivative_correction)] / (r^2)
    #
    # But actually, the cleanest approach: the modified EOM in the Weyl sector
    # effectively replaces W → J in the field equation contribution.
    # So: the "effective m''" that's consistent with J is:
    #   -r^2 m''_eff + 4r m' - 6m = 6r^3 * J
    #   => m''_eff = (4r m' - 6m - 6r^3 J) / r^2
    #
    # This is the SELF-CONSISTENCY condition: the metric produces Weyl W,
    # the form factor dresses it to J, and the metric must be consistent with J.

    mpp = (4.0*r*mp - 6.0*m - 6.0*r**3 * J) / r**2

    # Now the FULL Weyl amplitude including m'':
    W = (-r**2*mpp + 4*r*mp - 6*m) / (6*r**3)
    # By construction: W = J (self-consistency!)
    # But u_k evolve dynamically, so J ≠ W at intermediate steps.

    # Localized equations for u_k
    A_coeff = Hp + 2.0*H/r
    S_coeff = 6.0*H/r**2

    dydr = np.zeros_like(y)
    dydr[0] = mp
    dydr[1] = mpp

    for k in range(K):
        uk = u[k]
        vk = v[k]
        lam_k = Lam**2 / dtau[k]

        if k == 0:
            u_prev = W  # source: current Weyl amplitude
        else:
            u_prev = u[k-1]

        # H u_k'' + A u_k' - S u_k = lam_k(u_k - u_prev)
        if abs(H) < 1e-12:
            # At horizon: H = 0, degenerate
            # A_coeff u_k' - S u_k = lam_k(u_k - u_prev)
            # Can't determine u_k'' from this alone. Use regularity: u_k'' finite.
            # From L'Hopital: lim_{H->0} u_k'' = [lam_k(u-u_prev) - A v + S u] / H
            # Use Hp to regularize: u_k'' ~ [RHS] / (Hp * dr) at horizon crossing
            upp_k = 0.0  # safe fallback
        else:
            upp_k = (lam_k*(uk - u_prev) - A_coeff*vk + S_coeff*uk) / H

        dydr[2 + 2*k] = vk
        dydr[3 + 2*k] = upp_k

    return dydr


def run_stage2(M=1.0, Lam=1.0, alpha_C=13.0/120.0, K=8,
               r_start_frac=0.998, r_end_frac=0.005, n_points=10000):
    """Run coupled IVP from just inside horizon to near center."""
    layers = build_layers(K)
    r_h = 2.0*M
    r_start = r_h * r_start_frac
    r_end = r_h * r_end_frac

    print(f"\n  Stage 2: Coupled backreaction, M={M}, Lam={Lam}, alpha_C={alpha_C:.6f}")
    print(f"  r_h = {r_h:.4f}, r_start = {r_start:.4f}, r_end = {r_end:.6f}")

    # Initial conditions: Schwarzschild at r_start
    m0 = M  # Schwarzschild mass function = const
    mp0 = 0.0  # m' = 0 for Schwarzschild

    H0 = 1.0 - 2.0*M/r_start
    W0 = -2.0*M/r_start**3  # Schwarzschild Weyl

    y0 = np.zeros(2 + 2*K)
    y0[0] = m0
    y0[1] = mp0

    for k in range(K):
        tau_k = layers['tau'][k]
        u_init = W0 / (1.0 + 4.0*tau_k)
        v_init = 6.0*M/r_start**4 / (1.0 + 4.0*tau_k)
        y0[2 + 2*k] = u_init
        y0[3 + 2*k] = v_init

    print(f"  IC: m = {m0:.4f}, m' = {mp0:.6f}, W = {W0:.6e}")

    r_eval = np.linspace(r_start, r_end, n_points)

    sol = solve_ivp(
        lambda r, y: coupled_rhs(r, y, layers, Lam, alpha_C),
        [r_start, r_end],
        y0,
        method='RK45',
        t_eval=r_eval,
        rtol=1e-9,
        atol=1e-11,
        max_step=(r_start - r_end)/500,
    )

    print(f"  solve_ivp status = {sol.status}: {sol.message}")
    print(f"  Points: {sol.t.size}")
    return sol, layers


def analyze_stage2(sol, layers, M=1.0, Lam=1.0):
    """Analyze coupled solution: m(r), H(r), E^2, n_eff, compare with Stage 1."""
    K = layers['K']
    c = layers['c']
    r = sol.t
    m_arr = sol.y[0]
    mp_arr = sol.y[1]

    # Metric
    H_arr = 1.0 - 2.0*m_arr/r

    # J(r)
    J_arr = np.zeros(len(r))
    for k in range(K):
        J_arr += c[k] * sol.y[2 + 2*k]

    # Weyl amplitude from m(r)
    # W = (-r^2 m'' + 4r m' - 6m)/(6r^3)
    # m'' from self-consistency: mpp = (4r m' - 6m - 6r^3 J)/r^2
    mpp_arr = (4*r*mp_arr - 6*m_arr - 6*r**3*J_arr) / r**2
    W_arr = (-r**2*mpp_arr + 4*r*mp_arr - 6*m_arr) / (6*r**3)

    # E^2 from m(r): E^2 = (-r^2 m'' + 4r m' - 6m)^2 / (6r^6)
    A_arr = -r**2*mpp_arr + 4*r*mp_arr - 6*m_arr
    E2_arr = A_arr**2 / (6*r**6)

    # E^2 Schwarzschild for comparison
    E2_sch = 6*M**2/r**6

    # n_eff
    ln_r = np.log(r)
    ln_E2 = np.log(np.maximum(E2_arr, 1e-300))
    slope = np.gradient(ln_E2, ln_r)
    n_eff = (slope + 6) / 2

    # m(r) profile
    print(f"\n  === STAGE 2 ANALYSIS ===")
    print(f"  r range: [{r[-1]:.4e}, {r[0]:.4e}]")
    print()
    print(f"  {'r':>12s} {'m(r)':>12s} {'m_prime':>12s} {'H(r)':>12s} {'E2/E2_Sch':>12s} {'n_eff':>8s} {'J/W':>10s}")
    print("  " + "-"*82)

    for frac in [0.99, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        r_target = 2*M*frac
        if r_target < r[-1] or r_target > r[0]:
            continue
        idx = np.argmin(np.abs(r - r_target))
        ratio_E2 = E2_arr[idx]/(E2_sch[idx] + 1e-300) if E2_sch[idx] > 0 else 0
        ratio_JW = J_arr[idx]/(W_arr[idx] + 1e-300) if abs(W_arr[idx]) > 1e-300 else 0
        print(f"  {r[idx]:12.4e} {m_arr[idx]:12.6f} {mp_arr[idx]:12.4e} {H_arr[idx]:12.6f} "
              f"{ratio_E2:12.4e} {n_eff[idx]:8.3f} {ratio_JW:10.4f}")

    # Key diagnostics
    print(f"\n  m(r_end)/M = {m_arr[-1]/M:.6f}")
    print(f"  m'(r_end) = {mp_arr[-1]:.6e}")
    print(f"  H(r_end) = {H_arr[-1]:.6f}")
    print(f"  n_eff(r_end) = {n_eff[-1]:.4f}")

    # Classify: what n does m(r) approach near center?
    # If m ~ const: n = 0 (Schwarzschild-like)
    # If m ~ a*r: n = 1 (linearized SCT)
    # If m ~ a*r^3: n = 3 (regular)
    r_inner = r[-200:]
    m_inner = m_arr[-200:]
    if len(r_inner) > 10:
        # Fit m vs r^n near center
        ln_r_i = np.log(r_inner)
        ln_m_i = np.log(np.abs(m_inner) + 1e-300)
        coeffs = np.polyfit(ln_r_i, ln_m_i, 1)
        n_mass = coeffs[0]
        print(f"  m(r) power law near r_end: m ~ r^{n_mass:.3f}")
        print(f"  (n=0: Schwarzschild, n=1: SCT linear, n=3: regular)")

    # Stage 1 vs Stage 2 comparison
    print(f"\n  COMPARISON:")
    print(f"  Stage 1 (fixed bg): n_eff → 3-sqrt(6) ≈ 0.551")
    print(f"  Stage 2 (coupled):  n_eff → {n_eff[-1]:.4f}")
    if n_eff[-1] > 2.5:
        print(f"  >>> BACKREACTION PUSHES TOWARD REGULAR CENTER! <<<")
    elif n_eff[-1] > 1.5:
        print(f"  >>> BACKREACTION SIGNIFICANTLY MODIFIES SINGULARITY <<<")
    elif abs(n_eff[-1] - 0.551) < 0.1:
        print(f"  >>> BACKREACTION NEGLIGIBLE — same as Stage 1 <<<")
    elif n_eff[-1] < 0:
        print(f"  >>> BACKREACTION MAKES SINGULARITY WORSE! <<<")
    else:
        print(f"  >>> BACKREACTION MODERATE — partial modification <<<")

    return dict(r=r, m=m_arr, H=H_arr, E2=E2_arr, n_eff=n_eff, J=J_arr, W=W_arr)


if __name__ == "__main__":
    print("="*70)
    print("STAGE 2: COUPLED BACKREACTION — HORIZON TO CENTER")
    print("="*70)

    results = {}

    # Scan alpha_C values: 0 (no coupling), actual (13/120), enhanced
    for alpha_C in [0.0, 13.0/120.0, 1.0, 10.0]:
        label = f"alpha_{alpha_C:.4f}"
        print(f"\n{'='*50}")
        print(f"  alpha_C = {alpha_C:.4f}")
        print(f"{'='*50}")

        try:
            sol, layers = run_stage2(M=1.0, Lam=1.0, alpha_C=alpha_C, K=8,
                                     r_start_frac=0.998, r_end_frac=0.005)
            res = analyze_stage2(sol, layers, M=1.0, Lam=1.0)
            results[label] = {
                'alpha_C': alpha_C,
                'n_eff_end': float(res['n_eff'][-1]),
                'H_end': float(res['H'][-1]),
                'm_end': float(res['m'][-1]),
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[label] = {'alpha_C': alpha_C, 'error': str(e)}

    # Also try different Lam with physical alpha_C
    for Lam in [0.5, 2.0, 5.0]:
        label = f"Lam_{Lam}"
        print(f"\n{'='*50}")
        print(f"  Lam = {Lam}, alpha_C = 13/120")
        print(f"{'='*50}")
        try:
            sol, layers = run_stage2(M=1.0, Lam=Lam, alpha_C=13.0/120.0, K=8,
                                     r_start_frac=0.998, r_end_frac=0.005)
            res = analyze_stage2(sol, layers, M=1.0, Lam=Lam)
            results[label] = {
                'Lam': Lam,
                'n_eff_end': float(res['n_eff'][-1]),
                'H_end': float(res['H'][-1]),
                'm_end': float(res['m'][-1]),
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[label] = {'Lam': Lam, 'error': str(e)}

    outfile = OUTDIR / "horizon_to_center_stage2.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    print("\n" + "="*70)
    print("STAGE 2 SUMMARY")
    print("="*70)
    for key, val in results.items():
        if 'error' in val:
            print(f"  {key}: FAILED — {val['error'][:60]}")
        else:
            print(f"  {key}: n_eff = {val['n_eff_end']:.4f}, H = {val['H_end']:.4f}, m/M = {val['m_end']:.4f}")
