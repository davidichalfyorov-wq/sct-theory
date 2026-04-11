"""
Horizon-to-center IVP with full form factor localized system.

CRITICAL TEST: Does SCT dynamics SELECT the regular center (n=3)?
Or does it select the singular branch (n=1)?

Two stages:
  Stage 1: Fixed Schwarzschild background, evolve u_k fields inward from horizon.
  Stage 2: Full coupled system (F, H, u_k) — backreaction included.

The answer determines whether singularity resolution is DYNAMICAL or requires fine-tuning.

David Alfyorov, SCT Theory project, 2026-04-06
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
import json
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent.parent / "results" / "gap_g1"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ─── P(alpha) kernel quadrature (from P12) ───────────────────────────
def build_alpha_layers(K: int = 8):
    """Build tau-layers from the symmetric P(alpha) kernel.
    Integrates alpha in [0, 1/2] and doubles weights (symmetry)."""
    x, w = leggauss(K)
    alpha = 0.25 * (x + 1.0)         # [-1,1] → [0, 1/2]
    w_q = 0.25 * w * 2.0             # Jacobian × symmetry
    tau = alpha * (1.0 - alpha)
    P = -89.0/24.0 + (43.0/6.0)*tau + (236.0/3.0)*tau**2

    idx = np.argsort(tau)
    tau = tau[idx]; w_q = w_q[idx]; P = P[idx]; alpha = alpha[idx]
    c = w_q * P                       # quadrature weights for J = sum(c_k * u_k)
    dtau = np.diff(np.concatenate(([0.0], tau)))

    # Verify integral = alpha_C = 13/120
    integral = np.dot(w_q, P)
    print(f"  P(alpha) integral = {integral:.8f}  (expected 13/120 = {13/120:.8f})")
    return dict(tau=tau, c=c, dtau=dtau, P=P, w=w_q, K=K)


# ─── Schwarzschild background ────────────────────────────────────────
def schwarzschild_H(r, M):
    return 1.0 - 2.0*M/r

def schwarzschild_Hp(r, M):
    return 2.0*M/r**2

def schwarzschild_Hpp(r, M):
    return -4.0*M/r**3

def schwarzschild_Wamp(r, M):
    """Weyl amplitude W = (r^2 H'' - 2rH' + 2H - 2) / (6r^2) on Schwarzschild."""
    return -2.0*M/r**3


# ─── Stage 1: Fixed background IVP ──────────────────────────────────
def stage1_rhs(r, y, M, layers, Lam):
    """RHS for u_k ODE system on FIXED Schwarzschild background.

    For each k:
      H u_k'' + (H' + 2H/r) u_k' - 6H/r^2 u_k = (Lam^2/dtau_k)(u_k - u_{k-1})

    where u_0 = W_amp(r) for the first layer (k=0: source is background Weyl).

    State vector: y = [u_0, u_0', u_1, u_1', ..., u_{K-1}, u_{K-1}']
    """
    K = layers['K']
    dtau = layers['dtau']

    H  = schwarzschild_H(r, M)
    Hp = schwarzschild_Hp(r, M)
    W  = schwarzschild_Wamp(r, M)

    # Coefficients of the operator
    A_coeff = Hp + 2.0*H/r          # coefficient of u'
    S_coeff = 6.0*H/r**2            # coefficient of u (with minus sign)

    dydr = np.zeros(2*K)

    for k in range(K):
        uk  = y[2*k]
        vk  = y[2*k + 1]

        # Source: u_{k-1} for k>0, W_amp for k=0
        if k == 0:
            u_prev = W
        else:
            u_prev = y[2*(k-1)]

        lam_k = Lam**2 / dtau[k]

        # H u'' + A u' - S u = lam_k (u - u_prev)
        # u'' = [lam_k(u - u_prev) - A u' + S u] / H
        if abs(H) < 1e-14:
            # At horizon: degenerate. Use L'Hopital / constraint:
            # A u' = lam_k(u - u_prev) + S u
            # But this only gives u', not u''. Set u'' = 0 as limit.
            upp = 0.0
        else:
            upp = (lam_k*(uk - u_prev) - A_coeff*vk + S_coeff*uk) / H

        dydr[2*k]     = vk
        dydr[2*k + 1] = upp

    return dydr


def run_stage1(M=1.0, Lam=1.0, K=8, r_start_factor=0.999, r_end_factor=0.01,
               n_points=5000, method='RK45'):
    """Run Stage 1: fixed Schwarzschild background, u_k IVP from horizon inward."""
    layers = build_alpha_layers(K)

    r_h = 2.0*M
    r_start = r_h * r_start_factor   # just inside horizon
    r_end   = r_h * r_end_factor     # deep inside

    print(f"\n  Stage 1: Fixed Schwarzschild, M={M}, Lam={Lam}")
    print(f"  r_h = {r_h:.6f}, r_start = {r_start:.6f}, r_end = {r_end:.6f}")
    print(f"  K = {K} alpha-layers")

    # Initial conditions at r_start (Schwarzschild values)
    W0 = schwarzschild_Wamp(r_start, M)
    H0 = schwarzschild_H(r_start, M)
    Hp0 = schwarzschild_Hp(r_start, M)

    y0 = np.zeros(2*K)
    for k in range(K):
        tau_k = layers['tau'][k]
        # On Schwarzschild: u_k ≈ W / (1 + tau_k * something)
        # Leading order: u_k ≈ W_amp / (1 + 4*tau_k) is P12's initial guess
        # But more carefully at first order:
        u_k_init = W0 / (1.0 + 4.0*tau_k)  # stationary approximation

        # Derivative: du/dr ≈ dW/dr / (1 + 4*tau_k) = 6M/r^4 / (1 + 4*tau_k)
        dW_dr = 6.0*M/r_start**4
        v_k_init = dW_dr / (1.0 + 4.0*tau_k)

        y0[2*k]     = u_k_init
        y0[2*k + 1] = v_k_init

    print(f"  IC: W_amp = {W0:.6e}, u_0 = {y0[0]:.6e}")

    # Integrate INWARD (decreasing r)
    r_eval = np.linspace(r_start, r_end, n_points)

    sol = solve_ivp(
        lambda r, y: stage1_rhs(r, y, M, layers, Lam),
        [r_start, r_end],
        y0,
        method=method,
        t_eval=r_eval,
        rtol=1e-10,
        atol=1e-12,
        max_step=(r_start - r_end)/200,
    )

    if sol.status != 0:
        print(f"  WARNING: solve_ivp status = {sol.status}: {sol.message}")
    else:
        print(f"  Integration successful, {sol.t.size} points")

    return sol, layers


def analyze_stage1(sol, layers, M=1.0, Lam=1.0):
    """Analyze Stage 1 results: compute n_eff, E^2 profile, check regularity."""
    K = layers['K']
    r = sol.t
    c = layers['c']

    # Compute J(r) = sum(c_k * u_k)
    J = np.zeros(len(r))
    for k in range(K):
        J += c[k] * sol.y[2*k]

    # On fixed background: E^2 = 6M^2/r^6 (Schwarzschild)
    # But u_k encode the RESPONSE of the form factor system
    # The physical question: does J (form-factor weighted Weyl) stay regular?
    W_amp = schwarzschild_Wamp(r, M)

    # J/W_amp ratio: if → 1, form factors don't modify much
    # If → 0, form factors SUPPRESS Weyl (regularization!)
    # If → ∞, form factors AMPLIFY (instability!)
    ratio = J / W_amp

    # Compute n_eff from J profile (since CJ ∝ J^2 on fixed background)
    ln_r = np.log(r)
    J_sq = J**2
    ln_J_sq = np.log(np.maximum(J_sq, 1e-300))
    slope_J2 = np.gradient(ln_J_sq, ln_r)
    n_eff = (slope_J2 + 6) / 2  # same as for E^2

    # Check individual u_k behavior
    u_last = sol.y[2*(K-1)]
    u_first = sol.y[0]

    # Regularity indicators
    print(f"\n  === STAGE 1 ANALYSIS ===")
    print(f"  r range: [{r[-1]:.4e}, {r[0]:.4e}]")
    print(f"  J/W_amp at r_start: {ratio[0]:.6f}")
    print(f"  J/W_amp at r_end:   {ratio[-1]:.6f}")
    print(f"  max|J/W_amp|:       {np.max(np.abs(ratio)):.6f}")
    print(f"  min|J/W_amp|:       {np.min(np.abs(ratio)):.6f}")
    print()

    # n_eff profile
    print(f"  {'r':>12s} {'J/W_amp':>12s} {'n_eff':>8s} {'slope':>8s} {'u_0':>12s} {'u_last':>12s}")
    print("  " + "-"*68)
    for frac in [0.99, 0.9, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        r_target = 2*M * frac
        idx = np.argmin(np.abs(r - r_target))
        if idx < len(r):
            print(f"  {r[idx]:12.4e} {ratio[idx]:12.6f} {n_eff[idx]:8.3f} {slope_J2[idx]:+8.2f} "
                  f"{sol.y[0,idx]:12.4e} {sol.y[2*(K-1),idx]:12.4e}")

    # KEY VERDICT
    # Does J/W_amp → finite or → ∞ as r → 0?
    r_inner = r[-100:]  # last 100 points
    ratio_inner = ratio[-100:]
    is_growing = np.mean(np.diff(np.abs(ratio_inner))) > 0
    inner_slope = np.polyfit(np.log(r_inner), np.log(np.abs(ratio_inner) + 1e-300), 1)[0]

    print(f"\n  Inner behavior (r < {r_inner[0]:.3e}):")
    print(f"    |J/W_amp| trend: {'GROWING' if is_growing else 'STABLE/DECREASING'}")
    print(f"    Power-law slope of |J/W|: {inner_slope:+.3f}")
    print(f"    If slope ≈ 0: ratio const → n_eff stays same as Schwarzschild (n=0)")
    print(f"    If slope > 0: J grows faster → n_eff > 0 (modification)")
    print(f"    If slope = +1: J/W ~ r → E^2_eff ~ r^{-4} → n_eff = 1 (SCT linear)")
    print(f"    If slope = +3: J/W ~ r^3 → E^2_eff ~ r^0 → n_eff = 3 (REGULAR!)")

    return dict(r=r, J=J, W=W_amp, ratio=ratio, n_eff=n_eff, slope=slope_J2)


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("HORIZON-TO-CENTER IVP: Does SCT dynamics select regular center?")
    print("="*70)

    # Run at several Lam values to see how nonlocality affects the answer
    results = {}
    M = 1.0

    for Lam in [0.5, 1.0, 2.0, 5.0, 10.0]:
        print(f"\n{'='*50}")
        print(f"  Lambda = {Lam} (1/Lambda = {1/Lam:.3f})")
        print(f"  m2 = {2.148*Lam:.3f}, m2*r_h = {2.148*Lam*2*M:.3f}")
        print(f"{'='*50}")

        try:
            sol, layers = run_stage1(M=M, Lam=Lam, K=8,
                                     r_start_factor=0.999,
                                     r_end_factor=0.005,
                                     n_points=10000)
            res = analyze_stage1(sol, layers, M=M, Lam=Lam)
            results[f"Lam_{Lam}"] = {
                'Lam': Lam,
                'r_end': float(sol.t[-1]),
                'J_W_ratio_end': float(res['ratio'][-1]),
                'n_eff_end': float(res['n_eff'][-1]),
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[f"Lam_{Lam}"] = {'Lam': Lam, 'error': str(e)}

    # Save results
    outfile = OUTDIR / "horizon_to_center_stage1.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Final verdict
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for key, val in results.items():
        if 'error' in val:
            print(f"  {key}: FAILED - {val['error']}")
        else:
            print(f"  {key}: J/W_end = {val['J_W_ratio_end']:.6f}, n_eff_end = {val['n_eff_end']:.3f}")
