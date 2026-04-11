"""
Connection matrix M for the massive spin-2 operator on Schwarzschild interior.

Operator: D_massive u = H u'' + (H' + 2H/r) u' + (m2^2 - 6H/r^2) u = 0
where H(r) = 1 - 2M/r, H'(r) = 2M/r^2, M = 1 (geometric units).

Frobenius bases:
  At r -> 0: u1 = r^{+sqrt(6)}(1+...), u2 = r^{-sqrt(6)}(1+...)  [indicial s^2-6=0]
  At r -> 2M: v1 = 1+c1*rho+..., v2 = v1*ln(rho)+...  [indicial s^2=0, rho=r-2M]

Connection matrix: (u1, u2) integrated to horizon gives
  u_i(r_end) = M_i1 * v1(r_end) + M_i2 * v2(r_end)

Key question: Is M_12 = 0? (regular-at-center -> regular-at-horizon?)

Corrected mass: m2^2 = 1.2807 * Lambda^2  (NOT 60/13)
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import os

# ── Parameters ──
M_bh = 1.0        # Black hole mass in geometric units
r_horizon = 2 * M_bh  # Schwarzschild radius
sqrt6 = np.sqrt(6)

def H(r):
    """Metric function H = 1 - 2M/r (negative inside horizon for r < 2M)."""
    return 1.0 - 2 * M_bh / r

def Hp(r):
    """H' = 2M/r^2."""
    return 2 * M_bh / r**2

def ode_system(r, y, m2sq):
    """
    Convert D_massive u = 0 to first-order system.
    y = [u, u']

    H u'' + (H' + 2H/r) u' + (m2^2 - 6H/r^2) u = 0
    => u'' = -(1/H)[(H' + 2H/r) u' + (m2^2 - 6H/r^2) u]
    """
    u, up = y
    h = H(r)
    hp = Hp(r)

    coeff_up = hp + 2 * h / r
    coeff_u = m2sq - 6 * h / r**2

    upp = -(coeff_up * up + coeff_u * u) / h
    return [up, upp]


def frobenius_center_ic(eps, exponent):
    """
    Initial conditions at r=eps for Frobenius solution r^s near r=0.

    Near r=0: H ~ -2M/r, so H is large negative.
    Indicial equation from leading terms: s(s-1) - 2s - 6 = 0 => s^2 - 3s - 6 = 0
    Wait, let me re-derive carefully.

    Actually the indicial equation comes from:
    H u'' + (H' + 2H/r) u' + (m2^2 - 6H/r^2) u = 0

    Near r=0: H ~ -2M/r = -2/r, H' = 2/r^2, 2H/r ~ -4/r^2, 6H/r^2 ~ -12/r^3

    For u = r^s:
    u' = s*r^{s-1}, u'' = s(s-1)*r^{s-2}

    Leading balance (most singular terms ~ r^{s-3}):
    H*u'' ~ (-2/r)*s(s-1)*r^{s-2} = -2s(s-1)*r^{s-3}
    (2H/r)*u' ~ (-4/r^2)*s*r^{s-1} = -4s*r^{s-3}
    (-6H/r^2)*u ~ (12/r^3)*r^s = 12*r^{s-3}

    Also H'*u' ~ (2/r^2)*s*r^{s-1} = 2s*r^{s-3}

    Sum: [-2s(s-1) - 4s + 2s + 12]*r^{s-3} = 0
    => -2s^2 + 2s - 4s + 2s + 12 = 0
    => -2s^2 + 0s + 12 = 0
    => s^2 = 6
    => s = +/-sqrt(6)

    Good, matches the task specification.
    """
    s = exponent
    u = eps**s
    up = s * eps**(s - 1)
    return [u, up]


def frobenius_horizon_basis(rho, m2sq):
    """
    Frobenius basis at r = 2M (rho = r - 2M -> 0+).

    Near rho=0: H = 1 - 2/(2+rho) = rho/(2+rho) ~ rho/2
    H' = 2/(2+rho)^2 ~ 1/2

    Rewriting the ODE in terms of rho:
    H u'' + (H' + 2H/r) u' + (m2^2 - 6H/r^2) u = 0

    Near rho=0 (r=2M):
    H ~ rho/2, H' ~ 1/2, 2H/r ~ rho/2, 6H/r^2 ~ 3rho/4

    Leading: (rho/2) u'' + (1/2 + rho/2) u' + (m2^2 - 3rho/4) u = 0

    Indicial: s(s-1)*(1/2) + s*(1/2) = 0 => s^2/2 = 0 => s=0 (double root)

    So v1 = 1 + c1*rho + c2*rho^2 + ...
    v2 = v1*ln(rho) + d1*rho + d2*rho^2 + ...

    For v1: substitute u = 1 + c1*rho into the ODE at leading order:
    (rho/2)*0 + (1/2)*c1 + m2^2*1 = 0  => c1 = -2*m2^2

    Actually let me compute more carefully.
    Let r = 2 + rho, so H = rho/(2+rho).

    For u = 1 + c1*rho + c2*rho^2 + ...:
    u' = c1 + 2*c2*rho + ...
    u'' = 2*c2 + ...

    H*u'' = (rho/2)*2*c2 + ... = c2*rho + ...
    (H')*u' = (1/2)*c1 + ...  [at leading order in rho^0]
    Wait, H' = 2/(2+rho)^2, so at rho=0: H'(2) = 1/2
    (2H/r)*u' = (2*rho/(2*(2+rho)))*c1 ~ (rho/2)*c1 ~ O(rho)
    m2^2 * u = m2^2 + m2^2*c1*rho + ...
    -6H/r^2 * u = -6*(rho/(2+rho))/(2+rho)^2 * u ~ -6*rho/8 + ... ~ O(rho)

    At O(rho^0): (1/2)*c1 + m2^2 = 0 => c1 = -2*m2^2
    """
    # v1 = 1 + c1*rho + ...
    c1 = -2 * m2sq
    v1 = 1.0 + c1 * rho
    v1p = c1  # dv1/drho = dv1/dr

    # v2 = v1*ln(rho) + d1*rho + ...
    # At leading order, d1 can be computed but for extraction we mainly need
    # the ln(rho) coefficient
    ln_rho = np.log(abs(rho))
    v2 = v1 * ln_rho  # + regular corrections
    v2p = v1p * ln_rho + v1 / rho  # + regular corrections

    return v1, v1p, v2, v2p


def integrate_from_center(m2sq, eps=1e-3, delta=1e-3, exponent=None):
    """
    Integrate from r=eps to r=2M-delta.

    exponent: +sqrt(6) for u1, -sqrt(6) for u2
    """
    if exponent is None:
        exponent = sqrt6

    r_start = eps
    r_end = r_horizon - delta

    y0 = frobenius_center_ic(r_start, exponent)

    sol = solve_ivp(
        lambda r, y: ode_system(r, y, m2sq),
        [r_start, r_end],
        y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
        max_step=0.01
    )

    if not sol.success:
        print(f"  Integration failed: {sol.message}")
        return None, None, sol

    u_end = sol.y[0, -1]
    up_end = sol.y[1, -1]

    return u_end, up_end, sol


def extract_connection_coefficients(u_end, up_end, rho, m2sq):
    """
    At r = 2M - delta (rho = -delta, but we use |rho| = delta):

    u(r_end) = M_i1 * v1(rho) + M_i2 * v2(rho)
    u'(r_end) = M_i1 * v1'(rho) + M_i2 * v2'(rho)

    Solve 2x2 linear system for M_i1, M_i2.
    """
    v1, v1p, v2, v2p = frobenius_horizon_basis(rho, m2sq)

    # [v1  v2 ] [M_i1]   [u_end ]
    # [v1p v2p] [M_i2] = [up_end]

    A = np.array([[v1, v2], [v1p, v2p]])
    b = np.array([u_end, up_end])

    det = v1 * v2p - v2 * v1p

    Mi1 = (v2p * u_end - v2 * up_end) / det
    Mi2 = (-v1p * u_end + v1 * up_end) / det

    return Mi1, Mi2, det


def run_connection_matrix(Lambda, eps=1e-3, delta=1e-3):
    """
    Compute full 2x2 connection matrix for given Lambda.
    """
    m2sq = 1.2807 * Lambda**2

    print(f"\n{'='*60}")
    print(f"Lambda = {Lambda}, m2^2 = {m2sq:.6f}")
    print(f"eps = {eps}, delta = {delta}")
    print(f"{'='*60}")

    rho = delta  # distance from horizon (inside, so r < 2M, rho > 0 means approaching from below)
    # Actually rho = r_end - 2M. Since r_end = 2M - delta < 2M, rho = -delta
    # But the Frobenius expansion uses rho = r - 2M, which is negative inside.
    # For the log term: ln|rho| = ln(delta)

    # Let me be more careful about the sign.
    # Inside the horizon: r < 2M, so rho = r - 2M < 0.
    # H(r) = rho/(2+rho) with r = 2+rho, rho<0.
    # The Frobenius basis is in |rho|.

    rho_val = -delta  # r_end - 2M = (2M - delta) - 2M = -delta

    # Row 1: integrate u1 = r^{+sqrt(6)}
    print(f"\n--- u1 (exponent = +sqrt(6) = {sqrt6:.6f}) ---")
    u1_end, u1p_end, sol1 = integrate_from_center(m2sq, eps, delta, exponent=+sqrt6)
    if u1_end is None:
        return None
    print(f"  u1({r_horizon-delta:.4f}) = {u1_end:.12e}")
    print(f"  u1'({r_horizon-delta:.4f}) = {u1p_end:.12e}")

    M11, M12, det1 = extract_connection_coefficients(u1_end, u1p_end, rho_val, m2sq)
    print(f"  M11 = {M11:.12e}")
    print(f"  M12 = {M12:.12e}")
    print(f"  |M12/M11| = {abs(M12/M11):.6e}" if M11 != 0 else "  M11 = 0!")

    # Row 2: integrate u2 = r^{-sqrt(6)}
    print(f"\n--- u2 (exponent = -sqrt(6) = {-sqrt6:.6f}) ---")
    u2_end, u2p_end, sol2 = integrate_from_center(m2sq, eps, delta, exponent=-sqrt6)
    if u2_end is None:
        return None
    print(f"  u2({r_horizon-delta:.4f}) = {u2_end:.12e}")
    print(f"  u2'({r_horizon-delta:.4f}) = {u2p_end:.12e}")

    M21, M22, det2 = extract_connection_coefficients(u2_end, u2p_end, rho_val, m2sq)
    print(f"  M21 = {M21:.12e}")
    print(f"  M22 = {M22:.12e}")
    print(f"  |M22/M21| = {abs(M22/M21):.6e}" if M21 != 0 else "  M21 = 0!")

    # Full matrix
    print(f"\n--- Connection Matrix M ---")
    print(f"  [{M11:+.8e}  {M12:+.8e}]")
    print(f"  [{M21:+.8e}  {M22:+.8e}]")
    det_M = M11 * M22 - M12 * M21
    print(f"  det(M) = {det_M:.8e}")

    # Key diagnostic
    print(f"\n--- KEY DIAGNOSTIC ---")
    if abs(M12) < 1e-6 * abs(M11):
        print(f"  M12 ~ 0: regular-at-center -> regular-at-horizon AUTOMATICALLY!")
    else:
        print(f"  M12 != 0: regular-at-center acquires log-divergence at horizon")
        print(f"  |M12/M11| = {abs(M12/M11):.6e}")

    return {
        "Lambda": Lambda,
        "m2sq": m2sq,
        "eps": eps,
        "delta": delta,
        "M11": float(M11), "M12": float(M12),
        "M21": float(M21), "M22": float(M22),
        "det_M": float(det_M),
        "u1_end": float(u1_end), "u1p_end": float(u1p_end),
        "u2_end": float(u2_end), "u2p_end": float(u2p_end),
        "ratio_M12_M11": float(abs(M12/M11)) if M11 != 0 else None,
        "ratio_M22_M21": float(abs(M22/M21)) if M21 != 0 else None,
        "regular_at_horizon": bool(abs(M12) < 1e-6 * abs(M11)),
    }


def convergence_test(Lambda=1.0):
    """Test convergence w.r.t. eps and delta."""
    m2sq = 1.2807 * Lambda**2

    print("\n" + "="*60)
    print("CONVERGENCE TEST (Lambda=1)")
    print("="*60)

    results = []
    for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
        for delta in [1e-2, 1e-3, 1e-4]:
            u_end, up_end, sol = integrate_from_center(m2sq, eps, delta, exponent=+sqrt6)
            if u_end is not None:
                rho_val = -delta
                Mi1, Mi2, det = extract_connection_coefficients(u_end, up_end, rho_val, m2sq)
                ratio = abs(Mi2/Mi1) if Mi1 != 0 else float('inf')
                results.append({
                    "eps": eps, "delta": delta,
                    "M11": float(Mi1), "M12": float(Mi2),
                    "ratio": float(ratio)
                })
                print(f"  eps={eps:.0e}, delta={delta:.0e}: M11={Mi1:+.8e}, M12={Mi2:+.8e}, |M12/M11|={ratio:.4e}")

    return results


def main():
    # ── Convergence test first ──
    conv_results = convergence_test()

    # ── Main computation: Lambda scan ──
    Lambda_values = [0.5, 1.0, 2.0, 5.0]
    all_results = {}

    for Lambda in Lambda_values:
        result = run_connection_matrix(Lambda, eps=1e-4, delta=1e-4)
        if result is not None:
            all_results[f"Lambda_{Lambda}"] = result

    # ── Also test with different eps/delta for Lambda=1 ──
    # to verify robustness
    print("\n" + "="*60)
    print("ROBUSTNESS: Lambda=1, varying eps/delta")
    print("="*60)
    robustness = []
    for eps, delta in [(1e-3, 1e-3), (1e-4, 1e-4), (1e-5, 1e-5), (1e-3, 1e-4), (1e-4, 1e-3)]:
        r = run_connection_matrix(1.0, eps=eps, delta=delta)
        if r is not None:
            robustness.append(r)

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Lambda':>8} {'m2sq':>10} {'M11':>14} {'M12':>14} {'|M12/M11|':>14} {'Regular?':>10}")
    print("-" * 76)
    for key, r in all_results.items():
        print(f"{r['Lambda']:8.2f} {r['m2sq']:10.4f} {r['M11']:+14.6e} {r['M12']:+14.6e} {r['ratio_M12_M11']:14.6e} {'YES' if r['regular_at_horizon'] else 'NO':>10}")

    # ── Save ──
    output = {
        "description": "Connection matrix for massive spin-2 on Schwarzschild interior",
        "operator": "H u'' + (H'+2H/r) u' + (m2^2 - 6H/r^2) u = 0",
        "H": "1 - 2M/r, M=1",
        "mass_formula": "m2^2 = 1.2807 * Lambda^2",
        "frobenius_center_exponents": [float(sqrt6), float(-sqrt6)],
        "frobenius_horizon_exponents": [0, 0],
        "Lambda_scan": all_results,
        "convergence_test": conv_results,
        "robustness_tests": [r for r in robustness if r is not None],
    }

    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "gap_g1", "connection_matrix.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
