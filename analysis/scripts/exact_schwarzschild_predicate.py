#!/usr/bin/env python3
"""
Exact Schwarzschild causal predicate via numerical geodesic interval.

For two points in a local Schwarzschild patch, compute whether they are
causally related by evaluating the geodesic interval sigma(x,y) numerically.

The approach: instead of solving the full geodesic ODE (expensive), use a
HIGHER-ORDER synge expansion of the world function to 4th order in Riemann.
This captures the curvature gradient (nabla_R) that the quadratic jet misses.

Synge world function expansion in normal coordinates:
sigma(x,y) = eta_ab dx^a dx^b
           + (1/3) R_acbd x^c_mid x^d_mid dx^a dx^b
           + (1/12) nabla_e R_acbd x^c_mid x^d_mid x^e_mid dx^a dx^b
           + (1/20) [nabla_ef R_acbd + (2/9) R_acbe R^e_fgd] x^c_mid x^d_mid x^e_mid x^f_mid dx^a dx^b
           + ...

For Schwarzschild at r_0, we need:
1. R_abcd (Riemann tensor at r_0) - already have
2. nabla_e R_abcd (covariant derivative of Riemann at r_0)
3. R_acbe R^e_fgd (quadratic Riemann terms)

This gives a 4th-order predicate that should be much more accurate than
the 2nd-order jet predicate.
"""
import numpy as np


def schwarzschild_riemann_and_gradient(M, r0):
    """Compute Riemann tensor AND its covariant gradient at r=r0 in Schwarzschild.

    Uses orthonormal frame {e_t, e_r, e_theta, e_phi} at (r=r0, theta=pi/2).

    Returns:
        R: (4,4,4,4) Riemann tensor in orthonormal frame
        dR: (4,4,4,4,4) nabla_e R_abcd in orthonormal frame (5-index)
    """
    f = 1 - 2*M/r0
    sqf = np.sqrt(f)

    # Orthonormal frame Riemann components for Schwarzschild:
    # (indices: 0=t, 1=r, 2=theta, 3=phi)
    # Non-redundant:
    # R_0101 = -2M/r0^3 (electric radial)
    # R_0202 = R_0303 = M/r0^3 (electric tangential)
    # R_1212 = R_1313 = -M/r0^3 (magnetic... actually spatial tidal)
    # R_2323 = 2M/r0^3

    R = np.zeros((4,4,4,4))

    # Electric part: R_0i0j
    R[0,1,0,1] = -2*M/r0**3
    R[0,2,0,2] = M/r0**3
    R[0,3,0,3] = M/r0**3

    # Spatial part: R_ijkl
    R[1,2,1,2] = -M/r0**3
    R[1,3,1,3] = -M/r0**3
    R[2,3,2,3] = 2*M/r0**3

    # Fill all symmetries
    def fill_symmetries(R):
        Rf = np.copy(R)
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        v = R[a,b,c,d]
                        if abs(v) > 1e-20:
                            Rf[b,a,c,d] = -v
                            Rf[a,b,d,c] = -v
                            Rf[b,a,d,c] = v
                            Rf[c,d,a,b] = v
                            Rf[d,c,a,b] = v
                            Rf[c,d,b,a] = -v
                            Rf[d,c,b,a] = -v
        return Rf

    R = fill_symmetries(R)

    # Covariant derivative of Riemann in radial direction:
    # nabla_r R_0101 = d/dr(-2M/r^3) × (correction for frame) = +6M/r0^4 × (1/sqf)
    # (the 1/sqf comes from converting coordinate r-derivative to orthonormal frame)
    # nabla_r R_0202 = d/dr(M/r^3) × (1/sqf) = -3M/r0^4 × (1/sqf)
    # etc.

    # In orthonormal frame, nabla_1 (radial direction):
    dR = np.zeros((4,4,4,4,4))  # dR[e,a,b,c,d] = nabla_e R_abcd

    # Only radial gradient is nonzero (spherical symmetry):
    # nabla_r R_0101 = 6M/r0^4 / sqf (from d/dr of -2M/r^3 in orth. frame)
    dR[1,0,1,0,1] = 6*M/r0**4 / sqf
    dR[1,0,2,0,2] = -3*M/r0**4 / sqf
    dR[1,0,3,0,3] = -3*M/r0**4 / sqf
    dR[1,1,2,1,2] = 3*M/r0**4 / sqf
    dR[1,1,3,1,3] = 3*M/r0**4 / sqf
    dR[1,2,3,2,3] = -6*M/r0**4 / sqf

    # Fill symmetries of dR (R-part symmetries, e index is separate)
    for e in range(4):
        dR[e] = fill_symmetries(dR[e])

    return R, dR


def higher_order_interval(dx, x_mid, R, dR):
    """Compute geodesic interval to 3rd order in curvature expansion.

    sigma = eta_ab dx^a dx^b
          + (1/3) R_acbd x_mid^c x_mid^d dx^a dx^b
          + (1/12) nabla_e R_acbd x_mid^c x_mid^d x_mid^e dx^a dx^b

    Args:
        dx: separation vector (4,)
        x_mid: midpoint vector (4,)
        R: Riemann tensor (4,4,4,4)
        dR: gradient of Riemann (4,4,4,4,4) = nabla_e R_abcd

    Returns:
        sigma: geodesic interval (positive = timelike)
    """
    eta = np.array([-1., 1., 1., 1.])

    # 0th order: Minkowski interval
    sigma_0 = np.sum(eta * dx * dx)

    # 2nd order: Riemann correction (= jet predicate)
    sigma_2 = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    sigma_2 += R[a,c,b,d] * x_mid[c] * x_mid[d] * dx[a] * dx[b]
    sigma_2 /= 3.0

    # 3rd order: gradient correction (NEW — this is what jet misses)
    sigma_3 = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    for e in range(4):
                        sigma_3 += dR[e,a,c,b,d] * x_mid[c] * x_mid[d] * x_mid[e] * dx[a] * dx[b]
    sigma_3 /= 12.0

    return sigma_0 + sigma_2 + sigma_3


def exact_sch_preds_order3(pts, i, R, dR, tol=1e-12):
    """Higher-order Schwarzschild causal predicate.

    Uses 3rd-order Synge expansion (includes nabla_R correction).
    """
    x = pts[i]
    y = pts[:i]

    dx = x - y  # (i-1, 4)
    x_mid = 0.5 * (x + y)  # midpoint
    dt = dx[:, 0]

    # Vectorized computation for all predecessors
    mask = np.zeros(i, dtype=bool)

    for j in range(i):
        if dt[j] <= tol:
            continue
        sigma = higher_order_interval(dx[j], x_mid[j], R, dR)
        if sigma > tol:
            mask[j] = True

    return mask


if __name__ == '__main__':
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))
    from run_universal import (sprinkle_local_diamond, minkowski_preds,
        build_hasse_from_predicate, bulk_mask, jet_preds,
        riemann_schwarzschild_local)
    from sj_entropy_bridge import compute_CJ_from_hasse

    N, T = 2000, 1.0
    M_sch, r0 = 0.05, 0.50
    M_SEEDS = 15

    # Compute Riemann + gradient
    R_sch, dR_sch = schwarzschild_riemann_and_gradient(M_sch, r0)
    R_jet = riemann_schwarzschild_local(M_sch, r0)

    print("EXACT vs JET Schwarzschild predicate comparison")
    print(f"N={N}, M={M_sch}, r0={r0}, M_seeds={M_SEEDS}")
    print("="*60)

    cj_jet_v = []
    cj_o3_v = []

    for s in range(M_SEEDS):
        seed = 9500000 + s
        t0 = time.time()
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        par0, _ = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))

        # Jet predicate (order 2)
        parJ, _ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_jet))
        cj_jet = compute_CJ_from_hasse(pts, par0, parJ, T)
        cj_jet_v.append(cj_jet)

        # Order-3 predicate (includes nabla_R)
        par3, _ = build_hasse_from_predicate(
            pts, lambda P, i: exact_sch_preds_order3(P, i, R_sch, dR_sch))
        cj_o3 = compute_CJ_from_hasse(pts, par0, par3, T)
        cj_o3_v.append(cj_o3)

        dt = time.time() - t0
        print(f"  seed {s:2d}: CJ_jet={cj_jet:.6f}  CJ_o3={cj_o3:.6f}  "
              f"ratio={cj_o3/cj_jet:.3f}  ({dt:.1f}s)")

    mean_jet = np.mean(cj_jet_v)
    mean_o3 = np.mean(cj_o3_v)
    se_jet = np.std(cj_jet_v)/np.sqrt(len(cj_jet_v))
    se_o3 = np.std(cj_o3_v)/np.sqrt(len(cj_o3_v))

    print(f"\n<CJ_jet> = {mean_jet:.6f} +/- {se_jet:.6f}")
    print(f"<CJ_o3>  = {mean_o3:.6f} +/- {se_o3:.6f}")
    print(f"Ratio o3/jet = {mean_o3/mean_jet:.3f}")

    # Compare with 1/120 formula
    E2B2 = 6 * M_sch**2 / r0**6
    V = np.pi * T**4 / 24
    sigma_0 = 0.299 * N**0.25
    cj_formula = sigma_0 / (np.pi * 120) * E2B2 * V

    print(f"\n1/120 formula prediction: {cj_formula:.6f}")
    print(f"Ratio CJ_jet/formula  = {mean_jet/cj_formula:.3f}")
    print(f"Ratio CJ_o3/formula   = {mean_o3/cj_formula:.3f}")
    print(f"\nIf o3 brings ratio closer to 1.0 -> gradient correction helps")
