#!/usr/bin/env python3
"""
CJ in 2D: test the bridge formula dimension prediction.

Formula: CJ = [2d/((d-1)(2d+1)!)] × (angular×volume factor) × N^{2d/(2d+1)} × E² × T^d

For d=2: CJ_2D = (4/120) × F_2 × N^{4/5} × E² × T²

where F_2 is the 2D angular × volume factor (to be determined empirically
or derived analytically).

In 2D: 'sky' = S^0 = {±1}, angular integral = 2×E² = 2E².
Volume = T²/2. Combined: 2 × (1/2) = 1 (normalized per T²).

Prediction: CJ_2D = (1/30) × N^{4/5} × E² × T²
"""
import numpy as np
import time


def sprinkle_2d_diamond(N, T, rng):
    """Sprinkle N points in 2D causal diamond |t| + |x| < T/2."""
    pts = []
    while len(pts) < N:
        batch = rng.uniform(-T/2, T/2, size=(max(4096, 4*N), 2))
        mask = np.abs(batch[:, 0]) + np.abs(batch[:, 1]) < T/2
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:N], dtype=np.float64)
    return arr[np.argsort(arr[:, 0])]


def flat_causal_2d(pts, i):
    """Flat 2D causal predicate: j ≺ i iff Δt > |Δx|."""
    x = pts[i]
    y = pts[:i]
    dt = x[0] - y[:, 0]
    dx = np.abs(x[1] - y[:, 1])
    return (dt > 1e-12) & (dx < dt)


def curved_causal_2d(pts, i, eps):
    """Curved 2D 'pp-wave' predicate.

    Metric: ds² = -(1 + ε x²) dt² + dx²
    Causal: (1 + ε x_mid²) Δt² > Δx²
    """
    x = pts[i]
    y = pts[:i]
    dt = x[0] - y[:, 0]
    dx = x[1] - y[:, 1]
    x_mid = 0.5 * (x[1] + y[:, 1])
    sigma = (1.0 + eps * x_mid**2) * dt**2 - dx**2
    return (dt > 1e-12) & (sigma > 1e-12)


def build_hasse_2d(pts, pred_fn):
    """Build Hasse diagram with bitset transitive reduction."""
    n = len(pts)
    parents = [np.empty(0, dtype=np.int32) for _ in range(n)]
    children = [[] for _ in range(n)]
    past_bits = [0] * n

    for i in range(n):
        rel = np.flatnonzero(pred_fn(pts, i))
        if len(rel) == 0:
            continue
        covered = 0
        direct = []
        for j in rel[::-1]:
            j = int(j)
            bit = 1 << j
            if covered & bit:
                continue
            direct.append(j)
            covered |= past_bits[j] | bit
        direct.sort()
        parents[i] = np.array(direct, dtype=np.int32)
        pb = 0
        for j in direct:
            children[j].append(i)
            pb |= past_bits[j] | (1 << j)
        past_bits[i] = pb

    return parents, children


def Y_from_hasse(parents, n):
    """Link score Y = log2(p_down * p_up + 1)."""
    ch = [[] for _ in range(n)]
    for i in range(n):
        for j in parents[i]:
            ch[int(j)].append(i)
    p_d = np.ones(n)
    p_u = np.ones(n)
    for i in range(n):
        if len(parents[i]) > 0:
            p_d[i] = np.sum(p_d[parents[i]]) + 1
    for i in range(n-1, -1, -1):
        if ch[i]:
            p_u[i] = np.sum(p_u[ch[i]]) + 1
    return np.log2(p_d * p_u + 1)


def bulk_mask_2d(pts, T, zeta=0.15):
    """Bulk mask: exclude boundary elements."""
    slack = T/2 - np.abs(pts[:, 0]) - np.abs(pts[:, 1])
    return slack > zeta * T


def compute_CJ_2d(Y0, YC, pts, par0, T, zeta=0.15):
    """CJ = Σ_B w_B Cov_B(|X|, δY²) in 2D."""
    n = len(pts)
    bm = bulk_mask_2d(pts, T, zeta)
    delta = YC - Y0
    X = Y0[bm] - np.mean(Y0[bm])
    dY2 = delta[bm]**2

    # Simple stratification: time bins × depth bins
    tau = 2 * pts[:, 0] / T
    tau_bin = np.clip(np.floor((tau + 1) * 2.5).astype(int), 0, 4)
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if len(par0[i]) > 0:
            depth[i] = int(np.max(depth[par0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_bin = np.clip((depth * 3) // (max_d + 1), 0, 2)
    strata = tau_bin * 3 + depth_bin

    strata_m = strata[bm]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = np.mean(np.abs(X[idx]) * dY2[idx]) - np.mean(np.abs(X[idx])) * np.mean(dY2[idx])
        total += w * cov
    return float(total)


if __name__ == '__main__':
    from math import factorial

    T = 1.0
    M_SEEDS = 30

    # Formula prediction for d=2:
    d = 2
    A_pred = 2*d / ((d-1) * factorial(2*d+1))  # 4/120 = 1/30
    alpha_pred = 2*d / (2*d+1)  # 4/5 = 0.80

    # Angular factor for d=2:
    # int_{S^0} (E n)^2 = 2E^2
    # Volume_2D = T^2/2
    # Combined normalized: 2 × (1/2) = 1
    # But in d=4: pi^2/45 = (8pi/15)(pi/24). For d=2 we need the analogous factor.
    # In d=2: angular int = 2, volume coeff = 1/2 (V = T^2/2)
    # Factor = 2 × (1/2) = 1? Or include the d-dependent normalization differently?
    #
    # Let me just measure CJ_2D and extract A empirically.

    print("2D CJ Bridge Formula Test")
    print(f"Formula: CJ_2D = A × N^(4/5) × E^2 × T^2")
    print(f"Prediction: A = 4/120 = {A_pred:.6f}")
    print("=" * 70)

    for eps in [1.0, 2.0, 3.0, 5.0]:
        E2 = eps**2  # In 2D: E^2 = (R_0101)^2 = eps^2
        print(f"\neps={eps}, E^2={E2}")
        print(f"{'N':>6s}  {'CJ':>10s}  {'A_meas':>12s}  {'A_pred':>10s}  {'ratio':>7s}")

        for N in [200, 500, 1000, 2000]:
            cj_vals = []
            for s in range(M_SEEDS):
                rng = np.random.default_rng(7700000 + s + int(eps*1000) + N)
                pts = sprinkle_2d_diamond(N, T, rng)
                par0, ch0 = build_hasse_2d(pts, lambda P, i: flat_causal_2d(P, i))
                parC, chC = build_hasse_2d(pts, lambda P, i: curved_causal_2d(P, i, eps))
                Y0 = Y_from_hasse(par0, N)
                YC = Y_from_hasse(parC, N)
                cj = compute_CJ_2d(Y0, YC, pts, par0, T)
                cj_vals.append(cj)

            mean_cj = np.mean(cj_vals)
            # Extract A: CJ = A × N^alpha × E^2 × T^2
            # Use alpha = 4/5 (predicted) and see if A matches
            A_meas = mean_cj / (N**alpha_pred * E2 * T**2)
            ratio = A_meas / A_pred if A_pred > 0 else 0
            print(f"{N:6d}  {mean_cj:10.6f}  {A_meas:12.4e}  {A_pred:10.4e}  {ratio:7.3f}")
