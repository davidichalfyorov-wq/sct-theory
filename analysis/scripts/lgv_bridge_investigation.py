#!/usr/bin/env python3
"""
LGV/Talaska Free-Fermion Minor Bridge Investigation.

Replicates the independent pilot (r=0.700 at N=320) at production parameters:
- N=2000 with our EXACT pp-wave and Schwarzschild predicates
- M=20 seeds
- Full (k, t) parameter scan
- Also tests Schwarzschild (not just pp-wave)

LGV observable: Δlog|det M| = log|det G_curved[S,T]| - log|det G_flat[S,T]|
where G = (I - t·A)^{-1} and M = G[sources, sinks] is a k×k minor.

The LGV lemma guarantees det M counts nonintersecting k-path families.

Key questions:
1. Does corr(CJ, LGV) > 0.5 survive at production N=2000?
2. Is it stable across (k, t) parameters?
3. Does it work on Schwarzschild (not just pp-wave)?
4. Is it a genuine bridge or just another CJ proxy?
"""
import sys, os, time, json
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)


def hasse_to_adjacency(parents, n):
    """Build adjacency matrix A[j,i] = 1 if j is Hasse parent of i."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                A[int(j), i] = 1.0
    return A


def Y_from_graph(par, ch_or_none, n):
    """Compute link score Y = log2(p_down * p_up + 1)."""
    # Build children if not given
    ch = [[] for _ in range(n)]
    for i in range(n):
        if par[i] is not None:
            for j in par[i]:
                ch[int(j)].append(i)

    p_down = np.ones(n, dtype=np.float64)
    p_up = np.ones(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            p_down[i] = np.sum(p_down[list(par[i])]) + 1
    for i in range(n - 1, -1, -1):
        if ch[i]:
            p_up[i] = np.sum(p_up[ch[i]]) + 1
    return np.log2(p_down * p_up + 1)


def compute_CJ(Y0, YC, pts, par0, T, zeta=0.15):
    """Compute CJ = Σ_B w_B Cov_B(|X|, δY²)."""
    n = len(pts)
    bmask = bulk_mask(pts, T, zeta)
    delta = YC - Y0

    # Strata
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if par0[i] is not None and len(par0[i]) > 0:
            depth[i] = int(np.max(depth[list(par0[i])])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    strata = tau_bin * 9 + rho_bin * 3 + depth_terc

    X = Y0[bmask] - np.mean(Y0[bmask])
    dY2 = delta[bmask] ** 2
    strata_m = strata[bmask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = (np.mean(np.abs(X[idx]) * dY2[idx])
               - np.mean(np.abs(X[idx])) * np.mean(dY2[idx]))
        total += w * cov
    return float(total)


def select_boundary_sources_sinks(pts, bmask, k, mode='temporal'):
    """Select k source and k sink elements from bulk.

    mode='temporal': earliest/latest bulk elements (the independent choice)
    mode='spatial': innermost/outermost bulk elements
    mode='random': random bulk elements (control)
    """
    idx = np.where(bmask)[0]
    if mode == 'temporal':
        past = idx[np.argsort(pts[idx, 0])][:k]
        future = idx[np.argsort(-pts[idx, 0])][:k]
    elif mode == 'spatial':
        r = np.linalg.norm(pts[idx, 1:], axis=1)
        inner = idx[np.argsort(r)][:k]
        outer = idx[np.argsort(-r)][:k]
        past, future = inner, outer
    elif mode == 'random':
        perm = np.random.permutation(len(idx))
        past = idx[perm[:k]]
        future = idx[perm[k:2*k]]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.sort(past), np.sort(future)


def compute_lgv_minor(A_flat, A_curved, sources, sinks, t):
    """Compute Δlog|det M| for LGV minor.

    G = (I - t·A)^{-1}, M = G[sources, sinks]
    Returns log|det M_curved| - log|det M_flat|
    """
    n = A_flat.shape[0]
    I = np.eye(n)

    # Since A is nilpotent, (I - tA) is always invertible
    # Use solve for numerical stability
    G_flat = la.solve(I - t * A_flat, I, assume_a='gen')
    G_curved = la.solve(I - t * A_curved, I, assume_a='gen')

    M_flat = G_flat[np.ix_(sources, sinks)]
    M_curved = G_curved[np.ix_(sources, sinks)]

    _, logabs_flat = np.linalg.slogdet(M_flat)
    _, logabs_curved = np.linalg.slogdet(M_curved)

    return float(logabs_curved - logabs_flat)


def run_lgv_scan(N=2000, M_seeds=20, T=1.0, geometry='ppwave', eps=3.0,
                  M_sch=0.05, r0_sch=0.50,
                  k_values=[2, 3, 5, 8],
                  t_values=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
                  modes=['temporal']):
    """Full LGV parameter scan with CJ correlation."""
    print("=" * 72)
    print(f"LGV Bridge Investigation  N={N}  M={M_seeds}  {geometry}")
    print(f"k values: {k_values}")
    print(f"t values: {t_values}")
    print(f"modes: {modes}")
    print("=" * 72)

    if geometry == 'schwarzschild':
        R_abcd = riemann_schwarzschild_local(M_sch, r0_sch)

    # Collect data per seed
    seed_data = []

    for s in range(M_seeds):
        seed = 8500000 + s
        t0 = time.time()

        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        # Build Hasse (once each)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))

        if geometry == 'ppwave':
            parC, chC = build_hasse_from_predicate(
                pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        else:
            parC, chC = build_hasse_from_predicate(
                pts, lambda P, i: jet_preds(P, i, R_abcd))

        # Y and CJ (once)
        Y0 = Y_from_graph(par0, None, N)
        YC = Y_from_graph(parC, None, N)
        cj = compute_CJ(Y0, YC, pts, par0, T)

        # Adjacency matrices (once)
        A0 = hasse_to_adjacency(par0, N)
        AC = hasse_to_adjacency(parC, N)

        bmask = bulk_mask(pts, T, 0.15)

        # LGV scan over (k, t, mode)
        lgv_results = {}
        for mode in modes:
            for k in k_values:
                sources, sinks = select_boundary_sources_sinks(pts, bmask, k, mode)
                for t_val in t_values:
                    key = f"k{k}_t{t_val}_m{mode}"
                    lgv = compute_lgv_minor(A0, AC, sources, sinks, t_val)
                    lgv_results[key] = lgv

        dt = time.time() - t0
        entry = {'seed': seed, 'CJ': cj, 'lgv': lgv_results, 'time': dt}
        seed_data.append(entry)

        # Print progress with one representative (k=3, t=0.15)
        rep_key = f"k3_t0.15_mtemporal"
        rep_val = lgv_results.get(rep_key, 0)
        print(f"  seed {s:2d}: CJ={cj:+.6f}  LGV(k3,t.15)={rep_val:+.6f}  ({dt:.1f}s)")

    # Compute correlations for all (k, t, mode) combinations
    print()
    print("=" * 72)
    print("CORRELATION TABLE: corr(CJ, LGV)")
    print("=" * 72)

    cj_vals = [d['CJ'] for d in seed_data]

    best_corr = -1
    best_key = ""

    print(f"{'k':>3s}  {'t':>5s}  {'mode':>8s}  {'corr':>7s}  {'<LGV>':>10s}  {'σ(LGV)':>10s}")
    print("-" * 55)

    for mode in modes:
        for k in k_values:
            for t_val in t_values:
                key = f"k{k}_t{t_val}_m{mode}"
                lgv_vals = [d['lgv'][key] for d in seed_data]

                if np.std(lgv_vals) < 1e-15 or np.std(cj_vals) < 1e-15:
                    r = 0.0
                else:
                    r = float(np.corrcoef(cj_vals, lgv_vals)[0, 1])

                m_lgv = np.mean(lgv_vals)
                s_lgv = np.std(lgv_vals)

                print(f"{k:3d}  {t_val:5.2f}  {mode:>8s}  {r:+7.3f}  {m_lgv:+10.5f}  {s_lgv:10.5f}")

                if abs(r) > best_corr:
                    best_corr = abs(r)
                    best_key = key
                    best_r = r

    print()
    print(f"Best: {best_key}  corr={best_r:+.3f}")
    print(f"CJ mean: {np.mean(cj_vals):+.6f} ± {np.std(cj_vals)/np.sqrt(len(cj_vals)):.6f}")

    # Verdict
    print()
    if best_corr > 0.5:
        print(f"★ LGV SURVIVES: best |corr| = {best_corr:.3f} > 0.5")
    elif best_corr > 0.3:
        print(f"? LGV MARGINAL: best |corr| = {best_corr:.3f}")
    else:
        print(f"✗ LGV DEAD: best |corr| = {best_corr:.3f} < 0.3")

    return seed_data, cj_vals


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--M', type=int, default=20)
    parser.add_argument('--geometry', default='ppwave')
    parser.add_argument('--eps', type=float, default=3.0)
    parser.add_argument('--M_sch', type=float, default=0.05)
    parser.add_argument('--r0', type=float, default=0.50)
    args = parser.parse_args()

    data, cjs = run_lgv_scan(
        N=args.N, M_seeds=args.M, geometry=args.geometry,
        eps=args.eps, M_sch=args.M_sch, r0_sch=args.r0,
        k_values=[2, 3, 5, 8],
        t_values=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        modes=['temporal'])

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'fnd1_data',
                       f'lgv_bridge_{args.geometry}_N{args.N}.json')
    # Serialize
    save_data = []
    for d in data:
        save_data.append({'seed': d['seed'], 'CJ': d['CJ'],
                          'lgv': d['lgv'], 'time': d['time']})
    with open(out, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out}")
