#!/usr/bin/env python3
"""
Bridge Level 2 at N=10000: Physical Pauli-Jordan (Johnston link matrix).
Focused: pp-wave eps=3 + Schwarzschild, M=5 seeds.
Uses GPU (CuPy) for eigendecomposition if available.
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)

# Try GPU
USE_GPU = False
try:
    _c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.isdir(_c):
        os.add_dll_directory(_c)
    import cupy as cp
    USE_GPU = True
    print("GPU (CuPy) available — using for eigendecomposition")
except Exception as e:
    print(f"No GPU: {e} — using numpy")

N = 10000
T = 1.0
ZETA = 0.15
M_SEEDS = 5

SCH_M = 0.05
SCH_R0 = 0.50
SCH_R_ABCD = riemann_schwarzschild_local(SCH_M, SCH_R0)
SCH_E2 = 6.0 * SCH_M**2 / SCH_R0**6

PPW_EPS = 3.0


def hasse_to_link_matrix(parents, children, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def compute_spectral_det(L):
    n = L.shape[0]
    PJ = L - L.T
    threshold = math.sqrt(n) / (4 * math.pi)

    if USE_GPU:
        H_gpu = cp.asarray(1j * PJ)
        evals = cp.asnumpy(cp.linalg.eigvalsh(H_gpu))
        del H_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        H = 1j * PJ
        evals = np.linalg.eigvalsh(H)

    sigma = np.sort(evals[evals > 1e-14])[::-1]
    sigma_trunc = sigma[sigma > threshold]
    logdet = float(np.sum(np.log(sigma_trunc))) if len(sigma_trunc) > 0 else 0.0
    return {
        'n_modes_total': len(sigma),
        'n_modes_retained': len(sigma_trunc),
        'threshold': float(threshold),
        'logdet': logdet,
        'sigma_max': float(sigma[0]) if len(sigma) > 0 else 0.0,
    }


def Y_from_graph(par, ch):
    n = len(par)
    p_down = np.ones(n, dtype=np.float64)
    p_up = np.ones(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            p_down[i] = np.sum(p_down[par[i]]) + 1
    for i in range(n - 1, -1, -1):
        if ch[i] is not None and len(ch[i]) > 0:
            p_up[i] = np.sum(p_up[ch[i]]) + 1
    return np.log2(p_down * p_up + 1)


def make_strata(pts, par0, T):
    tau_hat = 2 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if par0[i] is not None and len(par0[i]) > 0:
            depth[i] = int(np.max(depth[par0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


def compute_CJ(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]
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


def run_seed(seed_idx, geometry, eps=None, base_seed=9400000):
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    t0 = time.time()

    # Flat Hasse
    print(f"    [{seed_idx}] flat Hasse...", end='', flush=True)
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    L_flat = hasse_to_link_matrix(par0, ch0, N)
    print(f" {time.time()-t0:.0f}s", end='', flush=True)

    # Curved Hasse
    print(f"  curved...", end='', flush=True)
    if geometry == 'ppwave':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        E2 = eps**2 / 2.0
    else:
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, SCH_R_ABCD))
        E2 = SCH_E2
    L_curved = hasse_to_link_matrix(parC, chC, N)
    print(f" {time.time()-t0:.0f}s", end='', flush=True)

    # Spectral
    print(f"  eigen({'GPU' if USE_GPU else 'CPU'})...", end='', flush=True)
    spec_flat = compute_spectral_det(L_flat)
    spec_curved = compute_spectral_det(L_curved)
    delta_gamma = 0.5 * (spec_curved['logdet'] - spec_flat['logdet'])
    print(f" {time.time()-t0:.0f}s", end='', flush=True)

    # CJ
    Y0 = Y_from_graph(par0, ch0)
    YC = Y_from_graph(parC, chC)
    mask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par0, T)
    cj = compute_CJ(Y0, YC - Y0, mask, strata)
    sigma0 = float(np.std(Y0[mask]))
    t_total = time.time() - t0

    print(f"  DONE {t_total:.0f}s  dG={delta_gamma:+.4f}  CJ={cj:.6f}", flush=True)

    return {
        'seed': seed_idx, 'geometry': geometry, 'eps': eps, 'E2': E2, 'N': N,
        'delta_gamma': delta_gamma, 'CJ': cj,
        'dG_over_E2': delta_gamma / E2 if E2 > 0 else 0,
        'sigma0': sigma0,
        'n_modes_flat': spec_flat['n_modes_retained'],
        'n_modes_total': spec_flat['n_modes_total'],
        'n_links_flat': int(L_flat.sum()),
        'n_links_curved': int(L_curved.sum()),
        'logdet_flat': spec_flat['logdet'],
        'logdet_curved': spec_curved['logdet'],
        'time_s': t_total,
    }


if __name__ == '__main__':
    print("=" * 72)
    print(f"BRIDGE LEVEL 2 @ N={N}: Johnston link matrix PJ")
    print(f"M={M_SEEDS}, T={T}, GPU={'YES' if USE_GPU else 'NO'}")
    print(f"Threshold = {math.sqrt(N)/(4*math.pi):.2f}")
    print("=" * 72, flush=True)

    results = {}

    # PP-WAVE eps=3
    E2 = PPW_EPS**2 / 2.0
    print(f"\n{'='*60}")
    print(f"PP-WAVE eps={PPW_EPS} (E^2={E2:.2f})")
    print(f"{'='*60}", flush=True)

    dg, cj = [], []
    seeds_ppw = []
    for si in range(M_SEEDS):
        res = run_seed(si, 'ppwave', eps=PPW_EPS)
        dg.append(res['delta_gamma'])
        cj.append(res['CJ'])
        seeds_ppw.append(res)

    dg = np.array(dg); cj = np.array(cj)
    corr = float(np.corrcoef(dg, cj)[0,1]) if len(dg) >= 3 else 0.0
    t_stat = float(dg.mean() / (dg.std(ddof=1)/math.sqrt(M_SEEDS) + 1e-30))
    results['ppwave'] = {
        'eps': PPW_EPS, 'E2': E2, 'N': N,
        'dG_mean': float(dg.mean()), 'dG_se': float(dg.std(ddof=1)/math.sqrt(M_SEEDS)),
        'CJ_mean': float(cj.mean()), 'CJ_se': float(cj.std(ddof=1)/math.sqrt(M_SEEDS)),
        'dG_over_E2': float(dg.mean()/E2),
        'corr': corr, 't_stat': t_stat,
        'seeds': seeds_ppw,
    }
    print(f"\n  SUMMARY ppw: dG={dg.mean():+.3f}±{dg.std(ddof=1)/math.sqrt(M_SEEDS):.3f} "
          f"(t={t_stat:+.2f})  CJ={cj.mean():.5f}  corr={corr:+.3f}", flush=True)

    # SCHWARZSCHILD
    print(f"\n{'='*60}")
    print(f"SCHWARZSCHILD E^2={SCH_E2:.4f}")
    print(f"{'='*60}", flush=True)

    dg_s, cj_s = [], []
    seeds_sch = []
    for si in range(M_SEEDS):
        res = run_seed(si, 'schwarzschild')
        dg_s.append(res['delta_gamma'])
        cj_s.append(res['CJ'])
        seeds_sch.append(res)

    dg_s = np.array(dg_s); cj_s = np.array(cj_s)
    corr_s = float(np.corrcoef(dg_s, cj_s)[0,1]) if len(dg_s) >= 3 else 0.0
    t_stat_s = float(dg_s.mean() / (dg_s.std(ddof=1)/math.sqrt(M_SEEDS) + 1e-30))
    results['schwarzschild'] = {
        'N': N, 'E2': SCH_E2,
        'dG_mean': float(dg_s.mean()), 'dG_se': float(dg_s.std(ddof=1)/math.sqrt(M_SEEDS)),
        'CJ_mean': float(cj_s.mean()), 'CJ_se': float(cj_s.std(ddof=1)/math.sqrt(M_SEEDS)),
        'dG_over_E2': float(dg_s.mean()/SCH_E2),
        'corr': corr_s, 't_stat': t_stat_s,
        'seeds': seeds_sch,
    }
    print(f"\n  SUMMARY sch: dG={dg_s.mean():+.3f}±{dg_s.std(ddof=1)/math.sqrt(M_SEEDS):.3f} "
          f"(t={t_stat_s:+.2f})  CJ={cj_s.mean():.5f}  corr={corr_s:+.3f}", flush=True)

    # N-SCALING TABLE
    print(f"\n{'='*72}")
    print("N-SCALING TABLE (pp-wave eps=3)")
    print(f"{'='*72}")
    print(f"  N=2000:  dG=+0.666±0.253 (t=2.63)   CJ=0.02542  corr=+0.43")
    print(f"  N=5000:  dG=+2.644±0.526 (t=5.03)   CJ=0.05840  corr=-0.16")
    print(f"  N=10000: dG={dg.mean():+.3f}±{dg.std(ddof=1)/math.sqrt(M_SEEDS):.3f} "
          f"(t={t_stat:+.2f})   CJ={cj.mean():.5f}  corr={corr:+.3f}")

    print(f"\nN-SCALING TABLE (Schwarzschild)")
    print(f"  N=2000:  dG=-0.010±0.119 (t=-0.08)  CJ=0.00186  corr=+0.11")
    print(f"  N=5000:  dG=-0.360±0.218 (t=-1.66)  CJ=0.00344  corr=-0.27")
    print(f"  N=10000: dG={dg_s.mean():+.3f}±{dg_s.std(ddof=1)/math.sqrt(M_SEEDS):.3f} "
          f"(t={t_stat_s:+.2f})  CJ={cj_s.mean():.5f}  corr={corr_s:+.3f}")

    # Save
    outfile = 'analysis/fnd1_data/bridge_level2_n10000.json'
    for key in results:
        if 'seeds' in results[key]:
            for sr in results[key]['seeds']:
                for k in list(sr.keys()):
                    if isinstance(sr[k], np.ndarray):
                        sr[k] = sr[k].tolist()
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\nSaved to {outfile}", flush=True)
