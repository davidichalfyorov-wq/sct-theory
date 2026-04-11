#!/usr/bin/env python3
"""
Deep dive tests on three findings from bridge data:

Test 1: corr(CJ, delta_links) across N and geometries
  - Is the +0.69 correlation on Sch stable across N?
  - Does it hold on ppw at different eps?
  - What drives CJ: link count change or spatial pattern?

Test 2: Precise N-scaling exponents for CJ
  - ppw eps=3: CJ ~ N^?
  - Sch: CJ ~ N^?
  - Are they truly different?

Test 3: Weyl-type discrimination via SJ vacuum response
  - frac_positive(dI) at eps=0.5, 1, 2, 3, 5 on ppw
  - Compare with Sch
  - Is frac_positive a Weyl-type classifier?
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)

T = 1.0
ZETA = 0.15
M_SEEDS = 20

SCH_M = 0.05
SCH_R0 = 0.50
SCH_R_ABCD = riemann_schwarzschild_local(SCH_M, SCH_R0)
SCH_E2 = 6.0 * SCH_M**2 / SCH_R0**6


def hasse_to_link_matrix(parents, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def compute_sj_wightman(L):
    n = L.shape[0]
    PJ = L - L.T
    H = 1j * PJ
    threshold = math.sqrt(n) / (4 * math.pi)
    evals, evecs = np.linalg.eigh(H)
    use_mask = evals > threshold
    n_modes = int(use_mask.sum())
    if n_modes == 0:
        return np.zeros((n, n)), 0
    V_pos = evecs[:, use_mask]
    sigma_pos = evals[use_mask]
    W = (V_pos * sigma_pos[None, :]) @ V_pos.conj().T
    return W, n_modes


def compute_field_intensity(W, parents, children, n):
    I = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                I[i] += abs(W[i, int(j)])**2
        if children[i] is not None and len(children[i]) > 0:
            for j in children[i]:
                I[i] += abs(W[i, j])**2
    return I


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


def run_full_seed(seed_idx, N, geometry, eps=None, base_seed=9800000):
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    bmask = bulk_mask(pts, T, ZETA)

    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    L_flat = hasse_to_link_matrix(par0, N)
    n_links_flat = int(L_flat.sum())

    if geometry == 'ppwave':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        E2 = eps**2 / 2.0
    else:
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, SCH_R_ABCD))
        E2 = SCH_E2

    L_curved = hasse_to_link_matrix(parC, N)
    n_links_curved = int(L_curved.sum())
    delta_links = n_links_curved - n_links_flat

    Y0 = Y_from_graph(par0, ch0)
    YC = Y_from_graph(parC, chC)
    deltaY = YC - Y0
    strata = make_strata(pts, par0, T)
    cj = compute_CJ(Y0, deltaY, bmask, strata)
    sigma0 = float(np.std(Y0[bmask]))

    # SJ vacuum for dI (only if N <= 3000 for speed)
    dI_mean = 0.0
    dI_std = 0.0
    frac_pos = 0.0
    if N <= 3000:
        W_flat, _ = compute_sj_wightman(L_flat)
        W_curved, _ = compute_sj_wightman(L_curved)
        I_flat = compute_field_intensity(W_flat, par0, ch0, N)
        I_curved = compute_field_intensity(W_curved, par0, ch0, N)
        dI = I_curved - I_flat
        dI_bulk = dI[bmask]
        dI_mean = float(np.mean(dI_bulk))
        dI_std = float(np.std(dI_bulk))
        frac_pos = float(np.mean(dI_bulk > 0))

    return {
        'seed': seed_idx, 'N': N, 'geometry': geometry, 'eps': eps, 'E2': E2,
        'CJ': cj,
        'n_links_flat': n_links_flat,
        'n_links_curved': n_links_curved,
        'delta_links': delta_links,
        'sigma0': sigma0,
        'dI_mean': dI_mean,
        'dI_std': dI_std,
        'frac_pos': frac_pos,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("DEEP DIVE TESTS")
    print("=" * 72, flush=True)

    all_results = {}

    # ============================================================
    # TEST 1: corr(CJ, delta_links) across N
    # ============================================================
    print("\n" + "=" * 72)
    print("TEST 1: corr(CJ, delta_links) across N and geometries")
    print("=" * 72, flush=True)

    for N in [2000, 5000]:
        for geometry, eps, label in [
            ('ppwave', 1.0, f'ppw1_N{N}'),
            ('ppwave', 3.0, f'ppw3_N{N}'),
            ('schwarzschild', None, f'sch_N{N}'),
        ]:
            print(f"\n  {label}...", end='', flush=True)
            cj_list, dl_list, nlf_list = [], [], []
            for si in range(M_SEEDS):
                res = run_full_seed(si, N, geometry, eps=eps)
                cj_list.append(res['CJ'])
                dl_list.append(res['delta_links'])
                nlf_list.append(res['n_links_flat'])
            cj_a = np.array(cj_list)
            dl_a = np.array(dl_list)
            nlf_a = np.array(nlf_list)

            r_cj_dl = float(np.corrcoef(cj_a, dl_a)[0, 1])
            r_cj_nlf = float(np.corrcoef(cj_a, nlf_a)[0, 1])
            cj_t = float(cj_a.mean() / (cj_a.std(ddof=1) / math.sqrt(M_SEEDS) + 1e-30))

            all_results[label] = {
                'N': N, 'geometry': geometry, 'eps': eps,
                'CJ_mean': float(cj_a.mean()),
                'CJ_t': cj_t,
                'delta_links_mean': float(dl_a.mean()),
                'delta_links_std': float(dl_a.std()),
                'corr_CJ_deltalinks': r_cj_dl,
                'corr_CJ_nlinksflat': r_cj_nlf,
            }
            print(f" CJ t={cj_t:+.1f}, corr(CJ,dl)={r_cj_dl:+.3f}, <dl>={dl_a.mean():+.0f}±{dl_a.std():.0f}", flush=True)

    # Summary table
    print(f"\n  {'Config':<15} {'CJ t':<8} {'r(CJ,dl)':<10} {'<dl>':<12}")
    print(f"  {'-'*45}")
    for label in sorted(all_results.keys()):
        r = all_results[label]
        if 'corr_CJ_deltalinks' in r:
            print(f"  {label:<15} {r['CJ_t']:+.1f}   {r['corr_CJ_deltalinks']:+.3f}    {r['delta_links_mean']:+.0f}±{r['delta_links_std']:.0f}")

    # ============================================================
    # TEST 2: N-scaling exponents
    # ============================================================
    print("\n" + "=" * 72)
    print("TEST 2: N-scaling exponents (from Tests 1 data + existing)")
    print("=" * 72, flush=True)

    for geometry_label, keys_pattern in [
        ('PP-wave eps=3', 'ppw3'),
        ('Schwarzschild', 'sch'),
    ]:
        ns, cjs = [], []
        for N in [2000, 5000]:
            k = f'{keys_pattern}_N{N}'
            if k in all_results:
                ns.append(N)
                cjs.append(all_results[k]['CJ_mean'])
        if len(ns) >= 2:
            ns = np.array(ns, dtype=float)
            cjs = np.array(cjs)
            alpha = np.polyfit(np.log(ns), np.log(np.abs(cjs)), 1)[0]
            print(f"  {geometry_label}: CJ ~ N^{alpha:.3f} (from N={ns.astype(int).tolist()})")

    # ============================================================
    # TEST 3: Weyl-type discrimination via frac_positive
    # ============================================================
    print("\n" + "=" * 72)
    print("TEST 3: frac_positive(dI) vs eps (Weyl-type discrimination)")
    print("=" * 72, flush=True)

    N_test3 = 2000
    eps_vals = [0.5, 1.0, 2.0, 3.0, 5.0]

    fp_results = {}
    for eps in eps_vals:
        label = f'ppw_e{eps}'
        print(f"\n  ppw eps={eps}...", end='', flush=True)
        fp_list, dI_list = [], []
        for si in range(M_SEEDS):
            res = run_full_seed(si, N_test3, 'ppwave', eps=eps)
            fp_list.append(res['frac_pos'])
            dI_list.append(res['dI_mean'])
        fp_a = np.array(fp_list)
        dI_a = np.array(dI_list)
        fp_results[label] = {
            'eps': eps,
            'frac_pos_mean': float(fp_a.mean()),
            'frac_pos_std': float(fp_a.std()),
            'dI_mean': float(dI_a.mean()),
        }
        print(f" frac_pos={fp_a.mean():.3f}±{fp_a.std():.3f}, dI={dI_a.mean():+.3f}", flush=True)

    # Schwarzschild
    print(f"\n  Sch...", end='', flush=True)
    fp_list, dI_list = [], []
    for si in range(M_SEEDS):
        res = run_full_seed(si, N_test3, 'schwarzschild')
        fp_list.append(res['frac_pos'])
        dI_list.append(res['dI_mean'])
    fp_a = np.array(fp_list)
    dI_a = np.array(dI_list)
    fp_results['sch'] = {
        'frac_pos_mean': float(fp_a.mean()),
        'frac_pos_std': float(fp_a.std()),
        'dI_mean': float(dI_a.mean()),
    }
    print(f" frac_pos={fp_a.mean():.3f}±{fp_a.std():.3f}, dI={dI_a.mean():+.3f}", flush=True)

    # Summary
    print(f"\n  {'Config':<15} {'frac_pos':<12} {'dI_mean':<10}")
    print(f"  {'-'*37}")
    for k in sorted(fp_results.keys()):
        r = fp_results[k]
        print(f"  {k:<15} {r['frac_pos_mean']:.3f}±{r.get('frac_pos_std',0):.3f}   {r['dI_mean']:+.4f}")

    all_results['test3_frac_pos'] = fp_results

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 72)
    print("FINDINGS")
    print("=" * 72)

    # Test 1
    print("\n  TEST 1: corr(CJ, delta_links)")
    for gtype in ['ppw3', 'sch']:
        vals = [(all_results[k]['N'], all_results[k]['corr_CJ_deltalinks'])
                for k in all_results if k.startswith(gtype + '_N')]
        vals.sort()
        if vals:
            label = 'PP-wave eps=3' if gtype == 'ppw3' else 'Schwarzschild'
            corrs = [v[1] for v in vals]
            print(f"    {label}: r = {', '.join(f'{v[1]:+.3f} (N={v[0]})' for v in vals)}")
            if all(abs(c) > 0.3 for c in corrs):
                print(f"    -> STABLE positive correlation. CJ reads link-change spatial pattern.")
            else:
                print(f"    -> Correlation varies with N.")

    # Test 3
    print("\n  TEST 3: Weyl-type discrimination")
    ppw_fps = [(fp_results[k]['eps'], fp_results[k]['frac_pos_mean'])
               for k in fp_results if k.startswith('ppw')]
    ppw_fps.sort()
    sch_fp = fp_results['sch']['frac_pos_mean']
    print(f"    PP-wave frac_pos vs eps: {', '.join(f'{e:.1f}:{fp:.3f}' for e,fp in ppw_fps)}")
    print(f"    Schwarzschild frac_pos: {sch_fp:.3f}")
    if ppw_fps and ppw_fps[-1][1] < 0.15 and sch_fp > 0.20:
        print(f"    -> CONFIRMED: Weyl type N (ppw) vs type D (Sch) have different SJ response.")

    # Save
    outfile = 'analysis/fnd1_data/bridge_deep_dive.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\nSaved to {outfile}", flush=True)
