"""
COMP-SCAN v8
Phase 0b: Screening of ~45 observables at N=500, M=20, CRN.
Dual metric: pp-wave quad eps=5, Schwarzschild eps=0.005.

New observable classes (vs v7):
  - Mobius function statistics (branch 4.1)
  - Scale-dependent ordering fraction (branch 6.2/SC-S6)
  - k-link Laplacian spectral (branch 1.3.3)
  - Normalized Laplacian (branch 1.3.2)
  - Interval internal structure
  - SVD vector localization (SI-2)

Anti-bias protocol v1.0: mandatory proxy check against degree distribution moments.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, sys, os, gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    build_link_graph, graph_statistics
)

RUN_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs',
                       'analysis_runs', 'run_20260326_105953')


def gini(x):
    """Gini coefficient of array x (all non-negative)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 2 or np.sum(x) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def compute_observables(C, A_link, degrees, pts):
    """Compute all candidate observables from causal matrix C and link graph."""
    N = C.shape[0]
    obs = {}

    # === Baselines ===
    tc = float(np.sum(C))
    obs['tc'] = tc
    obs['link_count'] = int(np.sum(A_link)) // 2
    obs['degree_cv'] = float(np.std(degrees) / np.mean(degrees)) if np.mean(degrees) > 0 else 0
    obs['degree_var'] = float(np.var(degrees))
    obs['degree_std'] = float(np.std(degrees))
    obs['degree_skew'] = float(stats.skew(degrees))
    obs['degree_kurt'] = float(stats.kurtosis(degrees))
    obs['max_degree'] = int(np.max(degrees))
    obs['mean_degree'] = float(np.mean(degrees))
    edges = np.array(sp.triu(A_link).nonzero()).T
    if len(edges) > 5:
        d_src = degrees[edges[:, 0]]
        d_dst = degrees[edges[:, 1]]
        rho_a = np.corrcoef(d_src, d_dst)[0, 1]
        obs['assortativity'] = float(rho_a) if not np.isnan(rho_a) else 0.0
    else:
        obs['assortativity'] = 0.0

    past_sizes = np.array(C.sum(axis=0)).ravel()
    future_sizes = np.array(C.sum(axis=1)).ravel()
    obs['sum_deg_sq'] = float(np.sum(degrees ** 2))

    C_sp = sp.csr_matrix(C)
    C2 = (C_sp @ C_sp).toarray()
    n2 = float(np.sum(C2 > 0))
    obs['n2'] = n2

    # === Positive controls ===
    col_norms_C2 = np.sqrt(np.sum(C2 ** 2, axis=0))
    obs['column_gini_C2'] = gini(col_norms_C2)
    obs['column_gini_C'] = gini(np.sqrt(past_sizes))
    obs['link_degree_skew'] = float(stats.skew(degrees))

    L_sp = sp.csr_matrix(A_link)
    lva_vals = []
    for x in range(N):
        nbrs = L_sp.getrow(x).indices
        if len(nbrs) >= 2:
            fan = future_sizes[nbrs]
            m = np.mean(fan)
            if m > 0:
                lva_vals.append(float(np.var(fan) / m**2))
    obs['lva'] = float(np.mean(lva_vals)) if lva_vals else 0.0

    fan_kurt_vals = []
    for x in range(N):
        nbrs = L_sp.getrow(x).indices
        if len(nbrs) >= 4:
            fan = future_sizes[nbrs].astype(np.float64)
            fan_kurt_vals.append(float(stats.kurtosis(fan)))
    obs['fan_kurtosis'] = float(np.mean(fan_kurt_vals)) if fan_kurt_vals else 0.0

    C_dense = C if isinstance(C, np.ndarray) else C.toarray()

    # ═══════════════════════════════════════════════════════════════
    # CLASS 1: MOBIUS FUNCTION (branch 4.1)
    # ═══════════════════════════════════════════════════════════════
    try:
        Z = np.eye(N) + C_dense
        M_mob = np.linalg.solve(Z, np.eye(N))
        causal_mask = C_dense > 0
        mu_vals = M_mob[causal_mask]
        if len(mu_vals) > 10:
            obs['mobius_mean'] = float(np.mean(mu_vals))
            obs['mobius_var'] = float(np.var(mu_vals))
            obs['mobius_skew'] = float(stats.skew(mu_vals))
            obs['mobius_kurt'] = float(stats.kurtosis(mu_vals))
            obs['mobius_frac_pos'] = float(np.mean(mu_vals > 0))
            obs['mobius_abs_mean'] = float(np.mean(np.abs(mu_vals)))
            obs['mobius_gini'] = gini(np.abs(mu_vals))
        else:
            for k in ['mobius_mean', 'mobius_var', 'mobius_skew', 'mobius_kurt',
                       'mobius_frac_pos', 'mobius_abs_mean', 'mobius_gini']:
                obs[k] = 0.0
        del M_mob
    except Exception:
        for k in ['mobius_mean', 'mobius_var', 'mobius_skew', 'mobius_kurt',
                   'mobius_frac_pos', 'mobius_abs_mean', 'mobius_gini']:
            obs[k] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # CLASS 2: SCALE-DEPENDENT ORDERING FRACTION (SC-S6)
    # ═══════════════════════════════════════════════════════════════
    C3 = C_sp @ sp.csr_matrix(C2)
    C3_dense = C3.toarray()
    interval_sizes = C2[C_dense > 0]
    c3_vals = C3_dense[C_dense > 0]

    mask_n3 = interval_sizes >= 3
    if np.sum(mask_n3) > 20:
        n_arr = interval_sizes[mask_n3].astype(np.float64)
        c3_arr = c3_vals[mask_n3].astype(np.float64)
        pairs_arr = n_arr * (n_arr - 1) / 2.0
        f_arr = c3_arr / pairs_arr

        small = n_arr < 10
        med = (n_arr >= 10) & (n_arr < 30)
        large = n_arr >= 30
        obs['sdmm_f_small'] = float(np.mean(f_arr[small])) if np.sum(small) > 5 else 0.0
        obs['sdmm_f_medium'] = float(np.mean(f_arr[med])) if np.sum(med) > 5 else 0.0
        obs['sdmm_f_large'] = float(np.mean(f_arr[large])) if np.sum(large) > 5 else 0.0

        log_n = np.log(n_arr + 1)
        if np.std(log_n) > 1e-10:
            slope, _, r_val, _, _ = stats.linregress(log_n, f_arr)
            obs['sdmm_slope'] = float(slope)
            obs['sdmm_r'] = float(r_val)
        else:
            obs['sdmm_slope'] = 0.0
            obs['sdmm_r'] = 0.0

        unique_n, counts = np.unique(n_arr.astype(int), return_counts=True)
        within_vars = []
        for nv, cnt in zip(unique_n, counts):
            if cnt >= 5:
                within_vars.append(np.var(f_arr[n_arr.astype(int) == nv]))
        obs['sdmm_within_var'] = float(np.mean(within_vars)) if within_vars else 0.0
    else:
        for k in ['sdmm_f_small', 'sdmm_f_medium', 'sdmm_f_large',
                   'sdmm_slope', 'sdmm_r', 'sdmm_within_var']:
            obs[k] = 0.0
    del C3, C3_dense

    # ═══════════════════════════════════════════════════════════════
    # CLASS 3: k-LINK LAPLACIAN SPECTRUM (branch 1.3.3)
    # ═══════════════════════════════════════════════════════════════
    A_link_f = sp.csr_matrix(A_link, dtype=np.float64)
    A2_link = (A_link_f @ A_link_f)
    A2_link.setdiag(0)
    A2_link.eliminate_zeros()

    if N <= 600:
        d2 = np.array(A2_link.sum(axis=1)).ravel()
        L2_dense = np.diag(d2) - A2_link.toarray()
        try:
            eigs_L2 = np.sort(np.linalg.eigvalsh(L2_dense))
            nz = eigs_L2[eigs_L2 > 1e-10]
            obs['klink2_gap'] = float(nz[0]) if len(nz) > 0 else 0.0
            obs['klink2_ratio23'] = float(nz[1] / nz[0]) if len(nz) > 1 and nz[0] > 1e-10 else 0.0
            obs['klink2_spectral_kurt'] = float(stats.kurtosis(eigs_L2))
            obs['klink2_heat_trace'] = float(np.sum(np.exp(-1.0 * eigs_L2)))
        except Exception:
            for k in ['klink2_gap', 'klink2_ratio23', 'klink2_spectral_kurt', 'klink2_heat_trace']:
                obs[k] = 0.0
        del L2_dense
    else:
        for k in ['klink2_gap', 'klink2_ratio23', 'klink2_spectral_kurt', 'klink2_heat_trace']:
            obs[k] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # CLASS 4: NORMALIZED LAPLACIAN (branch 1.3.2)
    # ═══════════════════════════════════════════════════════════════
    if N <= 600:
        A_dense = A_link.toarray() if sp.issparse(A_link) else A_link
        L_comb = np.diag(degrees.astype(np.float64)) - A_dense
        d_inv_sqrt = np.zeros(N)
        nz_mask = degrees > 0
        d_inv_sqrt[nz_mask] = 1.0 / np.sqrt(degrees[nz_mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_norm = D_inv_sqrt @ L_comb @ D_inv_sqrt
        try:
            eigs_Ln = np.sort(np.linalg.eigvalsh(L_norm))
            nz_Ln = eigs_Ln[eigs_Ln > 1e-10]
            obs['nlap_gap'] = float(nz_Ln[0]) if len(nz_Ln) > 0 else 0.0
            obs['nlap_ratio23'] = float(nz_Ln[1] / nz_Ln[0]) if len(nz_Ln) > 1 and nz_Ln[0] > 1e-10 else 0.0
            obs['nlap_kurt'] = float(stats.kurtosis(eigs_Ln))
            p_eig = nz_Ln / np.sum(nz_Ln) if np.sum(nz_Ln) > 0 else nz_Ln
            obs['nlap_entropy'] = float(-np.sum(p_eig * np.log(p_eig + 1e-30)))
        except Exception:
            for k in ['nlap_gap', 'nlap_ratio23', 'nlap_kurt', 'nlap_entropy']:
                obs[k] = 0.0
        del L_comb, L_norm
    else:
        for k in ['nlap_gap', 'nlap_ratio23', 'nlap_kurt', 'nlap_entropy']:
            obs[k] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # CLASS 5: INTERVAL INTERNAL STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    C_bool = (C_sp > 0).astype(np.float64)
    C2_bool_sp = sp.csr_matrix(C2 > 0, dtype=np.float64)
    L_dir = C_bool - C_bool.multiply(C2_bool_sp)
    L_dir.eliminate_zeros()
    L_csc = L_dir.tocsc()
    L_csr = L_dir.tocsr()

    heights = np.zeros(N, dtype=int)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            heights[j] = np.max(heights[parents]) + 1

    ii, jj = np.where((C_dense > 0) & (C2 >= 5))
    if len(ii) > 500:
        idx_s = np.random.default_rng(0).choice(len(ii), 500, replace=False)
        ii, jj = ii[idx_s], jj[idx_s]

    chain_fracs = []
    for a, b in zip(ii, jj):
        cl = heights[b] - heights[a]
        if cl > 0:
            chain_fracs.append(cl / (C2[a, b] + 1))
    obs['interval_chain_fraction'] = float(np.mean(chain_fracs)) if chain_fracs else 0.0
    obs['interval_chain_frac_var'] = float(np.var(chain_fracs)) if chain_fracs else 0.0

    # ═══════════════════════════════════════════════════════════════
    # CLASS 6: SVD VECTOR LOCALIZATION (SI-2)
    # ═══════════════════════════════════════════════════════════════
    try:
        U2, s2, Vt2 = np.linalg.svd(C2.astype(np.float64), full_matrices=False)
        r2 = np.sum(s2 > 1e-10)
        if r2 >= 5:
            U2_trunc = U2[:, :r2]
            ipr_C2 = N * np.sum(U2_trunc**4, axis=0)
            obs['ipr_C2_mean'] = float(np.mean(ipr_C2))
            obs['ipr_C2_gini'] = gini(ipr_C2)
            obs['ipr_C2_max'] = float(np.max(ipr_C2))
            obs['svd_gini_C2'] = gini(s2[:r2])
        else:
            obs['ipr_C2_mean'] = 0.0
            obs['ipr_C2_gini'] = 0.0
            obs['ipr_C2_max'] = 0.0
            obs['svd_gini_C2'] = 0.0
        del U2, Vt2
    except Exception:
        for k in ['ipr_C2_mean', 'ipr_C2_gini', 'ipr_C2_max', 'svd_gini_C2']:
            obs[k] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # ADDITIONAL
    # ═══════════════════════════════════════════════════════════════
    # Chain ratio N3/N2
    C3_sp = C_sp @ sp.csr_matrix(C2)
    n3 = float(np.sum(C3_sp.toarray() > 0))
    obs['chain_ratio_32'] = n3 / n2 if n2 > 0 else 0.0

    # Path count diversity
    p_down = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            p_down[j] = np.sum(p_down[parents])
        if p_down[j] == 0:
            p_down[j] = 1.0
    p_up = np.ones(N, dtype=np.float64)
    for i in range(N - 1, -1, -1):
        children = L_csr.getrow(i).indices
        if len(children) > 0:
            p_up[i] = np.sum(p_up[children])
        if p_up[i] == 0:
            p_up[i] = 1.0
    log_p = np.log2(p_down * p_up + 1)
    obs['path_count_gini'] = gini(log_p) if np.mean(log_p) > 0 else 0.0

    # Height-stratified cone Gini
    h_max = int(np.max(heights))
    lvl_g = []
    for k in range(1, h_max + 1):
        lm = heights == k
        if np.sum(lm) >= 5:
            lp = past_sizes[lm]
            if np.sum(lp) > 0:
                lvl_g.append(gini(lp))
    obs['height_cone_gini'] = float(np.mean(lvl_g)) if lvl_g else 0.0

    # C2 entry shape
    c2e = C2[C2 > 0].astype(np.float64)
    if len(c2e) > 10:
        c2n = c2e / np.mean(c2e)
        obs['c2_entry_kurtosis'] = float(stats.kurtosis(c2n))
        obs['c2_entry_skew'] = float(stats.skew(c2n))
    else:
        obs['c2_entry_kurtosis'] = 0.0
        obs['c2_entry_skew'] = 0.0

    # In/out spearman
    d_out = np.array(L_dir.sum(axis=1)).ravel()
    d_in = np.array(L_dir.sum(axis=0)).ravel()
    mask_io = (d_in > 0) & (d_out > 0)
    if np.sum(mask_io) > 10:
        rho_io, _ = stats.spearmanr(d_in[mask_io], d_out[mask_io])
        obs['in_out_spearman'] = float(rho_io) if not np.isnan(rho_io) else 0.0
    else:
        obs['in_out_spearman'] = 0.0

    del C2, C2_bool_sp, A2_link
    gc.collect()
    return obs


def crn_trial(seed, N, T, metric_fn, eps):
    """One CRN trial: flat vs curved on SAME points."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    _, deg_flat = graph_statistics(A_flat)
    obs_flat = compute_observables(C_flat, A_flat, deg_flat, pts)
    del C_flat, A_flat; gc.collect()
    C_curv = metric_fn(pts, eps)
    A_curv = build_link_graph(C_curv)
    _, deg_curv = graph_statistics(A_curv)
    obs_curv = compute_observables(C_curv, A_curv, deg_curv, pts)
    del C_curv, A_curv; gc.collect()
    deltas = {k: obs_curv[k] - obs_flat[k] for k in obs_flat}
    deltas['seed'] = seed
    return deltas, obs_flat, obs_curv


PROXY_NAMES = ['mean_degree', 'degree_var', 'degree_std', 'degree_skew',
               'degree_kurt', 'link_count', 'max_degree', 'assortativity']
POSITIVE_CONTROLS = ['column_gini_C2', 'column_gini_C', 'lva', 'fan_kurtosis',
                     'link_degree_skew']
SKIP = set(PROXY_NAMES + POSITIVE_CONTROLS + ['tc', 'n2', 'sum_deg_sq', 'seed',
           'degree_cv'])


def analyze_metric(results, label, n_total):
    print(f'\n=== {label} ===')
    alpha = 0.01 / (2 * n_total)
    print(f'Bonferroni alpha = {alpha:.6f}')
    obs_keys = [k for k in results[0] if k not in ('seed',)]
    all_d = {k: np.array([r[k] for r in results]) for k in obs_keys}
    proxy_d = {k: all_d[k] for k in PROXY_NAMES if k in all_d}
    summary = {}
    for k in sorted(obs_keys):
        if k in SKIP:
            continue
        d_arr = all_d[k]
        if np.all(d_arr == 0) or np.std(d_arr) < 1e-15:
            continue
        mu = np.mean(d_arr); sd = np.std(d_arr, ddof=1)
        cohen = mu / sd if sd > 0 else 0
        try:
            _, p_w = stats.wilcoxon(d_arr, alternative='two-sided')
        except Exception:
            p_w = 1.0
        _, p_t = stats.ttest_1samp(d_arr, 0)
        p_val = min(p_w, p_t)
        max_r2, best_px = 0.0, ''
        X_list = []
        for pk in PROXY_NAMES:
            if pk in proxy_d and np.std(proxy_d[pk]) > 1e-15:
                r2 = np.corrcoef(d_arr, proxy_d[pk])[0, 1]**2
                if r2 > max_r2:
                    max_r2, best_px = r2, pk
                X_list.append(proxy_d[pk])
        adj_r2 = 0.0
        if X_list:
            X = np.column_stack([np.ones(len(d_arr))] + X_list)
            try:
                b, _, _, _ = np.linalg.lstsq(X, d_arr, rcond=None)
                ss_r = np.sum((d_arr - X @ b)**2)
                ss_t = np.sum((d_arr - mu)**2)
                mr2 = 1 - ss_r / ss_t if ss_t > 0 else 0
                n_o, p_r = len(d_arr), len(X_list)
                adj_r2 = 1 - (1 - mr2) * (n_o - 1) / (n_o - p_r - 1) if n_o > p_r + 1 else mr2
            except Exception:
                adj_r2 = 0.0
        if p_val < alpha and adj_r2 < 0.50:
            v = 'GENUINE'
        elif adj_r2 > 0.80:
            v = 'KILLED_PROXY'
        elif p_val >= alpha:
            v = 'NOT_SIG'
        else:
            v = 'AMBIGUOUS'
        summary[k] = {'d': round(cohen, 3), 'p': float(f'{p_val:.2e}'),
                       'r2max': round(max_r2, 3), 'proxy': best_px,
                       'adj_r2': round(adj_r2, 3), 'verdict': v}
        f = '***' if v == 'GENUINE' else ('   ' if v == 'NOT_SIG' else ' ! ')
        print(f'  {f} {k:30s} d={cohen:+7.3f}  p={p_val:.1e}  '
              f'R2={max_r2:.3f}({best_px[:8]:8s})  adjR2={adj_r2:.3f}  {v}')
    print(f'\n  Controls:')
    for k in POSITIVE_CONTROLS:
        if k in all_d:
            d_arr = all_d[k]
            if np.std(d_arr) > 0:
                print(f'    {k:25s} d={np.mean(d_arr)/np.std(d_arr):+7.3f}')
    return summary


def main():
    N, T, M = 500, 1.0, 20
    metrics = [('ppwave_quad', causal_ppwave_quad, 5.0),
               ('schwarzschild', causal_schwarzschild, 0.005)]
    all_res = {}
    t0 = time.time()
    for mn, mf, eps in metrics:
        print(f'\n{"="*60}\nCRN: {mn}, N={N}, M={M}, eps={eps}\n{"="*60}')
        res = []
        for trial in range(M):
            seed = 10000 + trial * 100 + {'ppwave_quad': 1, 'schwarzschild': 3}[mn]
            d, _, _ = crn_trial(seed, N, T, mf, eps)
            res.append(d)
            if trial == 0:
                n_obs = len([k for k in d if k not in SKIP and k != 'seed'])
                print(f'  Screening {n_obs} observables...')
        s = analyze_metric(res, mn, n_obs)
        all_res[mn] = {'raw': [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                 for k, v in r.items()} for r in res], 'summary': s}
    el = time.time() - t0
    print(f'\nTotal: {el:.1f}s')
    out = {'version': 'COMP-SCAN v8', 'run': 'run_20260326_105953',
           'N': N, 'M': M, 'elapsed_sec': round(el, 1), 'results': {}}
    for mn in all_res:
        out['results'][mn] = all_res[mn]['summary']
    def nc(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    json.dump(out, open(os.path.join(RUN_DIR, 'comp_scan_v8.json'), 'w'),
              indent=2, default=nc)
    # Also save raw deltas
    raw_out = {mn: all_res[mn]['raw'] for mn in all_res}
    json.dump(raw_out, open(os.path.join(RUN_DIR, 'comp_scan_v8_raw.json'), 'w'),
              indent=2, default=nc)
    print(f'Saved to {RUN_DIR}/comp_scan_v8.json')


if __name__ == '__main__':
    main()
