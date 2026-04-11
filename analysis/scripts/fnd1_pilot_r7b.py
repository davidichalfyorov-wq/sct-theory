"""
Analysis run Session 2 — PILOT Phase 5
N=2000, M=20, CRN. 3 candidates + positive controls.
Dual metric: pp-wave quad eps=5, Schwarzschild eps=0.005.
Conformal null. TC-mediation test. Leakage check.

Candidates: chain_decay_slope, fan2_lva, path_kurtosis
Controls: column_gini_C2, lva, column_gini_C

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
                       'analysis_runs', 'run_20260326_110013')


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 2 or np.sum(x) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def compute_pilot_obs(C, A_link, degrees, pts):
    """Compute the 3 candidates + 3 controls + baselines."""
    N = C.shape[0]
    obs = {}
    C_sp = sp.csr_matrix(C) if not sp.issparse(C) else C
    tc = float(C_sp.sum())
    obs['tc'] = tc
    obs['link_count'] = int(np.sum(A_link)) // 2
    obs['mean_degree'] = float(np.mean(degrees))
    obs['degree_var'] = float(np.var(degrees))
    obs['degree_std'] = float(np.std(degrees))
    obs['degree_skew'] = float(stats.skew(degrees))
    obs['degree_kurt'] = float(stats.kurtosis(degrees))
    obs['max_degree'] = int(np.max(degrees))

    past_sizes = np.array(C_sp.sum(axis=0)).ravel().astype(np.float64)
    future_sizes = np.array(C_sp.sum(axis=1)).ravel().astype(np.float64)
    C2 = (C_sp @ C_sp).toarray()

    # Directed link graph
    C_bool = (C_sp > 0).astype(np.float64)
    C2_bool = (sp.csr_matrix(C2) > 0).astype(np.float64)
    L_dir = C_bool - C_bool.multiply(C2_bool)
    L_dir.eliminate_zeros()
    L_dir_csr = L_dir.tocsr()
    L_dir_csc = L_dir.tocsc()
    L_sp = sp.csr_matrix(A_link)

    # --- Controls ---
    col_norms_C2 = np.sqrt(np.sum(C2 ** 2, axis=0).astype(np.float64))
    obs['column_gini_C2'] = gini(col_norms_C2)
    obs['column_gini_C'] = gini(np.sqrt(past_sizes))

    lva_vals = []
    for x in range(N):
        nbrs = L_sp.getrow(x).indices
        if len(nbrs) >= 2:
            fan = future_sizes[nbrs]
            m = np.mean(fan)
            if m > 0:
                lva_vals.append(float(np.var(fan) / m**2))
    obs['lva'] = float(np.mean(lva_vals)) if lva_vals else 0.0

    # --- Candidate 1: chain_decay_slope ---
    n2_chains = tc
    n3_chains = float(np.sum(C2))
    C3_sp = C_sp @ sp.csr_matrix(C2)
    n4_chains = float(C3_sp.sum())
    chain_counts = [n2_chains, n3_chains, n4_chains]
    ks = [2, 3, 4]
    if all(c > 0 for c in chain_counts):
        log_counts = [np.log(c) for c in chain_counts]
        slope, _, _, _, _ = stats.linregress(ks, log_counts)
        obs['chain_decay_slope'] = float(slope)
    else:
        obs['chain_decay_slope'] = 0.0
    del C3_sp

    # --- Candidate 2: fan2_lva ---
    L2 = (L_sp @ L_sp).toarray()
    np.fill_diagonal(L2, 0)
    A_dense = A_link.toarray() if sp.issparse(A_link) else A_link
    L2[A_dense > 0] = 0
    fan2_vals = []
    for x in range(N):
        nbrs2 = np.where(L2[x] > 0)[0]
        if len(nbrs2) >= 3:
            fan = future_sizes[nbrs2]
            m = np.mean(fan)
            if m > 0:
                fan2_vals.append(float(np.var(fan) / m**2))
    obs['fan2_lva'] = float(np.mean(fan2_vals)) if fan2_vals else 0.0
    del L2, A_dense

    # --- Candidate 3: path_kurtosis ---
    p_down = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_dir_csc.getcol(j).indices
        if len(parents) > 0:
            p_down[j] = np.sum(p_down[parents])
        if p_down[j] == 0:
            p_down[j] = 1.0
    p_up = np.ones(N, dtype=np.float64)
    for i in range(N - 1, -1, -1):
        children = L_dir_csr.getrow(i).indices
        if len(children) > 0:
            p_up[i] = np.sum(p_up[children])
        if p_up[i] == 0:
            p_up[i] = 1.0
    log_p = np.log2(p_down * p_up + 1)
    obs['path_kurtosis'] = float(stats.kurtosis(log_p, fisher=True)) if len(log_p) > 10 else 0.0

    del C2
    gc.collect()
    return obs


def crn_trial(seed, N, T, metric_fn, eps):
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    _, deg_flat = graph_statistics(A_flat)
    obs_flat = compute_pilot_obs(C_flat, A_flat, deg_flat, pts)
    del C_flat, A_flat; gc.collect()
    C_curv = metric_fn(pts, eps)
    A_curv = build_link_graph(C_curv)
    _, deg_curv = graph_statistics(A_curv)
    obs_curv = compute_pilot_obs(C_curv, A_curv, deg_curv, pts)
    del C_curv, A_curv; gc.collect()
    deltas = {k: obs_curv[k] - obs_flat[k] for k in obs_flat}
    deltas['seed'] = seed
    return deltas, obs_flat, obs_curv


def main():
    N, T, M = 2000, 1.5, 20
    PROXY = ['mean_degree', 'degree_var', 'degree_std', 'degree_skew',
             'degree_kurt', 'max_degree', 'tc', 'link_count']
    CANDS = ['chain_decay_slope', 'fan2_lva', 'path_kurtosis']
    CTRLS = ['column_gini_C2', 'lva', 'column_gini_C']
    metrics = [("ppwave_quad", causal_ppwave_quad, 5.0),
               ("schwarzschild", causal_schwarzschild, 0.005)]

    all_res = {}
    for mn, mf, eps in metrics:
        print(f"\n{'='*60}\nPILOT: {mn} eps={eps}, N={N}, M={M}\n{'='*60}")
        dl, fl, cl = [], [], []
        t0 = time.time()
        for trial in range(M):
            seed = 50000 + trial * 1000
            d, f, c = crn_trial(seed, N, T, mf, eps)
            dl.append(d); fl.append(f); cl.append(c)
            if (trial+1) % 5 == 0:
                print(f"  Trial {trial+1}/{M} ({time.time()-t0:.1f}s)")
        print(f"  Total: {time.time()-t0:.1f}s")

        res = {}
        for key in CANDS + CTRLS:
            arr = np.array([d[key] for d in dl], dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            mu, sd = np.mean(arr), np.std(arr, ddof=1)
            d_c = mu / sd if sd > 0 else 0
            _, p = stats.ttest_1samp(arr, 0.0)
            # Proxy
            pr2 = {}
            for pk in PROXY:
                b = np.array([d[pk] for d in dl], dtype=np.float64)
                if np.std(b) > 0 and sd > 0:
                    pr2[pk] = float(np.corrcoef(arr, b)[0,1]**2)
                else:
                    pr2[pk] = 0.0
            max_r2 = max(pr2.values())
            # Multi R²adj
            Xc = [np.array([d[pk] for d in dl], dtype=np.float64) for pk in PROXY]
            X = np.column_stack([np.ones(M)] + Xc)
            try:
                b, _, _, _ = np.linalg.lstsq(X, arr, rcond=None)
                ss_r = np.sum((arr - X @ b)**2); ss_t = np.sum((arr - mu)**2)
                mr2 = 1 - ss_r/ss_t if ss_t > 0 else 0
                r2a = 1 - (1 - mr2)*(M-1)/(M-len(PROXY)-1) if M > len(PROXY)+1 else 0
            except: r2a = 0.0
            # TC-mediation
            tc_f = np.array([f['tc'] for f in fl])
            obs_f = np.array([f[key] for f in fl])
            obs_c = np.array([c[key] for c in cl])
            tc_c = np.array([c['tc'] for c in cl])
            try:
                sl, ic, _, _, _ = stats.linregress(tc_f, obs_f)
                pred = sl * tc_c + ic
                resid = obs_c - pred
                d_resid = float(np.mean(resid)/np.std(resid, ddof=1)) if np.std(resid, ddof=1) > 0 else 0
            except: d_resid = 0.0

            res[key] = {"d": round(d_c,3), "p": float(f"{p:.2e}"), "r2adj": round(r2a,3),
                        "max_r2": round(max_r2,3), "d_tc_resid": round(d_resid,3)}
            tag = "***" if key in CANDS else "(c)"
            print(f"  {tag} {key:25s} d={d_c:+.3f} p={p:.1e} R2a={r2a:.3f} d_tc={d_resid:+.3f}")
        all_res[mn] = {"results": res, "deltas": [{k:float(v) if isinstance(v,(float,np.floating)) else v for k,v in d.items()} for d in dl]}

    # Leakage
    print(f"\n{'='*60}\nLEAKAGE CHECK\n{'='*60}")
    for k1 in CANDS:
        for k2 in CANDS + CTRLS:
            if k1 >= k2: continue
            v1 = np.array([d[k1] for d in dl])
            v2 = np.array([d[k2] for d in dl])
            rho, _ = stats.spearmanr(v1, v2)
            if abs(rho) > 0.5:
                flag = " !!!KILL" if abs(rho)>0.8 else ""
                print(f"  rho({k1}, {k2}) = {rho:+.3f}{flag}")

    out = {"phase": "PILOT_s2", "N": N, "M": M}
    for mn in all_res:
        out[mn] = all_res[mn]["results"]
    json.dump(out, open(os.path.join(RUN_DIR, 'pilot_results.json'), 'w'), indent=2, default=str)
    print(f"\nSaved to {RUN_DIR}/pilot_results.json")

if __name__ == "__main__":
    main()
