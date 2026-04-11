"""
COMP-SCAN v7
Phase 0b: Adaptive screening of ~35 observables at N=500, M=20, CRN.
Dual metric: pp-wave quad eps=5, Schwarzschild eps=0.005.

Anti-bias protocol v1.0: mandatory proxy check against degree distribution moments.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.stats import rankdata
import json, time, sys, os, gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    build_link_graph, graph_statistics
)

RUN_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs',
                       'analysis_runs', 'run_20260326_102015')


# ═══════════════════════════════════════════════════════════════════════
# Observable computation functions
# ═══════════════════════════════════════════════════════════════════════

def gini(x):
    """Gini coefficient of array x (all non-negative)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 2 or np.sum(x) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def compute_observables(C, A_link, degrees, pts):
    """Compute all candidate observables from causal matrix C and link graph.

    Returns dict of observable_name -> value.
    """
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

    # Past/future cone sizes from C
    past_sizes = np.array(C.sum(axis=0)).ravel()   # col sums = |past(j)|
    future_sizes = np.array(C.sum(axis=1)).ravel()  # row sums = |future(i)|
    obs['sum_deg_sq'] = float(np.sum(degrees ** 2))

    # Interval abundances (from C²)
    C_sp = sp.csr_matrix(C)
    C2 = (C_sp @ C_sp).toarray()
    n2 = float(np.sum(C2 > 0))
    obs['n2'] = n2

    # ═══════════════════════════════════════════════════════════════
    # POSITIVE CONTROLS (already certified, for sanity check)
    # ═══════════════════════════════════════════════════════════════

    # column_gini_C2 (CERTIFIED)
    col_norms_C2 = np.sqrt(np.sum(C2 ** 2, axis=0))
    obs['column_gini_C2'] = gini(col_norms_C2)

    # column_gini_C (E2)
    obs['column_gini_C'] = gini(np.sqrt(past_sizes))

    # LVA (E2) — Var/Mean² of future fan sizes per element
    L_sp = sp.csr_matrix(A_link)
    # fan size k_+(x,y_i) = |J+(y_i)| for links y_i of x
    # Using C * C^T proxy: (CC^T)[x,y] = |J+(x) ∩ J+(y)|
    # For links: k_+(x,y) = future_sizes[y] exactly (proven identity)
    link_out_csr = sp.triu(L_sp, k=1)  # directed links in upper triangle
    rows_out, cols_out = link_out_csr.nonzero()

    lva_values = []
    # Group by source element
    from collections import defaultdict
    link_fans = defaultdict(list)
    for i, j in zip(rows_out, cols_out):
        link_fans[i].append(future_sizes[j])
    for i, j in zip(cols_out, rows_out):  # also reversed direction
        link_fans[i].append(future_sizes[j])

    # Actually, LVA uses FUTURE links of x: elements y such that x -> y is a link
    # In Hasse diagram: x -> y means x precedes y directly (link)
    # For undirected link graph, we use both directions
    # Simpler: for each element x with >=2 links, compute CV² of {future_sizes[y] : y linked to x}
    link_fans_clean = defaultdict(list)
    for x in range(N):
        row = L_sp.getrow(x)
        neighbors = row.indices
        if len(neighbors) >= 2:
            fan = future_sizes[neighbors]
            m = np.mean(fan)
            if m > 0:
                link_fans_clean[x] = float(np.var(fan) / m**2)

    if len(link_fans_clean) > 0:
        lva_vals = list(link_fans_clean.values())
        obs['lva'] = float(np.mean(lva_vals))
    else:
        obs['lva'] = 0.0

    # link_degree_skew (E2)
    obs['link_degree_skew'] = float(stats.skew(degrees))

    # ═══════════════════════════════════════════════════════════════
    # NEW CANDIDATES — SEED-CONSTRAINT
    # ═══════════════════════════════════════════════════════════════

    # SC1: col_kurt_C2 — kurtosis of C² column norms
    obs['col_kurt_C2'] = float(stats.kurtosis(col_norms_C2))

    # SC2: fan2_cv — CV² of 2-hop fan sizes
    # 2-hop: elements reachable in exactly 2 link steps
    L2 = (L_sp @ L_sp).toarray()
    np.fill_diagonal(L2, 0)
    fan2_sizes = np.sum(L2 > 0, axis=1)  # number of 2-hop neighbors per element
    mask2 = fan2_sizes > 0
    if np.sum(mask2) > 5:
        m2 = np.mean(fan2_sizes[mask2])
        if m2 > 0:
            obs['fan2_cv'] = float(np.var(fan2_sizes[mask2]) / m2**2)
        else:
            obs['fan2_cv'] = 0.0
    else:
        obs['fan2_cv'] = 0.0

    # SC3: ipr_cv_sv — CV² of IPR of right singular vectors of C
    # SVD of C (dense, compact)
    try:
        C_dense = C if isinstance(C, np.ndarray) else C.toarray()
        U, s, Vt = np.linalg.svd(C_dense, full_matrices=False)
        # Keep only nonzero singular values
        r = np.sum(s > 1e-10)
        if r >= 5:
            V = Vt[:r, :].T  # N x r, right singular vectors
            ipr = np.sum(V**4, axis=0)  # IPR for each vector
            m_ipr = np.mean(ipr)
            if m_ipr > 0:
                obs['ipr_cv_sv'] = float(np.var(ipr) / m_ipr**2)
            else:
                obs['ipr_cv_sv'] = 0.0
            obs['ipr_gini_sv'] = gini(ipr)
        else:
            obs['ipr_cv_sv'] = 0.0
            obs['ipr_gini_sv'] = 0.0
    except Exception:
        obs['ipr_cv_sv'] = 0.0
        obs['ipr_gini_sv'] = 0.0

    # SC4: link_wedge_kurtosis — kurtosis of d_in * d_out product
    # in-degree and out-degree in directed link graph
    L_dir = sp.triu(C_sp, k=1) - sp.triu(C_sp, k=1).multiply(sp.triu(sp.csr_matrix(C2 > 0), k=1))
    # Actually simpler: link = C minus elements with intervening
    # Use the link adjacency from build_link_graph but directed
    # For directed links: L_ij = 1 iff i->j is a link (i<j, C[i,j]=1, no k with i<k<j and C[i,k]*C[k,j]=1)
    C_bool = (C_sp > 0).astype(np.float64)
    C2_bool = (C_sp @ C_sp > 0).astype(np.float64)
    L_dir_mat = C_bool - C_bool.multiply(C2_bool)
    L_dir_mat.eliminate_zeros()

    d_out_link = np.array(L_dir_mat.sum(axis=1)).ravel()  # out-degree in directed link graph
    d_in_link = np.array(L_dir_mat.sum(axis=0)).ravel()   # in-degree
    wedge = d_in_link * d_out_link
    mask_w = (d_in_link > 0) & (d_out_link > 0)
    if np.sum(mask_w) > 10:
        obs['link_wedge_kurtosis'] = float(stats.kurtosis(wedge[mask_w]))
    else:
        obs['link_wedge_kurtosis'] = 0.0

    # SC5: path_diversity_cv — CV² of log(path_count) through each element
    # DP to count paths from sources to each node, and from each node to sinks
    # Using topological order (elements sorted by time coordinate)
    topo = np.arange(N)  # already sorted by time
    # Forward pass: count paths from any source
    p_down = np.ones(N, dtype=np.float64)  # each element is a path of length 0
    L_csc = L_dir_mat.tocsc()
    for j in range(N):
        col = L_csc.getcol(j)
        parents = col.indices
        if len(parents) > 0:
            p_down[j] = np.sum(p_down[parents])
        if p_down[j] == 0:
            p_down[j] = 1.0

    # Backward pass: count paths from each node to sinks
    p_up = np.ones(N, dtype=np.float64)
    L_csr = L_dir_mat.tocsr()
    for i in range(N - 1, -1, -1):
        row = L_csr.getrow(i)
        children = row.indices
        if len(children) > 0:
            p_up[i] = np.sum(p_up[children])
        if p_up[i] == 0:
            p_up[i] = 1.0

    p_through = p_down * p_up
    log_p = np.log2(p_through + 1)
    m_lp = np.mean(log_p)
    if m_lp > 0:
        obs['path_count_gini'] = gini(log_p)
        obs['path_count_cv'] = float(np.var(log_p) / m_lp**2)
    else:
        obs['path_count_gini'] = 0.0
        obs['path_count_cv'] = 0.0

    # SC7: spectral_kurtosis_L — kurtosis of link-graph Laplacian eigenvalues
    # Dense eigendecomposition of Laplacian (N <= 500, feasible)
    if N <= 1000:
        A_dense = A_link.toarray() if sp.issparse(A_link) else A_link
        L_dense = np.diag(degrees.astype(np.float64)) - A_dense
        try:
            eigs = np.linalg.eigvalsh(L_dense)
            obs['spectral_kurtosis_L'] = float(stats.kurtosis(eigs))
            obs['spectral_skew_L'] = float(stats.skew(eigs))
        except Exception:
            obs['spectral_kurtosis_L'] = 0.0
            obs['spectral_skew_L'] = 0.0
    else:
        obs['spectral_kurtosis_L'] = 0.0
        obs['spectral_skew_L'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # NEW CANDIDATES — SEED-INVERSION
    # ═══════════════════════════════════════════════════════════════

    # SI1: pf_rank_asym_kurtosis — kurtosis of rank asymmetry
    r_past = rankdata(past_sizes) / N
    r_future = rankdata(future_sizes) / N
    asym = r_past - r_future
    obs['pf_rank_asym_kurtosis'] = float(stats.kurtosis(asym))
    obs['pf_rank_asym_skew'] = float(stats.skew(asym))

    # SI2: link_trans_skew — skewness of directed link transitivity
    # For each x: fraction of (past-link, future-link) pairs that are themselves linked
    trans_values = []
    A_link_sp = sp.csr_matrix(A_link) if not sp.issparse(A_link) else A_link
    for x in range(N):
        past_links = L_csc.getcol(x).indices  # elements y with y->x link
        future_links = L_csr.getrow(x).indices  # elements z with x->z link
        n_past = len(past_links)
        n_future = len(future_links)
        if n_past >= 2 and n_future >= 2:
            count = 0
            for y in past_links:
                for z in future_links:
                    if A_link_sp[y, z] > 0:
                        count += 1
            denom = n_past * n_future
            trans_values.append(count / denom if denom > 0 else 0.0)

    if len(trans_values) > 10:
        obs['link_trans_skew'] = float(stats.skew(trans_values))
        obs['link_trans_mean'] = float(np.mean(trans_values))
    else:
        obs['link_trans_skew'] = 0.0
        obs['link_trans_mean'] = 0.0

    # SI4: path_count_gini already computed above

    # SI5: c2_entry_kurtosis — kurtosis of normalized C² entries
    c2_entries = C2[C2 > 0].astype(np.float64)
    if len(c2_entries) > 10:
        c2_norm = c2_entries / np.mean(c2_entries)
        obs['c2_entry_kurtosis'] = float(stats.kurtosis(c2_norm))
        obs['c2_entry_skew'] = float(stats.skew(c2_norm))
    else:
        obs['c2_entry_kurtosis'] = 0.0
        obs['c2_entry_skew'] = 0.0

    # SI6: height_cone_gini — mean Gini of past cone sizes per height level
    # Height = longest chain from minimal element using link graph
    heights = np.zeros(N, dtype=int)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            heights[j] = np.max(heights[parents]) + 1

    h_max = int(np.max(heights))
    level_ginis = []
    for k in range(1, h_max + 1):
        level_mask = heights == k
        if np.sum(level_mask) >= 5:
            level_past = past_sizes[level_mask]
            if np.sum(level_past) > 0:
                level_ginis.append(gini(level_past))
    obs['height_cone_gini'] = float(np.mean(level_ginis)) if level_ginis else 0.0

    # ═══════════════════════════════════════════════════════════════
    # ADDITIONAL EXPLORATORY OBSERVABLES
    # ═══════════════════════════════════════════════════════════════

    # Column norm higher moments of C (extends column_gini_C)
    sqrt_past = np.sqrt(past_sizes.astype(np.float64))
    obs['col_skew_C'] = float(stats.skew(sqrt_past))
    obs['col_kurt_C'] = float(stats.kurtosis(sqrt_past))

    # Fan kurtosis (proven E3 — sanity)
    fan_kurt_vals = []
    for x in range(N):
        row = L_sp.getrow(x)
        neighbors = row.indices
        if len(neighbors) >= 4:
            fan = future_sizes[neighbors].astype(np.float64)
            fan_kurt_vals.append(float(stats.kurtosis(fan)))
    obs['fan_kurtosis'] = float(np.mean(fan_kurt_vals)) if fan_kurt_vals else 0.0

    # Fan skewness (new — complements fan_kurtosis)
    fan_skew_vals = []
    for x in range(N):
        row = L_sp.getrow(x)
        neighbors = row.indices
        if len(neighbors) >= 4:
            fan = future_sizes[neighbors].astype(np.float64)
            fan_skew_vals.append(float(stats.skew(fan)))
    obs['fan_skewness'] = float(np.mean(fan_skew_vals)) if fan_skew_vals else 0.0

    # In-out degree Spearman correlation
    mask_io = (d_in_link > 0) & (d_out_link > 0)
    if np.sum(mask_io) > 10:
        rho_io, _ = stats.spearmanr(d_in_link[mask_io], d_out_link[mask_io])
        obs['in_out_spearman'] = float(rho_io) if not np.isnan(rho_io) else 0.0
    else:
        obs['in_out_spearman'] = 0.0

    # SVD Gini of C (CERTIFIED — sanity)
    if 'ipr_cv_sv' in obs and r >= 5:
        obs['svd_gini_C'] = gini(s[:r])
    else:
        try:
            C_dense = C if isinstance(C, np.ndarray) else C.toarray()
            s_vals = np.linalg.svd(C_dense, compute_uv=False)
            s_pos = s_vals[s_vals > 1e-10]
            obs['svd_gini_C'] = gini(s_pos)
        except Exception:
            obs['svd_gini_C'] = 0.0

    # Chain ratio N_3/N_2
    C3 = C_sp @ sp.csr_matrix(C2)
    n3 = float(np.sum(C3.toarray() > 0))
    obs['n3'] = n3
    obs['chain_ratio_32'] = n3 / n2 if n2 > 0 else 0.0

    del C2, L2, C3
    gc.collect()

    return obs


# ═══════════════════════════════════════════════════════════════════════
# CRN trial
# ═══════════════════════════════════════════════════════════════════════

def crn_trial(seed, N, T, metric_fn, eps):
    """One CRN trial: flat vs curved on SAME points."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)

    # Flat
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    gs_flat, deg_flat = graph_statistics(A_flat)
    obs_flat = compute_observables(C_flat, A_flat, deg_flat, pts)
    del C_flat, A_flat
    gc.collect()

    # Curved
    C_curv = metric_fn(pts, eps)
    A_curv = build_link_graph(C_curv)
    gs_curv, deg_curv = graph_statistics(A_curv)
    obs_curv = compute_observables(C_curv, A_curv, deg_curv, pts)
    del C_curv, A_curv
    gc.collect()

    # Deltas
    deltas = {}
    for key in obs_flat:
        deltas[key] = obs_curv[key] - obs_flat[key]
    deltas['seed'] = seed

    return deltas, obs_flat, obs_curv


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    N = 500
    T = 1.5
    M = 20

    # Pre-registered thresholds (anti-bias v1.0)
    ALPHA_BONF = 0.01 / 2  # 2 metrics
    R2_KILL = 0.80
    R2_GENUINE = 0.50

    metrics = [
        ("ppwave_quad", causal_ppwave_quad, 5.0),
        ("schwarzschild", causal_schwarzschild, 0.005),
    ]

    all_results = {}

    for metric_name, metric_fn, eps in metrics:
        print(f"\n{'='*60}")
        print(f"COMP-SCAN: {metric_name} eps={eps}, N={N}, M={M}")
        print(f"{'='*60}")

        deltas_list = []
        flat_list = []
        curv_list = []

        t0 = time.time()
        for trial in range(M):
            seed = 42000 + trial * 1000
            d, f, c = crn_trial(seed, N, T, metric_fn, eps)
            deltas_list.append(d)
            flat_list.append(f)
            curv_list.append(c)
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{M} done ({time.time()-t0:.1f}s)")

        elapsed = time.time() - t0
        print(f"  Total: {elapsed:.1f}s ({elapsed/M:.1f}s/trial)")

        # Analyze each observable
        obs_keys = [k for k in deltas_list[0] if k != 'seed']
        baseline_keys = ['tc', 'link_count', 'degree_cv', 'degree_var', 'degree_std',
                         'degree_skew', 'degree_kurt', 'max_degree', 'mean_degree',
                         'sum_deg_sq', 'n2']

        results = {}
        for key in obs_keys:
            if key in baseline_keys:
                continue
            vals = [d[key] for d in deltas_list]
            arr = np.array(vals, dtype=np.float64)
            # Replace NaN/Inf with 0
            arr = np.where(np.isfinite(arr), arr, 0.0)
            m = np.mean(arr)
            sd = np.std(arr, ddof=1)
            d_cohen = m / sd if sd > 0 else 0
            _, p = stats.ttest_1samp(arr, 0.0) if M >= 5 else (0, 1)

            # Proxy check: R² against degree statistics
            proxy_stats = {}
            for bkey in ['mean_degree', 'degree_var', 'degree_std', 'degree_skew',
                         'degree_kurt', 'max_degree']:
                b_vals = np.array([d[bkey] for d in deltas_list])
                if np.std(b_vals) > 0 and np.std(arr) > 0:
                    r = np.corrcoef(arr, b_vals)[0, 1]
                    proxy_stats[bkey] = float(r**2)
                else:
                    proxy_stats[bkey] = 0.0

            max_r2 = max(proxy_stats.values()) if proxy_stats else 0.0

            # Multiple regression R² against ALL proxies
            proxy_cols = []
            for bkey in ['mean_degree', 'degree_var', 'degree_std',
                         'degree_skew', 'degree_kurt', 'max_degree',
                         'tc', 'link_count']:
                col = np.array([d[bkey] for d in deltas_list], dtype=np.float64)
                col = np.where(np.isfinite(col), col, 0.0)
                proxy_cols.append(col)
            X_proxy = np.column_stack(proxy_cols)
            try:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(X_proxy, arr)
                r2_multi = reg.score(X_proxy, arr)
                # Adjusted R²
                n_obs = len(arr)
                n_feat = X_proxy.shape[1]
                r2_adj = 1 - (1 - r2_multi) * (n_obs - 1) / (n_obs - n_feat - 1)
            except ImportError:
                # Manual OLS
                X = np.column_stack([np.ones(len(arr)), X_proxy])
                try:
                    beta = np.linalg.lstsq(X, arr, rcond=None)[0]
                    pred = X @ beta
                    ss_res = np.sum((arr - pred)**2)
                    ss_tot = np.sum((arr - np.mean(arr))**2)
                    r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    n_obs = len(arr)
                    n_feat = X_proxy.shape[1]
                    r2_adj = 1 - (1 - r2_multi) * (n_obs - 1) / (n_obs - n_feat - 1)
                except Exception:
                    r2_multi = 0.0
                    r2_adj = 0.0

            # Classify
            if r2_adj > R2_KILL:
                status = "KILLED_PROXY"
            elif abs(d_cohen) < 0.3 or p > ALPHA_BONF:
                status = "WEAK"
            elif max_r2 > R2_GENUINE:
                status = "AMBIGUOUS"
            else:
                status = "PASS"

            results[key] = {
                "d_cohen": round(d_cohen, 3),
                "p_value": float(f"{p:.2e}"),
                "mean_delta": round(m, 6),
                "std_delta": round(sd, 6),
                "max_r2_proxy": round(max_r2, 3),
                "r2_multi_adj": round(r2_adj, 3),
                "status": status,
                "proxy_detail": {k: round(v, 3) for k, v in proxy_stats.items()},
            }

        all_results[metric_name] = results

    # Print summary
    print(f"\n{'='*70}")
    print("COMP-SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"{'Observable':<30} {'pp d':>7} {'pp p':>10} {'pp R²':>6} {'Sch d':>7} {'Sch p':>10} {'Sch R²':>6} {'Status':>12}")
    print("-" * 100)

    obs_union = set()
    for m in all_results:
        obs_union |= set(all_results[m].keys())

    for key in sorted(obs_union):
        pp = all_results.get("ppwave_quad", {}).get(key, {})
        sc = all_results.get("schwarzschild", {}).get(key, {})
        pp_d = pp.get("d_cohen", 0)
        pp_p = pp.get("p_value", 1)
        pp_r2 = pp.get("r2_multi_adj", 0)
        sc_d = sc.get("d_cohen", 0)
        sc_p = sc.get("p_value", 1)
        sc_r2 = sc.get("r2_multi_adj", 0)

        # Combined status
        pp_s = pp.get("status", "?")
        sc_s = sc.get("status", "?")
        if pp_s == "KILLED_PROXY" or sc_s == "KILLED_PROXY":
            status = "KILLED"
        elif pp_s == "PASS" and sc_s == "PASS":
            status = "DUAL-PASS"
        elif pp_s == "PASS" or sc_s == "PASS":
            status = "SINGLE"
        else:
            status = "WEAK"

        flag = " ***" if status == "DUAL-PASS" else " *" if status == "SINGLE" else ""
        print(f"{key:<30} {pp_d:>+7.2f} {pp_p:>10.2e} {pp_r2:>6.3f} {sc_d:>+7.2f} {sc_p:>10.2e} {sc_r2:>6.3f} {status:>12}{flag}")

    # Save results
    output_path = os.path.join(RUN_DIR, 'comp_scan_results.json')
    json.dump(all_results, open(output_path, 'w'), indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
