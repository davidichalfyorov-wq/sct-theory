#!/usr/bin/env python3
"""ALL skeptic tests + why-kurtosis investigation.

Tests:
A. Why kurtosis: 6 statistics of Y on same CRN (kurtosis, variance, Gini, skewness, mean_abs, entropy)
B. Random link perturbation (negative control)
C. A_E(N) convergence: N=5000 vs N=10000
D. r₀ variation: r₀=0.30, 0.50, 1.0 for Schwarzschild
E. path_count_gini T-scaling (observable class test)

All sequential, no multiprocessing.
"""
import sys, os, time, json, math
import numpy as np
from scipy import stats as sp_stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_schwarzschild_local
)


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    if n < 2 or np.sum(x) < 1e-15:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def entropy_Y(Y):
    """Shannon entropy of discretized Y distribution (20 bins)."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.size < 10:
        return 0.0
    counts, _ = np.histogram(Y, bins=20)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_all_statistics(Y0, YF, mask):
    """Compute 6 statistics of delta between YF and Y0 on mask."""
    y0 = Y0[mask]
    yf = YF[mask]

    dk_kurtosis = excess_kurtosis(yf) - excess_kurtosis(y0)
    dk_variance = float(np.var(yf) - np.var(y0))
    dk_gini = gini(yf) - gini(y0)
    dk_skewness = float(sp_stats.skew(yf) - sp_stats.skew(y0))
    dk_mean_abs = float(np.mean(np.abs(yf - np.mean(yf))) - np.mean(np.abs(y0 - np.mean(y0))))
    dk_entropy = entropy_Y(yf) - entropy_Y(y0)

    return {
        "kurtosis": dk_kurtosis,
        "variance": dk_variance,
        "gini": dk_gini,
        "skewness": dk_skewness,
        "mean_abs_dev": dk_mean_abs,
        "entropy": dk_entropy,
    }


if __name__ == "__main__":
    N = 10000
    ZETA = 0.15
    t_total = time.time()
    all_out = {}

    # ═══════════════════════════════════════════════════════
    # TEST A: Why kurtosis? 6 statistics on pp-wave CRN
    # ═══════════════════════════════════════════════════════
    print("=== TEST A: WHY KURTOSIS? ===", flush=True)
    print("pp-wave exact, T=1.0, eps=3.0, N=10000, M=20", flush=True)

    EPS = 3.0
    T_A = 1.0
    M_A = 20
    stat_names = ["kurtosis", "variance", "gini", "skewness", "mean_abs_dev", "entropy"]

    per_seed_A = {s: [] for s in stat_names}
    for si in range(M_A):
        seed = 900000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_A, rng)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YF = Y_from_graph(parF, chF)
        mask = bulk_mask(pts, T_A, ZETA)
        stats = compute_all_statistics(Y0, YF, mask)
        for s in stat_names:
            per_seed_A[s].append(stats[s])
        if (si + 1) % 5 == 0:
            print(f"  A: {si+1}/{M_A}", flush=True)

    print("\n  Results (T=1.0, eps=3, ppw exact):", flush=True)
    print(f"  {'Statistic':15s} {'mean':>12s} {'SE':>10s} {'d_cohen':>10s} {'|d|':>8s}", flush=True)
    test_A = {}
    for s in stat_names:
        arr = np.array(per_seed_A[s])
        m = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        d = m / np.std(arr, ddof=1) if np.std(arr, ddof=1) > 0 else 0
        print(f"  {s:15s} {m:+12.6f} {se:10.6f} {d:+10.3f} {abs(d):8.3f}", flush=True)
        test_A[s] = {"mean": float(m), "se": float(se), "d": float(d), "per_seed": [float(x) for x in arr]}
    all_out["test_A_why_kurtosis"] = test_A

    # Now T-scaling for ALL statistics (T=0.70)
    print("\n  T-scaling comparison (T=0.70, eps=3, M=15):", flush=True)
    T_A2 = 0.70
    M_A2 = 15
    per_seed_A2 = {s: [] for s in stat_names}
    for si in range(M_A2):
        seed = 910000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_A2, rng)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YF = Y_from_graph(parF, chF)
        mask = bulk_mask(pts, T_A2, ZETA)
        stats = compute_all_statistics(Y0, YF, mask)
        for s in stat_names:
            per_seed_A2[s].append(stats[s])
        if (si + 1) % 5 == 0:
            print(f"  A2: {si+1}/{M_A2}", flush=True)

    print(f"\n  T-scaling ratio dk(0.70)/dk(1.0) and ratio/T⁴:", flush=True)
    T4_ratio = 0.70**4
    for s in stat_names:
        m1 = np.mean(per_seed_A[s][:M_A2])  # use same number of seeds
        m07 = np.mean(per_seed_A2[s])
        ratio = m07 / m1 if abs(m1) > 1e-15 else float('nan')
        rt4 = ratio / T4_ratio if np.isfinite(ratio) else float('nan')
        print(f"  {s:15s}: dk(1.0)={m1:+.6f}, dk(0.70)={m07:+.6f}, ratio={ratio:.3f}, ratio/T⁴={rt4:.3f}", flush=True)

    # ═══════════════════════════════════════════════════════
    # TEST B: Random link perturbation (negative control)
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST B: RANDOM LINK PERTURBATION ===", flush=True)
    T_B = 1.0
    M_B = 15
    dk_curv_B = []
    dk_rand_B = []

    for si in range(M_B):
        seed = 920000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_B, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YF = Y_from_graph(parF, chF)

        mask = bulk_mask(pts, T_B, ZETA)
        dk_c = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
        dk_curv_B.append(dk_c)

        # Count how many causal pairs differ between flat and curved
        # Build causal matrices as sets of (j,i) pairs from Hasse parents
        def pairs_from_parents(parents_list):
            s = set()
            for i, pp in enumerate(parents_list):
                for j in pp:
                    s.add((int(j), i))
            return s

        links_flat = pairs_from_parents(par0)
        links_curv = pairs_from_parents(parF)
        added = links_curv - links_flat
        removed = links_flat - links_curv
        n_diff = len(added) + len(removed)

        # PROPER random perturbation: take flat Hasse links,
        # randomly remove len(removed) links and add len(added) fake links
        # Then rebuild path counts on modified graph
        rng_rand = np.random.default_rng(seed + 50000)
        n_flat_links = len(links_flat)

        # Remove random subset of flat links
        flat_list = list(links_flat)
        if len(removed) > 0 and len(flat_list) > len(removed):
            remove_idx = rng_rand.choice(len(flat_list), min(len(removed), len(flat_list)), replace=False)
            modified_links = set(flat_list)
            for idx in remove_idx:
                modified_links.discard(flat_list[idx])
        else:
            modified_links = set(flat_list)

        # Add random forward links (j < i, time-ordered)
        n_to_add = len(added)
        added_count = 0
        attempts = 0
        while added_count < n_to_add and attempts < n_to_add * 10:
            j = rng_rand.integers(0, N - 1)
            i = rng_rand.integers(j + 1, N)
            if (j, i) not in modified_links:
                modified_links.add((j, i))
                added_count += 1
            attempts += 1

        # Rebuild parents/children from modified links
        parents_rand = [[] for _ in range(N)]
        children_rand = [[] for _ in range(N)]
        for (j, i) in modified_links:
            parents_rand[i].append(j)
            children_rand[j].append(i)
        parents_rand = [np.array(sorted(p), dtype=np.int32) for p in parents_rand]
        children_rand = [np.array(sorted(c), dtype=np.int32) for c in children_rand]

        Y_rand = Y_from_graph(parents_rand, children_rand)
        dk_r = excess_kurtosis(Y_rand[mask]) - excess_kurtosis(Y0[mask])
        dk_rand_B.append(dk_r)

        if (si + 1) % 5 == 0:
            print(f"  B: {si+1}/{M_B} (n_diff={n_diff})", flush=True)

    arr_c = np.array(dk_curv_B)
    arr_r = np.array(dk_rand_B)
    print(f"\n  Curvature: dk={np.mean(arr_c):+.6f}±{np.std(arr_c,ddof=1)/np.sqrt(len(arr_c)):.6f}, d={np.mean(arr_c)/np.std(arr_c,ddof=1):+.3f}", flush=True)
    print(f"  Random:    dk={np.mean(arr_r):+.6f}±{np.std(arr_r,ddof=1)/np.sqrt(len(arr_r)):.6f}, d={np.mean(arr_r)/np.std(arr_r,ddof=1):+.3f}", flush=True)
    print(f"  |random/curvature| = {abs(np.mean(arr_r))/max(abs(np.mean(arr_c)),1e-15):.4f}", flush=True)
    all_out["test_B_random_link"] = {
        "curvature": {"mean": float(np.mean(arr_c)), "d": float(np.mean(arr_c)/np.std(arr_c,ddof=1))},
        "random": {"mean": float(np.mean(arr_r)), "d": float(np.mean(arr_r)/np.std(arr_r,ddof=1))},
        "ratio": float(abs(np.mean(arr_r))/max(abs(np.mean(arr_c)),1e-15)),
    }

    # ═══════════════════════════════════════════════════════
    # TEST C: A_E(N) convergence (N=5000 vs N=10000)
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST C: A_E(N) CONVERGENCE ===", flush=True)
    T_C = 1.0
    M_C = 15
    E2_ppw = 0.5 * EPS**2

    for N_test in [5000, 10000]:
        dks = []
        t0 = time.time()
        for si in range(M_C):
            seed = 930000 + N_test + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N_test, T_C, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
            YF = Y_from_graph(parF, chF)
            mask = bulk_mask(pts, T_C, ZETA)
            dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dks.append(dk)
            if (si + 1) % 5 == 0:
                print(f"  C N={N_test}: {si+1}/{M_C} ({time.time()-t0:.0f}s)", flush=True)
        arr = np.array(dks)
        AE = np.mean(arr) / (T_C**4 * E2_ppw)
        print(f"  N={N_test}: dk={np.mean(arr):+.6f}, A_E={AE:.6f}", flush=True)
    all_out["test_C_note"] = "See stdout for A_E(N) values"

    # ═══════════════════════════════════════════════════════
    # TEST D: r₀ variation for Schwarzschild
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST D: r₀ VARIATION ===", flush=True)
    T_D = 1.0
    M_D = 20
    # Use q_W that keeps M/r₀ < 0.15 (weak field) for ALL r₀
    # At r₀=0.30: M = q_W·r₀³ = q_W·0.027. M/r₀ = q_W·0.09. For M/r₀<0.15: q_W<1.67. OK.
    # At r₀=1.00: M = q_W·1.0. M/r₀ = q_W. For M/r₀<0.15: q_W<0.15.
    # Use q_W=0.10 → M/r₀ = 0.003 (r₀=0.30), 0.02 (r₀=0.50), 0.10 (r₀=1.0). All weak field.
    q_W_target = 0.10

    for r0 in [0.30, 0.50, 1.00]:
        M_sch = q_W_target * r0**3 / T_D**2
        R_sch = riemann_schwarzschild_local(M_sch, r0)
        E2_sch = 6.0 * (M_sch / r0**3)**2

        dks = []
        t0 = time.time()
        for si in range(M_D):
            seed = 940000 + int(r0 * 100) + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T_D, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            parF, chF = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_sch))
            YF = Y_from_graph(parF, chF)
            mask = bulk_mask(pts, T_D, ZETA)
            dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dks.append(dk)
            if (si + 1) % 5 == 0:
                print(f"  D r₀={r0}: {si+1}/{M_D} ({time.time()-t0:.0f}s)", flush=True)

        arr = np.array(dks)
        m = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        AE = m / (T_D**4 * E2_sch) if E2_sch > 1e-15 else 0
        print(f"  r₀={r0:.2f}: M={M_sch:.4f}, M/r₀={M_sch/r0:.3f}, E²={E2_sch:.4f}, dk={m:+.6f}±{se:.6f}, A_E={AE:.6f}", flush=True)

    # ═══════════════════════════════════════════════════════
    # TEST E: T-scaling for path_count_gini (observable class)
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST E: PATH_COUNT_GINI T-SCALING ===", flush=True)
    M_E = 15

    for T_E in [1.0, 0.70]:
        dks_gini = []
        dks_kurt = []
        for si in range(M_E):
            seed = 950000 + int(T_E * 1000) * 100 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T_E, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
            YF = Y_from_graph(parF, chF)
            mask = bulk_mask(pts, T_E, ZETA)

            dk_k = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dk_g = gini(YF[mask]) - gini(Y0[mask])
            dks_kurt.append(dk_k)
            dks_gini.append(dk_g)
            if (si + 1) % 5 == 0:
                print(f"  E T={T_E}: {si+1}/{M_E}", flush=True)

        ak = np.array(dks_kurt)
        ag = np.array(dks_gini)
        print(f"  T={T_E:.2f}: kurtosis dk={np.mean(ak):+.6f}, gini dk={np.mean(ag):+.6f}", flush=True)

    # ═══════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════
    total = time.time() - t_total
    print(f"\n=== TOTAL: {total:.0f}s = {total/60:.1f}min ===", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_skeptic_tests.json"), "w") as f:
        json.dump(all_out, f, indent=2)
    print("Saved.", flush=True)
