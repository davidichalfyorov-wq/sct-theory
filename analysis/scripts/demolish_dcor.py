#!/usr/bin/env python3
"""
DEMOLISH agent: Kill-tests for dcor(|X|, delta_Y^2).

Tests:
  1. ADVERSARY: random link flips vs geometric curvature
  2. N-CONVERGENCE: N=300,500,1000,2000 Cohen's d scaling
  3. PEARSON EQUIVALENCE: dcor vs r^2 information overlap
  4. STRATIFIED dcor: global vs within-stratum
  5. SUBSAMPLE SENSITIVITY: n=50,100,150,ALL

Author: David Alfyorov
"""
import sys, os, time, json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, Y_from_graph, bulk_mask,
)

# ═══════════════════════════════════════════════════════════
# Distance correlation
# ═══════════════════════════════════════════════════════════
def dcor(x, y):
    """Biased distance correlation (Szekely-Rizzo 2007)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)
    if n < 5:
        return 0.0
    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])
    Ax = ax - ax.mean(axis=0, keepdims=True) - ax.mean(axis=1, keepdims=True) + ax.mean()
    Ay = ay - ay.mean(axis=0, keepdims=True) - ay.mean(axis=1, keepdims=True) + ay.mean()
    dcov2 = float(np.mean(Ax * Ay))
    dvar_x = float(np.mean(Ax * Ax))
    dvar_y = float(np.mean(Ay * Ay))
    if dvar_x < 1e-30 or dvar_y < 1e-30:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y)))


def cohens_d(vals):
    """One-sample Cohen's d = mean / std."""
    vals = np.array(vals, dtype=float)
    mu = np.mean(vals)
    sd = np.std(vals, ddof=1)
    return mu / sd if sd > 1e-15 else 0.0


# ═══════════════════════════════════════════════════════════
# Strata
# ═══════════════════════════════════════════════════════════
def make_strata(pts, T, tau_bins=5, rho_bins=3):
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * tau_bins / 2.0).astype(int), 0, tau_bins - 1)
    rho_bin = np.clip(np.floor(rho_hat * rho_bins).astype(int), 0, rho_bins - 1)
    return tau_bin * rho_bins + rho_bin


# ═══════════════════════════════════════════════════════════
# TEST 1: GRAPH-THEORY ADVERSARY
# ═══════════════════════════════════════════════════════════
def adversary_test(N=300, M=15, eps=5.0, T=1.0, zeta=0.15, flip_frac=0.05):
    """Compare dcor from geometric curvature vs random link flips."""
    print("\n" + "=" * 60)
    print(f"TEST 1: ADVERSARY (N={N}, M={M}, flip_frac={flip_frac})")
    print("=" * 60, flush=True)

    dcor_geo = []
    dcor_rdag = []

    for trial in range(M):
        seed = 7777000 + trial * 1000
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        mask = bulk_mask(pts, T, zeta)
        idx = np.where(mask)[0]

        # Flat Hasse
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        # Curved Hasse (geometric)
        parC, chC = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        Y_curv = Y_from_graph(parC, chC)

        delta_Y_geo = Y_curv - Y0
        X = Y0 - np.mean(Y0[mask])
        absX = np.abs(X[idx])
        dY2_geo = delta_Y_geo[idx] ** 2

        dc_geo = dcor(absX, dY2_geo)
        dcor_geo.append(dc_geo)

        # Random perturbation: flip fraction of causal relations
        rng2 = np.random.default_rng(seed + 500)
        n = len(pts)
        # Count existing links
        n_links = sum(len(c) for c in ch0)
        n_flip = max(1, int(n_links * flip_frac))

        # Create perturbed Hasse by flipping random links
        par_r = [list(p) for p in par0]
        ch_r = [list(c) for c in ch0]

        # Remove random existing links
        all_links = []
        for i in range(n):
            for j in ch0[i]:
                all_links.append((i, j))
        if len(all_links) > n_flip:
            remove_idx = rng2.choice(len(all_links), n_flip, replace=False)
            for ri in remove_idx:
                i, j = all_links[ri]
                if j in ch_r[i]:
                    ch_r[i].remove(j)
                if i in par_r[j]:
                    par_r[j].remove(i)

        # Add random new links (respecting time ordering)
        added = 0
        attempts = 0
        while added < n_flip and attempts < n_flip * 20:
            i = rng2.integers(0, n - 1)
            j = rng2.integers(i + 1, n)
            if j not in ch_r[i]:
                ch_r[i].append(j)
                par_r[j].append(i)
                added += 1
            attempts += 1

        # Convert to numpy arrays as expected by Y_from_graph
        par_r_np = [np.array(p, dtype=np.intp) for p in par_r]
        ch_r_np = [np.array(c, dtype=np.intp) for c in ch_r]
        Y_rdag = Y_from_graph(par_r_np, ch_r_np)
        delta_Y_rdag = Y_rdag - Y0
        dY2_rdag = delta_Y_rdag[idx] ** 2

        dc_rdag = dcor(absX, dY2_rdag)
        dcor_rdag.append(dc_rdag)

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{M}: geo={dc_geo:.4f}, rdag={dc_rdag:.4f}", flush=True)

    d_geo = cohens_d(dcor_geo)
    d_rdag = cohens_d(dcor_rdag)
    ratio = abs(d_rdag / d_geo) if abs(d_geo) > 0.01 else float('inf')

    print(f"\nResult: d_geo={d_geo:.3f}, d_rdag={d_rdag:.3f}")
    print(f"  Ratio d_rdag/d_geo = {ratio:.3f}")
    print(f"  VERDICT: {'KILL' if ratio > 0.3 else 'PASS'} (threshold: 0.3)")

    return {
        "test": "adversary",
        "d_geo": round(d_geo, 3),
        "d_rdag": round(d_rdag, 3),
        "ratio": round(ratio, 3),
        "mean_geo": round(float(np.mean(dcor_geo)), 4),
        "mean_rdag": round(float(np.mean(dcor_rdag)), 4),
        "verdict": "KILL" if ratio > 0.3 else "PASS",
    }


# ═══════════════════════════════════════════════════════════
# TEST 2: N-CONVERGENCE
# ═══════════════════════════════════════════════════════════
def n_convergence_test(N_list=[300, 500, 1000], M=15, eps=5.0, T=1.0, zeta=0.15):
    """Check if Cohen's d grows, stays stable, or collapses with N."""
    print("\n" + "=" * 60)
    print(f"TEST 2: N-CONVERGENCE (N={N_list}, M={M})")
    print("=" * 60, flush=True)

    results = []
    for N in N_list:
        dcor_vals = []
        pearson_vals = []
        spearman_vals = []
        t0 = time.time()

        for trial in range(M):
            seed = 8888000 + trial * 1000 + N
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            mask = bulk_mask(pts, T, zeta)
            idx = np.where(mask)[0]

            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            parC, chC = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
            Y_curv = Y_from_graph(parC, chC)

            delta_Y = Y_curv - Y0
            X = Y0 - np.mean(Y0[mask])
            absX = np.abs(X[idx])
            dY2 = delta_Y[idx] ** 2

            # dcor on ALL interior elements (not subsampled)
            dc = dcor(absX, dY2)
            dcor_vals.append(dc)

            # Pearson for comparison
            if np.std(absX) > 1e-15 and np.std(dY2) > 1e-15:
                r_p = float(pearsonr(absX, dY2)[0])
                r_s = float(spearmanr(absX, dY2)[0])
            else:
                r_p, r_s = 0.0, 0.0
            pearson_vals.append(r_p)
            spearman_vals.append(r_s)

        elapsed = time.time() - t0
        d_dcor = cohens_d(dcor_vals)
        d_pear = cohens_d(pearson_vals)
        d_spear = cohens_d(spearman_vals)

        n_interior = int(np.mean([np.sum(bulk_mask(
            sprinkle_local_diamond(N, T, np.random.default_rng(8888000 + i * 1000 + N)),
            T, zeta)) for i in range(3)]))

        result = {
            "N": N,
            "n_interior": n_interior,
            "d_dcor": round(d_dcor, 3),
            "d_pearson": round(d_pear, 3),
            "d_spearman": round(d_spear, 3),
            "mean_dcor": round(float(np.mean(dcor_vals)), 4),
            "std_dcor": round(float(np.std(dcor_vals, ddof=1)), 4),
            "mean_pearson": round(float(np.mean(pearson_vals)), 4),
            "mean_spearman": round(float(np.mean(spearman_vals)), 4),
            "elapsed_s": round(elapsed, 1),
        }
        results.append(result)

        print(f"  N={N:5d}: d_dcor={d_dcor:>7.3f}, d_pear={d_pear:>7.3f}, "
              f"d_spear={d_spear:>7.3f}, n_int={n_interior}, "
              f"mean_dcor={np.mean(dcor_vals):.4f} ({elapsed:.1f}s)", flush=True)

    # Check convergence: is d stable or collapsing?
    d_values = [r["d_dcor"] for r in results]
    if len(d_values) >= 2 and d_values[0] > 0.5:
        collapse_ratio = d_values[-1] / d_values[0]
        verdict = "KILL" if collapse_ratio < 0.3 else ("CONDITIONAL" if collapse_ratio < 0.7 else "PASS")
    else:
        verdict = "KILL" if d_values[0] < 0.5 else "UNKNOWN"

    print(f"\n  N-convergence: d = {d_values}")
    if len(d_values) >= 2:
        print(f"  Ratio d(Nmax)/d(Nmin) = {d_values[-1]/max(d_values[0], 0.01):.3f}")
    print(f"  VERDICT: {verdict}")

    return {"test": "n_convergence", "results": results, "verdict": verdict}


# ═══════════════════════════════════════════════════════════
# TEST 3: PEARSON EQUIVALENCE
# ═══════════════════════════════════════════════════════════
def pearson_equivalence_test(N=300, M=20, eps=5.0, T=1.0, zeta=0.15):
    """Check if dcor adds information beyond Pearson r^2."""
    print("\n" + "=" * 60)
    print(f"TEST 3: PEARSON EQUIVALENCE (N={N}, M={M})")
    print("=" * 60, flush=True)

    dcor_vals = []
    pearson_vals = []
    pearson_sq_vals = []

    for trial in range(M):
        seed = 6666000 + trial * 1000
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        mask = bulk_mask(pts, T, zeta)
        idx = np.where(mask)[0]

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parC, chC = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        Y_curv = Y_from_graph(parC, chC)

        delta_Y = Y_curv - Y0
        X = Y0 - np.mean(Y0[mask])
        absX = np.abs(X[idx])
        dY2 = delta_Y[idx] ** 2

        dc = dcor(absX, dY2)
        dcor_vals.append(dc)

        if np.std(absX) > 1e-15 and np.std(dY2) > 1e-15:
            r = float(pearsonr(absX, dY2)[0])
        else:
            r = 0.0
        pearson_vals.append(r)
        pearson_sq_vals.append(r ** 2)

    # Correlation between dcor and r^2 across trials
    rho_dcor_r2 = float(spearmanr(dcor_vals, pearson_sq_vals)[0])
    rho_dcor_r = float(spearmanr(dcor_vals, pearson_vals)[0])

    print(f"  Spearman(dcor, r^2) = {rho_dcor_r2:.3f}")
    print(f"  Spearman(dcor, r)   = {rho_dcor_r:.3f}")
    print(f"  d(dcor) = {cohens_d(dcor_vals):.3f}")
    print(f"  d(r)    = {cohens_d(pearson_vals):.3f}")
    print(f"  d(r^2)  = {cohens_d(pearson_sq_vals):.3f}")

    redundant = abs(rho_dcor_r2) > 0.8
    print(f"  VERDICT: {'REDUNDANT' if redundant else 'INDEPENDENT'} (threshold: |rho| > 0.8)")

    return {
        "test": "pearson_equivalence",
        "spearman_dcor_r2": round(rho_dcor_r2, 3),
        "spearman_dcor_r": round(rho_dcor_r, 3),
        "d_dcor": round(cohens_d(dcor_vals), 3),
        "d_pearson": round(cohens_d(pearson_vals), 3),
        "d_pearson_sq": round(cohens_d(pearson_sq_vals), 3),
        "verdict": "REDUNDANT" if redundant else "INDEPENDENT",
    }


# ═══════════════════════════════════════════════════════════
# TEST 4: STRATIFIED dcor
# ═══════════════════════════════════════════════════════════
def stratified_dcor_test(N=300, M=20, eps=5.0, T=1.0, zeta=0.15):
    """Compare global dcor vs within-stratum dcor."""
    print("\n" + "=" * 60)
    print(f"TEST 4: STRATIFIED dcor (N={N}, M={M})")
    print("=" * 60, flush=True)

    dcor_global = []
    dcor_strat = []

    for trial in range(M):
        seed = 5555000 + trial * 1000
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        mask = bulk_mask(pts, T, zeta)
        idx = np.where(mask)[0]

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parC, chC = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        Y_curv = Y_from_graph(parC, chC)

        delta_Y = Y_curv - Y0
        X = Y0 - np.mean(Y0[mask])
        absX = np.abs(X[idx])
        dY2 = delta_Y[idx] ** 2

        dc_global = dcor(absX, dY2)
        dcor_global.append(dc_global)

        # Stratified dcor
        strata = make_strata(pts, T)
        st = strata[idx]
        dc_strat_vals = []
        weights = []
        for s in np.unique(st):
            sel = st == s
            if sel.sum() < 10:
                continue
            dc_s = dcor(absX[sel], dY2[sel])
            dc_strat_vals.append(dc_s)
            weights.append(float(sel.sum()))
        if dc_strat_vals:
            weights = np.array(weights) / sum(weights)
            dc_mean_strat = float(np.sum(np.array(dc_strat_vals) * weights))
        else:
            dc_mean_strat = 0.0
        dcor_strat.append(dc_mean_strat)

    d_global = cohens_d(dcor_global)
    d_strat = cohens_d(dcor_strat)
    ratio = d_strat / d_global if abs(d_global) > 0.01 else 0.0

    print(f"  d(global) = {d_global:.3f}, mean = {np.mean(dcor_global):.4f}")
    print(f"  d(strat)  = {d_strat:.3f}, mean = {np.mean(dcor_strat):.4f}")
    print(f"  Ratio d_strat/d_global = {ratio:.3f}")

    # If stratified dcor collapses, global dcor was measuring inter-stratum density gradient
    artifact = ratio < 0.3
    print(f"  VERDICT: {'DENSITY ARTIFACT' if artifact else 'GENUINE'}")

    return {
        "test": "stratified_dcor",
        "d_global": round(d_global, 3),
        "d_strat": round(d_strat, 3),
        "ratio": round(ratio, 3),
        "mean_global": round(float(np.mean(dcor_global)), 4),
        "mean_strat": round(float(np.mean(dcor_strat)), 4),
        "verdict": "DENSITY_ARTIFACT" if artifact else "GENUINE",
    }


# ═══════════════════════════════════════════════════════════
# TEST 5: SUBSAMPLE SENSITIVITY
# ═══════════════════════════════════════════════════════════
def subsample_sensitivity_test(N=300, M=15, eps=5.0, T=1.0, zeta=0.15):
    """Check sensitivity to subsample size."""
    print("\n" + "=" * 60)
    print(f"TEST 5: SUBSAMPLE SENSITIVITY (N={N}, M={M})")
    print("=" * 60, flush=True)

    sub_sizes = [30, 50, 100, 150, 999]  # 999 = ALL
    results = {}

    for sub_n in sub_sizes:
        dcor_vals = []
        for trial in range(M):
            seed = 4444000 + trial * 1000
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            mask = bulk_mask(pts, T, zeta)
            idx = np.where(mask)[0]

            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            parC, chC = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
            Y_curv = Y_from_graph(parC, chC)

            delta_Y = Y_curv - Y0
            X = Y0 - np.mean(Y0[mask])
            absX = np.abs(X[idx])
            dY2 = delta_Y[idx] ** 2

            if sub_n >= len(absX):
                dc = dcor(absX, dY2)
            else:
                sub_rng = np.random.default_rng(42)
                sub_idx = sub_rng.choice(len(absX), sub_n, replace=False)
                dc = dcor(absX[sub_idx], dY2[sub_idx])
            dcor_vals.append(dc)

        d = cohens_d(dcor_vals)
        actual_n = min(sub_n, len(idx)) if sub_n < 999 else len(idx)
        results[sub_n] = {"d": round(d, 3), "mean": round(float(np.mean(dcor_vals)), 4)}
        print(f"  sub_n={sub_n:>4d} (actual ~{actual_n}): d={d:>7.3f}, mean={np.mean(dcor_vals):.4f}")

    # Check if result is stable or depends heavily on subsample size
    d_values = [results[k]["d"] for k in sorted(results.keys())]
    cv = np.std(d_values) / max(np.mean(d_values), 0.01)
    unstable = cv > 0.5

    print(f"  CV of d across sub-sizes = {cv:.3f}")
    print(f"  VERDICT: {'UNSTABLE' if unstable else 'STABLE'}")

    return {"test": "subsample_sensitivity", "results": results,
            "cv": round(cv, 3), "verdict": "UNSTABLE" if unstable else "STABLE"}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("DEMOLISH: dcor(|X|, delta_Y^2)")
    print("=" * 60, flush=True)
    t0 = time.time()

    all_results = {}

    # Test 1: Adversary (most important)
    all_results["adversary"] = adversary_test(N=300, M=15)

    # Test 2: N-convergence
    all_results["n_convergence"] = n_convergence_test(N_list=[300, 500], M=12)

    # Test 3: Pearson equivalence
    all_results["pearson_equiv"] = pearson_equivalence_test(N=300, M=20)

    # Test 4: Stratified dcor
    all_results["stratified"] = stratified_dcor_test(N=300, M=15)

    # Test 5: Subsample sensitivity
    all_results["subsample"] = subsample_sensitivity_test(N=300, M=12)

    elapsed = time.time() - t0
    all_results["meta"] = {"elapsed_sec": round(elapsed, 1)}

    # Overall verdict
    kills = sum(1 for k, v in all_results.items()
                if isinstance(v, dict) and v.get("verdict") in ["KILL", "DENSITY_ARTIFACT", "REDUNDANT", "UNSTABLE"])
    total = sum(1 for k, v in all_results.items()
                if isinstance(v, dict) and "verdict" in v)

    print(f"\n{'='*60}")
    print(f"DEMOLISH SUMMARY ({elapsed:.1f}s)")
    print(f"  Kills: {kills}/{total}")
    for k, v in all_results.items():
        if isinstance(v, dict) and "verdict" in v:
            print(f"  {k}: {v['verdict']}")
    print(f"  OVERALL: {'KILL' if kills >= 1 else 'PASS'}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "results", "demolish_dcor.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(all_results, open(out_path, "w"), indent=2, default=float)
    print(f"\nSaved to {out_path}")
