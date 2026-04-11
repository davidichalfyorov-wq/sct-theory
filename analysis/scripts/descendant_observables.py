#!/usr/bin/env python3
"""Descendant observables: A_align and Moran's I.

Tests two analytical-suggested "cleaner" observables:
1. A_align = Σ_B w_B Cov_B(X², δY²) — alignment channel, predicted R_scr ≈ 0
2. Moran's I of δY on Hasse graph — spatial autocorrelation

Both computed on same CRN data as path_kurtosis.
Also computes R_scr for each (scrambling within strata).
"""
import sys, os, time, json
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, bulk_mask
)

N = 10000
ZETA = 0.15
EPS = 3.0
T = 1.0
M_SEEDS = 10
K_PERM = 50


def compute_A_align(Y0, delta, mask, strata, n_strata):
    """A_align = Σ_B w_B Cov_B(X², δY²) where X = Y0 - mean(Y0)."""
    X = Y0[mask] - np.mean(Y0[mask])
    X2 = X ** 2
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]

    total_cov = 0.0
    total_w = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        n_b = idx.sum()
        if n_b < 3:
            continue
        w_b = n_b / len(X)
        cov_b = np.mean(X2[idx] * dY2[idx]) - np.mean(X2[idx]) * np.mean(dY2[idx])
        total_cov += w_b * cov_b
        total_w += w_b

    return float(total_cov)


def compute_morans_I(delta, parents, children, mask):
    """Moran's I of δY on Hasse graph (links as adjacency)."""
    idx_map = np.where(mask)[0]
    n = len(idx_map)
    if n < 10:
        return 0.0

    # Build adjacency for bulk elements
    idx_set = set(idx_map.tolist())
    idx_to_local = {g: l for l, g in enumerate(idx_map)}

    dY = delta[idx_map]
    dY_centered = dY - np.mean(dY)

    numerator = 0.0
    W = 0
    for local_i, global_i in enumerate(idx_map):
        for j in parents[global_i]:
            j = int(j)
            if j in idx_to_local:
                local_j = idx_to_local[j]
                numerator += dY_centered[local_i] * dY_centered[local_j]
                W += 1
        for j in children[global_i]:
            j = int(j)
            if j in idx_to_local:
                local_j = idx_to_local[j]
                numerator += dY_centered[local_i] * dY_centered[local_j]
                W += 1

    denominator = np.sum(dY_centered ** 2)
    if W == 0 or denominator < 1e-15:
        return 0.0

    I = (n / W) * (numerator / denominator)
    return float(I)


def make_strata(pts, parents0, T, mask):
    n_tau, n_rho, n_depth = 5, 3, 3
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)

    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i].size > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)

    return tau_bin * 9 + rho_bin * 3 + depth_terc


def scramble_and_compute(Y0, delta, mask, strata, obs_fn, rng):
    delta_scr = delta.copy()
    for label in np.unique(strata[mask]):
        idx = np.where((strata == label) & mask)[0]
        if len(idx) > 1:
            delta_scr[idx] = delta[rng.permutation(idx)]
    return obs_fn(Y0, delta_scr, mask)


if __name__ == "__main__":
    print("=== DESCENDANT OBSERVABLES ===", flush=True)
    print(f"N={N}, T={T}, eps={EPS}, M={M_SEEDS}, K_perm={K_PERM}", flush=True)
    print(flush=True)

    t_total = time.time()
    results = {
        "kurtosis": {"curv": [], "scr": []},
        "A_align": {"curv": [], "scr": []},
        "morans_I": {"curv": [], "scr": []},
    }

    for si in range(M_SEEDS):
        seed = 1500000 + si
        rng_spr = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng_spr)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YF = Y_from_graph(parF, chF)

        mask = bulk_mask(pts, T, ZETA)
        delta = YF - Y0
        strata = make_strata(pts, par0, T, mask)
        n_strata = len(np.unique(strata[mask]))

        # Curvature values
        dk_curv = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
        aalign_curv = compute_A_align(Y0, delta, mask, strata, n_strata)
        moran_curv = compute_morans_I(delta, par0, ch0, mask)

        results["kurtosis"]["curv"].append(dk_curv)
        results["A_align"]["curv"].append(aalign_curv)
        results["morans_I"]["curv"].append(moran_curv)

        # Scrambled values
        rng_scr = np.random.default_rng(seed + 80000)
        scr_k, scr_a, scr_m = [], [], []
        for k in range(K_PERM):
            delta_scr = delta.copy()
            for label in np.unique(strata[mask]):
                idx = np.where((strata == label) & mask)[0]
                if len(idx) > 1:
                    delta_scr[idx] = delta[rng_scr.permutation(idx)]

            Yscr = Y0 + delta_scr
            scr_k.append(excess_kurtosis(Yscr[mask]) - excess_kurtosis(Y0[mask]))
            scr_a.append(compute_A_align(Y0, delta_scr, mask, strata, n_strata))
            scr_m.append(compute_morans_I(delta_scr, par0, ch0, mask))

        results["kurtosis"]["scr"].append(float(np.mean(scr_k)))
        results["A_align"]["scr"].append(float(np.mean(scr_a)))
        results["morans_I"]["scr"].append(float(np.mean(scr_m)))

        elapsed = time.time() - t_total
        print(f"  seed {si+1}/{M_SEEDS}: "
              f"kurt={dk_curv:+.4f}/{np.mean(scr_k):+.4f}, "
              f"A_align={aalign_curv:+.4f}/{np.mean(scr_a):+.4f}, "
              f"Moran={moran_curv:+.4f}/{np.mean(scr_m):+.4f} "
              f"({elapsed:.0f}s)", flush=True)

    # Summary
    print(f"\n=== RESULTS ===", flush=True)
    for name in ["kurtosis", "A_align", "morans_I"]:
        curv = np.array(results[name]["curv"])
        scr = np.array(results[name]["scr"])
        R = abs(np.mean(scr)) / max(abs(np.mean(curv)), 1e-15)
        d_curv = np.mean(curv) / np.std(curv, ddof=1) if np.std(curv, ddof=1) > 0 else 0
        print(f"  {name:12s}: curv={np.mean(curv):+.6f} (d={d_curv:+.3f}), "
              f"scr={np.mean(scr):+.6f}, R_scr={R:.4f}", flush=True)

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "descendant_observables.json"), "w") as f:
        json.dump({name: {"curv": results[name]["curv"],
                          "scr": results[name]["scr"]}
                   for name in results}, f, indent=2)
    print("Saved.", flush=True)
