#!/usr/bin/env python3
"""NC-3: Stratified δY-scrambling control.

Tests whether the kurtosis signal is carried by coherent spatial organization
of the response field, or merely by its one-point marginal distribution.

Method:
1. Compute δY = Y_curv − Y₀ (physical response)
2. Define strata: (τ̂, ρ̂) bins × flat depth tercile
3. Permute δY values within each stratum
4. Reconstruct Y_scr = Y₀ + δY_scr
5. Compute dk_scr = kurtosis(Y_scr) − kurtosis(Y₀)
6. Repeat K=100 permutations per seed

If dk_scr << dk_curv → signal is in SPATIAL ORGANIZATION (good!)
If dk_scr ≈ dk_curv → signal is just marginal shape (bad)
"""
import sys, os, time, json
import numpy as np

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
K_PERM = 100
N_TAU, N_RHO, N_DEPTH = 5, 3, 3


def compute_strata(pts, parents0, T, n_tau, n_rho, n_depth):
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)

    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 0.5 * n_tau).astype(int), 0, n_tau - 1)
    rho_bin = np.clip(np.floor(rho_hat * n_rho).astype(int), 0, n_rho - 1)

    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i].size > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * n_depth) // (max_d + 1), 0, n_depth - 1)

    return tau_bin * n_rho * n_depth + rho_bin * n_depth + depth_terc


def scramble_deltaY(delta, window_mask, strata_labels, rng):
    delta_scr = delta.copy()
    for label in np.unique(strata_labels[window_mask]):
        idx = np.where((strata_labels == label) & window_mask)[0]
        if len(idx) > 1:
            delta_scr[idx] = delta[rng.permutation(idx)]
    return delta_scr


if __name__ == "__main__":
    print(f"=== NC-3: STRATIFIED δY-SCRAMBLING CONTROL ===", flush=True)
    print(f"N={N}, T={T}, eps={EPS}, M_seeds={M_SEEDS}, K_perm={K_PERM}", flush=True)
    print(f"Strata: {N_TAU}τ × {N_RHO}ρ × {N_DEPTH}depth = {N_TAU*N_RHO*N_DEPTH}", flush=True)
    print(flush=True)

    all_dk_curv = []
    all_dk_scr = []

    t_total = time.time()
    for si in range(M_SEEDS):
        seed = 1100000 + si
        rng_spr = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng_spr)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YF = Y_from_graph(parF, chF)

        mask = bulk_mask(pts, T, ZETA)
        dk_curv = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
        all_dk_curv.append(dk_curv)

        delta = YF - Y0
        strata = compute_strata(pts, par0, T, N_TAU, N_RHO, N_DEPTH)

        scr_vals = []
        rng_scr = np.random.default_rng(seed + 50000)
        for k in range(K_PERM):
            delta_scr = scramble_deltaY(delta, mask, strata, rng_scr)
            Yscr = Y0 + delta_scr
            dk_scr = excess_kurtosis(Yscr[mask]) - excess_kurtosis(Y0[mask])
            scr_vals.append(dk_scr)

        mean_scr = float(np.mean(scr_vals))
        all_dk_scr.append(mean_scr)

        elapsed = time.time() - t_total
        print(f"  seed {si}: dk_curv={dk_curv:+.6f}, mean_dk_scr={mean_scr:+.6f}, "
              f"ratio={abs(mean_scr)/max(abs(dk_curv),1e-15):.3f} ({elapsed:.0f}s)", flush=True)

    arr_curv = np.array(all_dk_curv)
    arr_scr = np.array(all_dk_scr)
    R_scr = abs(np.mean(arr_scr)) / max(abs(np.mean(arr_curv)), 1e-15)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"  dk_curv:     mean={np.mean(arr_curv):+.6f} ± {np.std(arr_curv,ddof=1)/np.sqrt(len(arr_curv)):.6f}", flush=True)
    print(f"  dk_scrambled: mean={np.mean(arr_scr):+.6f} ± {np.std(arr_scr,ddof=1)/np.sqrt(len(arr_scr)):.6f}", flush=True)
    print(f"  R_scr = {R_scr:.4f}", flush=True)

    if R_scr < 0.25:
        verdict = "STRONG: spatial organization carries the signal"
    elif R_scr < 0.50:
        verdict = "MODERATE: partial organization dependence"
    elif R_scr < 0.75:
        verdict = "WEAK: marginal shape dominates"
    else:
        verdict = "BAD: scrambling preserves signal — no organization dependence"
    print(f"  VERDICT: {verdict}", flush=True)

    total = time.time() - t_total
    print(f"\n  Total: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "nc3_deltaY_scramble.json"), "w") as f:
        json.dump({
            "dk_curv": arr_curv.tolist(),
            "dk_scr_means": arr_scr.tolist(),
            "R_scr": float(R_scr),
            "verdict": verdict,
        }, f, indent=2)
    print("Saved.", flush=True)
