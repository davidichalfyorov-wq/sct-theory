#!/usr/bin/env python3
"""Epsilon-independence test: does C* depend on eps?"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, ppwave_exact_preds, bulk_mask, excess_kurtosis,
)

N = 10000; T = 1.0; ZETA = 0.15; M_SEEDS = 15
EPS_VALUES = [1.0, 2.0, 3.0, 5.0]

def make_strata(pts, par0, T):
    tau_hat = 2*pts[:,0]/T
    r = np.linalg.norm(pts[:,1:], axis=1)
    rmax = T/2 - np.abs(pts[:,0])
    rho_hat = np.clip(r/np.maximum(rmax,1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat+1)*2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat*3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if par0[i] is not None and len(par0[i]) > 0:
            depth[i] = int(np.max(depth[par0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth*3)//(max_d+1), 0, 2)
    return tau_bin*9 + rho_bin*3 + depth_terc

def compute_CJ(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask]**2
    strata_m = strata[mask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3: continue
        w = idx.sum()/len(X)
        cov = np.mean(np.abs(X[idx])*dY2[idx]) - np.mean(np.abs(X[idx]))*np.mean(dY2[idx])
        total += w * cov
    return float(total)

print("="*70, flush=True)
print(f"EPSILON-INDEPENDENCE TEST: eps={EPS_VALUES}", flush=True)
print(f"N={N}, T={T}, M={M_SEEDS}", flush=True)
print("="*70, flush=True)

results = {}
for eps in EPS_VALUES:
    t0 = time.time()
    E2 = eps**2 / 2.0
    cj_list = []; s0_list = []

    for si in range(M_SEEDS):
        seed = 8000000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T, ZETA)
        strata = make_strata(pts, par0, T)
        s0 = float(np.std(Y0[mask]))
        s0_list.append(s0)

        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        YC = Y_from_graph(parC, chC)
        delta = YC - Y0

        cj = compute_CJ(Y0, delta, mask, strata)
        cj_list.append(cj)

        if (si+1) % 5 == 0:
            print(f"  eps={eps}: {si+1}/{M_SEEDS} ({time.time()-t0:.0f}s)", flush=True)

    cj_arr = np.array(cj_list)
    s0_mean = np.mean(s0_list)
    cj_mean = cj_arr.mean()
    normalized = cj_mean / (s0_mean * T**4 * E2)

    results[str(eps)] = {
        "CJ_mean": float(cj_mean),
        "CJ_se": float(cj_arr.std(ddof=1)/np.sqrt(M_SEEDS)),
        "sigma0": float(s0_mean),
        "E2": float(E2),
        "CJ_over_s0_T4_E2": float(normalized),
    }
    print(f"  eps={eps}: CJ={cj_mean:.6f}, CJ/(s0*T4*E2)={normalized:.6f} ({time.time()-t0:.0f}s)", flush=True)

# Analysis
print(f"\n{'='*70}", flush=True)
print("ANALYSIS", flush=True)
print(f"{'='*70}", flush=True)

eps_arr = np.array(EPS_VALUES)
cj_all = np.array([results[str(e)]["CJ_mean"] for e in EPS_VALUES])
cj_over_E2 = cj_all / (eps_arr**2 / 2)

print("CJ / E^2:")
for e, v in zip(EPS_VALUES, cj_over_E2):
    print(f"  eps={e}: {v:.6f}")
cv = np.std(cj_over_E2)/np.mean(cj_over_E2)*100
print(f"CV = {cv:.1f}%")

alpha, _ = np.polyfit(np.log(eps_arr), np.log(cj_all), 1)
print(f"\nPower law: CJ ~ eps^{alpha:.3f} (expected 2.0)")

with open("analysis/fnd1_data/epsilon_independence_test.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved.", flush=True)
