#!/usr/bin/env python3
"""A_align exact-expmap midpoint at N=10000, M=20 seeds.
If ratio continues trend: 0.36(1k) → 0.61(5k) → ???(10k) → PASS?"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from schwarzschild_exact_local_tools import (
    map_rnc_to_schwarzschild_expmap,
    schwarzschild_exact_midpoint_preds_from_mapped
)
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_schwarzschild_local
)

N = 10000
ZETA = 0.15
T = 1.0
M_SEEDS = 50
EPS = 3.0
M_SCH = 0.05
R0 = 0.50
E2_PPW = EPS**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2
R_SCH = riemann_schwarzschild_local(M_SCH, R0)


def make_strata(pts, parents0):
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


def compute_A_align(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    X2 = X ** 2
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]
    total_cov = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w_b = idx.sum() / len(X)
        cov_b = np.mean(X2[idx] * dY2[idx]) - np.mean(X2[idx]) * np.mean(dY2[idx])
        total_cov += w_b * cov_b
    return float(total_cov)


if __name__ == "__main__":
    print(f"=== A_ALIGN EXACT-EXPMAP N={N} M={M_SEEDS} ===", flush=True)
    print(f"Trend: ratio 0.36(1k) → 0.61(5k) → ???(10k)", flush=True)
    print(flush=True)

    aalign_ppw = []
    aalign_sch_expmap = []
    aalign_sch_quad = []

    t0 = time.time()
    for si in range(M_SEEDS):
        seed = 3000000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        mapped = map_rnc_to_schwarzschild_expmap(pts, M_SCH, R0)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T, ZETA)
        strata = make_strata(pts, par0)

        parE, chE = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YE = Y_from_graph(parE, chE)
        aalign_ppw.append(compute_A_align(Y0, YE - Y0, mask, strata))

        parX, chX = build_hasse_from_predicate(
            pts, lambda P, i: schwarzschild_exact_midpoint_preds_from_mapped(mapped, i, M_SCH))
        YX = Y_from_graph(parX, chX)
        aalign_sch_expmap.append(compute_A_align(Y0, YX - Y0, mask, strata))

        parQ, chQ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
        YQ = Y_from_graph(parQ, chQ)
        aalign_sch_quad.append(compute_A_align(Y0, YQ - Y0, mask, strata))

        elapsed = time.time() - t0
        print(f"  {si+1}/{M_SEEDS}: ppw={aalign_ppw[-1]:+.4f} sch_x={aalign_sch_expmap[-1]:+.4f} "
              f"sch_q={aalign_sch_quad[-1]:+.4f} ({elapsed:.0f}s)", flush=True)

    def report(arr, E2):
        a = np.array(arr)
        m = float(np.mean(a))
        se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
        AE = m / E2 if E2 > 1e-15 else 0
        d = m / np.std(a, ddof=1) if np.std(a, ddof=1) > 0 else 0
        return m, se, AE, d

    m1, se1, AE1, d1 = report(aalign_ppw, E2_PPW)
    m2, se2, AE2, d2 = report(aalign_sch_expmap, E2_SCH)
    m3, se3, AE3, d3 = report(aalign_sch_quad, E2_SCH)

    ratio_expmap = AE2 / AE1 if abs(AE1) > 1e-15 else 0
    ratio_quad = AE3 / AE1 if abs(AE1) > 1e-15 else 0

    print(f"\n=== RESULTS ===", flush=True)
    print(f"  ppw exact:      A_align={m1:+.6f}±{se1:.6f}, A_E={AE1:.6f}, d={d1:+.3f}", flush=True)
    print(f"  Sch EXACT-EXP:  A_align={m2:+.6f}±{se2:.6f}, A_E={AE2:.6f}, d={d2:+.3f}", flush=True)
    print(f"  Sch quad jet:   A_align={m3:+.6f}±{se3:.6f}, A_E={AE3:.6f}, d={d3:+.3f}", flush=True)

    print(f"\n  Ratio Sch_expmap/ppw = {ratio_expmap:.3f} <-- THE NUMBER", flush=True)
    print(f"  Ratio Sch_quad/ppw   = {ratio_quad:.3f}", flush=True)

    print(f"\n=== N-SCALING OF RATIO ===", flush=True)
    print(f"  N=1000:  ratio_expmap = 0.363", flush=True)
    print(f"  N=5000:  ratio_expmap = 0.612", flush=True)
    print(f"  N=10000: ratio_expmap = {ratio_expmap:.3f}", flush=True)

    if 0.80 <= ratio_expmap <= 1.25:
        verdict = "PASS: universality!"
    elif 0.65 <= ratio_expmap <= 1.50:
        verdict = "BORDERLINE"
    elif ratio_expmap > 0.50:
        verdict = "ALIVE"
    else:
        verdict = f"FAIL: ratio {ratio_expmap:.3f}"
    print(f"\n  VERDICT: {verdict}", flush=True)

    total = time.time() - t0
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aalign_expmap_n10000.json"), "w") as f:
        json.dump({
            "ppw": {"mean": m1, "se": se1, "A_E": AE1, "d": d1, "per_seed": aalign_ppw},
            "sch_expmap": {"mean": m2, "se": se2, "A_E": AE2, "d": d2, "per_seed": aalign_sch_expmap},
            "sch_quad": {"mean": m3, "se": se3, "A_E": AE3, "d": d3, "per_seed": aalign_sch_quad},
            "ratio_expmap": ratio_expmap, "ratio_quad": ratio_quad, "verdict": verdict,
            "N_scaling": {"N1000": 0.363, "N5000": 0.612, "N10000": ratio_expmap},
        }, f, indent=2)
    print("Saved.", flush=True)
