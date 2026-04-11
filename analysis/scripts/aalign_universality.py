#!/usr/bin/env python3
"""A_align universality: cubic jet Sch vs exact ppw at N=5000."""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_ppwave_canonical, riemann_schwarzschild_local
)
from u1_cubic_jet_experiment import cubic_jet_preds, nabla_riemann_schwarzschild, nabla_riemann_ppwave

N = 5000
ZETA = 0.15
T = 1.0
M_SEEDS = 20
EPS = 3.0
M_SCH = 0.05
R0 = 0.50

R_PPW = riemann_ppwave_canonical(EPS)
NABR_PPW = nabla_riemann_ppwave(EPS)
R_SCH = riemann_schwarzschild_local(M_SCH, R0)
NABR_SCH = nabla_riemann_schwarzschild(M_SCH, R0)
E2_PPW = EPS**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2


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
    print(f"=== A_ALIGN UNIVERSALITY: N={N}, T={T}, M={M_SEEDS} ===", flush=True)

    aalign_ppw_exact = []
    aalign_sch_quad = []
    aalign_sch_cubic = []

    t0 = time.time()
    for si in range(M_SEEDS):
        seed = 2500000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T, ZETA)
        strata = make_strata(pts, par0)

        # ppw exact
        parE, chE = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YE = Y_from_graph(parE, chE)
        aalign_ppw_exact.append(compute_A_align(Y0, YE - Y0, mask, strata))

        # Sch quadratic
        parQ, chQ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
        YQ = Y_from_graph(parQ, chQ)
        aalign_sch_quad.append(compute_A_align(Y0, YQ - Y0, mask, strata))

        # Sch cubic
        parC, chC = build_hasse_from_predicate(pts, lambda P, i: cubic_jet_preds(P, i, R_abcd=R_SCH, nabla_R=NABR_SCH))
        YC = Y_from_graph(parC, chC)
        aalign_sch_cubic.append(compute_A_align(Y0, YC - Y0, mask, strata))

        if (si + 1) % 2 == 0:
            elapsed = time.time() - t0
            print(f"  {si+1}/{M_SEEDS} ({elapsed:.0f}s)", flush=True)

    def report(name, arr, E2):
        a = np.array(arr)
        m = float(np.mean(a))
        se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
        AE = m / E2 if E2 > 1e-15 else 0
        return m, se, AE

    m1, se1, AE1 = report("ppw_exact", aalign_ppw_exact, E2_PPW)
    m3, se3, AE3 = report("sch_quad", aalign_sch_quad, E2_SCH)
    m4, se4, AE4 = report("sch_cubic", aalign_sch_cubic, E2_SCH)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"  ppw exact:  A_align={m1:+.6f}±{se1:.6f}, A_E={AE1:.6f}", flush=True)
    print(f"  Sch quad:   A_align={m3:+.6f}±{se3:.6f}, A_E={AE3:.6f}", flush=True)
    print(f"  Sch CUBIC:  A_align={m4:+.6f}±{se4:.6f}, A_E={AE4:.6f}", flush=True)
    print(flush=True)

    ratio_quad = AE3 / AE1 if abs(AE1) > 1e-15 else 0
    ratio_cubic = AE4 / AE1 if abs(AE1) > 1e-15 else 0

    print(f"  A_E ratio Sch_quad/ppw  = {ratio_quad:.3f}", flush=True)
    print(f"  A_E ratio Sch_CUBIC/ppw = {ratio_cubic:.3f} <-- THE NUMBER", flush=True)
    print(f"  Cubic improvement: {ratio_cubic/ratio_quad:.2f}x over quadratic", flush=True)
    print(flush=True)

    if 0.80 <= ratio_cubic <= 1.25:
        verdict = "PASS: universality!"
    elif 0.65 <= ratio_cubic <= 1.50:
        verdict = "BORDERLINE"
    else:
        verdict = f"FAIL: ratio {ratio_cubic:.3f} outside [0.65, 1.50]"
    print(f"  VERDICT: {verdict}", flush=True)

    total = time.time() - t0
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aalign_universality.json"), "w") as f:
        json.dump({
            "ppw_exact": {"mean": m1, "se": se1, "A_E": AE1, "per_seed": aalign_ppw_exact},
            "sch_quad": {"mean": m3, "se": se3, "A_E": AE3, "per_seed": aalign_sch_quad},
            "sch_cubic": {"mean": m4, "se": se4, "A_E": AE4, "per_seed": aalign_sch_cubic},
            "ratio_quad": ratio_quad, "ratio_cubic": ratio_cubic, "verdict": verdict,
        }, f, indent=2)
    print("Saved.", flush=True)
