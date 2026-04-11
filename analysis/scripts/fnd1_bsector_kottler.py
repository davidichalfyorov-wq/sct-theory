#!/usr/bin/env python3
"""FND-1 CLOSURE: B-sector (transverse-boosted Schwarzschild) + Kottler (Weyl+Ricci).

Two new vacuum families to close FND-1:
1. Transverse-boosted Schwarzschild: B_ij ≠ 0 → test A_E vs A_B
2. Kottler (Schwarzschild-de Sitter): Weyl + Ricci → test Ricci subtraction on mixed spacetime

Uses run_universal.py infrastructure with new Riemann tensors.
"""
import sys, os, time, json
import numpy as np
import concurrent.futures as cf

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, jet_preds, bulk_mask, excess_kurtosis,
)

# ─────────────────────────────────────────────────────────────
# RIEMANN TENSORS (verified, from independent analysis + our verification)
# ─────────────────────────────────────────────────────────────

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

def _set_riemann(R, a, b, c, d, val):
    for i, j, k, l, v in [
        (a,b,c,d,+val),(b,a,c,d,-val),(a,b,d,c,-val),(b,a,d,c,+val),
        (c,d,a,b,+val),(d,c,a,b,-val),(c,d,b,a,-val),(d,c,b,a,+val),
    ]:
        R[i,j,k,l] = v

def riemann_schwarzschild_static(M, r0):
    q = M / r0**3
    R = np.zeros((4,4,4,4))
    _set_riemann(R, 0,1,0,1, -2*q)
    _set_riemann(R, 0,2,0,2, +q)
    _set_riemann(R, 0,3,0,3, +q)
    _set_riemann(R, 1,2,1,2, -q)
    _set_riemann(R, 1,3,1,3, -q)
    _set_riemann(R, 2,3,2,3, +2*q)
    return R

def riemann_boosted_schwarzschild_transverse(M, r0, v):
    """Transverse-boosted Schwarzschild (boost along e_2). Gives B≠0."""
    R0 = riemann_schwarzschild_static(M, r0)
    g = 1.0 / np.sqrt(1 - v**2)
    L = np.eye(4)
    L[0,0] = g; L[2,2] = g; L[0,2] = g*v; L[2,0] = g*v
    return np.einsum("pa,qb,rc,sd,pqrs->abcd", L, L, L, L, R0, optimize=True)

def riemann_kottler_local(M, r0, Lambda):
    """Kottler = Schwarzschild + constant-curvature Ricci part."""
    R_sch = riemann_schwarzschild_static(M, r0)
    K = Lambda / 3.0
    R_ric = np.zeros((4,4,4,4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    R_ric[a,b,c,d] = K * (ETA[a,c]*ETA[b,d] - ETA[a,d]*ETA[b,c])
    return R_sch + R_ric

def riemann_pure_ds(Lambda):
    """Pure de Sitter (constant curvature, no Weyl). For Ricci subtraction."""
    K = Lambda / 3.0
    R = np.zeros((4,4,4,4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    R[a,b,c,d] = K * (ETA[a,c]*ETA[b,d] - ETA[a,d]*ETA[b,c])
    return R

def electric_magnetic(R):
    E = R[0, 1:, 0, 1:].copy()
    eps = np.zeros((3,3,3))
    eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = +1.0
    eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0
    B = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    B[i,j] += 0.5 * eps[i,k,l] * R[k+1,l+1,0,j+1]
    return E, B

def compute_A12(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask]**2
    strata_m = strata[mask]
    total = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3: continue
        w = idx.sum() / len(X)
        cov = np.mean(np.abs(X[idx])*dY2[idx]) - np.mean(np.abs(X[idx]))*np.mean(dY2[idx])
        total += w * cov
    return float(total)

def make_strata(pts, parents0, T):
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i] is not None and len(parents0[i]) > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

N = 10000
T = 1.0
ZETA = 0.15
M_SCH = 0.05
R0 = 0.50
V_BOOST = 0.50
LAMBDA_KOTTLER = 0.50
M_SEEDS = 30

# Precompute Riemann tensors
R_static = riemann_schwarzschild_static(M_SCH, R0)
R_boosted = riemann_boosted_schwarzschild_transverse(M_SCH, R0, V_BOOST)
R_kottler = riemann_kottler_local(M_SCH, R0, LAMBDA_KOTTLER)
R_pure_ds = riemann_pure_ds(LAMBDA_KOTTLER)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("FND-1 CLOSURE: B-sector + Kottler", flush=True)
    print(f"N={N}, T={T}, M={M_SCH}, r0={R0}, v_boost={V_BOOST}, Lambda={LAMBDA_KOTTLER}", flush=True)
    print("=" * 70, flush=True)

    # Print E/B decomposition
    for name, R_tensor in [("Static Sch", R_static), ("Boosted Sch (axis=2)", R_boosted),
                            ("Kottler", R_kottler), ("Pure dS", R_pure_ds)]:
        E, B = electric_magnetic(R_tensor)
        E2 = np.sum(E**2); B2 = np.sum(B**2)
        print(f"\n  {name}: E^2={E2:.4f}, B^2={B2:.4f}, E^2+B^2={E2+B2:.4f}", flush=True)

    configs = {
        "static_sch": R_static,
        "boosted_sch": R_boosted,
        "kottler": R_kottler,
        "pure_ds": R_pure_ds,
    }

    results = {}
    t_total = time.time()

    for config_name, R_abcd in configs.items():
        print(f"\n--- {config_name} ---", flush=True)
        t0 = time.time()

        a12_list = []
        dk_list = []

        for si in range(M_SEEDS):
            seed = 7000000 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            # Flat Hasse
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)
            strata = make_strata(pts, par0, T)

            # Curved Hasse
            parC, chC = build_hasse_from_predicate(
                pts, lambda P, i: jet_preds(P, i, R_abcd=R_abcd))
            YC = Y_from_graph(parC, chC)
            delta = YC - Y0

            a12 = compute_A12(Y0, delta, mask, strata)
            dk = excess_kurtosis(YC[mask]) - excess_kurtosis(Y0[mask])

            a12_list.append(a12)
            dk_list.append(dk)

            if (si + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {si+1}/{M_SEEDS} ({elapsed:.0f}s)", flush=True)

        a12_arr = np.array(a12_list)
        dk_arr = np.array(dk_list)
        mn = a12_arr.mean()
        se = a12_arr.std(ddof=1) / np.sqrt(M_SEEDS)
        d_cohen = mn / (a12_arr.std(ddof=1) + 1e-15)

        E_tensor, B_tensor = electric_magnetic(R_abcd)
        E2 = np.sum(E_tensor**2)
        B2 = np.sum(B_tensor**2)

        elapsed = time.time() - t0
        results[config_name] = {
            "mean_A12": float(mn), "se_A12": float(se), "d_cohen": float(d_cohen),
            "mean_dk": float(dk_arr.mean()), "se_dk": float(dk_arr.std(ddof=1)/np.sqrt(M_SEEDS)),
            "E2": float(E2), "B2": float(B2),
            "per_seed_A12": a12_list, "per_seed_dk": dk_list,
            "elapsed_s": elapsed,
        }
        print(f"  A12={mn:.6f}±{se:.6f}, d={d_cohen:.2f}, E²={E2:.4f}, B²={B2:.4f} ({elapsed:.0f}s)", flush=True)

    # ─── ANALYSIS ───
    print(f"\n{'='*70}", flush=True)
    print("ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    # B-sector: A_E vs A_B
    # Static Sch: E²=0.96, B²=0. Boosted: E²=2.24, B²=1.28.
    # If A_E = A_B: CJ_boosted/CJ_static = (E²+B²)/(E²) = (2.24+1.28)/0.96 = 3.67
    # If A_E only: CJ_boosted/CJ_static = E²_boosted/E²_static = 2.24/0.96 = 2.33
    a12_static = results["static_sch"]["mean_A12"]
    a12_boosted = results["boosted_sch"]["mean_A12"]
    ratio_boost = a12_boosted / a12_static if abs(a12_static) > 1e-15 else 0

    E2_static = results["static_sch"]["E2"]
    E2_boosted = results["boosted_sch"]["E2"]
    B2_boosted = results["boosted_sch"]["B2"]

    pred_AE_only = E2_boosted / E2_static
    pred_AE_AB = (E2_boosted + B2_boosted) / E2_static

    print(f"\n  B-SECTOR TEST:", flush=True)
    print(f"    CJ_static = {a12_static:.6f}", flush=True)
    print(f"    CJ_boosted = {a12_boosted:.6f}", flush=True)
    print(f"    Ratio boosted/static = {ratio_boost:.3f}", flush=True)
    print(f"    Predicted if A_E only: {pred_AE_only:.3f}", flush=True)
    print(f"    Predicted if A_E = A_B: {pred_AE_AB:.3f}", flush=True)
    if abs(ratio_boost - pred_AE_AB) < abs(ratio_boost - pred_AE_only):
        print(f"    → CLOSER TO A_E=A_B", flush=True)
    else:
        print(f"    → CLOSER TO A_E ONLY", flush=True)

    # Kottler: Ricci subtraction
    a12_kottler = results["kottler"]["mean_A12"]
    a12_ds = results["pure_ds"]["mean_A12"]
    a12_weyl_subtracted = a12_kottler - a12_ds

    print(f"\n  KOTTLER (Weyl+Ricci) TEST:", flush=True)
    print(f"    CJ_kottler = {a12_kottler:.6f}", flush=True)
    print(f"    CJ_pure_dS = {a12_ds:.6f}", flush=True)
    print(f"    CJ_weyl = CJ_kottler - CJ_dS = {a12_weyl_subtracted:.6f}", flush=True)
    print(f"    CJ_static_sch = {a12_static:.6f}", flush=True)
    print(f"    Ratio weyl/static = {a12_weyl_subtracted/a12_static:.3f} (should be ~1 if Weyl part same)", flush=True)

    total_time = time.time() - t_total
    results["total_time_min"] = total_time / 60
    print(f"\nTotal: {total_time/60:.1f} min", flush=True)

    out_path = "analysis/fnd1_data/bsector_kottler_results.json"
    # Remove per_seed lists for JSON size
    results_save = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_save[k] = {kk: vv for kk, vv in v.items() if kk not in ("per_seed_A12", "per_seed_dk")}
        else:
            results_save[k] = v
    with open(out_path, "w") as f:
        json.dump(results_save, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
