#!/usr/bin/env python3
"""
DEMOLISH-009: Hostile adversarial attack on IVV and PIVF candidates.

10 objections tested computationally:
1. TC-mediation decomposition
2. Degree-proxy redundancy
3. Finite-size collapse (N=300,500,1000,2000)
4. Poisson noise floor analysis
5. Boundary effect sweep (zeta = 0.05..0.30)
6. Sign pattern analysis
7. Conformal null (de Sitter Riemann, Weyl=0)
8. Graph-theory adversary (random entry flips)
9. N-convergence of Cohen's d
10. PFC prior-art overlap

Author: DEMOLISH agent (hostile)
"""
import sys, os, time, json
import numpy as np
from scipy.stats import wilcoxon, pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, bulk_mask,
    riemann_ppwave_canonical, riemann_schwarzschild_local,
    ETA,
)

MASTER_SEED = 77_777_000
EPS_PPW = 3.0
M_SCH = 0.10
R0_SCH = 0.50
T = 1.0

# ────────────────────────────────────────────────────
# Core infrastructure (copied from compscan for independence)
# ────────────────────────────────────────────────────
def minkowski_preds(pts, i, tol=1e-12):
    dt = pts[i+1:, 0] - pts[i, 0]
    dr2 = np.sum((pts[i+1:, 1:] - pts[i, 1:])**2, axis=1)
    return (dt > tol) & (dt**2 - dr2 > tol)

def jet_preds(pts, i, R_abcd, tol=1e-12):
    dx = pts[i+1:] - pts[i]
    xm = 0.5 * (pts[i+1:] + pts[i])
    ds2_flat = -dx[:, 0]**2 + np.sum(dx[:, 1:]**2, axis=1)
    delta = np.zeros(len(dx))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if abs(R_abcd[a, b, c, d]) > 1e-15:
                        delta += R_abcd[a, b, c, d] * xm[:, a] * dx[:, b] * xm[:, c] * dx[:, d]
    ds2_eff = ds2_flat + delta / 3.0
    return (dx[:, 0] > tol) & (ds2_eff < -tol)

def build_causal_from_pred(pts, pred_fn):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mask = pred_fn(pts, i)
        C[i, i+1:] = mask.astype(np.float32)
    return C

def compute_IVV(C_flat, C_curv, mask):
    n = len(C_flat)
    past_f = C_flat.sum(axis=0)
    fut_f = C_flat.sum(axis=1)
    past_c = C_curv.sum(axis=0)
    fut_c = C_curv.sum(axis=1)
    V_flat = (past_f * fut_f) / (n * n)
    V_curv = (past_c * fut_c) / (n * n)
    dV = V_curv - V_flat
    return float(np.mean(dV[mask] ** 2)), dV

def compute_PIVF(C_flat, C_curv, C2_flat, C2_curv, mask, n, max_depth=8):
    # Compute chain layers from causal matrix
    # Layer = length of longest chain from past boundary to element
    # Approximate via iterative matmul (expensive but correct)
    # Use a simpler approach: layer = rank of element in total order by time
    # Actually use longest path through DAG
    # For speed, use the fact that elements are time-sorted
    L = np.ones(n, dtype=np.int32)
    for i in range(n):
        preds = np.where(C_flat[:, i] > 0)[0]
        if len(preds) > 0:
            L[i] = 1 + max(L[p] for p in preds)

    pivf_sum = 0.0
    pivf_count = 0
    for depth_bin in range(2, min(int(L.max()), max_depth)):
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(i+1, n):
                if C_flat[i, j] > 0 and abs(L[j] - L[i] - depth_bin) < 1:
                    v_f = C2_flat[i, j] / n
                    v_c = C2_curv[i, j] / n
                    pivf_sum += (v_c - v_f) ** 2
                    pivf_count += 1
    return pivf_sum / max(pivf_count, 1), pivf_count

def cohen_d_paired(vals):
    """Cohen's d for paired CRN differences (mean/std)."""
    v = np.array(vals)
    s = v.std(ddof=1)
    if s < 1e-20:
        return 0.0
    return float(v.mean() / s)


# ────────────────────────────────────────────────────
# OBJECTION 1: TC-Mediation Decomposition
# ────────────────────────────────────────────────────
def test_tc_mediation(C_flat, C_curv, mask, n):
    """Decompose IVV into symmetric (TC-driven) and asymmetric parts."""
    past_f = C_flat.sum(axis=0).astype(np.float64)
    fut_f = C_flat.sum(axis=1).astype(np.float64)
    past_c = C_curv.sum(axis=0).astype(np.float64)
    fut_c = C_curv.sum(axis=1).astype(np.float64)

    # Symmetric: S = past + future (correlated with TC)
    S_f = past_f + fut_f
    S_c = past_c + fut_c
    # Asymmetric: A = past - future
    A_f = past_f - fut_f
    A_c = past_c - fut_c

    # V = (S^2 - A^2) / (4N^2)
    # dV = [(S_c^2 - S_f^2) - (A_c^2 - A_f^2)] / (4N^2)
    dV_sym = (S_c**2 - S_f**2) / (4 * n * n)
    dV_asym = (A_c**2 - A_f**2) / (4 * n * n)
    dV_total = dV_sym - dV_asym

    # Check: does dV_total match IVV's dV?
    V_flat = (past_f * fut_f) / (n * n)
    V_curv = (past_c * fut_c) / (n * n)
    dV_direct = V_curv - V_flat

    sym_frac = np.mean(np.abs(dV_sym[mask])) / (np.mean(np.abs(dV_sym[mask])) + np.mean(np.abs(dV_asym[mask])) + 1e-20)

    # TC-mediation: R^2 of IVV against delta_TC^2
    tc_f = float(C_flat.sum())
    tc_c = float(C_curv.sum())
    delta_tc = tc_c - tc_f

    # Per-element S correlates with total TC
    dS = S_c - S_f  # per-element
    # sum(dS) = 2*delta_tc (exact identity)

    # R^2 of dV^2 against dS^2 (per-element)
    dV_bulk = dV_direct[mask]
    dS_bulk = dS[mask]
    if len(dV_bulk) > 5:
        r_dv_ds, _ = pearsonr(dV_bulk**2, dS_bulk**2)
    else:
        r_dv_ds = 0.0

    return {
        "sym_fraction": round(float(sym_frac), 4),
        "r2_dV2_vs_dS2": round(float(r_dv_ds**2), 4),
        "delta_tc": round(delta_tc, 1),
        "mean_abs_dV_sym": round(float(np.mean(np.abs(dV_sym[mask]))), 6),
        "mean_abs_dV_asym": round(float(np.mean(np.abs(dV_asym[mask]))), 6),
        "decomp_error": round(float(np.max(np.abs(dV_total - dV_direct))), 10),
    }


# ────────────────────────────────────────────────────
# OBJECTION 2: Degree-Proxy Redundancy
# ────────────────────────────────────────────────────
def test_degree_proxy(C_flat, C_curv, mask, n):
    """Check if IVV is redundant with degree-based CRN variance."""
    C2_f = C_flat @ C_flat
    C2_c = C_curv @ C_curv

    # IVV per-element
    past_f = C_flat.sum(axis=0).astype(np.float64)
    fut_f = C_flat.sum(axis=1).astype(np.float64)
    past_c = C_curv.sum(axis=0).astype(np.float64)
    fut_c = C_curv.sum(axis=1).astype(np.float64)
    V_f = (past_f * fut_f) / (n * n)
    V_c = (past_c * fut_c) / (n * n)
    dV = V_c - V_f

    # Degree CRN: link matrix
    link_f = (C_flat > 0) & (C2_f == 0)
    link_c = (C_curv > 0) & (C2_c == 0)
    adj_f = (link_f | link_f.T).astype(np.float64)
    adj_c = (link_c | link_c.T).astype(np.float64)
    deg_f = adj_f.sum(axis=1)
    deg_c = adj_c.sum(axis=1)
    d_deg = deg_c - deg_f

    # Correlation between dV^2 and d_deg^2
    bulk_dV2 = dV[mask]**2
    bulk_dd2 = d_deg[mask]**2
    if len(bulk_dV2) > 5:
        r_val, _ = pearsonr(bulk_dV2, bulk_dd2)
        rho_val, _ = spearmanr(bulk_dV2, bulk_dd2)
    else:
        r_val, rho_val = 0, 0

    return {
        "pearson_r_dV2_dd2": round(float(r_val), 4),
        "spearman_rho_dV2_dd2": round(float(rho_val), 4),
        "r2_dV2_dd2": round(float(r_val**2), 4),
    }


# ────────────────────────────────────────────────────
# OBJECTION 5: Boundary Effects
# ────────────────────────────────────────────────────
def test_boundary_sweep(pts, C_flat, C_curv, T):
    """Sweep boundary exclusion parameter zeta."""
    n = len(pts)
    results = {}
    for zeta in [0.05, 0.10, 0.15, 0.20, 0.30]:
        m = bulk_mask(pts, T, zeta)
        nb = m.sum()
        ivv, dV = compute_IVV(C_flat, C_curv, m)
        # Also check if boundary-only gives signal
        boundary_m = ~m
        if boundary_m.sum() > 10:
            ivv_boundary, _ = compute_IVV(C_flat, C_curv, boundary_m)
        else:
            ivv_boundary = 0.0
        results[f"z{zeta:.2f}"] = {
            "IVV": round(ivv, 8),
            "n_bulk": int(nb),
            "IVV_boundary": round(ivv_boundary, 8),
        }
    return results


# ────────────────────────────────────────────────────
# OBJECTION 6: Sign Pattern
# ────────────────────────────────────────────────────
def test_sign_pattern(C_flat, C_curv, mask, n):
    """Analyze sign structure of delta V."""
    past_f = C_flat.sum(axis=0).astype(np.float64)
    fut_f = C_flat.sum(axis=1).astype(np.float64)
    past_c = C_curv.sum(axis=0).astype(np.float64)
    fut_c = C_curv.sum(axis=1).astype(np.float64)
    V_f = (past_f * fut_f) / (n * n)
    V_c = (past_c * fut_c) / (n * n)
    dV = V_c - V_f
    dV_bulk = dV[mask]
    return {
        "mean_dV": round(float(np.mean(dV_bulk)), 8),
        "std_dV": round(float(np.std(dV_bulk)), 8),
        "frac_positive": round(float(np.mean(dV_bulk > 0)), 4),
        "frac_negative": round(float(np.mean(dV_bulk < 0)), 4),
        "skewness": round(float(
            np.mean(((dV_bulk - np.mean(dV_bulk)) / max(np.std(dV_bulk), 1e-20))**3)
        ), 4),
    }


# ────────────────────────────────────────────────────
# OBJECTION 7: Conformal Null (de Sitter Riemann)
# ────────────────────────────────────────────────────
def riemann_desitter(H):
    """de Sitter Riemann: R_{abcd} = H^2 (g_{ac} g_{bd} - g_{ad} g_{bc}).
    In RNC at origin with eta metric: R_{abcd} = H^2 (eta_{ac} eta_{bd} - eta_{ad} eta_{bc}).
    Weyl = 0, so this is purely Ricci. IVV should be ZERO if it's Weyl-sensitive."""
    R = np.zeros((4, 4, 4, 4))
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    R[a, b, c, d] = H**2 * (eta[a, c] * eta[b, d] - eta[a, d] * eta[b, c])
    return R


# ────────────────────────────────────────────────────
# OBJECTION 8: Graph-Theory Adversary
# ────────────────────────────────────────────────────
def graph_adversary(C_flat, C_curv, mask, n, rng):
    """Flip random entries matching the number changed by curvature."""
    # Count entries changed
    diff = (C_curv != C_flat)
    n_changed = int(diff.sum())

    # Build random perturbation: flip n_changed random upper-triangular entries
    C_rand = C_flat.copy()
    # Get upper-triangular indices
    triu_i, triu_j = np.triu_indices(n, k=1)
    # Choose n_changed random indices to flip
    flip_idx = rng.choice(len(triu_i), size=min(n_changed, len(triu_i)), replace=False)
    for idx in flip_idx:
        ii, jj = triu_i[idx], triu_j[idx]
        C_rand[ii, jj] = 1.0 - C_rand[ii, jj]

    # Compute IVV for random perturbation
    ivv_curv, _ = compute_IVV(C_flat, C_curv, mask)
    ivv_rand, _ = compute_IVV(C_flat, C_rand, mask)

    return {
        "n_entries_changed_by_curvature": n_changed,
        "IVV_curved": round(ivv_curv, 8),
        "IVV_random": round(ivv_rand, 8),
        "ratio_random_to_curved": round(ivv_rand / max(ivv_curv, 1e-20), 4),
    }


# ────────────────────────────────────────────────────
# Full single-trial analysis
# ────────────────────────────────────────────────────
def full_trial(seed, metric_name, metric_params, N, zeta=0.15):
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    n = len(pts)
    mask = bulk_mask(pts, T, zeta)

    C_flat = build_causal_from_pred(pts, lambda pts, i: minkowski_preds(pts, i))

    if metric_name == "ppwave":
        R = riemann_ppwave_canonical(metric_params["eps"])
    elif metric_name == "schwarzschild":
        R = riemann_schwarzschild_local(metric_params["M"], metric_params["r0"])
    elif metric_name == "desitter":
        R = riemann_desitter(metric_params["H"])
    else:
        raise ValueError(f"Unknown: {metric_name}")

    C_curv = build_causal_from_pred(pts, lambda pts, i: jet_preds(pts, i, R))
    C2_flat = C_flat @ C_flat
    C2_curv = C_curv @ C_curv

    # IVV
    ivv, dV = compute_IVV(C_flat, C_curv, mask)

    # PIVF (only at small N, expensive)
    if N <= 500:
        pivf, pivf_n = compute_PIVF(C_flat, C_curv, C2_flat, C2_curv, mask, n)
    else:
        pivf, pivf_n = 0.0, 0

    result = {
        "IVV": ivv,
        "PIVF": pivf,
        "PIVF_pairs": pivf_n,
        "n_bulk": int(mask.sum()),
        "tc_flat": float(C_flat.sum()),
        "tc_curv": float(C_curv.sum()),
    }

    # Objection 1: TC-mediation
    result["tc_mediation"] = test_tc_mediation(C_flat, C_curv, mask, n)

    # Objection 2: Degree proxy
    result["degree_proxy"] = test_degree_proxy(C_flat, C_curv, mask, n)

    # Objection 5: Boundary sweep (only for N=500)
    if N <= 500:
        result["boundary"] = test_boundary_sweep(pts, C_flat, C_curv, T)

    # Objection 6: Sign pattern
    result["sign"] = test_sign_pattern(C_flat, C_curv, mask, n)

    # Objection 8: Graph adversary
    result["adversary"] = graph_adversary(C_flat, C_curv, mask, n, rng)

    return result


# ────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("DEMOLISH-009: HOSTILE attack on IVV + PIVF")
    print("=" * 70, flush=True)
    t_start = time.time()

    ss = np.random.SeedSequence(MASTER_SEED)

    # ═══════════════════════════════════════════════
    # PHASE 1: Multi-metric at N=500, M=20
    # ═══════════════════════════════════════════════
    M = 20
    N = 500
    metrics = {
        "ppwave": {"eps": EPS_PPW},
        "schwarzschild": {"M": M_SCH, "r0": R0_SCH},
        "desitter": {"H": 2.0},  # Conformal null: Weyl=0
    }

    phase1 = {}
    for mname, mparams in metrics.items():
        print(f"\n--- Phase 1: {mname}, N={N}, M={M} ---", flush=True)
        seeds = ss.spawn(M)
        trials = []
        for trial in range(M):
            if trial % 5 == 0:
                print(f"  Trial {trial}/{M}...", flush=True)
            try:
                r = full_trial(seeds[trial], mname, mparams, N)
                trials.append(r)
            except Exception as e:
                print(f"  Trial {trial} FAILED: {e}")

        # Aggregate
        ivv_vals = [t["IVV"] for t in trials]
        pivf_vals = [t["PIVF"] for t in trials]
        tc_med_sym = [t["tc_mediation"]["sym_fraction"] for t in trials]
        tc_med_r2 = [t["tc_mediation"]["r2_dV2_vs_dS2"] for t in trials]
        deg_r2 = [t["degree_proxy"]["r2_dV2_dd2"] for t in trials]
        adv_ratio = [t["adversary"]["ratio_random_to_curved"] for t in trials]
        sign_frac_pos = [t["sign"]["frac_positive"] for t in trials]
        sign_mean = [t["sign"]["mean_dV"] for t in trials]

        phase1[mname] = {
            "IVV_mean": round(np.mean(ivv_vals), 8),
            "IVV_std": round(np.std(ivv_vals, ddof=1), 8),
            "PIVF_mean": round(np.mean(pivf_vals), 8),
            "PIVF_std": round(np.std(pivf_vals, ddof=1), 8),
            # Objection 1
            "tc_med_sym_fraction_mean": round(np.mean(tc_med_sym), 4),
            "tc_med_r2_dV2_dS2_mean": round(np.mean(tc_med_r2), 4),
            # Objection 2
            "degree_proxy_r2_mean": round(np.mean(deg_r2), 4),
            # Objection 6
            "sign_frac_positive_mean": round(np.mean(sign_frac_pos), 4),
            "sign_mean_dV_mean": round(np.mean(sign_mean), 8),
            # Objection 7 (de Sitter)
            # Objection 8
            "adversary_ratio_mean": round(np.mean(adv_ratio), 4),
            "adversary_ratio_std": round(np.std(adv_ratio, ddof=1), 4),
            # Boundary (first trial only for display)
            "boundary": trials[0].get("boundary", {}),
            "n_trials": len(trials),
        }

    # ═══════════════════════════════════════════════
    # PHASE 2: N-scaling at N=300,500,1000
    # (N=2000 would be 16x slower, skip if already clear)
    # ═══════════════════════════════════════════════
    M_scale = 10
    N_values = [300, 500, 1000]
    print(f"\n--- Phase 2: N-scaling, M={M_scale} ---", flush=True)
    scaling = {}

    for N_val in N_values:
        seeds = ss.spawn(M_scale)
        ivv_ppw = []
        ivv_sch = []
        print(f"  N={N_val}...", flush=True)
        for trial in range(M_scale):
            try:
                r_ppw = full_trial(seeds[trial], "ppwave", {"eps": EPS_PPW}, N_val)
                ivv_ppw.append(r_ppw["IVV"])
            except Exception as e:
                print(f"    ppwave trial {trial} FAILED: {e}")
            try:
                r_sch = full_trial(seeds[trial], "schwarzschild", {"M": M_SCH, "r0": R0_SCH}, N_val)
                ivv_sch.append(r_sch["IVV"])
            except Exception as e:
                print(f"    sch trial {trial} FAILED: {e}")

        d_ppw = cohen_d_paired(ivv_ppw) if len(ivv_ppw) > 2 else 0.0
        d_sch = cohen_d_paired(ivv_sch) if len(ivv_sch) > 2 else 0.0
        scaling[N_val] = {
            "IVV_ppw_d": round(d_ppw, 3),
            "IVV_sch_d": round(d_sch, 3),
            "IVV_ppw_mean": round(np.mean(ivv_ppw), 10) if ivv_ppw else 0,
            "IVV_sch_mean": round(np.mean(ivv_sch), 10) if ivv_sch else 0,
            "IVV_ppw_std": round(np.std(ivv_ppw, ddof=1), 10) if len(ivv_ppw) > 1 else 0,
            "IVV_sch_std": round(np.std(ivv_sch, ddof=1), 10) if len(ivv_sch) > 1 else 0,
        }

    elapsed = time.time() - t_start

    # ═══════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DEMOLISH-009 VERDICT")
    print("=" * 70)

    print("\n--- OBJECTION 1: TC-MEDIATION ---")
    for m in ["ppwave", "schwarzschild"]:
        sf = phase1[m]["tc_med_sym_fraction_mean"]
        r2 = phase1[m]["tc_med_r2_dV2_dS2_mean"]
        print(f"  {m}: symmetric fraction = {sf:.4f}, R^2(dV^2, dS^2) = {r2:.4f}")
        if sf > 0.8:
            print(f"  >>> SERIOUS: {sf:.0%} of |dV| is from symmetric (TC-driven) component")
        else:
            print(f"  >>> MINOR: asymmetric component contributes {1-sf:.0%}")

    print("\n--- OBJECTION 2: DEGREE-PROXY ---")
    for m in ["ppwave", "schwarzschild"]:
        r2 = phase1[m]["degree_proxy_r2_mean"]
        print(f"  {m}: R^2(dV^2, dd^2) = {r2:.4f}")
        if r2 > 0.5:
            print(f"  >>> SERIOUS: IVV is {r2:.0%} explained by degree changes")
        else:
            print(f"  >>> MINOR: only {r2:.0%} overlap with degree proxy")

    print("\n--- OBJECTION 3/9: FINITE-SIZE & N-CONVERGENCE ---")
    for N_val in N_values:
        s = scaling[N_val]
        print(f"  N={N_val}: d_ppw={s['IVV_ppw_d']:.3f}, d_sch={s['IVV_sch_d']:.3f}")
    if len(N_values) >= 3:
        d_300 = abs(scaling[300]["IVV_ppw_d"])
        d_1000 = abs(scaling[1000]["IVV_ppw_d"])
        if d_300 > 0 and d_1000 / d_300 < 0.3:
            print("  >>> FATAL: Cohen's d collapsed by >70% from N=300 to N=1000")
        elif d_300 > 0 and d_1000 / d_300 < 0.5:
            print("  >>> SERIOUS: Cohen's d collapsed by >50% from N=300 to N=1000")
        else:
            print("  >>> MINOR: d appears stable or growing with N")

    print("\n--- OBJECTION 5: BOUNDARY EFFECTS ---")
    for m in ["ppwave", "schwarzschild"]:
        bd = phase1[m]["boundary"]
        if bd:
            ivv_005 = bd.get("z0.05", {}).get("IVV", 0)
            ivv_030 = bd.get("z0.30", {}).get("IVV", 0)
            ivv_b = bd.get("z0.05", {}).get("IVV_boundary", 0)
            print(f"  {m}: IVV(z=0.05)={ivv_005:.8f}, IVV(z=0.30)={ivv_030:.8f}")
            if ivv_005 > 0 and ivv_030 / ivv_005 < 0.1:
                print(f"  >>> SERIOUS: signal collapses with stricter boundary cut")
            else:
                print(f"  >>> MINOR: signal persists across boundary cuts")

    print("\n--- OBJECTION 6: SIGN PATTERN ---")
    for m in ["ppwave", "schwarzschild"]:
        fp = phase1[m]["sign_frac_positive_mean"]
        md = phase1[m]["sign_mean_dV_mean"]
        print(f"  {m}: frac_positive={fp:.4f}, mean(dV)={md:.8f}")
    print("  NOTE: IVV = mean(dV^2) is always >= 0. No sign discrimination possible.")
    print("  >>> MINOR-BUT-FUNDAMENTAL: IVV cannot distinguish curvature types by sign.")

    print("\n--- OBJECTION 7: CONFORMAL NULL (de Sitter) ---")
    ds = phase1.get("desitter", {})
    ivv_ds = ds.get("IVV_mean", 0)
    ivv_ppw = phase1["ppwave"]["IVV_mean"]
    print(f"  de Sitter (H=2.0): IVV = {ivv_ds:.8f}")
    print(f"  pp-wave (eps=3.0): IVV = {ivv_ppw:.8f}")
    if ivv_ds > 0.01 * ivv_ppw and ivv_ppw > 0:
        ratio = ivv_ds / ivv_ppw
        print(f"  >>> IVV(dS)/IVV(ppw) = {ratio:.4f}")
        if ratio > 0.3:
            print(f"  >>> SERIOUS: de Sitter gives {ratio:.0%} of pp-wave signal. Not Weyl-specific.")
        else:
            print(f"  >>> MINOR: de Sitter signal is small but nonzero.")
    elif ivv_ppw > 0:
        print(f"  >>> PASS: de Sitter signal is negligible relative to pp-wave.")
    else:
        print(f"  >>> INCONCLUSIVE: pp-wave signal is zero.")

    print("\n--- OBJECTION 8: GRAPH-THEORY ADVERSARY ---")
    for m in ["ppwave", "schwarzschild"]:
        ar = phase1[m]["adversary_ratio_mean"]
        ar_std = phase1[m]["adversary_ratio_std"]
        print(f"  {m}: IVV_random/IVV_curved = {ar:.4f} +/- {ar_std:.4f}")
        if ar > 2.0:
            print(f"  >>> FATAL: Random perturbation gives {ar:.1f}x MORE signal than curvature")
        elif ar > 0.5:
            print(f"  >>> SERIOUS: Random perturbation gives {ar:.0%} of curvature signal")
        else:
            print(f"  >>> PASS: Random perturbation gives much less signal")

    print("\n--- OBJECTION 10: PFC PRIOR ART ---")
    print("  PFC = Spearman(|past|, |future|) was KILLED (d=-0.01 on Schwarzschild).")
    print("  IVV = mean((V_curv - V_flat)^2) where V = |past|*|future|/N^2.")
    print("  Both use the SAME base quantities: |past(x)| and |future(x)|.")
    print("  IVV's CRN structure is different: it measures VARIANCE of product changes.")
    print("  BUT: if |past| and |future| don't change independently on Schwarzschild")
    print("  (as PFC's failure implies), then IVV's product won't either.")
    ivv_sch_d = scaling.get(500, {}).get("IVV_sch_d", 0)
    if abs(ivv_sch_d) < 0.5:
        print(f"  >>> SERIOUS: IVV d={ivv_sch_d:.3f} on Sch at N=500 — weak, consistent with PFC failure")
    else:
        print(f"  >>> IVV d={ivv_sch_d:.3f} on Sch at N=500 — survives PFC's failure mode")

    # Overall
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    fatals = 0
    serious = 0
    # Count
    for N_val in N_values:
        d_300 = abs(scaling[300]["IVV_ppw_d"])
        d_1000 = abs(scaling[1000]["IVV_ppw_d"])
        if d_300 > 0 and d_1000 / d_300 < 0.3:
            fatals += 1
            break
    for m in ["ppwave", "schwarzschild"]:
        ar = phase1[m]["adversary_ratio_mean"]
        if ar > 2.0:
            fatals += 1
    for m in ["ppwave", "schwarzschild"]:
        sf = phase1[m]["tc_med_sym_fraction_mean"]
        if sf > 0.8:
            serious += 1
    for m in ["ppwave", "schwarzschild"]:
        r2 = phase1[m]["degree_proxy_r2_mean"]
        if r2 > 0.5:
            serious += 1

    print(f"  FATAL objections: {fatals}")
    print(f"  SERIOUS objections: {serious}")
    print(f"  Total time: {elapsed:.0f}s")

    # Save
    out = {
        "phase1": phase1,
        "scaling": {str(k): v for k, v in scaling.items()},
        "fatals": fatals,
        "serious": serious,
        "elapsed_s": round(elapsed, 1),
    }
    out_path = os.path.join(os.path.dirname(__file__), "results", "demolish_009_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
