#!/usr/bin/env python3
"""ALL killer tests for ratio 0.51 diagnosis.

Test 1: STF orbit-type test — jet(ppw_std) vs jet(ppw_m=0) at same E²
         Also: exact(ppw) vs jet(ppw) to quantify predicate bias
Test 4: A_align without strata (global Cov(X², δY²))
Test 5: A_align with τ̂ × ρ̂ only (no depth terciles)
Test 6: ppw at matched q=0.40 (same as Sch q_W)
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_ppwave_canonical, riemann_schwarzschild_local,
    riemann_vacuum_from_E, set_riemann_component
)
from schwarzschild_exact_local_tools import (
    map_rnc_to_schwarzschild_expmap,
    schwarzschild_exact_midpoint_preds_from_mapped
)

N = 10000
ZETA = 0.15
T = 1.0
M_SEEDS = 20
EPS_STANDARD = 3.0
M_SCH = 0.05
R0 = 0.50
EPS_LOW = 0.40
E2_PPW = EPS_STANDARD**2 / 2.0
E2_LOW = EPS_LOW**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2


def make_strata_full(pts, parents0):
    """5τ × 3ρ × 3depth = 45 strata"""
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


def make_strata_no_depth(pts):
    """5τ × 3ρ only (no depth) = 15 strata"""
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    return tau_bin * 3 + rho_bin


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


def compute_global_cov(Y0, delta, mask):
    """Global Cov(X², δY²) without any strata"""
    X = Y0[mask] - np.mean(Y0[mask])
    X2 = X ** 2
    dY2 = delta[mask] ** 2
    return float(np.mean(X2 * dY2) - np.mean(X2) * np.mean(dY2))


def riemann_ppwave_m0(eps):
    """pp-wave rotated to m=0: E_ij = (eps/2)·diag(-1,-1,+2) instead of diag(-1,+1,0).
    Same E² = 3eps²/2... wait, standard ppw E² = eps²/2.
    For m=0 diag(-1,-1,2): E² = 1+1+4 = 6 × (eps/2)² = 3eps²/2. DIFFERENT E²!

    To match E²: need diag(-a,-a,2a) with 6a² = eps²/2 → a = eps/(2√3).
    Or simpler: just use same eigenvalues as Sch: diag(-1,-1,+2) scaled to same E² as standard ppw.

    Standard ppw: E = (eps/2)diag(-1,+1,0). E² = (eps/2)²(1+1+0) = eps²/2.
    m=0 with same E²: E = c·diag(-1,-1,+2). E² = c²(1+1+4) = 6c². Set 6c² = eps²/2 → c = eps/√12.
    """
    c = eps / np.sqrt(12.0)
    E = np.diag([-c, -c, 2.0*c]).astype(np.float64)
    return riemann_vacuum_from_E(E)


R_SCH = riemann_schwarzschild_local(M_SCH, R0)


if __name__ == "__main__":
    print("=== ALL KILLER TESTS ===", flush=True)
    print(f"N={N}, T={T}, M={M_SEEDS}", flush=True)
    t_total = time.time()
    out = {}

    # ═══════════════════════════════════════════════════════
    # Precompute shared flat Hasse for each seed
    # ═══════════════════════════════════════════════════════
    all_data = []
    for si in range(M_SEEDS):
        seed = 3100000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T, ZETA)
        strata_full = make_strata_full(pts, par0)
        strata_nodepth = make_strata_no_depth(pts)

        # ppw standard (m=±2, eps=3) — EXACT predicate
        parP, chP = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_STANDARD))
        YP = Y_from_graph(parP, chP)
        delta_ppw = YP - Y0

        # ppw standard (m=±2, eps=3) — JET with vacuum_from_E (SAME construction as m=0!)
        # CRITICAL: must use vacuum_from_E, NOT riemann_ppwave_canonical,
        # because ppwave_canonical has wave-direction cross-components R_{0i3j}
        # that vacuum_from_E doesn't have. Using same construction = pure orbit-type test.
        E_ppw_std = (EPS_STANDARD / 2.0) * np.diag([-1.0, 1.0, 0.0])
        R_ppw_std = riemann_vacuum_from_E(E_ppw_std)
        parJ, chJ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_ppw_std))
        YJ = Y_from_graph(parJ, chJ)
        delta_ppw_jet = YJ - Y0

        # ppw m=0 (same E², different orbit I₃) — JET predicate
        R_m0 = riemann_ppwave_m0(EPS_STANDARD)
        parM, chM = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_m0))
        YM = Y_from_graph(parM, chM)
        delta_m0 = YM - Y0

        # ppw at low eps=0.40 (matched q to Sch q_W=0.40) — EXACT predicate
        parL, chL = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_LOW))
        YL = Y_from_graph(parL, chL)
        delta_low = YL - Y0

        # Sch quadratic jet
        parQ, chQ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
        YQ = Y_from_graph(parQ, chQ)
        delta_sch_q = YQ - Y0

        # Sch exact-expmap midpoint
        mapped = map_rnc_to_schwarzschild_expmap(pts, M_SCH, R0)
        parX, chX = build_hasse_from_predicate(
            pts, lambda P, i: schwarzschild_exact_midpoint_preds_from_mapped(mapped, i, M_SCH))
        YX = Y_from_graph(parX, chX)
        delta_sch_x = YX - Y0

        all_data.append({
            "mask": mask, "Y0": Y0,
            "strata_full": strata_full, "strata_nodepth": strata_nodepth,
            "delta_ppw": delta_ppw, "delta_ppw_jet": delta_ppw_jet,
            "delta_m0": delta_m0, "delta_low": delta_low,
            "delta_sch_q": delta_sch_q, "delta_sch_x": delta_sch_x,
        })

        if (si + 1) % 5 == 0:
            print(f"  Precomputed {si+1}/{M_SEEDS} ({time.time()-t_total:.0f}s)", flush=True)

    # ═══════════════════════════════════════════════════════
    # TEST 1: STF ORBIT-TYPE TEST
    # Both jet builds use vacuum_from_E (same construction, ONLY E_ij differs).
    # Three sub-comparisons:
    #   1a. jet(vac_E_std) vs jet(vac_E_m0) → PURE orbit-type (I₃) effect
    #   1b. ppw_exact vs jet(vac_E_std)     → total bias (jet+static vs exact+wave)
    #   1c. ppw_exact vs jet(vac_E_m0)      → combined
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST 1: STF ORBIT-TYPE (I₃) TEST ===", flush=True)
    E2_m0 = EPS_STANDARD**2 / 2.0  # same E² by construction

    a_ppw_exact = [compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], d["strata_full"]) for d in all_data]
    a_ppw_jet = [compute_A_align(d["Y0"], d["delta_ppw_jet"], d["mask"], d["strata_full"]) for d in all_data]
    a_m0 = [compute_A_align(d["Y0"], d["delta_m0"], d["mask"], d["strata_full"]) for d in all_data]

    AE_ppw_exact = np.mean(a_ppw_exact) / E2_PPW
    AE_ppw_jet = np.mean(a_ppw_jet) / E2_PPW
    AE_m0 = np.mean(a_m0) / E2_m0

    # 1a: PURE orbit-type test (same jet predicate, different I₃)
    ratio_orbit = AE_m0 / AE_ppw_jet if abs(AE_ppw_jet) > 1e-15 else 0
    # 1b: Predicate quality test (same geometry, different predicate)
    ratio_pred = AE_ppw_jet / AE_ppw_exact if abs(AE_ppw_exact) > 1e-15 else 0
    # 1c: Combined (old comparison)
    ratio_combined = AE_m0 / AE_ppw_exact if abs(AE_ppw_exact) > 1e-15 else 0

    print(f"  ppw exact (m=±2):  A_E = {AE_ppw_exact:.6f}", flush=True)
    print(f"  ppw jet   (m=±2):  A_E = {AE_ppw_jet:.6f}", flush=True)
    print(f"  ppw jet   (m=0):   A_E = {AE_m0:.6f}", flush=True)
    print(f"", flush=True)
    print(f"  1a. ORBIT-TYPE (jet m=0 / jet m=±2) = {ratio_orbit:.3f}", flush=True)
    print(f"  1b. TOTAL BIAS (jet vac_E / exact ppw) = {ratio_pred:.3f}", flush=True)
    print(f"  1c. COMBINED   (jet m=0 / exact m=±2) = {ratio_combined:.3f}", flush=True)
    print(f"", flush=True)

    if abs(ratio_orbit - 1.0) < 0.15:
        print(f"  → 1a: orbit-type (I₃) IRRELEVANT at current T/N (ratio≈1)", flush=True)
    else:
        print(f"  → 1a: orbit-type (I₃) MATTERS! ({ratio_orbit:.3f} ≠ 1.0)", flush=True)

    if abs(ratio_pred - 1.0) < 0.10:
        print(f"  → 1b: jet ≈ exact for ppw (predicate bias small)", flush=True)
    else:
        print(f"  → 1b: jet ≠ exact for ppw! predicate bias = {abs(1-ratio_pred)*100:.0f}%", flush=True)

    out["test1_orbit"] = {
        "AE_ppw_exact": AE_ppw_exact, "AE_ppw_jet": AE_ppw_jet, "AE_m0_jet": AE_m0,
        "ratio_orbit_pure": ratio_orbit,
        "ratio_predicate": ratio_pred,
        "ratio_combined": ratio_combined,
        "I3_std": 0.0,  # tr(E_std³) = (eps/2)³·((-1)³+1³+0) = 0
        "I3_m0": float((EPS_STANDARD/np.sqrt(12))**3 * ((-1)**3 + (-1)**3 + 2**3)),  # 6c³
    }

    # For backward compat in summary
    AE_ppw = AE_ppw_exact
    ratio_m0 = ratio_orbit

    # ═══════════════════════════════════════════════════════
    # TEST 4: A_align without strata (global Cov)
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST 4: GLOBAL COV (no strata) ===", flush=True)
    g_ppw = [compute_global_cov(d["Y0"], d["delta_ppw"], d["mask"]) for d in all_data]
    g_sch_q = [compute_global_cov(d["Y0"], d["delta_sch_q"], d["mask"]) for d in all_data]
    g_sch_x = [compute_global_cov(d["Y0"], d["delta_sch_x"], d["mask"]) for d in all_data]
    GE_ppw = np.mean(g_ppw) / E2_PPW
    GE_sch_q = np.mean(g_sch_q) / E2_SCH
    GE_sch_x = np.mean(g_sch_x) / E2_SCH
    r_global_q = GE_sch_q / GE_ppw if abs(GE_ppw) > 1e-15 else 0
    r_global_x = GE_sch_x / GE_ppw if abs(GE_ppw) > 1e-15 else 0
    print(f"  ppw:      global_A_E = {GE_ppw:.6f}", flush=True)
    print(f"  Sch quad: global_A_E = {GE_sch_q:.6f} (ratio = {r_global_q:.3f})", flush=True)
    print(f"  Sch exp:  global_A_E = {GE_sch_x:.6f} (ratio = {r_global_x:.3f})", flush=True)
    out["test4_global"] = {"ppw": GE_ppw, "sch_q": GE_sch_q, "sch_x": GE_sch_x,
                           "ratio_q": r_global_q, "ratio_x": r_global_x}

    # ═══════════════════════════════════════════════════════
    # TEST 5: A_align with τ̂ × ρ̂ only (no depth)
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST 5: NO-DEPTH STRATA (τ̂ × ρ̂ only) ===", flush=True)
    nd_ppw = [compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], d["strata_nodepth"]) for d in all_data]
    nd_sch_q = [compute_A_align(d["Y0"], d["delta_sch_q"], d["mask"], d["strata_nodepth"]) for d in all_data]
    nd_sch_x = [compute_A_align(d["Y0"], d["delta_sch_x"], d["mask"], d["strata_nodepth"]) for d in all_data]
    NE_ppw = np.mean(nd_ppw) / E2_PPW
    NE_sch_q = np.mean(nd_sch_q) / E2_SCH
    NE_sch_x = np.mean(nd_sch_x) / E2_SCH
    r_nd_q = NE_sch_q / NE_ppw if abs(NE_ppw) > 1e-15 else 0
    r_nd_x = NE_sch_x / NE_ppw if abs(NE_ppw) > 1e-15 else 0
    print(f"  ppw:      nodepth_A_E = {NE_ppw:.6f}", flush=True)
    print(f"  Sch quad: nodepth_A_E = {NE_sch_q:.6f} (ratio = {r_nd_q:.3f})", flush=True)
    print(f"  Sch exp:  nodepth_A_E = {NE_sch_x:.6f} (ratio = {r_nd_x:.3f})", flush=True)
    out["test5_nodepth"] = {"ppw": NE_ppw, "sch_q": NE_sch_q, "sch_x": NE_sch_x,
                            "ratio_q": r_nd_q, "ratio_x": r_nd_x}

    # ═══════════════════════════════════════════════════════
    # TEST 6: ppw at matched q=0.40
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST 6: PPW at matched q=0.40 ===", flush=True)
    a_low = [compute_A_align(d["Y0"], d["delta_low"], d["mask"], d["strata_full"]) for d in all_data]
    AE_low = np.mean(a_low) / E2_LOW
    ratio_matched = AE_low / AE_ppw if abs(AE_ppw) > 1e-15 else 0
    print(f"  ppw q=3.0 (eps=3):   A_E = {AE_ppw:.6f}", flush=True)
    print(f"  ppw q=0.40 (eps=0.4): A_E = {AE_low:.6f}", flush=True)
    print(f"  Ratio low_q/high_q = {ratio_matched:.3f}", flush=True)
    if ratio_matched < 0.7:
        print(f"  → q⁴ contamination SIGNIFICANT: ppw A_E inflated at high q!", flush=True)
    else:
        print(f"  → q⁴ contamination SMALL: ppw A_E stable across q", flush=True)

    # Compare ppw at matched q with Sch (SAME strata-based A_align, not global Cov)
    a_sch_x_strata = [compute_A_align(d["Y0"], d["delta_sch_x"], d["mask"], d["strata_full"]) for d in all_data]
    AE_sch_x_strata = np.mean(a_sch_x_strata) / E2_SCH
    r_sch_ppw_matched_q = AE_sch_x_strata / AE_low if abs(AE_low) > 1e-15 else 0
    print(f"  Sch strata A_E = {AE_sch_x_strata:.6f}", flush=True)
    print(f"  → KEY: Sch/ppw ratio AT SAME q≈0.40 = {r_sch_ppw_matched_q:.3f}", flush=True)
    # baseline_ratio computed below in SUMMARY, use Sch/ppw_exact as reference
    _ref_ratio = AE_sch_x_strata / AE_ppw if abs(AE_ppw) > 1e-15 else 0.508
    if abs(r_sch_ppw_matched_q - _ref_ratio) < 0.15:
        print(f"  → q-mismatch NOT the cause (ratio ≈ baseline)", flush=True)
    elif r_sch_ppw_matched_q > 0.7:
        print(f"  → q-mismatch WAS inflating ppw! Matched ratio IMPROVED!", flush=True)
    else:
        print(f"  → Ratio changed but unclear direction", flush=True)

    out["test6_matched_q"] = {"AE_ppw_high": AE_ppw, "AE_ppw_low": AE_low,
                              "AE_sch_strata": AE_sch_x_strata,
                              "ratio_low_high": ratio_matched,
                              "ratio_sch_ppw_matched_q": r_sch_ppw_matched_q}

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    # Baseline: strata A_E ratio Sch/ppw
    baseline_ratio = AE_sch_x_strata / AE_ppw if abs(AE_ppw) > 1e-15 else 0
    print(f"\n{'='*60}", flush=True)
    print(f"=== SUMMARY ===", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Baseline strata Sch/ppw:       {baseline_ratio:.3f} (this run, M={M_SEEDS})", flush=True)
    print(f"  Test 1a (orbit I₃, jet/jet):   {ratio_orbit:.3f}", flush=True)
    print(f"  Test 1b (predicate, jet/exact): {ratio_pred:.3f}", flush=True)
    print(f"  Test 4 (global Cov Sch/ppw):    {r_global_x:.3f}", flush=True)
    print(f"  Test 5 (no-depth Sch/ppw):      {r_nd_x:.3f}", flush=True)
    print(f"  Test 6 (ppw low_q/high_q):      {ratio_matched:.3f}", flush=True)
    print(f"  Test 6 (Sch/ppw at q≈0.40):     {r_sch_ppw_matched_q:.3f}", flush=True)
    print(f"", flush=True)
    print(f"  DIAGNOSIS:", flush=True)
    if abs(ratio_orbit - 1.0) > 0.15:
        print(f"  → Test 1a: orbit (I₃) MATTERS (jet m0/jet std={ratio_orbit:.3f})", flush=True)
    else:
        print(f"  → Test 1a: orbit irrelevant (jet m0/jet std={ratio_orbit:.3f}≈1)", flush=True)
    if abs(ratio_pred - 1.0) > 0.10:
        print(f"  → Test 1b: predicate bias SIGNIFICANT (jet/exact={ratio_pred:.3f})", flush=True)
    else:
        print(f"  → Test 1b: predicate bias small (jet/exact={ratio_pred:.3f}≈1)", flush=True)
    if abs(r_global_x - baseline_ratio) > 0.15:
        print(f"  → Test 4: strata affect ratio ({r_global_x:.3f} vs {baseline_ratio:.3f})", flush=True)
    else:
        print(f"  → Test 4: strata don't change ratio much", flush=True)
    if abs(r_nd_x - baseline_ratio) > 0.10:
        print(f"  → Test 5: depth terciles affect ratio ({r_nd_x:.3f} vs {baseline_ratio:.3f})", flush=True)
    else:
        print(f"  → Test 5: depth terciles don't matter much", flush=True)
    if ratio_matched < 0.7:
        print(f"  → Test 6: q-contamination significant (low/high={ratio_matched:.3f})", flush=True)
    else:
        print(f"  → Test 6: q-contamination small (low/high={ratio_matched:.3f})", flush=True)

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out["baseline_strata_ratio"] = baseline_ratio
    out["params"] = {"N": N, "T": T, "M_SEEDS": M_SEEDS, "EPS_STANDARD": EPS_STANDARD,
                     "EPS_LOW": EPS_LOW, "M_SCH": M_SCH, "R0": R0, "ZETA": ZETA}

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "killer_tests.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to killer_tests.json", flush=True)
