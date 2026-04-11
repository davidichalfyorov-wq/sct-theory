"""
FND-1 Critical Null Model Test: Is link_fiedler a curvature detector or a TC artifact?

Three null models applied to FLAT Minkowski causal sets (N=2000):
  A) Random deletion of causal pairs (TC decreases)
  B) Random addition of non-causal pairs (TC increases)
  C) Random permutation (TC constant): delete some, add same number

If link_fiedler drops in all three -> topology artifact, not curvature
If only in A -> connectivity artifact (fewer links)
If in none -> curvature-specific signal

Reference from d=4 experiment:
  cos*cosh eps=0.5: TC drops 28%, link_fiedler drops 0.90->0.64 (29%)
  quadrupole eps=40: TC rises 24%, link_fiedler drops 0.92->0.43 (53%)
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat,
    causal_matrix_4d,
    compute_layers_4d,
    build_link_graph,
    link_spectral_embedding,
)

# ── Parameters ──────────────────────────────────────────────────────────
N = 2000
T_DIAMOND = 1.0
N_SEEDS = 20
K_EMBED = 10          # embedding dimension for link_spectral_embedding

P_DELETE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
P_ADD    = [0.05, 0.10, 0.15, 0.20, 0.25]
P_PERMUTE = [0.10]    # keep TC constant, permute 10%


def get_link_fiedler(C):
    """Compute link_fiedler (second smallest eigenvalue of link Laplacian)."""
    C_sp = sp.csr_matrix(C)
    n_matrix = (C_sp @ C_sp).toarray()
    A_link = build_link_graph(C, n_matrix)
    _, lams = link_spectral_embedding(A_link, 2, C.shape[0])
    if len(lams) >= 1:
        return float(lams[0])   # first non-zero eigenvalue = Fiedler value
    return 0.0


def get_link_fiedler_and_stats(C):
    """Return (fiedler, total_causal, n_links)."""
    TC = float(np.sum(C))
    C_sp = sp.csr_matrix(C)
    n_matrix = (C_sp @ C_sp).toarray()
    A_link = build_link_graph(C, n_matrix)
    n_links = int(A_link.nnz // 2)  # undirected
    _, lams = link_spectral_embedding(A_link, 2, C.shape[0])
    fiedler = float(lams[0]) if len(lams) >= 1 else 0.0
    return fiedler, TC, n_links


def null_model_A(C, p, rng):
    """Delete p fraction of causal pairs (TC decreases)."""
    C_mod = C.copy()
    causal_i, causal_j = np.where(C > 0)
    n_pairs = len(causal_i)
    n_delete = int(p * n_pairs)
    idx = rng.choice(n_pairs, size=n_delete, replace=False)
    C_mod[causal_i[idx], causal_j[idx]] = 0.0
    return C_mod


def null_model_B(C, p, rng):
    """Add p fraction of non-causal pairs as causal (TC increases)."""
    C_mod = C.copy()
    n_causal = int(np.sum(C))
    n_add = int(p * n_causal)

    # Find non-causal future-directed pairs (upper triangle where C=0 and i<j in time)
    N_pts = C.shape[0]
    noncausal_i, noncausal_j = np.where((C == 0) & (np.triu(np.ones((N_pts, N_pts)), k=1) > 0))
    if len(noncausal_i) == 0:
        return C_mod
    n_add = min(n_add, len(noncausal_i))
    idx = rng.choice(len(noncausal_i), size=n_add, replace=False)
    C_mod[noncausal_i[idx], noncausal_j[idx]] = 1.0
    return C_mod


def null_model_C(C, p, rng):
    """Permute p fraction: delete some causal pairs, add same number of non-causal. TC constant."""
    C_mod = C.copy()
    N_pts = C.shape[0]

    causal_i, causal_j = np.where(C > 0)
    n_causal = len(causal_i)
    n_swap = int(p * n_causal)

    # Delete n_swap random causal pairs
    del_idx = rng.choice(n_causal, size=n_swap, replace=False)
    C_mod[causal_i[del_idx], causal_j[del_idx]] = 0.0

    # Add n_swap random non-causal pairs (future-directed)
    noncausal_i, noncausal_j = np.where(
        (C_mod == 0) & (np.triu(np.ones((N_pts, N_pts)), k=1) > 0)
    )
    if len(noncausal_i) < n_swap:
        n_swap = len(noncausal_i)
    add_idx = rng.choice(len(noncausal_i), size=n_swap, replace=False)
    C_mod[noncausal_i[add_idx], noncausal_j[add_idx]] = 1.0
    return C_mod


def run_experiment():
    """Run all null models across seeds."""
    print("=" * 75)
    print("FND-1 CRITICAL NULL MODEL TEST")
    print("Is link_fiedler curvature-specific or a TC/topology artifact?")
    print("=" * 75)
    print(f"\nN={N}, T={T_DIAMOND}, seeds={N_SEEDS}")
    print(f"Delete fractions: {P_DELETE}")
    print(f"Add fractions:    {P_ADD}")
    print(f"Permute fraction: {P_PERMUTE}")
    print()

    # Storage
    baseline_fiedler = []
    baseline_TC = []
    baseline_links = []

    results_A = {p: {"fiedler": [], "TC": [], "links": []} for p in P_DELETE}
    results_B = {p: {"fiedler": [], "TC": [], "links": []} for p in P_ADD}
    results_C = {p: {"fiedler": [], "TC": [], "links": []} for p in P_PERMUTE}

    t0 = time.time()

    for seed_idx in range(N_SEEDS):
        seed = 1000 + seed_idx
        rng = np.random.default_rng(seed)
        t_seed = time.time()

        # Sprinkle flat Minkowski
        pts = sprinkle_4d_flat(N, T_DIAMOND, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")

        # Baseline
        f0, tc0, nl0 = get_link_fiedler_and_stats(C)
        baseline_fiedler.append(f0)
        baseline_TC.append(tc0)
        baseline_links.append(nl0)

        # Null Model A: delete causal pairs
        for p in P_DELETE:
            rng_a = np.random.default_rng(seed * 100 + int(p * 1000))
            C_a = null_model_A(C, p, rng_a)
            fa, tca, nla = get_link_fiedler_and_stats(C_a)
            results_A[p]["fiedler"].append(fa)
            results_A[p]["TC"].append(tca)
            results_A[p]["links"].append(nla)

        # Null Model B: add non-causal pairs
        for p in P_ADD:
            rng_b = np.random.default_rng(seed * 200 + int(p * 1000))
            C_b = null_model_B(C, p, rng_b)
            fb, tcb, nlb = get_link_fiedler_and_stats(C_b)
            results_B[p]["fiedler"].append(fb)
            results_B[p]["TC"].append(tcb)
            results_B[p]["links"].append(nlb)

        # Null Model C: permute (TC constant)
        for p in P_PERMUTE:
            rng_c = np.random.default_rng(seed * 300 + int(p * 1000))
            C_c = null_model_C(C, p, rng_c)
            fc, tcc, nlc = get_link_fiedler_and_stats(C_c)
            results_C[p]["fiedler"].append(fc)
            results_C[p]["TC"].append(tcc)
            results_C[p]["links"].append(nlc)

        elapsed = time.time() - t_seed
        print(f"  Seed {seed_idx+1:2d}/{N_SEEDS}: baseline fiedler={f0:.4f}, "
              f"TC={tc0:.0f}, links={nl0}, dt={elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")

    # ── Report ──────────────────────────────────────────────────────────
    def mean_sem(arr):
        arr = np.array(arr)
        return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))

    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)

    bl_f, bl_f_sem = mean_sem(baseline_fiedler)
    bl_tc, bl_tc_sem = mean_sem(baseline_TC)
    bl_nl, bl_nl_sem = mean_sem(baseline_links)
    print(f"\nBASELINE (flat Minkowski, N={N}):")
    print(f"  link_fiedler = {bl_f:.4f} +/- {bl_f_sem:.4f}")
    print(f"  TC           = {bl_tc:.0f} +/- {bl_tc_sem:.0f}")
    print(f"  N_links      = {bl_nl:.0f} +/- {bl_nl_sem:.0f}")

    # ── Table A: Deletion ───────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("NULL MODEL A: Random deletion of causal pairs (TC decreases)")
    print(f"{'─'*75}")
    print(f"{'p':>6s} | {'fiedler':>12s} | {'dTC%':>8s} | {'dFiedler%':>10s} | {'dLinks%':>10s}")
    print(f"{'─'*6}-+-{'─'*12}-+-{'─'*8}-+-{'─'*10}-+-{'─'*10}")

    for p in P_DELETE:
        f_m, f_s = mean_sem(results_A[p]["fiedler"])
        tc_m, _ = mean_sem(results_A[p]["TC"])
        nl_m, _ = mean_sem(results_A[p]["links"])
        dtc = (tc_m - bl_tc) / bl_tc * 100
        df = (f_m - bl_f) / bl_f * 100
        dnl = (nl_m - bl_nl) / bl_nl * 100
        print(f"{p:6.0%} | {f_m:8.4f}+/-{f_s:.4f} | {dtc:+7.1f}% | {df:+9.1f}% | {dnl:+9.1f}%")

    # ── Table B: Addition ───────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("NULL MODEL B: Random addition of non-causal pairs (TC increases)")
    print(f"{'─'*75}")
    print(f"{'p':>6s} | {'fiedler':>12s} | {'dTC%':>8s} | {'dFiedler%':>10s} | {'dLinks%':>10s}")
    print(f"{'─'*6}-+-{'─'*12}-+-{'─'*8}-+-{'─'*10}-+-{'─'*10}")

    for p in P_ADD:
        f_m, f_s = mean_sem(results_B[p]["fiedler"])
        tc_m, _ = mean_sem(results_B[p]["TC"])
        nl_m, _ = mean_sem(results_B[p]["links"])
        dtc = (tc_m - bl_tc) / bl_tc * 100
        df = (f_m - bl_f) / bl_f * 100
        dnl = (nl_m - bl_nl) / bl_nl * 100
        print(f"{p:6.0%} | {f_m:8.4f}+/-{f_s:.4f} | {dtc:+7.1f}% | {df:+9.1f}% | {dnl:+9.1f}%")

    # ── Table C: Permutation ────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("NULL MODEL C: Random permutation (TC approximately constant)")
    print(f"{'─'*75}")
    print(f"{'p':>6s} | {'fiedler':>12s} | {'dTC%':>8s} | {'dFiedler%':>10s} | {'dLinks%':>10s}")
    print(f"{'─'*6}-+-{'─'*12}-+-{'─'*8}-+-{'─'*10}-+-{'─'*10}")

    for p in P_PERMUTE:
        f_m, f_s = mean_sem(results_C[p]["fiedler"])
        tc_m, _ = mean_sem(results_C[p]["TC"])
        nl_m, _ = mean_sem(results_C[p]["links"])
        dtc = (tc_m - bl_tc) / bl_tc * 100
        df = (f_m - bl_f) / bl_f * 100
        dnl = (nl_m - bl_nl) / bl_nl * 100
        print(f"{p:6.0%} | {f_m:8.4f}+/-{f_s:.4f} | {dtc:+7.1f}% | {df:+9.1f}% | {dnl:+9.1f}%")

    # ── Comparison with curvature experiment ────────────────────────────
    print(f"\n{'='*75}")
    print("COMPARISON WITH CURVATURE EXPERIMENT")
    print(f"{'='*75}")
    print(f"\nReference (from fnd1_4d_experiment.py, N=2000):")
    print(f"  cos*cosh eps=0.5: TC drops ~28%, fiedler drops 0.90->0.64 (29%)")
    print(f"  quadrupole eps=40: TC rises ~24%, fiedler drops 0.92->0.43 (53%)")

    # Find best-matching null model for each curvature case
    print(f"\nBest-matching null model comparisons:")

    # For cos*cosh (28% TC drop) -> compare with Model A at 25-30%
    a25_f, _ = mean_sem(results_A[0.25]["fiedler"])
    a30_f, _ = mean_sem(results_A[0.30]["fiedler"])
    a25_tc, _ = mean_sem(results_A[0.25]["TC"])
    a30_tc, _ = mean_sem(results_A[0.30]["TC"])
    dtc_a25 = (a25_tc - bl_tc) / bl_tc * 100
    dtc_a30 = (a30_tc - bl_tc) / bl_tc * 100
    print(f"  cos*cosh (28% TC drop):")
    print(f"    Model A p=25% -> TC change: {dtc_a25:+.1f}%, fiedler: {a25_f:.4f} ({(a25_f-bl_f)/bl_f*100:+.1f}%)")
    print(f"    Model A p=30% -> TC change: {dtc_a30:+.1f}%, fiedler: {a30_f:.4f} ({(a30_f-bl_f)/bl_f*100:+.1f}%)")
    print(f"    Curvature expt: fiedler ~0.64 ({-29:+.1f}%)")

    # For quadrupole (24% TC rise) -> compare with Model B at 25%
    b25_f, _ = mean_sem(results_B[0.25]["fiedler"])
    b25_tc, _ = mean_sem(results_B[0.25]["TC"])
    dtc_b25 = (b25_tc - bl_tc) / bl_tc * 100
    print(f"\n  quadrupole (24% TC rise):")
    print(f"    Model B p=25% -> TC change: {dtc_b25:+.1f}%, fiedler: {b25_f:.4f} ({(b25_f-bl_f)/bl_f*100:+.1f}%)")
    print(f"    Curvature expt: fiedler ~0.43 ({-53:+.1f}%)")

    # For TC-constant -> compare with Model C at 10%
    c10_f, _ = mean_sem(results_C[0.10]["fiedler"])
    c10_tc, _ = mean_sem(results_C[0.10]["TC"])
    dtc_c10 = (c10_tc - bl_tc) / bl_tc * 100
    print(f"\n  TC-constant control:")
    print(f"    Model C p=10% -> TC change: {dtc_c10:+.1f}%, fiedler: {c10_f:.4f} ({(c10_f-bl_f)/bl_f*100:+.1f}%)")

    # ── Verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("DIAGNOSTIC VERDICT")
    print(f"{'='*75}")

    # Check: does Model A (deletion) replicate the fiedler drop?
    a30_drop = (a30_f - bl_f) / bl_f * 100
    curv_drop_cos = -29.0

    # Check: does Model B (addition) replicate the quadrupole drop?
    b25_drop = (b25_f - bl_f) / bl_f * 100
    curv_drop_quad = -53.0

    # Check: does Model C (permutation) cause any drop?
    c10_drop = (c10_f - bl_f) / bl_f * 100

    print(f"\n  Model A (30% deletion): fiedler change = {a30_drop:+.1f}%")
    print(f"    vs curvature cos*cosh:                   {curv_drop_cos:+.1f}%")
    if abs(a30_drop) > 0.5 * abs(curv_drop_cos):
        print(f"    -> REPLICATES >50% of curvature effect. TC-deletion is a confounder.")
    else:
        print(f"    -> Does NOT replicate curvature effect. Signal is not from TC change alone.")

    print(f"\n  Model B (25% addition): fiedler change = {b25_drop:+.1f}%")
    print(f"    vs curvature quadrupole:                 {curv_drop_quad:+.1f}%")
    if abs(b25_drop) > 0.5 * abs(curv_drop_quad):
        print(f"    -> REPLICATES >50% of curvature effect. TC-addition is a confounder.")
    else:
        print(f"    -> Does NOT replicate curvature effect. Signal is not from TC change alone.")

    print(f"\n  Model C (10% permutation): fiedler change = {c10_drop:+.1f}%")
    if abs(c10_drop) > 5.0:
        print(f"    -> Topology change ALONE causes significant fiedler shift.")
    else:
        print(f"    -> Topology scrambling at constant TC has minimal effect.")

    # Overall
    print(f"\n  OVERALL ASSESSMENT:")
    a_replicates = abs(a30_drop) > 0.5 * abs(curv_drop_cos)
    b_replicates = abs(b25_drop) > 0.5 * abs(curv_drop_quad)
    c_significant = abs(c10_drop) > 5.0

    if a_replicates and b_replicates:
        print(f"    link_fiedler is sensitive to ANY graph perturbation.")
        print(f"    The curvature signal is likely a TC/topology artifact.")
        print(f"    VERDICT: ARTIFACT (link_fiedler not curvature-specific)")
    elif a_replicates and not b_replicates:
        print(f"    link_fiedler drops with deletion but not addition.")
        print(f"    The cos*cosh result is confounded by TC drop.")
        print(f"    But the quadrupole result (TC RISES, fiedler DROPS) remains unexplained.")
        print(f"    VERDICT: PARTIAL ARTIFACT (cos*cosh confounded, quadrupole possibly genuine)")
    elif not a_replicates and not b_replicates:
        print(f"    Neither deletion nor addition replicates the curvature effect.")
        if c_significant:
            print(f"    But topology scrambling does affect fiedler.")
            print(f"    VERDICT: TOPOLOGY-SENSITIVE (not TC-specific, but graph-structure dependent)")
        else:
            print(f"    And topology scrambling has minimal effect.")
            print(f"    VERDICT: CURVATURE-SPECIFIC (null models fail to replicate)")
    else:
        print(f"    Mixed results. Further investigation needed.")
        print(f"    VERDICT: INCONCLUSIVE")

    # ── Save JSON ───────────────────────────────────────────────────────
    output = {
        "experiment": "fnd1_null_model_test",
        "N": N, "T": T_DIAMOND, "seeds": N_SEEDS,
        "baseline": {
            "fiedler_mean": bl_f, "fiedler_sem": bl_f_sem,
            "TC_mean": bl_tc, "links_mean": bl_nl,
        },
        "model_A": {},
        "model_B": {},
        "model_C": {},
    }
    for p in P_DELETE:
        f_m, f_s = mean_sem(results_A[p]["fiedler"])
        tc_m, _ = mean_sem(results_A[p]["TC"])
        nl_m, _ = mean_sem(results_A[p]["links"])
        output["model_A"][str(p)] = {
            "fiedler_mean": f_m, "fiedler_sem": f_s,
            "TC_mean": tc_m, "links_mean": nl_m,
            "dTC_pct": (tc_m - bl_tc) / bl_tc * 100,
            "dFiedler_pct": (f_m - bl_f) / bl_f * 100,
        }
    for p in P_ADD:
        f_m, f_s = mean_sem(results_B[p]["fiedler"])
        tc_m, _ = mean_sem(results_B[p]["TC"])
        nl_m, _ = mean_sem(results_B[p]["links"])
        output["model_B"][str(p)] = {
            "fiedler_mean": f_m, "fiedler_sem": f_s,
            "TC_mean": tc_m, "links_mean": nl_m,
            "dTC_pct": (tc_m - bl_tc) / bl_tc * 100,
            "dFiedler_pct": (f_m - bl_f) / bl_f * 100,
        }
    for p in P_PERMUTE:
        f_m, f_s = mean_sem(results_C[p]["fiedler"])
        tc_m, _ = mean_sem(results_C[p]["TC"])
        nl_m, _ = mean_sem(results_C[p]["links"])
        output["model_C"][str(p)] = {
            "fiedler_mean": f_m, "fiedler_sem": f_s,
            "TC_mean": tc_m, "links_mean": nl_m,
            "dTC_pct": (tc_m - bl_tc) / bl_tc * 100,
            "dFiedler_pct": (f_m - bl_f) / bl_f * 100,
        }

    out_path = Path("speculative/numerics/fnd1_null_model_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
