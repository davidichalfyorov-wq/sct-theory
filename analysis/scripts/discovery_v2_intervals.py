"""
Discovery Run 001 — Variant 2: Interval-Based Observables
==========================================================

After killing ALL graph-curvature candidates (Forman, ORC, gap ratio)
because they collapse to degree statistics, we try observables that
depend on INTERVAL INTERIORS — the meso/global causal structure that
is structurally immune to degree-proxy collapse.

THREE CANDIDATES:

V2-A: Interval Volume Deficit
  For pairs (x,y) with x<y at proper time τ, define:
    V(x,y) = |{z : x < z < y}| = number of elements in causal interval
  In flat d-dim Minkowski: E[V] = C_d * ρ * τ^d
  Curvature changes V. The deficit δV = V_curved - V_flat (CRN paired)
  encodes integrated Ricci scalar in the diamond.

  WHY DEGREE-INDEPENDENT: V counts interior elements, not boundary links.
  Degree = number of links to immediate neighbors. V = number of elements
  in the entire causal interval. These are structurally different.

V2-B: Longest Chain Deviation
  For pairs (x,y), L_max = length of longest chain x < z_1 < ... < z_k < y.
  In flat space, L_max scales as τ^{1/d} (with known coefficient).
  Curvature modifies this scaling.
  CRN paired delta: ΔL = L_max_curved - L_max_flat.

  WHY DEGREE-INDEPENDENT: Longest chain is a GLOBAL property of the interval,
  not a local neighborhood property.

V2-C: Ordering Fraction in Intervals
  For an interval I(x,y), the ordering fraction f = |{(a,b) : a<b, a,b ∈ I}| / |I|²
  In flat d-dim: f depends on d only (Myrheim-Meyer).
  Curvature introduces corrections.
  CRN paired delta: Δf = f_curved - f_flat at fixed interval size.

PRE-REGISTERED ANTI-BIAS (v1.0 + sampling fix):
  - Bonferroni alpha = 0.01/15 = 0.000667
  - Adversarial proxy: same 9 stats + mean_forman + mean_orc
  - NULL control: conformal
  - ALL edges (no subsampling for V2-A,B,C — they use intervals, not edges)

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_conformal, build_link_graph, graph_statistics, forman_ricci,
    SEED_OFFSETS, METRIC_FNS
)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
N_PAIRS = 500   # number of random pairs to sample per trial


# ---------------------------------------------------------------------------
# Interval observables
# ---------------------------------------------------------------------------
def interval_observables(C, N, rng, n_pairs=500):
    """Compute interval-based observables from causal matrix.

    Args:
        C: N×N causal matrix (C[i,j]=1 iff i<j)
        N: number of elements
        rng: random generator (for pair sampling)
        n_pairs: number of pairs to sample

    Returns dict with:
        vol_mean: mean interval volume |I(x,y)|
        vol_std: std of interval volumes
        vol_by_layer: mean volume binned by proper time (chain length) bins
        chain_mean: mean longest chain length
        chain_std: std
        ofrac_mean: mean ordering fraction in intervals
        ofrac_std: std
        pair_count: number of valid pairs used
    """
    # Sample random causally related pairs
    # Use the causal matrix to find pairs
    rows, cols = np.where(C > 0.5)

    if len(rows) < n_pairs:
        idx = np.arange(len(rows))
    else:
        idx = rng.choice(len(rows), size=n_pairs, replace=False)

    volumes = []
    chain_lengths = []
    ordering_fracs = []
    proper_times = []  # longest chain as proxy for proper time

    for k in idx:
        i, j = rows[k], cols[k]

        # Interval I(i,j) = {z : i < z < j}
        # Elements that are in the future of i AND past of j
        future_i = C[i, :] > 0.5   # i < z
        past_j = C[:, j] > 0.5     # z < j
        interior = future_i & past_j
        interior[i] = False
        interior[j] = False

        vol = int(np.sum(interior))
        volumes.append(vol)

        if vol < 2:
            # Too small for meaningful statistics
            continue

        # Longest chain: approximate via powers of C restricted to interval
        # For small intervals, just count layers
        # Layer k = elements at chain distance k from i
        interior_idx = np.where(interior)[0]

        # Ordering fraction: among interior elements, what fraction are related?
        if vol >= 2:
            C_sub = C[np.ix_(interior_idx, interior_idx)]
            n_related = int(np.sum(C_sub))
            n_total = vol * (vol - 1)  # ordered pairs
            ofrac = n_related / n_total if n_total > 0 else 0
            ordering_fracs.append(ofrac)

        # Longest chain from i to j: BFS-like via matrix powers
        # Simple approach: the longest chain through the interval
        # Approximate: number of "layers" in the interval
        # Layer 0: elements linked to i (future links of i in the interval)
        # This is expensive for large intervals, use a simpler proxy:
        # the chain distance = largest k such that C^k[i,j] > 0
        # For now, use the interval size as proxy (correlated with chain length)
        chain_lengths.append(vol)  # placeholder — will compute properly below

    # Compute proper time (longest chain) for sampled pairs via BFS
    # Only for pairs with vol > 0
    proper_times = []
    for k in idx[:min(200, len(idx))]:  # limit to 200 for speed
        i, j = rows[k], cols[k]
        # BFS from i to j through the causal order
        # Longest path in a DAG: topological sort + DP
        # Simple version: just count how many layers between i and j
        future_i = C[i, :] > 0.5
        past_j = C[:, j] > 0.5
        interior = future_i & past_j
        interior[i] = False
        interior[j] = False

        if np.sum(interior) == 0:
            proper_times.append(1)  # direct link
            continue

        # Longest chain via DP on the restricted DAG
        interior_idx = np.where(interior)[0]
        all_idx = np.concatenate([[i], interior_idx, [j]])
        n_sub = len(all_idx)
        idx_map = {v: k for k, v in enumerate(all_idx)}

        # Build subgraph adjacency
        dist = np.zeros(n_sub, dtype=int)
        # Process in topological order (already sorted by time coordinate)
        for a_pos in range(n_sub):
            a = all_idx[a_pos]
            for b_pos in range(a_pos + 1, n_sub):
                b = all_idx[b_pos]
                if C[a, b] > 0.5:
                    if dist[a_pos] + 1 > dist[b_pos]:
                        dist[b_pos] = dist[a_pos] + 1

        proper_times.append(int(dist[-1]))  # distance to j

    vol_arr = np.array(volumes, dtype=float)
    pt_arr = np.array(proper_times, dtype=float)
    of_arr = np.array(ordering_fracs, dtype=float) if ordering_fracs else np.array([0.0])

    result = {
        "vol_mean": float(np.mean(vol_arr)) if len(vol_arr) > 0 else 0,
        "vol_std": float(np.std(vol_arr)) if len(vol_arr) > 0 else 0,
        "vol_median": float(np.median(vol_arr)) if len(vol_arr) > 0 else 0,
        "vol_q25": float(np.percentile(vol_arr, 25)) if len(vol_arr) > 0 else 0,
        "vol_q75": float(np.percentile(vol_arr, 75)) if len(vol_arr) > 0 else 0,
        "chain_mean": float(np.mean(pt_arr)) if len(pt_arr) > 0 else 0,
        "chain_std": float(np.std(pt_arr)) if len(pt_arr) > 0 else 0,
        "chain_median": float(np.median(pt_arr)) if len(pt_arr) > 0 else 0,
        "ofrac_mean": float(np.mean(of_arr)) if len(of_arr) > 0 else 0,
        "ofrac_std": float(np.std(of_arr)) if len(of_arr) > 0 else 0,
        "n_pairs": len(vol_arr),
        "n_chains": len(pt_arr),
        "n_ofracs": len(of_arr),
    }

    return result


# ---------------------------------------------------------------------------
# CRN trial
# ---------------------------------------------------------------------------
def crn_trial_v2(seed, N, T, metric_name, eps, n_pairs=500):
    """CRN trial with interval-based observables + degree stats for adversarial check."""
    seed_offset = SEED_OFFSETS.get(metric_name, 100)
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps}

    for tag, C_fn in [("flat", lambda: causal_flat(pts)),
                       ("curv", lambda: METRIC_FNS[metric_name](pts, eps))]:
        C = C_fn()

        # Interval observables (from full causal matrix, NOT link graph)
        rng_pairs = np.random.default_rng(seed + 9999)  # same pairs for flat/curved
        iv = interval_observables(C, N, rng_pairs, n_pairs)

        # Also compute degree stats (for adversarial check)
        A = build_link_graph(C)
        gs, deg = graph_statistics(A)
        fr, _ = forman_ricci(A, deg)

        for key, val in iv.items():
            result[f"{key}_{tag}"] = val
        result[f"forman_mean_{tag}"] = fr["F_mean"]
        for key in gs:
            result[f"{key}_{tag}"] = gs[key]

        del C, A; gc.collect()

    # Deltas
    for key in ["vol_mean", "vol_std", "vol_median", "vol_q25", "vol_q75",
                "chain_mean", "chain_std", "chain_median",
                "ofrac_mean", "ofrac_std"]:
        result[f"{key}_delta"] = result.get(f"{key}_curv", 0) - result.get(f"{key}_flat", 0)

    for key in ["mean_degree", "degree_var", "degree_std", "degree_skew",
                "degree_kurt", "edge_count", "max_degree", "assortativity"]:
        result[f"{key}_delta"] = result.get(f"{key}_curv", 0) - result.get(f"{key}_flat", 0)
    result["forman_mean_delta"] = result.get("forman_mean_curv", 0) - result.get("forman_mean_flat", 0)

    return result


# ---------------------------------------------------------------------------
# Adversarial analysis
# ---------------------------------------------------------------------------
def adversarial_v2(results, obs_key, label=""):
    """Full adversarial proxy check for interval-based observable."""
    M = len(results)
    obs = np.array([r[obs_key] for r in results])

    print(f"\n  === {label}: {obs_key} (M={M}) ===")
    m, se = np.mean(obs), np.std(obs) / np.sqrt(M)
    d_cohen = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p_obs = stats.ttest_1samp(obs, 0.0) if M >= 5 else (0, 1.0)
    print(f"  delta = {m:+.4f} +/- {se:.4f}, d={d_cohen:+.3f}, p={p_obs:.2e}")

    proxy_stats = [
        "mean_degree_delta", "degree_var_delta", "degree_std_delta",
        "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
        "max_degree_delta", "assortativity_delta", "forman_mean_delta",
    ]

    print(f"  ADVERSARIAL PROXY CHECK:")
    max_r2 = 0.0
    max_r2_name = ""
    for stat_name in proxy_stats:
        vals = np.array([r.get(stat_name, 0) for r in results], dtype=float)
        if np.std(vals) < 1e-15 or np.std(obs) < 1e-15:
            continue
        corr = np.corrcoef(obs, vals)[0, 1]
        r2 = corr**2
        if r2 > max_r2:
            max_r2 = r2
            max_r2_name = stat_name
        if r2 > 0.15:
            flag = " PROXY!!!" if r2 > 0.80 else (" AMBIGUOUS" if r2 > 0.50 else " notable")
            print(f"    R2({obs_key}, {stat_name}) = {r2:.3f}{flag}")

    # Multiple regression
    X_cols = []
    for stat_name in proxy_stats:
        vals = np.array([r.get(stat_name, 0) for r in results], dtype=float)
        if np.std(vals) > 1e-15:
            X_cols.append(vals)

    r2_multi = 0.0
    if X_cols and np.std(obs) > 1e-15:
        X = np.column_stack(X_cols)
        X_c = np.column_stack([np.ones(M), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_c, obs, rcond=None)
            y_pred = X_c @ beta
            ss_res = np.sum((obs - y_pred)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except:
            pass

    print(f"    MAX R2 = {max_r2:.3f} ({max_r2_name})")
    print(f"    Multiple regression R2 = {r2_multi:.3f}")

    ALPHA_BONF = 0.01 / 15
    if p_obs < ALPHA_BONF and max_r2 < 0.50 and r2_multi < 0.50:
        verdict = "DETECTED (genuine)"
    elif max_r2 > 0.80 or r2_multi > 0.80:
        verdict = "PROXY"
    elif p_obs < ALPHA_BONF and (0.50 <= max(max_r2, r2_multi) < 0.80):
        verdict = "AMBIGUOUS"
    elif p_obs < 0.05:
        verdict = "WEAK"
    else:
        verdict = "NULL"

    print(f"  VERDICT: {verdict}")

    return {
        "observable": obs_key, "mean_delta": float(m), "se_delta": float(se),
        "cohen_d": float(d_cohen), "p_value": float(p_obs),
        "max_r2_single": float(max_r2), "max_r2_name": max_r2_name,
        "r2_multiple": float(r2_multi), "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    N = 500
    M = 40

    TESTS = [
        ("conformal",     5.0),   # NULL CONTROL
        ("ppwave_quad",   10.0),  # known signal
        ("schwarzschild", 0.05),  # known signal
    ]

    OBSERVABLES = [
        "vol_mean_delta", "vol_median_delta",
        "chain_mean_delta", "chain_median_delta",
        "ofrac_mean_delta",
    ]

    print("=" * 70)
    print("VARIANT 2: Interval-Based Observables")
    print(f"N={N}, M={M}, n_pairs={N_PAIRS}")
    print(f"Bonferroni alpha = {0.01/15:.6f}")
    print("Observables: vol_mean, vol_median, chain_mean, chain_median, ofrac_mean")
    print("=" * 70)

    all_results = {}

    for geo, eps in TESTS:
        label = f"{geo}_eps{eps}"
        is_null = (geo == "conformal")
        note = " [NULL CONTROL]" if is_null else ""
        print(f"\n{'='*50}")
        print(f"  {label}{note}")
        print(f"{'='*50}")

        t0 = time.time()
        results = []
        for trial in range(M):
            res = crn_trial_v2(trial * 1000, N, T, geo, eps, n_pairs=N_PAIRS)
            results.append(res)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                vd = [r["vol_mean_delta"] for r in results]
                print(f"  trial {trial+1}/{M}: vol_delta={np.mean(vd):+.2f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        # Null check
        if is_null:
            vd = [r["vol_mean_delta"] for r in results]
            print(f"  NULL CONTROL: vol_mean_delta = {np.mean(vd):.4f} "
                  f"(should be ~0, may have small variance from pair sampling)")

        # Analyze each observable
        geo_results = {"geometry": geo, "eps": eps, "elapsed_sec": elapsed,
                       "is_null": is_null}

        for obs_key in OBSERVABLES:
            v = adversarial_v2(results, obs_key, label)
            geo_results[obs_key] = v

        geo_results["trials"] = results
        all_results[label] = geo_results

    # Save
    outpath = os.path.join(OUTDIR, "v2_intervals.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("VARIANT 2 SUMMARY")
    print("=" * 70)
    print(f"{'geometry':20s} {'observable':20s} {'delta':>10s} {'d':>8s} "
          f"{'p':>12s} {'R2max':>8s} {'R2multi':>8s} {'verdict':>22s}")
    print("-" * 105)

    genuine_count = 0
    for label, res in all_results.items():
        if res["is_null"]:
            continue
        for obs_key in OBSERVABLES:
            v = res[obs_key]
            print(f"{label:20s} {obs_key:20s} {v['mean_delta']:+10.3f} "
                  f"{v['cohen_d']:+8.3f} {v['p_value']:12.2e} "
                  f"{v['max_r2_single']:8.3f} {v['r2_multiple']:8.3f} "
                  f"{v['verdict']:>22s}")
            if v['verdict'] == "DETECTED (genuine)":
                genuine_count += 1

    print(f"\nGenuine detections: {genuine_count}")

    if genuine_count > 0:
        print("*** INTERVAL-BASED OBSERVABLES SURVIVE ADVERSARIAL PROTOCOL! ***")
    else:
        print("*** No genuine detections. Interval observables also fail. ***")


if __name__ == "__main__":
    main()
