"""
Discovery Run 001 — Attack 9 Resolution: Chain-Binned Scaling Exponent
=======================================================================

PROBLEM: Our scaling_exp bins pairs by coordinate proper time τ_coord,
which requires COORDINATES not available in a true causal set.

SOLUTION: Replace τ_coord with LONGEST CHAIN LENGTH between pairs.
Chain length is purely order-theoretic — computable from the partial
order alone, no coordinates needed.

In flat d-dimensional Minkowski, the longest chain between two elements
at proper time τ scales as τ^{1/d} (with known statistics). So chain
length IS a proxy for proper time, purely from the order.

We compute:
  V(k) = mean interval volume for pairs at chain distance k
  α_chain = slope of log(V) vs log(k)

In flat space: V ~ k^{d²/(d-1)} (since chain length ~ τ^{1/d} and V ~ τ^d).
Wait, actually: if chain_length ~ τ^{d/(d-1)} (Ilie-Thompson-Reid 2005),
then V ~ τ^d ~ chain_length^{d(d-1)/d} = chain_length^{d-1}.
In d=4: V ~ chain_length^3. So α_chain ≈ 3.

Curvature changes this exponent. The CRN delta Δα_chain should detect
curvature WITHOUT using any coordinate information.

PRE-REGISTERED:
  - If Δα_chain is GENUINE on pp-wave → Attack 9 RESOLVED
  - If Δα_chain is NULL → the observable requires coordinates, limitation stands
  - Bonferroni alpha = 0.01/10 = 0.001
  - Same adversarial proxy protocol

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0


def longest_chain_lengths(C, N, max_pairs=2000, rng=None):
    """Compute longest chain length between sampled causal pairs.

    Uses dynamic programming on the DAG (topologically sorted by time).
    Returns arrays: (pair_indices, chain_lengths, interval_volumes).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # C²[i,j] = number of elements between i and j = interval volume
    C2 = C @ C

    # Find causal pairs
    rows, cols = np.where(C > 0.5)
    n_pairs = len(rows)

    if n_pairs == 0:
        return np.array([]), np.array([]), np.array([])

    # Sample pairs
    if n_pairs > max_pairs:
        idx = rng.choice(n_pairs, size=max_pairs, replace=False)
        rows, cols = rows[idx], cols[idx]

    # For each pair, compute longest chain via DP on the restricted interval
    chain_lens = []
    volumes = []

    for k in range(len(rows)):
        i, j = rows[k], cols[k]
        vol = int(C2[i, j])
        volumes.append(vol)

        if vol == 0:
            chain_lens.append(1)  # direct link, chain length = 1
            continue

        # Get interior elements: those between i and j
        future_i = C[i, :] > 0.5
        past_j = C[:, j] > 0.5
        interior = future_i & past_j
        interior[i] = False
        interior[j] = False
        interior_idx = np.where(interior)[0]

        if len(interior_idx) == 0:
            chain_lens.append(1)
            continue

        # DP for longest path from i to j through interior
        # All elements are already sorted by time (sprinkle sorts by t)
        all_nodes = np.concatenate([[i], interior_idx, [j]])
        n_sub = len(all_nodes)

        # Distance array: dist[k] = longest chain from all_nodes[0]=i to all_nodes[k]
        dist = np.zeros(n_sub, dtype=int)

        for a_pos in range(n_sub):
            a = all_nodes[a_pos]
            for b_pos in range(a_pos + 1, n_sub):
                b = all_nodes[b_pos]
                if C[a, b] > 0.5:
                    new_dist = dist[a_pos] + 1
                    if new_dist > dist[b_pos]:
                        dist[b_pos] = new_dist

        chain_lens.append(int(dist[-1]))

    return rows, np.array(chain_lens), np.array(volumes)


def chain_binned_scaling(C, N, rng, k_min=2, k_max=10):
    """Compute V(k) — mean interval volume at chain length k.

    Returns tau-like bins (chain lengths) and corresponding mean volumes.
    Purely order-theoretic — no coordinates used!
    """
    _, chains, vols = longest_chain_lengths(C, N, max_pairs=3000, rng=rng)

    if len(chains) == 0:
        return np.array([]), np.array([]), np.array([])

    k_vals = []
    v_means = []
    v_counts = []

    for k in range(k_min, k_max + 1):
        mask = chains == k
        if np.sum(mask) >= 10:
            k_vals.append(k)
            v_means.append(float(np.mean(vols[mask])))
            v_counts.append(int(np.sum(mask)))

    return np.array(k_vals), np.array(v_means), np.array(v_counts)


def chain_scaling_exponent(k_vals, v_means):
    """Fit log(V) = α_chain × log(k) + const."""
    if len(k_vals) < 3 or np.any(v_means <= 0):
        return np.nan

    mask = v_means > 0.1
    if np.sum(mask) < 3:
        return np.nan

    slope, _ = np.polyfit(np.log(k_vals[mask].astype(float)),
                          np.log(v_means[mask]), 1)
    return float(slope)


def chain_var_ratio(C, N, rng, k_target=4):
    """Variance of interval volumes at fixed chain length k.

    Purely order-theoretic measure of anisotropy.
    """
    _, chains, vols = longest_chain_lengths(C, N, max_pairs=3000, rng=rng)

    mask = chains == k_target
    if np.sum(mask) < 10:
        return np.nan, np.nan

    return float(np.mean(vols[mask])), float(np.var(vols[mask]))


def crn_trial_chain(seed, N, T, metric_name, eps):
    """CRN trial with chain-binned observables (NO coordinates used for binning)."""
    seed_offset = SEED_OFFSETS.get(metric_name, 100)
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps}

    C_flat = causal_flat(pts)
    C_curv = METRIC_FNS[metric_name](pts, eps)

    # Chain-binned scaling (SAME rng for pair sampling → CRN)
    rng_c = np.random.default_rng(seed + 8888)
    k_f, v_f, n_f = chain_binned_scaling(C_flat, N, rng_c)
    rng_c = np.random.default_rng(seed + 8888)
    k_c, v_c, n_c = chain_binned_scaling(C_curv, N, rng_c)

    alpha_flat = chain_scaling_exponent(k_f, v_f)
    alpha_curv = chain_scaling_exponent(k_c, v_c)

    result["alpha_chain_flat"] = alpha_flat
    result["alpha_chain_curv"] = alpha_curv
    result["alpha_chain_delta"] = alpha_curv - alpha_flat if not (
        np.isnan(alpha_flat) or np.isnan(alpha_curv)) else np.nan

    # Chain-binned variance at k=4 and k=6
    for k_t in [3, 4, 5, 6]:
        rng_v = np.random.default_rng(seed + 9000 + k_t)
        m_f, var_f = chain_var_ratio(C_flat, N, rng_v, k_target=k_t)
        rng_v = np.random.default_rng(seed + 9000 + k_t)
        m_c, var_c = chain_var_ratio(C_curv, N, rng_v, k_target=k_t)

        if not (np.isnan(var_f) or np.isnan(var_c)) and var_f > 0:
            result[f"var_ratio_k{k_t}_delta"] = var_c / var_f - 1.0
        else:
            result[f"var_ratio_k{k_t}_delta"] = np.nan

    # Degree stats for adversarial
    A_flat = build_link_graph(C_flat)
    gs_flat, _ = graph_statistics(A_flat)
    fr_flat, _ = forman_ricci(A_flat)
    A_curv = build_link_graph(C_curv)
    gs_curv, _ = graph_statistics(A_curv)
    fr_curv, _ = forman_ricci(A_curv)

    for key in gs_flat:
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]
    result["forman_mean_delta"] = fr_curv["F_mean"] - fr_flat["F_mean"]

    del C_flat, C_curv, A_flat, A_curv; gc.collect()
    return result


def adversarial_chain(results, obs_key, label=""):
    """Adversarial proxy check for chain-binned observable."""
    M = len(results)
    obs = np.array([r.get(obs_key, np.nan) for r in results])
    valid = ~np.isnan(obs)
    obs = obs[valid]
    M_v = len(obs)

    if M_v < 10:
        print(f"  {label} {obs_key}: INSUFFICIENT ({M_v} valid)")
        return {"verdict": "INSUFFICIENT", "cohen_d": 0, "p_value": 1, "r2_multiple": 0}

    m, se = np.mean(obs), np.std(obs) / np.sqrt(M_v)
    d_c = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p = stats.ttest_1samp(obs, 0.0)

    proxy_stats = ["mean_degree_delta", "degree_var_delta", "degree_std_delta",
                   "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
                   "max_degree_delta", "assortativity_delta", "forman_mean_delta"]

    valid_res = [r for r in results if not np.isnan(r.get(obs_key, np.nan))][:M_v]
    max_r2, max_name = 0.0, ""
    X_cols = []
    for sn in proxy_stats:
        vals = np.array([r.get(sn, 0) for r in valid_res], dtype=float)
        if len(vals) == M_v and np.std(vals) > 1e-15 and np.std(obs) > 1e-15:
            r2 = np.corrcoef(obs, vals)[0, 1]**2
            if r2 > max_r2:
                max_r2, max_name = r2, sn
            X_cols.append(vals)

    r2_multi = 0.0
    if X_cols:
        X = np.column_stack(X_cols)
        X_c = np.column_stack([np.ones(M_v), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_c, obs, rcond=None)
            ss_res = np.sum((obs - X_c @ beta)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_multi = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        except:
            pass

    ALPHA_BONF = 0.01 / 10
    if p < ALPHA_BONF and max_r2 < 0.50 and r2_multi < 0.50:
        verdict = "DETECTED (genuine)"
    elif max_r2 > 0.80 or r2_multi > 0.80:
        verdict = "PROXY"
    elif p < ALPHA_BONF and max(max_r2, r2_multi) >= 0.50:
        verdict = "AMBIGUOUS"
    elif p < 0.05:
        verdict = "WEAK"
    else:
        verdict = "NULL"

    print(f"  {label} {obs_key:25s}: d={d_c:+.3f}, p={p:.2e}, "
          f"R²max={max_r2:.3f}, R²multi={r2_multi:.3f} → {verdict}")

    return {"verdict": verdict, "cohen_d": float(d_c), "p_value": float(p),
            "max_r2": float(max_r2), "r2_multiple": float(r2_multi),
            "mean_delta": float(m), "se": float(se), "M": M_v}


def main():
    N = 500
    M = 40

    TESTS = [
        ("conformal",     5.0),    # NULL
        ("ppwave_quad",   10.0),   # Weyl, no lapse
        ("ppwave_quad",   20.0),   # stronger
        ("schwarzschild", 0.05),   # Ricci + Weyl + lapse
    ]

    OBS_KEYS = ["alpha_chain_delta", "var_ratio_k4_delta", "var_ratio_k5_delta"]

    print("=" * 70)
    print("ATTACK 9 RESOLUTION: Chain-Binned Scaling Exponent")
    print("NO COORDINATES USED FOR BINNING — purely order-theoretic")
    print(f"N={N}, M={M}")
    print("=" * 70)

    all_results = {}

    for geo, eps in TESTS:
        label = f"{geo}_eps{eps}"
        is_null = (geo == "conformal")
        note = " [NULL]" if is_null else ""
        print(f"\n--- {label}{note} ---")
        t0 = time.time()

        results = []
        for trial in range(M):
            res = crn_trial_chain(trial * 1000, N, T, geo, eps)
            results.append(res)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                ad = [r.get("alpha_chain_delta", np.nan) for r in results]
                ad_v = [v for v in ad if not np.isnan(v)]
                print(f"  trial {trial+1}/{M}: α_chain_delta={np.mean(ad_v):+.4f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        geo_results = {"geometry": geo, "eps": eps, "is_null": is_null,
                       "elapsed_sec": elapsed}

        for obs_key in OBS_KEYS:
            v = adversarial_chain(results, obs_key, label)
            geo_results[obs_key] = v

        geo_results["trials"] = results
        all_results[label] = geo_results

        # Print absolute alpha values for context
        af = [r["alpha_chain_flat"] for r in results if not np.isnan(r["alpha_chain_flat"])]
        if af:
            print(f"  α_chain(flat) = {np.mean(af):.2f} (expected ~3 for d=4)")

    # Save
    outpath = os.path.join(OUTDIR, "chain_binned.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("CHAIN-BINNED SUMMARY (Attack 9 Resolution)")
    print("=" * 70)
    print(f"{'condition':25s} {'α_chain d':>10s} {'p':>12s} {'R²multi':>8s} {'verdict':>22s}")
    print("-" * 80)

    for label, res in all_results.items():
        v = res.get("alpha_chain_delta", {})
        if isinstance(v, dict) and "verdict" in v:
            print(f"{label:25s} {v.get('cohen_d',0):+10.3f} {v.get('p_value',1):12.2e} "
                  f"{v.get('r2_multiple',0):8.3f} {v['verdict']:>22s}")

    # THE VERDICT
    ppw10 = all_results.get("ppwave_quad_eps10.0", {}).get("alpha_chain_delta", {})
    ppw20 = all_results.get("ppwave_quad_eps20.0", {}).get("alpha_chain_delta", {})

    print("\n  ATTACK 9 VERDICT:")
    if ppw10.get("verdict") == "DETECTED (genuine)" or ppw20.get("verdict") == "DETECTED (genuine)":
        print("  *** RESOLVED: Chain-binned scaling_exp works WITHOUT coordinates! ***")
        print("  *** The observable is PURELY ORDER-THEORETIC. ***")
    elif ppw10.get("verdict") in ["WEAK", "AMBIGUOUS"] or ppw20.get("verdict") in ["WEAK", "AMBIGUOUS"]:
        print("  *** PARTIAL: Signal present but weaker without coordinates. ***")
    else:
        print("  *** FAILED: Chain-binned version does NOT detect curvature. ***")
        print("  *** Coordinate dependence is a real limitation. ***")


if __name__ == "__main__":
    main()
