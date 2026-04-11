"""
Discovery Run 001 — Test 1: Adversarial Proxy Check on RESIDUAL
================================================================

THE most dangerous test. Can raise score to 8/10 or drop to 4/10.

The RESIDUAL = scaling_exp(ppwave) - scaling_exp(synthetic_lapse)
captures the curvature component AFTER removing the lapse contribution.

QUESTION: Is this residual degree-independent? Or is it ALSO a
degree proxy (in which case the curvature claim is dead)?

PRE-REGISTERED:
  - Compute R² of RESIDUAL against ALL degree stat DIFFERENCES
    (degree stats of ppwave minus degree stats of synthetic lapse)
  - Also against: degree stats of ppwave minus degree stats of flat
  - Bonferroni alpha = 0.001
  - GENUINE: max R² < 0.50 AND R²_multi < 0.50
  - PROXY: max R² > 0.80 OR R²_multi > 0.80
  - This test is DESIGNED to kill the claim if the residual is a proxy

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_lapse_separation import causal_synthetic_lapse
from discovery_weyl_probes import interval_volumes_binned, weyl_observables

METRIC_FNS["synthetic_lapse"] = causal_synthetic_lapse
SEED_OFFSETS["synthetic_lapse"] = 100

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])


def crn_trial_residual_full(seed, N, T, eps):
    """Triple CRN with FULL degree stats for adversarial check on residual."""
    rng = np.random.default_rng(seed + 100)
    pts = sprinkle_4d(N, T, rng)

    C_flat = causal_flat(pts)
    C_ppw = causal_ppwave_quad(pts, eps)
    C_syn = causal_synthetic_lapse(pts, eps)

    result = {"seed": seed, "N": N, "eps": eps}

    # Weyl observables for ppw-flat and syn-flat
    rng_b = np.random.default_rng(seed + 7777)
    bins_flat = interval_volumes_binned(C_flat, pts, N, tau_bins, rng=rng_b)
    rng_b = np.random.default_rng(seed + 7777)
    bins_ppw = interval_volumes_binned(C_ppw, pts, N, tau_bins, rng=rng_b)
    rng_b = np.random.default_rng(seed + 7777)
    bins_syn = interval_volumes_binned(C_syn, pts, N, tau_bins, rng=rng_b)

    w_ppw = weyl_observables(bins_flat, bins_ppw, tau_bins)
    w_syn = weyl_observables(bins_flat, bins_syn, tau_bins)

    # Residuals
    for obs in ["scaling_exp_delta", "var_ratio_delta", "vol_ratio_delta"]:
        ppw_v = w_ppw.get(obs, np.nan)
        syn_v = w_syn.get(obs, np.nan)
        result[f"{obs}_ppw"] = ppw_v
        result[f"{obs}_syn"] = syn_v
        if not (np.isnan(ppw_v) or np.isnan(syn_v)):
            result[f"{obs}_residual"] = ppw_v - syn_v
        else:
            result[f"{obs}_residual"] = np.nan

    # Full degree stats for ALL THREE metrics
    for tag, C in [("flat", C_flat), ("ppw", C_ppw), ("syn", C_syn)]:
        A = build_link_graph(C)
        gs, _ = graph_statistics(A)
        fr, _ = forman_ricci(A)
        for key in gs:
            result[f"{key}_{tag}"] = gs[key]
        result[f"forman_mean_{tag}"] = fr["F_mean"]
        del A

    # Degree stat DELTAS (multiple reference frames for adversarial)
    stat_keys = ["mean_degree", "degree_var", "degree_std", "degree_skew",
                 "degree_kurt", "edge_count", "max_degree", "assortativity",
                 "forman_mean"]

    # ppw - flat (standard)
    for key in stat_keys:
        result[f"{key}_ppw_flat_delta"] = result.get(f"{key}_ppw", 0) - result.get(f"{key}_flat", 0)

    # syn - flat
    for key in stat_keys:
        result[f"{key}_syn_flat_delta"] = result.get(f"{key}_syn", 0) - result.get(f"{key}_flat", 0)

    # ppw - syn (DIFFERENCE of degree changes — most relevant for residual)
    for key in stat_keys:
        result[f"{key}_ppw_syn_delta"] = result.get(f"{key}_ppw", 0) - result.get(f"{key}_syn", 0)

    del C_flat, C_ppw, C_syn; gc.collect()
    return result


def adversarial_on_residual(results, obs_key, proxy_suffix, label=""):
    """Full adversarial proxy check on the residual."""
    M = len(results)
    obs = np.array([r.get(obs_key, np.nan) for r in results])
    valid = ~np.isnan(obs)
    obs = obs[valid]
    M_v = len(obs)

    if M_v < 10:
        print(f"  {label}: INSUFFICIENT ({M_v} valid)")
        return {"verdict": "INSUFFICIENT"}

    m = float(np.mean(obs))
    se = float(np.std(obs) / np.sqrt(M_v))
    d_c = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p = stats.ttest_1samp(obs, 0.0)

    print(f"\n  {label} {obs_key} (proxy ref: {proxy_suffix}):")
    print(f"    delta={m:+.4f} +/- {se:.4f}, d={d_c:+.3f}, p={p:.2e}")

    # Adversarial against degree stat deltas with given suffix
    stat_keys = ["mean_degree", "degree_var", "degree_std", "degree_skew",
                 "degree_kurt", "edge_count", "max_degree", "assortativity",
                 "forman_mean"]

    valid_res = [r for r in results if not np.isnan(r.get(obs_key, np.nan))][:M_v]
    max_r2, max_name = 0.0, ""
    X_cols = []
    col_names = []

    print(f"    ADVERSARIAL (vs {proxy_suffix} degree deltas):")
    for sn in stat_keys:
        full_name = f"{sn}_{proxy_suffix}"
        vals = np.array([r.get(full_name, 0) for r in valid_res], dtype=float)
        if len(vals) != M_v or np.std(vals) < 1e-15 or np.std(obs) < 1e-15:
            continue
        corr = np.corrcoef(obs, vals)[0, 1]
        r2 = corr**2
        if r2 > max_r2:
            max_r2, max_name = r2, full_name
        if r2 > 0.10:
            flag = " PROXY!!!" if r2 > 0.80 else (" AMBIG" if r2 > 0.50 else (" notable" if r2 > 0.20 else ""))
            print(f"      R²({obs_key}, {full_name}) = {r2:.3f}{flag}")
        X_cols.append(vals)
        col_names.append(full_name)

    r2_multi = 0.0
    adj_r2 = 0.0
    if X_cols:
        X = np.column_stack(X_cols)
        X_c = np.column_stack([np.ones(M_v), X])
        k = X.shape[1]
        try:
            beta, _, _, _ = np.linalg.lstsq(X_c, obs, rcond=None)
            ss_res = np.sum((obs - X_c @ beta)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            adj_r2 = 1 - (1 - r2_multi) * (M_v - 1) / max(M_v - k - 2, 1)
        except:
            pass

    print(f"    MAX R² = {max_r2:.3f} ({max_name})")
    print(f"    Multi R² = {r2_multi:.3f} (adj = {adj_r2:.3f})")

    ALPHA = 0.001
    if p < ALPHA and max_r2 < 0.50 and r2_multi < 0.50:
        verdict = "GENUINE (degree-independent)"
    elif max_r2 > 0.80 or r2_multi > 0.80:
        verdict = "PROXY ☠️"
    elif p < ALPHA and max(max_r2, r2_multi) >= 0.50:
        verdict = "AMBIGUOUS"
    elif p < 0.05:
        verdict = "WEAK"
    else:
        verdict = "NULL"

    print(f"    VERDICT: {verdict}")

    return {
        "verdict": verdict, "cohen_d": float(d_c), "p_value": float(p),
        "max_r2": float(max_r2), "max_r2_name": max_name,
        "r2_multiple": float(r2_multi), "adj_r2": float(adj_r2),
        "mean": float(m), "se": float(se), "M": M_v,
    }


def main():
    N = 500
    M = 60  # more trials for reliable R² estimates

    print("=" * 70)
    print("TEST 1: Adversarial Proxy Check on RESIDUAL")
    print(f"N={N}, M={M}")
    print("PRE-REGISTERED: R²_multi > 0.50 → PROXY → claim DEAD")
    print("=" * 70)

    all_results = {}

    for eps in [10.0, 20.0]:
        label = f"ppw_eps{eps}"
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

        t0 = time.time()
        results = []
        for trial in range(M):
            res = crn_trial_residual_full(trial * 1000, N, T, eps)
            results.append(res)
            if (trial + 1) % 20 == 0:
                elapsed = time.time() - t0
                r = [x.get("scaling_exp_delta_residual", np.nan) for x in results]
                r_v = [v for v in r if not np.isnan(v)]
                print(f"  trial {trial+1}/{M}: resid={np.mean(r_v):+.4f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        # Check residual against THREE sets of degree deltas:
        # 1. ppw-flat deltas (standard)
        # 2. syn-flat deltas
        # 3. ppw-syn deltas (most relevant — these are the degree DIFFERENCES
        #    between ppwave and synthetic lapse)

        for obs in ["scaling_exp_delta_residual", "var_ratio_delta_residual"]:
            for proxy_suffix in ["ppw_flat_delta", "syn_flat_delta", "ppw_syn_delta"]:
                v = adversarial_on_residual(results, obs, proxy_suffix, label)
                key = f"{label}_{obs}_{proxy_suffix}"
                all_results[key] = v

    # Save
    outpath = os.path.join(OUTDIR, "residual_proxy.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("RESIDUAL PROXY CHECK — SUMMARY")
    print("=" * 70)
    print(f"{'key':60s} {'R²multi':>8s} {'adj_R²':>8s} {'verdict':>25s}")
    print("-" * 105)

    any_proxy = False
    for key, v in all_results.items():
        if not isinstance(v, dict) or "verdict" not in v:
            continue
        print(f"{key:60s} {v.get('r2_multiple',0):8.3f} {v.get('adj_r2',0):8.3f} {v['verdict']:>25s}")
        if "PROXY" in v["verdict"]:
            any_proxy = True

    if any_proxy:
        print("\n☠️ RESIDUAL IS A PROXY. Curvature claim DEAD.")
    else:
        print("\n✅ RESIDUAL IS DEGREE-INDEPENDENT. Curvature claim SURVIVES.")


if __name__ == "__main__":
    main()
