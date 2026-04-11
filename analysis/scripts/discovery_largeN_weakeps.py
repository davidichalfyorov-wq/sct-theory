"""
Discovery Run 001 — Large-N Weak-Epsilon Regime
=================================================

N=2000, pp-wave eps=1-2. In this regime K·τ⁴ << 1,
so the perturbative formula V = V_flat(1 + a₂·K·τ⁴) should be valid.

If the formula fits: we extract the coefficient a₂.
If the signal is NULL at eps=1-2: we establish the sensitivity threshold.

Also tests: does scaling_exp remain GENUINE at N=2000?

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_analytical import volume_profile, TAU_FINE

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])


def crn_trial_largeN(seed, N, T, eps, tau_bins):
    """CRN trial at large N for pp-wave."""
    rng = np.random.default_rng(seed + 100)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "eps": eps}

    C_flat = causal_flat(pts)
    C_curv = causal_ppwave_quad(pts, eps)

    # Volume profiles (fine bins for ratio fitting)
    tau_f, vf, nf = volume_profile(C_flat, pts, N, TAU_FINE)
    tau_c, vc, nc = volume_profile(C_curv, pts, N, TAU_FINE)

    # Scaling exponent from coarse bins
    from discovery_weyl_probes import interval_volumes_binned, weyl_observables
    rng_b = np.random.default_rng(seed + 7777)
    bins_flat = interval_volumes_binned(C_flat, pts, N, tau_bins, rng=rng_b)
    rng_b = np.random.default_rng(seed + 7777)
    bins_curved = interval_volumes_binned(C_curv, pts, N, tau_bins, rng=rng_b)

    w = weyl_observables(bins_flat, bins_curved, tau_bins)
    result.update(w)

    # Degree stats for adversarial
    A_flat = build_link_graph(C_flat)
    gs_flat, _ = graph_statistics(A_flat)
    A_curv = build_link_graph(C_curv)
    gs_curv, _ = graph_statistics(A_curv)
    for key in gs_flat:
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]
    fr_f, _ = forman_ricci(A_flat)
    fr_c, _ = forman_ricci(A_curv)
    result["forman_mean_delta"] = fr_c["F_mean"] - fr_f["F_mean"]

    # Save volume ratios at fine bins for fitting
    ratios = []
    taus = []
    for i, t in enumerate(tau_f):
        idx_c = np.argmin(np.abs(tau_c - t))
        if abs(tau_c[idx_c] - t) < 0.005 and vf[i] > 0.5:
            ratios.append(vc[idx_c] / vf[i])
            taus.append(t)
    result["ratio_taus"] = taus
    result["ratio_vals"] = ratios

    del C_flat, C_curv, A_flat, A_curv; gc.collect()
    return result


def adversarial_simple(results, obs_key, label=""):
    """Quick adversarial check."""
    M = len(results)
    obs = np.array([r.get(obs_key, np.nan) for r in results])
    valid = ~np.isnan(obs)
    obs = obs[valid]
    M_v = len(obs)

    if M_v < 5:
        return {"verdict": "INSUFFICIENT", "cohen_d": 0, "p_value": 1, "r2_multiple": 0}

    m = np.mean(obs)
    d_c = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p = stats.ttest_1samp(obs, 0.0)

    proxy_stats = ["mean_degree_delta", "degree_var_delta", "degree_std_delta",
                   "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
                   "max_degree_delta", "assortativity_delta", "forman_mean_delta"]

    max_r2 = 0.0
    valid_res = [r for r in results if not np.isnan(r.get(obs_key, np.nan))][:M_v]
    X_cols = []
    for sn in proxy_stats:
        vals = np.array([r.get(sn, 0) for r in valid_res], dtype=float)
        if len(vals) == M_v and np.std(vals) > 1e-15 and np.std(obs) > 1e-15:
            r2 = np.corrcoef(obs, vals)[0, 1]**2
            if r2 > max_r2:
                max_r2 = r2
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

    print(f"  {label} {obs_key}: d={d_c:+.3f}, p={p:.2e}, "
          f"R²_multi={r2_multi:.3f} → {verdict}")

    return {"verdict": verdict, "cohen_d": float(d_c), "p_value": float(p),
            "r2_multiple": float(r2_multi), "mean_delta": float(m)}


def main():
    all_results = {}

    print("=" * 70)
    print("LARGE-N WEAK-EPSILON: Perturbative Regime")
    print("=" * 70)

    # Test at N=2000 with eps=1,2,5
    for N, eps_list, M in [(2000, [1.0, 2.0, 5.0], 20)]:
        for eps in eps_list:
            label = f"ppw_N{N}_eps{eps}"
            print(f"\n--- {label} (M={M}) ---")
            t0 = time.time()

            results = []
            for trial in range(M):
                res = crn_trial_largeN(trial * 1000, N, T, eps, tau_bins)
                results.append(res)
                if (trial + 1) % 5 == 0:
                    elapsed = time.time() - t0
                    se_d = [r.get("scaling_exp_delta", np.nan) for r in results]
                    se_v = [v for v in se_d if not np.isnan(v)]
                    print(f"  trial {trial+1}/{M}: scaling_exp={np.mean(se_v):+.4f} [{elapsed:.1f}s]")

            elapsed = time.time() - t0

            v_se = adversarial_simple(results, "scaling_exp_delta", label)
            v_vr = adversarial_simple(results, "var_ratio_delta", label)

            # Try to fit ratio = 1 + a*tau^4 at this eps
            all_taus = []
            all_ratios = []
            for r in results:
                for t, rv in zip(r.get("ratio_taus", []), r.get("ratio_vals", [])):
                    if t > 0.05 and rv > 0.5 and rv < 3.0:
                        all_taus.append(t)
                        all_ratios.append(rv)

            fit_result = {}
            if len(all_taus) > 20:
                tau_a = np.array(all_taus)
                rat_a = np.array(all_ratios)

                # Bin for cleaner fit
                tau_edges = np.linspace(0.05, 0.40, 15)
                tau_mid_b, rat_mean_b = [], []
                for i in range(len(tau_edges)-1):
                    mask = (tau_a >= tau_edges[i]) & (tau_a < tau_edges[i+1])
                    if np.sum(mask) >= 5:
                        tau_mid_b.append((tau_edges[i] + tau_edges[i+1])/2)
                        rat_mean_b.append(np.mean(rat_a[mask]))

                if len(tau_mid_b) >= 4:
                    tau_b = np.array(tau_mid_b)
                    rat_b = np.array(rat_mean_b)

                    try:
                        def model_t4(t, a):
                            return 1 + a * t**4
                        popt, _ = curve_fit(model_t4, tau_b, rat_b, p0=[10])
                        a4 = popt[0]
                        pred = model_t4(tau_b, a4)
                        ss_res = np.sum((rat_b - pred)**2)
                        ss_tot = np.sum((rat_b - np.mean(rat_b))**2)
                        r2_fit = 1 - ss_res/ss_tot if ss_tot > 0 else 0

                        K = 8 * eps**2
                        a2_coeff = a4 / K if K > 0 else 0

                        fit_result = {
                            "a4": float(a4), "r2": float(r2_fit),
                            "K": float(K), "a2": float(a2_coeff),
                        }
                        print(f"  Ratio fit: 1 + {a4:.2f}×τ⁴, R²={r2_fit:.3f}, "
                              f"a₂=a/K={a2_coeff:.4f}")

                        if r2_fit > 0.8:
                            print(f"  *** PERTURBATIVE FIT WORKS at eps={eps}! ***")
                    except Exception as e:
                        print(f"  Fit failed: {e}")

            all_results[label] = {
                "N": N, "eps": eps, "M": M, "elapsed_sec": elapsed,
                "scaling_exp": v_se, "var_ratio": v_vr,
                "ratio_fit": fit_result,
            }

    # Save
    outpath = os.path.join(OUTDIR, "largeN_weakeps.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("LARGE-N WEAK-EPSILON SUMMARY")
    print("=" * 70)
    print(f"{'condition':25s} {'SE d':>8s} {'SE p':>12s} {'SE verdict':>22s} {'fit R²':>8s} {'a₂':>10s}")
    print("-" * 90)

    for label, res in all_results.items():
        se = res.get("scaling_exp", {})
        fit = res.get("ratio_fit", {})
        print(f"{label:25s} {se.get('cohen_d',0):+8.3f} {se.get('p_value',1):12.2e} "
              f"{se.get('verdict','?'):>22s} {fit.get('r2',0):8.3f} {fit.get('a2',0):+10.4f}")


if __name__ == "__main__":
    main()
