#!/usr/bin/env python3
"""FND-1 TEST 6.1: N-convergence of A_E.

Tests whether A_E(N) converges as N→∞.
Runs ppw exact + Sch expmap at N∈{3000, 5000, 10000, 15000, 20000}.
Fits A_E(N) = A_∞ + c·N^{-γ}.

analytical PASS: γ>0, last doubling changes A_E by <15%.
analytical FAIL: monotonic drift >20% every doubling.
"""
import sys, os, time, json
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_schwarzschild_local,
)
from schwarzschild_exact_local_tools import (
    map_rnc_to_schwarzschild_expmap,
    schwarzschild_exact_midpoint_preds_from_mapped
)

N_VALUES = [3000, 5000, 10000, 15000, 20000]
M_SEEDS = 20
SEED_BASE = 5000000
T = 1.0
EPS = 3.0
M_SCH = 0.05
R0 = 0.50
ZETA = 0.15

E2_PPW = EPS**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2
R_SCH = riemann_schwarzschild_local(M_SCH, R0)


def make_strata(pts, parents0, T):
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
    total = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = np.mean(X2[idx] * dY2[idx]) - np.mean(X2[idx]) * np.mean(dY2[idx])
        total += w * cov
    return float(total)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("FND-1 TEST 6.1: N-CONVERGENCE", flush=True)
    print(f"N_VALUES={N_VALUES}, M={M_SEEDS}, T={T}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()
    results = {}

    for N in N_VALUES:
        print(f"\n─── N = {N} ───", flush=True)
        a_ppw_list = []
        a_sch_exp_list = []
        a_sch_jet_list = []

        for si in range(M_SEEDS):
            seed = SEED_BASE + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            # Flat
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)
            strata = make_strata(pts, par0, T)

            # PPW exact
            parP, chP = build_hasse_from_predicate(
                pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
            YP = Y_from_graph(parP, chP)
            a_ppw = compute_A_align(Y0, YP - Y0, mask, strata)
            a_ppw_list.append(a_ppw)

            # Sch jet
            parS, chS = build_hasse_from_predicate(
                pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
            YS = Y_from_graph(parS, chS)
            a_sch_jet = compute_A_align(Y0, YS - Y0, mask, strata)
            a_sch_jet_list.append(a_sch_jet)

            # Sch expmap (skip for N≥20000 if too slow — estimate first)
            if N <= 15000:
                mapped = map_rnc_to_schwarzschild_expmap(pts, M_SCH, R0)
                parX, chX = build_hasse_from_predicate(
                    pts, lambda P, i: schwarzschild_exact_midpoint_preds_from_mapped(
                        mapped, i, M_SCH))
                YX = Y_from_graph(parX, chX)
                a_sch_exp = compute_A_align(Y0, YX - Y0, mask, strata)
                a_sch_exp_list.append(a_sch_exp)

            if (si + 1) % 5 == 0:
                elapsed = time.time() - t_total
                print(f"  {si+1}/{M_SEEDS} ({elapsed/60:.0f}min)", flush=True)

        AE_ppw = np.mean(a_ppw_list) / E2_PPW
        AE_sch_jet = np.mean(a_sch_jet_list) / E2_SCH
        AE_sch_exp = np.mean(a_sch_exp_list) / E2_SCH if a_sch_exp_list else None
        ratio_jet = AE_sch_jet / AE_ppw if abs(AE_ppw) > 1e-15 else None
        ratio_exp = AE_sch_exp / AE_ppw if AE_sch_exp and abs(AE_ppw) > 1e-15 else None

        print(f"  AE_ppw = {AE_ppw:.6f}", flush=True)
        print(f"  AE_sch_jet = {AE_sch_jet:.6f} (ratio = {ratio_jet:.3f})" if ratio_jet else
              f"  AE_sch_jet = {AE_sch_jet:.6f}", flush=True)
        if AE_sch_exp is not None:
            print(f"  AE_sch_exp = {AE_sch_exp:.6f} (ratio = {ratio_exp:.3f})", flush=True)

        results[N] = {
            "AE_ppw": AE_ppw,
            "AE_ppw_se": float(np.std(a_ppw_list) / np.sqrt(len(a_ppw_list)) / E2_PPW),
            "AE_sch_jet": AE_sch_jet,
            "AE_sch_exp": AE_sch_exp,
            "ratio_jet": ratio_jet,
            "ratio_exp": ratio_exp,
            "n_seeds": M_SEEDS,
        }

    # ── FIT: A_E(N) = A_inf + c*N^{-gamma} ──
    print(f"\n{'='*60}", flush=True)
    print("CONVERGENCE FIT: A_E(N) = A_inf + c*N^(-gamma)", flush=True)
    Ns = np.array(sorted(results.keys()), dtype=float)
    AEs = np.array([results[int(n)]["AE_ppw"] for n in Ns])

    try:
        def model(N, A_inf, c, gamma):
            return A_inf + c * N**(-gamma)
        popt, pcov = curve_fit(model, Ns, AEs, p0=[AEs[-1], 1.0, 0.5],
                               bounds=([0, -np.inf, 0.01], [np.inf, np.inf, 5.0]))
        print(f"  A_inf = {popt[0]:.6f}, c = {popt[1]:.4f}, gamma = {popt[2]:.3f}", flush=True)
        results["fit_ppw"] = {"A_inf": float(popt[0]), "c": float(popt[1]),
                               "gamma": float(popt[2])}
    except Exception as e:
        print(f"  Fit failed: {e}", flush=True)
        results["fit_ppw"] = {"error": str(e)}

    # Check analytical criterion: last doubling changes <15%
    if len(Ns) >= 2:
        last_change = abs(AEs[-1] - AEs[-2]) / abs(AEs[-2]) if abs(AEs[-2]) > 1e-15 else 0
        print(f"  Last step change: {last_change*100:.1f}% (PASS < 15%)", flush=True)
        results["last_step_pct"] = float(last_change * 100)

    total_time = time.time() - t_total
    results["total_time_min"] = total_time / 60
    print(f"\nTotal: {total_time/60:.0f}min = {total_time/3600:.1f}h", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "n_convergence.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to fnd1_data/n_convergence.json", flush=True)
