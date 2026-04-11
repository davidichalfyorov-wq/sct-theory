#!/usr/bin/env python3
"""U1 QUICK: cubic jet universality test using WORKING Hasse builder.

Uses build_hasse_from_predicate (Python bigint, proven at N=10000)
and cubic_jet_preds from u1_cubic_jet_experiment.py.
Sequential. N=5000 for speed (A_E(5k) ≈ 0.021 already measured).
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, bulk_mask,
    riemann_ppwave_canonical, riemann_schwarzschild_local
)
from u1_cubic_jet_experiment import (
    cubic_jet_preds, nabla_riemann_schwarzschild, nabla_riemann_ppwave
)

N = 5000
ZETA = 0.15
M_SEEDS = 20
R0 = 0.50

CONDITIONS = [
    {"label": "ppw_cubic",  "T": 0.70, "q": 0.10, "geom": "ppw", "pred": "cubic"},
    {"label": "ppw_cubic",  "T": 0.70, "q": 0.20, "geom": "ppw", "pred": "cubic"},
    {"label": "ppw_exact",  "T": 0.70, "q": 0.10, "geom": "ppw", "pred": "exact"},
    {"label": "ppw_exact",  "T": 0.70, "q": 0.20, "geom": "ppw", "pred": "exact"},
    {"label": "sch_cubic",  "T": 0.70, "q": 0.10, "geom": "sch", "pred": "cubic"},
    {"label": "sch_cubic",  "T": 0.70, "q": 0.20, "geom": "sch", "pred": "cubic"},
]


if __name__ == "__main__":
    print(f"=== U1 QUICK: N={N}, M={M_SEEDS}, zeta={ZETA} ===", flush=True)
    print(f"Hasse: build_hasse_from_predicate (Python bigint, proven)", flush=True)
    print()

    all_results = {}
    t_total = time.time()

    for cond in CONDITIONS:
        T = cond["T"]
        q = cond["q"]
        geom = cond["geom"]
        pred_type = cond["pred"]

        if geom == "ppw":
            eps = q / T**2
            R = riemann_ppwave_canonical(eps)
            nabR = nabla_riemann_ppwave(eps)
            E2 = eps**2 / 2.0
        else:  # sch
            M_sch = q * R0**3 / T**2
            R = riemann_schwarzschild_local(M_sch, R0)
            nabR = nabla_riemann_schwarzschild(M_sch, R0)
            E2 = 6.0 * (M_sch / R0**3)**2

        cond_id = f"{cond['label']}_T{T}_q{q}"
        print(f"--- {cond_id} (E²={E2:.6f}) ---", flush=True)

        dks = []
        t0 = time.time()
        for si in range(M_SEEDS):
            seed = 990000 + hash(cond_id) % 10000 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            # Flat
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)

            # Curved
            if pred_type == "exact":
                parF, chF = build_hasse_from_predicate(
                    pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
            elif pred_type == "cubic":
                parF, chF = build_hasse_from_predicate(
                    pts, lambda P, i: cubic_jet_preds(P, i, R_abcd=R, nabla_R=nabR))
            YF = Y_from_graph(parF, chF)

            mask = bulk_mask(pts, T, ZETA)
            dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dks.append(dk)

            if (si + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"  {si+1}/{M_SEEDS} ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        arr = np.array(dks)
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        d = float(m / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0
        AE = m / (T**4 * E2) if E2 > 1e-15 else 0.0

        print(f"  dk={m:+.6f}±{se:.6f}, d={d:+.3f}, A_E={AE:.6f} ({elapsed:.0f}s)", flush=True)

        all_results[cond_id] = {
            "geom": geom, "pred": pred_type, "T": T, "q": q, "E2": E2,
            "dk": m, "se": se, "d": d, "A_E": AE, "per_seed": [float(x) for x in arr],
        }

    # Summary: universality ratio
    print("\n=== UNIVERSALITY COMPARISON ===", flush=True)
    for q in [0.10, 0.20]:
        ppw_key = f"ppw_cubic_T0.7_q{q}"
        sch_key = f"sch_cubic_T0.7_q{q}"
        ppw_exact_key = f"ppw_exact_T0.7_q{q}"

        if ppw_key in all_results and sch_key in all_results:
            AE_ppw = all_results[ppw_key]["A_E"]
            AE_sch = all_results[sch_key]["A_E"]
            ratio = AE_sch / AE_ppw if abs(AE_ppw) > 1e-15 else float('inf')
            print(f"  q={q}: A_E(ppw_cubic)={AE_ppw:.6f}, A_E(sch_cubic)={AE_sch:.6f}, ratio={ratio:.3f}", flush=True)

        if ppw_exact_key in all_results and ppw_key in all_results:
            J = all_results[ppw_exact_key]["dk"] / all_results[ppw_key]["dk"] if abs(all_results[ppw_key]["dk"]) > 1e-15 else 0
            print(f"  q={q}: J_pp(exact/cubic) = {J:.3f}", flush=True)

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "u1_quick.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.", flush=True)
