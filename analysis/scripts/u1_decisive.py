#!/usr/bin/env python3
"""U1 DECISIVE: cubic jet universality at STRONG signal.

Uses:
- build_hasse_from_predicate (Python bigint, PROVEN at N=10000)
- cubic_jet_preds from u1_cubic_jet_experiment.py (VALIDATED: identical to jet_preds when nabla_R=0)
- N=10000 (not 5000)
- q=0.50 and q=1.00 (not 0.10/0.20 which were underpowered)
- T=1.0 (strongest signal, even if not asymptotic)
- M_seeds=30

Conditions:
1. ppw cubic T=1.0 q=0.50  (eps=0.50)
2. ppw cubic T=1.0 q=1.00  (eps=1.00)
3. ppw exact T=1.0 q=0.50  (calibration J_pp)
4. ppw exact T=1.0 q=1.00  (calibration J_pp)
5. sch cubic T=1.0 q=0.50  (M=0.0625, r0=0.50)
6. sch cubic T=1.0 q=1.00  (M=0.125, r0=0.50)
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

N = 10000
ZETA = 0.15
T = 1.0
M_SEEDS = 30
R0 = 0.50

CONDITIONS = [
    {"id": "ppw_cubic_q0.5",  "geom": "ppw", "pred": "cubic", "q": 0.50},
    {"id": "ppw_cubic_q1.0",  "geom": "ppw", "pred": "cubic", "q": 1.00},
    {"id": "ppw_exact_q0.5",  "geom": "ppw", "pred": "exact", "q": 0.50},
    {"id": "ppw_exact_q1.0",  "geom": "ppw", "pred": "exact", "q": 1.00},
    {"id": "sch_cubic_q0.5",  "geom": "sch", "pred": "cubic", "q": 0.50},
    {"id": "sch_cubic_q1.0",  "geom": "sch", "pred": "cubic", "q": 1.00},
]


if __name__ == "__main__":
    print(f"=== U1 DECISIVE: N={N}, T={T}, M={M_SEEDS}, zeta={ZETA} ===", flush=True)
    print(f"Hasse: build_hasse_from_predicate (Python bigint, proven at N=10000)", flush=True)
    print(flush=True)

    all_results = {}
    t_total = time.time()

    for cond in CONDITIONS:
        q = cond["q"]
        geom = cond["geom"]
        pred_type = cond["pred"]
        cond_id = cond["id"]

        if geom == "ppw":
            eps = q / T**2
            R = riemann_ppwave_canonical(eps)
            nabR = nabla_riemann_ppwave(eps)
            E2 = eps**2 / 2.0
        else:
            M_sch = q * R0**3 / T**2
            R = riemann_schwarzschild_local(M_sch, R0)
            nabR = nabla_riemann_schwarzschild(M_sch, R0)
            E2 = 6.0 * (M_sch / R0**3)**2

        print(f"--- {cond_id} (eps/M={eps if geom=='ppw' else M_sch:.4f}, E²={E2:.6f}) ---", flush=True)

        dks = []
        t0 = time.time()
        for si in range(M_SEEDS):
            seed = 1200000 + hash(cond_id) % 10000 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)

            if pred_type == "exact":
                parF, chF = build_hasse_from_predicate(
                    pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
            else:
                parF, chF = build_hasse_from_predicate(
                    pts, lambda P, i: cubic_jet_preds(P, i, R_abcd=R, nabla_R=nabR))
            YF = Y_from_graph(parF, chF)

            mask = bulk_mask(pts, T, ZETA)
            dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dks.append(dk)

            if (si + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {si+1}/{M_SEEDS} ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        arr = np.array(dks)
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        std = float(np.std(arr, ddof=1))
        d = float(m / std) if std > 0 else 0.0
        AE = m / (T**4 * E2) if E2 > 1e-15 else 0.0

        sig = "***" if abs(d) > 3 else "**" if abs(d) > 2 else "*" if abs(d) > 1 else ""
        print(f"  dk={m:+.6f}±{se:.6f}, d={d:+.3f}{sig}, A_E={AE:.6f} ({elapsed:.0f}s)", flush=True)
        print(flush=True)

        all_results[cond_id] = {
            "geom": geom, "pred": pred_type, "q": q, "T": T, "E2": E2,
            "dk": m, "se": se, "std": std, "d": d, "A_E": AE,
            "per_seed": [float(x) for x in arr],
        }

    # Summary
    print("=== UNIVERSALITY COMPARISON ===", flush=True)
    for q in [0.50, 1.00]:
        ppw_c = all_results.get(f"ppw_cubic_q{q}")
        ppw_e = all_results.get(f"ppw_exact_q{q}")
        sch_c = all_results.get(f"sch_cubic_q{q}")

        if ppw_c and sch_c:
            ratio = sch_c["A_E"] / ppw_c["A_E"] if abs(ppw_c["A_E"]) > 1e-15 else float('inf')
            print(f"  q={q}: A_E(ppw_cubic)={ppw_c['A_E']:.6f}, A_E(sch_cubic)={sch_c['A_E']:.6f}, "
                  f"RATIO={ratio:.3f}", flush=True)

            if 0.80 <= ratio <= 1.25:
                print(f"    → PASS (universality)", flush=True)
            elif 0.65 <= ratio <= 1.50:
                print(f"    → BORDERLINE", flush=True)
            else:
                print(f"    → FAIL (ratio outside [0.65, 1.50])", flush=True)

        if ppw_c and ppw_e:
            J = ppw_e["dk"] / ppw_c["dk"] if abs(ppw_c["dk"]) > 1e-15 else float('inf')
            print(f"  q={q}: J_pp(exact/cubic) = {J:.3f} (should be ~1.0)", flush=True)

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "u1_decisive.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.", flush=True)
