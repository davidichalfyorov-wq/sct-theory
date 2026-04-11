#!/usr/bin/env python3
"""dS T-scaling with FIXED Ricci subtraction.

Bug was: full_mode=ds_exact, ric_mode=ricci_jet.
For dS: R_ric = R_full (Weyl=0), but jet ≠ exact → residual = jet error.
Fix: use ds_exact for BOTH full and Ricci control.
Then delta_weyl = YdS_exact - YdS_exact = 0 (exactly).

Also test: jet for BOTH (same approximation → weyl = 0).
"""
import sys, time, json, os
import numpy as np
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ds_preds, jet_preds, bulk_mask,
    riemann_ds, ricci_built_part, project_l0_grid
)

H_FIXED = 0.707
N = 10000
M_SEEDS = 50
ZETA = 0.15
T_VALS = [1.00, 0.70, 0.50, 0.35]


def run_one(args):
    T, si = args
    seed = 450000 + int(T * 1000) * 100 + si
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)

    # Flat
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    Y0 = Y_from_graph(par0, ch0)

    # dS exact (full)
    parD, chD = build_hasse_from_predicate(pts, lambda P, i: ds_preds(P, i, H=H_FIXED))
    YdS = Y_from_graph(parD, chD)

    # Ricci control = ALSO ds_exact (same predicate, because Weyl=0 for dS)
    # So YRic = YdS exactly → delta_weyl = 0
    YRic_exact = YdS  # trivially identical

    # Ricci control via jet (for comparison — shows the bug)
    R_ds = riemann_ds(H_FIXED)
    R_ric = ricci_built_part(R_ds)  # = R_ds for dS
    parJ, chJ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_ric))
    YRic_jet = Y_from_graph(parJ, chJ)

    mask_read = bulk_mask(pts, T, ZETA)
    mask_proj = bulk_mask(pts, T, ZETA / 2.0)
    k0 = excess_kurtosis(Y0[mask_read])

    dk_raw = excess_kurtosis(YdS[mask_read]) - k0

    # FIXED subtraction: exact - exact = 0
    delta_weyl_fixed = YdS - YRic_exact  # should be exactly 0
    Pi0_fixed = project_l0_grid(delta_weyl_fixed, pts, T, mask_proj)
    delta_W_fixed = delta_weyl_fixed - Pi0_fixed
    dk_weyl_fixed = excess_kurtosis((Y0 + delta_W_fixed)[mask_read]) - k0

    # BUGGY subtraction: exact - jet ≠ 0 (reproduces old bug)
    delta_weyl_buggy = YdS - YRic_jet
    Pi0_buggy = project_l0_grid(delta_weyl_buggy, pts, T, mask_proj)
    delta_W_buggy = delta_weyl_buggy - Pi0_buggy
    dk_weyl_buggy = excess_kurtosis((Y0 + delta_W_buggy)[mask_read]) - k0

    # Also: jet - jet (same approx → should be ~0)
    parJ2, chJ2 = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_ds))
    YdS_jet = Y_from_graph(parJ2, chJ2)
    delta_weyl_jj = YdS_jet - YRic_jet  # jet - jet = ~0
    dk_weyl_jj = excess_kurtosis((Y0 + delta_weyl_jj)[mask_read]) - k0

    return (T, si, dk_raw, dk_weyl_fixed, dk_weyl_buggy, dk_weyl_jj)


if __name__ == "__main__":
    print(f"=== dS RICCI SUBTRACTION FIX: H={H_FIXED}, R={12*H_FIXED**2:.2f}, N={N}, M={M_SEEDS} ===")
    print()

    tasks = [(T, si) for T in T_VALS for si in range(M_SEEDS)]
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        results_raw = list(ex.map(run_one, tasks))
    elapsed = time.time() - t0

    results = {}
    for T in T_VALS:
        rows = [(r, wf, wb, wj) for t, si, r, wf, wb, wj in results_raw if t == T]
        raw = np.array([x[0] for x in rows])
        wf = np.array([x[1] for x in rows])
        wb = np.array([x[2] for x in rows])
        wj = np.array([x[3] for x in rows])

        def stats(a):
            m = float(np.mean(a))
            se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
            d = float(m / np.std(a, ddof=1)) if np.std(a, ddof=1) > 0 else 0.0
            return m, se, d

        mr, ser, dr = stats(raw)
        mf, sef, df = stats(wf)
        mb, seb, db = stats(wb)
        mj, sej, dj = stats(wj)

        results[T] = {
            "raw": {"mean": mr, "se": ser, "d": dr},
            "weyl_fixed": {"mean": mf, "se": sef, "d": df},
            "weyl_buggy": {"mean": mb, "se": seb, "d": db},
            "weyl_jet_jet": {"mean": mj, "se": sej, "d": dj},
        }

        sig_r = "***" if abs(dr) > 3 else "**" if abs(dr) > 2 else "*" if abs(dr) > 1 else ""
        sig_f = "***" if abs(df) > 3 else "**" if abs(df) > 2 else "*" if abs(df) > 1 else ""
        sig_b = "***" if abs(db) > 3 else "**" if abs(db) > 2 else "*" if abs(db) > 1 else ""
        print(f"T={T:.2f}:")
        print(f"  raw:          dk={mr:+.6f}±{ser:.6f} d={dr:+.3f} {sig_r}")
        print(f"  weyl(FIXED):  dk={mf:+.6f}±{sef:.6f} d={df:+.3f} {sig_f}  ← exact-exact=0?")
        print(f"  weyl(BUGGY):  dk={mb:+.6f}±{seb:.6f} d={db:+.3f} {sig_b}  ← exact-jet≠0")
        print(f"  weyl(jet-jet): dk={mj:+.6f}±{sej:.6f} d={dj:+.3f}       ← jet-jet≈0?")
        print()

    print(f"Total: {elapsed:.0f}s = {elapsed/60:.1f}min")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ds_subtraction_fixed.json"), "w") as f:
        json.dump({"H": H_FIXED, "N": N, "M_seeds": M_SEEDS,
                   "results": {str(k): v for k, v in results.items()}, "elapsed_s": elapsed}, f, indent=2)
    print("Saved.")
