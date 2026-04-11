"""
FND-1 EXP-6: d=4 Higher N — Push to N=10000.

Extends EXP-1 to larger causal sets where finite-size effects are smaller
and the link-graph Fiedler signal should be stronger. Tests whether the
GENUINE d=4 curvature signal grows with N.

Designed for Verda H100 (32 CPU cores, 185 GB RAM).
Uses numpy (CPU) — no CuPy dependency for portability.
Each task needs ~3 GB RAM at N=10000 (C + n_matrix + L_dense).

Run:
  python analysis/scripts/fnd1_exp6_d4_higher_N.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat,
    causal_matrix_4d,
    compute_layers_4d,
    bd_action_4d,
    build_link_graph,
    link_spectral_embedding,
    _ppwave_profile,
)
from fnd1_4d_followup import quadrupole_profile
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [3000, 5000, 8000, 10000]
N_PRIMARY = 10000
M_ENSEMBLE = 30          # smaller M due to O(N^3) cost per task
T_DIAMOND = 1.0
MASTER_SEED = 1234
# Memory per worker: ~15 GB at N=10000 (11 NxN arrays in causal_matrix_4d + eigendecomp).
# N_WORKERS is already adaptive to system size. Cap per-N in main loop.
WORKERS = N_WORKERS

# PRIMARY: quadrupole x^2-y^2 (pure Weyl, zero mean, TC-stable at eps<=10)
EPS_QUADRUPOLE = [0.0, 2.0, 5.0, 10.0]
# SECONDARY: coscosh (has monopole contamination, for comparison with prior data)
EPS_COSCOSH = [0.0, 0.1, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph Fiedler for one d=4 sprinkling at high N."""
    seed_int, N, T, eps, profile = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    if abs(eps) < 1e-12:
        C = causal_matrix_4d(pts, 0.0, "flat")
    elif profile == "quadrupole":
        # Inline quadrupole causal matrix (pure Weyl, zero mean)
        t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dz = z[np.newaxis, :] - z[:, np.newaxis]
        dr2 = dx ** 2 + dy ** 2 + dz ** 2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        f_mid = quadrupole_profile(xm, ym)
        mink = dt ** 2 - dr2
        corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
        del dx, dy, xm, ym, f_mid, mink, corr  # free memory at high N
    else:
        C = causal_matrix_4d(pts, eps, "tidal")  # coscosh

    total_causal = float(np.sum(C))
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Link graph + Fiedler (dense eigendecomp, k=20 for d=4 per EXP-1)
    A_link = build_link_graph(C, n_matrix)
    mean_link_deg = float(A_link.sum() / N)

    embedding, link_lams = link_spectral_embedding(A_link, 20, N)
    fiedler = float(link_lams[0]) if len(link_lams) > 0 else 0.0

    # Geometry recovery (best-k from EXP-1: k=20 optimal in d=4)
    rng2 = np.random.default_rng(seed_int + 999)
    n_pairs = min(8000, N * 2)
    idx_i = rng2.integers(0, N, size=n_pairs)
    idx_j = rng2.integers(0, N, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    d_eucl = np.sqrt(np.sum((pts[idx_i] - pts[idx_j]) ** 2, axis=1))

    # Test k=2 and k=20
    best_rho = 0.0
    for k in [2, 20]:
        k_eff = min(k, embedding.shape[1])
        if k_eff > 0:
            emb = embedding[:, :k_eff]
            d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j]) ** 2, axis=1))
            rho_k, _ = stats.spearmanr(d_emb, d_eucl)
            if abs(rho_k) > abs(best_rho):
                best_rho = rho_k

    return {
        "fiedler": fiedler,
        "rho_spearman": float(best_rho),
        "total_causal": total_causal,
        "bd_action": bd,
        "mean_link_deg": mean_link_deg,
        "n_links": N0,
        "eps": eps,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp6_d4_higher_N",
        description="d=4 higher N: link Fiedler scaling to N=10000",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-6: d=4 HIGHER N (quadrupole + coscosh)", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Quadrupole eps: {EPS_QUADRUPOLE} (PRIMARY)", flush=True)
    print(f"Coscosh eps: {EPS_COSCOSH} (secondary)", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0, "quadrupole"))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * (len(EPS_QUADRUPOLE) + len(EPS_COSCOSH))
        par = tasks * dt / WORKERS
        print(f"  N={N:6d}: {dt:.1f}s/task, parallel({WORKERS}w): {par / 60:.1f} min",
              flush=True)

    # Profile configurations: (name, eps_list)
    PROFILES = [
        ("quadrupole", EPS_QUADRUPOLE),
        ("coscosh", EPS_COSCOSH),
    ]

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        # Per-N worker cap: ~15 GB at N=10000 (11 NxN arrays + eigendecomp)
        mem_per_worker_gb = max(0.5, 15.0 * (N / 10000) ** 2)
        try:
            import os as _os
            total_ram_gb = _os.sysconf('SC_PAGE_SIZE') * _os.sysconf('SC_PHYS_PAGES') / 1e9
        except (AttributeError, ValueError):
            total_ram_gb = 64.0  # conservative fallback
        workers_N = min(WORKERS, max(1, int(total_ram_gb * 0.7 / mem_per_worker_gb)))

        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N} (workers={workers_N})", flush=True)
        print("=" * 60, flush=True)

        all_results = []
        for prof_name, eps_list in PROFILES:
            print(f"  Profile: {prof_name}", flush=True)
            for eps in eps_list:
                eps_ss = ss.spawn(1)[0]
                seeds = eps_ss.spawn(M_ENSEMBLE)
                seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
                args = [(si, N, T_DIAMOND, eps, prof_name) for si in seed_ints]

                t0 = time.perf_counter()
                with Pool(workers_N, initializer=_init_worker) as pool:
                    raw = pool.map(_worker, args)
                elapsed = time.perf_counter() - t0

                fiedlers = [r["fiedler"] for r in raw]
                rhos = [r["rho_spearman"] for r in raw]
            print(f"  eps={eps:+.2f}: fiedler={np.mean(fiedlers):.4f}+-{np.std(fiedlers):.4f}"
                  f"  rho_S={np.mean(rhos):+.4f}"
                  f"  [{elapsed:.1f}s]", flush=True)
            all_results.extend(raw)

        # Separate by profile for clean analysis
        quad_results = [r for r in all_results if r["profile"] == "quadrupole"]
        cosc_results = [r for r in all_results if r["profile"] == "coscosh"]

        # PRIMARY analysis: quadrupole only (pure Weyl, no monopole contamination)
        flat_f = [r["fiedler"] for r in quad_results if abs(r["eps"]) < 1e-10]
        curv_f = [r["fiedler"] for r in quad_results if abs(r["eps"]) > 1e-10]
        if flat_f and curv_f:
            t_stat, p_val = stats.ttest_ind(flat_f, curv_f)
            delta = np.mean(curv_f) - np.mean(flat_f)
            d_cohen = delta / np.std(flat_f) if np.std(flat_f) > 0 else 0.0
        else:
            p_val = d_cohen = delta = 0.0

        # Mediation: partial r(eps, fiedler | TC + TC^2 + BD) — quadrupole only
        eps_arr = np.array([r["eps"] for r in quad_results])
        fiedler_arr = np.array([r["fiedler"] for r in quad_results])
        tc_arr = np.array([r["total_causal"] for r in quad_results])
        bd_arr = np.array([r["bd_action"] for r in quad_results])
        controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
        from numpy.linalg import lstsq
        cx, _, _, _ = lstsq(controls, eps_arr, rcond=None)
        cy, _, _, _ = lstsq(controls, fiedler_arr, rcond=None)
        rx = eps_arr - controls @ cx
        ry = fiedler_arr - controls @ cy
        if np.std(rx) > 1e-15 and np.std(ry) > 1e-15:
            r_partial, p_partial = stats.pearsonr(rx, ry)
        else:
            r_partial = p_partial = 0.0
        r_partial = float(r_partial)
        p_partial = float(p_partial)

        # Geometry at flat
        rho_flat = float(np.mean([r["rho_spearman"] for r in all_results
                                   if abs(r["eps"]) < 1e-10]))

        results_by_N[str(N)] = {
            "fiedler_flat": float(np.mean(flat_f)) if flat_f else 0.0,
            "fiedler_curved": float(np.mean(curv_f)) if curv_f else 0.0,
            "fiedler_delta": float(delta),
            "fiedler_cohen_d": float(d_cohen),
            "fiedler_p": float(p_val),
            "fiedler_r_partial": r_partial,
            "fiedler_p_partial": p_partial,
            "rho_flat": rho_flat,
            "mean_link_deg": float(np.mean([r["mean_link_deg"] for r in all_results])),
        }

        print(f"  Fiedler delta: {delta:+.4f} (d={d_cohen:+.2f}, p={p_val:.2e})",
              flush=True)
        print(f"  Mediation: r_partial={r_partial:+.4f}, p={p_partial:.2e}",
              flush=True)
        print(f"  Geometry (flat, best_k): rho={rho_flat:+.4f}", flush=True)

    # ==================================================================
    # SCALING
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'=' * 70}", flush=True)
    print("SCALING", flush=True)
    print("=" * 70, flush=True)

    Ns = np.array(N_VALUES, dtype=float)
    deltas = []
    cohens = []
    rho_flats = []

    for N in N_VALUES:
        r = results_by_N[str(N)]
        deltas.append(r["fiedler_delta"])
        cohens.append(r["fiedler_cohen_d"])
        rho_flats.append(r["rho_flat"])
        print(f"  N={N:6d}: delta={r['fiedler_delta']:+.4f}"
              f"  d={r['fiedler_cohen_d']:+.2f}"
              f"  p={r['fiedler_p']:.2e}"
              f"  rho={r['rho_flat']:+.4f}"
              f"  deg={r['mean_link_deg']:.1f}", flush=True)

    # Does effect size grow with N?
    if len(cohens) > 2:
        lr = stats.linregress(np.log(Ns), np.abs(cohens))
        print(f"\n  |Cohen d| vs log(N): slope={lr.slope:+.4f},"
              f" R^2={lr.rvalue**2:.4f}", flush=True)
        signal_grows = lr.slope > 0 and lr.rvalue ** 2 > 0.5
    else:
        signal_grows = False

    # Verdict
    r_best = results_by_N[str(N_PRIMARY)]
    rp = r_best["fiedler_r_partial"]
    pp = r_best["fiedler_p_partial"]
    cd = r_best["fiedler_cohen_d"]
    # Verdict requires BOTH t-test AND mediation (partial r survives TC+BD control)
    if abs(rp) > 0.10 and pp < 0.01 and abs(cd) > 0.5:
        verdict = (f"CONFIRMED at N={N_PRIMARY}: d={cd:+.2f},"
                   f" r_partial={rp:+.4f}, p={pp:.2e}")
    elif abs(rp) > 0.10 and pp < 0.05:
        verdict = (f"MARGINAL at N={N_PRIMARY}: r_partial={rp:+.4f},"
                   f" p={pp:.3f}, d={cd:+.2f}")
    elif abs(cd) > 0.5 and r_best["fiedler_p"] < 0.01:
        verdict = (f"MEDIATED at N={N_PRIMARY}: d={cd:+.2f} (t-test sig)"
                   f" but r_partial={rp:+.4f} (not sig after TC+BD)")
    else:
        verdict = (f"WEAK at N={N_PRIMARY}: d={cd:+.2f}, r_partial={rp:+.4f}")

    if signal_grows:
        verdict += " | SIGNAL GROWS WITH N"
    verdict += f" | Geometry rho(best_k)={rho_flats[-1]:+.4f}"

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "N_primary": N_PRIMARY,
            "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_quadrupole": EPS_QUADRUPOLE,
            "eps_coscosh": EPS_COSCOSH,
        },
        "results_by_N": results_by_N,
        "scaling": {
            "signal_grows": signal_grows,
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp6_d4_higher_N.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
