"""
FND-1 EXP-1: d=4 Link-Graph Verification.

Triple verification of the d4_followup GENUINE signal:
  link_fiedler correlates with pp-wave curvature after polynomial TC+BD mediation.

Tests:
  1. Reproducibility (5 N values, M=80 sprinklings each)
  2. Finite-size scaling: does partial r GROW with N?
  3. Conformal control: Weyl (tidal) vs density-only (conformal)
  4. Geometry recovery: Spearman rho(d_embed, d_eucl) in d=4
  5. Null model: shuffled coordinates
  6. Mediation: polynomial TC + TC^2 + BD

Also extends EXP-12 to d=4: does link-graph embedding recover 4D geometry?

Run:
  python analysis/scripts/fnd1_exp1_d4_link_verification.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat,
    sprinkle_4d_conformal,
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

N_VALUES = [500, 1000, 2000, 3000]
N_PRIMARY = 2000
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 991
WORKERS = N_WORKERS

K_VALUES_EMBED = [2, 5, 10, 20]  # d=4 needs more eigenvectors than d=2
K_MAX = 20                        # max eigenvectors for embedding
N_DISTANCE_PAIRS = 8000

# PP-wave coscosh profile (Weyl + monopole density change ~18% at eps=0.3)
EPS_COSCOSH = [0.0, 0.1, 0.2, 0.3, 0.5]

# PP-wave quadrupole profile x^2-y^2 (PURE Weyl, zero mean)
# TC-stable range: eps<=10 gives <3% TC change. eps=20 gives ~9%.
# Cap at eps=20 (consistent with EXP-2/3/4).
EPS_QUADRUPOLE = [0.0, 2.0, 5.0, 10.0, 20.0]

# Conformal control (density-only, flat causal structure)
EPS_CONFORMAL = [0.2, 0.5]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph observables for one d=4 sprinkling."""
    seed_int, N, T, eps, metric = args

    rng = np.random.default_rng(seed_int)

    if metric == "conformal" and abs(eps) > 1e-12:
        pts = sprinkle_4d_conformal(N, T, eps, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")
    elif metric in ("tidal", "quadrupole") and abs(eps) > 1e-12:
        pts = sprinkle_4d_flat(N, T, rng)
        # Build causal matrix with appropriate profile
        t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dz = z[np.newaxis, :] - z[:, np.newaxis]
        dr2 = dx ** 2 + dy ** 2 + dz ** 2
        mink_interval = dt ** 2 - dr2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        if metric == "quadrupole":
            f_mid = quadrupole_profile(xm, ym)
        else:
            f_mid = _ppwave_profile(xm, ym)
        ppwave_corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink_interval > ppwave_corr) & (dt > 0)).astype(np.float64)
    else:
        pts = sprinkle_4d_flat(N, T, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")

    total_causal = float(np.sum(C))

    # Layers + BD action
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Link graph
    A_link = build_link_graph(C, n_matrix)
    mean_link_deg = float(A_link.sum() / N)
    n_links = int(A_link.sum() / 2)

    # Spectral embedding (dense, robust) — compute max k, slice for each
    embedding, link_lams = link_spectral_embedding(A_link, K_MAX, N)
    fiedler = float(link_lams[0]) if len(link_lams) > 0 else 0.0

    # Distance pairs
    rng2 = np.random.default_rng(seed_int + 999)
    idx_i = rng2.integers(0, N, size=N_DISTANCE_PAIRS)
    idx_j = rng2.integers(0, N, size=N_DISTANCE_PAIRS)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    d_eucl = np.sqrt(np.sum((pts[idx_i] - pts[idx_j]) ** 2, axis=1))

    # Null model: shuffled coordinates
    rng_null = np.random.default_rng(seed_int + 55555)
    perm = rng_null.permutation(N)
    pts_shuf = pts[perm]
    d_eucl_shuf = np.sqrt(np.sum((pts_shuf[idx_i] - pts_shuf[idx_j]) ** 2, axis=1))

    # Test multiple k values (d=4 needs more eigenvectors than d=2)
    embed_by_k = {}
    for k in K_VALUES_EMBED:
        k_eff = min(k, embedding.shape[1])
        if k_eff > 0:
            emb = embedding[:, :k_eff]
            d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j]) ** 2, axis=1))
            rho_sp, _ = stats.spearmanr(d_emb, d_eucl)
            r_eucl_k, _ = stats.pearsonr(d_emb, d_eucl)
            r_null_k, _ = stats.pearsonr(d_emb, d_eucl_shuf)
        else:
            rho_sp = r_eucl_k = r_null_k = 0.0
        embed_by_k[k] = {
            "rho_spearman": float(rho_sp),
            "r_euclidean": float(r_eucl_k),
            "r_null": float(r_null_k),
        }

    # Best k (highest Spearman)
    best_k = max(K_VALUES_EMBED, key=lambda k: embed_by_k[k]["rho_spearman"])
    best_rho = embed_by_k[best_k]["rho_spearman"]

    return {
        "fiedler": fiedler,
        "embed_by_k": embed_by_k,
        "best_k": best_k,
        "rho_spearman": float(best_rho),
        "r_null": float(embed_by_k[best_k]["r_null"]),
        "total_causal": total_causal,
        "bd_action": bd,
        "mean_link_deg": mean_link_deg,
        "n_links": n_links,
        "eps": eps,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# Mediation analysis
# ---------------------------------------------------------------------------

def partial_correlation(x, y, controls):
    """Partial correlation between x and y, controlling for columns of controls."""
    from numpy.linalg import lstsq
    if controls.shape[1] == 0:
        return stats.pearsonr(x, y)
    # Residualize x and y
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx
    ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0, 1.0
    return stats.pearsonr(rx, ry)


def mediation_analysis(results, eps_values):
    """Polynomial mediation: partial r controlling TC + TC² + BD.

    Tests BOTH linear (eps) and quadratic (eps²) predictors because
    pp-wave Riemann is linear in eps, but Kretschner is quadratic.
    """
    eps_arr = np.array([r["eps"] for r in results])
    fiedler_arr = np.array([r["fiedler"] for r in results])
    tc_arr = np.array([r["total_causal"] for r in results])
    bd_arr = np.array([r["bd_action"] for r in results])
    rho_arr = np.array([r["rho_spearman"] for r in results])

    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(results))])

    out = {"n_samples": len(results)}

    # Test both linear and quadratic predictors
    for pred_name, predictor in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
        r_direct, p_direct = stats.pearsonr(predictor, fiedler_arr)
        r_partial, p_partial = partial_correlation(predictor, fiedler_arr, controls)
        out[f"fiedler_{pred_name}_r_direct"] = float(r_direct)
        out[f"fiedler_{pred_name}_p_direct"] = float(p_direct)
        out[f"fiedler_{pred_name}_r_partial"] = float(r_partial)
        out[f"fiedler_{pred_name}_p_partial"] = float(p_partial)

    # Best predictor (highest |partial r|)
    if abs(out["fiedler_linear_r_partial"]) >= abs(out["fiedler_quadratic_r_partial"]):
        out["best_predictor"] = "linear"
        out["fiedler_r_partial"] = out["fiedler_linear_r_partial"]
        out["fiedler_p_partial"] = out["fiedler_linear_p_partial"]
    else:
        out["best_predictor"] = "quadratic"
        out["fiedler_r_partial"] = out["fiedler_quadratic_r_partial"]
        out["fiedler_p_partial"] = out["fiedler_quadratic_p_partial"]

    return out


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------

def run_config(N, eps_values, metric, M, ss_parent, label):
    """Run M sprinklings per eps at one N and return results."""
    print(f"\n  {label}: N={N}, metric={metric}, {len(eps_values)} eps values",
          flush=True)

    all_results = []
    for eps in eps_values:
        eps_ss = ss_parent.spawn(1)[0]
        child_seeds = eps_ss.spawn(M)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N, T_DIAMOND, eps, metric) for si in seed_ints]

        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            raw = pool.map(_worker, args)
        elapsed = time.perf_counter() - t0

        # Summary
        fiedlers = [r["fiedler"] for r in raw]
        rhos = [r["rho_spearman"] for r in raw]
        nulls = [r["r_null"] for r in raw]
        print(f"    eps={eps:+.2f}: fiedler={np.mean(fiedlers):.4f}+-{np.std(fiedlers):.4f}"
              f"  rho_S={np.mean(rhos):+.4f}  null={np.mean(nulls):+.4f}"
              f"  [{elapsed:.1f}s]", flush=True)

        all_results.extend(raw)

    # Mediation analysis (any metric with >2 eps values)
    if len(eps_values) > 2:
        med = mediation_analysis(all_results, eps_values)
        print(f"    Mediation ({med['best_predictor']}): "
              f"r_partial={med['fiedler_r_partial']:+.4f}"
              f"  p={med['fiedler_p_partial']:.2e}"
              f"  (lin={med['fiedler_linear_r_partial']:+.3f},"
              f" quad={med['fiedler_quadratic_r_partial']:+.3f})", flush=True)
    else:
        med = None

    # Aggregate geometry recovery (flat eps=0 sprinklings only)
    flat_results = [r for r in all_results if abs(r["eps"]) < 1e-10]
    if flat_results:
        rho_mean = float(np.mean([r["rho_spearman"] for r in flat_results]))
        rho_sem = float(np.std([r["rho_spearman"] for r in flat_results], ddof=1)
                        / np.sqrt(len(flat_results)))
        null_mean = float(np.mean([r["r_null"] for r in flat_results]))
    else:
        rho_mean = rho_sem = null_mean = float("nan")

    return {
        "N": N,
        "metric": metric,
        "eps_values": eps_values,
        "n_sprinklings_total": len(all_results),
        "mediation": med,
        "geometry_flat": {
            "rho_spearman_mean": rho_mean,
            "rho_spearman_sem": rho_sem,
            "r_null_mean": null_mean,
        },
        "mean_link_deg": float(np.mean([r["mean_link_deg"] for r in all_results])),
        "mean_fiedler_flat": float(np.mean([r["fiedler"] for r in flat_results]))
        if flat_results else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp1_d4_link_verification",
        description="d=4 link-graph: verify GENUINE signal + geometry recovery",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-1: d=4 LINK-GRAPH VERIFICATION", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Coscosh eps: {EPS_COSCOSH}", flush=True)
    print(f"Quadrupole eps: {EPS_QUADRUPOLE} (pure Weyl, TC stable)", flush=True)
    print(f"Conformal eps: {EPS_CONFORMAL}", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK (3 seeds) ===", flush=True)
    for N in N_VALUES:
        times = []
        for seed in [100, 200, 300]:
            t0 = time.perf_counter()
            _worker((seed, N, T_DIAMOND, 0.0, "flat"))
            times.append(time.perf_counter() - t0)
        mean_t = np.mean(times)
        total_tasks = M_ENSEMBLE * (len(EPS_COSCOSH) + len(EPS_QUADRUPOLE) + len(EPS_CONFORMAL) + 1)
        par = total_tasks * mean_t / WORKERS
        print(f"  N={N:5d}: {mean_t:.3f}s/task, total tasks={total_tasks},"
              f" parallel: {par / 60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    all_configs = {}

    # --- COSCOSH (pp-wave, Weyl + monopole) at each N ---
    print(f"\n{'=' * 70}", flush=True)
    print("COSCOSH (PP-WAVE): Weyl + monopole density change", flush=True)
    print("=" * 70, flush=True)

    for N in N_VALUES:
        key = f"tidal_N{N}"
        all_configs[key] = run_config(N, EPS_COSCOSH, "tidal", M_ENSEMBLE, ss,
                                      f"Coscosh N={N}")

    # --- QUADRUPOLE (pure Weyl, zero mean, TC stable) at primary N ---
    print(f"\n{'=' * 70}", flush=True)
    print(f"QUADRUPOLE (PURE WEYL): TC-stable at N={N_PRIMARY}", flush=True)
    print("=" * 70, flush=True)

    all_configs["quadrupole"] = run_config(N_PRIMARY, EPS_QUADRUPOLE,
                                           "quadrupole", M_ENSEMBLE, ss,
                                           f"Quadrupole N={N_PRIMARY}")

    # --- CONFORMAL (density-only control) at primary N ---
    print(f"\n{'=' * 70}", flush=True)
    print(f"CONFORMAL CONTROL at N={N_PRIMARY}", flush=True)
    print("=" * 70, flush=True)

    all_configs["conformal"] = run_config(N_PRIMARY, [0.0] + EPS_CONFORMAL,
                                          "conformal", M_ENSEMBLE, ss,
                                          "Conformal control")

    # ==================================================================
    # SCALING ANALYSIS
    # ==================================================================

    print(f"\n{'=' * 70}", flush=True)
    print("SCALING: fiedler partial r vs N (tidal)", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'N':>6} {'r_lin_part':>10} {'r_best_part':>10} {'p_best':>10}"
          f" {'rho_flat':>10} {'null':>10} {'degree':>8}", flush=True)
    print(f"  {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}"
          f" {'-' * 10} {'-' * 10} {'-' * 8}", flush=True)

    partial_rs = []
    rho_flats = []
    for N in N_VALUES:
        c = all_configs[f"tidal_N{N}"]
        med = c["mediation"]
        g = c["geometry_flat"]
        if med:
            partial_rs.append(med["fiedler_r_partial"])
            print(f"  {N:6d} {med['fiedler_linear_r_partial']:+10.4f}"
                  f" {med['fiedler_r_partial']:+10.4f}"
                  f" {med['fiedler_p_partial']:10.2e}"
                  f" {g['rho_spearman_mean']:+10.4f}"
                  f" {g['r_null_mean']:+10.4f}"
                  f" {c['mean_link_deg']:8.2f}", flush=True)
        rho_flats.append(g["rho_spearman_mean"])

    # Quadrupole vs coscosh vs conformal comparison at N_PRIMARY
    print(f"\n  === Comparison at N={N_PRIMARY} ===", flush=True)
    for label, key in [("Coscosh (Weyl+monopole)", f"tidal_N{N_PRIMARY}"),
                       ("Quadrupole (pure Weyl)", "quadrupole"),
                       ("Conformal (density-only)", "conformal")]:
        c = all_configs.get(key, {})
        med = c.get("mediation")
        if med:
            print(f"  {label:30s}: r_partial={med['fiedler_r_partial']:+.4f}"
                  f" (p={med['fiedler_p_partial']:.3f},"
                  f" best={med['best_predictor']})", flush=True)
        else:
            print(f"  {label:30s}: no mediation", flush=True)

    # Geometry recovery scaling
    Ns = np.array(N_VALUES, dtype=float)
    rho_arr = np.array(rho_flats)
    if len(rho_arr) > 2:
        lr = stats.linregress(np.log(Ns), rho_arr)
        print(f"\n  Geometry rho(best_k) vs log(N): slope={lr.slope:+.4f},"
              f" R^2={lr.rvalue**2:.4f}", flush=True)

    # Is partial r growing with N?
    if len(partial_rs) > 2:
        lr_pr = stats.linregress(np.log(Ns[:len(partial_rs)]),
                                 np.abs(partial_rs))
        signal_grows = lr_pr.slope > 0 and lr_pr.rvalue ** 2 > 0.5
        print(f"  |partial_r| vs log(N): slope={lr_pr.slope:+.4f},"
              f" R^2={lr_pr.rvalue**2:.4f},"
              f" grows={'YES' if signal_grows else 'NO'}", flush=True)
    else:
        signal_grows = False

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total
    med_primary = all_configs[f"tidal_N{N_PRIMARY}"]["mediation"]

    if med_primary and abs(med_primary["fiedler_r_partial"]) > 0.10 and \
       med_primary["fiedler_p_partial"] < 0.01:
        curvature_verdict = (f"GENUINE: r_partial={med_primary['fiedler_r_partial']:+.4f},"
                             f" p={med_primary['fiedler_p_partial']:.2e}")
        if signal_grows:
            curvature_verdict += " | SIGNAL GROWS WITH N"
    elif med_primary and med_primary["fiedler_p_partial"] < 0.05:
        curvature_verdict = (f"MARGINAL: r_partial={med_primary['fiedler_r_partial']:+.4f},"
                             f" p={med_primary['fiedler_p_partial']:.3f}")
    else:
        curvature_verdict = (f"NOT CONFIRMED: r_partial="
                             f"{med_primary['fiedler_r_partial']:+.4f}" if med_primary
                             else "NO MEDIATION DATA")

    rho_best = rho_flats[-1] if rho_flats else 0.0
    if rho_best > 0.5:
        geo_verdict = f"GEOMETRY RECOVERED: rho(best_k)={rho_best:.4f}"
    elif rho_best > 0.3:
        geo_verdict = f"WEAK GEOMETRY: rho(best_k)={rho_best:.4f}"
    else:
        geo_verdict = f"NO GEOMETRY: rho(best_k)={rho_best:.4f}"

    verdict = f"Curvature: {curvature_verdict} | {geo_verdict}"

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    # Save
    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "N_primary": N_PRIMARY,
            "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_coscosh": EPS_COSCOSH, "eps_quadrupole": EPS_QUADRUPOLE,
            "eps_conformal": EPS_CONFORMAL,
            "k_values_embed": K_VALUES_EMBED, "k_max": K_MAX,
            "n_distance_pairs": N_DISTANCE_PAIRS,
        },
        "configs": all_configs,
        "scaling": {
            "N": N_VALUES,
            "partial_r": partial_rs,
            "rho_flat": rho_flats,
            "signal_grows": signal_grows,
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp1_d4_link_verification.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
