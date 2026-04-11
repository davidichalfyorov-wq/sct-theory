#!/usr/bin/env python3
"""FND-1 COMPREHENSIVE DATA COLLECTION.

Builds ALL Hasse diagrams for ALL tests from independent analysis test suite.
Saves per-element data for post-processing analysis.

Tests covered by this data:
  0.1  Reproducibility (new seeds)
  1.1  ppw A_align T⁴ scaling (4 T-points, M=30)
  1.2  Sch A_align T⁴ scaling (3 T-points)
  3.2  Nonlinear TC mediation (links, depth per seed)
  3.3  Cross-seed CRN decoupling (per-element X, δY)
  3.4  Flat-noise / random-strata null
  3.5  Independent sprinkling control
  4.1  Same-predicate ratio (jet ppw vs jet Sch)
  6.2  ζ/window robustness (recompute mask at different ζ)
  7.1  Observable (p,q) family

Estimated runtime: ~10-14 hours (N=10000, M=30, 4 T-values).
"""
import sys, os, time, json, pickle, gzip
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, ppwave_exact_preds, jet_preds, ds_preds,
    bulk_mask, riemann_schwarzschild_local, riemann_vacuum_from_E,
)
from schwarzschild_exact_local_tools import (
    map_rnc_to_schwarzschild_expmap,
    schwarzschild_exact_midpoint_preds_from_mapped
)

# ═══════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════
N = 10000
M_SEEDS = 30
SEED_BASE = 4000000  # non-overlapping with killer_tests (3100000+)

EPS = 3.0
M_SCH = 0.05
R0 = 0.50
H_DS = 0.30
ZETA = 0.15

T_VALUES = [1.0, 0.70, 0.50, 0.35]

# Build plan per T-value.
# T=1.0: full suite (7 builds) + independent flat for TEST 3.5
# T=0.7, 0.5: ppw + Sch (4 builds)
# T=0.35: ppw only (2 builds)
BUILD_PLAN = {
    1.0:  ["flat", "ppw_exact", "ppw_jet", "sch_jet", "sch_expmap", "ds_exact", "indep_flat"],
    0.70: ["flat", "ppw_exact", "sch_jet", "sch_expmap"],
    0.50: ["flat", "ppw_exact", "sch_jet", "sch_expmap"],
    0.35: ["flat", "ppw_exact"],
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
os.makedirs(OUT_DIR, exist_ok=True)


def make_strata(pts, parents0, T, depth_bins=3, tau_bins=5, rho_bins=3):
    """Standard 5τ × 3ρ × 3depth = 45 strata."""
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * tau_bins / 2.0).astype(int), 0, tau_bins - 1)
    rho_bin = np.clip(np.floor(rho_hat * rho_bins).astype(int), 0, rho_bins - 1)

    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i].size > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * depth_bins) // (max_d + 1), 0, depth_bins - 1)

    return tau_bin * (rho_bins * depth_bins) + rho_bin * depth_bins + depth_terc, depth


def count_links(parents):
    """Total link count = sum of |parents_i| over all elements."""
    return sum(p.size for p in parents)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("FND-1 COMPREHENSIVE DATA COLLECTION", flush=True)
    print(f"N={N}, M={M_SEEDS}, T={T_VALUES}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    t_total = time.time()

    # Total builds for ETA
    total_builds = sum(len(BUILD_PLAN[T]) * M_SEEDS for T in T_VALUES)
    builds_done = 0

    # Precompute Riemann tensors (constant across seeds)
    R_SCH = riemann_schwarzschild_local(M_SCH, R0)
    E_ppw_std = (EPS / 2.0) * np.diag([-1.0, 1.0, 0.0])
    R_PPW_JET = riemann_vacuum_from_E(E_ppw_std)

    summary = {
        "params": {
            "N": N, "M_SEEDS": M_SEEDS, "SEED_BASE": SEED_BASE,
            "EPS": EPS, "M_SCH": M_SCH, "R0": R0, "H_DS": H_DS, "ZETA": ZETA,
            "T_VALUES": T_VALUES,
            "E2_PPW": EPS**2 / 2.0,
            "E2_SCH": 6.0 * (M_SCH / R0**3)**2,
        },
        "total_builds": total_builds,
    }

    for T in T_VALUES:
        builds = BUILD_PLAN[T]
        t_label = f"T{T:.2f}"
        print(f"\n{'─'*60}", flush=True)
        print(f"T = {T}  ({len(builds)} builds × {M_SEEDS} seeds = {len(builds)*M_SEEDS})", flush=True)
        print(f"{'─'*60}", flush=True)

        for si in range(M_SEEDS):
            seed = SEED_BASE + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            t_seed = time.time()

            data = {"seed": seed, "T": T, "N": N}

            # ── FLAT ──
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)
            strata, depth0 = make_strata(pts, par0, T)
            links0 = count_links(par0)
            builds_done += 1

            data["pts"] = pts
            data["Y0"] = Y0
            data["mask"] = mask
            data["strata"] = strata
            data["depth0"] = depth0
            data["links0"] = links0

            # ── PPW EXACT ──
            if "ppw_exact" in builds:
                parP, chP = build_hasse_from_predicate(
                    pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
                YP = Y_from_graph(parP, chP)
                data["delta_ppw"] = YP - Y0
                data["links_ppw"] = count_links(parP)
                builds_done += 1

            # ── PPW JET (vacuum_from_E, for TEST 4.1 same-predicate) ──
            if "ppw_jet" in builds:
                parJ, chJ = build_hasse_from_predicate(
                    pts, lambda P, i: jet_preds(P, i, R_abcd=R_PPW_JET))
                YJ = Y_from_graph(parJ, chJ)
                data["delta_ppw_jet"] = YJ - Y0
                builds_done += 1

            # ── SCH JET (for TEST 4.1 same-predicate comparison) ──
            if "sch_jet" in builds:
                parS, chS = build_hasse_from_predicate(
                    pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
                YS = Y_from_graph(parS, chS)
                data["delta_sch_jet"] = YS - Y0
                data["links_sch_jet"] = count_links(parS)
                builds_done += 1

            # ── SCH EXPMAP MIDPOINT ──
            if "sch_expmap" in builds:
                mapped = map_rnc_to_schwarzschild_expmap(pts, M_SCH, R0)
                parX, chX = build_hasse_from_predicate(
                    pts, lambda P, i: schwarzschild_exact_midpoint_preds_from_mapped(
                        mapped, i, M_SCH))
                YX = Y_from_graph(parX, chX)
                data["delta_sch_exp"] = YX - Y0
                builds_done += 1

            # ── dS EXACT (for Ricci null verification) ──
            if "ds_exact" in builds:
                parD, chD = build_hasse_from_predicate(
                    pts, lambda P, i: ds_preds(P, i, H=H_DS))
                YD = Y_from_graph(parD, chD)
                data["delta_ds"] = YD - Y0
                builds_done += 1

            # ── INDEPENDENT FLAT (for TEST 3.5) ──
            if "indep_flat" in builds:
                rng2 = np.random.default_rng(seed + 90000000)
                pts2 = sprinkle_local_diamond(N, T, rng2)
                par2, ch2 = build_hasse_from_predicate(
                    pts2, lambda P, i: minkowski_preds(P, i))
                Y0_indep = Y_from_graph(par2, ch2)
                data["Y0_indep"] = Y0_indep
                data["pts_indep"] = pts2
                builds_done += 1

            # Save compressed
            fname = os.path.join(OUT_DIR, f"{t_label}_seed{si:03d}.pkl.gz")
            with gzip.open(fname, "wb") as f:
                pickle.dump(data, f, protocol=4)

            if (si + 1) % 5 == 0:
                elapsed = time.time() - t_total
                rate = builds_done / elapsed  # builds/sec
                remaining_builds = total_builds - builds_done
                eta_s = remaining_builds / rate if rate > 0 else 0
                print(f"  [{t_label}] {si+1}/{M_SEEDS}  "
                      f"({builds_done}/{total_builds} builds, "
                      f"{elapsed/60:.0f}min elapsed, "
                      f"ETA {eta_s/60:.0f}min)", flush=True)

    total_time = time.time() - t_total
    summary["total_time_s"] = total_time
    summary["total_time_min"] = total_time / 60
    summary["total_time_hours"] = total_time / 3600

    with open(os.path.join(OUT_DIR, "collection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"COMPLETE: {total_time/3600:.1f}h ({builds_done} builds)", flush=True)
    print(f"{'='*70}", flush=True)
