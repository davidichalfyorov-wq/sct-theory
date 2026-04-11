"""
Discovery Run 001 — Pilot Step 2: Forman-Ricci Curvature on Causal Sets
========================================================================

Tests whether Forman-Ricci curvature of the link graph (Hasse diagram)
is sensitive to spacetime curvature.

Forman-Ricci curvature for an edge e = (u,v) in an undirected graph:
    F(e) = 4 - deg(u) - deg(v)

This is the simplest version (Sreejith-Mohanraj-Jost-Saucan-Samal 2016).
The augmented version (with triangles) is:
    F_aug(e) = 4 - deg(u) - deg(v) + 3*T(e)
where T(e) = number of triangles containing edge e.

Saucan et al. (2018, arXiv:1809.07698) extended both to directed networks.
For a directed edge e = u -> v:
    F_dir(e) = w(u)/w(e) + w(v)/w(e)
             - sum_{e' || e, e' != e} w(e)/sqrt(w(e)*w(e'))
(with all weights = 1 for unweighted case)

For this pilot, we use the UNDIRECTED link graph (symmetric Hasse diagram)
since Ollivier-Ricci convergence (van der Hoorn+ 2020) is proven for
undirected graphs. We test both basic and augmented Forman-Ricci.

CRN protocol: same random seed, same points, different metric.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json
import time
import gc
import os

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 500          # causal set size
T = 1.0          # causal diamond half-size
M = 40           # CRN trials

GEOMETRIES = {
    "ppwave_quad":   {"eps": 10.0,  "seed_offset": 100},
    "schwarzschild": {"eps": 0.02,  "seed_offset": 300},
}

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")


# ---------------------------------------------------------------------------
# Sprinkling + Causal + Link graph (reused from gap_ratio pilot)
# ---------------------------------------------------------------------------
def sprinkle(N, T, rng):
    pts = np.empty((N, 4))
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count + n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


def causal_flat(pts):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    return ((dt**2 > dr2) & (dt > 0)).astype(np.float64)


def causal_ppwave_quad(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm**2 - ym**2
    mink = dt**2 - dr2
    corr = eps * f * (dt + dz)**2 / 2.0
    return ((mink > corr) & (dt > 0)).astype(np.float64)


def causal_schwarzschild(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    zm = (z[np.newaxis, :] + z[:, np.newaxis]) / 2.0
    rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3
    Phi = -eps / rm
    return (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.float64)


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "schwarzschild": causal_schwarzschild,
}


def build_link_graph(C):
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_intervening = (C2 != 0).astype(np.float64)
    link = C_sp - C_sp.multiply(has_intervening)
    link.eliminate_zeros()
    A = link + link.T
    A = (A > 0.5).astype(np.float64)
    return A.tocsr()


# ---------------------------------------------------------------------------
# Forman-Ricci curvature
# ---------------------------------------------------------------------------
def forman_ricci_basic(A_sp):
    """Basic Forman-Ricci curvature for each edge.

    F(e=(u,v)) = 4 - deg(u) - deg(v)

    Returns: array of F values for each edge, plus summary statistics.
    """
    A = A_sp.tocsr()
    degrees = np.array(A.sum(axis=1)).ravel()

    # Extract edges (upper triangle to avoid double-counting)
    rows, cols = sp.triu(A, k=1).nonzero()

    F_values = 4.0 - degrees[rows] - degrees[cols]

    return F_values, rows, cols, degrees


def forman_ricci_augmented(A_sp):
    """Augmented Forman-Ricci curvature (with triangle correction).

    F_aug(e=(u,v)) = 4 - deg(u) - deg(v) + 3*T(e)
    where T(e) = number of triangles containing edge e.

    Returns: array of F_aug values for each edge.
    """
    A = A_sp.tocsr()
    degrees = np.array(A.sum(axis=1)).ravel()

    # A^2 gives number of paths of length 2
    A2 = (A @ A).toarray()

    # Extract edges (upper triangle)
    rows, cols = sp.triu(A, k=1).nonzero()

    # T(e=(u,v)) = (A^2)[u,v] = number of common neighbors of u and v
    # (each triangle containing e contributes one common neighbor)
    triangles = np.array([A2[r, c] for r, c in zip(rows, cols)], dtype=np.float64)

    F_values = 4.0 - degrees[rows] - degrees[cols] + 3.0 * triangles

    return F_values, rows, cols, triangles


def forman_statistics(F_values):
    """Summary statistics of Forman-Ricci curvature distribution."""
    if len(F_values) == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan,
                "frac_positive": np.nan, "frac_negative": np.nan,
                "min": np.nan, "max": np.nan, "n_edges": 0}

    return {
        "mean": float(np.mean(F_values)),
        "std": float(np.std(F_values)),
        "median": float(np.median(F_values)),
        "frac_positive": float(np.mean(F_values > 0)),
        "frac_negative": float(np.mean(F_values < 0)),
        "min": float(np.min(F_values)),
        "max": float(np.max(F_values)),
        "q25": float(np.percentile(F_values, 25)),
        "q75": float(np.percentile(F_values, 75)),
        "skewness": float(stats.skew(F_values)),
        "kurtosis": float(stats.kurtosis(F_values)),
        "n_edges": len(F_values),
    }


# ---------------------------------------------------------------------------
# CRN single trial
# ---------------------------------------------------------------------------
def crn_trial(seed, N, T, metric_name, eps, seed_offset):
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle(N, T, rng)

    # Flat
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    del C_flat; gc.collect()

    F_basic_flat, _, _, deg_flat = forman_ricci_basic(A_flat)
    F_aug_flat, _, _, tri_flat = forman_ricci_augmented(A_flat)
    stats_basic_flat = forman_statistics(F_basic_flat)
    stats_aug_flat = forman_statistics(F_aug_flat)
    mean_degree_flat = float(np.mean(deg_flat))
    mean_tri_flat = float(np.mean(tri_flat))
    del A_flat; gc.collect()

    # Curved
    C_curv = METRIC_FNS[metric_name](pts, eps)
    A_curv = build_link_graph(C_curv)
    del C_curv; gc.collect()

    F_basic_curv, _, _, deg_curv = forman_ricci_basic(A_curv)
    F_aug_curv, _, _, tri_curv = forman_ricci_augmented(A_curv)
    stats_basic_curv = forman_statistics(F_basic_curv)
    stats_aug_curv = forman_statistics(F_aug_curv)
    mean_degree_curv = float(np.mean(deg_curv))
    mean_tri_curv = float(np.mean(tri_curv))
    del A_curv; gc.collect()

    return {
        "seed": seed,
        # Basic Forman-Ricci
        "F_basic_mean_flat": stats_basic_flat["mean"],
        "F_basic_mean_curv": stats_basic_curv["mean"],
        "F_basic_mean_delta": stats_basic_curv["mean"] - stats_basic_flat["mean"],
        "F_basic_std_flat": stats_basic_flat["std"],
        "F_basic_std_curv": stats_basic_curv["std"],
        "F_basic_frac_pos_flat": stats_basic_flat["frac_positive"],
        "F_basic_frac_pos_curv": stats_basic_curv["frac_positive"],
        # Augmented Forman-Ricci
        "F_aug_mean_flat": stats_aug_flat["mean"],
        "F_aug_mean_curv": stats_aug_curv["mean"],
        "F_aug_mean_delta": stats_aug_curv["mean"] - stats_aug_flat["mean"],
        "F_aug_std_flat": stats_aug_flat["std"],
        "F_aug_std_curv": stats_aug_curv["std"],
        "F_aug_median_flat": stats_aug_flat["median"],
        "F_aug_median_curv": stats_aug_curv["median"],
        "F_aug_skew_flat": stats_aug_flat["skewness"],
        "F_aug_skew_curv": stats_aug_curv["skewness"],
        "F_aug_frac_pos_flat": stats_aug_flat["frac_positive"],
        "F_aug_frac_pos_curv": stats_aug_curv["frac_positive"],
        # Graph statistics
        "mean_degree_flat": mean_degree_flat,
        "mean_degree_curv": mean_degree_curv,
        "mean_degree_delta": mean_degree_curv - mean_degree_flat,
        "mean_triangles_flat": mean_tri_flat,
        "mean_triangles_curv": mean_tri_curv,
        "n_edges_flat": stats_basic_flat["n_edges"],
        "n_edges_curv": stats_basic_curv["n_edges"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 70)
    print("Discovery Run 001 — Pilot Step 2: Forman-Ricci Curvature")
    print(f"N={N}, M={M}, geometries: {list(GEOMETRIES.keys())}")
    print("=" * 70)

    all_results = {}

    for geo_name, geo_params in GEOMETRIES.items():
        eps = geo_params["eps"]
        seed_off = geo_params["seed_offset"]
        print(f"\n--- {geo_name} (eps={eps}) ---")

        results = []
        t0 = time.time()

        for trial in range(M):
            seed = trial * 1000
            res = crn_trial(seed, N, T, geo_name, eps, seed_off)
            results.append(res)

            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                # Basic
                d_basic = [r["F_basic_mean_delta"] for r in results]
                d_aug = [r["F_aug_mean_delta"] for r in results]
                d_deg = [r["mean_degree_delta"] for r in results]
                print(f"  trial {trial+1}/{M}: "
                      f"F_basic_delta={np.mean(d_basic):+.3f}, "
                      f"F_aug_delta={np.mean(d_aug):+.3f}, "
                      f"deg_delta={np.mean(d_deg):+.3f}  "
                      f"[{elapsed:.1f}s]")

        elapsed = time.time() - t0

        # === Aggregate ===
        d_basic = [r["F_basic_mean_delta"] for r in results]
        d_aug = [r["F_aug_mean_delta"] for r in results]
        d_deg = [r["mean_degree_delta"] for r in results]
        d_tri = [r["mean_triangles_curv"] - r["mean_triangles_flat"] for r in results]
        d_frac_pos = [r["F_aug_frac_pos_curv"] - r["F_aug_frac_pos_flat"] for r in results]

        print(f"\n  RESULTS ({geo_name}, M={M}, {elapsed:.1f}s):")

        # Report each observable
        for name, deltas in [
            ("F_basic_mean", d_basic),
            ("F_aug_mean", d_aug),
            ("mean_degree", d_deg),
            ("mean_triangles", d_tri),
            ("F_aug_frac_positive", d_frac_pos),
        ]:
            mean_d = np.mean(deltas)
            se_d = np.std(deltas) / np.sqrt(len(deltas))
            cohen = mean_d / np.std(deltas) if np.std(deltas) > 0 else 0.0
            t_stat, p_val = stats.ttest_1samp(deltas, 0.0) if len(deltas) >= 5 else (0, 1)
            try:
                _, w_p = stats.wilcoxon(deltas, alternative='two-sided')
            except ValueError:
                w_p = 1.0

            if p_val < 0.001 and abs(cohen) > 0.5:
                v = "★★★"
            elif p_val < 0.01:
                v = "★★"
            elif p_val < 0.05:
                v = "★"
            else:
                v = "null"

            print(f"    {name:25s}: delta={mean_d:+.4f} +/- {se_d:.4f}, "
                  f"d={cohen:+.3f}, p={p_val:.2e} ({v})")

        # Also print absolute values for context
        fb_flat = np.mean([r["F_basic_mean_flat"] for r in results])
        fa_flat = np.mean([r["F_aug_mean_flat"] for r in results])
        deg_flat = np.mean([r["mean_degree_flat"] for r in results])
        tri_flat = np.mean([r["mean_triangles_flat"] for r in results])
        print(f"\n    Absolute values (flat): F_basic={fb_flat:.2f}, F_aug={fa_flat:.2f}, "
              f"deg={deg_flat:.1f}, tri={tri_flat:.2f}")

        # Compute residual after degree correction
        # F_basic = 4 - deg_u - deg_v ~ 4 - 2*mean_deg
        # If delta(F_basic) ~ -2*delta(mean_deg), then F is just degree proxy
        predicted_F_basic_delta = [-2 * d for d in d_deg]
        residual_F = [d_basic[i] - predicted_F_basic_delta[i] for i in range(len(d_basic))]
        mean_resid = np.mean(residual_F)
        se_resid = np.std(residual_F) / np.sqrt(len(residual_F))
        print(f"\n    Degree-corrected F_basic residual: {mean_resid:+.4f} +/- {se_resid:.4f}")
        print(f"    (If ~0, then F_basic is PURE degree proxy)")

        summary = {
            "geometry": geo_name, "eps": eps, "N": N, "M": M,
            "elapsed_sec": elapsed,
            "F_basic_mean_delta": {"mean": float(np.mean(d_basic)),
                                   "se": float(np.std(d_basic)/np.sqrt(M))},
            "F_aug_mean_delta": {"mean": float(np.mean(d_aug)),
                                 "se": float(np.std(d_aug)/np.sqrt(M))},
            "degree_delta": {"mean": float(np.mean(d_deg)),
                             "se": float(np.std(d_deg)/np.sqrt(M))},
            "degree_corrected_residual": {"mean": float(mean_resid),
                                          "se": float(se_resid)},
            "trials": results,
        }
        all_results[geo_name] = summary

    outpath = os.path.join(OUTDIR, "pilot_forman_ricci.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    print("\n" + "=" * 70)
    print("KEY QUESTION: Is Forman-Ricci MORE than just a degree proxy?")
    print("=" * 70)
    for geo, res in all_results.items():
        r = res["degree_corrected_residual"]
        print(f"  {geo:20s}: residual = {r['mean']:+.4f} +/- {r['se']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
