"""
Discovery Run 001 — Pilot Step 1: Spectral Gap Ratio <r>
=========================================================

Tests whether the spectral gap ratio statistic of the link-graph
Laplacian is sensitive to spacetime curvature.

The spectral gap ratio is:
    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    <r> = mean(r_i)
where s_i = lambda_{i+1} - lambda_i are consecutive eigenvalue spacings.

RMT predictions:
    Poisson (no correlations):  <r> ~ 0.386
    GOE (time-reversal inv.):   <r> ~ 0.536
    GUE (no time-reversal):     <r> ~ 0.603

Hypothesis: curved spacetime changes the RMT class of the link-graph
Laplacian spectrum, manifesting as a shift in <r>.

CRN protocol: same random seed, same points, different metric.
Any difference in <r> is caused by the metric change alone.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from scipy import stats
import json
import time
import gc
import os
import sys

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 500          # causal set size (dense fallback OK at this size)
T = 1.0          # causal diamond half-size
M = 40           # CRN trials per geometry pair
K_EIGSH = None   # None = full dense spectrum (N=500 is feasible)

# Geometries to test
GEOMETRIES = {
    "ppwave_quad":   {"eps": 10.0,  "seed_offset": 100},
    "schwarzschild": {"eps": 0.02, "seed_offset": 300},
}

# Reference values
R_POISSON = 0.38629  # 2 ln 2 - 1
R_GOE     = 0.53590
R_GUE     = 0.60266

# Output
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")

# ---------------------------------------------------------------------------
# Sprinkling (from Route 2)
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


# ---------------------------------------------------------------------------
# Causal conditions (from Route 2)
# ---------------------------------------------------------------------------
def causal_flat(pts):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    C = ((dt**2 > dr2) & (dt > 0)).astype(np.float64)
    return C


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
    C = ((mink > corr) & (dt > 0)).astype(np.float64)
    return C


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
    C = (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.float64)
    return C


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "schwarzschild": causal_schwarzschild,
}


# ---------------------------------------------------------------------------
# Link graph + Laplacian (from Route 2)
# ---------------------------------------------------------------------------
def build_link_graph(C):
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_intervening = (C2 != 0).astype(np.float64)
    link = C_sp - C_sp.multiply(has_intervening)
    link.eliminate_zeros()
    A = link + link.T
    A = (A > 0.5).astype(np.float64)
    return A.tocsr()


def build_laplacian(A_sp):
    degrees = np.array(A_sp.sum(axis=1)).ravel()
    D = sp.diags(degrees)
    return (D - A_sp).tocsr()


# ---------------------------------------------------------------------------
# Spectral gap ratio
# ---------------------------------------------------------------------------
def gap_ratio(evals):
    """Compute mean gap ratio <r> from sorted eigenvalues.

    Excludes zero eigenvalues (graph Laplacian always has >= 1).
    Uses bulk of spectrum (middle 80%) to avoid edge effects.
    """
    # Keep only nonzero eigenvalues
    ev = evals[evals > 1e-8]
    if len(ev) < 10:
        return np.nan, np.nan, 0

    # Spacings
    spacings = np.diff(ev)

    # Remove zero spacings (degeneracies)
    spacings = spacings[spacings > 1e-12]
    if len(spacings) < 5:
        return np.nan, np.nan, 0

    # Trim edges (10% from each side)
    n = len(spacings)
    lo, hi = n // 10, n - n // 10
    s = spacings[lo:hi]

    if len(s) < 3:
        return np.nan, np.nan, 0

    # Gap ratios
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])

    return float(np.mean(r)), float(np.std(r) / np.sqrt(len(r))), len(r)


def full_spectrum(L):
    """Full eigenvalue decomposition (dense, for small N)."""
    L_dense = L.toarray() if sp.issparse(L) else L
    return eigvalsh(L_dense)


# ---------------------------------------------------------------------------
# CRN single trial
# ---------------------------------------------------------------------------
def crn_trial(seed, N, T, metric_name, eps, seed_offset):
    """One CRN trial: sprinkle, build flat + curved, compute <r> for both."""
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle(N, T, rng)

    # Flat
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    del C_flat; gc.collect()
    L_flat = build_laplacian(A_flat)
    evals_flat = full_spectrum(L_flat)
    del A_flat, L_flat; gc.collect()

    # Curved
    C_curv = METRIC_FNS[metric_name](pts, eps)
    A_curv = build_link_graph(C_curv)
    del C_curv; gc.collect()
    L_curv = build_laplacian(A_curv)
    evals_curv = full_spectrum(L_curv)
    del A_curv, L_curv; gc.collect()

    # Gap ratios
    r_flat, r_flat_se, n_flat = gap_ratio(evals_flat)
    r_curv, r_curv_se, n_curv = gap_ratio(evals_curv)

    # Also save some spectral summary stats
    nz_flat = evals_flat[evals_flat > 1e-8]
    nz_curv = evals_curv[evals_curv > 1e-8]

    return {
        "seed": seed,
        "r_flat": r_flat,
        "r_curved": r_curv,
        "r_delta": r_curv - r_flat if not (np.isnan(r_curv) or np.isnan(r_flat)) else np.nan,
        "n_ratios_flat": n_flat,
        "n_ratios_curved": n_curv,
        "fiedler_flat": float(nz_flat[0]) if len(nz_flat) > 0 else 0.0,
        "fiedler_curved": float(nz_curv[0]) if len(nz_curv) > 0 else 0.0,
        "mean_degree_flat": float(np.sum(evals_flat > 1e-8)),  # approx
        "n_evals_total": len(evals_flat),
        # Save first 50 eigenvalues for debugging
        "evals_flat_50": evals_flat[:50].tolist(),
        "evals_curv_50": evals_curv[:50].tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 70)
    print("Discovery Run 001 — Pilot Step 1: Spectral Gap Ratio <r>")
    print(f"N={N}, M={M}, geometries: {list(GEOMETRIES.keys())}")
    print(f"RMT references: Poisson={R_POISSON:.4f}, GOE={R_GOE:.4f}, GUE={R_GUE:.4f}")
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
                r_deltas = [r["r_delta"] for r in results if not np.isnan(r["r_delta"])]
                if r_deltas:
                    mean_d = np.mean(r_deltas)
                    se_d = np.std(r_deltas) / np.sqrt(len(r_deltas))
                    print(f"  trial {trial+1}/{M}: <r_delta>={mean_d:+.5f} +/- {se_d:.5f}  [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        # Aggregate statistics
        r_flats = [r["r_flat"] for r in results if not np.isnan(r["r_flat"])]
        r_curvs = [r["r_curved"] for r in results if not np.isnan(r["r_curved"])]
        r_deltas = [r["r_delta"] for r in results if not np.isnan(r["r_delta"])]

        print(f"\n  RESULTS ({geo_name}, {len(r_deltas)} valid trials, {elapsed:.1f}s):")
        print(f"    <r>_flat    = {np.mean(r_flats):.5f} +/- {np.std(r_flats)/np.sqrt(len(r_flats)):.5f}")
        print(f"    <r>_curved  = {np.mean(r_curvs):.5f} +/- {np.std(r_curvs)/np.sqrt(len(r_curvs)):.5f}")
        print(f"    <r>_delta   = {np.mean(r_deltas):+.5f} +/- {np.std(r_deltas)/np.sqrt(len(r_deltas)):.5f}")

        # Cohen's d effect size
        if np.std(r_deltas) > 0:
            d_cohen = np.mean(r_deltas) / np.std(r_deltas)
        else:
            d_cohen = 0.0

        # Paired t-test (CRN)
        if len(r_deltas) >= 5:
            t_stat, p_val = stats.ttest_1samp(r_deltas, 0.0)
        else:
            t_stat, p_val = 0.0, 1.0

        # Wilcoxon signed-rank (nonparametric)
        if len(r_deltas) >= 5:
            try:
                w_stat, w_pval = stats.wilcoxon(r_deltas, alternative='two-sided')
            except ValueError:
                w_stat, w_pval = 0.0, 1.0
        else:
            w_stat, w_pval = 0.0, 1.0

        print(f"    Cohen's d   = {d_cohen:+.3f}")
        print(f"    t-test      = t={t_stat:.3f}, p={p_val:.2e}")
        print(f"    Wilcoxon    = W={w_stat:.0f}, p={w_pval:.2e}")

        # RMT class
        r_mean_flat = np.mean(r_flats)
        if abs(r_mean_flat - R_POISSON) < abs(r_mean_flat - R_GOE):
            rmt_class = "POISSON"
        elif abs(r_mean_flat - R_GOE) < abs(r_mean_flat - R_GUE):
            rmt_class = "GOE"
        else:
            rmt_class = "GUE"
        print(f"    RMT class (flat): {rmt_class} (<r>={r_mean_flat:.4f})")

        summary = {
            "geometry": geo_name,
            "eps": eps,
            "N": N,
            "M": M,
            "n_valid": len(r_deltas),
            "r_flat_mean": float(np.mean(r_flats)),
            "r_flat_se": float(np.std(r_flats) / np.sqrt(len(r_flats))),
            "r_curved_mean": float(np.mean(r_curvs)),
            "r_curved_se": float(np.std(r_curvs) / np.sqrt(len(r_curvs))),
            "r_delta_mean": float(np.mean(r_deltas)),
            "r_delta_se": float(np.std(r_deltas) / np.sqrt(len(r_deltas))),
            "cohen_d": float(d_cohen),
            "ttest_t": float(t_stat),
            "ttest_p": float(p_val),
            "wilcoxon_W": float(w_stat),
            "wilcoxon_p": float(w_pval),
            "rmt_class_flat": rmt_class,
            "elapsed_sec": elapsed,
            "trials": results,
        }

        all_results[geo_name] = summary

    # Save
    outpath = os.path.join(OUTDIR, "pilot_gap_ratio.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    # Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for geo, res in all_results.items():
        p = res["ttest_p"]
        d = res["cohen_d"]
        if p < 0.001 and abs(d) > 0.5:
            verdict = "DETECTED ★★★"
        elif p < 0.01 and abs(d) > 0.3:
            verdict = "DETECTED ★★"
        elif p < 0.05:
            verdict = "WEAK ★"
        else:
            verdict = "NULL"
        print(f"  {geo:20s}: delta={res['r_delta_mean']:+.5f}, d={d:+.3f}, p={p:.2e} → {verdict}")

    print("\nDone.")


if __name__ == "__main__":
    main()
