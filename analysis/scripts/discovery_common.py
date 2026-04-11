"""
Discovery Run 001 — Common functions for validation experiments.
Shared by val1/val2/val3 to avoid code duplication.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import gc


# ---------------------------------------------------------------------------
# Sprinkling
# ---------------------------------------------------------------------------
def sprinkle_4d(N, T, rng):
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
# Causal conditions
# ---------------------------------------------------------------------------
def causal_flat(pts):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
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


def causal_ppwave_cross(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm * ym
    mink = dt**2 - dr2
    corr = eps * f * (dt + dz)**2 / 2.0
    return ((mink > corr) & (dt > 0)).astype(np.float64)


def causal_schwarzschild(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    zm = (z[np.newaxis, :] + z[:, np.newaxis]) / 2.0
    rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3
    Phi = -eps / rm
    return (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.float64)


def causal_flrw(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2.0
    a2 = (1 + eps * tm**2)**2
    return ((dt**2 > a2 * dr2) & (dt > 0)).astype(np.float64)


def causal_conformal(pts, eps):
    """Conformal rescaling: identical causal structure to flat. NULL CONTROL."""
    return causal_flat(pts)


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "ppwave_cross": causal_ppwave_cross,
    "schwarzschild": causal_schwarzschild,
    "flrw": causal_flrw,
    "conformal": causal_conformal,
}

SEED_OFFSETS = {
    "ppwave_quad": 100, "ppwave_cross": 200, "schwarzschild": 300,
    "flrw": 400, "conformal": 500,
}


# ---------------------------------------------------------------------------
# Link graph + Laplacian
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


# ---------------------------------------------------------------------------
# Forman-Ricci + full graph statistics
# ---------------------------------------------------------------------------
def graph_statistics(A_sp):
    """Compute comprehensive graph statistics for bias control.

    Returns dict with: mean_degree, degree_var, degree_skew, degree_kurt,
    max_degree, min_degree, edge_count, assortativity.
    """
    degrees = np.array(A_sp.sum(axis=1)).ravel()
    n_edges = int(A_sp.sum()) // 2

    result = {
        "mean_degree": float(np.mean(degrees)),
        "degree_var": float(np.var(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_skew": float(stats.skew(degrees)),
        "degree_kurt": float(stats.kurtosis(degrees)),
        "max_degree": int(np.max(degrees)),
        "min_degree": int(np.min(degrees)),
        "edge_count": n_edges,
    }

    # Assortativity (degree-degree correlation)
    rows, cols = sp.triu(A_sp, k=1).nonzero()
    if len(rows) > 0:
        d_src = degrees[rows]
        d_tgt = degrees[cols]
        if np.std(d_src) > 0 and np.std(d_tgt) > 0:
            result["assortativity"] = float(np.corrcoef(d_src, d_tgt)[0, 1])
        else:
            result["assortativity"] = 0.0
    else:
        result["assortativity"] = 0.0

    return result, degrees


def forman_ricci(A_sp, degrees=None):
    """Basic Forman-Ricci: F(e=(u,v)) = 4 - deg(u) - deg(v).

    Returns per-edge F values and summary statistics.
    """
    if degrees is None:
        degrees = np.array(A_sp.sum(axis=1)).ravel()

    rows, cols = sp.triu(A_sp, k=1).nonzero()
    F = 4.0 - degrees[rows] - degrees[cols]

    return {
        "F_mean": float(np.mean(F)),
        "F_std": float(np.std(F)),
        "F_median": float(np.median(F)),
        "F_skew": float(stats.skew(F)) if len(F) > 2 else 0.0,
        "F_kurt": float(stats.kurtosis(F)) if len(F) > 3 else 0.0,
        "n_edges": len(F),
    }, F


# ---------------------------------------------------------------------------
# CRN trial with full statistics
# ---------------------------------------------------------------------------
def crn_trial_full(seed, N, T, metric_name, eps):
    """One CRN trial: flat + curved, Forman + full graph stats."""
    seed_offset = SEED_OFFSETS[metric_name]
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    # Flat
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    del C_flat; gc.collect()
    gs_flat, deg_flat = graph_statistics(A_flat)
    fr_flat, F_flat = forman_ricci(A_flat, deg_flat)
    del A_flat; gc.collect()

    # Curved
    C_curv = METRIC_FNS[metric_name](pts, eps)
    A_curv = build_link_graph(C_curv)
    del C_curv; gc.collect()
    gs_curv, deg_curv = graph_statistics(A_curv)
    fr_curv, F_curv = forman_ricci(A_curv, deg_curv)
    del A_curv; gc.collect()

    # Deltas
    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps}

    # Forman deltas
    result["F_mean_flat"] = fr_flat["F_mean"]
    result["F_mean_curv"] = fr_curv["F_mean"]
    result["F_mean_delta"] = fr_curv["F_mean"] - fr_flat["F_mean"]

    # Graph stat deltas
    for key in gs_flat:
        result[f"{key}_flat"] = gs_flat[key]
        result[f"{key}_curv"] = gs_curv[key]
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]

    # Degree-corrected Forman residual: F_delta - (-2 * mean_degree_delta)
    result["F_residual"] = result["F_mean_delta"] - (-2 * result["mean_degree_delta"])

    return result


def analyze_deltas(results, label=""):
    """Analyze CRN delta arrays with pre-registered thresholds."""
    M = len(results)

    # Extract key deltas
    F_deltas = [r["F_mean_delta"] for r in results]
    residuals = [r["F_residual"] for r in results]
    deg_deltas = [r["mean_degree_delta"] for r in results]
    degvar_deltas = [r["degree_var_delta"] for r in results]

    print(f"\n  === {label} (M={M}) ===")

    # 1. Raw Forman delta
    m, se = np.mean(F_deltas), np.std(F_deltas) / np.sqrt(M)
    d_cohen = m / np.std(F_deltas) if np.std(F_deltas) > 0 else 0
    _, p = stats.ttest_1samp(F_deltas, 0.0) if M >= 5 else (0, 1)
    print(f"  F_mean_delta     = {m:+.4f} +/- {se:.4f}, d={d_cohen:+.3f}, p={p:.2e}")

    # 2. Degree-corrected residual
    m, se = np.mean(residuals), np.std(residuals) / np.sqrt(M)
    d_cohen = m / np.std(residuals) if np.std(residuals) > 0 else 0
    _, p_resid = stats.ttest_1samp(residuals, 0.0) if M >= 5 else (0, 1)
    print(f"  F_residual       = {m:+.4f} +/- {se:.4f}, d={d_cohen:+.3f}, p={p_resid:.2e}")

    # 3. Degree delta
    m_deg = np.mean(deg_deltas)
    _, p_deg = stats.ttest_1samp(deg_deltas, 0.0) if M >= 5 else (0, 1)
    print(f"  mean_degree_delta= {m_deg:+.4f}, p={p_deg:.2e}")

    # 4. Degree-variance delta
    m_dv = np.mean(degvar_deltas)
    _, p_dv = stats.ttest_1samp(degvar_deltas, 0.0) if M >= 5 else (0, 1)
    print(f"  degree_var_delta = {m_dv:+.4f}, p={p_dv:.2e}")

    # 5. ADVERSARIAL: R^2 of residual against ALL simple graph stats
    simple_stats = ["mean_degree_delta", "degree_var_delta", "degree_std_delta",
                    "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
                    "max_degree_delta", "assortativity_delta"]

    print(f"\n  ADVERSARIAL PROXY CHECK:")
    max_r2 = 0.0
    max_r2_name = ""
    for stat_name in simple_stats:
        vals = [r.get(stat_name, 0) for r in results]
        if np.std(vals) > 1e-15 and np.std(residuals) > 1e-15:
            corr = np.corrcoef(residuals, vals)[0, 1]
            r2 = corr**2
        else:
            r2 = 0.0
        if r2 > max_r2:
            max_r2 = r2
            max_r2_name = stat_name
        if r2 > 0.2:
            print(f"    R2(residual, {stat_name}) = {r2:.3f}  {'!!!' if r2 > 0.5 else ''}")

    print(f"    MAX R2 = {max_r2:.3f} ({max_r2_name})")

    # 6. VERDICT (pre-registered thresholds)
    # Bonferroni adjusted alpha: 0.01 / 30 = 0.000333
    ALPHA_BONF = 0.01 / 30

    if p_resid < ALPHA_BONF and max_r2 < 0.50:
        verdict = "DETECTED (genuine)"
    elif p_resid < ALPHA_BONF and max_r2 >= 0.80:
        verdict = "PROXY (explains >80%)"
    elif p_resid < ALPHA_BONF and 0.50 <= max_r2 < 0.80:
        verdict = "AMBIGUOUS (50-80% proxy)"
    elif p_resid < 0.05:
        verdict = "WEAK (p<0.05, not Bonferroni)"
    else:
        verdict = "NULL"

    print(f"\n  VERDICT: {verdict}")
    print(f"    (Bonferroni alpha = {ALPHA_BONF:.6f}, p_resid = {p_resid:.2e}, max_R2 = {max_r2:.3f})")

    return {
        "verdict": verdict,
        "p_residual": float(p_resid),
        "max_r2_proxy": float(max_r2),
        "max_r2_name": max_r2_name,
        "cohen_d_residual": float(np.mean(residuals) / np.std(residuals)) if np.std(residuals) > 0 else 0,
        "mean_residual": float(np.mean(residuals)),
        "se_residual": float(np.std(residuals) / np.sqrt(M)),
    }
