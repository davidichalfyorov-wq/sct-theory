"""
FND-1 Discovery Module: Unsupervised pattern finding in per-sprinkling data.

Looks for things we DON'T know to look for:
- Cross-observable correlation matrix (what correlates with what?)
- PCA on spectral data (hidden structure?)
- Random forest feature importance (what predicts curvature?)
- Anomaly detection (outlier sprinklings with unusual physics?)
- Mutual information (nonlinear dependencies beyond correlation?)
- Eigenvalue ratio universality (dimension-independent constants?)

Requires: per_sprinkling_multioperator.json from fnd1_per_sprinkling.py

Run:
    python analysis/scripts/fnd1_discovery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "speculative" / "numerics" / "ensemble_results"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "fnd1_discovery"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _init_style():
    try:
        plt.style.use(["science", "high-vis"])
    except Exception:
        plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})


# ---------------------------------------------------------------------------
# Load per-sprinkling data
# ---------------------------------------------------------------------------

def load_data() -> list[dict]:
    path = RESULTS_DIR / "per_sprinkling_multioperator.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run fnd1_per_sprinkling.py first.")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("per_sprinkling", [])


# ---------------------------------------------------------------------------
# 1. Cross-observable correlation matrix
# ---------------------------------------------------------------------------

def correlation_matrix(records: list[dict]) -> dict:
    """Compute Spearman correlation between ALL numeric observables."""
    if not records:
        return {"fields": [], "corr_matrix": [], "n_surprising": 0, "surprising_correlations": []}
    # Extract numeric fields
    skip = {"seed", "N", "link_top_eigs", "comm_top_eigs", "eig_ratios"}
    fields = [k for k in records[0].keys()
              if k not in skip and isinstance(records[0][k], (int, float))]

    n = len(fields)
    corr = np.zeros((n, n))
    pvals = np.zeros((n, n))

    arrays = {}
    for f in fields:
        arrays[f] = np.array([r[f] for r in records], dtype=float)

    for i, fi in enumerate(fields):
        for j, fj in enumerate(fields):
            if i <= j:
                r, p = stats.spearmanr(arrays[fi], arrays[fj])
                corr[i, j] = corr[j, i] = r if not np.isnan(r) else 0
                pvals[i, j] = pvals[j, i] = p if not np.isnan(p) else 1

    # Find surprising correlations (high |r| between non-obvious pairs)
    surprising = []
    obvious_pairs = {
        ("eps", "total_causal"), ("eps", "bd_action"), ("total_causal", "bd_action"),
        ("total_causal", "n_links"), ("total_causal", "mean_link_deg"),
        ("n_links", "mean_link_deg"), ("fiedler", "gap_ratio"),
        ("fiedler", "alg_conn_ratio"), ("fiedler", "link_entropy"),
        ("fiedler", "t2K_tau1"), ("fiedler", "t2K_tau5"),
        ("comm_frobenius", "comm_max"), ("comm_frobenius", "comm_entropy"),
        ("comm_max", "comm_entropy"), ("t2K_tau1", "t2K_tau5"),
        ("link_entropy", "t2K_tau1"), ("link_entropy", "t2K_tau5"),
    }

    for i in range(n):
        for j in range(i + 1, n):
            pair = (fields[i], fields[j])
            pair_rev = (fields[j], fields[i])
            if pair in obvious_pairs or pair_rev in obvious_pairs:
                continue
            if abs(corr[i, j]) > 0.3 and pvals[i, j] < 0.001:
                surprising.append({
                    "pair": f"{fields[i]} × {fields[j]}",
                    "rho": round(float(corr[i, j]), 4),
                    "p": float(pvals[i, j]),
                })

    surprising.sort(key=lambda x: abs(x["rho"]), reverse=True)

    return {
        "fields": fields,
        "corr_matrix": corr.tolist(),
        "n_surprising": len(surprising),
        "surprising_correlations": surprising[:20],
    }


def fig_correlation_matrix(result: dict):
    """Heatmap of cross-observable correlations."""
    fields = result["fields"]
    corr = np.array(result["corr_matrix"])
    if len(fields) < 3:
        return

    _init_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(fields)))
    ax.set_yticks(range(len(fields)))
    ax.set_xticklabels(fields, fontsize=6, rotation=90)
    ax.set_yticklabels(fields, fontsize=6)

    # Annotate strong correlations
    for i in range(len(fields)):
        for j in range(len(fields)):
            if abs(corr[i, j]) > 0.3 and i != j:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(corr[i, j]) > 0.5 else "black")

    ax.set_title("Cross-Observable Correlation Matrix (Spearman)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: correlation_matrix.pdf")


# ---------------------------------------------------------------------------
# 2. PCA on spectral data
# ---------------------------------------------------------------------------

def pca_analysis(records: list[dict]) -> dict:
    """PCA on normalized observables — find hidden structure."""
    if not records:
        return {"error": "no records"}
    skip = {"seed", "N", "eps", "link_top_eigs", "comm_top_eigs", "eig_ratios"}
    fields = [k for k in records[0].keys()
              if k not in skip and isinstance(records[0][k], (int, float))]

    X = np.array([[r[f] for f in fields] for r in records], dtype=float)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-15] = 1
    X_std = (X - mean) / std

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    explained = (S ** 2) / np.sum(S ** 2)

    # Project onto first 3 PCs
    proj = X_std @ Vt[:3].T

    eps_arr = np.array([r["eps"] for r in records])
    N_arr = np.array([r["N"] for r in records])

    # Check if PCs separate curvature
    pc_eps_corr = []
    for pc in range(min(3, proj.shape[1])):
        r, p = stats.spearmanr(eps_arr, proj[:, pc])
        pc_eps_corr.append({"pc": pc, "rho_eps": round(float(r), 4), "p": float(p)})

    # Top loadings for each PC
    loadings = {}
    for pc in range(min(3, Vt.shape[0])):
        idx = np.argsort(np.abs(Vt[pc]))[::-1][:5]
        loadings[f"PC{pc}"] = {fields[i]: round(float(Vt[pc, i]), 3) for i in idx}

    return {
        "n_features": len(fields),
        "explained_variance": [round(float(e), 4) for e in explained[:5]],
        "cumulative_80pct": int(np.searchsorted(np.cumsum(explained), 0.80) + 1),
        "pc_eps_correlation": pc_eps_corr,
        "top_loadings": loadings,
        "projection": proj.tolist(),
        "eps": eps_arr.tolist(),
        "N": N_arr.tolist(),
    }


def fig_pca(result: dict):
    """PCA scatter: PC1 vs PC2, colored by eps."""
    proj = np.array(result["projection"])
    eps = np.array(result["eps"])
    if proj.shape[1] < 2:
        return

    _init_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: PC1 vs PC2, color=eps
    sc = ax1.scatter(proj[:, 0], proj[:, 1], c=eps, cmap="viridis", s=10, alpha=0.7)
    ax1.set_xlabel(f"PC1 ({result['explained_variance'][0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({result['explained_variance'][1]*100:.1f}%)")
    ax1.set_title("PCA: Spectral Data Colored by Curvature (eps)")
    fig.colorbar(sc, ax=ax1, label="eps")

    # Right: explained variance
    evr = result["explained_variance"]
    ax2.bar(range(len(evr)), [e * 100 for e in evr], color="#2196F3")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title(f"PCA: {result['cumulative_80pct']} PCs for 80% variance")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_spectral.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "pca_spectral.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: pca_spectral.pdf")


# ---------------------------------------------------------------------------
# 3. Random forest feature importance
# ---------------------------------------------------------------------------

def feature_importance(records: list[dict]) -> dict:
    """Which features best predict curvature (eps)?
    Uses GroupKFold by N to prevent cross-N information leakage.
    """
    if not records:
        return {"error": "no records"}
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score, GroupKFold
    except ImportError:
        return {"error": "scikit-learn not installed"}

    skip = {"seed", "N", "eps", "link_top_eigs", "comm_top_eigs", "eig_ratios"}
    fields = [k for k in records[0].keys()
              if k not in skip and isinstance(records[0][k], (int, float))]

    X = np.array([[r[f] for f in fields] for r in records], dtype=float)
    y = np.array([r["eps"] for r in records], dtype=float)
    groups = np.array([r["N"] for r in records])

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Per-N z-score normalization (features have different scales at different N)
    for n_val in np.unique(groups):
        mask = groups == n_val
        m = X[mask].mean(axis=0)
        s = X[mask].std(axis=0)
        s[s < 1e-15] = 1
        X[mask] = (X[mask] - m) / s

    # Fit random forest — per-N normalized, so cross-N CV is valid
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=5, scoring="r2")

    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

    top_features = []
    for i in idx[:10]:
        top_features.append({
            "feature": fields[i],
            "importance": round(float(importances[i]), 4),
        })

    # Also fit WITHOUT confounders (TC, BD) to see what's left
    confounder_idx = [i for i, f in enumerate(fields) if f in ("total_causal", "bd_action", "n_links")]
    non_confounder_fields = [f for i, f in enumerate(fields) if i not in confounder_idx]
    X_clean = np.delete(X, confounder_idx, axis=1)

    rf_clean = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    scores_clean = cross_val_score(rf_clean, X_clean, y, cv=5, scoring="r2")

    rf_clean.fit(X_clean, y)
    imp_clean = rf_clean.feature_importances_
    idx_clean = np.argsort(imp_clean)[::-1]

    top_clean = []
    for i in idx_clean[:10]:
        top_clean.append({
            "feature": non_confounder_fields[i],
            "importance": round(float(imp_clean[i]), 4),
        })

    return {
        "r2_cv_all": round(float(np.mean(scores)), 4),
        "r2_cv_no_confounders": round(float(np.mean(scores_clean)), 4),
        "top_features_all": top_features,
        "top_features_no_confounders": top_clean,
        "note": "R2 > 0.5 without confounders = genuine curvature information in spectral data",
    }


def fig_feature_importance(result: dict):
    """Bar chart of feature importances."""
    if "error" in result:
        return

    _init_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # All features
    feats = result["top_features_all"][:8]
    ax1.barh(range(len(feats)), [f["importance"] for f in feats], color="#2196F3")
    ax1.set_yticks(range(len(feats)))
    ax1.set_yticklabels([f["feature"] for f in feats], fontsize=8)
    ax1.set_xlabel("Importance")
    ax1.set_title(f"All Features (R²={result['r2_cv_all']:.3f})")
    ax1.invert_yaxis()

    # Without confounders
    feats_c = result["top_features_no_confounders"][:8]
    ax2.barh(range(len(feats_c)), [f["importance"] for f in feats_c], color="#FF9800")
    ax2.set_yticks(range(len(feats_c)))
    ax2.set_yticklabels([f["feature"] for f in feats_c], fontsize=8)
    ax2.set_xlabel("Importance")
    ax2.set_title(f"Without TC/BD/N_links (R²={result['r2_cv_no_confounders']:.3f})")
    ax2.invert_yaxis()

    fig.suptitle("Random Forest: What Predicts Curvature?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: feature_importance.pdf")


# ---------------------------------------------------------------------------
# 4. Anomaly detection
# ---------------------------------------------------------------------------

def anomaly_detection(records: list[dict]) -> dict:
    """Find outlier sprinklings using PCA-based anomaly score (handles correlated features)."""
    if not records:
        return {"n_anomalies": 0, "top_anomalies": [], "anomaly_eps_correlation": 0, "anomaly_eps_p": 1}

    skip = {"seed", "N", "eps", "link_top_eigs", "comm_top_eigs", "eig_ratios"}
    fields = [k for k in records[0].keys()
              if k not in skip and isinstance(records[0][k], (int, float))]

    X = np.array([[r[f] for f in fields] for r in records], dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-15] = 1
    Z = (X - mean) / std

    # PCA-based anomaly: project onto PCs, normalize by eigenvalue, sum
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    # Score = sum of (projection / sqrt(eigenvalue))^2, normalized by n_features
    n_feat = len(fields)
    S_safe = np.maximum(S, 1e-10)
    proj = Z @ Vt.T  # projection onto PCs
    anomaly_score = np.sum((proj / S_safe[np.newaxis, :]) ** 2, axis=1) / n_feat

    # Top anomalies
    top_idx = np.argsort(anomaly_score)[::-1][:10]
    anomalies = []
    for i in top_idx:
        r = records[i]
        # Which fields are most anomalous?
        field_z = {fields[j]: round(float(Z[i, j]), 2) for j in range(len(fields))}
        top_fields = sorted(field_z.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        anomalies.append({
            "idx": int(i),
            "seed": r["seed"],
            "N": r["N"],
            "eps": r["eps"],
            "score": round(float(anomaly_score[i]), 2),
            "top_deviations": {k: v for k, v in top_fields},
        })

    # Is anomaly score correlated with eps?
    eps_arr = np.array([r["eps"] for r in records])
    r_anom, p_anom = stats.spearmanr(eps_arr, anomaly_score)

    return {
        "n_anomalies": len(anomalies),
        "top_anomalies": anomalies,
        "anomaly_eps_correlation": round(float(r_anom), 4),
        "anomaly_eps_p": float(p_anom),
        "note": "High anomaly-eps correlation means curvature creates unusual sprinklings",
    }


# ---------------------------------------------------------------------------
# 5. Clustering — do sprinklings naturally group by curvature?
# ---------------------------------------------------------------------------

def clustering_analysis(records: list[dict]) -> dict:
    """K-means on PCA-projected spectral data. Do flat and curved cluster separately?"""
    if not records or len(records) < 10:
        return {"error": "insufficient records"}

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError:
        return {"error": "scikit-learn not installed"}

    skip = {"seed", "N", "eps", "link_top_eigs", "comm_top_eigs", "eig_ratios"}
    fields = [k for k in records[0].keys()
              if k not in skip and isinstance(records[0][k], (int, float))]

    X = np.array([[r[f] for f in fields] for r in records], dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    groups = np.array([r["N"] for r in records])

    # Per-N normalization
    for n_val in np.unique(groups):
        mask = groups == n_val
        m = X[mask].mean(axis=0)
        s = X[mask].std(axis=0)
        s[s < 1e-15] = 1
        X[mask] = (X[mask] - m) / s

    # PCA to 5 components
    mean = X.mean(axis=0)
    X_c = X - mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_pca = X_c @ Vt[:5].T

    eps_arr = np.array([r["eps"] for r in records])
    # Binary label: flat (eps=0) vs curved (eps>0)
    true_labels = (eps_arr > 0.01).astype(int)

    results = {}
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        pred = km.fit_predict(X_pca)

        ari = adjusted_rand_score(true_labels, pred)
        sil = silhouette_score(X_pca, pred) if k < len(X_pca) else 0

        results[f"k={k}"] = {
            "ARI_vs_curvature": round(float(ari), 4),
            "silhouette": round(float(sil), 4),
            "curvature_separable": ari > 0.1,
        }

    # Best k
    best_k = max(results.keys(), key=lambda k: results[k]["ARI_vs_curvature"])
    best = results[best_k]

    return {
        "per_k": results,
        "best_k": best_k,
        "curvature_naturally_clusters": best["ARI_vs_curvature"] > 0.1,
        "note": "ARI > 0.1 means clusters align with flat/curved distinction without supervision",
    }


# ---------------------------------------------------------------------------
# 6. Mutual information
# ---------------------------------------------------------------------------

def mutual_information_analysis(records: list[dict]) -> dict:
    """Mutual information between eps and observables.

    Uses sklearn k-NN MI estimator when available (more accurate than binning).
    Falls back to histogram-based estimation with Miller-Madow bias correction.
    Compares MI to shuffled null to detect genuine nonlinear dependencies.
    """
    if not records:
        return {"all": {}, "nonlinear_dependencies": {}}

    eps_arr = np.array([r["eps"] for r in records], dtype=float)

    skip = {"seed", "N", "eps", "link_top_eigs", "comm_top_eigs", "eig_ratios"}
    fields = [k for k in records[0].keys()
              if k not in skip and isinstance(records[0][k], (int, float))]

    # Try sklearn MI (k-NN based, more accurate)
    use_sklearn = False
    try:
        from sklearn.feature_selection import mutual_info_regression
        use_sklearn = True
    except ImportError:
        pass

    results = {}
    for f in fields:
        arr = np.array([r[f] for r in records], dtype=float)
        arr = np.nan_to_num(arr, nan=np.nanmedian(arr) if not np.all(np.isnan(arr)) else 0)

        # Skip constant fields
        if np.std(arr) < 1e-15:
            results[f] = {"MI": 0.0, "MI_null": 0.0, "MI_excess": 0.0,
                          "spearman": 0.0, "significant": False}
            continue

        # Spearman
        rho_s, p_s = stats.spearmanr(eps_arr, arr)
        rho_s = 0.0 if np.isnan(rho_s) else float(rho_s)

        if use_sklearn:
            # k-NN MI estimation (Kraskov et al.)
            mi = float(mutual_info_regression(
                arr.reshape(-1, 1), eps_arr, n_neighbors=5, random_state=42)[0])
            # Null: shuffled eps
            rng = np.random.default_rng(42)
            mi_nulls = []
            for _ in range(20):
                eps_shuf = rng.permutation(eps_arr)
                mi_null = float(mutual_info_regression(
                    arr.reshape(-1, 1), eps_shuf, n_neighbors=5, random_state=42)[0])
                mi_nulls.append(mi_null)
            mi_null_mean = float(np.mean(mi_nulls))
            mi_excess = mi - mi_null_mean
        else:
            # Histogram fallback with proper binning
            n_bins = 8
            eps_range = eps_arr.max() - eps_arr.min()
            arr_range = np.nanmax(arr) - np.nanmin(arr)
            if eps_range < 1e-15 or arr_range < 1e-15:
                results[f] = {"MI": 0.0, "MI_null": 0.0, "MI_excess": 0.0,
                              "spearman": rho_s, "significant": False}
                continue

            joint, _, _ = np.histogram2d(eps_arr, arr, bins=n_bins)
            joint = joint / joint.sum()
            p_eps = joint.sum(axis=1)
            p_obs = joint.sum(axis=0)

            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint[i, j] > 1e-10 and p_eps[i] > 1e-10 and p_obs[j] > 1e-10:
                        mi += joint[i, j] * np.log(joint[i, j] / (p_eps[i] * p_obs[j]))

            # Miller-Madow bias correction
            B_nonzero = np.sum(joint > 1e-10)
            N_total = len(records)
            mi_corrected = max(0, mi - (B_nonzero - 1) / (2 * N_total))
            mi = mi_corrected
            mi_null_mean = (n_bins * n_bins - 1) / (2 * N_total)  # expected MI under null
            mi_excess = mi - mi_null_mean

        results[f] = {
            "MI": round(float(mi), 4),
            "MI_null": round(float(mi_null_mean), 4),
            "MI_excess": round(float(mi_excess), 4),
            "spearman": round(rho_s, 4),
            "significant": mi_excess > 2 * mi_null_mean and mi_excess > 0.01,
        }

    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["MI_excess"], reverse=True))
    nonlinear = {k: v for k, v in sorted_results.items() if v["significant"]}

    return {
        "all": sorted_results,
        "nonlinear_dependencies": nonlinear,
        "method": "sklearn_knn" if use_sklearn else "histogram_miller_madow",
        "note": "MI_excess = MI - MI_null. Significant if excess > 2*null and > 0.01",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FND-1 DISCOVERY MODULE: Unsupervised Pattern Finding")
    print("=" * 70)
    print()

    records = load_data()
    if not records:
        return

    print(f"Loaded {len(records)} per-sprinkling records")
    print(f"Fields: {len([k for k in records[0] if isinstance(records[0][k], (int, float))])} numeric")
    print()

    # 1. Correlation matrix
    print("=" * 70)
    print("1. CROSS-OBSERVABLE CORRELATION MATRIX")
    print("=" * 70)
    corr_result = correlation_matrix(records)
    print(f"  {corr_result['n_surprising']} surprising correlations (|rho|>0.3, p<0.001)")
    for s in corr_result["surprising_correlations"][:10]:
        print(f"    {s['pair']}: rho={s['rho']:+.4f}")

    # 2. PCA
    print("\n" + "=" * 70)
    print("2. PCA — HIDDEN STRUCTURE IN SPECTRAL DATA")
    print("=" * 70)
    pca_result = pca_analysis(records)
    print(f"  {pca_result['cumulative_80pct']} PCs for 80% variance")
    print(f"  Explained: {pca_result['explained_variance'][:5]}")
    for pc_info in pca_result["pc_eps_correlation"]:
        print(f"  PC{pc_info['pc']} vs eps: rho={pc_info['rho_eps']:+.4f} (p={pc_info['p']:.2e})")
    print(f"  Top loadings:")
    for pc, loads in pca_result["top_loadings"].items():
        top3 = list(loads.items())[:3]
        print(f"    {pc}: {', '.join(f'{k}={v:+.3f}' for k, v in top3)}")

    # 3. Feature importance
    print("\n" + "=" * 70)
    print("3. RANDOM FOREST — WHAT PREDICTS CURVATURE?")
    print("=" * 70)
    fi_result = feature_importance(records)
    if "error" not in fi_result:
        print(f"  R² (all features): {fi_result['r2_cv_all']:.4f}")
        print(f"  R² (no confounders): {fi_result['r2_cv_no_confounders']:.4f}")
        print(f"  Top features (no confounders):")
        for f in fi_result["top_features_no_confounders"][:5]:
            print(f"    {f['feature']}: {f['importance']:.4f}")
    else:
        print(f"  {fi_result['error']}")

    # 4. Anomaly detection
    print("\n" + "=" * 70)
    print("4. ANOMALY DETECTION — UNUSUAL SPRINKLINGS")
    print("=" * 70)
    anom_result = anomaly_detection(records)
    print(f"  Anomaly-eps correlation: rho={anom_result['anomaly_eps_correlation']:+.4f}"
          f" (p={anom_result['anomaly_eps_p']:.2e})")
    print(f"  Top anomalies:")
    for a in anom_result["top_anomalies"][:5]:
        print(f"    seed={a['seed']}, N={a['N']}, eps={a['eps']}, "
              f"score={a['score']:.1f}, deviations={a['top_deviations']}")

    # 5. Mutual information
    print("\n" + "=" * 70)
    print("5. CLUSTERING — DO SPRINKLINGS GROUP BY CURVATURE?")
    print("=" * 70)
    clust_result = clustering_analysis(records)
    if "error" not in clust_result:
        for k, info in clust_result["per_k"].items():
            flag = " ← SEPARABLE" if info["curvature_separable"] else ""
            print(f"  {k}: ARI={info['ARI_vs_curvature']:+.4f}, "
                  f"silhouette={info['silhouette']:.4f}{flag}")
        print(f"  Natural curvature clustering: "
              f"{'YES' if clust_result['curvature_naturally_clusters'] else 'NO'}")
    else:
        print(f"  {clust_result['error']}")

    # 6. Mutual information
    print("\n" + "=" * 70)
    print("6. MUTUAL INFORMATION — NONLINEAR DEPENDENCIES")
    print("=" * 70)
    mi_result = mutual_information_analysis(records)
    print(f"  Method: {mi_result.get('method', 'unknown')}")
    print(f"  Top by MI_excess:")
    for k, v in list(mi_result["all"].items())[:8]:
        flag = " ← SIGNIFICANT" if v.get("significant") else ""
        print(f"    {k}: MI={v['MI']:.4f}, excess={v['MI_excess']:+.4f}, "
              f"spearman={v['spearman']:+.4f}{flag}")
    if mi_result["nonlinear_dependencies"]:
        print(f"\n  Features with significant nonlinear curvature info:")
        for k, v in mi_result["nonlinear_dependencies"].items():
            print(f"    {k}: MI_excess={v['MI_excess']:+.4f}")

    # Figures
    print("\n" + "=" * 70)
    print("GENERATING DISCOVERY FIGURES")
    print("=" * 70)
    fig_correlation_matrix(corr_result)
    fig_pca(pca_result)
    fig_feature_importance(fi_result)

    # Save
    output = {
        "n_records": len(records),
        "correlation_matrix": {
            "n_surprising": corr_result["n_surprising"],
            "surprising": corr_result["surprising_correlations"],
        },
        "pca": {
            "explained_variance": pca_result["explained_variance"],
            "cumulative_80pct": pca_result["cumulative_80pct"],
            "pc_eps_correlation": pca_result["pc_eps_correlation"],
            "top_loadings": pca_result["top_loadings"],
        },
        "feature_importance": fi_result,
        "anomaly_detection": anom_result,
        "clustering": clust_result,
        "mutual_information": mi_result,
    }

    out_path = RESULTS_DIR / "fnd1_discovery.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer, np.bool_)) else str(x))
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"  Surprising correlations: {corr_result['n_surprising']}")
    print(f"  PCA dimensionality: {pca_result['cumulative_80pct']} PCs for 80%")
    if "error" not in fi_result:
        r2 = fi_result["r2_cv_no_confounders"]
        if r2 > 0.5:
            print(f"  RF curvature prediction (no confounders): R²={r2:.3f} — STRONG")
        elif r2 > 0.2:
            print(f"  RF curvature prediction (no confounders): R²={r2:.3f} — MODERATE")
        else:
            print(f"  RF curvature prediction (no confounders): R²={r2:.3f} — WEAK")
    nl = len(mi_result.get("nonlinear_dependencies", {}))
    print(f"  Nonlinear dependencies found: {nl}")
    print("=" * 70)


if __name__ == "__main__":
    main()
