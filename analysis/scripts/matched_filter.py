#!/usr/bin/env python3
"""Matched filter: find optimal linear combination of 6 statistics.

Reads Test A per-seed data from all_skeptic_tests.json.
Computes:
1. Optimal weight vector w = Σ⁻¹ · μ (maximizes d_cohen)
2. d_cohen of optimal combination vs each individual statistic
3. Leave-one-out cross-validated d to check for overfitting

NO bias: the optimization maximizes SNR purely from data.
Both positive and negative results are reported identically.
The optimal combination might turn out WORSE than individual statistics
(if they're highly correlated), which would also be reported honestly.
"""
import json, sys
import numpy as np

def main():
    with open("analysis/universal_runs_v2/all_skeptic_tests.json") as f:
        data = json.load(f)

    test_A = data["test_A_why_kurtosis"]
    stat_names = list(test_A.keys())
    n_stats = len(stat_names)

    # Build matrix: rows = seeds, columns = statistics
    n_seeds = len(test_A[stat_names[0]]["per_seed"])
    X = np.zeros((n_seeds, n_stats), dtype=np.float64)
    for j, s in enumerate(stat_names):
        X[:, j] = np.array(test_A[s]["per_seed"])

    # Mean response vector
    mu = np.mean(X, axis=0)

    # Covariance matrix of fluctuations
    Sigma = np.cov(X, rowvar=False, ddof=1)

    print(f"=== MATCHED FILTER: OPTIMAL LINEAR COMBINATION ===")
    print(f"Statistics: {stat_names}")
    print(f"N_seeds: {n_seeds}")
    print()

    # Individual d_cohen
    print("Individual statistics:")
    for j, s in enumerate(stat_names):
        std_j = np.std(X[:, j], ddof=1)
        d_j = mu[j] / std_j if std_j > 1e-15 else 0.0
        print(f"  {s:15s}: mean={mu[j]:+.6f}, std={std_j:.6f}, d={d_j:+.3f}")
    print()

    # Optimal matched filter
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        # Regularize if singular
        Sigma_inv = np.linalg.inv(Sigma + 1e-10 * np.eye(n_stats))
        print("WARNING: Sigma singular, regularized.")

    w = Sigma_inv @ mu  # optimal weight vector
    w_normalized = w / np.linalg.norm(w)

    # Optimal combination per seed
    Z = X @ w_normalized
    mu_Z = np.mean(Z)
    std_Z = np.std(Z, ddof=1)
    d_optimal = mu_Z / std_Z if std_Z > 1e-15 else 0.0

    print(f"Optimal weight vector (normalized):")
    for j, s in enumerate(stat_names):
        print(f"  {s:15s}: w={w_normalized[j]:+.6f}")
    print()
    print(f"Optimal combination: mean={mu_Z:+.6f}, std={std_Z:.6f}, d={d_optimal:+.3f}")
    print()

    # Sample size warning
    if n_seeds < 3 * n_stats:
        print(f"WARNING: n_seeds={n_seeds} < 3×n_stats={3*n_stats}. "
              f"Covariance matrix may be ill-conditioned. "
              f"LOO cross-validation is the reliable metric, not in-sample d.")
        print()

    # Comparison
    best_individual = max(stat_names, key=lambda s: abs(test_A[s]["d"]))
    d_best_ind = abs(test_A[best_individual]["d"])
    print(f"Best individual: {best_individual} (|d|={d_best_ind:.3f})")
    print(f"Optimal combination: |d|={abs(d_optimal):.3f}")
    print(f"Improvement: {abs(d_optimal)/d_best_ind:.2f}x")
    print()

    # Leave-one-out cross-validation (guards against overfitting)
    print("Leave-one-out cross-validated d:")
    loo_Z = np.zeros(n_seeds)
    for i in range(n_seeds):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i+1, n_seeds)])
        X_train = X[train_idx]
        mu_train = np.mean(X_train, axis=0)
        Sigma_train = np.cov(X_train, rowvar=False, ddof=1)
        try:
            w_train = np.linalg.inv(Sigma_train) @ mu_train
        except:
            w_train = np.linalg.inv(Sigma_train + 1e-10 * np.eye(n_stats)) @ mu_train
        w_train_n = w_train / np.linalg.norm(w_train)
        loo_Z[i] = X[i] @ w_train_n

    mu_loo = np.mean(loo_Z)
    std_loo = np.std(loo_Z, ddof=1)
    d_loo = mu_loo / std_loo if std_loo > 1e-15 else 0.0
    print(f"  LOO d = {d_loo:+.3f} (vs in-sample d = {d_optimal:+.3f})")
    print(f"  Overfitting ratio: {abs(d_optimal)/max(abs(d_loo),1e-15):.2f}")
    if abs(d_loo) < abs(d_optimal) * 0.7:
        print("  WARNING: significant overfitting detected!")
    else:
        print("  OK: LOO consistent with in-sample.")
    print()

    # Correlation matrix between statistics
    print("Correlation matrix:")
    corr = np.corrcoef(X, rowvar=False)
    header = "            " + " ".join(f"{s[:7]:>7s}" for s in stat_names)
    print(header)
    for j, s in enumerate(stat_names):
        row = f"  {s[:10]:10s}" + " ".join(f"{corr[j,k]:+7.3f}" for k in range(n_stats))
        print(row)

    # Save
    result = {
        "stat_names": stat_names,
        "mu": mu.tolist(),
        "weights_normalized": w_normalized.tolist(),
        "d_optimal": float(d_optimal),
        "d_loo": float(d_loo),
        "individual_d": {s: float(test_A[s]["d"]) for s in stat_names},
        "correlation_matrix": corr.tolist(),
    }
    with open("analysis/universal_runs_v2/matched_filter.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to matched_filter.json")


if __name__ == "__main__":
    main()
