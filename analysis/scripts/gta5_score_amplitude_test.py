#!/usr/bin/env python3
"""GTA-5: Score Amplitude Optimality Test.
Is CJ an approximate projection of SA through path resolvent?
"""
import numpy as np, gzip, pickle, json, os
from scipy.stats import kurtosis as scipy_kurtosis

DATA_DIR = "analysis/fnd1_data"

def compute_A12(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask]**2
    strata_m = strata[mask]
    total = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = np.mean(np.abs(X[idx]) * dY2[idx]) - np.mean(np.abs(X[idx])) * np.mean(dY2[idx])
        total += w * cov
    return float(total)


print("=" * 70, flush=True)
print("GTA-5: Score Amplitude Optimality Test", flush=True)
print("=" * 70, flush=True)

cj_seeds = []
sa_unstrat_seeds = []
sa_global_seeds = []
profiles = []

for si in range(30):
    with gzip.open(os.path.join(DATA_DIR, f"T1.00_seed{si:03d}.pkl.gz"), "rb") as f:
        d = pickle.load(f)

    Y0 = d["Y0"]; mask = d["mask"]; strata = d["strata"]
    delta = d["delta_ppw"]; pts = d["pts"]

    # CJ standard (stratified)
    cj = compute_A12(Y0, delta, mask, strata)
    cj_seeds.append(cj)

    # SA proxy: unstratified Cov(|X|, dY^2)
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask]**2
    sa_unstrat = np.mean(np.abs(X) * dY2) - np.mean(np.abs(X)) * np.mean(dY2)
    sa_unstrat_seeds.append(sa_unstrat)

    # SA global: mean(dY^2)
    sa_global_seeds.append(np.mean(dY2))

    # 45-dim stratum profile of Cov_B(|X|, dY^2)
    strata_m = strata[mask]
    profile = np.zeros(45)
    for b in range(45):
        idx = strata_m == b
        if idx.sum() >= 3:
            profile[b] = (np.mean(np.abs(X[idx]) * dY2[idx])
                          - np.mean(np.abs(X[idx])) * np.mean(dY2[idx]))
    profiles.append(profile)

cj_arr = np.array(cj_seeds)
sa_unstrat_arr = np.array(sa_unstrat_seeds)
sa_global_arr = np.array(sa_global_seeds)
profiles = np.array(profiles)

# --- Seed-to-seed correlations ---
print("\n--- Seed-to-seed correlations ---", flush=True)
r_unstrat = np.corrcoef(cj_arr, sa_unstrat_arr)[0, 1]
r_global = np.corrcoef(cj_arr, sa_global_arr)[0, 1]
print(f"  corr(CJ, SA_unstrat)  = {r_unstrat:.4f}  R^2 = {r_unstrat**2:.4f}", flush=True)
print(f"  corr(CJ, SA_global)   = {r_global:.4f}  R^2 = {r_global**2:.4f}", flush=True)

# CJ / SA_unstrat ratio
ratio = cj_arr / sa_unstrat_arr
print(f"  CJ / SA_unstrat ratio: mean={np.mean(ratio):.4f}, CV={np.std(ratio)/np.mean(ratio)*100:.1f}%", flush=True)

# --- Profile PCA ---
print("\n--- Profile PCA (45-dim) ---", flush=True)
from numpy.linalg import svd
U, S, Vt = svd(profiles - profiles.mean(axis=0), full_matrices=False)
frac = (S ** 2) / np.sum(S ** 2)
for k in range(5):
    print(f"  PC{k+1}: {frac[k]*100:.1f}%", flush=True)
print(f"  Top 3: {sum(frac[:3])*100:.1f}%", flush=True)

# Regress CJ on top 3 PCs
scores = U[:, :3] * S[:3]
X_pca = np.column_stack([scores, np.ones(30)])
beta_pca = np.linalg.lstsq(X_pca, cj_arr, rcond=None)[0]
pred_pca = X_pca @ beta_pca
R2_pca = 1 - np.var(cj_arr - pred_pca) / np.var(cj_arr)
print(f"  R^2(CJ ~ top 3 profile PCs) = {R2_pca:.4f}", flush=True)

# --- Also test: does SA_unstrat predict CJ better than SA_global? ---
print("\n--- Linear models ---", flush=True)
# CJ ~ a * SA_unstrat + b
c1 = np.polyfit(sa_unstrat_arr, cj_arr, 1)
pred1 = c1[0] * sa_unstrat_arr + c1[1]
R2_1 = 1 - np.var(cj_arr - pred1) / np.var(cj_arr)
print(f"  CJ ~ SA_unstrat: slope={c1[0]:.4f}, R^2={R2_1:.4f}", flush=True)

# CJ ~ a * SA_global + b
c2 = np.polyfit(sa_global_arr, cj_arr, 1)
pred2 = c2[0] * sa_global_arr + c2[1]
R2_2 = 1 - np.var(cj_arr - pred2) / np.var(cj_arr)
print(f"  CJ ~ SA_global:  slope={c2[0]:.4f}, R^2={R2_2:.4f}", flush=True)

# --- VERDICT ---
print(f"\n{'='*70}", flush=True)
print("GTA-5 VERDICT", flush=True)
print(f"{'='*70}", flush=True)

if r_unstrat ** 2 > 0.90:
    verdict = "CJ IS a projection of the score field. CLOSED."
elif r_unstrat ** 2 > 0.70:
    verdict = "CJ is APPROXIMATELY a projection. Stratification adds conditioning."
else:
    verdict = "CJ contains information BEYOND simple score projection."

print(f"  corr(CJ, SA_unstrat) = {r_unstrat:.4f}, R^2 = {r_unstrat**2:.4f}", flush=True)
print(f"  {verdict}", flush=True)

# Save
results = {
    "corr_CJ_SA_unstrat": float(r_unstrat),
    "R2_CJ_SA_unstrat": float(r_unstrat ** 2),
    "corr_CJ_SA_global": float(r_global),
    "R2_CJ_SA_global": float(r_global ** 2),
    "ratio_CJ_SA_mean": float(np.mean(ratio)),
    "ratio_CJ_SA_CV": float(np.std(ratio) / np.mean(ratio) * 100),
    "profile_PCA_fracs": [float(f) for f in frac[:5]],
    "R2_CJ_top3PCs": float(R2_pca),
    "verdict": verdict,
}
out_path = os.path.join(DATA_DIR, "gta5_score_amplitude.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
