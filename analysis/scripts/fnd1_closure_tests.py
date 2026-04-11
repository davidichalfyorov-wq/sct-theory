#!/usr/bin/env python3
"""FND-1 CLOSURE TESTS — all attacks + nice-to-haves."""
import sys, os, json
import numpy as np, gzip, pickle
from scipy.stats import kurtosis as scipy_kurtosis

def excess_kurtosis(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4 or np.var(x) < 1e-15:
        return 0.0
    return float(scipy_kurtosis(x, fisher=True, bias=True))

def compute_A_pq(Y0, delta, mask, strata, p, q):
    X = Y0[mask] - np.mean(Y0[mask])
    Xp = np.abs(X) ** p
    dYq = np.abs(delta[mask]) ** q
    strata_m = strata[mask]
    total = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(Xp)
        cov = np.mean(Xp[idx] * dYq[idx]) - np.mean(Xp[idx]) * np.mean(dYq[idx])
        total += w * cov
    return float(total)

def make_strata(pts, depth0, T, n_tau, n_rho, n_depth):
    t = pts[:, 0]
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    tau_hat = 2 * t / T
    r_max = T / 2 - np.abs(t)
    rho_hat = np.clip(r / np.maximum(r_max, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * n_tau / 2).astype(int), 0, n_tau - 1)
    rho_bin = np.clip(np.floor(rho_hat * n_rho).astype(int), 0, n_rho - 1)
    if n_depth > 1:
        max_d = max(int(depth0.max()), 1)
        depth_bin = np.clip((depth0 * n_depth) // (max_d + 1), 0, n_depth - 1)
    else:
        depth_bin = np.zeros(len(pts), dtype=int)
    return tau_bin * (n_rho * n_depth) + rho_bin * n_depth + depth_bin

DATA_DIR = "analysis/fnd1_data"

print("=" * 70, flush=True)
print("FND-1 CLOSURE TESTS", flush=True)
print("=" * 70, flush=True)

# ============================================================
# TEST 1: True continuum normalization
# ============================================================
print("\n=== TEST 1: True continuum normalization ===", flush=True)

nc = json.load(open(os.path.join(DATA_DIR, "normalized_convergence.json")))

# H_max from data
H_max_known = {3000: 12, 5000: 15}
for si in range(20):
    fpath = os.path.join(DATA_DIR, f"T1.00_seed{si:03d}.pkl.gz")
    try:
        with gzip.open(fpath, "rb") as f:
            d = pickle.load(f)
        H_max_known.setdefault(10000, [])
        if isinstance(H_max_known[10000], list):
            H_max_known[10000].append(int(d["depth0"].max()))
    except:
        pass
H_max_known[10000] = np.mean(H_max_known[10000])

# Extrapolate H_max for 15000, 20000
Ns_h = np.array([3000, 5000, 10000], dtype=float)
Hs_h = np.array([H_max_known[3000], H_max_known[5000], H_max_known[10000]])
a_h, b_h = np.polyfit(np.log(Ns_h), np.log(Hs_h), 1)
H_max_known[15000] = np.exp(a_h * np.log(15000) + b_h)
H_max_known[20000] = np.exp(a_h * np.log(20000) + b_h)

Atilde_12 = []
Atilde_22 = []
Ns_sorted = sorted([int(k) for k in nc if k.isdigit()])
for N in Ns_sorted:
    v = nc[str(N)]
    s0 = v["sigma0_mean"]
    H = H_max_known[N]
    a12 = v["raw_AE_12"]
    a22 = v["raw_AE_22"]
    at12 = a12 / (s0 * H**2)
    at22 = a22 / (s0**2 * H**2)
    Atilde_12.append(at12)
    Atilde_22.append(at22)
    print(f"  N={N:>5}: At12={at12:.4e}, At22={at22:.4e}, H={H:.1f}", flush=True)

cv12 = np.std(Atilde_12) / np.mean(Atilde_12) * 100
cv22 = np.std(Atilde_22) / np.mean(Atilde_22) * 100
steps = [f"{(Atilde_12[i]-Atilde_12[i-1])/Atilde_12[i-1]*100:+.1f}%" for i in range(1, len(Atilde_12))]
print(f"  CV(At12) = {cv12:.1f}%, steps: {steps}", flush=True)
print(f"  CV(At22) = {cv22:.1f}%", flush=True)
verdict1 = "CONVERGING" if cv12 < 10 else "NOT CONVERGING"
print(f"  VERDICT: {verdict1}", flush=True)

# ============================================================
# TEST 2: Strata sweep
# ============================================================
print("\n=== TEST 2: Strata sweep ===", flush=True)

# Average over 10 seeds
configs = [
    (1, 1, 1, "1 bin (global)"),
    (3, 1, 1, "3 bins"),
    (5, 1, 1, "5 bins"),
    (5, 3, 1, "15 bins"),
    (3, 3, 3, "27 bins"),
    (5, 3, 3, "45 bins (STD)"),
    (5, 5, 3, "75 bins"),
    (5, 5, 5, "125 bins"),
]

a12_per_config = {label: [] for _, _, _, label in configs}

for si in range(10):
    fpath = os.path.join(DATA_DIR, f"T1.00_seed{si:03d}.pkl.gz")
    with gzip.open(fpath, "rb") as f:
        d = pickle.load(f)
    Y0 = d["Y0"]; delta = d["delta_ppw"]; mask = d["mask"]
    pts = d["pts"]; depth0 = d["depth0"]; T_val = d["T"]

    for n_tau, n_rho, n_dep, label in configs:
        strata_custom = make_strata(pts, depth0, T_val, n_tau, n_rho, n_dep)
        a12 = compute_A_pq(Y0, delta, mask, strata_custom, 1, 2)
        a12_per_config[label].append(a12)

ref_mean = np.mean(a12_per_config["45 bins (STD)"])
for _, _, _, label in configs:
    vals = a12_per_config[label]
    mn = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    ratio = mn / ref_mean
    print(f"  {label:>20}: A12={mn:.5f}+-{se:.5f}, ratio_to_std={ratio:.3f}", flush=True)

all_means = [np.mean(a12_per_config[label]) for _, _, _, label in configs]
cv_strata = np.std(all_means) / np.mean(all_means) * 100
verdict2 = "STRATA-ROBUST" if cv_strata < 20 else "STRATA-DEPENDENT"
print(f"  CV across configs: {cv_strata:.1f}%", flush=True)
print(f"  VERDICT: {verdict2}", flush=True)

# ============================================================
# TEST 3: PCA at different T
# ============================================================
print("\n=== TEST 3: PCA at T=0.35, 0.70, 1.00 ===", flush=True)

for T_str in ["0.35", "0.70", "1.00"]:
    obs_matrix = []
    for si in range(30):
        fpath = os.path.join(DATA_DIR, f"T{T_str}_seed{si:03d}.pkl.gz")
        try:
            with gzip.open(fpath, "rb") as f:
                d = pickle.load(f)
        except:
            continue

        Y0 = d["Y0"]; mask = d["mask"]; strata = d["strata"]
        delta = d["delta_ppw"]
        if delta is None or np.all(delta == 0):
            continue

        X = Y0[mask] - np.mean(Y0[mask])
        dY = delta[mask]
        strata_m = strata[mask]

        obs = []
        for p, q in [(1, 2), (2, 2), (1, 1)]:
            Xp = np.abs(X) ** p
            dYq = np.abs(dY) ** q
            total = 0.0
            for label in np.unique(strata_m):
                idx = strata_m == label
                if idx.sum() < 3:
                    continue
                w = idx.sum() / len(Xp)
                cov = np.mean(Xp[idx] * dYq[idx]) - np.mean(Xp[idx]) * np.mean(dYq[idx])
                total += w * cov
            obs.append(total)

        Y_c = Y0 + delta
        dk = excess_kurtosis(Y_c[mask]) - excess_kurtosis(Y0[mask])
        obs.append(dk)
        obs_matrix.append(obs)

    if len(obs_matrix) < 10:
        print(f"  T={T_str}: {len(obs_matrix)} seeds, SKIP", flush=True)
        continue

    obs_matrix = np.array(obs_matrix)
    corr = np.corrcoef(obs_matrix.T)
    eigvals = np.linalg.eigvalsh(corr)[::-1]
    frac = eigvals / eigvals.sum()
    corr_a12_dk = corr[0, 3]
    print(f"  T={T_str} (n={len(obs_matrix)}): PC1={frac[0]:.3f}, PC2={frac[1]:.3f}, corr(A12,dk)={corr_a12_dk:+.3f}", flush=True)

# ============================================================
# TEST 4: Killer attack — R^2(A12, proxy)
# ============================================================
print("\n=== TEST 4: KILLER — is A12 just a volume proxy? ===", flush=True)

a12_seeds = []
link_change = []
dY_var_seeds = []
f_sq_seeds = []

for si in range(30):
    fpath = os.path.join(DATA_DIR, f"T1.00_seed{si:03d}.pkl.gz")
    with gzip.open(fpath, "rb") as f:
        d = pickle.load(f)

    Y0 = d["Y0"]; mask = d["mask"]; strata = d["strata"]
    delta = d["delta_ppw"]; pts = d["pts"]

    a12 = compute_A_pq(Y0, delta, mask, strata, 1, 2)
    a12_seeds.append(a12)

    # Link count change
    lc = d["links_ppw"] - d["links0"]
    link_change.append(lc)

    # Variance of delta Y
    dY_var = np.var(delta[mask])
    dY_var_seeds.append(dY_var)

    # Mean f^2
    f_val = (pts[mask, 1]**2 - pts[mask, 2]**2) / 2
    f_sq_seeds.append(np.mean(f_val**2))

a12_arr = np.array(a12_seeds)
link_arr = np.array(link_change)
dYvar_arr = np.array(dY_var_seeds)
fsq_arr = np.array(f_sq_seeds)

for name, proxy in [("delta_TC", link_arr), ("Var(dY)", dYvar_arr), ("mean(f^2)", fsq_arr)]:
    corr_val = np.corrcoef(a12_arr, proxy)[0, 1]
    R2 = corr_val ** 2
    print(f"  corr(A12, {name}) = {corr_val:+.3f}, R^2 = {R2:.3f}", flush=True)

# Per-element: R^2(dY^2, f^2)
with gzip.open(os.path.join(DATA_DIR, "T1.00_seed000.pkl.gz"), "rb") as f:
    d0 = pickle.load(f)
delta0 = d0["delta_ppw"]; mask0 = d0["mask"]; pts0 = d0["pts"]
f0 = (pts0[mask0, 1]**2 - pts0[mask0, 2]**2) / 2
R2_elem = np.corrcoef(delta0[mask0]**2, f0**2)[0, 1] ** 2
print(f"  Per-element R^2(dY^2, f^2) = {R2_elem:.4f}", flush=True)

# Multiple regression: A12 ~ c0 + c1*delta_TC + c2*Var(dY) + c3*mean(f^2)
X_reg = np.column_stack([link_arr, dYvar_arr, fsq_arr, np.ones(30)])
beta_reg = np.linalg.lstsq(X_reg, a12_arr, rcond=None)[0]
pred_reg = X_reg @ beta_reg
R2_multi = 1 - np.var(a12_arr - pred_reg) / np.var(a12_arr)
print(f"  Multiple R^2(A12 ~ TC+Var+f^2) = {R2_multi:.3f}", flush=True)

if R2_multi > 0.80:
    print("  KILLER VERDICT: A12 may be a PROXY (R^2 > 0.80)", flush=True)
elif R2_multi < 0.50:
    print("  KILLER VERDICT: A12 is NOT a simple proxy (R^2 < 0.50)", flush=True)
else:
    print("  KILLER VERDICT: AMBIGUOUS (0.50 < R^2 < 0.80)", flush=True)

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "test1_cv12": cv12, "test1_cv22": cv22, "test1_verdict": verdict1,
    "test2_cv_strata": cv_strata, "test2_verdict": verdict2,
    "test4_R2_multi": R2_multi,
    "test4_R2_elem_dY2_f2": R2_elem,
}
with open(os.path.join(DATA_DIR, "closure_tests.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70, flush=True)
print("ALL TESTS COMPLETE", flush=True)
print("Saved to analysis/fnd1_data/closure_tests.json", flush=True)
print("=" * 70, flush=True)
