"""
O(M²) separation for Schwarzschild path_kurtosis.
N=5000, r_min=0.10, M_seeds=15.

Masses: 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02
(3 new + 3 existing for cross-check)

Fit: Δκ(M) = α·M + β·M² with bootstrap CI on (α, β).
"""
import sys, time, json, numpy as np
sys.path.insert(0, 'analysis')
from sct_tools.hasse import crn_trial_schwarzschild

N = 5000
M_VALUES = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
M_SEEDS = 15
T = 1.0
R_MIN = 0.10

# C_1,rms from independent analysis formula
a = R_MIN
C1_rms = np.sqrt(8) * (T - 2*a) / (T**2 + 4*T*a + 12*a**2)
print(f"C_1,rms(T={T}, r_min={R_MIN}) = {C1_rms:.4f}", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"O(M²) FIT: N={N}, r_min={R_MIN}, M_seeds={M_SEEDS}", flush=True)
print(f"Masses: {M_VALUES}", flush=True)
print(f"{'='*70}", flush=True)

results = {}
t_start = time.time()

for M in M_VALUES:
    print(f"\n--- M = {M} ---", flush=True)
    dks = []
    for m in range(M_SEEDS):
        seed = 3000000 + int(M * 100000) + m
        t0 = time.time()
        dk = crn_trial_schwarzschild(N, M, seed, T=T, r_min=R_MIN)
        dt = time.time() - t0
        dks.append(dk)
        print(f"  m={m:2d}: dk={dk:+.6f} ({dt:.1f}s)", flush=True)

    dks = np.array(dks)
    mn = dks.mean()
    se = dks.std(ddof=1) / np.sqrt(M_SEEDS)
    d = mn / se if se > 1e-15 else 0.0

    B_eff = mn / (M * N**0.25 * C1_rms)
    B_se = se / (M * N**0.25 * C1_rms)

    results[str(M)] = {
        'mean': float(mn), 'se': float(se), 'd': float(d),
        'B_eff': float(B_eff), 'B_se': float(B_se),
        'dks': dks.tolist(),
    }

    print(f"  dk_mean = {mn:+.6f} ± {se:.6f}, d = {d:.2f}", flush=True)
    print(f"  B_eff = {B_eff:.4f} ± {B_se:.4f}", flush=True)

elapsed = time.time() - t_start
print(f"\n{'='*70}", flush=True)
print(f"Data collection: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

# ─── ANALYSIS ─────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"ANALYSIS: LINEAR vs QUADRATIC", flush=True)
print(f"{'='*70}", flush=True)

Ms = np.array(M_VALUES)
dk_means = np.array([results[str(M)]['mean'] for M in M_VALUES])
dk_ses = np.array([results[str(M)]['se'] for M in M_VALUES])

# dk/M test (linear)
print(f"\ndk/M (constant if pure linear):", flush=True)
for i, M in enumerate(M_VALUES):
    print(f"  M={M}: dk/M = {dk_means[i]/M:.4f}", flush=True)

# dk/M² test (quadratic)
print(f"\ndk/M² (constant if pure quadratic):", flush=True)
for i, M in enumerate(M_VALUES):
    print(f"  M={M}: dk/M² = {dk_means[i]/M**2:.2f}", flush=True)

# ─── WEIGHTED LEAST SQUARES FIT ────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"WLS FIT: Δκ = α·M + β·M²", flush=True)
print(f"{'='*70}", flush=True)

# Design matrix: [M, M²], no intercept (Δκ(0) = 0 by construction)
X = np.column_stack([Ms, Ms**2])
W = np.diag(1.0 / dk_ses**2)  # inverse variance weights

# WLS: (X^T W X)^{-1} X^T W y
XtW = X.T @ W
beta_hat = np.linalg.solve(XtW @ X, XtW @ dk_means)
alpha_fit, beta_fit = beta_hat

# Covariance matrix
cov = np.linalg.inv(XtW @ X)
alpha_se_wls = np.sqrt(cov[0, 0])
beta_se_wls = np.sqrt(cov[1, 1])

# Residuals
resid = dk_means - X @ beta_hat
chi2 = float(resid @ W @ resid)
ndof = len(M_VALUES) - 2

print(f"  α (linear coeff)  = {alpha_fit:.4f} ± {alpha_se_wls:.4f}", flush=True)
print(f"  β (quadratic coeff) = {beta_fit:.2f} ± {beta_se_wls:.2f}", flush=True)
print(f"  χ²/ndof = {chi2:.2f}/{ndof} = {chi2/ndof if ndof > 0 else float('inf'):.2f}", flush=True)
print(f"  β/α = {beta_fit/alpha_fit:.2f} (relative importance of M² term)", flush=True)

# Compare: pure linear fit (β=0)
alpha_lin = float(np.sum(dk_means * Ms / dk_ses**2) / np.sum(Ms**2 / dk_ses**2))
resid_lin = dk_means - alpha_lin * Ms
chi2_lin = float(np.sum((resid_lin / dk_ses)**2))

print(f"\n  Pure linear fit: α_lin = {alpha_lin:.4f}", flush=True)
print(f"  χ²_lin/ndof = {chi2_lin:.2f}/{len(M_VALUES)-1} = {chi2_lin/(len(M_VALUES)-1):.2f}", flush=True)
print(f"  Δχ² (linear − quadratic) = {chi2_lin - chi2:.2f}", flush=True)

# ─── BOOTSTRAP ────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"BOOTSTRAP: 10000 resamples", flush=True)
print(f"{'='*70}", flush=True)

N_BOOT = 10000
rng = np.random.default_rng(42)

# Collect all per-seed dk arrays
all_dks = {M: np.array(results[str(M)]['dks']) for M in M_VALUES}

alpha_boot = np.empty(N_BOOT)
beta_boot = np.empty(N_BOOT)

for b in range(N_BOOT):
    # Resample seeds within each mass
    dk_b = np.empty(len(M_VALUES))
    se_b = np.empty(len(M_VALUES))
    for i, M in enumerate(M_VALUES):
        idx = rng.integers(0, M_SEEDS, size=M_SEEDS)
        sample = all_dks[M][idx]
        dk_b[i] = sample.mean()
        se_b[i] = sample.std(ddof=1) / np.sqrt(M_SEEDS)

    # WLS fit on bootstrap sample
    W_b = np.diag(1.0 / np.maximum(se_b, 1e-12)**2)
    XtW_b = X.T @ W_b
    try:
        b_hat = np.linalg.solve(XtW_b @ X, XtW_b @ dk_b)
        alpha_boot[b] = b_hat[0]
        beta_boot[b] = b_hat[1]
    except np.linalg.LinAlgError:
        alpha_boot[b] = np.nan
        beta_boot[b] = np.nan

# Remove NaN
mask = ~(np.isnan(alpha_boot) | np.isnan(beta_boot))
alpha_boot = alpha_boot[mask]
beta_boot = beta_boot[mask]

alpha_ci = np.percentile(alpha_boot, [2.5, 97.5])
beta_ci = np.percentile(beta_boot, [2.5, 97.5])
beta_frac_zero = np.mean(beta_boot > 0)  # fraction where β > 0

print(f"  α: {np.median(alpha_boot):.4f} [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}] (95% CI)", flush=True)
print(f"  β: {np.median(beta_boot):.2f} [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}] (95% CI)", flush=True)
print(f"  P(β > 0) = {beta_frac_zero:.3f} (should be ~0 if β negative)", flush=True)
print(f"  P(β < 0) = {1 - beta_frac_zero:.3f}", flush=True)

# ─── B_eff TREND ──────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"B_eff CONSISTENCY", flush=True)
print(f"{'='*70}", flush=True)
for M in M_VALUES:
    r = results[str(M)]
    print(f"  M={M}: B_eff = {r['B_eff']:.4f} ± {r['B_se']:.4f}", flush=True)

# ─── SAVE ─────────────────────────────────────────────────
out = {
    'N': N, 'T': T, 'r_min': R_MIN, 'M_seeds': M_SEEDS,
    'C1_rms': float(C1_rms),
    'M_values': M_VALUES,
    'results': results,
    'fit': {
        'alpha': float(alpha_fit), 'alpha_se': float(alpha_se_wls),
        'beta': float(beta_fit), 'beta_se': float(beta_se_wls),
        'chi2': float(chi2), 'ndof': int(ndof),
        'alpha_lin': float(alpha_lin), 'chi2_lin': float(chi2_lin),
    },
    'bootstrap': {
        'N_boot': N_BOOT,
        'alpha_median': float(np.median(alpha_boot)),
        'alpha_ci95': [float(alpha_ci[0]), float(alpha_ci[1])],
        'beta_median': float(np.median(beta_boot)),
        'beta_ci95': [float(beta_ci[0]), float(beta_ci[1])],
        'P_beta_positive': float(beta_frac_zero),
    },
    'elapsed_s': elapsed,
}
outpath = 'analysis/discovery_runs/run_001/schwarzschild_om2_fit.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}", flush=True)

print(f"\n{'='*70}", flush=True)
print("DONE", flush=True)
