"""
Correct derivation of the 0.55 overcounting factor.
=====================================================

Key findings from derive_beta_and_055.py:
- corr(xi_down, xi_up) = +0.22 (POSITIVE, not negative!)
- corr(log2_pd_flat, log2_pu_flat) = -0.93 (this was confused with the above)
- Var(xi) / Var_indep = 1.22 (amplified, not reduced)
- The 0.55 factor is NOT from anticorrelation of perturbations

This script properly diagnoses the 0.55 factor using multiple eps values
and by comparing the perturbative formula to observed delta_kurtosis.

Author: David Alfyorov
"""
import numpy as np
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# GPU
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp
from scipy.stats import kurtosis as kurt
from scipy import stats
import json


def build_causal_link_gpu(pts_g, eps_val=0.0):
    t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
    dt = t_g[None, :] - t_g[:, None]
    dx = x_g[None, :] - x_g[:, None]
    dy = y_g[None, :] - y_g[:, None]
    dz = z_g[None, :] - z_g[:, None]

    if abs(eps_val) < 1e-15:
        tau2 = dt**2 - dx**2 - dy**2 - dz**2
    else:
        du = (dt + dz) / np.sqrt(2)
        x_i = x_g[:, None]; y_i = y_g[:, None]
        H_int = x_i**2 + x_i * dx + dx**2 / 3 - y_i**2 - y_i * dy - dy**2 / 3
        delta_sigma = eps_val / 2 * du**2 * H_int
        tau2 = dt**2 - dx**2 - dy**2 - dz**2 - 2 * delta_sigma

    C = ((dt > 0) & (tau2 > 0)).astype(cp.float32)
    C2 = C @ C
    L = ((C > 0.5) & (C2 < 0.5)).astype(cp.float32)
    return L


def compute_path_counts(L_np, order):
    N = L_np.shape[0]
    p_down = np.zeros(N, dtype=np.float64)
    p_up = np.zeros(N, dtype=np.float64)

    for idx in order:
        preds = np.where(L_np[:, idx] > 0.5)[0]
        p_down[idx] = p_down[preds].sum() if len(preds) > 0 else 1.0

    for idx in reversed(order.tolist()):
        succs = np.where(L_np[idx, :] > 0.5)[0]
        p_up[idx] = p_up[succs].sum() if len(succs) > 0 else 1.0

    return p_down, p_up


N = 2000
T = 1.0
M = 20  # more sprinklings for precision
eps_list = [1.0, 2.0, 5.0, 10.0]  # range of eps

print("=" * 70)
print(f"FACTOR 0.55: CORRECT DIAGNOSIS (N={N}, M={M})")
print("=" * 70)

results = {}
t0 = time.time()

for eps_val in eps_list:
    dk_obs_list = []
    formula_pred_list = []

    for m in range(M):
        np.random.seed(3000 + m)
        pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
        pts_g = cp.asarray(pts)
        order = np.argsort(pts[:, 0])

        # Flat
        L_flat = build_causal_link_gpu(pts_g, 0.0)
        L_flat_np = cp.asnumpy(L_flat)
        pd_flat, pu_flat = compute_path_counts(L_flat_np, order)

        # PP-wave
        L_ppw = build_causal_link_gpu(pts_g, eps_val)
        L_ppw_np = cp.asnumpy(L_ppw)
        pd_ppw, pu_ppw = compute_path_counts(L_ppw_np, order)

        # Y = log2(P+1), P = p_down * p_up
        P_flat = pd_flat * pu_flat
        P_ppw = pd_ppw * pu_ppw
        Y_flat = np.log2(P_flat + 1)
        Y_ppw = np.log2(P_ppw + 1)

        # NO boundary exclusion (consistent with our discovery scripts)
        # Observed delta kurtosis
        dk_obs = kurt(Y_ppw, fisher=True) - kurt(Y_flat, fisher=True)

        # Perturbation field (using ALL elements, not just interior)
        xi = (Y_ppw - Y_flat) / eps_val

        # Formula prediction
        Y0 = Y_flat
        sigma0_sq = np.var(Y0)
        kappa0 = kurt(Y0, fisher=True)
        xi_sq = np.mean(xi**2)
        Y0sq_xi_sq = np.mean(Y0**2 * xi**2)

        formula = eps_val**2 * (
            6 * Y0sq_xi_sq / sigma0_sq**2
            - 2 * (kappa0 + 3) * xi_sq / sigma0_sq
        )

        dk_obs_list.append(dk_obs)
        formula_pred_list.append(formula)

    dk_obs_arr = np.array(dk_obs_list)
    formula_arr = np.array(formula_pred_list)
    factor_arr = dk_obs_arr / formula_arr

    results[eps_val] = {
        "dk_obs_mean": float(dk_obs_arr.mean()),
        "dk_obs_se": float(dk_obs_arr.std() / np.sqrt(M)),
        "formula_mean": float(formula_arr.mean()),
        "formula_se": float(formula_arr.std() / np.sqrt(M)),
        "factor_mean": float(factor_arr.mean()),
        "factor_se": float(factor_arr.std() / np.sqrt(M)),
    }

    print(f"eps={eps_val:5.1f}: dk_obs={dk_obs_arr.mean():+.6f} +/- {dk_obs_arr.std()/np.sqrt(M):.6f}  "
          f"formula={formula_arr.mean():+.6f}  "
          f"factor={factor_arr.mean():.4f} +/- {factor_arr.std()/np.sqrt(M):.4f}")

    if (time.time() - t0) > 30:
        print(f"  [{time.time()-t0:.1f}s elapsed]")

print()
print("=" * 70)
print("ANALYSIS: What is the 0.55 factor?")
print("=" * 70)
for eps_val in eps_list:
    r = results[eps_val]
    print(f"eps={eps_val:5.1f}: observed/formula = {r['factor_mean']:.4f} +/- {r['factor_se']:.4f}")

print()
print("If the factor is approximately constant across eps, it's a structural overcounting.")
print("If it varies, higher-order terms in eps are important.")
print()

# Also check: does the formula work at SMALL eps?
# At small eps, O(eps^4) terms are negligible and the factor should be closer to 1
# (or whatever the true value is).
print("Testing at VERY small eps (where perturbation theory is most accurate)...")
for eps_small in [0.5, 1.0]:
    dk_list = []
    form_list = []
    for m in range(M):
        np.random.seed(3000 + m)
        pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
        pts_g = cp.asarray(pts)
        order = np.argsort(pts[:, 0])

        L_flat = build_causal_link_gpu(pts_g, 0.0)
        L_flat_np = cp.asnumpy(L_flat)
        pd_flat, pu_flat = compute_path_counts(L_flat_np, order)

        L_ppw = build_causal_link_gpu(pts_g, eps_small)
        L_ppw_np = cp.asnumpy(L_ppw)
        pd_ppw, pu_ppw = compute_path_counts(L_ppw_np, order)

        P_flat = pd_flat * pu_flat
        P_ppw = pd_ppw * pu_ppw
        Y_flat = np.log2(P_flat + 1)
        Y_ppw = np.log2(P_ppw + 1)

        dk = kurt(Y_ppw, fisher=True) - kurt(Y_flat, fisher=True)
        xi = (Y_ppw - Y_flat) / eps_small
        Y0 = Y_flat
        s2 = np.var(Y0)
        k0 = kurt(Y0, fisher=True)
        form = eps_small**2 * (6*np.mean(Y0**2*xi**2)/s2**2 - 2*(k0+3)*np.mean(xi**2)/s2)
        dk_list.append(dk)
        form_list.append(form)

    dk_a = np.array(dk_list)
    form_a = np.array(form_list)
    fac = dk_a / form_a
    print(f"  eps={eps_small}: dk={dk_a.mean():+.6f}, formula={form_a.mean():+.6f}, "
          f"factor={fac.mean():.4f} +/- {fac.std()/np.sqrt(M):.4f}")

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.1f}s")

# Save
outdir = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
with open(os.path.join(outdir, "factor_055_diagnosis.json"), "w") as f:
    json.dump(results, f, indent=2)
