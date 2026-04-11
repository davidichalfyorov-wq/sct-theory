"""
Factor 0.55 diagnosis v3: Using CORRECT sprinkling (diamond, not cube!)
=========================================================================

BUG FIX: Previous versions used np.random.uniform(0,T,(N,4)) = cube.
The correct function sprinkle_4d uses rejection sampling into a causal
diamond: |t| + r < T/2, centered at origin.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
import os, sys, time, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy import stats
from scipy.stats import kurtosis as kurt

# Use the ORIGINAL functions from discovery_common.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad


def path_kurtosis_val(C, N):
    """Exact copy from discovery_s4_wprop.py."""
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L = C_sp - C_sp.multiply(has_int)
    L.eliminate_zeros()
    L_csr = L.tocsr()
    L_csc = L.tocsc()

    p_down = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            p_down[j] = np.sum(p_down[parents])
        if p_down[j] == 0:
            p_down[j] = 1.0

    p_up = np.ones(N, dtype=np.float64)
    for i in range(N - 1, -1, -1):
        children = L_csr.getrow(i).indices
        if len(children) > 0:
            p_up[i] = np.sum(p_up[children])
        if p_up[i] == 0:
            p_up[i] = 1.0

    log_p = np.log2(p_down * p_up + 1)
    return float(kurt(log_p, fisher=True)), log_p, p_down, p_up


N = 2000
T = 1.0
M = 20

print("=" * 70)
print(f"FACTOR 0.55 v3: CORRECT DIAMOND SPRINKLING (N={N}, M={M})")
print("=" * 70)

# Step 1: Reproduce the memory numbers (eps=5, N=2000)
print("\n--- STEP 1: Reproduce memory delta(path_kurtosis) ---")
deltas_eps5 = []
for trial in range(M):
    rng = np.random.default_rng(trial * 1000 + 100)  # SAME seeds as discovery scripts
    pts = sprinkle_4d(N, T, rng)
    C_f = causal_flat(pts)
    pk_f, _, _, _ = path_kurtosis_val(C_f, N)
    del C_f; gc.collect()
    C_p = causal_ppwave_quad(pts, 5.0)
    pk_p, _, _, _ = path_kurtosis_val(C_p, N)
    del C_p; gc.collect()
    deltas_eps5.append(pk_p - pk_f)
    if (trial + 1) % 5 == 0:
        print(f"  trial {trial+1}/{M}: delta={deltas_eps5[-1]:+.6f}")

d = np.array(deltas_eps5)
print(f"\ndelta(path_kurtosis) at eps=5: {d.mean():+.6f} +/- {d.std()/np.sqrt(M):.6f}")
print(f"Expected from memory: +0.059749 +/- 0.00694")
print(f"Match: {'YES' if abs(d.mean() - 0.060) < 0.02 else 'NO'}")

# Step 2: Compute the perturbative formula and the factor
print("\n--- STEP 2: Perturbative formula and factor at multiple eps ---")
eps_list = [1.0, 2.0, 5.0, 10.0]

for eps_val in eps_list:
    dk_list = []
    form_list = []
    t0 = time.time()

    for trial in range(M):
        rng = np.random.default_rng(trial * 1000 + 100)
        pts = sprinkle_4d(N, T, rng)

        C_f = causal_flat(pts)
        pk_f, Y_flat, pd_f, pu_f = path_kurtosis_val(C_f, N)
        del C_f; gc.collect()

        C_p = causal_ppwave_quad(pts, eps_val)
        pk_p, Y_ppw, pd_p, pu_p = path_kurtosis_val(C_p, N)
        del C_p; gc.collect()

        dk = pk_p - pk_f
        dk_list.append(dk)

        # Perturbation field
        xi = (Y_ppw - Y_flat) / eps_val

        # Formula
        Y0 = Y_flat
        s2 = np.var(Y0)
        k0 = kurt(Y0, fisher=True)
        xi2 = np.mean(xi**2)
        Y0xi2 = np.mean(Y0**2 * xi**2)
        form = eps_val**2 * (6 * Y0xi2 / s2**2 - 2 * (k0 + 3) * xi2 / s2)
        form_list.append(form)

    dk_a = np.array(dk_list)
    form_a = np.array(form_list)
    fac = dk_a / form_a

    print(f"eps={eps_val:5.1f}: dk={dk_a.mean():+.6f}+/-{dk_a.std()/np.sqrt(M):.6f}  "
          f"formula={form_a.mean():+.6f}+/-{form_a.std()/np.sqrt(M):.6f}  "
          f"factor={fac.mean():.4f}+/-{fac.std()/np.sqrt(M):.4f}  "
          f"[{time.time()-t0:.1f}s]")

# Step 3: Decompose the 0.55 factor at eps=5
print("\n--- STEP 3: Decompose factor at eps=5 ---")
eps_val = 5.0
rho_xi_list = []
var_ratio_list = []
xi2_ratio_list = []
cross_ratio_list = []

for trial in range(min(M, 10)):  # fewer for speed
    rng = np.random.default_rng(trial * 1000 + 100)
    pts = sprinkle_4d(N, T, rng)

    C_f = causal_flat(pts)
    _, Y_flat, pd_f, pu_f = path_kurtosis_val(C_f, N)
    del C_f; gc.collect()

    C_p = causal_ppwave_quad(pts, eps_val)
    _, Y_ppw, pd_p, pu_p = path_kurtosis_val(C_p, N)
    del C_p; gc.collect()

    xi = (Y_ppw - Y_flat) / eps_val

    # Decompose xi into xi_down and xi_up
    log_pd_f = np.log2(np.maximum(pd_f, 1))
    log_pu_f = np.log2(np.maximum(pu_f, 1))
    log_pd_p = np.log2(np.maximum(pd_p, 1))
    log_pu_p = np.log2(np.maximum(pu_p, 1))

    xi_d = (log_pd_p - log_pd_f) / eps_val
    xi_u = (log_pu_p - log_pu_f) / eps_val

    mask = (pd_f > 1) & (pu_f > 1)
    if mask.sum() > 100:
        rho_xi = np.corrcoef(xi_d[mask], xi_u[mask])[0, 1]
        rho_xi_list.append(rho_xi)

        var_ratio = np.var(xi[mask]) / (np.var(xi_d[mask]) + np.var(xi_u[mask]))
        var_ratio_list.append(var_ratio)

        Y0 = Y_flat[mask]
        xi_m = xi[mask]
        xi_dm = xi_d[mask]
        xi_um = xi_u[mask]
        xi2_r = np.mean(xi_m**2) / (np.mean(xi_dm**2) + np.mean(xi_um**2))
        xi2_ratio_list.append(xi2_r)

        Y0xi2_total = np.mean(Y0**2 * xi_m**2)
        Y0xi2_indep = np.mean(Y0**2 * xi_dm**2) + np.mean(Y0**2 * xi_um**2)
        cross_ratio_list.append(Y0xi2_total / Y0xi2_indep)

        rho_Y = np.corrcoef(log_pd_f[mask], log_pu_f[mask])[0, 1]

    if (trial + 1) % 5 == 0:
        print(f"  trial {trial+1}: rho_xi={rho_xi:.3f}, var_ratio={var_ratio:.3f}")

if rho_xi_list:
    print(f"\ncorr(log_pd_flat, log_pu_flat) = {rho_Y:.4f}")
    print(f"corr(xi_down, xi_up) = {np.mean(rho_xi_list):.4f} +/- {np.std(rho_xi_list)/np.sqrt(len(rho_xi_list)):.4f}")
    print(f"Var(xi)/Var_indep = {np.mean(var_ratio_list):.4f}")
    print(f"<xi^2> ratio = {np.mean(xi2_ratio_list):.4f}")
    print(f"<Y0^2*xi^2> ratio = {np.mean(cross_ratio_list):.4f}")
