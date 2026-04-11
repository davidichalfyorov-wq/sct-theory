"""
Factor 0.55 diagnosis v2: using CORRECT pp-wave causal condition from discovery_common.py.
===========================================================================================

The correct formula (midpoint approximation, as in discovery_common.py):
  mink = dt^2 - dx^2 - dy^2 - dz^2
  corr = eps * (xm^2 - ym^2) * du^2 / 2    where xm=(xi+xj)/2, du=(dt+dz)
  Causal iff: mink > corr AND dt > 0

This is different from the Synge world function integral I was using.

Author: David Alfyorov
"""
import numpy as np
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp
from scipy.stats import kurtosis as kurt


def causal_link_gpu(pts_g, eps_val=0.0):
    """Build causal + link matrix using CORRECT pp-wave formula."""
    t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
    dt = t_g[None, :] - t_g[:, None]
    dx = x_g[None, :] - x_g[:, None]
    dy = y_g[None, :] - y_g[:, None]
    dz = z_g[None, :] - z_g[:, None]
    dr2 = dx**2 + dy**2 + dz**2
    mink = dt**2 - dr2

    if abs(eps_val) < 1e-15:
        C = ((mink > 0) & (dt > 0)).astype(cp.float32)
    else:
        # Midpoint approximation (same as discovery_common.py)
        xm = (x_g[None, :] + x_g[:, None]) / 2.0
        ym = (y_g[None, :] + y_g[:, None]) / 2.0
        f = xm**2 - ym**2
        du = dt + dz  # NOT divided by sqrt(2) — matches discovery_common.py
        corr = eps_val * f * du**2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(cp.float32)

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
M = 20

print("=" * 70)
print(f"FACTOR 0.55 v2: CORRECT PP-WAVE (N={N}, M={M})")
print("=" * 70)

eps_list = [1.0, 2.0, 5.0, 10.0]
t0 = time.time()

for eps_val in eps_list:
    dk_list = []
    form_list = []
    factor_list = []

    for m in range(M):
        np.random.seed(42 + m)  # match original seeds
        pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
        pts_g = cp.asarray(pts)
        order = np.argsort(pts[:, 0])

        L_flat = causal_link_gpu(pts_g, 0.0)
        L_flat_np = cp.asnumpy(L_flat)
        pd_flat, pu_flat = compute_path_counts(L_flat_np, order)

        L_ppw = causal_link_gpu(pts_g, eps_val)
        L_ppw_np = cp.asnumpy(L_ppw)
        pd_ppw, pu_ppw = compute_path_counts(L_ppw_np, order)

        P_flat = pd_flat * pu_flat
        P_ppw = pd_ppw * pu_ppw
        Y_flat = np.log2(P_flat + 1)
        Y_ppw = np.log2(P_ppw + 1)

        dk_obs = kurt(Y_ppw, fisher=True) - kurt(Y_flat, fisher=True)

        # Perturbation field
        xi = (Y_ppw - Y_flat) / eps_val

        # Formula
        Y0 = Y_flat
        s2 = np.var(Y0)
        k0 = kurt(Y0, fisher=True)
        xi2 = np.mean(xi**2)
        Y0xi2 = np.mean(Y0**2 * xi**2)

        form = eps_val**2 * (6 * Y0xi2 / s2**2 - 2 * (k0 + 3) * xi2 / s2)

        dk_list.append(dk_obs)
        form_list.append(form)
        if abs(form) > 1e-10:
            factor_list.append(dk_obs / form)

    dk_a = np.array(dk_list)
    form_a = np.array(form_list)
    fac_a = np.array(factor_list) if factor_list else np.array([np.nan])

    print(f"eps={eps_val:5.1f}: dk_obs={dk_a.mean():+.6f} +/- {dk_a.std()/np.sqrt(M):.6f}  "
          f"formula={form_a.mean():+.6f} +/- {form_a.std()/np.sqrt(M):.6f}  "
          f"factor={fac_a.mean():.4f} +/- {fac_a.std()/np.sqrt(max(len(fac_a),1)):.4f}")

print(f"\nElapsed: {time.time()-t0:.1f}s")

# Also check: first CRN realization at eps=5 to match memory file
print("\n--- Single CRN check (seed=42, eps=5, N=2000) ---")
np.random.seed(42)
pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
pts_g = cp.asarray(pts)
order = np.argsort(pts[:, 0])

L_flat = causal_link_gpu(pts_g, 0.0)
L_flat_np = cp.asnumpy(L_flat)
pd_flat, pu_flat = compute_path_counts(L_flat_np, order)

L_ppw = causal_link_gpu(pts_g, 5.0)
L_ppw_np = cp.asnumpy(L_ppw)
pd_ppw, pu_ppw = compute_path_counts(L_ppw_np, order)

P_flat = pd_flat * pu_flat
P_ppw = pd_ppw * pu_ppw
Y_flat = np.log2(P_flat + 1)
Y_ppw = np.log2(P_ppw + 1)

dk = kurt(Y_ppw, fisher=True) - kurt(Y_flat, fisher=True)
print(f"delta(path_kurtosis) = {dk:+.6f}")
print(f"Expected from memory: ~+0.060")
print(f"Mean Y_flat: {Y_flat.mean():.2f}, std: {Y_flat.std():.2f}, kurtosis: {kurt(Y_flat, fisher=True):.4f}")
print(f"Mean Y_ppw:  {Y_ppw.mean():.2f}, std: {Y_ppw.std():.2f}, kurtosis: {kurt(Y_ppw, fisher=True):.4f}")
print(f"Links flat: {int(L_flat.sum())}, ppw: {int(L_ppw.sum())}")
