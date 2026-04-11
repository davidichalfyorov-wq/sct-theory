"""
a_down scaling at large N using GPU for causal matrix + link construction.
Path counts still CPU (sequential DAG traversal).

Verifies a_down ~ N^{0.35} (from alpha(N) ~ N^{-0.31} * CLT).

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
import os, sys, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from discovery_common import sprinkle_4d
from scipy import stats

# GPU for causal matrix
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp

T = 1.0
eps_val = 0.5
M = 5  # fewer for large N


def causal_link_gpu(pts, eps_val=0.0):
    """Build link matrix using GPU for causal+C2, return sparse CPU."""
    N = len(pts)
    pts_g = cp.asarray(pts.astype(np.float32))
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
        xm = (x_g[None, :] + x_g[:, None]) / 2
        ym = (y_g[None, :] + y_g[:, None]) / 2
        f = xm**2 - ym**2
        du = dt + dz
        corr = eps_val * f * du**2 / 2
        C = ((mink > corr) & (dt > 0)).astype(cp.float32)

    C2 = C @ C
    L = ((C > 0.5) & (C2 < 0.5)).astype(cp.float32)

    # Transfer to CPU sparse
    L_np = cp.asnumpy(L)
    del C, C2, L, dt, dx, dy, dz, dr2, mink, pts_g
    cp.get_default_memory_pool().free_all_blocks()
    return sp.csr_matrix(L_np)


def compute_p_down(L_sp, N):
    """Compute p_down from sparse link matrix."""
    L_csc = L_sp.tocsc()
    p_down = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            p_down[j] = np.sum(p_down[parents])
        if p_down[j] == 0:
            p_down[j] = 1.0
    return p_down


def compute_p_up(L_sp, N):
    """Compute p_up from sparse link matrix."""
    L_csr = L_sp.tocsr()
    p_up = np.ones(N, dtype=np.float64)
    for i in range(N - 1, -1, -1):
        children = L_csr.getrow(i).indices
        if len(children) > 0:
            p_up[i] = np.sum(p_up[children])
        if p_up[i] == 0:
            p_up[i] = 1.0
    return p_up


print("=" * 70)
print(f"a_down SCALING AT LARGE N (eps={eps_val}, M={M}, GPU causal)")
print("=" * 70)

N_list = [1000, 2000, 3000, 5000]

for N in N_list:
    adown_list = []
    t0 = time.time()

    for m in range(M):
        rng = np.random.default_rng(m * 1000 + 200)
        pts = sprinkle_4d(N, T, rng)
        f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0

        # Flat
        L_f = causal_link_gpu(pts, 0.0)
        pd_f = compute_p_down(L_f, N)
        pu_f = compute_p_up(L_f, N)
        logP_f = np.log2(pd_f * pu_f + 1)
        del L_f; gc.collect()

        # PP-wave
        L_p = causal_link_gpu(pts, eps_val)
        pd_p = compute_p_down(L_p, N)
        pu_p = compute_p_up(L_p, N)
        logP_p = np.log2(pd_p * pu_p + 1)
        del L_p; gc.collect()

        # a_down from P = p_down * p_up (full observable, not just p_down)
        delta_logP = logP_p - logP_f
        mask = logP_f > 2
        if mask.sum() > 50:
            slope, _, r, p_val, se = stats.linregress(f_vals[mask], delta_logP[mask])
            a_full = slope / eps_val
            adown_list.append(a_full)

    ad = np.array(adown_list)
    elapsed = time.time() - t0
    print(f"N={N:5d}: a_full={ad.mean():+.3f}+/-{ad.std()/np.sqrt(M):.3f}  "
          f"a/sqrt(N)={ad.mean()/np.sqrt(N):.4f}  "
          f"a/N^0.35={ad.mean()/N**0.35:.4f}  "
          f"[{elapsed:.1f}s]")

# Power law fit
print("\nPOWER LAW FIT:")
Ns = np.array(N_list)
if len(adown_list) > 0:
    # Collect means
    means = []
    for N in N_list:
        rng_test = np.random.default_rng(200)
        pts_test = sprinkle_4d(N, T, rng_test)
        f_test = (pts_test[:, 1]**2 - pts_test[:, 2]**2) / 2.0
        L_f = causal_link_gpu(pts_test, 0.0)
        pd_f = compute_p_down(L_f, N)
        pu_f = compute_p_up(L_f, N)
        logP_f = np.log2(pd_f * pu_f + 1)
        del L_f; gc.collect()
        L_p = causal_link_gpu(pts_test, eps_val)
        pd_p = compute_p_down(L_p, N)
        pu_p = compute_p_up(L_p, N)
        logP_p = np.log2(pd_p * pu_p + 1)
        del L_p; gc.collect()
        delta = logP_p - logP_f
        mask = logP_f > 2
        slope, _, _, _, _ = stats.linregress(f_test[mask], delta[mask])
        means.append(abs(slope / eps_val))

    means = np.array(means)
    mask_pos = means > 0
    if mask_pos.sum() >= 2:
        fit = np.polyfit(np.log(Ns[mask_pos]), np.log(means[mask_pos]), 1)
        print(f"  a_full ~ N^{fit[0]:.3f} (expected 0.35 from CLT+alpha)")
        print(f"  Coefficient: {np.exp(fit[1]):.4f}")
