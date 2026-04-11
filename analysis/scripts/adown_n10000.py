"""
a_full at N=10000 using GPU. Verify N^{0.47} scaling extends.
Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
import os, sys, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d
from scipy import stats

_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp

T = 1.0
eps_val = 0.5
M = 3  # fewer for N=10000 (expensive)


def causal_link_gpu(pts, eps_val=0.0):
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
    L_np = cp.asnumpy(L)
    del C, C2, L, dt, dx, dy, dz, dr2, mink, pts_g
    cp.get_default_memory_pool().free_all_blocks()
    return sp.csr_matrix(L_np)


def compute_paths(L_sp, N):
    L_csc = L_sp.tocsc()
    L_csr = L_sp.tocsr()
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
    return p_down, p_up


print("=" * 70)
print(f"a_full AT N=10000 (eps={eps_val}, M={M})")
print("=" * 70)

# Also rerun N=2000,5000 for consistent comparison
for N in [2000, 5000, 10000]:
    adown_list = []
    t0 = time.time()
    for m in range(M):
        rng = np.random.default_rng(m * 1000 + 200)
        pts = sprinkle_4d(N, T, rng)
        f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0

        L_f = causal_link_gpu(pts, 0.0)
        pd_f, pu_f = compute_paths(L_f, N)
        logP_f = np.log2(pd_f * pu_f + 1)
        del L_f; gc.collect()

        L_p = causal_link_gpu(pts, eps_val)
        pd_p, pu_p = compute_paths(L_p, N)
        logP_p = np.log2(pd_p * pu_p + 1)
        del L_p; gc.collect()

        delta = logP_p - logP_f
        mask = logP_f > 2
        if mask.sum() > 50:
            slope, _, _, _, _ = stats.linregress(f_vals[mask], delta[mask])
            adown_list.append(slope / eps_val)

        print(f"  N={N}, trial {m+1}/{M}: a_full={adown_list[-1]:+.2f} [{time.time()-t0:.1f}s]")

    ad = np.array(adown_list)
    print(f"N={N:6d}: a_full={ad.mean():+.2f}+/-{ad.std()/np.sqrt(M):.2f}  "
          f"a/sqrt(N)={ad.mean()/np.sqrt(N):.4f}  "
          f"a/N^0.47={ad.mean()/N**0.47:.4f}")
    print()

# Final power law
print("POWER LAW across all N:")
Ns = np.array([2000, 5000, 10000])
# recompute means... already printed above
