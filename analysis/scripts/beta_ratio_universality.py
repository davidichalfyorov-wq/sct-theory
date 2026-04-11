"""
Test universality of beta_int/beta_bdy ratio across N.
=========================================================

Claim: beta_int/beta_bdy = -0.595 (from hyperboloid integral ratio).
This should be N-independent if it's a genuine geometric property.

Test at N=500,1000,1500,2000,3000,5000 with M=15 sprinklings each.
Use DIAMOND sprinkling + CORRECT pp-wave (midpoint formula).

Also test at different eps to check eps-independence.
Also test on Schwarzschild to check geometry-independence.

Author: David Alfyorov
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad
from scipy import stats

# GPU
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp

T = 1.0


def decompose_beta_gpu(N, eps_val, M, seeds_start=100):
    """Compute beta decomposition using GPU. Returns (beta_tot, beta_int, beta_bdy) arrays."""
    bt, bi, bb = [], [], []

    for m in range(M):
        rng = np.random.default_rng(m * 1000 + seeds_start)
        pts = sprinkle_4d(N, T, rng)
        f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0
        pts_g = cp.asarray(pts.astype(np.float32))

        t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
        dt = t_g[None, :] - t_g[:, None]
        dx = x_g[None, :] - x_g[:, None]
        dy = y_g[None, :] - y_g[:, None]
        dz = z_g[None, :] - z_g[:, None]
        dr2 = dx**2 + dy**2 + dz**2
        mink = dt**2 - dr2

        # Flat
        C_flat = ((mink > 0) & (dt > 0))
        C_f32 = C_flat.astype(cp.float32)
        C2_f = C_f32 @ C_f32
        L_flat = (C_flat & (C2_f < 0.5))

        # PP-wave
        xm = (x_g[None, :] + x_g[:, None]) / 2
        ym = (y_g[None, :] + y_g[:, None]) / 2
        f_mid = xm**2 - ym**2
        du_cart = dt + dz
        corr = eps_val * f_mid * du_cart**2 / 2
        C_ppw = ((mink > corr) & (dt > 0))
        C_p32 = C_ppw.astype(cp.float32)
        C2_p = C_p32 @ C_p32
        L_ppw = (C_ppw & (C2_p < 0.5))

        k_flat = cp.asnumpy((L_flat.astype(cp.float32).sum(0) +
                             L_flat.astype(cp.float32).sum(1)))
        k_ppw = cp.asnumpy((L_ppw.astype(cp.float32).sum(0) +
                            L_ppw.astype(cp.float32).sum(1)))
        dk_total = k_ppw - k_flat

        # Decompose
        new_causal = ((~C_flat) & C_ppw)
        lost_causal = (C_flat & (~C_ppw))
        links_gained = (new_causal & L_ppw)
        links_lost = (lost_causal & L_flat)
        dk_bdy = cp.asnumpy(
            (links_gained.astype(cp.float32).sum(0) +
             links_gained.astype(cp.float32).sum(1)) -
            (links_lost.astype(cp.float32).sum(0) +
             links_lost.astype(cp.float32).sum(1)))
        dk_int = dk_total - dk_bdy

        mask = k_flat > 5
        f_m = f_vals[mask]
        k0 = np.maximum(k_flat[mask], 1)

        s_t, _, _, _, _ = stats.linregress(f_m, dk_total[mask] / k0)
        s_i, _, _, _, _ = stats.linregress(f_m, dk_int[mask] / k0)
        s_b, _, _, _, _ = stats.linregress(f_m, dk_bdy[mask] / k0)
        bt.append(s_t / eps_val)
        bi.append(s_i / eps_val)
        bb.append(s_b / eps_val)

        del pts_g, C_flat, C_ppw, L_flat, L_ppw, C_f32, C_p32, C2_f, C2_p
        del new_causal, lost_causal, links_gained, links_lost
        cp.get_default_memory_pool().free_all_blocks()

    return np.array(bt), np.array(bi), np.array(bb)


# ====================================================================
# TEST 1: N-universality at fixed eps=0.1
# ====================================================================
print("=" * 70)
print("TEST 1: RATIO UNIVERSALITY ACROSS N (eps=0.1)")
print("=" * 70)

eps_val = 0.1
M = 15

for N in [500, 1000, 1500, 2000, 3000, 5000]:
    t0 = time.time()
    bt, bi, bb = decompose_beta_gpu(N, eps_val, M)
    ratio = bi / bb  # element-wise ratio for each sprinkling
    elapsed = time.time() - t0

    print(f"N={N:5d}: int={bi.mean():+.3f}+/-{bi.std()/np.sqrt(M):.3f}  "
          f"bdy={bb.mean():+.3f}+/-{bb.std()/np.sqrt(M):.3f}  "
          f"ratio={ratio.mean():.4f}+/-{ratio.std()/np.sqrt(M):.4f}  "
          f"tot={bt.mean():+.3f}  [{elapsed:.1f}s]")

print()
print("ANALYTICAL PREDICTION: ratio = -0.595")

# ====================================================================
# TEST 2: eps-independence at fixed N=2000
# ====================================================================
print()
print("=" * 70)
print("TEST 2: RATIO vs EPS (N=2000)")
print("=" * 70)

N = 2000
M = 10
for eps_val in [0.02, 0.05, 0.1, 0.2, 0.5]:
    bt, bi, bb = decompose_beta_gpu(N, eps_val, M, seeds_start=500)
    ratio = bi / bb
    print(f"eps={eps_val:.3f}: int={bi.mean():+.3f}  bdy={bb.mean():+.3f}  "
          f"ratio={ratio.mean():.4f}+/-{ratio.std()/np.sqrt(M):.4f}")

# ====================================================================
# TEST 3: Schwarzschild (different geometry)
# ====================================================================
print()
print("=" * 70)
print("TEST 3: SCHWARZSCHILD (N=2000, eps=0.005)")
print("=" * 70)

# For Schwarzschild we need the causal_schwarzschild function
from discovery_common import causal_schwarzschild

N = 2000
eps_val = 0.005
M = 10

bt_s, bi_s, bb_s = [], [], []
for m in range(M):
    rng = np.random.default_rng(m * 1000 + 700)
    pts = sprinkle_4d(N, T, rng)
    # f for Schwarzschild: f(r) = 1/r type, but let's use the same regression on f=(x^2-y^2)/2
    # Actually Schwarzschild perturbation is different, skip this test for now.
    # The causal_schwarzschild uses a different metric perturbation.
    pass

print("  [Schwarzschild test requires different f(x) definition — skipped for now]")
print("  [PP-wave ratio test is the primary universality check]")
