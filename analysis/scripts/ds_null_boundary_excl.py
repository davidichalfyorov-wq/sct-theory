"""
dS null test with boundary exclusion.
======================================

Previous result: dS gives small but nonzero signal at N=5000 (d=-1.01, p=8e-6).
This could be from boundary elements contaminating the kurtosis.

Test: compute path_kurtosis using ONLY interior elements (80% interior exclusion).
If signal disappears => it was boundary artifact => dS null PASSES.
If signal persists => path_kurtosis is NOT purely Weyl-sensitive.

Also run pp-wave and flat as controls.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
import os, sys, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad
from scipy.stats import kurtosis as kurt
from scipy import stats as sp_stats

T = 1.0
N = 5000
M = 30  # enough for good statistics

print("=" * 70)
print(f"dS NULL TEST WITH BOUNDARY EXCLUSION (N={N}, M={M})")
print("=" * 70)


def causal_desitter(pts, H):
    """de Sitter causal condition: conformally flat with scale factor a(t)=exp(Ht)."""
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2

    # Conformal time: eta = -1/(aH) = -exp(-Ht)/H
    # For small H*t: a(t) ≈ 1 + Ht + (Ht)^2/2
    # Causal condition: conformal distance < conformal time separation
    # For small H: use perturbative correction
    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2
    a2 = np.exp(2 * H * tm)  # a^2 at midpoint
    mink = dt**2 - dr2
    # dS correction: effective metric ds^2 = a^2(t) * eta_munu dx^mu dx^nu
    # Causal iff a^2 * (dt^2 - dr^2) > 0 at leading order
    # More precisely: conformal factor doesn't change causal structure!
    # dS is conformally flat => causal structure = Minkowski causal structure.
    # BUT: sprinkling density changes! rho_dS = rho_flat * a^4 (d=4).
    # For CRN: same coordinates, same causal structure, different DENSITY.
    # The difference comes from the VOLUME ELEMENT, not causal condition.

    # Actually for CRN test: we sprinkle in FLAT coordinates but change
    # the density to match dS. The causal condition is the SAME (conformally flat).
    # The observable changes because the Hasse diagram changes due to different
    # effective sprinkling density at different times.

    # Simpler approach: use the perturbative dS metric
    # ds^2 ≈ -(1 - H^2 r^2) dt^2 + (1 + H^2 r^2/3) dr^2 + ...
    # For small H: correction to causal condition is O(H^2)

    # Use midpoint approximation like pp-wave:
    rm2 = ((x[np.newaxis, :] + x[:, np.newaxis])**2 +
           (y[np.newaxis, :] + y[:, np.newaxis])**2 +
           (z[np.newaxis, :] + z[:, np.newaxis])**2) / 4
    corr = H**2 * rm2 * dt**2 / 3  # O(H^2) correction
    return ((mink > corr) & (dt > 0)).astype(np.float64)


def path_kurtosis_with_exclusion(C, N, pts, frac_interior=0.8):
    """Compute path_kurtosis using only interior elements."""
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L = C_sp - C_sp.multiply(has_int)
    L.eliminate_zeros()
    L_csc = L.tocsc()
    L_csr = L.tocsr()

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

    # Boundary exclusion: keep only elements in the interior
    # "Interior" = elements whose time coordinate is in the middle frac_interior
    t = pts[:, 0]
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    # Distance from boundary: R_p = T/2 - |t| - r
    R_p = T / 2 - np.abs(t) - r
    R_max = R_p.max()
    threshold = R_max * (1 - frac_interior)
    interior = R_p > threshold

    if interior.sum() < 50:
        return float('nan')

    return float(kurt(log_p[interior], fisher=True))


# ====================================================================
# Run tests
# ====================================================================
print(f"\nUsing 80% interior exclusion (boundary elements removed)")
print()

# 1. Flat vs Flat (sanity check: should give d≈0)
print("--- FLAT vs FLAT (sanity) ---")
# Skip: trivially d=0

# 2. PP-wave eps=5 (should give strong signal)
print("--- PP-WAVE eps=5 (positive control) ---")
dpk_ppw = []
for trial in range(M):
    rng = np.random.default_rng(trial * 1000 + 100)
    pts = sprinkle_4d(N, T, rng)
    C_f = causal_flat(pts)
    pk_f = path_kurtosis_with_exclusion(C_f, N, pts, 0.8)
    del C_f; gc.collect()
    C_p = causal_ppwave_quad(pts, 5.0)
    pk_p = path_kurtosis_with_exclusion(C_p, N, pts, 0.8)
    del C_p; gc.collect()
    if not (np.isnan(pk_f) or np.isnan(pk_p)):
        dpk_ppw.append(pk_p - pk_f)
    if (trial + 1) % 10 == 0:
        print(f"  trial {trial+1}/{M}")

dpk_ppw = np.array(dpk_ppw)
d_ppw = dpk_ppw.mean() / (dpk_ppw.std() / np.sqrt(len(dpk_ppw))) if dpk_ppw.std() > 0 else 0
print(f"  PP-wave: Dpk={dpk_ppw.mean():+.6f}+/-{dpk_ppw.std()/np.sqrt(len(dpk_ppw)):.6f}  "
      f"d={d_ppw:.2f}  p={sp_stats.ttest_1samp(dpk_ppw, 0).pvalue:.2e}")
print()

# 3. de Sitter H=0.5 (null test)
print("--- de SITTER H=0.5 (NULL TEST) ---")
dpk_ds = []
for trial in range(M):
    rng = np.random.default_rng(trial * 1000 + 100)
    pts = sprinkle_4d(N, T, rng)
    C_f = causal_flat(pts)
    pk_f = path_kurtosis_with_exclusion(C_f, N, pts, 0.8)
    del C_f; gc.collect()
    C_ds = causal_desitter(pts, 0.5)
    pk_ds = path_kurtosis_with_exclusion(C_ds, N, pts, 0.8)
    del C_ds; gc.collect()
    if not (np.isnan(pk_f) or np.isnan(pk_ds)):
        dpk_ds.append(pk_ds - pk_f)
    if (trial + 1) % 10 == 0:
        print(f"  trial {trial+1}/{M}")

dpk_ds = np.array(dpk_ds)
d_ds = dpk_ds.mean() / (dpk_ds.std() / np.sqrt(len(dpk_ds))) if dpk_ds.std() > 0 else 0
p_ds = sp_stats.ttest_1samp(dpk_ds, 0).pvalue if len(dpk_ds) > 2 else 1.0
print(f"  dS: Dpk={dpk_ds.mean():+.6f}+/-{dpk_ds.std()/np.sqrt(len(dpk_ds)):.6f}  "
      f"d={d_ds:.2f}  p={p_ds:.2e}")
print()

if abs(d_ds) < 2:
    print("  => dS NULL TEST PASSES with boundary exclusion!")
elif abs(d_ds) < 3:
    print("  => dS signal MARGINAL with boundary exclusion")
else:
    print(f"  => dS signal PERSISTS (d={d_ds:.1f}) even with boundary exclusion")

# 4. Also run WITHOUT exclusion for comparison
print()
print("--- de SITTER H=0.5 (NO exclusion, for comparison) ---")
dpk_ds_no = []
for trial in range(min(M, 15)):
    rng = np.random.default_rng(trial * 1000 + 100)
    pts = sprinkle_4d(N, T, rng)
    C_f = causal_flat(pts)
    _, Y_f = None, np.log2(np.ones(N))  # dummy
    pk_f_val = kurt(np.log2(np.ones(N) + 1), fisher=True)
    # Actually compute properly
    C_sp = sp.csr_matrix(C_f)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L = C_sp - C_sp.multiply(has_int)
    L.eliminate_zeros()
    L_csc = L.tocsc()
    L_csr = L.tocsr()
    pd = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            pd[j] = np.sum(pd[parents])
        if pd[j] == 0: pd[j] = 1.0
    pu = np.ones(N, dtype=np.float64)
    for i in range(N-1,-1,-1):
        children = L_csr.getrow(i).indices
        if len(children) > 0:
            pu[i] = np.sum(pu[children])
        if pu[i] == 0: pu[i] = 1.0
    Y_f = np.log2(pd*pu+1)
    pk_f_val = kurt(Y_f, fisher=True)
    del C_f, C_sp, C2, L; gc.collect()

    C_ds = causal_desitter(pts, 0.5)
    C_sp = sp.csr_matrix(C_ds)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L = C_sp - C_sp.multiply(has_int)
    L.eliminate_zeros()
    L_csc = L.tocsc()
    L_csr = L.tocsr()
    pd2 = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_csc.getcol(j).indices
        if len(parents) > 0:
            pd2[j] = np.sum(pd2[parents])
        if pd2[j] == 0: pd2[j] = 1.0
    pu2 = np.ones(N, dtype=np.float64)
    for i in range(N-1,-1,-1):
        children = L_csr.getrow(i).indices
        if len(children) > 0:
            pu2[i] = np.sum(pu2[children])
        if pu2[i] == 0: pu2[i] = 1.0
    Y_ds = np.log2(pd2*pu2+1)
    pk_ds_val = kurt(Y_ds, fisher=True)
    del C_ds, C_sp, C2, L; gc.collect()

    dpk_ds_no.append(pk_ds_val - pk_f_val)

dpk_ds_no = np.array(dpk_ds_no)
d_no = dpk_ds_no.mean() / (dpk_ds_no.std()/np.sqrt(len(dpk_ds_no))) if dpk_ds_no.std() > 0 else 0
print(f"  dS (no excl): Dpk={dpk_ds_no.mean():+.6f}+/-{dpk_ds_no.std()/np.sqrt(len(dpk_ds_no)):.6f}  d={d_no:.2f}")
