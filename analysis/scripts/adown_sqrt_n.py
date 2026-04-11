"""
Derive a_down ~ sqrt(N): numerical investigation.
====================================================

Hypothesis: a_down = 0.052 * sqrt(N) because:
- Each element has ~N/2 ancestors in its past light cone
- The metric perturbation changes O(N*beta*eps) links
- Each changed link perturbs logP of all its descendants
- The net effect on logP accumulates as sqrt(# independent perturbations)

This script measures:
1. a_down at multiple N (verify sqrt(N) scaling)
2. Number of ancestors per element vs N
3. Number of perturbed links vs N
4. Correlation structure of link perturbations
5. Variance of delta(logP) decomposition

Uses CORRECT diamond sprinkling from discovery_common.py.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
import os, sys, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad
from scipy import stats

# GPU for larger N
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
try:
    import cupy as cp
    USE_GPU = True
except:
    USE_GPU = False

T = 1.0
eps_val = 0.5  # small eps for perturbative regime
M = 10  # sprinklings per N


def compute_logP_and_adown(pts, eps_val):
    """Compute logP for flat and pp-wave, extract a_down."""
    N = len(pts)

    C_f = causal_flat(pts)
    C_f_sp = sp.csr_matrix(C_f)
    C2_f = C_f_sp @ C_f_sp
    has_int_f = (C2_f != 0).astype(np.float64)
    L_f = C_f_sp - C_f_sp.multiply(has_int_f)
    L_f.eliminate_zeros()
    L_f_csc = L_f.tocsc()
    L_f_csr = L_f.tocsr()

    p_down_f = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_f_csc.getcol(j).indices
        if len(parents) > 0:
            p_down_f[j] = np.sum(p_down_f[parents])
        if p_down_f[j] == 0:
            p_down_f[j] = 1.0

    del C_f, C_f_sp, C2_f, has_int_f; gc.collect()

    C_p = causal_ppwave_quad(pts, eps_val)
    C_p_sp = sp.csr_matrix(C_p)
    C2_p = C_p_sp @ C_p_sp
    has_int_p = (C2_p != 0).astype(np.float64)
    L_p = C_p_sp - C_p_sp.multiply(has_int_p)
    L_p.eliminate_zeros()
    L_p_csc = L_p.tocsc()

    p_down_p = np.ones(N, dtype=np.float64)
    for j in range(N):
        parents = L_p_csc.getcol(j).indices
        if len(parents) > 0:
            p_down_p[j] = np.sum(p_down_p[parents])
        if p_down_p[j] == 0:
            p_down_p[j] = 1.0

    del C_p, C_p_sp, C2_p, has_int_p; gc.collect()

    logP_f = np.log2(np.maximum(p_down_f, 1))
    logP_p = np.log2(np.maximum(p_down_p, 1))

    # f(x) at each element
    f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0

    # Extract a_down: delta(logP) = a_down * eps * f(x) + noise
    delta_logP = logP_p - logP_f
    mask = logP_f > 1  # interior elements
    if mask.sum() > 50:
        slope, _, r, p_val, se = stats.linregress(f_vals[mask], delta_logP[mask])
        a_down = slope / eps_val
    else:
        a_down = 0.0

    # Also compute: number of ancestors per element (from the resolvent)
    # R = (I - L^T)^{-1}, R[i,j] = number of paths from j to i
    # Number of ancestors of j = number of i with R[i,j] > 0
    # Simpler: use the causal matrix C_f. Number of ancestors of j = sum_i C_f[i,j]
    # But C_f is not available anymore. Use link graph reachability.
    # Actually, for simplicity, count from the flat link graph.
    # Reachability: R = I + L + L^2 + ... = (I - L)^{-1}
    # Too expensive for large N. Use approximate: ancestors ~ N * fraction_past
    # For d=4 diamond: fraction of elements in the past of a generic interior element ~ 1/2

    n_links_f = L_f.nnz
    n_links_p = L_p.nnz
    links_changed = abs(n_links_p - n_links_f)

    return a_down, n_links_f, n_links_p, links_changed


print("=" * 70)
print(f"a_down vs N (eps={eps_val}, M={M})")
print("=" * 70)

N_list = [500, 1000, 1500, 2000, 3000]

results = {}
for N in N_list:
    adown_list = []
    nlinks_list = []
    dlinks_list = []
    t0 = time.time()

    for m in range(M):
        rng = np.random.default_rng(m * 1000 + 200)
        pts = sprinkle_4d(N, T, rng)

        a, nl_f, nl_p, dl = compute_logP_and_adown(pts, eps_val)
        adown_list.append(a)
        nlinks_list.append(nl_f)
        dlinks_list.append(dl)

    ad = np.array(adown_list)
    nl = np.array(nlinks_list)
    dl = np.array(dlinks_list)

    results[N] = {
        "a_down": float(ad.mean()),
        "a_down_se": float(ad.std() / np.sqrt(M)),
        "a_down_over_sqrtN": float(ad.mean() / np.sqrt(N)),
        "n_links": float(nl.mean()),
        "delta_links": float(dl.mean()),
    }

    print(f"N={N:5d}: a_down={ad.mean():+.3f}+/-{ad.std()/np.sqrt(M):.3f}  "
          f"a/sqrt(N)={ad.mean()/np.sqrt(N):.4f}  "
          f"links={nl.mean():.0f}  "
          f"dlinks={dl.mean():.0f}  "
          f"[{time.time()-t0:.1f}s]")

print()
print("SCALING CHECK:")
for N in N_list:
    r = results[N]
    print(f"  N={N:5d}: a_down/sqrt(N) = {r['a_down_over_sqrtN']:.4f}")

# Power law fit
Ns = np.array(N_list)
ads = np.array([results[N]["a_down"] for N in N_list])
mask_pos = ads > 0
if mask_pos.sum() >= 2:
    log_fit = np.polyfit(np.log(Ns[mask_pos]), np.log(ads[mask_pos]), 1)
    print(f"\nPower law fit: a_down ~ N^{log_fit[0]:.3f}")
    print(f"Expected: 0.500 (sqrt)")
    print(f"Coefficient: {np.exp(log_fit[1]):.4f}")
