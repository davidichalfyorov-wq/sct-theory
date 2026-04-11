"""
Verify analytical scaling law: Dpk ~ eps^2 * N^{1/2} * C2
=========================================================

If true: Dpk / (eps^2 * sqrt(N) * C2) = A_eff = const across N.
C2 = <f^2> = T^4/1120 (at T=1: 8.929e-4).

Test at N=500,1000,2000,3000,5000 with M=20, eps=5.
Also test with full centered quadratic formula for comparison.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
import os, sys, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad
from scipy.stats import kurtosis as kurt

T = 1.0
eps_val = 5.0
M = 20
C2 = 1.0 / 1120  # T^4/1120 at T=1

print("=" * 70)
print(f"Dpk SCALING TEST: Dpk / (eps^2 * sqrt(N) * C2) = const?")
print(f"eps={eps_val}, M={M}, C2={C2:.6e}")
print("=" * 70)


def path_kurtosis_val(C, N):
    C_sp = sp.csr_matrix(C)
    C2m = C_sp @ C_sp
    has_int = (C2m != 0).astype(np.float64)
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
    return float(kurt(log_p, fisher=True)), log_p


for N in [500, 1000, 2000, 3000, 5000]:
    dpk_list = []
    full_formula_list = []
    t0 = time.time()

    for trial in range(M):
        rng = np.random.default_rng(trial * 1000 + 100)
        pts = sprinkle_4d(N, T, rng)

        C_f = causal_flat(pts)
        pk_f, Y0 = path_kurtosis_val(C_f, N)
        del C_f; gc.collect()

        C_p = causal_ppwave_quad(pts, eps_val)
        pk_p, Y_eps = path_kurtosis_val(C_p, N)
        del C_p; gc.collect()

        dpk = pk_p - pk_f
        dpk_list.append(dpk)

        # Full centered quadratic formula
        delta_Y = Y_eps - Y0
        X = Y0 - np.mean(Y0)
        eta = delta_Y - np.mean(delta_Y)
        sigma2 = np.var(Y0)
        sigma4 = sigma2**2
        kappa0 = kurt(Y0, fisher=True)

        a = np.mean(X * eta)
        b = np.mean(eta**2)
        c = np.mean(X**3 * eta)
        d = np.mean(X**2 * eta**2)

        D1 = 4 * c / sigma4 - 4 * (kappa0 + 3) * a / sigma2
        D2 = (6 * d / sigma4
              - 2 * (kappa0 + 3) * b / sigma2
              - 16 * a * c / sigma2**3
              + 12 * (kappa0 + 3) * a**2 / sigma4)
        full_formula_list.append(D1 + D2)

    dpk_a = np.array(dpk_list)
    ff_a = np.array(full_formula_list)

    # analytical scaling: Dpk / (eps^2 * sqrt(N) * C2)
    A_eff = dpk_a.mean() / (eps_val**2 * np.sqrt(N) * C2)
    A_eff_formula = ff_a.mean() / (eps_val**2 * np.sqrt(N) * C2)

    elapsed = time.time() - t0
    print(f"N={N:5d}: Dpk={dpk_a.mean():+.6f}+/-{dpk_a.std()/np.sqrt(M):.6f}  "
          f"A_eff(obs)={A_eff:.2f}  A_eff(formula)={A_eff_formula:.2f}  "
          f"full_fac={dpk_a.mean()/ff_a.mean():.3f}  "
          f"[{elapsed:.1f}s]")

print()
print("If A_eff(obs) is constant across N => Dpk ~ eps^2 * N^{1/2} * C2 CONFIRMED")
print("If A_eff grows => exponent > 1/2")
print("If A_eff shrinks => exponent < 1/2")

# Power law fit
Ns = np.array([500, 1000, 2000, 3000, 5000])
# Recompute dpk means for fit (use first trial for speed)
dpk_means = []
for N in Ns:
    rng = np.random.default_rng(100)
    pts = sprinkle_4d(N, T, rng)
    C_f = causal_flat(pts)
    pk_f, _ = path_kurtosis_val(C_f, N)
    del C_f; gc.collect()
    C_p = causal_ppwave_quad(pts, eps_val)
    pk_p, _ = path_kurtosis_val(C_p, N)
    del C_p; gc.collect()
    dpk_means.append(pk_p - pk_f)

dpk_arr = np.array(dpk_means)
mask = dpk_arr > 0
if mask.sum() >= 2:
    fit = np.polyfit(np.log(Ns[mask]), np.log(dpk_arr[mask]), 1)
    print(f"\nPower law fit: Dpk ~ N^{fit[0]:.3f}")
    print(f"analytical prediction: 0.500")
