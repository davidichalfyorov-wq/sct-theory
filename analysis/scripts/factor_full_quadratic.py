"""
Full centered quadratic expansion for kurtosis change.
=======================================================

independent analysis identified that factor_055_v3.py uses INCOMPLETE formula.
The full centered expansion is:

Let X = Y0 - <Y0>, eta = deltaY - <deltaY>
a = <X*eta>, b = <eta^2>, c = <X^3*eta>, d = <X^2*eta^2>

Delta_kappa = [4c/sigma^4 - 4(kappa0+3)a/sigma^2]
            + [6d/sigma^4 - 2(kappa0+3)b/sigma^2 - 16ac/sigma^6 + 12(kappa0+3)a^2/sigma^4]
            + O(eta^3)

The OLD formula only kept the d and b terms (in uncentered form).
The MISSING terms (a, c, ac, a^2) can be large!

Author: David Alfyorov
"""
import numpy as np
import os, sys, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad
import scipy.sparse as sp
from scipy.stats import kurtosis as kurt

N = 2000
T = 1.0
M = 20

print("=" * 70)
print(f"FULL vs INCOMPLETE QUADRATIC KURTOSIS FORMULA (N={N}, M={M})")
print("=" * 70)


def path_kurtosis_full(C, N):
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
    return log_p


for eps_val in [1.0, 2.0, 5.0, 10.0]:
    dk_obs_list = []
    formula_old_list = []
    formula_new_list = []

    for trial in range(M):
        rng = np.random.default_rng(trial * 1000 + 100)
        pts = sprinkle_4d(N, T, rng)

        C_f = causal_flat(pts)
        Y0 = path_kurtosis_full(C_f, N)
        del C_f; gc.collect()

        C_p = causal_ppwave_quad(pts, eps_val)
        Y_eps = path_kurtosis_full(C_p, N)
        del C_p; gc.collect()

        # Observed
        dk_obs = kurt(Y_eps, fisher=True) - kurt(Y0, fisher=True)

        # Perturbation
        delta_Y = Y_eps - Y0

        # Centered quantities
        X = Y0 - np.mean(Y0)
        eta = delta_Y - np.mean(delta_Y)
        sigma2 = np.var(Y0)      # = <X^2>
        sigma4 = sigma2**2
        kappa0 = kurt(Y0, fisher=True)

        # Moments
        a = np.mean(X * eta)           # <X*eta>
        b = np.mean(eta**2)            # <eta^2>
        c = np.mean(X**3 * eta)        # <X^3*eta>
        d = np.mean(X**2 * eta**2)     # <X^2*eta^2>

        # OLD formula (incomplete): only d and b terms, uncentered
        xi = delta_Y / eps_val
        Y0sq_xi2 = np.mean(Y0**2 * xi**2)
        xi2 = np.mean(xi**2)
        formula_old = eps_val**2 * (6 * Y0sq_xi2 / sigma4 - 2 * (kappa0 + 3) * xi2 / sigma2)

        # NEW formula (full centered quadratic)
        # First order terms (should cancel by symmetry over ensemble):
        D1 = 4 * c / sigma4 - 4 * (kappa0 + 3) * a / sigma2
        # Second order terms:
        D2 = (6 * d / sigma4
              - 2 * (kappa0 + 3) * b / sigma2
              - 16 * a * c / sigma2**3
              + 12 * (kappa0 + 3) * a**2 / sigma4)
        formula_new = D1 + D2

        dk_obs_list.append(dk_obs)
        formula_old_list.append(formula_old)
        formula_new_list.append(formula_new)

    dk_a = np.array(dk_obs_list)
    old_a = np.array(formula_old_list)
    new_a = np.array(formula_new_list)

    fac_old = dk_a / old_a
    fac_new = dk_a / new_a

    print(f"eps={eps_val:5.1f}: obs={dk_a.mean():+.6f}  "
          f"old={old_a.mean():+.6f} (fac={fac_old.mean():.4f})  "
          f"NEW={new_a.mean():+.6f} (fac={fac_new.mean():.4f})")

print()
print("If analytical is correct: NEW factor should be ~1.0-1.25 (not 0.08)")
