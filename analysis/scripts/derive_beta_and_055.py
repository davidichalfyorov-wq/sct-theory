"""
Derive beta=0.30 and 0.55 factor from first principles.
=========================================================
Two tasks:
  A. Precise beta measurement via CRN at small eps (convergence check)
  B. 0.55 factor decomposition: p_down x p_up anticorrelation analysis

Author: David Alfyorov
"""
import numpy as np
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# GPU preamble
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp

from scipy import stats
from scipy.stats import kurtosis as kurt


def build_causal_link_gpu(pts_g, eps_val=0.0):
    """Build causal and link matrices on GPU. Returns (C, L) as GPU arrays."""
    t_g = pts_g[:, 0]
    x_g = pts_g[:, 1]
    y_g = pts_g[:, 2]
    z_g = pts_g[:, 3]

    dt = t_g[None, :] - t_g[:, None]
    dx = x_g[None, :] - x_g[:, None]
    dy = y_g[None, :] - y_g[:, None]
    dz = z_g[None, :] - z_g[:, None]

    if abs(eps_val) < 1e-15:
        tau2 = dt**2 - dx**2 - dy**2 - dz**2
    else:
        du = (dt + dz) / np.sqrt(2)
        x_i = x_g[:, None]
        y_i = y_g[:, None]
        H_int = x_i**2 + x_i * dx + dx**2 / 3 - y_i**2 - y_i * dy - dy**2 / 3
        delta_sigma = eps_val / 2 * du**2 * H_int
        tau2 = dt**2 - dx**2 - dy**2 - dz**2 - 2 * delta_sigma

    C = ((dt > 0) & (tau2 > 0)).astype(cp.float32)
    C2 = C @ C
    L = ((C > 0.5) & (C2 < 0.5)).astype(cp.float32)
    return C, L


def compute_path_counts(L_np, time_order):
    """Compute p_down and p_up from link matrix (CPU numpy)."""
    N = L_np.shape[0]
    p_down = np.zeros(N, dtype=np.float64)
    p_up = np.zeros(N, dtype=np.float64)

    for idx in time_order:
        preds = np.where(L_np[:, idx] > 0.5)[0]
        if len(preds) == 0:
            p_down[idx] = 1.0
        else:
            p_down[idx] = p_down[preds].sum()

    for idx in reversed(time_order.tolist()):
        succs = np.where(L_np[idx, :] > 0.5)[0]
        if len(succs) == 0:
            p_up[idx] = 1.0
        else:
            p_up[idx] = p_up[succs].sum()

    return p_down, p_up


# ====================================================================
# PART A: Precise beta measurement
# ====================================================================
def measure_beta(N=2000, T=1.0, M=20, eps_list=None):
    if eps_list is None:
        eps_list = [0.02, 0.05, 0.1, 0.2, 0.5]

    print("=" * 70)
    print(f"PART A: PRECISE beta MEASUREMENT (N={N}, M={M})")
    print("=" * 70)

    all_betas = {e: [] for e in eps_list}
    t0 = time.time()

    for m in range(M):
        np.random.seed(1000 + m)
        pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
        pts_g = cp.asarray(pts)
        f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0

        _, L_flat = build_causal_link_gpu(pts_g, 0.0)
        k_flat = cp.asnumpy((L_flat.sum(0) + L_flat.sum(1)).astype(cp.float32))

        for eps_val in eps_list:
            _, L_ppw = build_causal_link_gpu(pts_g, eps_val)
            k_ppw = cp.asnumpy((L_ppw.sum(0) + L_ppw.sum(1)).astype(cp.float32))

            dk = k_ppw - k_flat
            k0 = np.maximum(k_flat, 1.0)
            dk_rel = dk / k0

            mask = k_flat > 10
            if mask.sum() > 100:
                slope, _, r, p, se = stats.linregress(f_vals[mask], dk_rel[mask])
                beta_m = slope / eps_val
                all_betas[eps_val].append(beta_m)

        if (m + 1) % 5 == 0:
            print(f"  Sprinkling {m+1}/{M} done ({time.time()-t0:.1f}s)")

    print()
    print("CONVERGENCE OF beta AT eps -> 0:")
    results_beta = {}
    for eps_val in eps_list:
        betas = np.array(all_betas[eps_val])
        mean_b = float(betas.mean())
        se_b = float(betas.std() / np.sqrt(M))
        std_b = float(betas.std())
        results_beta[eps_val] = {"mean": mean_b, "se": se_b, "std": std_b}
        print(f"  eps={eps_val:6.3f}: beta = {mean_b:+.4f} +/- {se_b:.4f}  (std={std_b:.4f})")

    return results_beta


# ====================================================================
# PART B: 0.55 factor derivation
# ====================================================================
def derive_055_factor(N=2000, T=1.0, M=10, eps_val=5.0):
    print()
    print("=" * 70)
    print(f"PART B: 0.55 FACTOR DERIVATION (N={N}, M={M}, eps={eps_val})")
    print("=" * 70)

    factors = []
    rho_Y_list = []
    rho_xi_list = []
    var_ratio_list = []
    cross_ratio_list = []
    xi2_ratio_list = []

    t0 = time.time()

    for m in range(M):
        np.random.seed(2000 + m)
        pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
        pts_g = cp.asarray(pts)
        order = np.argsort(pts[:, 0])

        # Flat
        _, L_flat = build_causal_link_gpu(pts_g, 0.0)
        L_flat_np = cp.asnumpy(L_flat)
        pd_flat, pu_flat = compute_path_counts(L_flat_np, order)

        # PP-wave
        _, L_ppw = build_causal_link_gpu(pts_g, eps_val)
        L_ppw_np = cp.asnumpy(L_ppw)
        pd_ppw, pu_ppw = compute_path_counts(L_ppw_np, order)

        # Y values
        P_flat = pd_flat * pu_flat
        P_ppw = pd_ppw * pu_ppw
        Y_flat = np.log2(P_flat + 1)
        Y_ppw = np.log2(P_ppw + 1)

        # Perturbation
        xi = (Y_ppw - Y_flat) / eps_val

        # Interior mask
        mask = (pd_flat > 1) & (pu_flat > 1) & (pd_ppw > 1) & (pu_ppw > 1)

        # Correlation of log(p_down), log(p_up)
        log_pd = np.log2(np.maximum(pd_flat, 1))
        log_pu = np.log2(np.maximum(pu_flat, 1))
        rho_Y = np.corrcoef(log_pd[mask], log_pu[mask])[0, 1]
        rho_Y_list.append(rho_Y)

        # xi decomposition
        xi_d = (np.log2(np.maximum(pd_ppw, 1)) - log_pd) / eps_val
        xi_u = (np.log2(np.maximum(pu_ppw, 1)) - log_pu) / eps_val
        rho_xi = np.corrcoef(xi_d[mask], xi_u[mask])[0, 1]
        rho_xi_list.append(rho_xi)

        # Variance ratio
        var_xi = np.var(xi[mask])
        var_indep = np.var(xi_d[mask]) + np.var(xi_u[mask])
        var_ratio_list.append(var_xi / var_indep if var_indep > 0 else np.nan)

        # Cross-moment ratio
        Y0 = Y_flat[mask]
        xi_m = xi[mask]
        xi_dm = xi_d[mask]
        xi_um = xi_u[mask]

        Y0_xi2 = np.mean(Y0**2 * xi_m**2)
        Y0_xid2 = np.mean(Y0**2 * xi_dm**2)
        Y0_xiu2 = np.mean(Y0**2 * xi_um**2)
        cross_ratio = Y0_xi2 / (Y0_xid2 + Y0_xiu2) if (Y0_xid2 + Y0_xiu2) > 0 else np.nan
        cross_ratio_list.append(cross_ratio)

        # <xi^2> ratio
        xi2_actual = np.mean(xi_m**2)
        xi2_indep = np.mean(xi_dm**2) + np.mean(xi_um**2)
        xi2_ratio_list.append(xi2_actual / xi2_indep if xi2_indep > 0 else np.nan)

        # Formula prediction vs observed
        sigma0_sq = np.var(Y0)
        kappa0 = kurt(Y0, fisher=True)
        xi_sq_mean = np.mean(xi_m**2)
        Y0sq_xi_sq = np.mean(Y0**2 * xi_m**2)

        formula_pred = eps_val**2 * (
            6 * Y0sq_xi_sq / sigma0_sq**2
            - 2 * (kappa0 + 3) * xi_sq_mean / sigma0_sq
        )
        dk_obs = kurt(Y_ppw[mask], fisher=True) - kurt(Y_flat[mask], fisher=True)
        factor = dk_obs / formula_pred if abs(formula_pred) > 1e-10 else np.nan
        factors.append(factor)

        if (m + 1) % 5 == 0:
            print(f"  Sprinkling {m+1}/{M} done ({time.time()-t0:.1f}s)")

    # Summary
    print()
    print("RESULTS:")
    print(f"  corr(log2_pd, log2_pu) = {np.mean(rho_Y_list):.4f} +/- {np.std(rho_Y_list)/np.sqrt(M):.4f}")
    print(f"  corr(xi_down, xi_up)   = {np.mean(rho_xi_list):.4f} +/- {np.std(rho_xi_list)/np.sqrt(M):.4f}")
    print()
    print(f"  Var(xi) / [Var(xi_d)+Var(xi_u)] = {np.mean(var_ratio_list):.4f} +/- {np.std(var_ratio_list)/np.sqrt(M):.4f}")
    print(f"  <Y0^2*xi^2> / [<Y0^2*xi_d^2>+<Y0^2*xi_u^2>] = {np.mean(cross_ratio_list):.4f} +/- {np.std(cross_ratio_list)/np.sqrt(M):.4f}")
    print(f"  <xi^2> / [<xi_d^2>+<xi_u^2>] = {np.mean(xi2_ratio_list):.4f} +/- {np.std(xi2_ratio_list)/np.sqrt(M):.4f}")
    print()
    print(f"  Observed/Formula factor = {np.mean(factors):.4f} +/- {np.std(factors)/np.sqrt(M):.4f}")
    print(f"  Expected from 1+rho_xi = {1+np.mean(rho_xi_list):.4f}")
    print()

    # Theoretical prediction for the factor:
    # The kurtosis formula involves two terms:
    # Term 1: 6*<Y0^2*xi^2>/sigma^4 * eps^2
    # Term 2: -2*(kappa0+3)*<xi^2>/sigma^2 * eps^2
    # Both involve <xi^2>, which is reduced by factor (1+rho_xi) from anticorrelation.
    # But the <Y0^2*xi^2> cross-moment has a DIFFERENT reduction factor.
    # The overall factor is a weighted combination.
    rho_xi_avg = np.mean(rho_xi_list)
    cross_r = np.mean(cross_ratio_list)
    xi2_r = np.mean(xi2_ratio_list)
    print(f"  ANALYSIS:")
    print(f"    rho_xi (perturbation correlation) = {rho_xi_avg:.4f}")
    print(f"    1 + rho_xi = {1+rho_xi_avg:.4f} (naive variance reduction)")
    print(f"    Actual <xi^2> ratio = {xi2_r:.4f}")
    print(f"    Actual cross-moment ratio = {cross_r:.4f}")
    print(f"    These should be ≈ (1+rho_xi) if anticorrelation is the only effect")

    return {
        "factor_mean": float(np.mean(factors)),
        "factor_se": float(np.std(factors) / np.sqrt(M)),
        "rho_Y": float(np.mean(rho_Y_list)),
        "rho_xi": float(np.mean(rho_xi_list)),
        "var_ratio": float(np.mean(var_ratio_list)),
        "cross_ratio": float(np.mean(cross_ratio_list)),
        "xi2_ratio": float(np.mean(xi2_ratio_list)),
    }


if __name__ == "__main__":
    # Part A: beta
    beta_results = measure_beta(N=2000, T=1.0, M=20, eps_list=[0.02, 0.05, 0.1, 0.2, 0.5])

    # Part B: 0.55 factor
    factor_results = derive_055_factor(N=2000, T=1.0, M=10, eps_val=5.0)

    # Save results
    outdir = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "beta_and_055_derivation.json"), "w") as f:
        json.dump({"beta": beta_results, "factor_055": factor_results}, f, indent=2)
    print(f"\nResults saved to {outdir}/beta_and_055_derivation.json")
