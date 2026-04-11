#!/usr/bin/env python3
"""
Bridge Test: Spectral Determinant of Pauli-Jordan function on causal sets.

Tests whether the one-loop spectral determinant delta-Gamma computed from
causal matrices on Poisson sprinklings detects Weyl curvature on pp-wave
spacetimes.

Three levels of analysis:
  Level 1: Naive PJ = C - C^T (causal matrix, not physical Green function)
  Level 2: Correlation with CJ across seeds
  Level 3: Null tests (epsilon=0, dS Ricci-only)

Output: real-time progress + JSON results.
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, bulk_mask, excess_kurtosis,
)

# ============================================================
# PARAMETERS
# ============================================================
N = 2000
T = 1.0
ZETA = 0.15
M_SEEDS = 20
EPS_VALUES = [0.5, 1.0, 2.0, 3.0]  # include small eps for scaling check

# ============================================================
# CAUSAL MATRIX BUILDERS
# ============================================================

def build_causal_matrix_flat(pts):
    """N x N causal matrix for Minkowski: C[i,j]=1 iff i < j causally."""
    N = len(pts)
    t = pts[:, 0]
    x = pts[:, 1:]
    dt = t[None, :] - t[:, None]          # dt[i,j] = t_j - t_i
    dx2 = np.sum((x[None, :, :] - x[:, None, :]) ** 2, axis=2)
    C = ((dt > 1e-12) & (dt**2 > dx2)).astype(np.float64)
    return C


def build_causal_matrix_ppwave(pts, eps):
    """N x N causal matrix for exact pp-wave.
    Uses the same exact predicate as our CJ computations.
    """
    N = len(pts)
    C = np.zeros((N, N), dtype=np.float64)
    for i in range(1, N):
        mask = ppwave_exact_preds(pts, i, eps=eps)
        C[:len(mask), i] = mask.astype(np.float64)
    return C


# ============================================================
# SPECTRAL DETERMINANT
# ============================================================

def compute_spectral_det(C, truncation_factor=None):
    """Compute spectral data of Pauli-Jordan function PJ = C - C^T.

    PJ is real antisymmetric -> eigenvalues of (i*PJ) are real.
    Positive eigenvalues = sigma_k.

    Returns dict with sigma spectrum and log-determinant.
    """
    N = C.shape[0]
    PJ = C - C.T

    # i*PJ is Hermitian (real symmetric in this case since i*(real antisym) is Hermitian)
    # But numpy: i*PJ has complex entries. Use eigvalsh for Hermitian.
    H = 1j * PJ
    evals = np.linalg.eigvalsh(H)  # real eigenvalues

    # Positive eigenvalues (modes)
    sigma = np.sort(evals[evals > 1e-14])[::-1]

    # Truncation options
    if truncation_factor is None:
        # Sorkin-Yazdi: threshold = sqrt(N) / (4*pi)
        threshold = math.sqrt(N) / (4 * math.pi)
    else:
        threshold = truncation_factor

    sigma_trunc = sigma[sigma > threshold]

    if len(sigma_trunc) == 0:
        logdet = 0.0
    else:
        logdet = float(np.sum(np.log(sigma_trunc)))

    return {
        'sigma_all': sigma,
        'sigma_trunc': sigma_trunc,
        'n_modes_total': len(sigma),
        'n_modes_retained': len(sigma_trunc),
        'threshold': float(threshold),
        'logdet': logdet,
        'sigma_max': float(sigma[0]) if len(sigma) > 0 else 0.0,
        'sigma_min_retained': float(sigma_trunc[-1]) if len(sigma_trunc) > 0 else 0.0,
    }


# ============================================================
# CJ COMPUTATION (for correlation test)
# ============================================================

def Y_from_graph(par, ch):
    """Compute Y = log2(p_down * p_up + 1) from Hasse graph."""
    n = len(par)
    p_down = np.ones(n, dtype=np.float64)
    p_up = np.ones(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            p_down[i] = np.sum(p_down[par[i]]) + 1
    for i in range(n - 1, -1, -1):
        if ch[i] is not None and len(ch[i]) > 0:
            p_up[i] = np.sum(p_up[ch[i]]) + 1
    return np.log2(p_down * p_up + 1)


def make_strata(pts, par0, T):
    """45-bin strata (5 tau x 3 rho x 3 depth)."""
    tau_hat = 2 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if par0[i] is not None and len(par0[i]) > 0:
            depth[i] = int(np.max(depth[par0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


def compute_CJ(Y0, delta, mask, strata):
    """Stratified covariance CJ."""
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = (np.mean(np.abs(X[idx]) * dY2[idx])
               - np.mean(np.abs(X[idx])) * np.mean(dY2[idx]))
        total += w * cov
    return float(total)


# ============================================================
# MAIN TEST
# ============================================================

def run_one_seed(seed_idx, eps, base_seed=9000000):
    """Run bridge test for one CRN seed at given epsilon."""
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)

    t0 = time.time()

    # --- Build causal matrices ---
    C_flat = build_causal_matrix_flat(pts)
    t_cmat_flat = time.time() - t0

    C_curved = build_causal_matrix_ppwave(pts, eps)
    t_cmat_curved = time.time() - t0

    # --- Spectral determinants ---
    spec_flat = compute_spectral_det(C_flat)
    t_spec_flat = time.time() - t0

    spec_curved = compute_spectral_det(C_curved)
    t_spec_curved = time.time() - t0

    # --- Delta Gamma ---
    delta_gamma = 0.5 * (spec_curved['logdet'] - spec_flat['logdet'])

    # --- CJ for correlation ---
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    Y0 = Y_from_graph(par0, ch0)
    mask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par0, T)

    parC, chC = build_hasse_from_predicate(
        pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
    YC = Y_from_graph(parC, chC)
    delta_Y = YC - Y0

    cj = compute_CJ(Y0, delta_Y, mask, strata)
    t_total = time.time() - t0

    E2 = eps**2 / 2.0

    return {
        'seed': seed_idx,
        'eps': eps,
        'E2': E2,
        'delta_gamma': delta_gamma,
        'CJ': cj,
        'CJ_over_E2': cj / E2 if E2 > 0 else 0,
        'delta_gamma_over_E2': delta_gamma / E2 if E2 > 0 else 0,
        'n_modes_flat': spec_flat['n_modes_retained'],
        'n_modes_curved': spec_curved['n_modes_retained'],
        'n_modes_total_flat': spec_flat['n_modes_total'],
        'sigma_max_flat': spec_flat['sigma_max'],
        'sigma_max_curved': spec_curved['sigma_max'],
        'logdet_flat': spec_flat['logdet'],
        'logdet_curved': spec_curved['logdet'],
        'threshold': spec_flat['threshold'],
        'time_s': t_total,
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("=" * 72)
    print("BRIDGE TEST: Spectral Determinant of Pauli-Jordan Function")
    print(f"N={N}, T={T}, zeta={ZETA}, M={M_SEEDS}")
    print(f"eps values: {EPS_VALUES}")
    print(f"Sorkin-Yazdi truncation: sigma > sqrt(N)/(4*pi) = {math.sqrt(N)/(4*math.pi):.2f}")
    print("=" * 72)
    print(flush=True)

    all_results = {}

    for eps in EPS_VALUES:
        E2 = eps**2 / 2.0
        print(f"\n--- eps = {eps} (E^2 = {E2:.2f}) ---", flush=True)

        dg_list = []
        cj_list = []
        dg_over_E2_list = []
        seed_results = []

        for si in range(M_SEEDS):
            t0 = time.time()
            res = run_one_seed(si, eps)
            elapsed = time.time() - t0

            dg_list.append(res['delta_gamma'])
            cj_list.append(res['CJ'])
            dg_over_E2_list.append(res['delta_gamma_over_E2'])
            seed_results.append(res)

            # Real-time progress
            if (si + 1) % 5 == 0 or si == 0:
                dg_mean = np.mean(dg_list)
                cj_mean = np.mean(cj_list)
                # Running correlation
                if len(dg_list) >= 3:
                    corr = np.corrcoef(dg_list, cj_list)[0, 1]
                    corr_str = f"corr(dG,CJ)={corr:+.3f}"
                else:
                    corr_str = "corr=n/a"
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"dG={res['delta_gamma']:+.4f}  "
                      f"CJ={res['CJ']:.6f}  "
                      f"dG/E2={res['delta_gamma_over_E2']:+.4f}  "
                      f"modes={res['n_modes_flat']}/{res['n_modes_total_flat']}  "
                      f"<dG>={dg_mean:+.4f}  {corr_str}  "
                      f"({elapsed:.1f}s)",
                      flush=True)

        dg_arr = np.array(dg_list)
        cj_arr = np.array(cj_list)
        dg_e2_arr = np.array(dg_over_E2_list)

        summary = {
            'eps': eps,
            'E2': E2,
            'delta_gamma_mean': float(dg_arr.mean()),
            'delta_gamma_se': float(dg_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'delta_gamma_std': float(dg_arr.std(ddof=1)),
            'CJ_mean': float(cj_arr.mean()),
            'CJ_se': float(cj_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'dG_over_E2_mean': float(dg_e2_arr.mean()),
            'dG_over_E2_se': float(dg_e2_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'correlation_dG_CJ': float(np.corrcoef(dg_arr, cj_arr)[0, 1]),
            'seeds': seed_results,
        }
        all_results[str(eps)] = summary

        print(f"\n  SUMMARY eps={eps}:")
        print(f"    delta_Gamma = {summary['delta_gamma_mean']:+.4f} +/- {summary['delta_gamma_se']:.4f}")
        print(f"    CJ          = {summary['CJ_mean']:.6f} +/- {summary['CJ_se']:.6f}")
        print(f"    dG/E^2      = {summary['dG_over_E2_mean']:+.4f} +/- {summary['dG_over_E2_se']:.4f}")
        print(f"    Corr(dG,CJ) = {summary['correlation_dG_CJ']:+.4f}")
        print(flush=True)

    # ============================================================
    # CROSS-EPSILON ANALYSIS
    # ============================================================
    print("\n" + "=" * 72)
    print("CROSS-EPSILON ANALYSIS")
    print("=" * 72)

    eps_arr = np.array(EPS_VALUES)
    dg_means = np.array([all_results[str(e)]['delta_gamma_mean'] for e in EPS_VALUES])
    dg_over_E2 = np.array([all_results[str(e)]['dG_over_E2_mean'] for e in EPS_VALUES])

    print("\ndG/E^2 constancy (epsilon-independence test):")
    for e, v in zip(EPS_VALUES, dg_over_E2):
        print(f"  eps={e:.1f}: dG/E^2 = {v:+.4f}")
    if len(dg_over_E2) > 1 and np.mean(np.abs(dg_over_E2)) > 0:
        cv = np.std(dg_over_E2) / np.abs(np.mean(dg_over_E2)) * 100
        print(f"  CV = {cv:.1f}%")
    else:
        cv = float('inf')
        print("  CV: undefined (dG/E2 ~ 0)")

    # Power-law fit: dG ~ eps^alpha
    if all(dg_means > 0):
        alpha, _ = np.polyfit(np.log(eps_arr), np.log(dg_means), 1)
        print(f"\n  Power law: dG ~ eps^{alpha:.3f} (expected 2.0)")
    elif all(dg_means < 0):
        alpha, _ = np.polyfit(np.log(eps_arr), np.log(-dg_means), 1)
        print(f"\n  Power law: |dG| ~ eps^{alpha:.3f} (expected 2.0) [dG < 0]")
    else:
        alpha = float('nan')
        print("\n  Power law: mixed signs, cannot fit")

    # Correlation summary
    print("\nCorrelation(delta_Gamma, CJ) per epsilon:")
    for e in EPS_VALUES:
        c = all_results[str(e)]['correlation_dG_CJ']
        print(f"  eps={e:.1f}: corr = {c:+.4f}")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    # Is dG significantly nonzero?
    overall_dg = np.concatenate([np.array([r['delta_gamma'] for r in all_results[str(e)]['seeds']])
                                  for e in EPS_VALUES])
    # Check at largest eps
    e_max = max(EPS_VALUES)
    dg_max = all_results[str(e_max)]
    t_stat = dg_max['delta_gamma_mean'] / (dg_max['delta_gamma_se'] + 1e-30)
    print(f"\n  At eps={e_max}: dG = {dg_max['delta_gamma_mean']:+.4f}, "
          f"t-stat = {t_stat:+.2f} "
          f"({'SIGNIFICANT (p<0.05)' if abs(t_stat) > 2.09 else 'NOT SIGNIFICANT'})")

    # Is dG proportional to eps^2?
    print(f"  eps scaling: alpha = {alpha:.3f} (expected 2.0)")
    if not np.isnan(alpha):
        if abs(alpha - 2.0) < 0.3:
            print("  -> CONSISTENT with quadratic Weyl scaling")
        else:
            print(f"  -> INCONSISTENT with quadratic (deviation {alpha-2.0:+.2f})")

    # Is dG correlated with CJ?
    mean_corr = np.mean([all_results[str(e)]['correlation_dG_CJ'] for e in EPS_VALUES])
    print(f"  Mean corr(dG, CJ) = {mean_corr:+.3f}")
    if abs(mean_corr) > 0.5:
        print("  -> STRONG correlation: spectral det and CJ probe same curvature")
    elif abs(mean_corr) > 0.3:
        print("  -> MODERATE correlation")
    else:
        print("  -> WEAK/NO correlation: independent observables")

    # Save
    outfile = 'analysis/fnd1_data/bridge_spectral_test.json'
    # Strip non-serializable numpy arrays from seed results
    for eps_key in all_results:
        for sr in all_results[eps_key]['seeds']:
            for k in list(sr.keys()):
                if isinstance(sr[k], np.ndarray):
                    sr[k] = sr[k].tolist()
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\nSaved to {outfile}", flush=True)
