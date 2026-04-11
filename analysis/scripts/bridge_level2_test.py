#!/usr/bin/env python3
"""
Bridge Level 2: Physical Pauli-Jordan spectral determinant.

Uses Johnston 4D massless retarded Green function:
    K_0(x,x') = (1/2pi) * sqrt(rho/6) * L_0(x,x')
where L_0 = link (Hasse) adjacency matrix.

Physical Pauli-Jordan: i*Delta_phys = K_R - K_A ~ L_0 - L_0^T

Tests on TWO backgrounds:
  1) pp-wave (VSI, C^2=0) — check nonzero, scaling, CJ correlation
  2) Schwarzschild local (C^2 != 0) — extract coefficient, compare with 1/120

Key difference from Level 1: uses LINK matrix (Hasse neighbors only),
not full causal matrix C. This is the physical 4D scalar field
Pauli-Jordan function per Johnston/Nomaan-Dowker-Surya (1701.07212).
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)

# ============================================================
# PARAMETERS
# ============================================================
N = 2000
T = 1.0
ZETA = 0.15
M_SEEDS = 20

# pp-wave amplitudes
PPW_EPS = [0.5, 1.0, 2.0, 3.0]

# Schwarzschild parameters (same as FND-1 production)
SCH_M = 0.05
SCH_R0 = 0.50
SCH_R_ABCD = riemann_schwarzschild_local(SCH_M, SCH_R0)
SCH_E2 = 6.0 * SCH_M**2 / SCH_R0**6  # = 0.96


# ============================================================
# LINK MATRIX FROM HASSE
# ============================================================

def hasse_to_link_matrix(parents, children, n):
    """Build N x N link (adjacency) matrix from Hasse diagram.
    L[j, i] = 1 iff j is a Hasse parent of i (j -> i link).
    This is the LOWER-triangular part (j < i since sorted by time).
    """
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


# ============================================================
# PHYSICAL SPECTRAL DETERMINANT
# ============================================================

def compute_physical_spectral_det(L, truncation_mode='sorkin_yazdi'):
    """Compute spectral data of PHYSICAL Pauli-Jordan from link matrix.

    PJ_phys = L - L^T (antisymmetrized link matrix).
    Johnston normalization (1/2pi)*sqrt(rho/6) is a CONSTANT that
    cancels in CRN difference, so we use unnormalized L.

    Returns dict with sigma spectrum and log-pseudodeterminant.
    """
    n = L.shape[0]
    PJ = L - L.T  # antisymmetric

    # i*PJ is Hermitian -> real eigenvalues
    H = 1j * PJ
    evals = np.linalg.eigvalsh(H)

    # Positive eigenvalues (physical modes)
    sigma = np.sort(evals[evals > 1e-14])[::-1]

    # Truncation
    if truncation_mode == 'sorkin_yazdi':
        threshold = math.sqrt(n) / (4 * math.pi)
    elif truncation_mode == 'generous':
        # Keep more modes: threshold = N^{1/4}
        threshold = n ** 0.25
    else:
        threshold = 0.1

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
# CJ COMPUTATION (same as Level 1, for correlation)
# ============================================================

def Y_from_graph(par, ch):
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
# SINGLE SEED RUNNER
# ============================================================

def run_seed_ppwave(seed_idx, eps, base_seed=9100000):
    """Run Level 2 bridge test on pp-wave for one CRN seed."""
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    t0 = time.time()

    # Flat Hasse + link matrix
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    L_flat = hasse_to_link_matrix(par0, ch0, N)

    # Curved Hasse + link matrix
    parC, chC = build_hasse_from_predicate(
        pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
    L_curved = hasse_to_link_matrix(parC, chC, N)

    # Spectral determinants (physical PJ from link matrix)
    spec_flat = compute_physical_spectral_det(L_flat)
    spec_curved = compute_physical_spectral_det(L_curved)

    delta_gamma = 0.5 * (spec_curved['logdet'] - spec_flat['logdet'])

    # CJ for correlation
    Y0 = Y_from_graph(par0, ch0)
    YC = Y_from_graph(parC, chC)
    mask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par0, T)
    delta_Y = YC - Y0
    cj = compute_CJ(Y0, delta_Y, mask, strata)

    E2 = eps**2 / 2.0
    sigma0 = float(np.std(Y0[mask]))

    return {
        'seed': seed_idx, 'eps': eps, 'E2': E2,
        'delta_gamma': delta_gamma,
        'CJ': cj,
        'CJ_over_E2': cj / E2,
        'dG_over_E2': delta_gamma / E2 if E2 > 0 else 0,
        'sigma0': sigma0,
        'n_modes_flat': spec_flat['n_modes_retained'],
        'n_modes_curved': spec_curved['n_modes_retained'],
        'n_modes_total': spec_flat['n_modes_total'],
        'n_links_flat': int(L_flat.sum()),
        'n_links_curved': int(L_curved.sum()),
        'logdet_flat': spec_flat['logdet'],
        'logdet_curved': spec_curved['logdet'],
        'threshold': spec_flat['threshold'],
        'time_s': time.time() - t0,
    }


def run_seed_schwarzschild(seed_idx, base_seed=9200000):
    """Run Level 2 bridge test on local Schwarzschild for one CRN seed."""
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    t0 = time.time()

    # Flat Hasse + link matrix
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    L_flat = hasse_to_link_matrix(par0, ch0, N)

    # Schwarzschild (jet predicate) Hasse + link matrix
    parS, chS = build_hasse_from_predicate(
        pts, lambda P, i: jet_preds(P, i, SCH_R_ABCD))
    L_curved = hasse_to_link_matrix(parS, chS, N)

    # Spectral determinants
    spec_flat = compute_physical_spectral_det(L_flat)
    spec_curved = compute_physical_spectral_det(L_curved)

    delta_gamma = 0.5 * (spec_curved['logdet'] - spec_flat['logdet'])

    # CJ for correlation
    Y0 = Y_from_graph(par0, ch0)
    YS = Y_from_graph(parS, chS)
    mask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par0, T)
    delta_Y = YS - Y0
    cj = compute_CJ(Y0, delta_Y, mask, strata)
    sigma0 = float(np.std(Y0[mask]))

    return {
        'seed': seed_idx, 'geometry': 'schwarzschild',
        'M': SCH_M, 'r0': SCH_R0, 'E2': SCH_E2,
        'delta_gamma': delta_gamma,
        'CJ': cj,
        'CJ_over_E2': cj / SCH_E2,
        'dG_over_E2': delta_gamma / SCH_E2,
        'sigma0': sigma0,
        'n_modes_flat': spec_flat['n_modes_retained'],
        'n_modes_curved': spec_curved['n_modes_retained'],
        'n_modes_total': spec_flat['n_modes_total'],
        'n_links_flat': int(L_flat.sum()),
        'n_links_curved': int(L_curved.sum()),
        'logdet_flat': spec_flat['logdet'],
        'logdet_curved': spec_curved['logdet'],
        'threshold': spec_flat['threshold'],
        'time_s': time.time() - t0,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 72)
    print("BRIDGE LEVEL 2: Physical Pauli-Jordan (Johnston link matrix)")
    print(f"N={N}, T={T}, zeta={ZETA}, M={M_SEEDS}")
    print(f"Johnston K_0 = (1/2pi)*sqrt(rho/6) * L_0  [link projector]")
    print(f"Physical PJ = L_0 - L_0^T  (antisymmetrized Hasse adjacency)")
    print(f"Truncation: sigma > sqrt(N)/(4*pi) = {math.sqrt(N)/(4*math.pi):.2f}")
    print("=" * 72)

    all_results = {}

    # ========================
    # PART 1: PP-WAVE
    # ========================
    print("\n" + "=" * 72)
    print("PART 1: PP-WAVE (VSI, C^2 = 0)")
    print("=" * 72)

    for eps in PPW_EPS:
        E2 = eps**2 / 2.0
        print(f"\n--- eps = {eps} (E^2 = {E2:.2f}) ---", flush=True)

        dg_list, cj_list, dg_e2_list = [], [], []
        seed_results = []

        for si in range(M_SEEDS):
            res = run_seed_ppwave(si, eps)
            dg_list.append(res['delta_gamma'])
            cj_list.append(res['CJ'])
            dg_e2_list.append(res['dG_over_E2'])
            seed_results.append(res)

            if (si + 1) % 5 == 0 or si == 0:
                dg_mean = np.mean(dg_list)
                corr_str = f"r={np.corrcoef(dg_list, cj_list)[0,1]:+.3f}" if len(dg_list) >= 3 else "r=n/a"
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"dG={res['delta_gamma']:+.6f}  "
                      f"CJ={res['CJ']:.6f}  "
                      f"links={res['n_links_curved']}  "
                      f"modes={res['n_modes_flat']}/{res['n_modes_total']}  "
                      f"<dG>={dg_mean:+.6f}  {corr_str}  "
                      f"({res['time_s']:.1f}s)", flush=True)

        dg_arr = np.array(dg_list)
        cj_arr = np.array(cj_list)

        summary = {
            'eps': eps, 'E2': E2,
            'dG_mean': float(dg_arr.mean()),
            'dG_se': float(dg_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'CJ_mean': float(cj_arr.mean()),
            'CJ_se': float(cj_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'dG_over_E2_mean': float(np.mean(dg_e2_list)),
            'dG_over_E2_se': float(np.std(dg_e2_list, ddof=1) / math.sqrt(M_SEEDS)),
            'corr_dG_CJ': float(np.corrcoef(dg_arr, cj_arr)[0, 1]),
            'seeds': seed_results,
        }
        all_results[f'ppw_eps{eps}'] = summary

        t_stat = summary['dG_mean'] / (summary['dG_se'] + 1e-30)
        print(f"\n  SUMMARY eps={eps}:")
        print(f"    dG      = {summary['dG_mean']:+.6f} +/- {summary['dG_se']:.6f}  (t={t_stat:+.2f})")
        print(f"    CJ      = {summary['CJ_mean']:.6f} +/- {summary['CJ_se']:.6f}")
        print(f"    dG/E^2  = {summary['dG_over_E2_mean']:+.6f} +/- {summary['dG_over_E2_se']:.6f}")
        print(f"    Corr    = {summary['corr_dG_CJ']:+.4f}")
        print(flush=True)

    # PP-WAVE CROSS-EPSILON
    print("\n--- PP-WAVE CROSS-EPSILON ---")
    eps_arr = np.array(PPW_EPS)
    dg_means_ppw = np.array([all_results[f'ppw_eps{e}']['dG_mean'] for e in PPW_EPS])
    dg_e2_ppw = np.array([all_results[f'ppw_eps{e}']['dG_over_E2_mean'] for e in PPW_EPS])

    print("dG/E^2 constancy:")
    for e, v in zip(PPW_EPS, dg_e2_ppw):
        print(f"  eps={e}: dG/E^2 = {v:+.6f}")
    if np.mean(np.abs(dg_e2_ppw)) > 1e-10:
        cv = np.std(dg_e2_ppw) / np.abs(np.mean(dg_e2_ppw)) * 100
        print(f"  CV = {cv:.1f}%")

    # Power law
    all_pos = all(dg_means_ppw > 0)
    all_neg = all(dg_means_ppw < 0)
    if all_pos:
        alpha, _ = np.polyfit(np.log(eps_arr), np.log(dg_means_ppw), 1)
        print(f"  Power law: dG ~ eps^{alpha:.3f} (expected 2.0)")
    elif all_neg:
        alpha, _ = np.polyfit(np.log(eps_arr), np.log(-dg_means_ppw), 1)
        print(f"  Power law: |dG| ~ eps^{alpha:.3f} (dG<0, expected 2.0)")
    else:
        alpha = float('nan')
        print("  Power law: mixed signs")

    # ========================
    # PART 2: SCHWARZSCHILD
    # ========================
    print("\n" + "=" * 72)
    print(f"PART 2: SCHWARZSCHILD LOCAL (M={SCH_M}, r0={SCH_R0}, E^2={SCH_E2:.4f})")
    print("=" * 72)

    dg_list_s, cj_list_s = [], []
    seed_results_s = []

    for si in range(M_SEEDS):
        res = run_seed_schwarzschild(si)
        dg_list_s.append(res['delta_gamma'])
        cj_list_s.append(res['CJ'])
        seed_results_s.append(res)

        if (si + 1) % 5 == 0 or si == 0:
            dg_mean = np.mean(dg_list_s)
            corr_str = f"r={np.corrcoef(dg_list_s, cj_list_s)[0,1]:+.3f}" if len(dg_list_s) >= 3 else "r=n/a"
            print(f"  seed {si+1:2d}/{M_SEEDS}: "
                  f"dG={res['delta_gamma']:+.6f}  "
                  f"CJ={res['CJ']:.6f}  "
                  f"links={res['n_links_curved']}  "
                  f"modes={res['n_modes_flat']}/{res['n_modes_total']}  "
                  f"<dG>={dg_mean:+.6f}  {corr_str}  "
                  f"({res['time_s']:.1f}s)", flush=True)

    dg_arr_s = np.array(dg_list_s)
    cj_arr_s = np.array(cj_list_s)

    summary_s = {
        'geometry': 'schwarzschild', 'M': SCH_M, 'r0': SCH_R0, 'E2': SCH_E2,
        'dG_mean': float(dg_arr_s.mean()),
        'dG_se': float(dg_arr_s.std(ddof=1) / math.sqrt(M_SEEDS)),
        'CJ_mean': float(cj_arr_s.mean()),
        'CJ_se': float(cj_arr_s.std(ddof=1) / math.sqrt(M_SEEDS)),
        'dG_over_E2_mean': float(dg_arr_s.mean() / SCH_E2),
        'corr_dG_CJ': float(np.corrcoef(dg_arr_s, cj_arr_s)[0, 1]),
        'seeds': seed_results_s,
    }
    all_results['schwarzschild'] = summary_s

    t_stat_s = summary_s['dG_mean'] / (summary_s['dG_se'] + 1e-30)
    print(f"\n  SUMMARY Schwarzschild:")
    print(f"    dG      = {summary_s['dG_mean']:+.6f} +/- {summary_s['dG_se']:.6f}  (t={t_stat_s:+.2f})")
    print(f"    CJ      = {summary_s['CJ_mean']:.6f} +/- {summary_s['CJ_se']:.6f}")
    print(f"    dG/E^2  = {summary_s['dG_over_E2_mean']:+.6f}")
    print(f"    Corr    = {summary_s['corr_dG_CJ']:+.4f}")

    # ========================
    # FINAL VERDICT
    # ========================
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)

    # PP-wave: is dG nonzero? scaling? correlation?
    e_max = max(PPW_EPS)
    ppw_max = all_results[f'ppw_eps{e_max}']
    t_ppw = ppw_max['dG_mean'] / (ppw_max['dG_se'] + 1e-30)
    print(f"\n  PP-WAVE (eps={e_max}):")
    print(f"    dG nonzero? t={t_ppw:+.2f} {'YES' if abs(t_ppw) > 2.09 else 'NO'}")
    if not np.isnan(alpha):
        print(f"    Scaling: dG ~ eps^{alpha:.2f} (expected 2.0 for Weyl^2)")
    corrs_ppw = [all_results[f'ppw_eps{e}']['corr_dG_CJ'] for e in PPW_EPS]
    print(f"    Mean Corr(dG, CJ) = {np.mean(corrs_ppw):+.3f}")

    # Schwarzschild: coefficient extraction
    print(f"\n  SCHWARZSCHILD:")
    print(f"    dG nonzero? t={t_stat_s:+.2f} {'YES' if abs(t_stat_s) > 2.09 else 'NO'}")
    print(f"    dG/E^2 = {summary_s['dG_over_E2_mean']:+.6f}")
    print(f"    Corr(dG, CJ) = {summary_s['corr_dG_CJ']:+.4f}")

    # Comparison with 1/120
    if abs(summary_s['dG_over_E2_mean']) > 1e-10:
        ratio = summary_s['dG_over_E2_mean'] / (1.0/120.0)
        print(f"    Ratio dG/E^2 / (1/120) = {ratio:.4f}")

    # Save
    outfile = 'analysis/fnd1_data/bridge_level2_test.json'
    for key in all_results:
        if 'seeds' in all_results[key]:
            for sr in all_results[key]['seeds']:
                for k in list(sr.keys()):
                    if isinstance(sr[k], np.ndarray):
                        sr[k] = sr[k].tolist()
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\nSaved to {outfile}", flush=True)
