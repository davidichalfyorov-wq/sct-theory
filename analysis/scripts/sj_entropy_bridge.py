#!/usr/bin/env python3
"""
SJ Entanglement Entropy Bridge: Phases 1-5.

Extracts the type-B anomaly coefficient c = 1/120 from the curvature-dependent
logarithmic term of double-truncated Sorkin entropy on 4D causal sets.

Pipeline:
  1. Sprinkle N points in outer 4D causal diamond
  2. Build Hasse diagram (link matrix L) for flat and curved metrics (CRN)
  3. Construct PJ = L - L^T, then H = i*PJ (Hermitian)
  4. Global truncation of H at threshold kappa_O
  5. Build SJ Wightman W = Pos(H_truncated)
  6. Restrict W and Delta to inner subdiamond U
  7. Local truncation at kappa_U
  8. Compute Sorkin entropy S = Σ λ ln|λ| via generalized eigenproblem
  9. CRN difference δS = S_curved - S_flat

References:
  - BBLL 2017 (arXiv:1712.04227): double-truncated SJ entropy, area law
  - Solodukhin 2008 (arXiv:0802.3117): type-B anomaly surface functional
  - Johnston (arXiv:1701.07212): 4D massless retarded Green function G = α_J L
  - Sorkin 2012 (arXiv:1205.2953): spacetime entanglement entropy

Target: c_cs = 4 δα_log / δJ_B → 1/120 for free real scalar.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)


# ===================================================================
#  Core SJ Entropy Functions
# ===================================================================

def hasse_to_link_matrix(parents, n):
    """Build dense link matrix from Hasse parent lists.
    L[j, i] = 1 if j is a Hasse parent of i (j ≺ i, no intermediary).
    L is strictly lower-triangular (time-sorted points).
    """
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def global_truncation(Delta, kappa):
    """Global (outer) spectral truncation of PJ operator.

    Input:
        Delta: real antisymmetric matrix (PJ operator = L - L^T)
        kappa: spectral threshold on |σ| of H = i*Delta

    Returns:
        W_trunc: Wightman = positive spectral part of truncated H
        Delta_trunc: truncated PJ (real antisymmetric)
        info: dict with diagnostic counts
    """
    H = 1j * Delta
    sigma, U = np.linalg.eigh(H)

    keep = np.abs(sigma) > kappa
    pos = sigma > kappa

    n_kept = int(keep.sum())
    n_pos = int(pos.sum())

    # Truncated Wightman = positive part
    U_pos = U[:, pos]
    s_pos = sigma[pos]
    W_trunc = (U_pos * s_pos[None, :]) @ U_pos.conj().T

    # Truncated PJ
    U_keep = U[:, keep]
    s_keep = sigma[keep]
    H_trunc = (U_keep * s_keep[None, :]) @ U_keep.conj().T
    Delta_trunc = np.real(-1j * H_trunc)

    info = {
        'n_total': len(sigma),
        'n_kept': n_kept,
        'n_pos': n_pos,
        'sigma_max': float(sigma.max()),
        'sigma_min': float(sigma.min()),
        'kappa': float(kappa),
    }
    return W_trunc, Delta_trunc, info


def inner_subdiamond_indices(pts, T_inner):
    """Select points inside inner concentric subdiamond.
    Subdiamond: |t| + |x| < T_inner/2.
    """
    t = pts[:, 0]
    r = np.linalg.norm(pts[:, 1:], axis=1)
    return np.where(np.abs(t) + r < T_inner / 2.0)[0]


def local_truncation_and_entropy(W_U0, Delta_U0, kappa_U, tol=1e-10):
    """Local (inner) truncation + Sorkin entropy computation.

    Steps:
    1. Eigendecompose H_U = i*Delta_U0 (restricted outer PJ)
    2. Keep eigenvalues with |τ| > kappa_U → support basis B
    3. Project BOTH W_U0 and Delta_U0 onto this support
    4. Solve A = (i*Delta_proj)^{-1} * W_proj for eigenvalues λ
    5. S = Σ λ ln|λ|

    Critical: W is PROJECTED (not recomputed as Pos).
    This preserves the state information from the outer truncation.
    """
    H_U = 1j * Delta_U0
    tau, E = np.linalg.eigh(H_U)

    # Support = eigenvalues above inner threshold
    support = np.abs(tau) > kappa_U
    n_support = int(support.sum())

    if n_support < 4:
        return 0.0, np.array([]), {'n_support': n_support, 'error': 'too_few_modes'}

    B = E[:, support]  # support basis (N_U × n_support)

    # Project BOTH onto support
    W_proj = B.conj().T @ W_U0 @ B       # n_support × n_support
    iD_proj = B.conj().T @ H_U @ B        # diagonal in this basis (eigenvalues τ)

    # iD_proj is diagonal (eigenbasis of H_U restricted to support)
    iD_diag = np.diag(iD_proj)

    # Check diagonality
    offdiag_norm = np.linalg.norm(iD_proj - np.diag(iD_diag))
    if offdiag_norm > 1e-8 * np.linalg.norm(iD_proj):
        # Not diagonal — use general solve
        A = np.linalg.solve(iD_proj, W_proj)
    else:
        # Diagonal — efficient inversion
        inv_iD = np.diag(1.0 / iD_diag)
        A = inv_iD @ W_proj

    # Eigenvalues of A = generalized eigenvalues λ
    lam = np.linalg.eigvals(A)

    # Check: eigenvalues should be real
    max_imag = float(np.max(np.abs(lam.imag)))
    lam_real = lam.real

    # Sorkin entropy: S = Σ λ ln|λ|
    valid = np.abs(lam_real) > 1e-14
    S = float(np.sum(lam_real[valid] * np.log(np.abs(lam_real[valid]))))

    info = {
        'n_support': n_support,
        'kappa_U': float(kappa_U),
        'max_imag': max_imag,
        'lam_min': float(lam_real.min()),
        'lam_max': float(lam_real.max()),
        'n_valid': int(valid.sum()),
        'offdiag_norm': float(offdiag_norm),
    }
    return S, lam_real, info


def compute_sj_entropy(pts, parents, T_outer, T_inner, kappa_factor_O=1.0,
                        kappa_factor_U=1.0):
    """Full SJ entropy pipeline: sprinkle → Hasse → PJ → double truncation → S.

    Args:
        pts: sprinkled points (N × 4), time-sorted
        parents: list of parent arrays from build_hasse_from_predicate
        T_outer: outer diamond duration
        T_inner: inner diamond duration (T_outer/√2 for V_O/V_U=4)
        kappa_factor_O: multiplier for outer threshold √N/(4π)
        kappa_factor_U: multiplier for inner threshold √N_U/(4π)

    Returns:
        S: Sorkin entropy of inner region
        info: diagnostic dict
    """
    N = len(pts)

    # Build link matrix
    L = hasse_to_link_matrix(parents, N)

    # PJ operator (unnormalized — α_J cancels in entropy)
    Delta = L - L.T  # real antisymmetric N×N

    # Global truncation
    kappa_O = kappa_factor_O * np.sqrt(N) / (4.0 * np.pi)
    W_O, Delta_O, info_O = global_truncation(Delta, kappa_O)

    # Inner subdiamond
    idx_U = inner_subdiamond_indices(pts, T_inner)
    N_U = len(idx_U)

    if N_U < 20:
        return 0.0, {'error': 'too_few_inner', 'N_U': N_U, **info_O}

    # Restrict to inner region
    W_U0 = W_O[np.ix_(idx_U, idx_U)]
    Delta_U0 = Delta_O[np.ix_(idx_U, idx_U)]

    # Local truncation + entropy
    kappa_U = kappa_factor_U * np.sqrt(N_U) / (4.0 * np.pi)
    S, lambdas, info_U = local_truncation_and_entropy(W_U0, Delta_U0, kappa_U)

    info = {
        'N': N,
        'N_U': N_U,
        'T_outer': T_outer,
        'T_inner': T_inner,
        **{f'outer_{k}': v for k, v in info_O.items()},
        **{f'inner_{k}': v for k, v in info_U.items()},
        'S': S,
    }
    return S, info


def compute_CJ_from_hasse(pts, par0, parC, T, zeta=0.15):
    """Compute CJ for the same seed (for correlation test).
    Reuses the GTA path-resolvent machinery from bridge scripts.
    """
    n = len(pts)
    bmask = bulk_mask(pts, T, zeta)

    # Y from flat and curved Hasse
    def Y_from_graph(par):
        p_down = np.ones(n, dtype=np.float64)
        p_up = np.ones(n, dtype=np.float64)
        ch_lists = [[] for _ in range(n)]
        for i in range(n):
            if par[i] is not None and len(par[i]) > 0:
                for j in par[i]:
                    ch_lists[int(j)].append(i)
        for i in range(n):
            if par[i] is not None and len(par[i]) > 0:
                p_down[i] = np.sum(p_down[list(par[i])]) + 1
        for i in range(n - 1, -1, -1):
            if ch_lists[i]:
                p_up[i] = np.sum(p_up[ch_lists[i]]) + 1
        return np.log2(p_down * p_up + 1)

    Y0 = Y_from_graph(par0)
    YC = Y_from_graph(parC)
    delta = YC - Y0

    # Strata
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if par0[i] is not None and len(par0[i]) > 0:
            depth[i] = int(np.max(depth[list(par0[i])])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    strata = tau_bin * 9 + rho_bin * 3 + depth_terc

    # CJ = Σ_B w_B Cov_B(|X|, δY²)
    X = Y0[bmask] - np.mean(Y0[bmask])
    dY2 = delta[bmask] ** 2
    strata_m = strata[bmask]
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


# ===================================================================
#  CRN Entropy Run
# ===================================================================

def crn_entropy_run(seed, N, T, T_inner, geometry, eps=None, M_sch=None,
                    r0_sch=None, kf_O=1.0, kf_U=1.0, compute_cj=False):
    """One CRN entropy seed: compute S_flat, S_curved, δS, and optionally CJ.

    Args:
        seed: random seed
        N: number of points
        T: outer diamond duration
        T_inner: inner diamond duration
        geometry: 'ppwave' or 'schwarzschild'
        eps: pp-wave amplitude (if geometry='ppwave')
        M_sch, r0_sch: Schwarzschild parameters (if geometry='schwarzschild')
        kf_O, kf_U: threshold multipliers
        compute_cj: if True, also compute CJ for correlation test

    Returns:
        dict with S_flat, S_curved, deltaS, CJ (if requested), and diagnostics
    """
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)

    # Flat Hasse
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))

    # Curved Hasse (CRN: same points, different causal order)
    if geometry == 'ppwave':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
    elif geometry == 'schwarzschild':
        R_abcd = riemann_schwarzschild_local(M_sch, r0_sch)
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, R_abcd))
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # Flat entropy
    S_flat, info_flat = compute_sj_entropy(pts, par0, T, T_inner, kf_O, kf_U)

    # Curved entropy
    S_curved, info_curved = compute_sj_entropy(pts, parC, T, T_inner, kf_O, kf_U)

    result = {
        'seed': seed,
        'S_flat': S_flat,
        'S_curved': S_curved,
        'deltaS': S_curved - S_flat,
        'N': N,
        'N_U_flat': info_flat.get('N_U', 0),
        'N_U_curved': info_curved.get('N_U', 0),
        'modes_flat': info_flat.get('inner_n_support', 0),
        'modes_curved': info_curved.get('inner_n_support', 0),
    }

    # Optionally compute CJ for correlation test
    if compute_cj:
        cj = compute_CJ_from_hasse(pts, par0, parC, T)
        result['CJ'] = cj

    return result


# ===================================================================
#  Phase runners
# ===================================================================

def run_phase1_4d_flat(N=2000, M_seeds=5, T=1.0):
    """Phase 1: 4D flat calibration.
    Compute S_U on flat Minkowski nested diamonds.
    Calibrate threshold, measure noise.
    """
    print("=" * 72)
    print(f"Phase 1: 4D Flat Calibration  N={N}  M={M_seeds}  T={T}")
    print("=" * 72)

    T_inner = T / np.sqrt(2)
    results = []

    for s in range(M_seeds):
        seed = 8000000 + s
        t0 = time.time()
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        par0, ch0 = build_hasse_from_predicate(
            pts, lambda P, i: minkowski_preds(P, i))

        # Try several kappa factors to find right calibration
        S_vals = {}
        for kf in [1.0, 2.0, 3.0]:
            S, info = compute_sj_entropy(pts, par0, T, T_inner,
                                          kappa_factor_O=kf, kappa_factor_U=kf)
            N_U = info.get('N_U', 0)
            modes = info.get('inner_n_support', 0)
            S_vals[kf] = (S, N_U, modes)

        dt = time.time() - t0
        print(f"  seed {s}: ", end="")
        for kf, (S, N_U, modes) in S_vals.items():
            print(f"kf={kf}: S={S:+.3f}(N_U={N_U},m={modes})  ", end="")
        print(f"({dt:.1f}s)")
        results.append({'seed': seed, 'S_vals': S_vals, 'time': dt})

    # Summary
    print()
    for kf in [1.0, 2.0, 3.0]:
        S_list = [r['S_vals'][kf][0] for r in results]
        N_U_list = [r['S_vals'][kf][1] for r in results]
        modes_list = [r['S_vals'][kf][2] for r in results]
        print(f"  kf={kf}: <S>={np.mean(S_list):+.3f} ± {np.std(S_list):.3f}  "
              f"<N_U>={np.mean(N_U_list):.0f}  <modes>={np.mean(modes_list):.0f}")
        # Area law check: S should scale as √N_U in 4D
        for r in results:
            S, N_U, m = r['S_vals'][kf]
            if N_U > 0:
                pass  # would need multiple N to check scaling

    return results


def run_phase2_crn_pilot(N=2000, M_seeds=20, T=1.0, geometry='schwarzschild',
                          eps=3.0, M_sch=0.05, r0_sch=0.50, kf=1.0):
    """Phase 2: CRN Schwarzschild pilot.
    Compute δS for flat-curved pairs. Also compute CJ for correlation.
    """
    print("=" * 72)
    print(f"Phase 2: CRN {geometry} Pilot  N={N}  M={M_seeds}")
    print("=" * 72)

    T_inner = T / np.sqrt(2)
    results = []

    for s in range(M_seeds):
        seed = 9000000 + s
        t0 = time.time()
        r = crn_entropy_run(seed, N, T, T_inner, geometry,
                            eps=eps, M_sch=M_sch, r0_sch=r0_sch,
                            kf_O=kf, kf_U=kf, compute_cj=True)
        dt = time.time() - t0

        print(f"  seed {s:2d}: S_flat={r['S_flat']:+.4f}  S_curved={r['S_curved']:+.4f}  "
              f"δS={r['deltaS']:+.5f}  CJ={r.get('CJ', 0):+.6f}  ({dt:.1f}s)")

        r['time'] = dt
        results.append(r)

    # Summary
    print()
    deltaS = [r['deltaS'] for r in results]
    CJ_vals = [r.get('CJ', 0) for r in results]
    mean_dS = np.mean(deltaS)
    se_dS = np.std(deltaS) / np.sqrt(len(deltaS))
    t_stat_dS = mean_dS / se_dS if se_dS > 0 else 0

    mean_CJ = np.mean(CJ_vals)
    se_CJ = np.std(CJ_vals) / np.sqrt(len(CJ_vals))
    t_stat_CJ = mean_CJ / se_CJ if se_CJ > 0 else 0

    print(f"  <δS>  = {mean_dS:+.5f} ± {se_dS:.5f}  (t={t_stat_dS:+.2f})")
    print(f"  <CJ>  = {mean_CJ:+.6f} ± {se_CJ:.6f}  (t={t_stat_CJ:+.2f})")

    if len(deltaS) > 3:
        corr = np.corrcoef(deltaS, CJ_vals)[0, 1]
        print(f"  corr(δS, CJ) = {corr:+.3f}")
    else:
        corr = None

    return results, {'mean_dS': mean_dS, 'se_dS': se_dS, 't_dS': t_stat_dS,
                     'mean_CJ': mean_CJ, 'corr_dS_CJ': corr}


# ===================================================================
#  Main
# ===================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, help='Phase to run (1 or 2)')
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--kf', type=float, default=1.0, help='Threshold factor')
    parser.add_argument('--geometry', default='schwarzschild')
    parser.add_argument('--eps', type=float, default=3.0)
    parser.add_argument('--M_sch', type=float, default=0.05)
    parser.add_argument('--r0', type=float, default=0.50)
    args = parser.parse_args()

    if args.phase == 1:
        results = run_phase1_4d_flat(N=args.N, M_seeds=args.M)
    elif args.phase == 2:
        results, summary = run_phase2_crn_pilot(
            N=args.N, M_seeds=args.M, geometry=args.geometry,
            eps=args.eps, M_sch=args.M_sch, r0_sch=args.r0, kf=args.kf)

        # Save
        out_path = os.path.join(os.path.dirname(__file__), '..', 'fnd1_data',
                                f'sj_entropy_phase2_{args.geometry}.json')
        with open(out_path, 'w') as f:
            json.dump({'results': results, 'summary': summary}, f, indent=2)
        print(f"\nSaved to {out_path}")
