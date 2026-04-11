#!/usr/bin/env python3
"""
Optimized κ-scan: build Hasse ONCE per seed, scan kf on cached eigendecomposition.

This avoids redundant Hasse builds and eigendecompositions across kf values.
For each seed:
  1. Sprinkle points (once)
  2. Build flat + curved Hasse (once each)
  3. Build link matrix + PJ + eigendecompose H = i(L-L^T) (once each)
  4. For each kf: apply threshold, restrict, inner truncation, entropy

Saves factor ~5× in compute time vs naive kf-loop.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds,
    build_hasse_from_predicate, riemann_schwarzschild_local, jet_preds,
)


def hasse_to_link_matrix(parents, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def entropy_from_cached_eig(sigma, U, pts, kf, T_inner):
    """Compute SJ entropy from cached outer eigendecomposition.

    Args:
        sigma: eigenvalues of H = i(L-L^T), ascending
        U: eigenvectors
        pts: point coordinates (for inner region selection)
        kf: threshold factor
        T_inner: inner diamond duration

    Returns:
        S, N_U, n_support
    """
    N = len(sigma)
    kappa_O = kf * np.sqrt(N) / (4.0 * np.pi)

    keep_O = np.abs(sigma) > kappa_O
    pos_O = sigma > kappa_O

    # Build W_O and Delta_O from cached eigendecomposition
    U_pos = U[:, pos_O]
    s_pos = sigma[pos_O]
    W_O = (U_pos * s_pos[None, :]) @ U_pos.conj().T

    U_keep = U[:, keep_O]
    s_keep = sigma[keep_O]
    H_O_cut = (U_keep * s_keep[None, :]) @ U_keep.conj().T
    Delta_O = np.real(-1j * H_O_cut)

    # Inner subdiamond
    t_coord = pts[:, 0]
    r_coord = np.linalg.norm(pts[:, 1:], axis=1)
    idx_U = np.where(np.abs(t_coord) + r_coord < T_inner / 2.0)[0]
    N_U = len(idx_U)

    if N_U < 20:
        return 0.0, N_U, 0

    # Restrict
    W_U0 = W_O[np.ix_(idx_U, idx_U)]
    Delta_U0 = Delta_O[np.ix_(idx_U, idx_U)]

    # Inner eigendecomposition
    H_U = 1j * Delta_U0
    tau, E = np.linalg.eigh(H_U)
    kappa_U = kf * np.sqrt(N_U) / (4.0 * np.pi)
    keep_U = np.abs(tau) > kappa_U
    n_support = int(keep_U.sum())

    if n_support < 4:
        return 0.0, N_U, n_support

    B = E[:, keep_U]
    W_proj = B.conj().T @ W_U0 @ B
    iD_proj = B.conj().T @ H_U @ B
    iD_diag = np.diag(iD_proj)

    inv_iD = np.diag(1.0 / iD_diag)
    A = inv_iD @ W_proj

    lam = np.linalg.eigvals(A).real
    valid = np.abs(lam) > 1e-14
    S = float(np.sum(lam[valid] * np.log(np.abs(lam[valid]))))

    return S, N_U, n_support


def run_optimized_kappa_scan(N, M_seeds, T=1.0, M_sch=0.05, r0_sch=0.50,
                              kf_values=[1.0, 2.0, 3.0, 4.0, 5.0]):
    """Optimized κ-scan: eigendecompose once per seed, scan kf."""
    print("=" * 72)
    print(f"Optimized κ-scan  N={N}  M={M_seeds}  Sch M={M_sch} r₀={r0_sch}")
    print(f"kf values: {kf_values}")
    print("=" * 72)

    T_inner = T / np.sqrt(2)
    R_abcd = riemann_schwarzschild_local(M_sch, r0_sch)

    # Collect δS for each (seed, kf)
    all_deltaS = {kf: [] for kf in kf_values}

    for s in range(M_seeds):
        seed = 7000000 + s
        t0 = time.time()

        # 1. Sprinkle (once)
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        # 2. Build Hasse (once each)
        t1 = time.time()
        par0, _ = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        parC, _ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd))
        t_hasse = time.time() - t1

        # 3. Build link matrices and eigendecompose (once each)
        L0 = hasse_to_link_matrix(par0, N)
        LC = hasse_to_link_matrix(parC, N)

        t1 = time.time()
        H0 = 1j * (L0 - L0.T)
        sigma0, U0 = np.linalg.eigh(H0)
        HC = 1j * (LC - LC.T)
        sigmaC, UC = np.linalg.eigh(HC)
        t_eig = time.time() - t1

        # 4. Scan kf
        kf_results = []
        for kf in kf_values:
            S_flat, N_U, m0 = entropy_from_cached_eig(sigma0, U0, pts, kf, T_inner)
            S_curved, _, mC = entropy_from_cached_eig(sigmaC, UC, pts, kf, T_inner)
            dS = S_curved - S_flat
            all_deltaS[kf].append(dS)
            kf_results.append(f"kf={kf}: δS={dS:+.5f}(m={m0})")

        dt = time.time() - t0
        print(f"  seed {s:2d} ({dt:.0f}s, hasse={t_hasse:.0f}s, eig={t_eig:.0f}s): "
              + "  ".join(kf_results))

    # Summary
    print()
    print("SUMMARY:")
    print(f"{'kf':>5s}  {'<δS>':>10s}  {'SE':>8s}  {'σ':>8s}  {'t':>7s}")
    print("-" * 45)
    for kf in kf_values:
        vals = all_deltaS[kf]
        m = np.mean(vals)
        sd = np.std(vals)
        se = sd / np.sqrt(len(vals))
        t = m / se if se > 0 else 0
        print(f"{kf:5.1f}  {m:+10.5f}  {se:8.5f}  {sd:8.5f}  {t:+7.2f}")

    # Expected signal
    delta_JB = -8.0 * M_sch / r0_sch
    b_log_expected = delta_JB / (4.0 * 120.0)  # c=1/120
    signal_expected = b_log_expected * np.log(N)
    print(f"\nExpected: δS_log = {signal_expected:+.5f}  (δJ_B={delta_JB:+.3f}, c=1/120)")

    return all_deltaS


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--M_sch', type=float, default=0.05)
    parser.add_argument('--r0', type=float, default=0.50)
    args = parser.parse_args()

    results = run_optimized_kappa_scan(
        args.N, args.M, M_sch=args.M_sch, r0_sch=args.r0,
        kf_values=[1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])

    out = os.path.join(os.path.dirname(__file__), '..', 'fnd1_data',
                       f'sj_entropy_kappa_scan_opt_N{args.N}.json')
    save = {str(kf): v for kf, v in results.items()}
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved to {out}")
