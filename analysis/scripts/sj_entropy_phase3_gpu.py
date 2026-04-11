#!/usr/bin/env python3
"""
Phase 3: SJ Entropy κ-scan and radius-scan at N=8000 with GPU.

Step 1: κ-scan — find plateau in δS vs kf at fixed r₀
Step 2: radius-scan — measure δS at several r₀ values, fixed N, fixed kf
Step 3: extract c_cs = -[δS(r₁) - δS(r₂)] / [2M(1/r₁ - 1/r₂) ln N]

GPU: CuPy for dense Hermitian eigendecomposition (N×N complex128).
"""
import sys, os, time, json
import numpy as np

# GPU preamble
_cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_cuda):
    os.add_dll_directory(_cuda)

try:
    import cupy as cp
    from cupyx.scipy.linalg import eigh as cp_eigh
    GPU_AVAILABLE = True
    print(f"GPU: {cp.cuda.runtime.getDeviceCount()} device(s), "
          f"VRAM: {cp.cuda.Device(0).mem_info[1] / 1e9:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU not available, using CPU")

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds,
    build_hasse_from_predicate, riemann_schwarzschild_local, jet_preds,
    ppwave_exact_preds,
)


def hasse_to_link_matrix(parents, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def eigh_gpu_or_cpu(H):
    """Hermitian eigendecomposition, GPU if available."""
    if GPU_AVAILABLE and H.shape[0] >= 1000:
        H_gpu = cp.asarray(H)
        sigma_gpu, U_gpu = cp.linalg.eigh(H_gpu)
        sigma = cp.asnumpy(sigma_gpu)
        U = cp.asnumpy(U_gpu)
        del H_gpu, sigma_gpu, U_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return sigma, U
    else:
        return np.linalg.eigh(H)


def compute_entropy_pipeline(pts, parents, T_outer, T_inner, kf=1.0):
    """Full SJ entropy pipeline with GPU eigendecomposition.

    Returns S, diagnostic dict.
    """
    N = len(pts)
    t_start = time.time()

    # Build link matrix
    L = hasse_to_link_matrix(parents, N)
    t_hasse = time.time() - t_start

    # PJ operator
    Delta = (L - L.T).astype(np.float64)
    H = 1j * Delta  # Hermitian complex

    # Global truncation
    kappa_O = kf * np.sqrt(N) / (4.0 * np.pi)
    t0 = time.time()
    sigma, U = eigh_gpu_or_cpu(H)
    t_eig_O = time.time() - t0

    keep_O = np.abs(sigma) > kappa_O
    pos_O = sigma > kappa_O
    n_kept_O = int(keep_O.sum())
    n_pos_O = int(pos_O.sum())

    # Build W_O (positive part) and Delta_O (truncated)
    U_pos = U[:, pos_O]
    s_pos = sigma[pos_O]
    W_O = (U_pos * s_pos[None, :]) @ U_pos.conj().T

    U_keep = U[:, keep_O]
    s_keep = sigma[keep_O]
    H_O_cut = (U_keep * s_keep[None, :]) @ U_keep.conj().T
    Delta_O = np.real(-1j * H_O_cut)

    del U, U_pos, U_keep  # free memory

    # Inner subdiamond
    t_coord = pts[:, 0]
    r_coord = np.linalg.norm(pts[:, 1:], axis=1)
    idx_U = np.where(np.abs(t_coord) + r_coord < T_inner / 2.0)[0]
    N_U = len(idx_U)

    if N_U < 20:
        return 0.0, {'error': 'too_few_inner', 'N_U': N_U}

    # Restrict
    W_U0 = W_O[np.ix_(idx_U, idx_U)]
    Delta_U0 = Delta_O[np.ix_(idx_U, idx_U)]
    del W_O, Delta_O  # free

    # Inner eigendecomposition
    H_U = 1j * Delta_U0
    t0 = time.time()
    tau, E = eigh_gpu_or_cpu(H_U)
    t_eig_U = time.time() - t0

    kappa_U = kf * np.sqrt(N_U) / (4.0 * np.pi)
    keep_U = np.abs(tau) > kappa_U
    n_support = int(keep_U.sum())

    if n_support < 4:
        return 0.0, {'error': 'too_few_support', 'n_support': n_support}

    B = E[:, keep_U]

    # Project W and iΔ onto support
    W_proj = B.conj().T @ W_U0 @ B
    iD_proj = B.conj().T @ H_U @ B
    iD_diag = np.diag(iD_proj)

    # A = (iΔ)^{-1} W on support
    inv_iD = np.diag(1.0 / iD_diag)
    A = inv_iD @ W_proj

    lam = np.linalg.eigvals(A)
    lam_real = lam.real
    max_imag = float(np.max(np.abs(lam.imag)))

    # Sorkin entropy: S = Σ λ ln|λ|
    valid = np.abs(lam_real) > 1e-14
    S = float(np.sum(lam_real[valid] * np.log(np.abs(lam_real[valid]))))

    t_total = time.time() - t_start

    info = {
        'N': N, 'N_U': N_U, 'S': S,
        'kf': kf, 'kappa_O': float(kappa_O), 'kappa_U': float(kappa_U),
        'n_kept_O': n_kept_O, 'n_pos_O': n_pos_O,
        'n_support': n_support, 'max_imag': max_imag,
        'lam_min': float(lam_real.min()), 'lam_max': float(lam_real.max()),
        't_hasse': t_hasse, 't_eig_O': t_eig_O, 't_eig_U': t_eig_U,
        't_total': t_total, 'T_inner': T_inner,
    }
    return S, info


def crn_run(seed, N, T, T_inner, geometry, kf=1.0,
            eps=None, M_sch=None, r0_sch=None):
    """One CRN pair: flat + curved entropy."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)

    # Flat Hasse
    par0, _ = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))

    # Curved Hasse
    if geometry == 'ppwave':
        parC, _ = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
    elif geometry == 'schwarzschild':
        R_abcd = riemann_schwarzschild_local(M_sch, r0_sch)
        parC, _ = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, R_abcd))
    else:
        raise ValueError(f"Unknown: {geometry}")

    S_flat, info_flat = compute_entropy_pipeline(pts, par0, T, T_inner, kf)
    S_curved, info_curved = compute_entropy_pipeline(pts, parC, T, T_inner, kf)

    return {
        'seed': seed,
        'S_flat': S_flat,
        'S_curved': S_curved,
        'deltaS': S_curved - S_flat,
        'N_U': info_flat.get('N_U', 0),
        'modes': info_flat.get('n_support', 0),
        't_total': info_flat.get('t_total', 0) + info_curved.get('t_total', 0),
    }


def run_kappa_scan(N, M_seeds, T=1.0, M_sch=0.05, r0_sch=0.50,
                    kf_values=[1.0, 2.0, 3.0, 4.0, 5.0]):
    """Step 1: find plateau in δS vs kf."""
    print("=" * 72)
    print(f"Phase 3 Step 1: κ-scan  N={N}  M={M_seeds}")
    print("=" * 72)

    T_inner = T / np.sqrt(2)

    all_results = {}
    for kf in kf_values:
        ds_vals = []
        for s in range(M_seeds):
            seed = 7000000 + s
            r = crn_run(seed, N, T, T_inner, 'schwarzschild',
                        kf=kf, M_sch=M_sch, r0_sch=r0_sch)
            ds_vals.append(r['deltaS'])
            if s == 0:
                print(f"  kf={kf:.1f}: seed 0 → δS={r['deltaS']:+.5f}  "
                      f"N_U={r['N_U']}  modes={r['modes']}  ({r['t_total']:.1f}s)")

        m = np.mean(ds_vals)
        sd = np.std(ds_vals)
        se = sd / np.sqrt(len(ds_vals))
        t_stat = m / se if se > 0 else 0
        print(f"  kf={kf:.1f}: <δS>={m:+.5f} ± {se:.5f}  (σ={sd:.5f}, t={t_stat:+.2f})")
        all_results[kf] = {'mean': m, 'se': se, 'sigma': sd, 't': t_stat,
                           'values': ds_vals}

    return all_results


def run_radius_scan(N, M_seeds, kf, T=1.0, M_sch=0.05, r0_sch=0.50,
                     r0_values=[0.30, 0.35, 0.40, 0.45, 0.50]):
    """Step 2: radius-scan at fixed N and kf."""
    print("=" * 72)
    print(f"Phase 3 Step 2: Radius-scan  N={N}  kf={kf}  M={M_seeds}")
    print("=" * 72)

    # For each r₀, the inner diamond has T_inner = 2*r₀
    # (equatorial radius of diamond = T_inner/2 = r₀)
    # δJ_B = -8M/r₀ for Schwarzschild

    all_results = {}
    for r0 in r0_values:
        T_inner = 2.0 * r0  # so equatorial radius = r0
        ds_vals = []
        for s in range(M_seeds):
            seed = 7000000 + s
            r = crn_run(seed, N, T, T_inner, 'schwarzschild',
                        kf=kf, M_sch=M_sch, r0_sch=r0_sch)
            ds_vals.append(r['deltaS'])

        m = np.mean(ds_vals)
        sd = np.std(ds_vals)
        se = sd / np.sqrt(len(ds_vals))
        t_stat = m / se if se > 0 else 0

        inv_r0 = 1.0 / r0
        delta_JB = -8.0 * M_sch / r0

        print(f"  r₀={r0:.2f}  1/r₀={inv_r0:.2f}  δJ_B={delta_JB:+.3f}  "
              f"<δS>={m:+.5f} ± {se:.5f}  (t={t_stat:+.2f})")

        all_results[r0] = {
            'mean': m, 'se': se, 'sigma': sd, 't': t_stat,
            'inv_r0': inv_r0, 'delta_JB': delta_JB,
            'values': ds_vals,
        }

    # Fit δS vs 1/r₀
    r0s = sorted(all_results.keys())
    inv_r0s = [1.0 / r for r in r0s]
    means = [all_results[r]['mean'] for r in r0s]

    if len(r0s) >= 3:
        # Linear fit: δS = a / r₀ + b
        X = np.column_stack([inv_r0s, np.ones(len(r0s))])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, means, rcond=None)
        slope, intercept = coeffs

        # c_cs from slope: slope = -2cM ln(N)
        # → c = -slope / (2M ln(N))
        ln_N = np.log(N)
        c_cs = -slope / (2.0 * M_sch * ln_N)

        print(f"\n  Fit: δS = {slope:+.5f}/r₀ + {intercept:+.5f}")
        print(f"  c_cs = -slope / (2M ln N) = {c_cs:.6f}")
        print(f"  Target: 1/120 = {1/120:.6f}")
        print(f"  Ratio: c_cs / (1/120) = {c_cs * 120:.3f}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=1, help='1=kappa-scan, 2=radius-scan')
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--kf', type=float, default=3.0)
    parser.add_argument('--M_sch', type=float, default=0.05)
    parser.add_argument('--r0', type=float, default=0.50)
    args = parser.parse_args()

    if args.step == 1:
        results = run_kappa_scan(args.N, args.M, M_sch=args.M_sch, r0_sch=args.r0)

        out = os.path.join(os.path.dirname(__file__), '..', 'fnd1_data',
                           f'sj_entropy_kappa_scan_N{args.N}.json')
        # Convert for JSON
        save = {}
        for kf, v in results.items():
            save[str(kf)] = {k: v2 for k, v2 in v.items()}
        with open(out, 'w') as f:
            json.dump(save, f, indent=2)
        print(f"\nSaved to {out}")

    elif args.step == 2:
        results = run_radius_scan(args.N, args.M, args.kf,
                                   M_sch=args.M_sch, r0_sch=args.r0)
        out = os.path.join(os.path.dirname(__file__), '..', 'fnd1_data',
                           f'sj_entropy_radius_scan_N{args.N}_kf{args.kf}.json')
        save = {}
        for r0, v in results.items():
            save[str(r0)] = {k: v2 for k, v2 in v.items()}
        with open(out, 'w') as f:
            json.dump(save, f, indent=2)
        print(f"\nSaved to {out}")
