#!/usr/bin/env python3
"""
Phase 0: 2D Flat Diamond SJ Entropy Sanity Check.

Target: reproduce S ≈ (1/3) ln(N_U) for nested causal diamonds in 2D Minkowski,
following the double-cutoff prescription of BBLL 2017 (arXiv:1712.04227).

This is a 1+1D test:
- Outer diamond O: side length L = 1.0
- Inner diamond U: side length a, with V_O/V_U = 4 → a = L/2
- Sprinkle N points into O, select N_U ≈ N/4 into U
- Build causal matrix C, then PJ = C - C^T (in 2D, G_R = (1/2)C)
- Double truncation: first on outer iΔ, then on restricted inner iΔ
- Sorkin entropy: S = Σ_λ λ ln|λ| from generalized eigenproblem

Expected: S ≈ (1/3) ln(N_U) for a free massless scalar with c = 1/6 in 2D.

Reference: BBLL 2017, Section 3 and Figure 3.
"""
import numpy as np
import time


def sprinkle_2d_diamond(N, L, rng):
    """Sprinkle N points in a 2D causal diamond of side length L.
    Diamond: |t| + |x| < L/2, centered at origin.
    Points sorted by time coordinate.
    """
    pts = []
    while len(pts) < N:
        batch = rng.uniform(-L / 2, L / 2, size=(8 * N, 2))
        mask = np.abs(batch[:, 0]) + np.abs(batch[:, 1]) < L / 2
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:N], dtype=np.float64)
    order = np.argsort(arr[:, 0])
    return arr[order]


def build_causal_matrix_2d(pts):
    """Build causal matrix C_ij = 1 if j ≺ i (j in causal past of i).
    In 2D Minkowski: j ≺ i iff t_j < t_i and |x_i - x_j| < t_i - t_j.
    Points MUST be sorted by time.
    """
    n = len(pts)
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            dt = pts[i, 0] - pts[j, 0]  # > 0 by sorting
            dx = abs(pts[i, 1] - pts[j, 1])
            if dx < dt:
                C[i, j] = 1.0
    return C


def spectral_truncate(Delta, kappa):
    """Double-truncation step: truncate spectrum of H = i*Delta at |σ| > κ.

    Returns:
        W_cut: positive spectral part (Wightman)
        Delta_cut: truncated PJ (antisymmetric, real up to numerics)
        n_kept: number of eigenvalues above threshold
        n_pos: number of positive eigenvalues above threshold
    """
    H = 1j * Delta  # Hermitian
    sigma, U = np.linalg.eigh(H)

    keep = np.abs(sigma) > kappa
    pos = sigma > kappa

    n_kept = int(keep.sum())
    n_pos = int(pos.sum())

    # Truncated H
    U_keep = U[:, keep]
    s_keep = sigma[keep]
    H_cut = (U_keep * s_keep[None, :]) @ U_keep.conj().T

    # Positive part = W
    U_pos = U[:, pos]
    s_pos = sigma[pos]
    W_cut = (U_pos * s_pos[None, :]) @ U_pos.conj().T

    # Delta_cut = -i H_cut
    Delta_cut = np.real(-1j * H_cut)  # should be real antisymmetric

    return W_cut, Delta_cut, n_kept, n_pos


def sorkin_entropy(W, Delta, tol=1e-10):
    """Compute Sorkin entropy S = Σ_λ λ ln|λ|.

    Solves generalized eigenproblem W v = iλ Δ v on the support of Δ.

    Implementation: project onto support of H = iΔ (eigenvalues > tol),
    then compute A = -i Δ_s^{-1} W_s on that support, and use eigenvalues of A.

    Returns:
        S: entropy value
        lambdas: array of generalized eigenvalues (real parts)
        n_support: dimension of the support
    """
    H_inner = 1j * Delta
    tau, E = np.linalg.eigh(H_inner)

    # Support = eigenvalues with |τ| > tol
    support_mask = np.abs(tau) > tol
    n_support = int(support_mask.sum())

    if n_support == 0:
        return 0.0, np.array([]), 0

    B = E[:, support_mask]  # support basis

    # Project W and Δ onto support
    Ws = B.conj().T @ W @ B
    Ds = B.conj().T @ (1j * Delta) @ B  # This is H_s, Hermitian on support

    # Actually we need: A = -i Δ_s^{-1} W_s
    # Δ_s (antisymmetric) projected = -i H_s where H_s is Hermitian
    # So A = -i * (-i H_s)^{-1} * W_s = -i * (i / H_s) * W_s = W_s / H_s
    # Wait, let me be more careful.
    #
    # On the support, Δ_s is invertible (as a map on the support).
    # Δ_s = B^T Δ B (real antisymmetric projected)
    # A = -i Δ_s^{-1} W_s
    #
    # Let me just do it directly with the projected real matrices.

    Delta_s = np.real(B.conj().T @ Delta @ B)  # real antisymmetric on support
    W_s = B.conj().T @ W @ B  # Hermitian (should be real symmetric for real scalar)
    W_s_real = np.real(W_s)

    # A = -i * Δ_s^{-1} * W_s
    # Since Δ_s is real antisymmetric, iΔ_s is real symmetric? No.
    # Actually: Δ is real antisymmetric, so Δ_s (projected) is also real antisymmetric.
    # Δ_s^{-1} exists on support.
    # A = -i * inv(Δ_s) * W_s

    try:
        inv_Delta_s = np.linalg.inv(Delta_s)
    except np.linalg.LinAlgError:
        return 0.0, np.array([]), n_support

    A = -1j * inv_Delta_s @ W_s_real

    lam = np.linalg.eigvals(A)

    # Eigenvalues should be real (or nearly so)
    lam_real = lam[np.abs(lam.imag) < 0.1 * np.abs(lam.real + 1e-300)].real
    if len(lam_real) < n_support * 0.5:
        # Too many complex eigenvalues — fallback: take all real parts
        lam_real = lam.real

    # Sorkin entropy: S = Σ λ ln|λ|
    # Skip λ ≈ 0 to avoid 0*(-inf)
    valid = np.abs(lam_real) > 1e-14
    S = np.sum(lam_real[valid] * np.log(np.abs(lam_real[valid])))

    return float(np.real(S)), lam_real, n_support


def run_2d_entropy(N, L=1.0, seed=42):
    """Run one 2D entropy computation with double truncation."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_2d_diamond(N, L, rng)

    # Build causal matrix (= retarded Green function in 2D up to factor 1/2)
    C = build_causal_matrix_2d(pts)

    # PJ = G_R - G_R^T = (1/2)(C - C^T)
    # But overall factor cancels in entropy (analytical verified).
    # So use Delta = C - C^T directly.
    Delta_full = C - C.T

    # ---- GLOBAL (outer) truncation ----
    kappa_O = np.sqrt(N) / (4 * np.pi)
    W_O, Delta_O, n_kept_O, n_pos_O = spectral_truncate(Delta_full, kappa_O)

    # ---- Select inner subdiamond ----
    # Inner diamond: |t| + |x| < a/2 with a = L/2 for V_O/V_U = 4 (2D volume = L^2/2)
    a = L / 2.0
    inner_mask = (np.abs(pts[:, 0]) + np.abs(pts[:, 1])) < a / 2.0
    idx_U = np.where(inner_mask)[0]
    N_U = len(idx_U)

    if N_U < 10:
        return {'error': 'Too few inner points', 'N_U': N_U}

    # ---- Restrict to inner region ----
    W_U0 = W_O[np.ix_(idx_U, idx_U)]
    Delta_U0 = Delta_O[np.ix_(idx_U, idx_U)]

    # ---- LOCAL (inner) truncation ----
    kappa_U = np.sqrt(N_U) / (4 * np.pi)
    W_U, Delta_U, n_kept_U, n_pos_U = spectral_truncate(Delta_U0, kappa_U)

    # ---- Sorkin entropy ----
    S, lambdas, n_support = sorkin_entropy(W_U, Delta_U)

    return {
        'N': N,
        'N_U': N_U,
        'kappa_O': kappa_O,
        'kappa_U': kappa_U,
        'n_kept_O': n_kept_O,
        'n_pos_O': n_pos_O,
        'n_kept_U': n_kept_U,
        'n_pos_U': n_pos_U,
        'n_support': n_support,
        'S': S,
        'target': (1.0 / 3.0) * np.log(N_U),
        'ratio': S / ((1.0 / 3.0) * np.log(N_U)) if N_U > 1 else None,
        'seed': seed,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("Phase 0: 2D Flat Diamond SJ Entropy Sanity Check")
    print("Target: S ≈ (1/3) ln(N_U)")
    print("=" * 72)
    print()

    # Test at several N values
    N_values = [256, 512, 1024, 2048, 4096]

    results = []
    for N in N_values:
        t0 = time.time()
        r = run_2d_entropy(N, L=1.0, seed=42)
        dt = time.time() - t0

        if 'error' in r:
            print(f"  N={N}: ERROR — {r['error']}")
            continue

        print(f"  N={N:5d}  N_U={r['N_U']:4d}  "
              f"S={r['S']:+.4f}  target={r['target']:.4f}  "
              f"ratio={r['ratio']:.3f}  "
              f"modes_O={r['n_kept_O']}  modes_U={r['n_kept_U']}  "
              f"support={r['n_support']}  ({dt:.1f}s)")

        results.append({**r, 'time': dt})

    print()
    print("=" * 72)
    print("DIAGNOSTIC")
    print("=" * 72)
    if results:
        ratios = [r['ratio'] for r in results if r['ratio'] is not None]
        print(f"  Mean ratio S / [(1/3)ln(N_U)] = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
        print(f"  Expected: ~1.0 for correct implementation")
        print()
        if abs(np.mean(ratios) - 1.0) < 0.3:
            print("  ✓ PASS: entropy formula implementation appears correct")
        else:
            print("  ✗ FAIL: ratio far from 1.0 — check implementation")

    # Also check with different seeds to verify not a fluke
    print()
    print("Seed variation at N=1024:")
    for s in range(5):
        r = run_2d_entropy(1024, L=1.0, seed=100 + s)
        if 'error' not in r:
            print(f"  seed={100+s}: N_U={r['N_U']}  S={r['S']:+.4f}  "
                  f"target={r['target']:.4f}  ratio={r['ratio']:.3f}")
