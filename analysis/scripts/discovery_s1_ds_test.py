"""
SYNTHESIS TEST S1: path_kurtosis на de Sitter — Weyl-null test
===============================================================

ВОПРОС: Является ли path_kurtosis чисто Weyl-чувствительной?

dS: R ≠ 0, C² = 0 (conformally flat, Weyl = 0)
pp-wave: R = 0, C² ≠ 0, K = 0 (VSI, Weyl ≠ 0)

Если path_kurtosis(dS) = NULL → видит ТОЛЬКО Weyl
Если path_kurtosis(dS) ≠ NULL → видит и Ricci

РАСХОЖДЕНИЕ: Run #007 (N=5000): dS d=-2.54. Run #008 (N=2000): dS d=-0.02.
Этот тест разрешает расхождение при N=5000 с M=30.

PRE-REGISTERED:
  - |d| < 0.5 → NULL → path_kurtosis = Weyl-only
  - |d| > 1.0 → ≠ NULL → path_kurtosis видит Ricci тоже
  - Bonferroni alpha = 0.01/3 = 0.0033

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import time, gc, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sct_tools.gpu import xp, USE_GPU, gpu_info, to_cpu, free_gpu
gpu_info()

T = 1.0


def sprinkle_4d(N, T, rng):
    pts = np.empty((N, 4), dtype=np.float32)
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4)).astype(np.float32)
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count + n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


def causal_flat(pts):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    return ((dt**2 > dr2) & (dt > 0)).astype(np.float64)


def causal_desitter(pts, H):
    """de Sitter: ds² = -dt² + exp(2Ht)(dx² + dy² + dz²). Midpoint approx."""
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2.0
    a2 = np.exp(2.0 * H * tm)
    return ((dt**2 > a2 * dr2) & (dt > 0)).astype(np.float64)


def causal_ppwave(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm**2 - ym**2
    mink = dt**2 - dr2
    corr = eps * f * (dt + dz)**2 / 2.0
    return ((mink > corr) & (dt > 0)).astype(np.float64)


def causal_schwarzschild(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    zm = (z[np.newaxis, :] + z[:, np.newaxis]) / 2.0
    rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3
    Phi = -eps / rm
    return (((1 + 2 * Phi) * dt**2 > (1 - 2 * Phi) * dr2) & (dt > 0)).astype(np.float64)


def build_link_sparse(C):
    """Directed Hasse diagram as sparse CSR + CSC."""
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L = C_sp - C_sp.multiply(has_int)
    L.eliminate_zeros()
    return L.tocsr(), L.tocsc()


def path_kurtosis(pts, C, N):
    """Compute path_kurtosis from causal matrix C."""
    L_csr, L_csc = build_link_sparse(C)

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
    if len(log_p) < 10:
        return 0.0
    return float(stats.kurtosis(log_p, fisher=True))


def crn_trial(seed, N, T, metric_fn, eps, seed_offset=0):
    """CRN: flat vs curved, return path_kurtosis delta."""
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    C_flat = causal_flat(pts)
    pk_flat = path_kurtosis(pts, C_flat, N)
    del C_flat; gc.collect()

    C_curv = metric_fn(pts, eps)
    pk_curv = path_kurtosis(pts, C_curv, N)
    del C_curv; gc.collect()

    return pk_curv - pk_flat


def main():
    N = 5000
    M = 30

    TESTS = [
        ("ppwave eps=5", causal_ppwave, 5.0, 100),
        ("schwarzschild eps=0.005", causal_schwarzschild, 0.005, 300),
        ("deSitter H=0.5", causal_desitter, 0.5, 500),
    ]

    print("=" * 70)
    print("TEST S1: path_kurtosis на de Sitter — Weyl-null test")
    print(f"N={N}, M={M}")
    print("PRE-REGISTERED: dS |d| < 0.5 → NULL → Weyl-only")
    print("=" * 70)

    for label, metric_fn, eps, seed_off in TESTS:
        print(f"\n--- {label} ---")
        t0 = time.time()
        deltas = []

        for trial in range(M):
            d = crn_trial(trial * 1000, N, T, metric_fn, eps, seed_off)
            deltas.append(d)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                m = np.mean(deltas)
                print(f"  trial {trial + 1}/{M}: delta={m:+.4f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0
        deltas = np.array(deltas)
        m = np.mean(deltas)
        se = np.std(deltas) / np.sqrt(M)
        d_cohen = m / np.std(deltas) if np.std(deltas) > 0 else 0
        _, p = stats.ttest_1samp(deltas, 0.0)

        print(f"\n  RESULT: delta={m:+.4f} ± {se:.4f}, d={d_cohen:+.3f}, p={p:.2e}")

        if "deSitter" in label:
            if abs(d_cohen) < 0.5:
                print(f"  ✅ dS = NULL (|d|={abs(d_cohen):.2f} < 0.5)")
                print(f"  → path_kurtosis = WEYL-ONLY PROBE")
            elif abs(d_cohen) > 1.0 and p < 0.003:
                print(f"  ☠️ dS ≠ NULL (d={d_cohen:+.2f}, p={p:.2e})")
                print(f"  → path_kurtosis видит и Ricci")
            else:
                print(f"  ⚠️ AMBIGUOUS (|d|={abs(d_cohen):.2f})")

    print(f"\nВсего: {time.time() - time.time():.0f}с")


if __name__ == "__main__":
    main()
