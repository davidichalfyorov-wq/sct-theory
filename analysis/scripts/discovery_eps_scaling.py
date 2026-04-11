"""
Discovery Run 001 - Epsilon scaling test for Forman-Ricci on pp-wave.
Tests whether the degree-corrected residual scales with epsilon.
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import gc

N, T, M = 500, 1.0, 30

def sprinkle(N, T, rng):
    pts = np.empty((N, 4))
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count + n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]

def causal_flat(pts):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    return ((dt**2 > dr2) & (dt > 0)).astype(np.float64)

def causal_ppwave_quad(pts, eps):
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

def build_link_graph(C):
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_intervening = (C2 != 0).astype(np.float64)
    link = C_sp - C_sp.multiply(has_intervening)
    link.eliminate_zeros()
    A = link + link.T
    A = (A > 0.5).astype(np.float64)
    return A.tocsr()

def forman_basic(A_sp):
    degrees = np.array(A_sp.sum(axis=1)).ravel()
    rows, cols = sp.triu(A_sp, k=1).nonzero()
    F = 4.0 - degrees[rows] - degrees[cols]
    return float(np.mean(F)), float(np.mean(degrees))

print("pp-wave Forman-Ricci epsilon scaling (N=500, M=30)")
print(f"{'eps':>6s} {'F_delta':>10s} {'deg_delta':>10s} {'residual':>10s} {'resid_SE':>10s} {'p_resid':>10s} {'sigma':>8s}")
print("-" * 66)

for eps in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    d_F = []; d_deg = []
    for trial in range(M):
        seed = trial * 1000
        rng = np.random.default_rng(seed + 100)
        pts = sprinkle(N, T, rng)

        C_flat = causal_flat(pts)
        A_flat = build_link_graph(C_flat); del C_flat
        mf, md_f = forman_basic(A_flat)
        del A_flat; gc.collect()

        C_curv = causal_ppwave_quad(pts, eps)
        A_curv = build_link_graph(C_curv); del C_curv
        mc, md_c = forman_basic(A_curv)
        del A_curv; gc.collect()

        d_F.append(mc - mf)
        d_deg.append(md_c - md_f)

    resid = [d_F[i] - (-2*d_deg[i]) for i in range(M)]
    mean_r = np.mean(resid)
    se_r = np.std(resid) / np.sqrt(M)
    _, p_r = stats.ttest_1samp(resid, 0.0)
    sig = abs(mean_r / se_r) if se_r > 0 else 0

    print(f"{eps:6.1f} {np.mean(d_F):+10.3f} {np.mean(d_deg):+10.3f} "
          f"{mean_r:+10.4f} {se_r:10.4f} {p_r:10.2e} {sig:7.1f}s")

print("\nIf residual ~ eps^1: genuine linear curvature response")
print("If residual ~ eps^2: genuine quadratic curvature response")
print("If residual saturates: finite-size artifact")
