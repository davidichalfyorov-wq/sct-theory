"""S4: W-proportionality test for path_kurtosis on pp-wave.
If delta(path_kurtosis)/eps^2 = const -> proportional to Bel-Robinson W.
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import time, gc, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sct_tools.gpu import xp, USE_GPU, gpu_info
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

def path_kurtosis_val(C, N):
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L = C_sp - C_sp.multiply(has_int)
    L.eliminate_zeros()
    L_csr = L.tocsr()
    L_csc = L.tocsc()

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
    return float(stats.kurtosis(log_p, fisher=True)) if len(log_p) > 10 else 0.0

def main():
    N = 2000
    M = 20

    print("=" * 60)
    print("S4: W-proportionality (path_kurtosis vs eps on pp-wave)")
    print(f"N={N}, M={M}")
    print("W = eps^2 (Bel-Robinson). If delta/W = const -> proportional to W")
    print("=" * 60)

    results = []
    for eps in [1.0, 2.0, 5.0, 10.0, 20.0]:
        deltas = []
        t0 = time.time()
        for trial in range(M):
            rng = np.random.default_rng(trial * 1000 + 100)
            pts = sprinkle_4d(N, T, rng)
            C_f = causal_flat(pts)
            pk_f = path_kurtosis_val(C_f, N)
            del C_f; gc.collect()
            C_p = causal_ppwave(pts, eps)
            pk_p = path_kurtosis_val(C_p, N)
            del C_p; gc.collect()
            deltas.append(pk_p - pk_f)

        m = np.mean(deltas)
        se = np.std(deltas) / np.sqrt(M)
        W = eps**2
        ratio = m / W if W > 0 else 0
        elapsed = time.time() - t0
        print(f"  eps={eps:5.1f}: delta={m:+.6f} +/- {se:.6f}, "
              f"W={W:6.1f}, delta/W={ratio:+.8f} [{elapsed:.1f}s]")
        results.append({"eps": eps, "delta": m, "W": W, "ratio": ratio})

    # Check constancy
    ratios = [r["ratio"] for r in results if abs(r["ratio"]) > 1e-10]
    if len(ratios) >= 3:
        cv = np.std(ratios) / abs(np.mean(ratios)) * 100
        print(f"\n  delta/W ratios: {[f'{r:.8f}' for r in ratios]}")
        print(f"  CV = {cv:.0f}%")
        if cv < 30:
            print(f"  -> PROPORTIONAL to W (CV < 30%)")
        elif cv < 50:
            print(f"  -> APPROXIMATELY proportional (30% < CV < 50%)")
        else:
            print(f"  -> NOT proportional (CV > 50%)")

        # Fit power law: delta ~ eps^beta
        eps_arr = np.array([r["eps"] for r in results])
        d_arr = np.abs(np.array([r["delta"] for r in results]))
        mask = d_arr > 1e-8
        if np.sum(mask) >= 3:
            slope, _ = np.polyfit(np.log(eps_arr[mask]), np.log(d_arr[mask]), 1)
            print(f"  Power law: |delta| ~ eps^{slope:.2f}")
            print(f"  If ~2: proportional to W=eps^2 (Bel-Robinson)")
            print(f"  If ~1: proportional to |Riem|~eps")

if __name__ == "__main__":
    main()
