"""
a2 extraction for N=20000,50000 using SPARSE matmul (no GPU needed).
Sparse C@C is 5x faster than dense CPU at N=20000 and avoids GPU OOM.

Strategy:
- Build causal matrix in chunks (CPU, low memory)
- Store as scipy.sparse.csr_matrix (~5% density)
- C² = C_sparse @ C_sparse (scipy optimized, 5x faster)
- Extract C²[i,j] values for tau-binned pairs

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
import json, time, gc, os, sys

OUTDIR = "gcp_results"
T = 1.0
TAU_EDGES = np.linspace(0.03, 0.45, 26)


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


def build_causal_sparse(pts, N, metric, eps, chunk=2000):
    """Build causal matrix as sparse CSR, chunked to control memory."""
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
    rows, cols = [], []

    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        dt = t[np.newaxis, :] - t[i0:i1, np.newaxis]
        dx = x[np.newaxis, :] - x[i0:i1, np.newaxis]
        dy = y[np.newaxis, :] - y[i0:i1, np.newaxis]
        dz = z[np.newaxis, :] - z[i0:i1, np.newaxis]
        dr2 = dx**2 + dy**2 + dz**2

        if metric == "flat":
            mask = (dt**2 > dr2) & (dt > 0)
        elif metric == "ppwave":
            xm = (x[np.newaxis, :] + x[i0:i1, np.newaxis]) / 2.0
            ym = (y[np.newaxis, :] + y[i0:i1, np.newaxis]) / 2.0
            f = xm**2 - ym**2
            mink = dt**2 - dr2
            corr = eps * f * (dt + dz)**2 / 2.0
            mask = (mink > corr) & (dt > 0)
            del xm, ym, f, mink, corr
        elif metric == "synlapse":
            xm = (x[np.newaxis, :] + x[i0:i1, np.newaxis]) / 2.0
            ym = (y[np.newaxis, :] + y[i0:i1, np.newaxis]) / 2.0
            f = xm**2 - ym**2
            g_tt = 1.0 - eps * f / 2.0
            mask = (g_tt * dt**2 > dr2) & (dt > 0) & (g_tt > 0)
            del xm, ym, f, g_tt

        del dt, dx, dy, dz, dr2

        ri, ci = np.where(mask)
        rows.append(ri + i0)
        cols.append(ci)
        del mask

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)
    C = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return C


def run_trial(seed, N, eps):
    """One CRN trial: flat + ppwave + synlapse, sparse matmul."""
    rng = np.random.default_rng(seed + 100)
    pts = sprinkle_4d(N, T, rng)
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]

    # Build sparse causal matrices
    t0b = time.time()
    C_flat = build_causal_sparse(pts, N, "flat", eps)
    C_ppw = build_causal_sparse(pts, N, "ppwave", eps)
    C_syn = build_causal_sparse(pts, N, "synlapse", eps)
    t_build = time.time() - t0b

    # Sparse matmul: C² = C @ C
    t0m = time.time()
    C2_flat = C_flat @ C_flat
    C2_ppw = C_ppw @ C_ppw
    C2_syn = C_syn @ C_syn
    t_matmul = time.time() - t0m

    # Tau-binning (chunked to control memory)
    n_bins = len(TAU_EDGES) - 1
    sum_vf = np.zeros(n_bins)
    sum_vp = np.zeros(n_bins)
    sum_vs = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=np.int64)

    chunk = min(N, max(500, int(2e9 / (N * 4 * 6))))
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)

        dt_c = t[np.newaxis, :] - t[i0:i1, np.newaxis]
        dx_c = x[np.newaxis, :] - x[i0:i1, np.newaxis]
        dy_c = y[np.newaxis, :] - y[i0:i1, np.newaxis]
        dz_c = z[np.newaxis, :] - z[i0:i1, np.newaxis]
        tau2 = dt_c**2 - dx_c**2 - dy_c**2 - dz_c**2
        del dx_c, dy_c, dz_c, dt_c

        # Causal in both ppw and syn
        ppw_chunk = C_ppw[i0:i1].toarray()
        syn_chunk = C_syn[i0:i1].toarray()
        causal = (ppw_chunk > 0.5) & (syn_chunk > 0.5) & (tau2 > 0)
        tau_vals = np.sqrt(np.maximum(tau2, 0))
        del tau2, ppw_chunk, syn_chunk

        # C² values (convert sparse slices to dense for indexing)
        c2f = C2_flat[i0:i1].toarray().astype(np.float32)
        c2p = C2_ppw[i0:i1].toarray().astype(np.float32)
        c2s = C2_syn[i0:i1].toarray().astype(np.float32)

        for b in range(n_bins):
            m = causal & (tau_vals >= TAU_EDGES[b]) & (tau_vals < TAU_EDGES[b + 1])
            c = np.sum(m)
            if c > 0:
                sum_vf[b] += np.sum(c2f[m])
                sum_vp[b] += np.sum(c2p[m])
                sum_vs[b] += np.sum(c2s[m])
                cnt[b] += c

        del causal, tau_vals, c2f, c2p, c2s

    del C_flat, C_ppw, C_syn, C2_flat, C2_ppw, C2_syn
    gc.collect()

    # Profile
    tau_mid, dr_prof, vf_prof, np_list = [], [], [], []
    for b in range(n_bins):
        if cnt[b] < 50:
            continue
        vf = sum_vf[b] / cnt[b]
        if vf < 1.0:
            continue
        vp = sum_vp[b] / cnt[b]
        vs = sum_vs[b] / cnt[b]
        dr_prof.append((vp - vs) / vf)
        tau_mid.append((TAU_EDGES[b] + TAU_EDGES[b + 1]) / 2)
        vf_prof.append(vf)
        np_list.append(int(cnt[b]))

    return {
        "tau_mid": tau_mid, "dr_profile": dr_prof,
        "vf_profile": vf_prof, "n_pairs": np_list,
        "t_build": t_build, "t_matmul": t_matmul,
    }


def fit_a2(tau_mid, dr_profile, W, tau_min=0.25):
    """Fit delta_V/V = A * tau^4. W = Bel-Robinson superenergy = eps^2."""
    mask = np.array(tau_mid) >= tau_min
    if np.sum(mask) < 3:
        return {"a2_W": None, "r2": None, "n_points": 0}
    tf = np.array(tau_mid)[mask]
    df = np.array(dr_profile)[mask]
    try:
        popt, _ = curve_fit(lambda t, A: A * t**4, tf, df, p0=[1.0])
        A = popt[0]
        pred = A * tf**4
        ss_r = np.sum((df - pred)**2)
        ss_t = np.sum((df - np.mean(df))**2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        a2_W = A / W if W > 0 else 0

        popt2, _ = curve_fit(lambda t, A, n: A * t**n, tf, df, p0=[1.0, 4.0], maxfev=5000)
        n_free = popt2[1]

        return {"a2_W": float(a2_W), "A": float(A), "r2": float(r2),
                "n_free": float(n_free), "n_points": int(np.sum(mask))}
    except Exception as e:
        return {"a2_W": None, "r2": None, "error": str(e)}


def main():
    path = os.path.join(OUTDIR, "a2_extraction.json")
    data = json.load(open(path))

    CONFIGS = [
        ("N20000_eps0.5", 20000, 0.5, 15),
        ("N20000_eps1.0", 20000, 1.0, 15),
        ("N50000_eps0.5", 50000, 0.5, 8),
        ("N50000_eps1.0", 50000, 1.0, 8),
    ]

    print("=" * 60)
    print("a2 EXTRACTION — SPARSE MATMUL (no GPU needed)")
    print("=" * 60)

    for name, N, eps, M in CONFIGS:
        if name in data and "fit_025" in data[name]:
            print(f"\nSKIP {name} (already done)")
            continue

        W = eps**2  # Bel-Robinson superenergy
        print(f"\n{'=' * 50}")
        print(f"  {name} | N={N} eps={eps} M={M} | W={W}")
        print(f"{'=' * 50}")

        t0 = time.time()
        trials = []
        for m in range(M):
            tt = time.time()
            trial = run_trial(m * 1000, N, eps)
            el = time.time() - tt
            trials.append(trial)
            rem = (M - m - 1) * el
            print(f"  {m + 1}/{M}: {el:.1f}s "
                  f"(build={trial['t_build']:.1f}s matmul={trial['t_matmul']:.1f}s) "
                  f"rem: {rem / 60:.1f}m")
            sys.stdout.flush()

        elapsed = time.time() - t0

        valid = [t for t in trials if len(t["dr_profile"]) > 0]
        if not valid:
            data[name] = {"N": N, "eps": eps, "status": "no_data",
                          "elapsed_sec": elapsed}
            json.dump(data, open(path, "w"), indent=2, default=str)
            continue

        ml = min(len(t["dr_profile"]) for t in valid)
        da = np.array([t["dr_profile"][:ml] for t in valid])
        tm = np.array(valid[0]["tau_mid"][:ml])
        dm = np.mean(da, axis=0)
        ds = np.std(da, axis=0) / np.sqrt(len(da))

        # Print profile
        print(f"\n  Profile ({len(valid)} trials):")
        for i, t_val in enumerate(tm):
            print(f"  {t_val:8.4f} {dm[i]:+14.8f} {ds[i]:12.8f}")

        f025 = fit_a2(tm, dm, W, 0.25)
        f015 = fit_a2(tm, dm, W, 0.15)

        data[name] = {
            "N": N, "eps": eps, "M": len(valid),
            "tau_mid": tm.tolist(), "dr_mean": dm.tolist(), "dr_se": ds.tolist(),
            "fit_025": f025, "fit_015": f015, "elapsed_sec": elapsed,
        }
        json.dump(data, open(path, "w"), indent=2, default=str)

        print(f"\n  >> a2_W={f025.get('a2_W', '?')}, R²={f025.get('r2', '?')}, "
              f"n_free={f025.get('n_free', '?')}, {elapsed / 60:.1f}min")
        print(f"  >> Target: a2_W = 0.002275")

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'config':20s} {'a2_W':>10s} {'R2':>8s} {'n_free':>7s}")
    print("-" * 50)
    for k in sorted(data.keys()):
        r = data[k]
        if "fit_025" in r:
            f = r["fit_025"]
            print(f"{k:20s} {str(f.get('a2_W', '?')):>10s} "
                  f"{str(f.get('r2', '?')):>8s} {str(f.get('n_free', '?')):>7s}")
    print(f"\nTarget: a2_W = 0.002275")
    print("DONE.")


if __name__ == "__main__":
    main()
