#!/usr/bin/env python3
"""
FND-1: SJ Vacuum N=5000 — GPU via CuPy (WSL2).

Self-contained: no imports from Windows-only modules.
Run from Windows:
  wsl.exe -d Ubuntu bash -c '. ~/.sct_env && source ~/sct-wsl/bin/activate && cd /mnt/f/Black\ Mesa\ Research\ Facility/Main\ Facility/Physics\ department/SCT\ Theory && python3 analysis/scripts/fnd1_sj_gpu_wsl.py'
"""

import time, json
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 5000
M = 40
T = 1.0
SEED = 555
EPS_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]

# ---------------------------------------------------------------------------
# 4D Causal Diamond
# ---------------------------------------------------------------------------
def sprinkle_4d(N, T, rng):
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 500) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        inside = np.abs(c[:, 0]) + r < half
        valid = c[inside]
        n_take = min(len(valid), N - count)
        pts[count:count + n_take] = valid[:n_take]
        count += n_take
    return pts[np.argsort(pts[:, 0])]

def causal_matrix_4d_ppwave(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[None, :] - t[:, None]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dz = z[None, :] - z[:, None]
    dr2 = dx**2 + dy**2 + dz**2
    mink = dt**2 - dr2
    if abs(eps) > 1e-12:
        xm = (x[None, :] + x[:, None]) / 2
        ym = (y[None, :] + y[:, None]) / 2
        f = np.cos(np.pi * xm) * np.cosh(np.pi * ym)
        corr = eps * f * (dt + dz)**2 / 2
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
    else:
        C = ((mink > 0) & (dt > 0)).astype(np.float64)
    return C

def compute_layers(C):
    import scipy.sparse as sp
    C_sp = sp.csr_matrix(C)
    n_mat = (C_sp @ C_sp).toarray()
    past = C.T
    n_past = n_mat.T
    N0 = int(np.sum((past > 0) & (n_past == 0)))
    N1 = int(np.sum((past > 0) & (n_past == 1)))
    N2 = int(np.sum((past > 0) & (n_past == 2)))
    N3 = int(np.sum((past > 0) & (n_past == 3)))
    return n_mat, N0, N1, N2, N3

# ---------------------------------------------------------------------------
# SJ Construction (GPU eigendecomposition)
# ---------------------------------------------------------------------------
def build_sj(C, n_matrix, rho):
    a = np.sqrt(rho) / (2.0 * np.pi * np.sqrt(6.0))
    past = C.T
    n_past = n_matrix.T
    link_lower = ((past > 0) & (n_past == 0)).astype(np.float64)
    L_upper = link_lower.T
    Delta = a * (L_upper - L_upper.T)
    iDelta = 1j * Delta

    # GPU eigendecomposition
    import cupy as cp
    iD_gpu = cp.asarray(iDelta)
    evals_gpu = cp.linalg.eigvalsh(iD_gpu)
    evals = cp.asnumpy(evals_gpu)
    del iD_gpu, evals_gpu
    cp.get_default_memory_pool().free_all_blocks()

    pos = evals[evals > 1e-15]
    return pos, evals

def sj_obs(pos_evals, all_evals, N_pts):
    if len(pos_evals) == 0:
        return {k: 0.0 for k in ["trace_W", "n_modes", "spectral_gap_ratio",
                "entropy_spectral", "lambda_max", "lambda_median", "spectral_width",
                "trace_trunc", "entropy_trunc", "ssee_proxy"]}
    sp = np.sort(pos_evals)[::-1]
    trace_W = float(np.sum(pos_evals))
    n_modes = len(pos_evals)
    lmax = float(sp[0])
    lmed = float(np.median(sp))
    gap_ratio = float(sp[0] / sp[1]) if len(sp) >= 2 and sp[1] > 1e-20 else float('inf')
    p = pos_evals / trace_W
    entropy = float(-np.sum(p * np.log(p + 1e-300)))
    le = np.log(pos_evals[pos_evals > 1e-20])
    sw = float(np.std(le)) if len(le) > 1 else 0.0
    n_max = max(1, int(N_pts**0.75))
    tr = sp[:n_max]
    trace_trunc = float(np.sum(tr))
    pt = tr / trace_trunc if trace_trunc > 0 else np.ones_like(tr) / len(tr)
    entropy_trunc = float(-np.sum(pt * np.log(pt + 1e-300)))
    ssee = float(np.sum(np.where(tr > 1e-20, (tr+1)*np.log(tr+1) - tr*np.log(tr+1e-300), 0)))
    return {"trace_W": trace_W, "n_modes": n_modes, "spectral_gap_ratio": gap_ratio,
            "entropy_spectral": entropy, "lambda_max": lmax, "lambda_median": lmed,
            "spectral_width": sw, "trace_trunc": trace_trunc,
            "entropy_trunc": entropy_trunc, "ssee_proxy": ssee}

# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------
def mediation_poly(eps_a, obs_a, tc_a, nl_a, bd_a):
    n = len(eps_a)
    tc2, tc3 = tc_a**2, tc_a**3
    def pr(x, y, ctrls):
        X = np.column_stack([*ctrls, np.ones(n)])
        try:
            bx = np.linalg.lstsq(X, x, rcond=None)[0]
            by = np.linalg.lstsq(X, y, rcond=None)[0]
        except: return 0.0, 1.0
        xr, yr = x - X @ bx, y - X @ by
        if np.std(xr) > 1e-15 and np.std(yr) > 1e-15:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0
    r_dir = stats.pearsonr(eps_a, obs_a)[0] if np.std(obs_a) > 1e-15 else 0
    r_tc = pr(eps_a, obs_a, [tc_a])[0]
    r_full = pr(eps_a, obs_a, [tc_a, nl_a, bd_a])[0]
    r_poly3, p_poly3 = pr(eps_a, obs_a, [tc_a, tc2, tc3, nl_a, bd_a])
    return {"r_direct": float(r_dir), "r_tc": float(r_tc), "r_full": float(r_full),
            "r_poly3": float(r_poly3), "p_poly3": float(p_poly3)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(0)
    mem = cp.cuda.runtime.memGetInfo()
    print(f"FND-1 SJ N={N} GPU: {dev['name'].decode()}, "
          f"Free={mem[0]/1e9:.1f}GB", flush=True)

    # Benchmark
    rng = np.random.default_rng(42)
    pts = sprinkle_4d(N, T, rng)
    C = causal_matrix_4d_ppwave(pts, 0.0)
    n_mat, *_ = compute_layers(C)
    V = np.pi * T**4 / 24.0
    rho = N / V
    tb = time.perf_counter()
    build_sj(C, n_mat, rho)
    print(f"Benchmark: {time.perf_counter()-tb:.2f}s/eigendecomp", flush=True)

    ss = np.random.SeedSequence(SEED)
    data = []

    for eps in EPS_VALUES:
        seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(1)[0].spawn(M)]
        print(f"\n  eps={eps:+.2f}:", end=" ", flush=True)
        t1 = time.perf_counter()
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(s)
            pts = sprinkle_4d(N, T, rng)
            C = causal_matrix_4d_ppwave(pts, eps)
            tc = float(np.sum(C))
            n_mat, N0, N1, N2, N3 = compute_layers(C)
            bd = (-4*N + 4*N0 - 36*N1 + 64*N2 - 32*N3) / np.sqrt(6.0)
            pos, all_ev = build_sj(C, n_mat, rho)
            obs = sj_obs(pos, all_ev, N)
            import scipy.sparse as sp2
            past = C.T; n_past = n_mat.T
            lm = ((past > 0) & (n_past == 0)).astype(np.float64)
            A = lm + lm.T; A = (A > 0).astype(np.float64)
            degs = np.sum(A, axis=1)
            deg_cv = float(np.std(degs)/np.mean(degs)) if np.mean(degs) > 0 else 0
            data.append({"eps": eps, "total_causal": tc, "n_links": N0,
                         "bd_action": float(bd), "deg_cv": deg_cv, **obs})
            if (i+1) % 10 == 0: print(f"{i+1}", end=" ", flush=True)
        d_eps = [d for d in data if abs(d["eps"]-eps)<0.01]
        print(f" ({time.perf_counter()-t1:.0f}s, TC={np.mean([d['total_causal'] for d in d_eps]):.0f})",
              flush=True)

    # Analysis
    eps_a = np.array([d["eps"] for d in data])
    tc_a = np.array([d["total_causal"] for d in data])
    nl_a = np.array([d["n_links"] for d in data])
    bd_a = np.array([d["bd_action"] for d in data])

    metrics = ["trace_W", "spectral_width", "trace_trunc", "lambda_median",
               "spectral_gap_ratio", "entropy_trunc", "ssee_proxy"]

    print(f"\n{'='*70}\nMEDIATION (N={N})\n{'='*70}", flush=True)
    print(f"  {'metric':>20} {'r_dir':>7} {'r|TC':>7} {'r|poly3':>7} {'p|poly3':>10}", flush=True)

    results = {}
    for metric in metrics:
        obs = np.array([d[metric] for d in data])
        if np.std(obs) < 1e-15: continue
        med = mediation_poly(eps_a, obs, tc_a, nl_a, bd_a)
        results[metric] = med
        print(f"  {metric:>20} {med['r_direct']:+7.3f} {med['r_tc']:+7.3f} "
              f"{med['r_poly3']:+7.3f} {med['p_poly3']:10.2e}", flush=True)

    print(f"\n{'='*70}\nSCALING: trace_W r|poly3\n{'='*70}", flush=True)
    print(f"  N=1000: -0.054\n  N=2000: -0.285\n  N=3000: -0.405", flush=True)
    tw = results.get("trace_W", {})
    print(f"  N=5000: {tw.get('r_poly3',0):+.3f}  <-- THIS", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\n  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    out = {"N": N, "M": M, "results": results, "wall_time": total_time}
    with open("speculative/numerics/ensemble_results/sj_n5000.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved. Done.", flush=True)

if __name__ == "__main__":
    main()
