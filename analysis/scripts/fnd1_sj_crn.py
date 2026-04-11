#!/usr/bin/env python3
"""
FND-1: SJ Vacuum CRN (Common Random Numbers) Test.

The CLEANEST possible test: same points, different metric.
No mediation, no poly3, no regression. Just paired differences.

For each sprinkling:
  1. Generate N points (flat Minkowski diamond)
  2. Compute C_flat (Minkowski causal condition)
  3. Compute C_ppwave (pp-wave causal condition on SAME points)
  4. Build SJ vacuum for both: W_flat, W_ppwave
  5. Paired difference: Delta_obs = obs(ppwave) - obs(flat)
  6. If Delta_obs != 0 systematically -> genuine causal structure effect

Density is IDENTICAL (same points). Any difference is PURELY from
lightcone change (the pp-wave tilts lightcones).

Run via WSL (GPU):
  wsl.exe -d Ubuntu bash -c '. ~/.sct_env && source ~/sct-wsl/bin/activate && \
  cd "/mnt/f/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory" && \
  python3 analysis/scripts/fnd1_sj_crn.py'
"""

import time, json
import numpy as np
from scipy import stats

N = 5000
M = 40  # paired sprinklings
T = 1.0
SEED = 777
EPS_VALUES = [0.1, 0.2, 0.3, 0.5]  # no eps=0 needed (flat is the reference)


def sprinkle_4d(N, T, rng):
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 500) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        valid = c[np.abs(c[:, 0]) + r < half]
        n_take = min(len(valid), N - count)
        pts[count:count + n_take] = valid[:n_take]
        count += n_take
    return pts[np.argsort(pts[:, 0])]


def causal_matrix(pts, eps):
    """Build causal matrix. eps=0 -> Minkowski, eps>0 -> pp-wave."""
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
        return ((mink > corr) & (dt > 0)).astype(np.float64)
    return ((mink > 0) & (dt > 0)).astype(np.float64)


def sj_eigenvalues(C, rho):
    """Compute positive eigenvalues of the SJ Wightman function."""
    import scipy.sparse as sp
    C_sp = sp.csr_matrix(C)
    n_mat = (C_sp @ C_sp).toarray()
    past = C.T
    n_past = n_mat.T
    link_lower = ((past > 0) & (n_past == 0)).astype(np.float64)
    L = link_lower.T  # upper-tri links

    a = np.sqrt(rho) / (2.0 * np.pi * np.sqrt(6.0))
    Delta = a * (L - L.T)
    iDelta = 1j * Delta

    import cupy as cp
    evals = cp.asnumpy(cp.linalg.eigvalsh(cp.asarray(iDelta)))
    cp.get_default_memory_pool().free_all_blocks()

    pos = evals[evals > 1e-15]
    return pos


def extract_obs(pos_evals, N_pts):
    """Extract SJ observables from positive eigenvalues."""
    if len(pos_evals) == 0:
        return {}
    sp = np.sort(pos_evals)[::-1]
    trace = float(np.sum(sp))
    n_modes = len(sp)
    gap_ratio = float(sp[0] / sp[1]) if len(sp) >= 2 and sp[1] > 1e-20 else 0
    p = sp / trace
    entropy = float(-np.sum(p * np.log(p + 1e-300)))
    le = np.log(sp[sp > 1e-20])
    width = float(np.std(le)) if len(le) > 1 else 0
    n_max = max(1, int(N_pts**0.75))
    tr = sp[:n_max]
    trace_trunc = float(np.sum(tr))
    lmed = float(np.median(sp))
    return {"trace_W": trace, "n_modes": n_modes, "spectral_gap_ratio": gap_ratio,
            "entropy_spectral": entropy, "spectral_width": width,
            "trace_trunc": trace_trunc, "lambda_median": lmed}


def main():
    t0 = time.perf_counter()
    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"FND-1 SJ CRN TEST: N={N}, M={M}, GPU={dev['name'].decode()}", flush=True)
    print(f"Same points, different metric. No mediation needed.", flush=True)

    V = np.pi * T**4 / 24.0
    rho = N / V

    # Benchmark
    rng = np.random.default_rng(42)
    pts = sprinkle_4d(N, T, rng)
    tb = time.perf_counter()
    C0 = causal_matrix(pts, 0.0)
    C1 = causal_matrix(pts, 0.5)
    ev0 = sj_eigenvalues(C0, rho)
    ev1 = sj_eigenvalues(C1, rho)
    bench = time.perf_counter() - tb
    tc0, tc1 = np.sum(C0), np.sum(C1)
    n_changed = int(np.sum(np.abs(C1 - C0)))
    print(f"Benchmark: {bench:.1f}s/pair, TC_flat={tc0:.0f}, TC_ppwave={tc1:.0f}, "
          f"pairs_changed={n_changed} ({100*n_changed/max(tc0,1):.1f}%)", flush=True)
    total_est = M * len(EPS_VALUES) * bench
    print(f"Est total: {total_est/60:.0f} min", flush=True)

    ss = np.random.SeedSequence(SEED)
    all_results = {}

    for eps in EPS_VALUES:
        seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(1)[0].spawn(M)]
        print(f"\n  eps={eps:+.2f}:", end=" ", flush=True)
        t1 = time.perf_counter()

        deltas = {k: [] for k in ["trace_W", "n_modes", "spectral_gap_ratio",
                                    "entropy_spectral", "spectral_width",
                                    "trace_trunc", "lambda_median", "delta_TC"]}

        for i, s in enumerate(seeds):
            rng = np.random.default_rng(s)
            pts = sprinkle_4d(N, T, rng)

            C_flat = causal_matrix(pts, 0.0)
            C_curved = causal_matrix(pts, eps)

            ev_flat = sj_eigenvalues(C_flat, rho)
            ev_curved = sj_eigenvalues(C_curved, rho)

            obs_flat = extract_obs(ev_flat, N)
            obs_curved = extract_obs(ev_curved, N)

            for k in deltas:
                if k == "delta_TC":
                    deltas[k].append(float(np.sum(C_curved) - np.sum(C_flat)))
                elif k in obs_flat and k in obs_curved:
                    deltas[k].append(obs_curved[k] - obs_flat[k])

            if (i + 1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)

        elapsed = time.perf_counter() - t1
        print(f" ({elapsed:.0f}s)", flush=True)

        # Paired t-test for each observable: is Delta != 0?
        print(f"    {'observable':>20} {'mean_delta':>12} {'SEM':>10} {'t':>8} {'p':>10} {'sig':>4}",
              flush=True)

        eps_result = {}
        for k, vals in deltas.items():
            arr = np.array(vals)
            mean_d = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            if sem > 1e-20:
                t_stat, p_val = stats.ttest_1samp(arr, 0.0)
            else:
                t_stat, p_val = 0.0, 1.0
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            print(f"    {k:>20} {mean_d:+12.4f} {sem:10.4f} {t_stat:+8.2f} {p_val:10.2e} {sig:>4}",
                  flush=True)
            eps_result[k] = {"mean": mean_d, "sem": sem, "t": float(t_stat), "p": float(p_val)}

        all_results[str(eps)] = eps_result

    # Summary: which observables show CONSISTENT significant Delta across all eps?
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY: CONSISTENT SIGNAL ACROSS ALL EPS?", flush=True)
    print(f"{'='*70}", flush=True)

    obs_names = ["trace_W", "n_modes", "spectral_gap_ratio", "entropy_spectral",
                 "spectral_width", "trace_trunc", "lambda_median"]

    print(f"\n  {'observable':>20}", end="", flush=True)
    for eps in EPS_VALUES:
        print(f" {'eps='+str(eps):>12}", end="")
    print(f" {'consistent':>10}", flush=True)

    for obs in obs_names:
        print(f"  {obs:>20}", end="", flush=True)
        signs = []
        all_sig = True
        for eps in EPS_VALUES:
            r = all_results[str(eps)].get(obs, {})
            mean_d = r.get("mean", 0)
            p = r.get("p", 1)
            sig = p < 0.05
            sign = "+" if mean_d > 0 else "-"
            marker = f"{sign}{'*' if sig else ' '}"
            print(f" {mean_d:+11.3f}{'' if sig else '(ns)'}", end="")
            signs.append(np.sign(mean_d))
            if not sig:
                all_sig = False
        consistent = all_sig and len(set(s for s in signs if s != 0)) == 1
        print(f" {'YES' if consistent else 'no':>10}", flush=True)

    # The CRITICAL number: delta_TC
    print(f"\n  Causal pair changes (delta_TC):", flush=True)
    for eps in EPS_VALUES:
        r = all_results[str(eps)].get("delta_TC", {})
        print(f"    eps={eps}: delta_TC = {r.get('mean',0):+.0f} "
              f"({100*r.get('mean',0)/1258934:.1f}% of flat TC)", flush=True)

    # VERDICT
    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'='*70}", flush=True)

    # Count observables with consistent significant paired difference
    n_consistent = 0
    for obs in obs_names:
        signs = []
        all_sig = True
        for eps in EPS_VALUES:
            r = all_results[str(eps)].get(obs, {})
            if r.get("p", 1) >= 0.05:
                all_sig = False
            signs.append(np.sign(r.get("mean", 0)))
        if all_sig and len(set(s for s in signs if s != 0)) == 1:
            n_consistent += 1

    if n_consistent >= 3:
        verdict = f"CRN SIGNAL: {n_consistent}/7 observables show consistent paired difference"
    elif n_consistent >= 1:
        verdict = f"WEAK CRN SIGNAL: {n_consistent}/7 consistent"
    else:
        verdict = "NO CRN SIGNAL: pp-wave does not change SJ spectrum beyond TC"

    print(f"\n  {verdict}", flush=True)
    total_time = time.perf_counter() - t0
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    with open("speculative/numerics/ensemble_results/sj_crn.json", "w") as f:
        json.dump({"N": N, "M": M, "results": all_results, "verdict": verdict,
                   "wall_time": total_time}, f, indent=2)
    print("Saved. Done.", flush=True)


if __name__ == "__main__":
    main()
