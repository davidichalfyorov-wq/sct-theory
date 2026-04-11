"""
FND-1 EXP-7: Dang-Wrochna a_0 Normalization Test.

Tests the single new prediction from the DW Lorentzian spectral zeta
extraction (FND1_ROUTE3_DW_EXTRACTION.md):

  tau * K_corr(tau) -> 1/(2*pi) ~ 0.15915  (plateau at intermediate tau)

where K_corr(tau) = K(tau) - n_zero/N is the zero-mode-corrected
normalized heat trace of H = (L+L^T)/2.

If FAIL -> the DW recipe for curvature extraction is invalidated.
If PASS -> first quantitative confirmation that H connects to DW zeta.
If BD_ABOVE_GOE -> H carries geometric content even if DW target not met.

Includes:
  - Zero-mode correction (K_corr = K - n_zero/N)
  - Derivative-based plateau detection (not crossing)
  - GOE null model (random matrix with matched spectral range)
  - Confidence intervals on plateau value
  - Spectral zeta residue extraction

Finite-size scaling: N = 200, 500, 1000, 2000, 5000.
Flat d=2 Minkowski only (no curvature parameter).

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_exp7_dw_a0.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    sprinkle_diamond,
    compute_interval_cardinalities,
    build_bd_L,
    ZERO_THRESHOLD,
)
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [200, 500, 1000, 2000, 5000]
M_ENSEMBLE = 100
T_DIAMOND = 1.0
MASTER_SEED = 314
WORKERS = N_WORKERS

# Heat trace tau grid — extended to 1000 to capture full behavior
N_TAU = 300
TAU_MIN = 1e-3
TAU_MAX = 1000.0
TAU_GRID = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)

# Spectral zeta
N_S = 80
S_MIN = 0.2
S_MAX = 3.0
S_GRID = np.linspace(S_MIN, S_MAX, N_S)
MU_VALUES = [0.01, 0.05, 0.1, 0.5]

# DW prediction
A0_TARGET = 1.0 / (2.0 * np.pi)  # 0.159155...

# Plateau detection: d(tK)/d(log tau) threshold
PLATEAU_DERIV_THRESHOLD = 0.02  # |d(tK)/d(log tau)| < this counts as flat
PLATEAU_MIN_DECADES = 0.3       # minimum width to call it a plateau


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute heat trace (BD + GOE null) and spectral zeta."""
    seed_int, N, T = args
    V = T ** 2 / 2.0
    rho = N / V

    rng = np.random.default_rng(seed_int)
    pts, C = sprinkle_diamond(N, T, rng)
    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)

    # --- BD operator ---
    H = (L + L.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(H)
    sigma = np.abs(eigenvalues) / rho

    n_zero = int(np.sum(np.abs(eigenvalues) < ZERO_THRESHOLD))
    sigma_max = float(np.max(sigma))
    sigma_nz = sigma[sigma > ZERO_THRESHOLD / rho]
    sigma_min_nz = float(np.min(sigma_nz)) if len(sigma_nz) > 0 else 0.0

    # BD heat trace + zero-mode correction
    K_bd = np.mean(np.exp(-TAU_GRID[:, None] * sigma[None, :]), axis=1)
    K_corr = K_bd - n_zero / N  # subtract asymptotic floor

    # --- GOE null model: random symmetric, matched spectral range ---
    rng_null = np.random.default_rng(seed_int + 77777)
    G = rng_null.standard_normal((N, N))
    G = (G + G.T) / 2.0
    eigs_goe = np.linalg.eigvalsh(G)
    # Rescale to match BD sigma_max
    goe_max = np.max(np.abs(eigs_goe))
    if goe_max > 0:
        eigs_goe = eigs_goe * (sigma_max * rho / goe_max)
    sigma_goe = np.abs(eigs_goe) / rho
    n_zero_goe = int(np.sum(np.abs(eigs_goe) < ZERO_THRESHOLD))
    K_goe = np.mean(np.exp(-TAU_GRID[:, None] * sigma_goe[None, :]), axis=1)
    K_goe_corr = K_goe - n_zero_goe / N

    # --- Spectral zeta ---
    zeta_by_mu = {}
    for mu in MU_VALUES:
        shifted = sigma + mu
        zeta = np.mean(shifted[None, :] ** (-S_GRID[:, None]), axis=1)
        zeta_by_mu[str(mu)] = zeta.tolist()

    # --- Eigenvalue density near zero ---
    delta = 0.5
    f_near_zero = float(np.sum(sigma < delta) / N / delta)
    f_near_zero_goe = float(np.sum(sigma_goe < delta) / N / delta)

    return {
        "K_bd": K_bd.tolist(),
        "K_corr": K_corr.tolist(),
        "K_goe": K_goe.tolist(),
        "K_goe_corr": K_goe_corr.tolist(),
        "zeta_by_mu": zeta_by_mu,
        "n_zero": n_zero,
        "n_zero_goe": n_zero_goe,
        "sigma_max": sigma_max,
        "sigma_min_nz": sigma_min_nz,
        "f_near_zero": f_near_zero,
        "f_near_zero_goe": f_near_zero_goe,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def find_plateau(tau_grid, tK_mean, tK_sem):
    """Derivative-based plateau detection.

    A plateau is a contiguous region where |d(tK)/d(log tau)| < threshold
    spanning at least PLATEAU_MIN_DECADES in log-tau.

    Returns dict with plateau properties, or None if no plateau found.
    """
    log_tau = np.log10(tau_grid)
    # Numerical derivative d(tK)/d(log tau)
    d_tK = np.gradient(tK_mean, log_tau)

    # Find contiguous regions where |derivative| < threshold
    # AND tK is in a physically meaningful range (not just UV zero region)
    is_flat = (np.abs(d_tK) < PLATEAU_DERIV_THRESHOLD) & (tK_mean > 0.01)
    if not np.any(is_flat):
        return None

    # Find contiguous runs
    changes = np.diff(is_flat.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    # Handle edge cases
    if is_flat[0]:
        starts = np.concatenate([[0], starts])
    if is_flat[-1]:
        ends = np.concatenate([ends, [len(is_flat)]])
    if len(starts) == 0 or len(ends) == 0:
        return None

    # Find the widest plateau
    best_width = 0
    best_start = best_end = 0
    for s, e in zip(starts, ends):
        width = log_tau[min(e, len(log_tau) - 1)] - log_tau[s]
        if width > best_width:
            best_width = width
            best_start, best_end = s, e

    if best_width < PLATEAU_MIN_DECADES:
        return None

    # Plateau statistics
    sl = slice(best_start, best_end)
    plateau_tK = tK_mean[sl]
    plateau_sem = tK_sem[sl]

    value = float(np.mean(plateau_tK))
    # Uncertainty: combine ensemble SEM with intra-plateau spread
    sem_combined = float(np.sqrt(np.mean(plateau_sem ** 2)
                                 + np.var(plateau_tK, ddof=1)))
    ci_lo = value - 2.0 * sem_combined
    ci_hi = value + 2.0 * sem_combined

    return {
        "value": value,
        "sem": sem_combined,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "tau_lo": float(tau_grid[best_start]),
        "tau_hi": float(tau_grid[min(best_end, len(tau_grid) - 1)]),
        "decades": best_width,
        "target_in_ci": bool(ci_lo <= A0_TARGET <= ci_hi),
        "rel_error": float((value - A0_TARGET) / A0_TARGET),
    }


def analyze_heat_trace(K_key, raw_results):
    """Analyze ensemble-averaged heat trace with plateau detection."""
    K_arr = np.array([r[K_key] for r in raw_results])  # (M, N_TAU)
    M = K_arr.shape[0]
    K_mean = np.mean(K_arr, axis=0)
    K_sem = np.std(K_arr, axis=0, ddof=1) / np.sqrt(M)

    tK_mean = TAU_GRID * K_mean
    tK_sem = TAU_GRID * K_sem

    # Plateau detection
    plateau = find_plateau(TAU_GRID, tK_mean, tK_sem)

    # Max of tK in the "intermediate" range (not UV, not IR zero-mode)
    # Intermediate: tau in [0.1, 100]
    intermed = (TAU_GRID > 0.1) & (TAU_GRID < 100)
    if np.any(intermed):
        tK_max_intermed = float(np.max(tK_mean[intermed]))
        tau_at_max = float(TAU_GRID[intermed][np.argmax(tK_mean[intermed])])
    else:
        tK_max_intermed = tau_at_max = float("nan")

    # Full ensemble-mean curve (50 log-spaced points for plotting)
    plot_idx = np.unique(np.linspace(0, N_TAU - 1, 50, dtype=int))
    tK_curve = {f"{TAU_GRID[i]:.6g}": round(float(tK_mean[i]), 8)
                for i in plot_idx}

    return {
        "plateau": plateau,
        "tK_curve": tK_curve,
        "tK_max_intermediate": tK_max_intermed,
        "tau_at_max_intermediate": tau_at_max,
    }


def analyze_spectral_zeta(zeta_ensemble, mu):
    """Analyze ensemble-averaged spectral zeta: extract residue at s=1."""
    zeta_arr = np.array(zeta_ensemble)
    zeta_mean = np.mean(zeta_arr, axis=0)

    fit_mask = (S_GRID > 0.4) & (S_GRID < 0.9)
    if np.sum(fit_mask) > 3:
        s_fit = S_GRID[fit_mask]
        sm1 = s_fit - 1.0
        sz = sm1 * zeta_mean[fit_mask]
        try:
            lr = stats.linregress(sm1, sz)
            residue = float(lr.intercept)
            residue_err = float((residue - A0_TARGET) / A0_TARGET)
        except Exception:
            residue = residue_err = float("nan")
    else:
        residue = residue_err = float("nan")

    return {"residue": residue, "residue_rel_error": residue_err}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=3, name="exp7_dw_a0",
        description="DW a0 normalization: tK_corr plateau vs 1/(2pi), with GOE null",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-7: DANG-WROCHNA a0 NORMALIZATION TEST", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}", flush=True)
    print(f"M = {M_ENSEMBLE}, workers = {WORKERS}", flush=True)
    print(f"Target: plateau of tau*K_corr(tau) = 1/(2pi) = {A0_TARGET:.6f}",
          flush=True)
    print(f"Includes: zero-mode correction, GOE null model, plateau detection",
          flush=True)
    print(flush=True)

    # Benchmark (3 seeds for variance estimate)
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        times = []
        for seed in [42, 100, 200]:
            t0 = time.perf_counter()
            _worker((seed, N, T_DIAMOND))
            times.append(time.perf_counter() - t0)
        mean_t = np.mean(times)
        par = M_ENSEMBLE * mean_t / WORKERS
        print(f"  N={N:5d}: {mean_t:.3f}s/task (x3), parallel({WORKERS}w): "
              f"{par:.1f}s = {par / 60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print("=" * 60, flush=True)

        n_ss = ss.spawn(1)[0]
        child_seeds = n_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N, T_DIAMOND) for si in seed_ints]

        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            raw = pool.map(_worker, args)
        elapsed = time.perf_counter() - t0
        print(f"  Computation: {elapsed:.1f}s ({elapsed / M_ENSEMBLE:.3f}s/task)",
              flush=True)

        # ---- BD heat trace (zero-mode corrected) ----
        ht_corr = analyze_heat_trace("K_corr", raw)
        p = ht_corr["plateau"]
        if p is not None:
            print(f"  BD CORRECTED plateau: {p['value']:.6f} +- {p['sem']:.6f}"
                  f"  (target {A0_TARGET:.6f}, err {p['rel_error']:+.2%})",
                  flush=True)
            print(f"    tau in [{p['tau_lo']:.3f}, {p['tau_hi']:.3f}]"
                  f" ({p['decades']:.2f} dec), target in 95% CI: {p['target_in_ci']}",
                  flush=True)
        else:
            print(f"  BD CORRECTED: NO PLATEAU detected (deriv threshold "
                  f"{PLATEAU_DERIV_THRESHOLD})", flush=True)
            print(f"    tK_max in [0.1, 100]: {ht_corr['tK_max_intermediate']:.6f}"
                  f" at tau={ht_corr['tau_at_max_intermediate']:.3f}", flush=True)

        # ---- GOE null heat trace (zero-mode corrected) ----
        ht_goe = analyze_heat_trace("K_goe_corr", raw)
        p_goe = ht_goe["plateau"]
        if p_goe is not None:
            print(f"  GOE NULL plateau: {p_goe['value']:.6f} +- {p_goe['sem']:.6f}",
                  flush=True)
        else:
            print(f"  GOE NULL: NO PLATEAU (tK_max_inter="
                  f"{ht_goe['tK_max_intermediate']:.6f})", flush=True)

        # ---- BD vs GOE comparison ----
        bd_max = ht_corr["tK_max_intermediate"]
        goe_max = ht_goe["tK_max_intermediate"]
        if goe_max > 0:
            ratio = bd_max / goe_max
            print(f"  BD/GOE ratio (intermediate max): {ratio:.3f}", flush=True)
        else:
            ratio = float("nan")

        # ---- Spectral zeta ----
        zeta_results = {}
        for mu in MU_VALUES:
            z_ens = [r["zeta_by_mu"][str(mu)] for r in raw]
            zr = analyze_spectral_zeta(z_ens, mu)
            zeta_results[f"mu={mu}"] = zr
            if not np.isnan(zr["residue"]):
                print(f"  zeta(s;mu={mu}): residue = {zr['residue']:.6f}"
                      f"  (err {zr['residue_rel_error']:+.2%})", flush=True)

        # ---- Eigenvalue density ----
        mean_f0 = float(np.mean([r["f_near_zero"] for r in raw]))
        mean_f0_goe = float(np.mean([r["f_near_zero_goe"] for r in raw]))
        mean_n_zero = float(np.mean([r["n_zero"] for r in raw]))
        mean_sigma_max = float(np.mean([r["sigma_max"] for r in raw]))

        print(f"  f(0): BD={mean_f0:.6f}, GOE={mean_f0_goe:.6f}, "
              f"target={A0_TARGET:.6f}", flush=True)
        print(f"  Mean zeros: {mean_n_zero:.1f}/{N}, sigma_max: "
              f"{mean_sigma_max:.2f}", flush=True)

        results_by_N[str(N)] = {
            "bd_corrected": {
                "plateau": p,
                "tK_max_intermediate": bd_max,
                "tau_at_max_intermediate": ht_corr["tau_at_max_intermediate"],
                "tK_curve": ht_corr["tK_curve"],
            },
            "goe_null": {
                "plateau": p_goe,
                "tK_max_intermediate": goe_max,
            },
            "bd_goe_ratio": ratio,
            "zeta": zeta_results,
            "f_near_zero": mean_f0,
            "f_near_zero_goe": mean_f0_goe,
            "mean_n_zero": mean_n_zero,
            "mean_sigma_max": mean_sigma_max,
        }

    # ==================================================================
    # CONVERGENCE
    # ==================================================================

    print(f"\n{'=' * 70}", flush=True)
    print("CONVERGENCE", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'N':>6} {'plateau':>10} {'+-CI':>10} {'GOE':>10} "
          f"{'BD/GOE':>8} {'f(0)':>8} {'f_GOE':>8}", flush=True)
    print(f"  {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} "
          f"{'-' * 8} {'-' * 8} {'-' * 8}", flush=True)

    plateau_Ns = []
    plateau_vals = []
    for N in N_VALUES:
        r = results_by_N[str(N)]
        p = r["bd_corrected"]["plateau"]
        p_g = r["goe_null"]["plateau"]
        if p is not None:
            pv = f"{p['value']:.6f}"
            ci = f"[{p['ci_lo']:.4f},{p['ci_hi']:.4f}]"
            plateau_Ns.append(N)
            plateau_vals.append(p["value"])
        else:
            pv = "no plat."
            ci = f"max={r['bd_corrected']['tK_max_intermediate']:.4f}"
        goe_str = (f"{p_g['value']:.6f}" if p_g is not None
                   else f"{r['goe_null']['tK_max_intermediate']:.4f}")
        print(f"  {N:6d} {pv:>10} {ci:>10} {goe_str:>10} "
              f"{r['bd_goe_ratio']:8.3f} {r['f_near_zero']:8.5f} "
              f"{r['f_near_zero_goe']:8.5f}", flush=True)

    # Convergence of plateau value (only N values that have a genuine plateau)
    conv_exp = conv_r2 = float("nan")
    if len(plateau_vals) >= 3:
        pv_Ns = np.array(plateau_Ns, dtype=float)
        pv_arr = np.array(plateau_vals)
        residuals = np.abs(pv_arr - A0_TARGET)
        mask_pos = residuals > 1e-10
        if np.sum(mask_pos) > 2:
            lr = stats.linregress(np.log(pv_Ns[mask_pos]),
                                  np.log(residuals[mask_pos]))
            conv_exp = float(lr.slope)
            conv_r2 = float(lr.rvalue ** 2)
    else:
        print(f"\n  Convergence: only {len(plateau_vals)} N values have plateau "
              f"(need >=3)", flush=True)

    print(f"\n  |plateau - 1/(2pi)| ~ N^{conv_exp:.3f} (R^2={conv_r2:.4f})",
          flush=True)
    print(f"  Target: {A0_TARGET:.6f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total
    best_N = N_VALUES[-1]
    rb = results_by_N[str(best_N)]
    p_best = rb["bd_corrected"]["plateau"]

    if p_best is not None and p_best["target_in_ci"]:
        verdict = (f"PASS: plateau = {p_best['value']:.6f} +- {p_best['sem']:.6f}"
                   f", 1/(2pi) in 95% CI [{p_best['ci_lo']:.5f}, "
                   f"{p_best['ci_hi']:.5f}]")
    elif p_best is not None and abs(p_best["rel_error"]) < 0.10:
        verdict = (f"MARGINAL: plateau = {p_best['value']:.6f}, "
                   f"error {p_best['rel_error']:+.2%} but target not in CI")
    elif p_best is not None:
        verdict = (f"FAIL: plateau = {p_best['value']:.6f}, "
                   f"error {p_best['rel_error']:+.2%}")
        if rb["bd_goe_ratio"] > 1.3:
            verdict += f" | BD > GOE by {rb['bd_goe_ratio']:.2f}x (geometric content)"
    elif not np.isnan(conv_exp) and conv_exp < -0.3 and conv_r2 > 0.7:
        verdict = (f"CONVERGING (no plateau yet): tK_max = "
                   f"{rb['bd_corrected']['tK_max_intermediate']:.6f}, "
                   f"rate N^{conv_exp:.2f}")
    else:
        verdict = (f"FAIL: no plateau at N={best_N}, tK_max = "
                   f"{rb['bd_corrected']['tK_max_intermediate']:.6f} "
                   f"(target {A0_TARGET:.6f})")
        if rb["bd_goe_ratio"] > 1.3:
            verdict += f" | BD > GOE by {rb['bd_goe_ratio']:.2f}x (geometric content)"

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    # Save
    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "M": M_ENSEMBLE, "T": T_DIAMOND,
            "tau_range": [TAU_MIN, TAU_MAX, N_TAU],
            "s_range": [S_MIN, S_MAX, N_S],
            "mu_values": MU_VALUES,
            "plateau_deriv_threshold": PLATEAU_DERIV_THRESHOLD,
            "plateau_min_decades": PLATEAU_MIN_DECADES,
        },
        "prediction": {"a0": A0_TARGET, "formula": "tau*K_corr(tau) plateau -> 1/(2pi)"},
        "results_by_N": results_by_N,
        "convergence": {"exponent": conv_exp, "r2": conv_r2},
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp7_dw_a0.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
