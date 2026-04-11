"""
FND-1 Step (a): Does H = (L_BD + L_BD^T)/2 converge to the d'Alembertian?

This is the ZERO-LEVEL TEST. If H does not approximate Box on test functions,
all heat kernel / spectral action experiments with H are meaningless.

Method:
  1. Sprinkle N points into flat 4D Minkowski diamond
  2. Build the BD retarded operator L_BD using d=4 coefficients
  3. Symmetrize: H = (L_BD + L_BD^T) / 2
  4. Apply H to known test functions f(x_i)
  5. Compare <Hf>_interior with analytical Box(f), Delta(f), Spatial(f), -d_tt(f)
  6. The RATIO <Hf>/(Op f) identifies the operator
  7. Cross-consistency of ratio across functions determines if H -> Op

Key discriminator:
  - If <H t^2> / <H x^2> ~ -1 : H converges to Box (d'Alembertian)
  - If <H t^2> / <H x^2> ~ -24 : H converges to -d_tt (like L_link)
  - If ratio varies with N: not yet converged

BD d=4 coefficients (SymPy-verified against Benincasa-Dowker 2010, Aslanbeigi+ 2014):
  Relative: diagonal=-1, layer0=+1, layer1=-9, layer2=+16, layer3=-8
  Code: (-4, +4, -36, +64, -32)/sqrt(6) [factor 4 from 2/l^2 normalization]

Pre-registration: PRIMARY TEST = ratio <H t^2> / <H x^2>.
  If ~ -1 -> Box. If ~ -24 -> -d_tt. Anything else -> unknown operator.

Run:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_h_convergence.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d,
)
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000, 3000]
M_SEEDS = 30
T_DIAMOND = 1.0
MASTER_SEED = 314

# Boundary exclusion: 3 link lengths
# In d=4: link length ~ (V/N)^(1/4) where V = pi*T^4/24
# At N=2000, T=1: l ~ (pi/24/2000)^0.25 ~ 0.081. Margin = 3*0.081 = 0.24
# So exclude points with |t|+r > T/2 - 0.24 = 0.26
# Parametrize as fraction: margin_frac such that boundary_dist > margin_frac * T/2
BOUNDARY_MARGIN_FRAC = 0.48  # at N=2000: margin ~ 0.24 = 3 link lengths


# ---------------------------------------------------------------------------
# Build H = (L_BD + L_BD^T) / 2 in d=4
# ---------------------------------------------------------------------------

def build_H_4d(C, n_matrix, rho):
    """Build symmetrized BD operator H = (L + L^T)/2 in d=4.

    BD d=4 (Benincasa-Dowker 2010, verified with SymPy):
      L_ij = prefactor * [diag_coeff * delta_ij + layer_coeff * past_layer_mask]

    Coefficients (relative): diag=-1, layer0=+1, layer1=-9, layer2=+16, layer3=-8
    Prefactor: 4 * rho^(1/2) / sqrt(6)  [from 2/l_4^2 where l_4^2 = 1/(rho*V_4)^(1/2)]
    """
    N = C.shape[0]
    prefactor = 4.0 * np.sqrt(rho) / np.sqrt(6.0)

    past = C.T  # past[i,j] = 1 means j precedes i
    n_past = n_matrix.T

    # Build L (retarded: acts on past)
    L = np.zeros((N, N), dtype=np.float64)

    # Diagonal: -1 * prefactor
    np.fill_diagonal(L, -1.0 * prefactor)

    # Layer 0 (links, n=0): +1
    mask0 = ((past > 0) & (n_past == 0)).astype(np.float64)
    L += 1.0 * prefactor * mask0

    # Layer 1 (n=1): -9
    mask1 = ((past > 0) & (n_past == 1)).astype(np.float64)
    L += -9.0 * prefactor * mask1

    # Layer 2 (n=2): +16
    mask2 = ((past > 0) & (n_past == 2)).astype(np.float64)
    L += 16.0 * prefactor * mask2

    # Layer 3 (n=3): -8
    mask3 = ((past > 0) & (n_past == 3)).astype(np.float64)
    L += -8.0 * prefactor * mask3

    # Symmetrize: H = (L + L^T) / 2
    H = (L + L.T) / 2.0

    return H, L


# ---------------------------------------------------------------------------
# Test functions (module-level for clarity, no lambdas)
# ---------------------------------------------------------------------------

def f_t2(pts):
    return pts[:, 0]**2

def f_x2(pts):
    return pts[:, 1]**2

def f_y2(pts):
    return pts[:, 2]**2

def f_z2(pts):
    return pts[:, 3]**2

def f_t2_minus_r2(pts):
    return pts[:, 0]**2 - pts[:, 1]**2 - pts[:, 2]**2 - pts[:, 3]**2

def f_t2_plus_r2(pts):
    return pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2

def f_x2_minus_y2(pts):
    return pts[:, 1]**2 - pts[:, 2]**2

def f_tx(pts):
    return pts[:, 0] * pts[:, 1]

def f_cos_pit(pts):
    return np.cos(np.pi * pts[:, 0])

def f_cos_pix(pts):
    return np.cos(np.pi * pts[:, 1])


# Analytical operator values (SymPy-verified)
# Box = -d_tt + d_xx + d_yy + d_zz
# -d_tt = just the temporal part
TEST_FUNCTIONS = [
    # (name, function, Box value, -d_tt value)
    # For quadratics: operator values are constants
    ("t^2",       f_t2,           -2.0,  -2.0),
    ("x^2",       f_x2,            2.0,   0.0),
    ("y^2",       f_y2,            2.0,   0.0),
    ("z^2",       f_z2,            2.0,   0.0),
    ("t^2-r^2",   f_t2_minus_r2,  -8.0,  -2.0),
    ("t^2+r^2",   f_t2_plus_r2,    4.0,  -2.0),
    ("x^2-y^2",   f_x2_minus_y2,   0.0,   0.0),  # harmonic
    ("t*x",       f_tx,            0.0,   0.0),  # harmonic
]

# Position-dependent test functions: Box f varies spatially
POS_TEST_FUNCTIONS = [
    # (name, function, box_function, neg_dtt_function)
    ("cos(pi*t)", f_cos_pit,
     lambda p: np.pi**2 * np.cos(np.pi * p[:, 0]),    # Box
     lambda p: np.pi**2 * np.cos(np.pi * p[:, 0])),   # -d_tt (same!)
    ("cos(pi*x)", f_cos_pix,
     lambda p: -np.pi**2 * np.cos(np.pi * p[:, 1]),   # Box
     lambda p: np.zeros(len(p))),                       # -d_tt = 0
]


# ---------------------------------------------------------------------------
# Single sprinkling analysis
# ---------------------------------------------------------------------------

def analyze_one(seed, N, T):
    """One sprinkling: build H, apply to test functions, return ratios."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d_flat(N, T, rng)
    C = causal_matrix_4d(pts, 0.0, "flat")

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    V = np.pi * T**4 / 24.0
    rho = N / V

    H, L = build_H_4d(C, n_matrix, rho)

    # Interior mask: exclude boundary
    half = T / 2.0
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    boundary_dist = half - (np.abs(pts[:, 0]) + r)
    interior = boundary_dist > BOUNDARY_MARGIN_FRAC * half
    n_int = int(np.sum(interior))

    if n_int < 30:
        return None

    results = {}

    # Constant-operator test functions
    for name, fn, box_val, neg_dtt_val in TEST_FUNCTIONS:
        f_vals = fn(pts)
        Hf = H @ f_vals
        Hf_int = Hf[interior]
        mean_Hf = float(np.mean(Hf_int))
        std_Hf = float(np.std(Hf_int))

        results[name] = {
            "mean_Hf": mean_Hf,
            "std_Hf": std_Hf,
            "box_val": box_val,
            "neg_dtt_val": neg_dtt_val,
        }

    # Position-dependent test functions
    for name, fn, box_fn, neg_dtt_fn in POS_TEST_FUNCTIONS:
        f_vals = fn(pts)
        Hf = H @ f_vals
        Hf_int = Hf[interior]
        box_int = box_fn(pts)[interior]
        neg_dtt_int = neg_dtt_fn(pts)[interior]

        r_box = float(stats.pearsonr(Hf_int, box_int)[0]) if np.std(box_int) > 1e-15 and np.std(Hf_int) > 1e-15 else 0.0
        r_dtt = float(stats.pearsonr(Hf_int, neg_dtt_int)[0]) if np.std(neg_dtt_int) > 1e-15 and np.std(Hf_int) > 1e-15 else 0.0

        # Regression slopes
        a_box = float(stats.linregress(box_int, Hf_int).slope) if np.std(box_int) > 1e-15 else 0.0
        a_dtt = float(stats.linregress(neg_dtt_int, Hf_int).slope) if np.std(neg_dtt_int) > 1e-15 else 0.0

        results[name] = {
            "r_box": r_box,
            "r_neg_dtt": r_dtt,
            "alpha_box": a_box,
            "alpha_neg_dtt": a_dtt,
        }

    return {
        "N": N, "n_interior": n_int, "rho": rho,
        "mean_degree_H": float(np.mean(np.sum(np.abs(H) > 0, axis=1))),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70, flush=True)
    print("FND-1 STEP (a): DOES H = (L_BD + L_BD^T)/2 CONVERGE TO BOX?", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, seeds: {M_SEEDS}", flush=True)
    print(f"Boundary exclusion: {BOUNDARY_MARGIN_FRAC*100:.0f}% of diamond half-width", flush=True)
    print(f"BD d=4 coefficients: diag=-1, layer0=+1, layer1=-9, layer2=+16, layer3=-8", flush=True)
    print(f"Pre-registered primary: ratio <H t^2> / <H x^2>", flush=True)
    print(f"  If ~ -1: H -> Box. If ~ -24: H -> -d_tt. If ~ 0: H -> Spatial.", flush=True)
    print(flush=True)

    # Smoke test
    t0 = time.perf_counter()
    r = analyze_one(42, 500, T_DIAMOND)
    print(f"Smoke test (N=500): {time.perf_counter()-t0:.2f}s, "
          f"n_interior={r['n_interior']}, mean_deg_H={r['mean_degree_H']:.0f}", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    all_results = {}

    for N in N_VALUES:
        seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(M_SEEDS)]
        print(f"\n  N={N}: {M_SEEDS} sprinklings...", flush=True)
        t0 = time.perf_counter()
        raw = [analyze_one(s, N, T_DIAMOND) for s in seeds]
        results = [r for r in raw if r is not None]
        elapsed = time.perf_counter() - t0
        n_int_avg = np.mean([r["n_interior"] for r in results])
        print(f"    Done {elapsed:.1f}s, valid={len(results)}/{M_SEEDS}, "
              f"n_interior_avg={n_int_avg:.0f}", flush=True)
        all_results[N] = results

    # ==================================================================
    # ANALYSIS
    # ==================================================================

    print(f"\n{'='*70}", flush=True)
    print("CONSTANT-OPERATOR TEST FUNCTIONS", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  If H -> alpha*Box, then alpha = <Hf>/Box(f) should be CONSTANT across all f.", flush=True)
    print(f"  If H -> alpha*(-d_tt), then alpha = <Hf>/(-d_tt f) should be CONSTANT.", flush=True)

    const_fns = ["t^2", "x^2", "y^2", "z^2", "t^2-r^2", "t^2+r^2"]

    for N in N_VALUES:
        results = all_results[N]
        print(f"\n  N={N}:", flush=True)
        print(f"  {'func':>10} {'<Hf>':>10} {'std':>8} {'Box(f)':>7} {'a_box':>8} "
              f"{'-dtt(f)':>7} {'a_dtt':>8}", flush=True)
        print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*8}", flush=True)

        for fn_name in const_fns:
            mean_Hf = np.mean([r["results"][fn_name]["mean_Hf"] for r in results])
            std_Hf = np.mean([r["results"][fn_name]["std_Hf"] for r in results])
            box_v = results[0]["results"][fn_name]["box_val"]
            dtt_v = results[0]["results"][fn_name]["neg_dtt_val"]
            a_box = mean_Hf / box_v if abs(box_v) > 0.01 else float('nan')
            a_dtt = mean_Hf / dtt_v if abs(dtt_v) > 0.01 else float('nan')
            print(f"  {fn_name:>10} {mean_Hf:+10.4f} {std_Hf:8.4f} {box_v:+7.1f} {a_box:+8.4f} "
                  f"{dtt_v:+7.1f} {a_dtt:+8.4f}", flush=True)

    # PRIMARY TEST: ratio <H t^2> / <H x^2>
    print(f"\n{'='*70}", flush=True)
    print("PRIMARY TEST: RATIO <H t^2> / <H x^2>", flush=True)
    print("=" * 70, flush=True)
    print(f"  Box: ratio = -2/2 = -1.0", flush=True)
    print(f"  -d_tt: ratio = -2/0 = undefined (x^2 gives 0)", flush=True)
    print(f"  Spatial: ratio = 0/2 = 0.0", flush=True)
    print(f"  Delta: ratio = 2/2 = 1.0", flush=True)

    for N in N_VALUES:
        results = all_results[N]
        mean_t2 = np.mean([r["results"]["t^2"]["mean_Hf"] for r in results])
        mean_x2 = np.mean([r["results"]["x^2"]["mean_Hf"] for r in results])
        ratio = mean_t2 / mean_x2 if abs(mean_x2) > 1e-10 else float('inf')
        print(f"  N={N}: <H t^2>={mean_t2:+.4f}, <H x^2>={mean_x2:+.4f}, "
              f"ratio={ratio:+.4f}", flush=True)

    # CROSS-CONSISTENCY
    print(f"\n{'='*70}", flush=True)
    print("CROSS-CONSISTENCY: alpha = <Hf>/(Op f) for each candidate", flush=True)
    print("=" * 70, flush=True)

    for N in N_VALUES:
        results = all_results[N]
        alphas_box = []
        alphas_dtt = []
        for fn_name in const_fns:
            mean_Hf = np.mean([r["results"][fn_name]["mean_Hf"] for r in results])
            box_v = results[0]["results"][fn_name]["box_val"]
            dtt_v = results[0]["results"][fn_name]["neg_dtt_val"]
            if abs(box_v) > 0.01:
                alphas_box.append(mean_Hf / box_v)
            if abs(dtt_v) > 0.01:
                alphas_dtt.append(mean_Hf / dtt_v)

        cv_box = np.std(alphas_box) / abs(np.mean(alphas_box)) if len(alphas_box) >= 2 and abs(np.mean(alphas_box)) > 0 else float('inf')
        cv_dtt = np.std(alphas_dtt) / abs(np.mean(alphas_dtt)) if len(alphas_dtt) >= 2 and abs(np.mean(alphas_dtt)) > 0 else float('inf')

        print(f"  N={N}:", flush=True)
        print(f"    Box: alpha = {np.mean(alphas_box):+.4f} +/- {np.std(alphas_box):.4f} "
              f"(CV={cv_box:.3f}, n={len(alphas_box)})", flush=True)
        print(f"    -d_tt: alpha = {np.mean(alphas_dtt):+.4f} +/- {np.std(alphas_dtt):.4f} "
              f"(CV={cv_dtt:.3f}, n={len(alphas_dtt)})", flush=True)

    # HARMONIC TEST
    print(f"\n{'='*70}", flush=True)
    print("HARMONIC TEST: <Hf> should be ~0 for harmonic f", flush=True)
    print("=" * 70, flush=True)

    for N in N_VALUES:
        results = all_results[N]
        for fn_name in ["x^2-y^2", "t*x"]:
            mean_Hf = np.mean([r["results"][fn_name]["mean_Hf"] for r in results])
            mean_t2 = abs(np.mean([r["results"]["t^2"]["mean_Hf"] for r in results]))
            ratio = abs(mean_Hf) / mean_t2 if mean_t2 > 0 else 0
            print(f"  N={N}, {fn_name}: <Hf>={mean_Hf:+.6f}, |<Hf>|/|<H t^2>|={ratio:.4f}",
                  flush=True)

    # POSITION-DEPENDENT FUNCTIONS
    print(f"\n{'='*70}", flush=True)
    print("POSITION-DEPENDENT: cos(pi*t) discriminates Box from -d_tt", flush=True)
    print("=" * 70, flush=True)
    print(f"  cos(pi*t): Box = +pi^2 cos, -d_tt = +pi^2 cos (SAME! Does not discriminate)", flush=True)
    print(f"  cos(pi*x): Box = -pi^2 cos, -d_tt = 0 (DISCRIMINATES!)", flush=True)

    for N in N_VALUES:
        results = all_results[N]
        for fn_name in ["cos(pi*t)", "cos(pi*x)"]:
            r_box = np.mean([r["results"][fn_name]["r_box"] for r in results])
            r_dtt = np.mean([r["results"][fn_name]["r_neg_dtt"] for r in results])
            a_box = np.mean([r["results"][fn_name]["alpha_box"] for r in results])
            print(f"  N={N}, {fn_name}: r_box={r_box:+.4f}, r_dtt={r_dtt:+.4f}, "
                  f"alpha_box={a_box:+.4f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    # Use largest N for final determination
    N_final = max(N_VALUES)
    results = all_results[N_final]

    mean_t2 = np.mean([r["results"]["t^2"]["mean_Hf"] for r in results])
    mean_x2 = np.mean([r["results"]["x^2"]["mean_Hf"] for r in results])
    ratio = mean_t2 / mean_x2 if abs(mean_x2) > 1e-10 else float('inf')

    # Compute CV for both candidates
    alphas_box, alphas_dtt = [], []
    for fn_name in const_fns:
        mean_Hf = np.mean([r["results"][fn_name]["mean_Hf"] for r in results])
        box_v = results[0]["results"][fn_name]["box_val"]
        dtt_v = results[0]["results"][fn_name]["neg_dtt_val"]
        if abs(box_v) > 0.01:
            alphas_box.append(mean_Hf / box_v)
        if abs(dtt_v) > 0.01:
            alphas_dtt.append(mean_Hf / dtt_v)

    cv_box = np.std(alphas_box) / abs(np.mean(alphas_box)) if alphas_box else float('inf')
    cv_dtt = np.std(alphas_dtt) / abs(np.mean(alphas_dtt)) if alphas_dtt else float('inf')

    if abs(ratio + 1.0) < 0.3 and cv_box < 0.3:
        verdict = f"H CONVERGES TO BOX (ratio={ratio:+.3f}, CV_box={cv_box:.3f})"
    elif cv_dtt < 0.3:
        verdict = f"H CONVERGES TO -d_tt (CV_dtt={cv_dtt:.3f})"
    elif abs(ratio) > 10:
        verdict = f"H IS ANISOTROPIC (~-d_tt dominant, ratio={ratio:+.1f})"
    else:
        verdict = f"H CONVERGES TO UNKNOWN OPERATOR (ratio={ratio:+.3f}, CV_box={cv_box:.3f}, CV_dtt={cv_dtt:.3f})"

    print(f"\n  {verdict}", flush=True)
    print(f"  Primary: <H t^2>/<H x^2> = {ratio:+.4f} (Box predicts -1.0)", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    meta = ExperimentMeta(route=2, name="h_convergence",
                          description="Step (a): H convergence test on d=4 test functions",
                          N=N_final, M=M_SEEDS, status="completed", verdict=verdict)
    meta.wall_time_sec = total_time
    output = {"verdict": verdict, "ratio_t2_x2": float(ratio),
              "cv_box": float(cv_box), "cv_dtt": float(cv_dtt),
              "wall_time_sec": total_time}
    save_experiment(meta, output, RESULTS_DIR / "h_convergence.json")
    print(f"  Saved: {RESULTS_DIR / 'h_convergence.json'}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
