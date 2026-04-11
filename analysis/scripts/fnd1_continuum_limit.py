"""
FND-1: Numerical Identification of the Continuum Limit of L_link.

Core question: what differential operator does the link-graph Laplacian
converge to when applied to smooth test functions?

Method:
  1. Sprinkle N points into flat 4D Minkowski diamond
  2. Build link graph, compute L_link
  3. Apply L_link to known smooth functions f evaluated at sprinkled points
  4. Compare (L_link f)(x_i) with (Op f)(x_i) for candidate operators Op:
     - Box (d'Alembertian): -d^2f/dt^2 + d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2
     - Delta (Laplacian): d^2f/dt^2 + d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2
     - Spatial Laplacian: d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2
  5. Measure: Pearson r between (L_link f)_i and (Op f)_i across all points i
  6. The operator with highest r IS the continuum limit
  7. Finite-size scaling: repeat at N=500, 1000, 2000, 3000 to check convergence
  8. Extract normalization: fit (L_link f) = alpha * (Op f) + beta

Test functions (chosen for discriminating power):
  f1 = t                    -> Box=0, Delta=0, Spatial=0
  f2 = x^2 + y^2 + z^2     -> Box=6, Delta=8, Spatial=6  (wait: Box=6, Delta=6+2=8? No.)
  Actually: Box f = -d_tt f + d_xx f + d_yy f + d_zz f
  Delta f = d_tt f + d_xx f + d_yy f + d_zz f  (Euclidean 4D Laplacian)
  Spatial f = d_xx f + d_yy f + d_zz f

  f1 = t               -> Box=0, Delta=0, Spatial=0 (linear: all zero)
  f2 = t^2             -> Box=-2, Delta=2, Spatial=0
  f3 = x^2             -> Box=2, Delta=2, Spatial=2
  f4 = t^2 - x^2       -> Box=-4, Delta=0, Spatial=-2
  f5 = x^2 + y^2 + z^2 -> Box=6, Delta=6, Spatial=6  (doesn't discriminate Box vs Delta)
  f6 = t^2 + x^2 + y^2 + z^2 -> Box=4, Delta=8, Spatial=6  (DISCRIMINATES)
  f7 = t*x             -> Box=0, Delta=0, Spatial=0 (all zero, tests linearity)
  f8 = t^2 - x^2 - y^2 - z^2 -> Box=-8, Delta=0, Spatial=-6 (BEST DISCRIMINATOR)

Key: f6 and f8 discriminate Box from Delta from Spatial.
  f8: Box=-8, Delta=0, Spatial=-6. If L_link f8 ~ -8*alpha -> Box.
                                    If L_link f8 ~ 0 -> Delta.
                                    If L_link f8 ~ -6*alpha -> Spatial.

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_continuum_limit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d,
    build_link_graph, link_spectral_embedding,
)
from fnd1_parallel import N_WORKERS, _init_worker
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000, 3000]
M_SEEDS = 30          # seeds per N for averaging
T_DIAMOND = 1.0
MASTER_SEED = 123
WORKERS = N_WORKERS

# We exclude boundary points (within fraction BOUNDARY_MARGIN of the diamond edge)
# because L_link has boundary artifacts (fewer neighbors near the edge)
BOUNDARY_MARGIN = 0.15  # exclude points with |t|+r > (1-margin)*T/2


# ---------------------------------------------------------------------------
# Test functions (module-level for pickling on Windows)
# ---------------------------------------------------------------------------

_GA = 5.0  # Gaussian width

def _f_t2(p): return p[:, 0]**2
def _f_x2(p): return p[:, 1]**2
def _f_t2mr2(p): return p[:, 0]**2 - p[:, 1]**2 - p[:, 2]**2 - p[:, 3]**2
def _f_t2pr2(p): return p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2
def _f_x2my2(p): return p[:, 1]**2 - p[:, 2]**2
def _f_tx(p): return p[:, 0] * p[:, 1]
def _f_cospt(p): return np.cos(np.pi * p[:, 0])
def _f_cospx(p): return np.cos(np.pi * p[:, 1])
def _f_gauss(p): return np.exp(-_GA * np.sum(p**2, axis=1))

def _const(val):
    """Return a picklable constant-function factory."""
    def fn(p): return np.full(len(p), val)
    return fn

# Box = -d_tt + d_xx + d_yy + d_zz  (signature -+++)
# Delta = d_tt + d_xx + d_yy + d_zz (4D Euclidean)
# Spatial = d_xx + d_yy + d_zz

def _box_cospt(p): return np.pi**2 * np.cos(np.pi * p[:, 0])    # -(-pi^2 cos) = +pi^2 cos
def _del_cospt(p): return -np.pi**2 * np.cos(np.pi * p[:, 0])   # -pi^2 cos
def _box_cospx(p): return -np.pi**2 * np.cos(np.pi * p[:, 1])   # same for all 3 ops
def _box_gauss(p):
    r2 = p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2
    return (-4*_GA + 4*_GA**2*(r2 - p[:, 0]**2)) * _f_gauss(p)
def _del_gauss(p):
    r4d2 = np.sum(p**2, axis=1)
    return (-8*_GA + 4*_GA**2*r4d2) * _f_gauss(p)
def _spa_gauss(p):
    r2 = p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2
    return (-6*_GA + 4*_GA**2*r2) * _f_gauss(p)


def make_test_functions():
    """Return list of (name, f, Box_f, Delta_f, Spatial_f).

    Box = -d_tt + d_xx + d_yy + d_zz  (d'Alembertian, -+++ signature)
    Delta = d_tt + d_xx + d_yy + d_zz  (4D Euclidean Laplacian)
    Spatial = d_xx + d_yy + d_zz       (3D spatial Laplacian)
    """
    tests = []

    # f1: t^2  -> discriminates Box (-2) vs Delta (+2) vs Spatial (0)
    tests.append({
        "name": "t^2",
        "f": lambda p: p[:, 0]**2,
        "box": lambda p: np.full(len(p), -2.0),
        "delta": lambda p: np.full(len(p), 2.0),
        "spatial": lambda p: np.full(len(p), 0.0),
    })

    # f2: x^2  -> Box=2, Delta=2, Spatial=2 (doesn't discriminate Box vs Delta)
    tests.append({
        "name": "x^2",
        "f": lambda p: p[:, 1]**2,
        "box": lambda p: np.full(len(p), 2.0),
        "delta": lambda p: np.full(len(p), 2.0),
        "spatial": lambda p: np.full(len(p), 2.0),
    })

    # f3: t^2 - x^2 - y^2 - z^2  (Lorentz interval from origin)
    # Box = -2 + 2 + 2 + 2 = -8 (WRONG, let me recalculate)
    # Actually: Box f = -d_tt f + d_xx f + d_yy f + d_zz f
    # d_tt(t^2) = 2, d_xx(-x^2) = -2, d_yy(-y^2) = -2, d_zz(-z^2) = -2
    # Box f = -2 + (-2) + (-2) + (-2) = -8
    # Delta f = 2 + (-2) + (-2) + (-2) = -4
    # Spatial f = -2 + -2 + -2 = -6
    tests.append({
        "name": "t^2-r^2",
        "f": lambda p: p[:, 0]**2 - p[:, 1]**2 - p[:, 2]**2 - p[:, 3]**2,
        "box": lambda p: np.full(len(p), -8.0),
        "delta": lambda p: np.full(len(p), -4.0),
        "spatial": lambda p: np.full(len(p), -6.0),
    })

    # f4: t^2 + x^2 + y^2 + z^2  (Euclidean distance^2 from origin)
    # Box = -2 + 2 + 2 + 2 = 4
    # Delta = 2 + 2 + 2 + 2 = 8
    # Spatial = 2 + 2 + 2 = 6
    tests.append({
        "name": "t^2+r^2",
        "f": lambda p: p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2,
        "box": lambda p: np.full(len(p), 4.0),
        "delta": lambda p: np.full(len(p), 8.0),
        "spatial": lambda p: np.full(len(p), 6.0),
    })

    # f5: x^2 - y^2  (anisotropic spatial, tidal-like)
    # Box = 0 + 2 - 2 + 0 = 0
    # Delta = 0 + 2 - 2 + 0 = 0
    # Spatial = 2 - 2 + 0 = 0
    # All zero -- tests that L_link gives zero for harmonic functions
    tests.append({
        "name": "x^2-y^2",
        "f": lambda p: p[:, 1]**2 - p[:, 2]**2,
        "box": lambda p: np.full(len(p), 0.0),
        "delta": lambda p: np.full(len(p), 0.0),
        "spatial": lambda p: np.full(len(p), 0.0),
    })

    # f6: t*x  (mixed, all operators give 0)
    tests.append({
        "name": "t*x",
        "f": lambda p: p[:, 0] * p[:, 1],
        "box": lambda p: np.full(len(p), 0.0),
        "delta": lambda p: np.full(len(p), 0.0),
        "spatial": lambda p: np.full(len(p), 0.0),
    })

    # f7: cos(pi*t) (oscillatory temporal)
    # d_tt = -pi^2 cos(pi*t)
    # Box = -d_tt = pi^2 cos(pi*t)
    # Delta = d_tt = -pi^2 cos(pi*t)
    # Spatial = 0
    tests.append({
        "name": "cos(pi*t)",
        "f": lambda p: np.cos(np.pi * p[:, 0]),
        "box": lambda p: np.pi**2 * np.cos(np.pi * p[:, 0]),
        "delta": lambda p: -np.pi**2 * np.cos(np.pi * p[:, 0]),
        "spatial": lambda p: np.full(len(p), 0.0),
    })

    # f8: cos(pi*x) (oscillatory spatial)
    # d_xx = -pi^2 cos(pi*x)
    # Box = d_xx = -pi^2 cos(pi*x)
    # Delta = d_xx = -pi^2 cos(pi*x)
    # Spatial = d_xx = -pi^2 cos(pi*x)
    tests.append({
        "name": "cos(pi*x)",
        "f": lambda p: np.cos(np.pi * p[:, 1]),
        "box": lambda p: -np.pi**2 * np.cos(np.pi * p[:, 1]),
        "delta": lambda p: -np.pi**2 * np.cos(np.pi * p[:, 1]),
        "spatial": lambda p: -np.pi**2 * np.cos(np.pi * p[:, 1]),
    })

    # f9: exp(-5*(t^2+x^2+y^2+z^2))  (Gaussian, position-dependent derivatives)
    # Laplacian of exp(-a*r4d^2) = (-2a*d + 4a^2*r4d^2) * exp(-a*r4d^2) where d=4
    # Box = (2a - 4a^2*t^2) + (-2a + 4a^2*x^2) + (-2a + 4a^2*y^2) + (-2a + 4a^2*z^2)
    #      = 2a - 6a + 4a^2(-t^2+x^2+y^2+z^2) = -4a + 4a^2*(r^2-t^2)
    # with a=5: Box = -20 + 100*(r^2-t^2)
    # Delta = -8a + 4a^2*(t^2+r^2) = -40 + 100*(t^2+r^2)
    # Spatial = -6a + 4a^2*r^2 = -30 + 100*r^2
    a = 5.0
    tests.append({
        "name": "gaussian",
        "f": lambda p: np.exp(-a * (p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)),
        "box": lambda p: (-4*a + 4*a**2 * (p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2 - p[:, 0]**2)) *
                          np.exp(-a * (p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)),
        "delta": lambda p: (-8*a + 4*a**2 * (p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)) *
                            np.exp(-a * (p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)),
        "spatial": lambda p: (-6*a + 4*a**2 * (p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)) *
                              np.exp(-a * (p[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2 + p[:, 3]**2)),
    })

    return tests


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_continuum(args):
    """One sprinkling: apply L_link to all test functions, compare with operators."""
    seed_int, N, T, test_functions = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)
    C = causal_matrix_4d(pts, 0.0, "flat")

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    A_link = build_link_graph(C, n_matrix)

    degrees = np.array(A_link.sum(axis=1)).ravel()
    L = sp.diags(degrees) - A_link

    # Interior mask: exclude boundary points
    half = T / 2.0
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    boundary_dist = half - (np.abs(pts[:, 0]) + r)
    interior = boundary_dist > BOUNDARY_MARGIN * half
    n_interior = int(np.sum(interior))

    if n_interior < 50:
        return None

    mean_degree = float(np.mean(degrees[interior]))

    results = {}
    for tf in test_functions:
        name = tf["name"]

        # Evaluate f at sprinkled points
        f_vals = tf["f"](pts)

        # Apply L_link
        Lf = np.array(L @ f_vals).ravel()

        # Analytical operator values (at interior points only)
        box_vals = tf["box"](pts)
        delta_vals = tf["delta"](pts)
        spatial_vals = tf["spatial"](pts)

        # Restrict to interior
        Lf_int = Lf[interior]
        box_int = box_vals[interior]
        delta_int = delta_vals[interior]
        spatial_int = spatial_vals[interior]

        # Correlations (Lf vs each operator)
        r_box, p_box = (0.0, 1.0)
        r_delta, p_delta = (0.0, 1.0)
        r_spatial, p_spatial = (0.0, 1.0)

        if np.std(Lf_int) > 1e-15:
            if np.std(box_int) > 1e-15:
                r_box, p_box = stats.pearsonr(Lf_int, box_int)
            if np.std(delta_int) > 1e-15:
                r_delta, p_delta = stats.pearsonr(Lf_int, delta_int)
            if np.std(spatial_int) > 1e-15:
                r_spatial, p_spatial = stats.pearsonr(Lf_int, spatial_int)

        # For constant-valued operators (like f=t^2 where Box=-2 everywhere),
        # correlation is undefined. Use ratio instead.
        mean_Lf = float(np.mean(Lf_int))
        std_Lf = float(np.std(Lf_int))

        # Regression: Lf = alpha * Op + beta (for position-dependent test functions)
        alpha_box, alpha_delta, alpha_spatial = 0.0, 0.0, 0.0
        if np.std(box_int) > 1e-15:
            sl, _, _, _, _ = stats.linregress(box_int, Lf_int)
            alpha_box = float(sl)
        if np.std(delta_int) > 1e-15:
            sl, _, _, _, _ = stats.linregress(delta_int, Lf_int)
            alpha_delta = float(sl)
        if np.std(spatial_int) > 1e-15:
            sl, _, _, _, _ = stats.linregress(spatial_int, Lf_int)
            alpha_spatial = float(sl)

        results[name] = {
            "mean_Lf": mean_Lf,
            "std_Lf": std_Lf,
            "r_box": float(r_box),
            "r_delta": float(r_delta),
            "r_spatial": float(r_spatial),
            "alpha_box": alpha_box,
            "alpha_delta": alpha_delta,
            "alpha_spatial": alpha_spatial,
            "n_interior": n_interior,
        }

    return {
        "N": N, "seed": seed_int,
        "mean_degree": mean_degree,
        "n_interior": n_interior,
        "n_links": N0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    test_functions = make_test_functions()
    tf_names = [tf["name"] for tf in test_functions]

    print("=" * 70, flush=True)
    print("FND-1: CONTINUUM LIMIT OF L_link", flush=True)
    print("=" * 70, flush=True)
    print(f"Question: L_link f -> Box f, Delta f, Spatial f, or something else?", flush=True)
    print(f"N values: {N_VALUES}, seeds per N: {M_SEEDS}", flush=True)
    print(f"Test functions: {tf_names}", flush=True)
    print(f"Boundary exclusion: {BOUNDARY_MARGIN*100:.0f}%", flush=True)
    print(flush=True)

    # Benchmark
    t0 = time.perf_counter()
    _worker_continuum((42, 1000, T_DIAMOND, test_functions))
    print(f"Benchmark: {time.perf_counter()-t0:.2f}s/task at N=1000", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    all_results = {}

    for N in N_VALUES:
        seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(M_SEEDS)]
        args = [(si, N, T_DIAMOND, test_functions) for si in seeds]

        print(f"\n  N={N}: {M_SEEDS} sprinklings...", flush=True)
        t0 = time.perf_counter()
        raw = [_worker_continuum(a) for a in args]  # serial: avoids lambda pickling
        results = [r for r in raw if r is not None]
        elapsed = time.perf_counter() - t0

        mean_deg = np.mean([r["mean_degree"] for r in results])
        print(f"    Done in {elapsed:.1f}s, mean_degree={mean_deg:.1f}, "
              f"valid={len(results)}/{M_SEEDS}", flush=True)

        all_results[N] = results

    # ==================================================================
    # ANALYSIS: For each test function, which operator matches best?
    # ==================================================================

    print(f"\n{'='*70}", flush=True)
    print("OPERATOR IDENTIFICATION", flush=True)
    print("=" * 70, flush=True)

    # For CONSTANT-valued operators (t^2, x^2, t^2-r^2, t^2+r^2, x^2-y^2, t*x):
    # Use the RATIO mean(L_link f) / (Op f) to find the normalization constant alpha.
    # For POSITION-DEPENDENT operators (cos(pi*t), cos(pi*x), gaussian):
    # Use correlation r(L_link f, Op f) and regression slope alpha.

    print(f"\n--- CONSTANT-OPERATOR TEST FUNCTIONS ---", flush=True)
    print(f"  For these, Op f = const. We report <L_link f> / <degree> and compare ratios.", flush=True)

    const_fns = ["t^2", "x^2", "t^2-r^2", "t^2+r^2"]
    op_vals = {
        "t^2":     {"box": -2, "delta": 2, "spatial": 0},
        "x^2":     {"box": 2, "delta": 2, "spatial": 2},
        "t^2-r^2": {"box": -8, "delta": -4, "spatial": -6},
        "t^2+r^2": {"box": 4, "delta": 8, "spatial": 6},
    }

    for N in N_VALUES:
        results = all_results[N]
        mean_deg = np.mean([r["mean_degree"] for r in results])

        print(f"\n  N={N} (deg={mean_deg:.1f}):", flush=True)
        print(f"  {'func':>10} {'<Lf>':>10} {'<Lf>/deg':>10} "
              f"{'Box*a':>8} {'Del*a':>8} {'Spa*a':>8} {'best':>6}", flush=True)

        for fn in const_fns:
            mean_Lf = np.mean([r["results"][fn]["mean_Lf"] for r in results])
            ratio = mean_Lf / mean_deg if mean_deg > 0 else 0

            # Which operator, when multiplied by the same alpha, gives the right ratio?
            # If L_link f ~ alpha * Op f, then <Lf> = alpha * (Op f)
            # and alpha = <Lf> / (Op f) for each Op.
            # The consistent alpha across all test functions identifies the operator.
            alphas = {}
            for op in ["box", "delta", "spatial"]:
                v = op_vals[fn][op]
                if abs(v) > 0.01:
                    alphas[op] = mean_Lf / v
                else:
                    alphas[op] = float('nan')

            best = min(["box", "delta", "spatial"],
                       key=lambda op: abs(alphas.get(op, 1e10) - ratio) if not np.isnan(alphas.get(op, float('nan'))) else 1e10)

            print(f"  {fn:>10} {mean_Lf:+10.4f} {ratio:+10.6f} "
                  f"{alphas.get('box', float('nan')):+8.4f} "
                  f"{alphas.get('delta', float('nan')):+8.4f} "
                  f"{alphas.get('spatial', float('nan')):+8.4f}", flush=True)

    # Cross-consistency: compute alpha = <Lf>/(Op f) for each operator across all fns
    print(f"\n--- CROSS-CONSISTENCY: alpha = <Lf>/(Op f) ---", flush=True)
    print(f"  If L_link = alpha * Op, then alpha should be CONSTANT across all test functions.", flush=True)

    for N in N_VALUES:
        results = all_results[N]
        print(f"\n  N={N}:", flush=True)
        print(f"  {'func':>10} {'alpha_box':>10} {'alpha_del':>10} {'alpha_spa':>10}", flush=True)
        for fn in const_fns:
            mean_Lf = np.mean([r["results"][fn]["mean_Lf"] for r in results])
            row = f"  {fn:>10}"
            for op in ["box", "delta", "spatial"]:
                v = op_vals[fn][op]
                if abs(v) > 0.01:
                    alpha = mean_Lf / v
                    row += f" {alpha:+10.4f}"
                else:
                    row += f" {'n/a':>10}"
            print(row, flush=True)

    # Position-dependent test functions: use correlation
    print(f"\n--- POSITION-DEPENDENT TEST FUNCTIONS ---", flush=True)
    print(f"  For these, Op f varies spatially. We report r(L_link f, Op f).", flush=True)

    pos_fns = ["cos(pi*t)", "cos(pi*x)", "gaussian"]

    for N in N_VALUES:
        results = all_results[N]
        print(f"\n  N={N}:", flush=True)
        print(f"  {'func':>12} {'r_box':>8} {'r_delta':>8} {'r_spatial':>10} "
              f"{'a_box':>8} {'a_delta':>8} {'a_spatial':>10} {'best':>8}", flush=True)

        for fn in pos_fns:
            r_b = np.mean([r["results"][fn]["r_box"] for r in results])
            r_d = np.mean([r["results"][fn]["r_delta"] for r in results])
            r_s = np.mean([r["results"][fn]["r_spatial"] for r in results])
            a_b = np.mean([r["results"][fn]["alpha_box"] for r in results])
            a_d = np.mean([r["results"][fn]["alpha_delta"] for r in results])
            a_s = np.mean([r["results"][fn]["alpha_spatial"] for r in results])

            rs = {"box": r_b, "delta": r_d, "spatial": r_s}
            best = max(rs, key=lambda k: abs(rs[k]))

            print(f"  {fn:>12} {r_b:+8.4f} {r_d:+8.4f} {r_s:+10.4f} "
                  f"{a_b:+8.4f} {a_d:+8.4f} {a_s:+10.4f} {best:>8}", flush=True)

    # ==================================================================
    # HARMONIC TEST: L_link should give ~0 for harmonic functions
    # ==================================================================

    print(f"\n--- HARMONIC TEST ---", flush=True)
    print(f"  For x^2-y^2 and t*x (harmonic: Box=Delta=Spatial=0),", flush=True)
    print(f"  L_link f should be ~0 (up to boundary/discretization effects).", flush=True)

    harmonic_fns = ["x^2-y^2", "t*x"]
    for N in N_VALUES:
        results = all_results[N]
        print(f"\n  N={N}:", flush=True)
        for fn in harmonic_fns:
            mean_Lf = np.mean([r["results"][fn]["mean_Lf"] for r in results])
            std_Lf = np.mean([r["results"][fn]["std_Lf"] for r in results])
            # Compare with non-harmonic function to see relative magnitude
            mean_Lf_t2 = abs(np.mean([r["results"]["t^2"]["mean_Lf"] for r in results]))
            ratio = abs(mean_Lf) / mean_Lf_t2 if mean_Lf_t2 > 0 else 0
            print(f"    {fn}: <Lf>={mean_Lf:+.6f}, std={std_Lf:.4f}, "
                  f"|<Lf>|/|<Lf(t^2)>|={ratio:.4f}", flush=True)

    # ==================================================================
    # FINITE-SIZE SCALING: does alpha converge?
    # ==================================================================

    print(f"\n{'='*70}", flush=True)
    print("FINITE-SIZE SCALING: alpha = <Lf>/(Op f) vs N", flush=True)
    print("=" * 70, flush=True)

    # Use f = t^2-r^2 as the most discriminating function
    # Box(t^2-r^2) = -8, Delta = -4, Spatial = -6
    ref_fn = "t^2-r^2"
    print(f"\n  Reference: f = {ref_fn}", flush=True)
    print(f"  {'N':>6} {'deg':>6} {'<Lf>':>10} {'alpha_box':>10} {'alpha_del':>10} {'alpha_spa':>10}", flush=True)
    for N in N_VALUES:
        results = all_results[N]
        mean_deg = np.mean([r["mean_degree"] for r in results])
        mean_Lf = np.mean([r["results"][ref_fn]["mean_Lf"] for r in results])
        a_box = mean_Lf / (-8.0)
        a_del = mean_Lf / (-4.0)
        a_spa = mean_Lf / (-6.0)
        print(f"  {N:6d} {mean_deg:6.1f} {mean_Lf:+10.4f} "
              f"{a_box:+10.4f} {a_del:+10.4f} {a_spa:+10.4f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    # Determine which operator by cross-consistency of alpha
    # Using the largest N for final answer
    N_final = max(N_VALUES)
    results = all_results[N_final]

    alphas_by_op = {"box": [], "delta": [], "spatial": []}
    for fn in const_fns:
        mean_Lf = np.mean([r["results"][fn]["mean_Lf"] for r in results])
        for op in ["box", "delta", "spatial"]:
            v = op_vals[fn][op]
            if abs(v) > 0.01:
                alphas_by_op[op].append(mean_Lf / v)

    print(f"\n  Cross-consistency of alpha at N={N_final}:", flush=True)
    for op in ["box", "delta", "spatial"]:
        vals = alphas_by_op[op]
        if len(vals) >= 2:
            mean_a = np.mean(vals)
            std_a = np.std(vals)
            cv = std_a / abs(mean_a) if abs(mean_a) > 0 else float('inf')
            print(f"    {op:>8}: alpha = {mean_a:+.4f} +/- {std_a:.4f} "
                  f"(CV = {cv:.3f}, n = {len(vals)} functions)", flush=True)
        else:
            print(f"    {op:>8}: insufficient data", flush=True)

    # The operator with smallest CV (most consistent alpha) is the winner
    best_op = min(alphas_by_op, key=lambda op:
                  np.std(alphas_by_op[op]) / abs(np.mean(alphas_by_op[op]))
                  if len(alphas_by_op[op]) >= 2 and abs(np.mean(alphas_by_op[op])) > 0
                  else float('inf'))
    best_alpha = np.mean(alphas_by_op[best_op])
    best_cv = np.std(alphas_by_op[best_op]) / abs(best_alpha) if abs(best_alpha) > 0 else float('inf')

    verdict = (f"L_link converges to alpha * {best_op.upper()} with "
               f"alpha = {best_alpha:+.4f} (CV = {best_cv:.3f})")

    print(f"\n  {verdict}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    meta = ExperimentMeta(route=2, name="continuum_limit",
                          description="Numerical identification of L_link continuum limit operator",
                          N=N_final, M=M_SEEDS, status="completed", verdict=verdict)
    meta.wall_time_sec = total_time

    output = {"verdict": verdict, "best_op": best_op, "best_alpha": best_alpha,
              "best_cv": best_cv, "wall_time_sec": total_time}
    save_experiment(meta, output, RESULTS_DIR / "continuum_limit.json")
    print(f"  Saved: {RESULTS_DIR / 'continuum_limit.json'}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
