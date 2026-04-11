"""
FND-1 Research Pipeline
========================

WHAT THIS CODE DOES (plain language)
-------------------------------------
Einstein showed that gravity is the curvature of spacetime.  There is a
theory (causal set theory) proposing that spacetime is not smooth but made
of discrete points, like pixels.  Between these points there are
cause-and-effect links: "this event can influence that one."

From these links you can build a matrix (the Benincasa-Dowker operator, L).
It is already known that L can recover one type of curvature -- the Ricci
scalar R (Benincasa & Dowker, PRL 2010, arXiv:1001.2725).  But R = 0 for
many important spacetimes: black holes, gravitational waves, etc.  Their
curvature is encoded in the Weyl tensor, which L alone does not see.

This code tests a new idea: instead of looking at L itself, look at how
L interacts with its transpose L^T.  L encodes "influence from the past"
(it is lower-triangular); L^T encodes "influence from the future."
In flat spacetime, past and future are statistically symmetric, so
L^T L and L L^T are nearly equal.  Curvature breaks this symmetry.

The observable:  [H,M] = (L^T L - L L^T) / 2.
This is a standard measure of matrix non-normality called the Henrici
departure (Trefethen & Embree, "Spectra and Pseudospectra", 2005).
We compute its eigenvalues and check whether they change when the metric
changes from flat to curved.

The experiment: sprinkle the same random points twice -- once compute
the links assuming flat spacetime, once assuming curved.  Same points,
same density.  Any difference in the eigenvalues must come from the
change in the causal condition (the geometry), not from density noise.
This technique is called Common Random Numbers (CRN) and is standard in
simulation (Law & Kelton, "Simulation Modeling and Analysis"), but has
not been used on causal sets before.

EXPERIMENTAL DESIGN
-------------------
Common Random Numbers (CRN): for each trial, sprinkle N points once, then
compute the causal matrix TWICE — once with flat Minkowski metric, once
with the curved test metric.  Same points → same density fluctuations →
any difference in observables is due to the causal condition change, not
density.  This is standard variance-reduction (Law & Kelton, "Simulation
Modeling and Analysis"), applied here for the first time to causal sets.

PRE-REGISTERED PRIMARY HYPOTHESIS:
    H0: mean(frobenius_delta) = 0 for ppwave_quad at eps=10, N=30000, M=80.
    H1: mean(frobenius_delta) != 0.
    One-sample paired t-test, alpha=0.01, two-sided.
    All other tests are EXPLORATORY.

    Note: frobenius was selected based on EXP-14 results (strongest
    observable at N>=5000).  This is conditional pre-registration, not
    blind pre-registration.

METRICS (5 test spacetimes)
---------------------------
Each metric deforms flat spacetime in a different way.  The deformation
strength is controlled by a parameter eps.  At eps=0, everything is flat.

  1. PP-wave quadrupole  (f = x^2 - y^2)
     A gravitational wave passing through spacetime.  Think of it as a
     ripple that stretches space in the x-direction and squeezes it in y.
     This is a vacuum solution (R = 0) -- no matter, pure geometry.
     The standard curvature detector (BD action ~ integral of R) gives
     zero here.  If our commutator gives nonzero -- it sees something
     the BD action cannot.
     Reference: Brinkmann 1925; exact solution of Einstein's equations.

  2. PP-wave cross  (f = x * y)
     Same type of gravitational wave, rotated 45 degrees.  If the
     commutator gives a similar signal -- the result does not depend
     on the specific shape of the wave, just on its strength.

  3. Weak Schwarzschild  (Phi = -eps / (r + 0.3))
     The gravity of a mass at the centre -- like the Sun's gravity but
     very weak (linearised).  Unlike the pp-wave, this has a nonzero
     C^2 (a specific curvature measure).  Tests a fundamentally
     different type of curvature.  The regularisation r+0.3 softens
     the singularity at the centre; this introduces a small amount of
     Ricci curvature near r=0, but most of the volume is vacuum.
     Reference: Poisson, "A Relativist's Toolkit", section 1.5.

  4. FLRW conformally flat control  (a(t) = 1 + eps * t^2)
     An expanding universe (like our cosmos, simplified).  This spacetime
     is "conformally flat" -- its causal structure (which events can
     influence which) is the same as flat spacetime, just rescaled.
     Prediction: the commutator signal should be WEAKER than pp-wave,
     because the causal structure barely changes.  If confirmed -- our
     method is specifically sensitive to Weyl curvature, not just any
     metric change.
     Reference: Malament 1977 -- causal structure determines the
     conformal class of the metric.

  5. Conformal null test
     Flat spacetime with a conformal rescaling g = Omega^2 * eta.
     The causal condition is identical to flat -- by construction,
     the difference must be exactly zero.  This is a code validation
     check, not a physics test.  If we get nonzero -- there is a bug.

CODE STRUCTURE
--------------
Single file, flat functions, no classes.  Reasons:
  - Runs on any machine with numpy + scipy.  No project-level imports.
  - A collaborator can read it top-to-bottom without jumping between files.
  - Easy to profile: python -m cProfile fnd1_gcp_pipeline.py --calibrate

The commutator matrix is stored as a dense (not sparse) array, because
benchmarking showed it is ~80% filled at N >= 10000.  Sparse format
actually uses MORE memory for near-dense matrices due to index overhead.

Checkpoints are saved every 5 sprinklings in case the machine is
interrupted (cloud spot VMs can be killed at any moment).  The save
uses atomic write (write to .tmp, then rename) to avoid corruption
if killed mid-write.  Re-running the same command skips finished work.

WHAT TAKES THE MOST TIME
-------------------------
73% of the runtime is numpy.linalg.eigvalsh -- computing all eigenvalues
of a dense symmetric matrix.  This uses LAPACK (dsyevd algorithm), which
is the fastest known method for this problem.  No known algorithm
computes all eigenvalues of a dense symmetric matrix faster than O(N^3).  The only way to finish sooner is to run
many independent tasks in parallel on multiple CPU cores or machines.

GPU does not help: eigvalsh requires double-precision (FP64) arithmetic,
and consumer GPUs (RTX 3090 Ti, 4060, 4070) have very low FP64
throughput (~1/64 of their advertised FP32 speed).  Benchmarked:
GPU eigvalsh was SLOWER than CPU on the same machine.

Ray distributes independent tasks across machines for linear speedup:
2 machines = 2x faster, 3 machines = 3x faster.

USAGE
-----
    # Calibrate (measure speed on this machine):
    python fnd1_gcp_pipeline.py --calibrate

    # Run one metric:
    python fnd1_gcp_pipeline.py --metric ppwave_quad --N 10000 --M 80

    # Run all 5 metrics:
    python fnd1_gcp_pipeline.py --all --N 10000 --M 80

    # Test T-independence:
    python fnd1_gcp_pipeline.py --metric ppwave_quad --N 10000 --T 0.5
    python fnd1_gcp_pipeline.py --metric ppwave_quad --N 10000 --T 2.0

    # With GCS backup (for spot VMs):
    python fnd1_gcp_pipeline.py --all --N 30000 --gcs my-bucket-name
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy import stats

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
GPU_OK = False
try:
    import cupy as cp
    cp.linalg.eigvalsh(cp.eye(2, dtype=cp.float64))  # test cusolver
    GPU_OK = True
except Exception:
    cp = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
T_DIAMOND = 1.0  # default; use --T flag to test T-independence
MASTER_SEED = 31415
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

METRICS = {
    "ppwave_quad":  {"label": "PP-wave x²-y²",       "eps": [0.0, 2.0, 5.0, 10.0]},       # eps≤10: linear regime
    "ppwave_cross": {"label": "PP-wave xy",           "eps": [0.0, 2.0, 5.0, 10.0]},       # eps≤10: linear regime
    "schwarzschild":{"label": "Weak Schwarzschild (distributed Weyl + localised Ricci near r=0)",
                     "eps": [0.0, 0.005, 0.01, 0.02]},   # |Φ|_max = eps/0.3 ≤ 0.067
    "flrw":         {"label": "FLRW conformally flat CONTROL (predict: delta suppressed vs pp-wave)",
                     "eps": [0.0, 0.5, 1.0, 2.0, 5.0]},
    "conformal":    {"label": "Conformal (NULL: must give exactly 0)",
                     "eps": [0.0, 5.0, 10.0]},  # fewer eps (all give 0 anyway)
}

# Deterministic metric→seed mapping (BUG FIX: hash() is non-deterministic)
METRIC_SEED_OFFSETS = {
    "ppwave_quad": 100, "ppwave_cross": 200, "schwarzschild": 300,
    "flrw": 400, "conformal": 500,
}

# ---------------------------------------------------------------------------
# Sprinkling
# ---------------------------------------------------------------------------
def sprinkle(N: int, T: float, rng) -> np.ndarray:
    """Poisson sprinkle into 4D Alexandrov interval (causal diamond).

    Rejection sampling: generate uniform points in a 4D cube, keep those
    inside the diamond {|t| + |r| < T/2}.  Acceptance rate ~13% in 4D.
    Points sorted by time coordinate (needed for causal ordering).

    Why rejection (not inverse CDF): the 4D diamond has no simple closed-form
    inverse CDF.  Rejection is standard in the causal set literature
    (Bombelli-Lee-Meyer-Sorkin 1987).
    """
    pts = np.empty((N, 4))
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count+n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


# ---------------------------------------------------------------------------
# Causal conditions (one per metric)
# ---------------------------------------------------------------------------
def _pairwise(pts):
    """Common pairwise differences (memory-optimised)."""
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    return t, x, y, z, dt, dx, dy, dz, dr2


def causal_flat(pts, _eps=0.0) -> np.ndarray:
    """Minkowski causal condition."""
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    del dx, dy, dz
    C = ((dt**2 > dr2) & (dt > 0)).astype(np.float64)
    del dt, dr2
    return C


def causal_ppwave_quad(pts, eps) -> np.ndarray:
    """PP-wave with quadrupole profile f = x² - y²."""
    t, x, y, z, dt, dx, dy, dz, dr2 = _pairwise(pts)
    del dx, dy
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm**2 - ym**2
    del xm, ym
    mink = dt**2 - dr2; del dr2
    corr = eps * f * (dt + dz)**2 / 2.0; del f
    C = ((mink > corr) & (dt > 0)).astype(np.float64)
    del dt, dz, mink, corr
    return C


def causal_ppwave_cross(pts, eps) -> np.ndarray:
    """PP-wave with cross profile f = x·y (harmonic, ∇²f=0)."""
    t, x, y, z, dt, dx, dy, dz, dr2 = _pairwise(pts)
    del dx, dy
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm * ym
    del xm, ym
    mink = dt**2 - dr2; del dr2
    corr = eps * f * (dt + dz)**2 / 2.0; del f
    C = ((mink > corr) & (dt > 0)).astype(np.float64)
    del dt, dz, mink, corr
    return C


def causal_schwarzschild(pts, eps) -> np.ndarray:
    """Weak-field Schwarzschild: Φ = -eps/(r+0.3).
    ds² = -(1+2Φ)dt² + (1-2Φ)dr².  Approximately vacuum for r >> 0.3."""
    t, x, y, z, dt, dx, dy, dz, dr2 = _pairwise(pts)
    del dx, dy, dz
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    zm = (z[np.newaxis, :] + z[:, np.newaxis]) / 2.0
    rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3  # larger regularisation → R≈0 in diamond interior
    del xm, ym, zm
    Phi = -eps / rm; del rm
    # (1+2Φ)dt² > (1-2Φ)dr²
    C = (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.float64)
    del dt, dr2, Phi
    return C


def causal_flrw(pts, eps) -> np.ndarray:
    """FLRW-like: a(t) = 1 + eps·t².
    ds² = -dt² + a²(t_m)·dr².  Conformally flat."""
    t, x, y, z, dt, dx, dy, dz, dr2 = _pairwise(pts)
    del dx, dy, dz
    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2.0
    a2 = (1 + eps * tm**2)**2; del tm
    C = ((dt**2 > a2 * dr2) & (dt > 0)).astype(np.float64)
    del dt, dr2, a2
    return C


def causal_conformal(pts, eps) -> np.ndarray:
    """Pure conformal: g = Ω²·η. Same causal condition as flat.
    NULL TEST: CRN difference must be exactly 0."""
    return causal_flat(pts, 0.0)


METRIC_FNS = {
    "ppwave_quad":   causal_ppwave_quad,
    "ppwave_cross":  causal_ppwave_cross,
    "schwarzschild": causal_schwarzschild,
    "flrw":          causal_flrw,
    "conformal":     causal_conformal,
}


# ---------------------------------------------------------------------------
# BD operator and GTA
# ---------------------------------------------------------------------------
def _build_layers(C: np.ndarray, N: int, T: float):
    """Extract layer counts and build sparse layer data.

    For each causally related pair (i,j) where j≺i, compute the "layer"
    = number of elements between j and i in the causal order.  This is
    (C @ C)[j,i] where C is the causal matrix.

    Layers 0-3 are used by the Benincasa-Dowker d'Alembertian (Dowker &
    Glaser 2013, arXiv:1305.2588).  Layer 0 = links (nearest causal
    neighbors), layer k = k elements between.

    Returns (layer_data, rho, layer_counts, bd_action):
      layer_data[k] = (row_indices, col_indices) for pairs in layer k
      rho = N/V = sprinkling density
      layer_counts[k] = number of pairs in layer k
      bd_action = (-4N + 4*N0 - 36*N1 + 64*N2 - 32*N3) / sqrt(6)

    Why .toarray() for n_arr:
      C_sp @ C_sp is sparse but nearly as dense as C (~5%).
      Element-wise comparison (n_int == k) is faster on dense arrays.
      Memory: N^2 × 8 bytes (7.2 GB at N=30k).  Acceptable on 64 GB machines.

    Why past = C.T.copy():
      C.T is a VIEW (shares memory with C).  Without .copy(), del C
      cannot free the array because the view holds a reference.
      With .copy(), del C frees 7.2 GB immediately.
    """
    V = np.pi * T**4 / 24.0
    rho = N / V

    C_sp = sp.csr_matrix(C)
    n_arr = (C_sp @ C_sp).toarray()
    past = C.T.copy()   # .copy() to allow freeing C
    n_past = n_arr.T.copy()
    del C, C_sp, n_arr; gc.collect()

    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5
    del past

    layer_data = {}
    layer_counts = {}
    for k in range(4):
        mask = causal_mask & (n_int == k)
        r, c = np.nonzero(mask)
        layer_data[k] = (r, c)
        layer_counts[k] = len(r)

    # BD action: S = (-4N + 4N0 - 36N1 + 64N2 - 32N3) / sqrt(6)
    bd_action = (
        -4 * N
        + 4 * layer_counts[0]
        - 36 * layer_counts[1]
        + 64 * layer_counts[2]
        - 32 * layer_counts[3]
    ) / np.sqrt(6.0)

    del causal_mask, n_int; gc.collect()
    return layer_data, rho, layer_counts, float(bd_action)


def _build_L_sparse(layer_data: dict, rho: float, N: int,
                    coeffs: dict | None = None) -> sp.csr_matrix:
    """Build sparse BD operator from layer data with given coefficients."""
    if coeffs is None:
        coeffs = {0: 4.0, 1: -36.0, 2: 64.0, 3: -32.0}
    scale = np.sqrt(rho)
    rows, cols, vals = [], [], []
    for k, coeff in coeffs.items():
        if k not in layer_data:
            continue
        r, c = layer_data[k]
        rows.append(r); cols.append(c)
        vals.append(np.full(len(r), coeff * scale))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def build_gta_observables(C: np.ndarray, N: int, T: float, mode: str = "dense"):
    """Build BD operator L and compute GTA observables + BD action.

    Returns dict with: entropy, frobenius, max_abs, bd_action.
    mode="dense": full eigvalsh → all eigenvalue observables.
    mode="matfree": matrix-free eigsh(k=10) + stochastic Frobenius.
    """
    layer_data, rho, layer_counts, bd_action = _build_layers(C, N, T)

    # BD operator with standard coefficients
    L_sp = _build_L_sparse(layer_data, rho, N)
    del layer_data; gc.collect()

    if mode == "dense":
        obs = _gta_dense(L_sp, N)
    else:
        obs = _gta_matfree(L_sp, N)

    obs["bd_action"] = bd_action
    obs["layer_counts"] = {str(k): int(v) for k, v in layer_counts.items()}
    return obs


def build_gta_shuffled_layers(C: np.ndarray, N: int, T: float,
                              rand_seed: int = 456,
                              precomputed_layers=None) -> dict:
    """Shuffle layer assignments among causal pairs, keeping BD coefficients.

    THE key null test: if detection survives shuffling, then the specific
    layer→coefficient mapping is irrelevant and the signal comes from the
    causal structure alone, not from the BD operator's geometric content.

    If detection FAILS after shuffling → the layer structure IS meaningful
    → the BD operator encodes genuine geometric information.
    """
    if precomputed_layers is not None:
        layer_data, rho, layer_counts = precomputed_layers
    else:
        layer_data, rho, layer_counts, _ = _build_layers(C, N, T)

    # Collect all causal pairs from layers 0-3
    all_rows = np.concatenate([layer_data[k][0] for k in range(4)])
    all_cols = np.concatenate([layer_data[k][1] for k in range(4)])

    # True layer labels for each pair
    true_labels = np.concatenate([
        np.full(layer_counts[k], k, dtype=int) for k in range(4)
    ])

    # Shuffle labels (break layer→coefficient mapping)
    rng = np.random.default_rng(rand_seed)
    shuffled = true_labels.copy()
    rng.shuffle(shuffled)

    # Build L with shuffled assignments but same BD coefficients
    bd_coeffs = {0: 4.0, 1: -36.0, 2: 64.0, 3: -32.0}
    scale = np.sqrt(rho)
    vals = np.array([bd_coeffs[int(k)] * scale for k in shuffled])

    L_shuf = sp.csr_matrix((vals, (all_rows, all_cols)), shape=(N, N))
    del layer_data; gc.collect()
    return _gta_dense(L_shuf, N)


def build_gta_random_causal(C: np.ndarray, N: int, T: float,
                            rand_seed: int = 789,
                            precomputed_layers=None) -> dict:
    """Random weights on causal pairs, ignoring layers entirely.

    Tests whether RAW causal structure (without any layer information)
    is sufficient for curvature detection.  If yes → layers are
    redundant.  If no → layer structure adds information.
    """
    if precomputed_layers is not None:
        layer_data, rho, layer_counts = precomputed_layers
    else:
        layer_data, rho, layer_counts, _ = _build_layers(C, N, T)

    all_rows = np.concatenate([layer_data[k][0] for k in range(4)])
    all_cols = np.concatenate([layer_data[k][1] for k in range(4)])

    rng = np.random.default_rng(rand_seed)
    scale = np.sqrt(rho)
    bd_norm = np.sqrt(4**2 + 36**2 + 64**2 + 32**2)
    # Random weights with same overall magnitude as BD
    vals = rng.standard_normal(len(all_rows)) * scale * bd_norm / 2.0

    L_rand = sp.csr_matrix((vals, (all_rows, all_cols)), shape=(N, N))
    del layer_data; gc.collect()
    return _gta_dense(L_rand, N)


def build_gta_random_operator(C: np.ndarray, N: int, T: float,
                              rand_seed: int = 123, n_random: int | None = None,
                              precomputed_layers=None) -> dict:
    """Build GTA with RANDOM coefficients instead of BD.

    Null test: if random coefficients detect curvature equally well,
    then the BD structure is irrelevant and the result is trivial.

    Tests n_random different random coefficient sets, all normalised to
    have the same L2 norm as BD coefficients {4,-36,64,-32}.
    """
    if precomputed_layers is not None:
        layer_data, rho, layer_counts = precomputed_layers
    else:
        layer_data, rho, _, _ = _build_layers(C, N, T)

    # Adaptive n_random: expensive eigvalsh at large N → fewer random sets
    if n_random is None:
        n_random = 20 if N <= 5000 else 10 if N <= 15000 else 5

    bd_norm = np.sqrt(4**2 + 36**2 + 64**2 + 32**2)  # ≈ 78.4
    rng = np.random.default_rng(rand_seed)

    bd_signs = [+1, -1, +1, -1]  # BD sign structure

    results_full_random = []    # fully random coefficients
    results_sign_preserved = [] # same signs as BD, random magnitudes

    for _ in range(n_random):
        # Type A: fully random (normalised)
        raw = {k: float(rng.standard_normal()) for k in range(4)}
        raw_norm = np.sqrt(sum(v**2 for v in raw.values()))
        coeffs_a = {k: v * bd_norm / (raw_norm + 1e-10) for k, v in raw.items()}
        L_a = _build_L_sparse(layer_data, rho, N, coeffs=coeffs_a)
        obs_a = _gta_dense(L_a, N)
        results_full_random.append({
            "frobenius": obs_a["frobenius"], "max_abs": obs_a["max_abs"],
            "coeffs": {str(k): round(v, 4) for k, v in coeffs_a.items()},
        })

        # Type B: same sign structure as BD, random magnitudes
        raw_mag = {k: abs(float(rng.standard_normal())) for k in range(4)}
        raw_mag_norm = np.sqrt(sum(v**2 for v in raw_mag.values()))
        coeffs_b = {k: bd_signs[k] * v * bd_norm / (raw_mag_norm + 1e-10)
                    for k, v in raw_mag.items()}
        L_b = _build_L_sparse(layer_data, rho, N, coeffs=coeffs_b)
        obs_b = _gta_dense(L_b, N)
        results_sign_preserved.append({
            "frobenius": obs_b["frobenius"], "max_abs": obs_b["max_abs"],
            "coeffs": {str(k): round(v, 4) for k, v in coeffs_b.items()},
        })

    del layer_data; gc.collect()

    return {
        "frobenius": float(np.mean([r["frobenius"] for r in results_full_random])),
        "max_abs": float(np.mean([r["max_abs"] for r in results_full_random])),
        "full_random": results_full_random,
        "sign_preserved": results_sign_preserved,
        "n_random": n_random,
    }


def _gta_dense(L_sp, N):
    """Dense eigvalsh — full spectrum, all observables.

    Why dense (not sparse eigsh):
      The commutator L^TL-LL^T is ~80% dense at N>=10k (benchmarked).
      Sparse eigsh is SLOWER than dense eigvalsh for near-dense matrices
      because sparse format adds index overhead without reducing flops.

    Why eigvalsh (not eigh):
      We need eigenvalues only, not eigenvectors.  eigvalsh is faster.

    Why symmetrize (comm + comm.T)/2:
      Floating-point errors in sparse matmul break exact symmetry.
      eigvalsh requires symmetric input; without this, it can return
      complex eigenvalues or fail to converge.
    """
    comm = (L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0
    comm = ((comm + comm.T) / 2.0).toarray()
    del L_sp; gc.collect()

    if GPU_OK:
        comm_gpu = cp.asarray(comm)
        evals = cp.asnumpy(cp.linalg.eigvalsh(comm_gpu))
        del comm_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        evals = np.linalg.eigvalsh(comm)
    del comm; gc.collect()

    a = np.abs(evals)
    s = float(np.sum(a))
    return {
        "entropy":   float(-np.sum((a/s)*np.log(a/s + 1e-300))) if s > 0 else 0.0,
        "frobenius": float(np.sqrt(np.sum(evals**2))),
        "max_abs":   float(np.max(a)),
    }


def _gta_matfree(L_sp, N, k_eig=10, m_stoch=200):
    """Matrix-free: top-k eigsh + stochastic Frobenius.

    Memory: only L_sp (~2.4 GB at N=200k), no N×N dense matrix.
    Trades accuracy for scalability: only top-k eigenvalues (not all N),
    and stochastic Frobenius estimate (not exact).

    Why matrix-free:
      At N>50k, the dense commutator doesn't fit in RAM (N=100k → 80 GB).
      Matrix-free computes comm @ v = (L^T(Lv) - L(L^Tv))/2 using only L.

    Why fixed seed for stochastic vectors (rng seed=0):
      CRN variance reduction.  Same random vectors for flat and curved →
      stochastic errors are CORRELATED → cancel in the paired difference.
      This makes the Frobenius DELTA more accurate than either absolute value.

    Why m=200 stochastic vectors:
      Hutchinson estimator: E[||Av||^2] = tr(A^2) = ||A||_F^2.
      Relative error ~ sqrt(2/m) ≈ 10% for m=200.  Acceptable for detection
      (paired t-test), not for precision measurement.
    """
    def matvec(v):
        Lv = L_sp @ v
        LTv = L_sp.T @ v
        return (L_sp.T @ Lv - L_sp @ LTv) / 2.0

    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)

    # Top-k eigenvalues
    k_actual = min(k_eig, N - 2)
    evals = eigsh(op, k=k_actual, which="LM", return_eigenvectors=False)
    max_abs = float(np.max(np.abs(evals)))

    # Stochastic Frobenius: tr(comm²) ≈ (N/m) Σ ||comm·v_i||²
    rng = np.random.default_rng(0)  # fixed seed for CRN variance reduction
    frob2 = 0.0
    for _ in range(m_stoch):
        v = rng.standard_normal(N)
        Av = matvec(v)
        frob2 += float(np.dot(Av, Av))
    frob2 /= m_stoch  # unbiased estimator of tr(A²)/1 (not /N)
    frobenius = float(np.sqrt(max(frob2, 0.0)))

    return {
        "entropy":   0.0,  # not available in matfree mode
        "frobenius": frobenius,
        "max_abs":   max_abs,
    }


# ---------------------------------------------------------------------------
# CRN paired worker
# ---------------------------------------------------------------------------
def crn_one(seed: int, N: int, T: float, eps: float,
            metric_fn, mode: str = "dense",
            include_random_operator: bool = False) -> dict:
    """One CRN paired sprinkling: flat vs curved, same points.

    This is the ATOMIC UNIT of computation.  Each call is independent —
    no shared state, no inter-call communication.  Ray distributes these
    across machines.  ~75 seconds at N=10k on i9-12900KS (CPU).

    Flow:
      1. Sprinkle N points into 4D diamond (same for flat and curved)
      2. Build flat causal matrix → compute GTA observables
      3. Build curved causal matrix (same points!) → compute GTA observables
      4. Return paired differences (curved - flat)

    Why paired differences (not absolute values):
      CRN design: same points → same density fluctuations → delta isolates
      the effect of the metric change.  This cancels the dominant source of
      variance (random sprinkling noise), giving much smaller error bars.

    Why include_random_operator is optional:
      Null operator tests are 4× more expensive (shuffled + random_causal +
      random_coefficients, each requiring eigvalsh).  Run on a subset of
      sprinklings (first 20 at the two highest eps values) to save time.
    """
    rng = np.random.default_rng(seed)
    pts = sprinkle(N, T, rng)

    # Flat
    C_flat = causal_flat(pts)
    tc_flat = float(np.sum(C_flat))
    if include_random_operator:
        C_flat_copy = C_flat.copy()
    gta_flat = build_gta_observables(C_flat, N, T, mode)  # consumes C_flat

    # Curved
    C_curved = metric_fn(pts, eps)
    tc_curved = float(np.sum(C_curved))
    if include_random_operator:
        C_curved_copy = C_curved.copy()
    gta_curved = build_gta_observables(C_curved, N, T, mode)  # consumes C_curved

    result = {"seed": seed, "N": N, "eps": eps}

    # BD commutator observables
    for key in ["entropy", "frobenius", "max_abs"]:
        result[f"{key}_flat"] = gta_flat[key]
        result[f"{key}_curved"] = gta_curved[key]
        result[f"{key}_delta"] = gta_curved[key] - gta_flat[key]

    # BD action (scalar — proven to converge to ∫R√g)
    result["bd_action_flat"] = gta_flat["bd_action"]
    result["bd_action_curved"] = gta_curved["bd_action"]
    result["bd_action_delta"] = gta_curved["bd_action"] - gta_flat["bd_action"]

    # TC
    result["tc_flat"] = tc_flat
    result["tc_curved"] = tc_curved
    result["tc_delta_pct"] = (tc_curved - tc_flat) / max(tc_flat, 1) * 100

    # Layer counts (free diagnostic: is commutator info beyond layer counts?)
    for k in ["0", "1", "2", "3"]:
        lf = gta_flat.get("layer_counts", {}).get(k, 0)
        lc = gta_curved.get("layer_counts", {}).get(k, 0)
        result[f"layer{k}_flat"] = lf
        result[f"layer{k}_curved"] = lc
        result[f"layer{k}_delta"] = lc - lf

    # Null operator tests (expensive — only run for subset)
    # NOTE: _build_layers(C,...) does del C on its LOCAL ref only.
    # The caller's copy survives — no .copy() needed between calls.
    # Precompute layers ONCE per C to avoid 8× recomputation of C@C.
    if include_random_operator:
        ld_flat, rho_f, lc_flat, _ = _build_layers(C_flat_copy, N, T)
        pre_flat = (ld_flat, rho_f, lc_flat)
        ld_curv, rho_c, lc_curv, _ = _build_layers(C_curved_copy, N, T)
        pre_curv = (ld_curv, rho_c, lc_curv)

        # Level 1: Shuffled layers (BD coeffs, randomised layer mapping)
        shuf_flat = build_gta_shuffled_layers(None, N, T, precomputed_layers=pre_flat)
        shuf_curved = build_gta_shuffled_layers(None, N, T, precomputed_layers=pre_curv)
        result["shuf_frobenius_delta"] = shuf_curved["frobenius"] - shuf_flat["frobenius"]

        # Level 2: Random causal (random weights, no layers)
        rcaus_flat = build_gta_random_causal(None, N, T, precomputed_layers=pre_flat)
        rcaus_curved = build_gta_random_causal(None, N, T, precomputed_layers=pre_curv)
        result["rcaus_frobenius_delta"] = rcaus_curved["frobenius"] - rcaus_flat["frobenius"]

        # Level 3: Random layer coefficients (within layer family)
        rand_flat = build_gta_random_operator(None, N, T, precomputed_layers=pre_flat)
        rand_curved = build_gta_random_operator(None, N, T, precomputed_layers=pre_curv)
        result["rand_frobenius_delta"] = rand_curved["frobenius"] - rand_flat["frobenius"]
        result["rand_sign_frob_delta"] = (
            float(np.mean([r["frobenius"] for r in rand_curved["sign_preserved"]]))
            - float(np.mean([r["frobenius"] for r in rand_flat["sign_preserved"]]))
        )
        result["rand_n_random"] = rand_flat["n_random"]
        del C_flat_copy, C_curved_copy, ld_flat, ld_curv
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def ckpt_path(metric: str, N: int, eps: float) -> Path:
    return CHECKPOINT_DIR / f"ckpt_{metric}_N{N}_eps{eps}.json"


def save_ckpt(data: list[dict], metric: str, N: int, eps: float):
    """Atomic checkpoint save: write to .tmp then rename (safe on spot kill)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = ckpt_path(metric, N, eps)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    tmp.replace(p)  # atomic on Linux (GCP)


def load_ckpt(metric: str, N: int, eps: float) -> list[dict]:
    p = ckpt_path(metric, N, eps)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def upload_gcs(bucket: str | None):
    """Upload checkpoints to GCS (if bucket specified)."""
    if not bucket:
        return
    try:
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", str(CHECKPOINT_DIR), f"gs://{bucket}/fnd1/"],
            timeout=60, capture_output=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Run one experiment (metric × N)
# ---------------------------------------------------------------------------
def run_experiment(metric: str, N: int, M: int, mode: str, gcs_bucket: str | None,
                   T: float = 1.0):
    """Run full CRN experiment for one metric at one N."""
    cfg = METRICS[metric]
    metric_fn = METRIC_FNS[metric]
    eps_values = cfg["eps"]

    print(f"\n{'='*60}")
    print(f"{cfg['label']}  N={N:,}  M={M}  mode={mode}")
    print(f"{'='*60}")

    ss = np.random.SeedSequence(MASTER_SEED + METRIC_SEED_OFFSETS.get(metric, 999))
    all_seeds = ss.spawn(len(eps_values) * M)
    seed_idx = 0

    all_results = {}

    for eps in eps_values:
        existing = load_ckpt(metric, N, eps)
        done_seeds = {r["seed"] for r in existing}
        results = list(existing)

        # eps=0: only 5 sprinklings (sanity check, all deltas must be 0)
        M_this = 5 if abs(eps) < 1e-12 else M
        seeds = []
        for i in range(M):  # always advance seed_idx by M for reproducibility
            s = int(all_seeds[seed_idx + i].generate_state(1)[0])
            if i < M_this:
                seeds.append(s)
        seed_idx += M

        remaining = [s for s in seeds if s not in done_seeds]
        print(f"  eps={eps}: {len(existing)}/{M} done, {len(remaining)} remaining")

        t0 = time.perf_counter()
        for j, seed in enumerate(remaining):
            # Random operator test: at two highest eps, first 20 sprinklings
            nonzero_eps = sorted([e for e in eps_values if abs(e) > 1e-12])
            top_two_eps = set(nonzero_eps[-2:]) if len(nonzero_eps) >= 2 else set(nonzero_eps)
            do_random = (eps in top_two_eps) and (j < 20) and (mode == "dense")
            r = crn_one(seed, N, T, eps, metric_fn, mode,
                        include_random_operator=do_random)
            results.append(r)

            if (j + 1) % 5 == 0 or j == len(remaining) - 1:
                save_ckpt(results, metric, N, eps)
                elapsed = time.perf_counter() - t0
                rate = (j + 1) / elapsed * 60
                eta = (len(remaining) - j - 1) / rate if rate > 0 else 0
                print(f"    {j+1}/{len(remaining)}  "
                      f"{rate:.1f} spr/min  ETA {eta:.0f} min")

        if remaining:
            upload_gcs(gcs_bucket)

        # Paired t-test
        for obs in ["entropy", "frobenius", "max_abs"]:
            deltas = [r[f"{obs}_delta"] for r in results if abs(r["eps"]) > 1e-12]
            if len(deltas) < 2:
                continue
            t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
            d_mean = float(np.mean(deltas))
            d_std = float(np.std(deltas, ddof=1))
            cohen_d = d_mean / d_std if d_std > 1e-20 else 0
            if abs(eps) > 1e-12:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"    {obs}: d={cohen_d:+.3f} p={p_val:.2e} {sig}")

        all_results[str(eps)] = results

    # Save final
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "metric": metric,
        "label": cfg["label"],
        "N": N,
        "M": M,
        "mode": mode,
        "eps_values": eps_values,
        "results": {k: v for k, v in all_results.items()},
    }
    out_path = RESULTS_DIR / f"crn_{metric}_N{N}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"  Saved: {out_path}")

    # Post-experiment analysis
    analyze_experiment(all_results, cfg["label"], N)


# ---------------------------------------------------------------------------
# Statistical analysis (skeptic-proof)
# ---------------------------------------------------------------------------

def permutation_test(deltas: list[float], n_perm: int = 2000) -> dict:
    """Permutation null test: randomly flip signs of paired differences.

    Under H0 (no curvature effect), each delta is equally likely to be
    positive or negative.  We compare the observed |mean(delta)| against
    the distribution of |mean| under random sign flips.
    """
    deltas_arr = np.array(deltas)
    observed = abs(float(np.mean(deltas_arr)))
    rng = np.random.default_rng(777)
    n_exceed = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(deltas_arr))
        perm_mean = abs(float(np.mean(deltas_arr * signs)))
        if perm_mean >= observed:
            n_exceed += 1
    p_perm = (n_exceed + 1) / (n_perm + 1)  # +1 for conservative estimate
    return {"observed_abs_mean": round(observed, 8),
            "p_permutation": round(p_perm, 6),
            "n_permutations": n_perm}


def tc_matched_analysis(results: list[dict], obs: str = "frobenius") -> dict:
    """TC-matched analysis: does signal persist after controlling for TC?

    Bin sprinklings by TC change.  Within each bin (similar TC change),
    test if frobenius_delta is still nonzero.  If yes → signal is NOT
    an artifact of TC change alone.
    """
    tc_deltas = np.array([r["tc_delta_pct"] for r in results])
    obs_deltas = np.array([r[f"{obs}_delta"] for r in results])

    if len(results) < 20:
        return {"skip": "too few samples"}

    # Partial correlation: obs_delta vs eps controlling for tc_delta + tc_delta²
    eps_arr = np.array([r["eps"] for r in results])
    # Residualise both obs_delta and eps on tc_delta (linear + quadratic)
    tc_design = np.column_stack([tc_deltas, tc_deltas**2, np.ones(len(tc_deltas))])
    coeff_obs, _, _, _ = np.linalg.lstsq(tc_design, obs_deltas, rcond=None)
    coeff_eps, _, _, _ = np.linalg.lstsq(tc_design, eps_arr, rcond=None)
    resid_obs = obs_deltas - tc_design @ coeff_obs
    resid_eps = eps_arr - tc_design @ coeff_eps

    if np.std(resid_obs) < 1e-20 or np.std(resid_eps) < 1e-20:
        return {"partial_r": 0.0, "partial_p": 1.0}

    r_partial, p_partial = stats.pearsonr(resid_eps, resid_obs)

    # Also: bin by TC change quartiles, test within each
    quartiles = np.percentile(tc_deltas, [25, 50, 75])
    bin_results = []
    edges = [-np.inf] + list(quartiles) + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (tc_deltas >= lo) & (tc_deltas < hi)
        if np.sum(mask) < 5:
            continue
        bin_deltas = obs_deltas[mask]
        if np.std(bin_deltas) < 1e-20:
            continue
        t_stat, p_val = stats.ttest_1samp(bin_deltas, 0.0)
        bin_results.append({
            "tc_range": f"[{lo:.1f}, {hi:.1f})",
            "n": int(np.sum(mask)),
            "mean_delta": round(float(np.mean(bin_deltas)), 6),
            "p_value": round(float(p_val), 6),
            "significant": p_val < 0.05,
        })

    return {
        "partial_r_controlling_TC": round(float(r_partial), 4),
        "partial_p_controlling_TC": float(p_partial),
        "tc_bin_tests": bin_results,
        "signal_survives_tc_control": p_partial < 0.01,
    }


def sensitivity_curve(all_results: dict, obs: str = "frobenius") -> dict:
    """Detection threshold: at what eps does signal become significant?"""
    curve = []
    for eps_str, results in sorted(all_results.items(), key=lambda x: float(x[0])):
        eps = float(eps_str)
        if abs(eps) < 1e-12:
            continue
        deltas = [r[f"{obs}_delta"] for r in results]
        if len(deltas) < 2:
            continue
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1))
        cohen_d = mean_d / std_d if std_d > 1e-20 else 0.0
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        curve.append({
            "eps": eps,
            "cohen_d": round(cohen_d, 4),
            "p_value": float(p_val),
            "significant_001": p_val < 0.001,
        })

    # Find minimum detectable eps
    min_eps = None
    for pt in curve:
        if pt["significant_001"]:
            min_eps = pt["eps"]
            break

    return {
        "observable": obs,
        "curve": curve,
        "min_detectable_eps": min_eps,
    }


def curvature_extraction(all_results: dict, obs: str = "frobenius") -> dict:
    """Extract curvature coefficient from frobenius_delta vs eps.

    For pp-wave: Riemann ~ eps, so quadratic curvature invariants ~ eps².
    Fit: mean(|frobenius_delta|) = A * eps^beta.
    If beta ≈ 2 → observable scales as curvature².
    If A is stable across N → converges to a continuum geometric quantity.

    Also extract: frobenius_delta / eps^2 = C(N).
    If C(N) → const as N → ∞ → the constant IS the curvature coefficient.
    """
    eps_vals = []
    mean_deltas = []
    for eps_str, results in sorted(all_results.items(), key=lambda x: float(x[0])):
        eps = float(eps_str)
        if abs(eps) < 1e-12:
            continue
        deltas = [r[f"{obs}_delta"] for r in results]
        if len(deltas) < 5:
            continue
        # Use signed mean (not |mean|) to preserve direction
        # Only include eps where signal is significant (p<0.05)
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        if p_val > 0.10:
            continue  # skip noise-dominated eps
        eps_vals.append(eps)
        mean_deltas.append(float(np.mean(deltas)))

    if len(eps_vals) < 3:
        return {"skip": "need >= 3 nonzero eps values"}

    eps_arr = np.array(eps_vals)
    delta_arr = np.array(mean_deltas)

    # Fit: log(|delta|) = beta * log(eps) + log(A)
    abs_delta = np.abs(delta_arr)
    sign = np.sign(np.mean(delta_arr))  # overall sign direction
    log_fit = np.polyfit(np.log(eps_arr), np.log(abs_delta + 1e-300), 1)
    beta = log_fit[0]
    A = float(np.exp(log_fit[1]))

    # Curvature coefficient: |delta| / eps^2 at each eps
    coefficients = abs_delta / eps_arr ** 2
    C_mean = float(np.mean(coefficients))
    C_std = float(np.std(coefficients, ddof=1)) if len(coefficients) > 1 else 0.0

    # Quality: how well does eps^beta fit?
    predicted = A * eps_arr ** beta
    residuals = abs_delta - predicted
    r_squared = 1 - np.sum(residuals**2) / np.sum((abs_delta - np.mean(abs_delta))**2)

    return {
        "observable": obs,
        "fitted_beta": round(beta, 3),
        "fitted_A": round(A, 8),
        "r_squared": round(float(r_squared), 4),
        "eps_squared_consistent": abs(beta - 2.0) < 0.5,  # hypothesis: curvature² scaling
        "curvature_coefficient_mean": round(C_mean, 8),
        "curvature_coefficient_std": round(C_std, 8),
        "curvature_coefficient_by_eps": {
            str(e): round(float(c), 8) for e, c in zip(eps_vals, coefficients)
        },
        "interpretation": (
            f"frobenius_delta ~ eps^{beta:.2f} (expect 2.0 for curvature²). "
            f"Curvature coefficient C = {C_mean:.4e} ± {C_std:.4e}. "
            f"{'CONSISTENT with quadratic curvature scaling.' if abs(beta - 2.0) < 0.5 else 'DEVIATES from quadratic — may indicate nonlinear regime or non-curvature effect.'}"
        ),
    }


def convergence_fit(results_by_N: dict, obs: str = "frobenius") -> dict:
    """Fit observable_delta / N^alpha → const.

    If converges → genuine geometric quantity.
    If diverges → finite-size artifact.
    """
    Ns = []
    means = []
    for N_str, results in results_by_N.items():
        N_val = int(N_str)
        # Use the highest eps results for strongest signal
        deltas = [r[f"{obs}_delta"] for r in results if abs(r["eps"]) > 1e-12]
        if len(deltas) < 5:
            continue
        Ns.append(N_val)
        means.append(float(np.mean(np.abs(deltas))))

    if len(Ns) < 2:
        return {"skip": "need results at multiple N values"}

    Ns = np.array(Ns, dtype=float)
    means = np.array(means)

    # Fit: log(mean) = alpha * log(N) + const
    coeffs = np.polyfit(np.log(Ns), np.log(means + 1e-300), 1)
    alpha = coeffs[0]

    # Check: does mean / N^alpha converge?
    normalised = means / Ns ** alpha
    cv = float(np.std(normalised) / np.mean(normalised))  # coefficient of variation

    return {
        "observable": obs,
        "N_values": [int(n) for n in Ns],
        "mean_abs_delta": [round(float(m), 6) for m in means],
        "fitted_alpha": round(alpha, 3),
        "normalised_values": [round(float(v), 6) for v in normalised],
        "coefficient_of_variation": round(cv, 4),
        "converges": cv < 0.3,  # CV < 30% → reasonably stable
    }


def analyze_experiment(all_results: dict, label: str, N: int):
    """Run all post-experiment analyses and print results."""
    print(f"\n  ── POST-EXPERIMENT ANALYSIS: {label} N={N:,} ──")

    # Pre-registered observable: frobenius only
    obs = "frobenius"
    perm = {}

    # 0. Conformal null validation
    if "NULL" in label.upper() or "conformal" in label.lower():
        all_deltas = [r[f"{obs}_delta"] for results in all_results.values()
                      for r in results]
        max_delta = max(abs(d) for d in all_deltas) if all_deltas else 0
        print(f"  NULL TEST: max|delta| = {max_delta:.2e}  "
              f"({'PASS (exactly 0)' if max_delta < 1e-10 else 'FAIL — BUG IN CODE'})")
        return  # no further analysis needed for null test

    # 1. Permutation test (on highest eps)
    eps_keys = sorted(all_results.keys(), key=lambda x: float(x))
    if len(eps_keys) >= 2:
        highest_eps = eps_keys[-1]
        deltas = [r[f"{obs}_delta"] for r in all_results[highest_eps]]
        if len(deltas) >= 10:
            perm = permutation_test(deltas)
            print(f"  Permutation test (eps={highest_eps}, {obs}): "
                  f"p={perm['p_permutation']:.4f}  "
                  f"({'SIGNIFICANT' if perm['p_permutation'] < 0.01 else 'not sig'})")
            # Also Wilcoxon signed-rank
            try:
                w_stat, w_p = stats.wilcoxon(deltas)
                print(f"  Wilcoxon signed-rank: p={w_p:.2e}")
            except ValueError:
                pass

    # 2. TC-matched analysis
    all_flat = [r for results in all_results.values() for r in results]
    tc_res = tc_matched_analysis(all_flat, obs)
    if "partial_r_controlling_TC" in tc_res:
        print(f"  TC-matched (partial r controlling TC): "
              f"r={tc_res['partial_r_controlling_TC']:.3f}  "
              f"p={tc_res['partial_p_controlling_TC']:.2e}  "
              f"({'SURVIVES' if tc_res['signal_survives_tc_control'] else 'FAILS'})")
        for b in tc_res.get("tc_bin_tests", []):
            sig = "**" if b["p_value"] < 0.01 else "*" if b["p_value"] < 0.05 else "ns"
            print(f"    TC bin {b['tc_range']}: n={b['n']}  "
                  f"mean_delta={b['mean_delta']:.6f}  p={b['p_value']:.3f} {sig}")

    # 3. Sensitivity curve
    sens = sensitivity_curve(all_results, obs)
    if sens.get("curve"):
        print(f"  Sensitivity ({obs}):")
        for pt in sens["curve"]:
            sig = "***" if pt["p_value"] < 0.001 else "**" if pt["p_value"] < 0.01 else "*" if pt["p_value"] < 0.05 else "ns"
            print(f"    eps={pt['eps']:>6.1f}: Cohen d={pt['cohen_d']:+.3f}  "
                  f"p={pt['p_value']:.2e} {sig}")
        if sens["min_detectable_eps"] is not None:
            print(f"  Min detectable eps (p<0.001): {sens['min_detectable_eps']}")

    # 4. Curvature extraction
    curv = curvature_extraction(all_results, obs)
    if "fitted_beta" in curv:
        print(f"  Curvature extraction ({obs}):")
        print(f"    frobenius_delta ~ eps^{curv['fitted_beta']:.2f}  "
              f"(expect 2.0)  R²={curv['r_squared']:.3f}")
        print(f"    Curvature coeff C = {curv['curvature_coefficient_mean']:.4e} "
              f"± {curv['curvature_coefficient_std']:.4e}")
        for e, c in curv["curvature_coefficient_by_eps"].items():
            print(f"      eps={e}: C = {c:.4e}")
        status = "CONSISTENT" if curv["eps_squared_consistent"] else "DEVIATES"
        print(f"    Quadratic scaling: {status}")

    # 5. BD action comparison (THE key test for pp-wave: R=0 → S_BD should be ~0)
    bd_deltas = [r["bd_action_delta"] for r in all_flat if abs(r["eps"]) > 1e-12]
    frob_deltas = [r[f"{obs}_delta"] for r in all_flat if abs(r["eps"]) > 1e-12]
    if len(bd_deltas) >= 10:
        bd_t, bd_p = stats.ttest_1samp(bd_deltas, 0.0)
        fr_t, fr_p = stats.ttest_1samp(frob_deltas, 0.0)
        bd_d = float(np.mean(bd_deltas)) / (float(np.std(bd_deltas, ddof=1)) + 1e-20)
        fr_d = float(np.mean(frob_deltas)) / (float(np.std(frob_deltas, ddof=1)) + 1e-20)
        print(f"  BD action vs Commutator comparison:")
        print(f"    BD action:   Cohen d={bd_d:+.3f}  p={bd_p:.2e}  "
              f"({'detects' if bd_p < 0.01 else 'NULL'})")
        print(f"    Commutator:  Cohen d={fr_d:+.3f}  p={fr_p:.2e}  "
              f"({'detects' if fr_p < 0.01 else 'NULL'})")
        if bd_p > 0.05 and fr_p < 0.001:
            print(f"    >>> COMMUTATOR DETECTS WHAT BD ACTION MISSES <<<")
        bd_comparison = {
            "bd_cohen_d": round(bd_d, 4), "bd_p": float(bd_p),
            "comm_cohen_d": round(fr_d, 4), "comm_p": float(fr_p),
            "commutator_adds_information": bd_p > 0.05 and fr_p < 0.001,
        }
    else:
        bd_comparison = {}

    # 6. Hierarchical null operator tests
    null_subset = [r for r in all_flat
                   if "shuf_frobenius_delta" in r and abs(r["eps"]) > 1e-12]
    if len(null_subset) >= 8:
        bd_frob = [r[f"{obs}_delta"] for r in null_subset]
        bd_d_val = float(np.mean(bd_frob)) / (float(np.std(bd_frob, ddof=1)) + 1e-20)

        def _cohen(key):
            vals = [r[key] for r in null_subset if key in r]
            if len(vals) < 3:
                return 0.0, 1.0
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            d = m / (s + 1e-20)
            _, p = stats.ttest_1samp(vals, 0.0)
            return round(d, 4), float(p)

        shuf_d, shuf_p = _cohen("shuf_frobenius_delta")
        rcaus_d, rcaus_p = _cohen("rcaus_frobenius_delta")
        rand_d, rand_p = _cohen("rand_frobenius_delta")
        sign_d, sign_p = _cohen("rand_sign_frob_delta")

        print(f"  Hierarchical null operator tests ({len(null_subset)} sprinklings):")
        print(f"    BD operator:          d={bd_d_val:+.3f}")
        print(f"    Shuffled layers:      d={shuf_d:+.3f}  p={shuf_p:.2e}  "
              f"{'DETECTS' if shuf_p < 0.01 else 'null'}")
        print(f"    Sign-preserved rand:  d={sign_d:+.3f}  p={sign_p:.2e}  "
              f"{'DETECTS' if sign_p < 0.01 else 'null'}")
        print(f"    Full random coeffs:   d={rand_d:+.3f}  p={rand_p:.2e}  "
              f"{'DETECTS' if rand_p < 0.01 else 'null'}")
        print(f"    Random causal:        d={rcaus_d:+.3f}  p={rcaus_p:.2e}  "
              f"{'DETECTS' if rcaus_p < 0.01 else 'null'}")

        # Interpretation
        if shuf_p > 0.05:
            print(f"    >>> LAYER STRUCTURE IS ESSENTIAL — shuffling kills signal <<<")
        elif rcaus_p > 0.05:
            print(f"    >>> LAYER STRUCTURE helps but RAW CAUSAL is insufficient <<<")
        else:
            print(f"    >>> ANY causal operator detects — signal is from causal structure itself <<<")

        rand_comparison = {
            "bd_d": round(bd_d_val, 4),
            "shuffled_layers_d": shuf_d, "shuffled_p": shuf_p,
            "random_causal_d": rcaus_d, "random_causal_p": rcaus_p,
            "random_coeffs_d": rand_d, "random_p": rand_p,
            "sign_preserved_d": sign_d, "sign_preserved_p": sign_p,
            "layer_structure_essential": shuf_p > 0.05,
            "raw_causal_sufficient": rcaus_p < 0.01,
        }
    else:
        rand_comparison = {}

    # 7. Layer count information test: does commutator add info beyond layers?
    layer_info = {}
    layer_deltas_all = {}
    for k in ["0", "1", "2", "3"]:
        key = f"layer{k}_delta"
        vals = [r[key] for r in all_flat if key in r and abs(r["eps"]) > 1e-12]
        if vals:
            layer_deltas_all[key] = vals
    frob_for_layers = [r[f"{obs}_delta"] for r in all_flat
                       if f"layer0_delta" in r and abs(r["eps"]) > 1e-12]
    if len(frob_for_layers) >= 20 and len(layer_deltas_all) == 4:
        # Regress frobenius_delta on layer deltas: how much do layers explain?
        Y = np.array(frob_for_layers)
        X = np.column_stack([np.array(layer_deltas_all[f"layer{k}_delta"])
                             for k in ["0", "1", "2", "3"]]
                            + [np.ones(len(frob_for_layers))])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_pred = X @ coeffs
        ss_res = float(np.sum((Y - Y_pred) ** 2))
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        r2_layers = 1 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
        layer_info = {
            "r2_layers_explain_frobenius": round(r2_layers, 4),
            "commutator_adds_info": r2_layers < 0.90,
        }
        print(f"  Layer count information test:")
        print(f"    R² (layer deltas → frobenius_delta): {r2_layers:.3f}")
        if r2_layers < 0.90:
            print(f"    >>> COMMUTATOR carries info BEYOND layer counts ({(1-r2_layers)*100:.0f}% unexplained) <<<")
        else:
            print(f"    WARNING: layer counts explain {r2_layers*100:.0f}% — commutator may be redundant")

    # Save analysis
    analysis = {"permutation": perm if 'perm' in dir() else {},
                "tc_matched": tc_res, "sensitivity": sens,
                "curvature_extraction": curv,
                "bd_action_comparison": bd_comparison,
                "random_operator_test": rand_comparison,
                "layer_info_test": layer_info}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    apath = RESULTS_DIR / f"analysis_{label.replace(' ', '_')}_N{N}.json"
    with open(apath, "w") as f:
        json.dump(analysis, f, indent=1, default=str)
    print(f"  Analysis saved: {apath.name}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate():
    """Quick calibration: 1 sprinkling per N, measure time."""
    print("CALIBRATION")
    print("-" * 40)

    for N in [5000, 10000, 20000, 30000]:
        mem_needed = N**2 * 8 * 5 / 1e9
        try:
            import psutil
            avail = psutil.virtual_memory().available / 1e9
            if mem_needed > avail * 0.8:
                print(f"  N={N:>6,}: skip (need {mem_needed:.0f} GB, avail {avail:.0f} GB)")
                continue
        except ImportError:
            pass

        t0 = time.perf_counter()
        r = crn_one(42, N, T_DIAMOND, 5.0, causal_ppwave_quad, "dense")
        elapsed = time.perf_counter() - t0
        print(f"  N={N:>6,}: {elapsed:>6.1f}s per CRN pair  "
              f"→ 400 pairs = {elapsed*400/3600:.1f}h")
        del r; gc.collect()
        if GPU_OK:
            cp.get_default_memory_pool().free_all_blocks()

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FND-1 GCP Pipeline")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--metric", type=str, default=None,
                        help="Run single metric (ppwave_quad, ppwave_cross, "
                             "schwarzschild, flrw, conformal)")
    parser.add_argument("--N", type=int, default=30000)
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--T", type=float, default=1.0,
                        help="Diamond half-extent. Test T-independence with 0.5, 1.0, 2.0")
    parser.add_argument("--mode", choices=["dense", "matfree"], default="dense")
    parser.add_argument("--gcs", type=str, default=None, help="GCS bucket name")
    parser.add_argument("--all", action="store_true", help="Run all metrics")
    args = parser.parse_args()

    print("=" * 60)
    print("FND-1 GCP RESEARCH PIPELINE")
    print("=" * 60)
    print(f"GPU: {'YES — ' + str(cp.cuda.runtime.getDeviceProperties(0)['name'].decode()) if GPU_OK else 'NO (CPU mode)'}")
    print()

    if args.calibrate:
        calibrate()
        return

    if args.all:
        metrics = list(METRICS.keys())
    elif args.metric:
        metrics = [args.metric]
    else:
        metrics = ["ppwave_quad"]  # default

    for metric in metrics:
        if metric not in METRICS:
            print(f"Unknown metric: {metric}. Options: {list(METRICS.keys())}")
            continue
        run_experiment(metric, args.N, args.M, args.mode, args.gcs, T=args.T)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
