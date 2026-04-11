#!/usr/bin/env python3
"""
Universal local-geometry experiment skeleton for path_kurtosis.

Self-contained implementation using only numpy/scipy + Python stdlib.
Designed for Windows/Python 3.12 with multiprocessing-safe entry point.

Primary features:
- Local diamond sprinkling in 4D Minkowski coordinates (t, x, y, z)
- Generic Hasse builder from arbitrary predecessor predicates
- Stable log-path counting and path-kurtosis readout
- Ricci-built control metric in quadratic RNC jet
- l=0 shell-average subtraction on Poisson sprinklings
- Orientation averaging
- Stage-based run list (0-4) from an internal JSON-serializable config
- One JSON output per expanded condition + stage/global summary

Important caveats:
- The generic jet predicate is a quadratic-RNC midpoint test, not an exact geodesic interval.
- The exact pp-wave predicate implemented here assumes the convention
    ds^2 = -du dv + dx^2 + dy^2 + (eps/2)(x^2-y^2) du^2,
  with u=t+z, v=t-z in the local chart.
  If your production pp-wave exact module uses a different convention,
  replace `ppwave_exact_preds` accordingly.
- The Schwarzschild local vacuum patch uses the Weyl electric tensor model in a static orthonormal frame.
  This is the intended local small-diamond object, distinct from the global shell/Shapiro pipeline.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import kurtosis as scipy_kurtosis

# --------------------------------------------------------------------------------------
# Basic constants and helpers
# --------------------------------------------------------------------------------------
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
SPATIAL_ID = np.eye(3)


def excess_kurtosis(x: np.ndarray) -> float:
    """Return Fisher excess kurtosis, matching scipy.stats.kurtosis(..., fisher=True)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return 0.0
    if np.allclose(np.var(x), 0.0):
        return 0.0
    return float(scipy_kurtosis(x, fisher=True, bias=True))


def summarize_samples(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "se": 0.0, "std": 0.0, "d": 0.0, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    d = float(mean / std) if std > 1e-15 else 0.0
    return {"mean": mean, "se": se, "std": std, "d": d, "n": int(arr.size)}


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


# --------------------------------------------------------------------------------------
# Geometry-independent local diamond tools
# --------------------------------------------------------------------------------------

def sprinkle_local_diamond(N: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson sprinkling with exactly N accepted points in a 4D causal diamond of duration T."""
    pts: List[List[float]] = []
    tmin, tmax = -T / 2.0, T / 2.0
    batch_size = max(4096, 8 * N)
    while len(pts) < N:
        batch = rng.uniform(tmin, tmax, size=(batch_size, 4))
        r = np.linalg.norm(batch[:, 1:], axis=1)
        mask = (np.abs(batch[:, 0]) + r) < (T / 2.0)
        if np.any(mask):
            pts.extend(batch[mask].tolist())
    arr = np.asarray(pts[:N], dtype=np.float64)
    order = np.argsort(arr[:, 0], kind="mergesort")
    return arr[order]


def bulk_mask(pts: np.ndarray, T: float, zeta: float) -> np.ndarray:
    tau = pts[:, 0]
    r = np.linalg.norm(pts[:, 1:], axis=1)
    slack = T / 2.0 - (np.abs(tau) + r)
    return slack >= (zeta * T)


def rotate_points_spatial(pts: np.ndarray, Q: np.ndarray) -> np.ndarray:
    out = np.array(pts, copy=True)
    out[:, 1:] = pts[:, 1:] @ Q.T
    return out


# --------------------------------------------------------------------------------------
# Orientation sets
# --------------------------------------------------------------------------------------

def orientation_set_O1() -> List[np.ndarray]:
    return [np.eye(3)]


def orientation_set_O6() -> List[np.ndarray]:
    mats = []
    basis = [
        np.eye(3),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),   # z -> x
        np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),    # z -> -x
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),    # z -> y
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),    # z -> -y
        np.diag([1.0, -1.0, -1.0]),                                        # z -> -z
    ]
    for Q in basis:
        mats.append(Q)
    return mats


def orientation_set_O24() -> List[np.ndarray]:
    mats = []
    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]
    signs = [
        np.diag([sx, sy, sz]).astype(np.float64)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    for p in perms:
        P = np.zeros((3, 3), dtype=np.float64)
        for i, j in enumerate(p):
            P[i, j] = 1.0
        for S in signs:
            Q = S @ P
            if np.linalg.det(Q) > 0.5:
                mats.append(Q)
    # unique
    unique = []
    for Q in mats:
        if not any(np.allclose(Q, U) for U in unique):
            unique.append(Q)
    return unique


def get_orientation_set(name: str) -> List[np.ndarray]:
    table = {
        "O1": orientation_set_O1,
        "O6": orientation_set_O6,
        "O24": orientation_set_O24,
    }
    if name not in table:
        raise ValueError(f"Unknown orientation set: {name}")
    return table[name]()


# --------------------------------------------------------------------------------------
# Riemann tensor utilities
# --------------------------------------------------------------------------------------

def set_riemann_component(R: np.ndarray, a: int, b: int, c: int, d: int, val: float) -> None:
    """Fill all antisymmetry/pair-symmetry related entries for one lower-index component."""
    entries = [
        (a, b, c, d, +val),
        (b, a, c, d, -val),
        (a, b, d, c, -val),
        (b, a, d, c, +val),
        (c, d, a, b, +val),
        (d, c, a, b, -val),
        (c, d, b, a, -val),
        (d, c, b, a, +val),
    ]
    for i, j, k, l, v in entries:
        R[i, j, k, l] = v


def ricci_built_part(R: np.ndarray) -> np.ndarray:
    """Return the Ricci-built part of the 4D Riemann tensor in an orthonormal frame."""
    eta_inv = ETA.copy()
    Ric = np.einsum("ac,abcd->bd", eta_inv, R, optimize=True)
    Rsc = float(np.einsum("bd,bd->", eta_inv, Ric, optimize=True))
    Rric = np.zeros_like(R)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    Rric[a, b, c, d] = 0.5 * (
                        ETA[a, c] * Ric[b, d]
                        - ETA[a, d] * Ric[b, c]
                        - ETA[b, c] * Ric[a, d]
                        + ETA[b, d] * Ric[a, c]
                    ) - (Rsc / 6.0) * (
                        ETA[a, c] * ETA[b, d] - ETA[a, d] * ETA[b, c]
                    )
    return Rric


def rotate_riemann(R: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotate spatial frame by Q, keeping time axis fixed."""
    L = np.eye(4)
    L[1:, 1:] = Q
    return np.einsum("ap,bq,cr,ds,pqrs->abcd", L, L, L, L, R, optimize=True)


def riemann_constant_curvature(K: float) -> np.ndarray:
    """Riemann for 4D constant-curvature space with sectional curvature K."""
    R = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    R[a, b, c, d] = K * (ETA[a, c] * ETA[b, d] - ETA[a, d] * ETA[b, c])
    return R


def riemann_ds(H: float) -> np.ndarray:
    return riemann_constant_curvature(H * H)


def riemann_flrw(gamma: float) -> np.ndarray:
    """Quadratic RNC jet at t=0 for a(t)=1 + 0.5*gamma*t^2."""
    R = np.zeros((4, 4, 4, 4), dtype=np.float64)
    # R_{0i0j} = -gamma delta_ij, spatial R_{ijkl}=0 at t=0 because H=0.
    for i in range(1, 4):
        set_riemann_component(R, 0, i, 0, i, -gamma)
    return R


def riemann_ppwave_canonical(eps: float) -> np.ndarray:
    """Canonical pp-wave Riemann in (t,x,y,z) local coordinates.

    Assumes the exact predicate convention implemented below:
        ds^2 = -du dv + dx^2 + dy^2 + (eps/2)(x^2-y^2) du^2,
    with u=t+z, v=t-z.
    """
    R = np.zeros((4, 4, 4, 4), dtype=np.float64)
    ex = -0.5 * eps
    ey = +0.5 * eps
    # x-sector
    set_riemann_component(R, 0, 1, 0, 1, ex)  # t x t x
    set_riemann_component(R, 0, 1, 3, 1, ex)  # t x z x
    set_riemann_component(R, 3, 1, 3, 1, ex)  # z x z x
    # y-sector
    set_riemann_component(R, 0, 2, 0, 2, ey)  # t y t y
    set_riemann_component(R, 0, 2, 3, 2, ey)  # t y z y
    set_riemann_component(R, 3, 2, 3, 2, ey)  # z y z y
    return R


def riemann_vacuum_from_E(E: np.ndarray) -> np.ndarray:
    """Vacuum Riemann from electric Weyl tensor E_ij (B=0) in orthonormal frame."""
    R = np.zeros((4, 4, 4, 4), dtype=np.float64)
    delta = np.eye(3)
    # R_{0i0j}=E_{ij}
    for i in range(3):
        for j in range(3):
            if abs(E[i, j]) > 0:
                set_riemann_component(R, 0, i + 1, 0, j + 1, float(E[i, j]))
    # R_{ijkl} = -(δ_{ik}E_{jl}+δ_{jl}E_{ik}-δ_{il}E_{jk}-δ_{jk}E_{il})
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = -(
                        delta[i, k] * E[j, l]
                        + delta[j, l] * E[i, k]
                        - delta[i, l] * E[j, k]
                        - delta[j, k] * E[i, l]
                    )
                    if abs(val) > 0:
                        set_riemann_component(R, i + 1, j + 1, k + 1, l + 1, float(val))
    return R


def riemann_schwarzschild_local(M: float, r0: float) -> np.ndarray:
    """Static local vacuum patch, radial direction along +z.

    Uses the canonical Schwarzschild electric Weyl tensor in an orthonormal static frame:
        E = diag(-q, -q, 2q), q = M/r0^3.
    """
    q = M / (r0 ** 3)
    E = np.diag([-q, -q, 2.0 * q]).astype(np.float64)
    return riemann_vacuum_from_E(E)


def geometry_riemann(geometry: str, params: Dict[str, float]) -> np.ndarray:
    if geometry == "ppwave":
        return riemann_ppwave_canonical(float(params["eps"]))
    if geometry == "ds":
        return riemann_ds(float(params["H"]))
    if geometry == "flrw":
        return riemann_flrw(float(params["gamma"]))
    if geometry == "schwarzschild_local":
        return riemann_schwarzschild_local(float(params["M"]), float(params["r0"]))
    raise ValueError(f"Unknown geometry for Riemann construction: {geometry}")


# --------------------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------------------

def minkowski_preds(pts: np.ndarray, i: int, tol: float = 1e-12) -> np.ndarray:
    x = pts[i]
    y = pts[:i]
    d = x[None, :] - y
    s2 = -d[:, 0] ** 2 + np.sum(d[:, 1:] ** 2, axis=1)
    return (d[:, 0] > tol) & (s2 <= tol)


def jet_preds(pts: np.ndarray, i: int, R_abcd: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    x = pts[i]
    y = pts[:i]
    d = x[None, :] - y
    m = 0.5 * (x[None, :] + y)
    gmid = ETA[None, :, :] - (1.0 / 3.0) * np.einsum(
        "aubv,nu,nv->nab", R_abcd, m, m, optimize=True
    )
    s2 = np.einsum("na,nab,nb->n", d, gmid, d, optimize=True)
    return (d[:, 0] > tol) & (s2 <= tol)


def ds_preds(pts: np.ndarray, i: int, H: float, tol: float = 1e-12) -> np.ndarray:
    tB = float(pts[i, 0])
    xB = pts[i, 1:]
    tA = pts[:i, 0]
    xA = pts[:i, 1:]
    chi = (np.exp(-H * tA) - np.exp(-H * tB)) / H
    R = np.linalg.norm(xB[None, :] - xA, axis=1)
    return (tB > tA + tol) & (chi >= R - tol)


def flrw_preds(pts: np.ndarray, i: int, gamma: float, tol: float = 1e-12) -> np.ndarray:
    tB = float(pts[i, 0])
    xB = pts[i, 1:]
    tA = pts[:i, 0]
    xA = pts[:i, 1:]
    fac = math.sqrt(2.0 / gamma)
    chi = fac * (
        np.arctan(math.sqrt(gamma / 2.0) * tB)
        - np.arctan(np.sqrt(gamma / 2.0) * tA)
    )
    R = np.linalg.norm(xB[None, :] - xA, axis=1)
    return (tB > tA + tol) & (chi >= R - tol)


def vneeded_ppwave_exact(U: np.ndarray, xA: np.ndarray, xB: np.ndarray,
                         yA: np.ndarray, yB: np.ndarray, eps: float,
                         tol: float = 1e-12, series_threshold: float = 1e-4) -> np.ndarray:
    U, xA, xB, yA, yB = np.broadcast_arrays(U, xA, xB, yA, yB)
    out = np.full(U.shape, np.inf, dtype=np.float64)
    maskU = U > tol
    if not np.any(maskU):
        return out

    Um = U[maskU]
    x0 = xA[maskU]
    x1 = xB[maskU]
    y0 = yA[maskU]
    y1 = yB[maskU]

    if abs(eps) < tol:
        out[maskU] = ((x1 - x0) ** 2 + (y1 - y0) ** 2) / Um
        return out

    def series2(Um_, x0_, x1_, y0_, y1_, eps_):
        term0 = ((x1_ - x0_) ** 2 + (y1_ - y0_) ** 2) / Um_
        term1 = (eps_ * Um_ / 6.0) * (
            (x0_ * x0_ + x0_ * x1_ + x1_ * x1_)
            - (y0_ * y0_ + y0_ * y1_ + y1_ * y1_)
        )
        term2 = -(eps_ * eps_) * (Um_ ** 3) / 720.0 * (
            4.0 * (x0_ * x0_ + x1_ * x1_ + y0_ * y0_ + y1_ * y1_)
            + 7.0 * (x0_ * x1_ + y0_ * y1_)
        )
        return term0 + term1 + term2

    w = math.sqrt(abs(eps) / 2.0)
    eta = w * Um
    small = np.abs(eta) < series_threshold
    conj = eta >= (math.pi - tol)
    vals = np.empty_like(Um)
    vals[conj] = -np.inf

    reg = ~conj
    if np.any(reg & small):
        idx = reg & small
        vals[idx] = series2(Um[idx], x0[idx], x1[idx], y0[idx], y1[idx], eps)

    if np.any(reg & ~small):
        idx = reg & ~small
        et = eta[idx]
        xa = x0[idx]
        xb = x1[idx]
        ya = y0[idx]
        yb = y1[idx]
        if eps > 0:
            Sx = w * (((xa * xa + xb * xb) * np.cosh(et) - 2.0 * xa * xb) / np.sinh(et))
            Sy = w * (((ya * ya + yb * yb) * np.cos(et) - 2.0 * ya * yb) / np.sin(et))
        else:
            Sx = w * (((xa * xa + xb * xb) * np.cos(et) - 2.0 * xa * xb) / np.sin(et))
            Sy = w * (((ya * ya + yb * yb) * np.cosh(et) - 2.0 * ya * yb) / np.sinh(et))
        vals[idx] = Sx + Sy

    out[maskU] = vals
    return out


def ppwave_exact_preds(pts: np.ndarray, i: int, eps: float, tol: float = 1e-12) -> np.ndarray:
    """Exact pp-wave predecessor test in canonical local coordinates (t,x,y,z).

    Metric convention:
        ds^2 = -du dv + dx^2 + dy^2 + (eps/2)(x^2-y^2) du^2,
        u=t+z, v=t-z.
    """
    t = pts[:, 0]
    x = pts[:, 1]
    y = pts[:, 2]
    z = pts[:, 3]

    U = (t[i] + z[i]) - (t[:i] + z[:i])
    V = (t[i] - z[i]) - (t[:i] - z[:i])
    Vneed = vneeded_ppwave_exact(
        U,
        x[:i], np.full(i, x[i]),
        y[:i], np.full(i, y[i]),
        eps,
        tol=tol,
    )
    return (U > tol) & (V >= Vneed - tol)


# --------------------------------------------------------------------------------------
# Generic Hasse builder from predecessor predicate
# --------------------------------------------------------------------------------------

def build_hasse_from_predicate(pts: np.ndarray, pred_fn) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build Hasse diagram (transitive reduction) using a bitset transitive-closure sweep.

    The point set must be sorted by increasing time coordinate so that all edges point from j < i to i.
    pred_fn(pts, i) must return a boolean mask of length i marking causal predecessors of i.
    """
    n = len(pts)
    parents: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n)]
    children_lists: List[List[int]] = [[] for _ in range(n)]
    past_bits: List[int] = [0] * n

    for i in range(n):
        rel_mask = np.asarray(pred_fn(pts, i), dtype=bool)
        if rel_mask.size != i:
            raise ValueError(f"Predicate for node {i} returned mask of length {rel_mask.size}, expected {i}")
        rel_preds = np.flatnonzero(rel_mask)
        if rel_preds.size == 0:
            continue

        covered = 0
        direct: List[int] = []
        # Iterate newest predecessors first; if an older one lies in the ancestry of an accepted parent,
        # it is not a Hasse parent.
        for j in rel_preds[::-1]:
            bit = 1 << int(j)
            if covered & bit:
                continue
            jj = int(j)
            direct.append(jj)
            covered |= past_bits[jj] | bit

        direct.sort()
        arr = np.asarray(direct, dtype=np.int32)
        parents[i] = arr

        pb = 0
        for jj in direct:
            children_lists[jj].append(i)
            pb |= past_bits[jj] | (1 << jj)
        past_bits[i] = pb

    children = [np.asarray(ch, dtype=np.int32) for ch in children_lists]
    return parents, children


# --------------------------------------------------------------------------------------
# Path counts in the log domain
# --------------------------------------------------------------------------------------

def log_path_counts(parents: List[np.ndarray], children: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(parents)
    log_pd = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = parents[i]
        if p.size == 0:
            log_pd[i] = 0.0
        else:
            log_pd[i] = float(logsumexp(log_pd[p]))

    log_pu = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        ch = children[i]
        if ch.size == 0:
            log_pu[i] = 0.0
        else:
            log_pu[i] = float(logsumexp(log_pu[ch]))

    return log_pd, log_pu


def Y_from_graph(parents: List[np.ndarray], children: List[np.ndarray]) -> np.ndarray:
    log_pd, log_pu = log_path_counts(parents, children)
    L = log_pd + log_pu  # natural log of p_down * p_up
    # log2(exp(L) + 1) = (max(L,0) + log(1 + exp(-|L|))) / ln2
    return (np.maximum(L, 0.0) + np.log1p(np.exp(-np.abs(L)))) / math.log(2.0)


# --------------------------------------------------------------------------------------
# l=0 projection
# --------------------------------------------------------------------------------------

def project_l0_grid(deltaY: np.ndarray, pts: np.ndarray, T: float, mask_support: np.ndarray,
                    n_t: int = 10, n_r: int = 8, n_min: int = 40) -> np.ndarray:
    """Project onto shell-average (l=0) using adaptive bins in (tau_hat, rho_hat)."""
    tau = pts[:, 0]
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(tau)

    idx = np.where(mask_support)[0]
    if idx.size == 0:
        return np.zeros_like(deltaY)

    th = 2.0 * tau[idx] / T
    rh = np.clip(r[idx] / np.maximum(rmax[idx], 1e-12), 0.0, 0.999999)

    it = np.floor((th + 1.0) * 0.5 * n_t).astype(int)
    it = np.clip(it, 0, n_t - 1)
    ir = np.floor(rh * n_r).astype(int)
    ir = np.clip(ir, 0, n_r - 1)

    cell = it * n_r + ir
    counts = np.bincount(cell, minlength=n_t * n_r)

    remap = np.arange(n_t * n_r)
    for tbin in range(n_t):
        for rbin in range(n_r):
            c = tbin * n_r + rbin
            if counts[c] >= n_min:
                continue
            candidates = []
            if rbin > 0:
                candidates.append(tbin * n_r + (rbin - 1))
            if rbin < n_r - 1:
                candidates.append(tbin * n_r + (rbin + 1))
            if candidates:
                target = max(candidates, key=lambda z: counts[z])
                remap[c] = target

    cell = remap[cell]
    sums = np.bincount(cell, weights=deltaY[idx], minlength=n_t * n_r)
    cnts = np.bincount(cell, minlength=n_t * n_r)

    means = np.zeros(n_t * n_r, dtype=np.float64)
    good = cnts > 0
    means[good] = sums[good] / cnts[good]

    Pi0 = np.zeros(len(deltaY), dtype=np.float64)
    Pi0[idx] = means[cell]
    return Pi0


def project_l0_kernel(deltaY: np.ndarray, pts: np.ndarray, T: float, mask_support: np.ndarray,
                      h_t: float = 0.10, h_r: float = 0.10) -> np.ndarray:
    """Validation-only kernel smoother in (tau_hat, rho_hat). O(N^2) on the support set."""
    tau = pts[:, 0]
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(tau)

    idx = np.where(mask_support)[0]
    if idx.size == 0:
        return np.zeros_like(deltaY)

    th = 2.0 * tau[idx] / T
    rh = np.clip(r[idx] / np.maximum(rmax[idx], 1e-12), 0.0, 0.999999)
    dt = th[:, None] - th[None, :]
    dr = rh[:, None] - rh[None, :]
    W = np.exp(-0.5 * (dt / h_t) ** 2 - 0.5 * (dr / h_r) ** 2)
    num = W @ deltaY[idx]
    den = np.sum(W, axis=1)
    Pi0 = np.zeros(len(deltaY), dtype=np.float64)
    Pi0[idx] = num / np.maximum(den, 1e-15)
    return Pi0


# --------------------------------------------------------------------------------------
# Geometry parameter laws and proxies
# --------------------------------------------------------------------------------------

def resolve_params(cond: Dict[str, Any]) -> Dict[str, float]:
    law = cond["param_law"]
    T = float(cond["T"])
    kind = law["kind"]
    if kind == "ppwave_fixed":
        return {"eps": float(law["value"])}
    if kind == "ppwave_q":
        return {"eps": float(law["q"]) / (T * T)}
    if kind == "ds_qR":
        Rsc = float(law["qR"]) / (T * T)
        H = math.sqrt(max(Rsc, 0.0) / 12.0)
        return {"R": Rsc, "H": H}
    if kind == "flrw_qR":
        gamma = float(law["qR"]) / (T * T)
        return {"gamma": gamma, "R": 6.0 * gamma}
    if kind == "sch_qW":
        r0 = float(cond["r0"])
        M = float(law["qW"]) * (r0 ** 3) / (T * T)
        return {"M": M, "r0": r0}
    raise ValueError(f"Unknown parameter law: {kind}")


def weyl_density_proxy(cond: Dict[str, Any], params: Dict[str, float]) -> float:
    g = cond["geometry"]
    if g == "ppwave":
        return float(params["eps"] ** 2)
    if g == "schwarzschild_local":
        q = float(params["M"]) / (float(params["r0"]) ** 3)
        return float(6.0 * q * q)
    return 0.0


def ricci_density_proxy(cond: Dict[str, Any], params: Dict[str, float]) -> float:
    g = cond["geometry"]
    if g == "ds":
        return float(abs(params["R"]))
    if g == "flrw":
        return float(abs(params["R"]))
    return 0.0


def is_vacuum_geometry(cond: Dict[str, Any]) -> bool:
    return cond["geometry"] in ("ppwave", "schwarzschild_local")


# --------------------------------------------------------------------------------------
# Predicate preparation helpers
# --------------------------------------------------------------------------------------

def prepare_predicate(pts: np.ndarray, cond: Dict[str, Any], params: Dict[str, float],
                      Q: np.ndarray | None, mode: str, tol: float) -> Tuple[np.ndarray, Any, np.ndarray | None]:
    """Return (working_points, pred_fn, optional_riemann_used)."""
    geometry = cond["geometry"]

    if mode == "flat":
        return pts, lambda P, i: minkowski_preds(P, i, tol=tol), None

    if mode == "ppwave_exact":
        pts_work = rotate_points_spatial(pts, Q) if Q is not None else pts
        eps = float(params["eps"])
        return pts_work, lambda P, i: ppwave_exact_preds(P, i, eps=eps, tol=tol), None

    if mode == "ds_exact":
        H = float(params["H"])
        return pts, lambda P, i: ds_preds(P, i, H=H, tol=tol), None

    if mode == "flrw_exact":
        gamma = float(params["gamma"])
        return pts, lambda P, i: flrw_preds(P, i, gamma=gamma, tol=tol), None

    if mode == "jet":
        R = geometry_riemann(geometry, params)
        if Q is not None:
            R = rotate_riemann(R, Q)
        return pts, lambda P, i: jet_preds(P, i, R_abcd=R, tol=tol), R

    if mode == "ricci_jet":
        R = geometry_riemann(geometry, params)
        R = ricci_built_part(R)
        if Q is not None:
            R = rotate_riemann(R, Q)
        if np.max(np.abs(R)) < 1e-15:
            return pts, lambda P, i: minkowski_preds(P, i, tol=tol), R
        return pts, lambda P, i: jet_preds(P, i, R_abcd=R, tol=tol), R

    raise ValueError(f"Unknown predicate mode: {mode}")


# --------------------------------------------------------------------------------------
# Per-seed workflow
# --------------------------------------------------------------------------------------

def _compute_projected_observables(Y0: np.ndarray, Yfull: np.ndarray, Yric: np.ndarray,
                                   pts: np.ndarray, T: float, zeta: float,
                                   projector_mode: str, validate_projector: bool) -> Dict[str, float]:
    mask_read = bulk_mask(pts, T, zeta)
    mask_proj = bulk_mask(pts, T, zeta / 2.0)

    delta_full = Yfull - Y0
    delta_ric = Yric - Y0
    delta_weyl = delta_full - delta_ric

    Pi0_full_grid = project_l0_grid(delta_full, pts, T, mask_proj)
    Pi0_weyl_grid = project_l0_grid(delta_weyl, pts, T, mask_proj)

    if validate_projector:
        Pi0_weyl_kernel = project_l0_kernel(delta_weyl, pts, T, mask_proj)
        proj_relerr = float(
            np.linalg.norm((Pi0_weyl_grid - Pi0_weyl_kernel)[mask_read])
            / max(np.linalg.norm(delta_weyl[mask_read]), 1e-15)
        )
    else:
        proj_relerr = float("nan")

    delta_cf = delta_full - Pi0_full_grid
    delta_W = delta_weyl - Pi0_weyl_grid

    k0 = excess_kurtosis(Y0[mask_read])
    return {
        "raw": excess_kurtosis(Yfull[mask_read]) - k0,
        "cf": excess_kurtosis((Y0 + delta_cf)[mask_read]) - k0,
        "ric": excess_kurtosis((Y0 + delta_weyl)[mask_read]) - k0,
        "weyl": excess_kurtosis((Y0 + delta_W)[mask_read]) - k0,
        "n_read": int(mask_read.sum()),
        "n_proj": int(mask_proj.sum()),
        "projector_relerr": proj_relerr,
    }


def _run_seed_worker(task: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    cond, seed_index = task
    tol = float(cond.get("tol", 1e-12))
    validate_projector = bool(cond.get("projector_validation", False))
    projector_mode = cond.get("projector_mode", "grid")

    seed = int(cond["seed_base"]) + int(seed_index)
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(int(cond["N"]), float(cond["T"]), rng)

    # Flat graph once per seed
    parents0, children0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i, tol=tol))
    Y0 = Y_from_graph(parents0, children0)

    params = resolve_params(cond)
    Qs = get_orientation_set(cond["orientation_set"])

    orient_records = []
    for Q in Qs:
        # Full graph
        pts_full, pred_full, _ = prepare_predicate(pts, cond, params, Q, cond["full_mode"], tol)
        parentsF, childrenF = build_hasse_from_predicate(pts_full, pred_full)
        Yfull = Y_from_graph(parentsF, childrenF)

        # Ricci-control graph (flat for vacuum)
        if is_vacuum_geometry(cond):
            Yric = Y0
        else:
            pts_ric, pred_ric, _ = prepare_predicate(pts, cond, params, Q, cond.get("ric_mode", "ricci_jet"), tol)
            parentsR, childrenR = build_hasse_from_predicate(pts_ric, pred_ric)
            Yric = Y_from_graph(parentsR, childrenR)

        per_zeta = {}
        for zeta in cond["zeta_values"]:
            per_zeta[str(zeta)] = _compute_projected_observables(
                Y0=Y0,
                Yfull=Yfull,
                Yric=Yric,
                pts=pts,
                T=float(cond["T"]),
                zeta=float(zeta),
                projector_mode=projector_mode,
                validate_projector=validate_projector,
            )

        # Optional calibration comparison (e.g. pp-wave exact vs jet)
        calib = {}
        compare_mode = cond.get("compare_mode")
        if compare_mode:
            pts_cmp, pred_cmp, _ = prepare_predicate(pts, cond, params, Q, compare_mode, tol)
            parentsC, childrenC = build_hasse_from_predicate(pts_cmp, pred_cmp)
            Ycmp = Y_from_graph(parentsC, childrenC)
            for zeta in cond["zeta_values"]:
                zkey = str(zeta)
                mask_read = bulk_mask(pts, float(cond["T"]), float(zeta))
                k0 = excess_kurtosis(Y0[mask_read])
                dk_full = excess_kurtosis(Yfull[mask_read]) - k0
                dk_cmp = excess_kurtosis(Ycmp[mask_read]) - k0
                relerr = abs(dk_full - dk_cmp) / max(abs(dk_full), 1e-15)
                calib[zkey] = {
                    "dk_full": float(dk_full),
                    "dk_compare": float(dk_cmp),
                    "relerr_raw": float(relerr),
                }

        orient_records.append({"per_zeta": per_zeta, "calibration": calib})

    # Orientation average per seed
    seed_record: Dict[str, Any] = {"seed": seed, "params": params, "zeta": {}}
    for zeta in cond["zeta_values"]:
        zkey = str(zeta)
        obs_keys = ["raw", "cf", "ric", "weyl", "projector_relerr"]
        seed_record["zeta"][zkey] = {}
        for ok in obs_keys:
            vals = [rec["per_zeta"][zkey][ok] for rec in orient_records]
            seed_record["zeta"][zkey][ok] = float(np.mean(vals))
        seed_record["zeta"][zkey]["n_read"] = int(np.mean([rec["per_zeta"][zkey]["n_read"] for rec in orient_records]))
        seed_record["zeta"][zkey]["n_proj"] = int(np.mean([rec["per_zeta"][zkey]["n_proj"] for rec in orient_records]))
        if cond.get("compare_mode"):
            relerrs = [rec["calibration"][zkey]["relerr_raw"] for rec in orient_records]
            seed_record["zeta"][zkey]["calibration_relerr_raw"] = float(np.mean(relerrs))

    return seed_record


# --------------------------------------------------------------------------------------
# Condition runner and summaries
# --------------------------------------------------------------------------------------

def condition_normalized_stats(cond: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    params = summary["params_mean"]
    T = float(cond["T"])
    wproxy = weyl_density_proxy(cond, params)
    rproxy = ricci_density_proxy(cond, params)

    for zeta, obs_stats in summary["stats_by_zeta"].items():
        entry = {}
        if wproxy > 0.0:
            entry["A_weyl"] = obs_stats["weyl"]["mean"] / (T ** 4 * wproxy)
            entry["A_ric"] = obs_stats["ric"]["mean"] / (T ** 4 * wproxy)
        if rproxy > 0.0:
            entry["B_raw"] = obs_stats["raw"]["mean"] / (T ** 2 * rproxy)
            entry["B_cf"] = obs_stats["cf"]["mean"] / (T ** 2 * rproxy)
            entry["B_weyl"] = obs_stats["weyl"]["mean"] / max(T ** 2 * rproxy, 1e-15)
        out[zeta] = entry
    return out


def run_condition(cond: Dict[str, Any], output_dir: str | Path, workers: int = 1) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = int(cond["n_seeds"])
    tasks = [(cond, s) for s in range(n_seeds)]
    if workers <= 1:
        seed_records = [_run_seed_worker(t) for t in tasks]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            seed_records = list(ex.map(_run_seed_worker, tasks))

    params_mean: Dict[str, float] = {}
    for key in seed_records[0]["params"]:
        params_mean[key] = float(np.mean([sr["params"][key] for sr in seed_records]))

    stats_by_zeta: Dict[str, Any] = {}
    for zeta in cond["zeta_values"]:
        zkey = str(zeta)
        stats_by_zeta[zkey] = {}
        for obs in ["raw", "cf", "ric", "weyl", "projector_relerr", "calibration_relerr_raw"]:
            vals = [sr["zeta"][zkey][obs] for sr in seed_records if obs in sr["zeta"][zkey]]
            if vals:
                stats_by_zeta[zkey][obs] = summarize_samples(vals)
        stats_by_zeta[zkey]["n_read"] = int(np.mean([sr["zeta"][zkey]["n_read"] for sr in seed_records]))
        stats_by_zeta[zkey]["n_proj"] = int(np.mean([sr["zeta"][zkey]["n_proj"] for sr in seed_records]))

    # Local boundary CV on the primary observable (weyl for vacuum, raw for controls)
    primary_key = "weyl" if is_vacuum_geometry(cond) else "raw"
    zetas = list(map(str, cond["zeta_values"]))
    means = np.array([stats_by_zeta[z][primary_key]["mean"] for z in zetas], dtype=np.float64)
    cv = float(np.std(means, ddof=0) / max(abs(np.mean(means)), 1e-15)) if means.size > 1 else 0.0
    if cv <= 0.15:
        boundary_status = "PASS"
    elif cv <= 0.25:
        boundary_status = "BORDERLINE"
    else:
        boundary_status = "FAIL"

    summary = {
        "condition": cond,
        "params_mean": params_mean,
        "seed_records": seed_records,
        "stats_by_zeta": stats_by_zeta,
        "normalized": condition_normalized_stats(cond, {"params_mean": params_mean, "stats_by_zeta": stats_by_zeta}),
        "criteria_local": {
            "boundary_cv": cv,
            "boundary_status": boundary_status,
        },
    }

    out_path = out_dir / f"{cond['id']}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------------------
# Gap criteria evaluation across multiple conditions
# --------------------------------------------------------------------------------------

def _load_condition_jsons(results_dir: Path) -> List[Dict[str, Any]]:
    items = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.startswith("stage_") or p.name.startswith("global_") or p.name.endswith("config.json"):
            continue
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def fit_gap_criteria(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    zeta_primary = str(config["global"].get("zeta_primary", 0.15))
    out: Dict[str, Any] = {
        "boundary_gap": {},
        "continuum_gap": {},
        "control_gap": {},
        "universality_gap": {},
    }

    # Gap 1: boundary dependence per condition (already local, but summarize again)
    for item in results:
        cid = item["condition"]["id"]
        out["boundary_gap"][cid] = item["criteria_local"]

    # Group by family line for continuum and universality checks.
    family_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        cond = item["condition"]
        family = cond.get("family", cond["geometry"])
        family_groups.setdefault(family, []).append(item)

    # Gap 2: continuum/small-T limit for vacuum families using normalized A_weyl at primary zeta.
    for family, items in family_groups.items():
        vac = [it for it in items if is_vacuum_geometry(it["condition"]) and zeta_primary in it["normalized"] and "A_weyl" in it["normalized"][zeta_primary]]
        if len(vac) < 3:
            continue
        vac = sorted(vac, key=lambda it: float(it["condition"]["T"]))
        Ts = np.array([float(it["condition"]["T"]) for it in vac], dtype=np.float64)
        A = np.array([float(it["normalized"][zeta_primary]["A_weyl"]) for it in vac], dtype=np.float64)
        # Use the three smallest T values (last three after sorting descending? smaller T means smaller numbers)
        order = np.argsort(Ts)
        Ts = Ts[order][:3]
        A = A[order][:3]
        Amean = float(np.mean(A))
        chi2 = float(np.sum((A - Amean) ** 2 / np.maximum(np.abs(Amean), 1e-12) ** 2))
        chi2_per_dof = chi2 / max(len(A) - 1, 1)
        ratios = [float(A[1] / A[0]) if abs(A[0]) > 1e-15 else float("nan"),
                  float(A[2] / A[1]) if abs(A[1]) > 1e-15 else float("nan")]
        if chi2_per_dof < 2.0 and all(0.90 <= r <= 1.10 for r in ratios if np.isfinite(r)):
            status = "PASS"
        elif chi2_per_dof < 3.0 and all(0.85 <= r <= 1.15 for r in ratios if np.isfinite(r)):
            status = "BORDERLINE"
        else:
            status = "FAIL"
        out["continuum_gap"][family] = {
            "Ts": Ts.tolist(),
            "A_weyl": A.tolist(),
            "chi2_per_dof": chi2_per_dof,
            "ratios": ratios,
            "status": status,
        }

    # Gap 3: Ricci/control suppression relative to pp-wave matched T.
    ppw_by_T: Dict[float, float] = {}
    for item in results:
        cond = item["condition"]
        if cond["geometry"] == "ppwave" and zeta_primary in item["stats_by_zeta"]:
            ppw_by_T[float(cond["T"])] = float(item["stats_by_zeta"][zeta_primary]["weyl"]["mean"])

    for item in results:
        cond = item["condition"]
        if cond["geometry"] not in ("ds", "flrw"):
            continue
        T = float(cond["T"])
        if T not in ppw_by_T or zeta_primary not in item["stats_by_zeta"]:
            continue
        ctrl = float(item["stats_by_zeta"][zeta_primary]["weyl"]["mean"])
        ppw = ppw_by_T[T]
        ratio = abs(ctrl) / max(abs(ppw), 1e-15)
        d_abs = abs(item["stats_by_zeta"][zeta_primary]["weyl"]["d"])
        if d_abs < 2.0 and ratio < 0.10:
            status = "PASS"
        elif ratio < 0.25:
            status = "BORDERLINE"
        else:
            status = "FAIL"
        out["control_gap"][cond["id"]] = {
            "T": T,
            "ctrl_mean": ctrl,
            "ppw_mean": ppw,
            "ratio": ratio,
            "|d|": d_abs,
            "status": status,
        }

    # Gap 4: universality collapse across vacuum families on the two smallest T values.
    vac_rows = []
    for item in results:
        cond = item["condition"]
        if not is_vacuum_geometry(cond):
            continue
        if zeta_primary not in item["normalized"] or "A_weyl" not in item["normalized"][zeta_primary]:
            continue
        vac_rows.append({
            "id": cond["id"],
            "geometry": cond["geometry"],
            "T": float(cond["T"]),
            "A_weyl": float(item["normalized"][zeta_primary]["A_weyl"]),
        })

    if len(vac_rows) >= 4:
        # Keep two smallest T for each geometry family.
        selected = []
        by_geom: Dict[str, List[Dict[str, Any]]] = {}
        for row in vac_rows:
            by_geom.setdefault(row["geometry"], []).append(row)
        for geom, rows in by_geom.items():
            rows = sorted(rows, key=lambda r: r["T"])
            selected.extend(rows[:2])
        if len(selected) >= 4:
            A = np.array([r["A_weyl"] for r in selected], dtype=np.float64)
            Amean = float(np.mean(A))
            chi2 = float(np.sum((A - Amean) ** 2 / np.maximum(np.abs(Amean), 1e-12) ** 2))
            chi2_per_dof = chi2 / max(len(A) - 1, 1)
            rel_diffs = []
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    rel = abs(A[i] - A[j]) / max(0.5 * (abs(A[i]) + abs(A[j])), 1e-15)
                    rel_diffs.append(float(rel))
            max_rel = max(rel_diffs) if rel_diffs else 0.0
            if chi2_per_dof < 2.5 and max_rel < 0.20 and np.all(np.sign(A) == np.sign(A[0])):
                status = "PASS"
            elif chi2_per_dof < 5.0 and max_rel < 0.35:
                status = "BORDERLINE"
            else:
                status = "FAIL"
            out["universality_gap"] = {
                "selected": selected,
                "chi2_per_dof": chi2_per_dof,
                "max_relative_difference": max_rel,
                "status": status,
            }
    return out


# --------------------------------------------------------------------------------------
# Configuration and stage expansion
# --------------------------------------------------------------------------------------

def _template(template_id: str, family: str, geometry: str, full_mode: str, param_law: Dict[str, Any],
              T_values: List[float], N: int, n_seeds: int, zeta_values: List[float],
              orientation_set: str, **extra) -> Dict[str, Any]:
    d = {
        "id": template_id,
        "family": family,
        "geometry": geometry,
        "full_mode": full_mode,
        "param_law": param_law,
        "T_values": T_values,
        "N": N,
        "n_seeds": n_seeds,
        "zeta_values": zeta_values,
        "orientation_set": orientation_set,
    }
    d.update(extra)
    return d


DEFAULT_CONFIG: Dict[str, Any] = {
    "global": {
        "output_dir": "universal_runs",
        "workers": 8,
        "tol": 1e-12,
        "seed_base": 910000,
        "zeta_primary": 0.15,
    },
    "stages": {
        "0": [
            _template(
                template_id="C0_ppwave_exact_vs_jet",
                family="ppwave_calibration",
                geometry="ppwave",
                full_mode="ppwave_exact",
                compare_mode="jet",
                param_law={"kind": "ppwave_fixed", "value": 2.0},
                T_values=[0.70, 0.50, 0.35],
                N=10000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O1",
            ),
            _template(
                template_id="C1_ds_exact_ricci",
                family="ds_control",
                geometry="ds",
                full_mode="ds_exact",
                ric_mode="ricci_jet",
                param_law={"kind": "ds_qR", "qR": 0.10},
                T_values=[0.70, 0.50, 0.35],
                N=10000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O1",
            ),
            _template(
                template_id="C2_ppwave_projector_validation",
                family="ppwave_projval",
                geometry="ppwave",
                full_mode="ppwave_exact",
                param_law={"kind": "ppwave_q", "q": 0.10},
                T_values=[0.50, 0.25],
                N=10000,
                n_seeds=5,
                zeta_values=[0.15],
                orientation_set="O1",
                projector_validation=True,
            ),
            _template(
                template_id="C2_ds_projector_validation",
                family="ds_projval",
                geometry="ds",
                full_mode="ds_exact",
                ric_mode="ricci_jet",
                param_law={"kind": "ds_qR", "qR": 0.10},
                T_values=[0.50, 0.25],
                N=10000,
                n_seeds=5,
                zeta_values=[0.15],
                orientation_set="O1",
                projector_validation=True,
            ),
        ],
        "1": [
            _template(
                template_id="S1_ppwave_q010",
                family="ppwave_q010",
                geometry="ppwave",
                full_mode="ppwave_exact",
                param_law={"kind": "ppwave_q", "q": 0.10},
                T_values=[0.70, 0.50, 0.35, 0.25],
                N=10000,
                n_seeds=20,
                zeta_values=[0.10, 0.15, 0.20],
                orientation_set="O6",
            ),
            _template(
                template_id="S1_ppwave_q015",
                family="ppwave_q015",
                geometry="ppwave",
                full_mode="ppwave_exact",
                param_law={"kind": "ppwave_q", "q": 0.15},
                T_values=[0.70, 0.50, 0.35, 0.25],
                N=10000,
                n_seeds=20,
                zeta_values=[0.10, 0.15, 0.20],
                orientation_set="O6",
            ),
            _template(
                template_id="S1_ppwave_fixed_eps2",
                family="ppwave_fixed_eps2",
                geometry="ppwave",
                full_mode="ppwave_exact",
                param_law={"kind": "ppwave_fixed", "value": 2.0},
                T_values=[0.70, 0.50, 0.35],
                N=10000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O1",
            ),
        ],
        "2": [
            _template(
                template_id="S2_ds_qR010",
                family="ds_qR010",
                geometry="ds",
                full_mode="ds_exact",
                ric_mode="ricci_jet",
                param_law={"kind": "ds_qR", "qR": 0.10},
                T_values=[0.70, 0.50, 0.35, 0.25],
                N=10000,
                n_seeds=20,
                zeta_values=[0.10, 0.15, 0.20],
                orientation_set="O1",
            ),
            _template(
                template_id="S2_flrw_qR010",
                family="flrw_qR010",
                geometry="flrw",
                full_mode="flrw_exact",
                ric_mode="ricci_jet",
                param_law={"kind": "flrw_qR", "qR": 0.10},
                T_values=[0.70, 0.50, 0.35, 0.25],
                N=10000,
                n_seeds=15,
                zeta_values=[0.10, 0.15, 0.20],
                orientation_set="O1",
            ),
        ],
        "3": [
            _template(
                template_id="S3_sch_r050_q010",
                family="sch_q010_r050",
                geometry="schwarzschild_local",
                full_mode="jet",
                ric_mode="ricci_jet",
                param_law={"kind": "sch_qW", "qW": 0.010},
                T_values=[0.50, 0.35, 0.25],
                N=10000,
                n_seeds=20,
                zeta_values=[0.10, 0.15, 0.20],
                orientation_set="O6",
                r0=0.50,
            ),
            _template(
                template_id="S3_sch_r050_q015",
                family="sch_q015_r050",
                geometry="schwarzschild_local",
                full_mode="jet",
                ric_mode="ricci_jet",
                param_law={"kind": "sch_qW", "qW": 0.015},
                T_values=[0.50, 0.35, 0.25],
                N=10000,
                n_seeds=20,
                zeta_values=[0.10, 0.15, 0.20],
                orientation_set="O6",
                r0=0.50,
            ),
            _template(
                template_id="S3_sch_r070_q010",
                family="sch_q010_r070",
                geometry="schwarzschild_local",
                full_mode="jet",
                ric_mode="ricci_jet",
                param_law={"kind": "sch_qW", "qW": 0.010},
                T_values=[0.35, 0.25],
                N=10000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O6",
                r0=0.70,
            ),
        ],
        "4": [
            _template(
                template_id="S4_ppwave_q010_N20k",
                family="ppwave_q010_confirm",
                geometry="ppwave",
                full_mode="ppwave_exact",
                param_law={"kind": "ppwave_q", "q": 0.10},
                T_values=[0.35, 0.25],
                N=20000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O6",
            ),
            _template(
                template_id="S4_ds_qR010_N20k",
                family="ds_qR010_confirm",
                geometry="ds",
                full_mode="ds_exact",
                ric_mode="ricci_jet",
                param_law={"kind": "ds_qR", "qR": 0.10},
                T_values=[0.35, 0.25],
                N=20000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O1",
            ),
            _template(
                template_id="S4_sch_r050_q010_N20k",
                family="sch_q010_r050_confirm",
                geometry="schwarzschild_local",
                full_mode="jet",
                ric_mode="ricci_jet",
                param_law={"kind": "sch_qW", "qW": 0.010},
                T_values=[0.35, 0.25],
                N=20000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O6",
                r0=0.50,
            ),
            _template(
                template_id="S4_ppwave_q010_T018_N20k",
                family="ppwave_q010_stretch",
                geometry="ppwave",
                full_mode="ppwave_exact",
                param_law={"kind": "ppwave_q", "q": 0.10},
                T_values=[0.18],
                N=20000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O6",
            ),
            _template(
                template_id="S4_ds_qR010_T018_N20k",
                family="ds_qR010_stretch",
                geometry="ds",
                full_mode="ds_exact",
                ric_mode="ricci_jet",
                param_law={"kind": "ds_qR", "qR": 0.10},
                T_values=[0.18],
                N=20000,
                n_seeds=10,
                zeta_values=[0.15],
                orientation_set="O1",
            ),
        ],
    },
}


def expand_stage_conditions(config: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    if stage not in config["stages"]:
        raise ValueError(f"Unknown stage {stage!r}")
    global_cfg = config["global"]
    out: List[Dict[str, Any]] = []
    stage_templates = config["stages"][stage]
    running_seed_base = int(global_cfg.get("seed_base", 910000)) + 100000 * int(stage)
    for tidx, tmpl in enumerate(stage_templates):
        for j, T in enumerate(tmpl["T_values"]):
            cond = {k: v for k, v in tmpl.items() if k != "T_values"}
            cond["T"] = float(T)
            cond["tol"] = float(global_cfg.get("tol", 1e-12))
            cond["id"] = f"{tmpl['id']}_T{str(T).replace('.', 'p')}_N{tmpl['N']}"
            cond["stage"] = stage
            cond["seed_base"] = int(running_seed_base + 1000 * tidx + 50 * j)
            out.append(cond)
    return out


def dump_default_config(path: str | Path) -> None:
    Path(path).write_text(json.dumps(DEFAULT_CONFIG, indent=2, default=json_default), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Experiment class
# --------------------------------------------------------------------------------------

class LocalGeometryExperiment:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config if config is not None else DEFAULT_CONFIG

    def expand_stage(self, stage: str) -> List[Dict[str, Any]]:
        return expand_stage_conditions(self.config, stage)

    def run_stage(self, stage: str, output_dir: str | Path | None = None, workers: int | None = None,
                  only_condition: str | None = None) -> Dict[str, Any]:
        out_dir = Path(output_dir or self.config["global"]["output_dir"]) / f"stage_{stage}"
        out_dir.mkdir(parents=True, exist_ok=True)
        w = int(workers if workers is not None else self.config["global"].get("workers", 1))

        conditions = self.expand_stage(stage)
        if only_condition is not None:
            conditions = [c for c in conditions if c["id"] == only_condition]
            if not conditions:
                raise ValueError(f"Condition {only_condition!r} not found in stage {stage}")

        summaries = []
        for cond in conditions:
            print(f"[stage {stage}] running {cond['id']} ...", flush=True)
            summaries.append(run_condition(cond, out_dir, workers=w))

        criteria = fit_gap_criteria(summaries, self.config)
        stage_summary = {
            "stage": stage,
            "n_conditions": len(summaries),
            "conditions": [s["condition"]["id"] for s in summaries],
            "criteria": criteria,
        }
        (out_dir / f"stage_{stage}_summary.json").write_text(
            json.dumps(stage_summary, indent=2, default=json_default), encoding="utf-8"
        )
        return stage_summary

    def run_all(self, output_dir: str | Path | None = None, workers: int | None = None) -> Dict[str, Any]:
        out_dir = Path(output_dir or self.config["global"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        all_summaries = {}
        for stage in sorted(self.config["stages"].keys(), key=int):
            all_summaries[stage] = self.run_stage(stage, output_dir=out_dir, workers=workers)
        all_results = _load_condition_jsons(out_dir / "stage_0") + _load_condition_jsons(out_dir / "stage_1") + _load_condition_jsons(out_dir / "stage_2") + _load_condition_jsons(out_dir / "stage_3") + _load_condition_jsons(out_dir / "stage_4")
        global_summary = fit_gap_criteria(all_results, self.config)
        (out_dir / "global_gap_summary.json").write_text(
            json.dumps(global_summary, indent=2, default=json_default), encoding="utf-8"
        )
        return all_summaries


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run universal local-geometry path_kurtosis experiments.")
    p.add_argument("--stage", default="0", help="Stage to run: 0,1,2,3,4 or all")
    p.add_argument("--config", default=None, help="Optional JSON config path")
    p.add_argument("--out", default=None, help="Output directory (defaults to config/global/output_dir)")
    p.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    p.add_argument("--only-condition", default=None, help="Run only one expanded condition id")
    p.add_argument("--dump-config", default=None, help="Write the default JSON config and exit")
    return p.parse_args()


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def main() -> None:
    args = parse_args()
    if args.dump_config is not None:
        dump_default_config(args.dump_config)
        print(f"Wrote default config to {args.dump_config}")
        return

    cfg = load_config(args.config)
    exp = LocalGeometryExperiment(cfg)

    if args.stage == "all":
        summary = exp.run_all(output_dir=args.out, workers=args.workers)
        print(json.dumps(summary, indent=2, default=json_default))
        return

    summary = exp.run_stage(
        stage=args.stage,
        output_dir=args.out,
        workers=args.workers,
        only_condition=args.only_condition,
    )
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
