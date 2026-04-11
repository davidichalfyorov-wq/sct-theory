
from __future__ import annotations

import json
import warnings

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from sct_bvp_driver import rhs, center_series, initial_guess, f6_eval, h6_eval

# Leading strict asymptotic coefficients extracted from the algebraic recurrence:
#   F = r + a5/r^5 + ...
#   H = 1 - 2M/r + b6/r^6 + ...
CF = 4.5416507936508
CH = 27.0939682539683

def asymp_strict(rho: np.ndarray | float, M: float) -> tuple[np.ndarray | float, ...]:
    a5 = -CF * M**2
    b6 = CH * M**2
    F = rho + a5 / np.power(rho, 5)
    F1 = 1.0 - 5.0 * a5 / np.power(rho, 6)
    H = 1.0 - 2.0 * M / rho + b6 / np.power(rho, 6)
    H1 = 2.0 * M / np.power(rho, 2) - 6.0 * b6 / np.power(rho, 7)
    return F, F1, H, H1

def bc_strict(ya: np.ndarray, yb: np.ndarray, p: np.ndarray, eps: float, Rmax: float, M: float) -> np.ndarray:
    left = ya - center_series(eps, p)
    F, F1, H, H1 = asymp_strict(Rmax, M)
    right = np.array([yb[0] - F, yb[1] - F1, yb[6] - H, yb[7] - H1], dtype=float)
    return np.concatenate([left, right])

def solve_strict(
    M: float,
    Rmax: float = 40.0,
    eps: float = 0.02,
    Lam: float = 1.0,
    p0: np.ndarray | None = None,
    l: float = 2.0,
    nodes: int = 180,
    tol: float = 2e-3,
    max_nodes: int = 50000,
    verbose: int = 0,
):
    if p0 is None:
        p0 = np.array([-2*M/l**3, 0.0, 0.0, 0.0], dtype=float)
    rho = np.geomspace(eps, Rmax, nodes)
    y0 = initial_guess(rho, p0, M, l=l)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_bvp(
            lambda r, y, p: rhs(r, y, p, Lam=Lam),
            lambda ya, yb, p: bc_strict(ya, yb, p, eps, Rmax, M),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=verbose
        )
    return sol

def continue_strict(M: float, R_values: list[float], eps: float = 0.02, Lam: float = 1.0,
                    tol: float = 2e-3, max_nodes: int = 50000, verbose: int = 0):
    sols = []
    prev = None
    for Rmax in R_values:
        if prev is None or prev.status != 0:
            sol = solve_strict(M, Rmax=Rmax, eps=eps, Lam=Lam, tol=tol, max_nodes=max_nodes, verbose=verbose)
        else:
            rho = np.geomspace(eps, Rmax, 220)
            y_guess = np.zeros((12, rho.size))
            r_prev_max = prev.x[-1]
            rho_common = np.minimum(rho, r_prev_max)
            y_prev = prev.sol(rho_common)
            y_guess[:] = y_prev
            mask = rho > r_prev_max
            if np.any(mask):
                y_guess[:, mask] = y_prev[:, -1, None]
                F, F1, H, H1 = asymp_strict(rho[mask], M)
                y_guess[0, mask] = F
                y_guess[1, mask] = F1
                y_guess[2:6, mask] = 0.0
                y_guess[6, mask] = H
                y_guess[7, mask] = H1
                y_guess[8:, mask] = 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_bvp(
                    lambda r, y, p: rhs(r, y, p, Lam=Lam),
                    lambda ya, yb, p: bc_strict(ya, yb, p, eps, Rmax, M),
                    rho, y_guess, p=prev.p,
                    tol=tol, max_nodes=max_nodes, verbose=verbose
                )
        sols.append(sol)
        if sol.status == 0:
            prev = sol
    return sols

def misner_sharp_profile(sol, n: int = 1000):
    rho = np.geomspace(sol.x[0], sol.x[-1], n)
    y = sol.sol(rho)
    F = y[0]
    F1 = y[1]
    H = y[6]
    m = 0.5 * F * (1.0 - H * F1**2)
    return rho, F, H, m

def mass_profile_stats(sol, frac_list=(0.1, 0.5, 0.9), n: int = 1000):
    rho, F, H, m = misner_sharp_profile(sol, n=n)
    stats = {}
    for frac in frac_list:
        inds = np.where(m >= frac)[0]
        stats[f"rho{int(100*frac):02d}"] = float(rho[inds[0]]) if len(inds) else None
    return stats, rho, m

def summary_record(sol, M: float):
    rec = {
        "Rmax": float(sol.x[-1]),
        "status": int(sol.status),
        "message": sol.message,
        "M": float(M),
    }
    if sol.status == 0:
        stats, _, _ = mass_profile_stats(sol)
        rec.update({
            "h2": float(sol.p[0]),
            "h4": float(sol.p[1]),
            "f3": float(sol.p[2]),
            "f5": float(sol.p[3]),
            "Hmin": float(sol.sol(sol.x)[6].min()),
            **stats
        })
    return rec

if __name__ == "__main__":
    M = 1.0
    R_values = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 220.0]
    sols = continue_strict(M, R_values, verbose=0)
    data = [summary_record(sol, M) for sol in sols]
    with open("/mnt/data/sct_strict_scan_M1_manual.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
