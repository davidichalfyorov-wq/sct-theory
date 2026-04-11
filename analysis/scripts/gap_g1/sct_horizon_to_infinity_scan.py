
from __future__ import annotations
import json, itertools, math, statistics, sys
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import importlib.util

sys.path.append("/mnt/data")

spec = importlib.util.spec_from_file_location("drv", "/mnt/data/sct_bvp_driver.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

with open("/mnt/data/sct_horizon_series_data.json", "r", encoding="utf-8") as f:
    hdata = json.load(f)

r0, f1, f2, f3, h2, h3 = sp.symbols('r0 f1 f2 f3 h2 h3')
locals_dict = {'r0':r0, 'f1':f1, 'f2':f2, 'f3':f3, 'h2':h2, 'h3':h3}
h4_expr = sp.sympify(hdata['expressions_h1_eq_1']['h4'], locals=locals_dict)
f4_expr = sp.sympify(hdata['expressions_h1_eq_1']['f4'], locals=locals_dict)
h5_expr = sp.sympify(hdata['expressions_h1_eq_1']['h5'], locals=locals_dict)
f5_expr = sp.sympify(hdata['expressions_h1_eq_1']['f5'], locals=locals_dict)
h6_expr = sp.sympify(hdata['expressions_h1_eq_1']['h6'], locals=locals_dict)

h4_fun = sp.lambdify((r0,f1,f2,f3,h2,h3), h4_expr, 'numpy')
f4_fun = sp.lambdify((r0,f1,f2,f3,h2,h3), f4_expr, 'numpy')
h5_fun = sp.lambdify((r0,f1,f2,f3,h2,h3), h5_expr, 'numpy')
f5_fun = sp.lambdify((r0,f1,f2,f3,h2,h3), f5_expr, 'numpy')
h6_fun = sp.lambdify((r0,f1,f2,f3,h2,h3), h6_expr, 'numpy')

def horizon_state(eps: float, pars):
    r0v, f1v, f2v, f3v, h2v, h3v = pars
    h4v = h4_fun(r0v, f1v, f2v, f3v, h2v, h3v)
    f4v = f4_fun(r0v, f1v, f2v, f3v, h2v, h3v)
    h5v = h5_fun(r0v, f1v, f2v, f3v, h2v, h3v)
    f5v = f5_fun(r0v, f1v, f2v, f3v, h2v, h3v)
    h6v = h6_fun(r0v, f1v, f2v, f3v, h2v, h3v)
    x = eps
    F = r0v + f1v*x + f2v*x**2 + f3v*x**3 + f4v*x**4 + f5v*x**5
    F1 = f1v + 2*f2v*x + 3*f3v*x**2 + 4*f4v*x**3 + 5*f5v*x**4
    F2 = 2*f2v + 6*f3v*x + 12*f4v*x**2 + 20*f5v*x**3
    F3 = 6*f3v + 24*f4v*x + 60*f5v*x**2
    F4 = 24*f4v + 120*f5v*x
    F5 = 120*f5v
    H = x + h2v*x**2 + h3v*x**3 + h4v*x**4 + h5v*x**5 + h6v*x**6
    H1 = 1 + 2*h2v*x + 3*h3v*x**2 + 4*h4v*x**3 + 5*h5v*x**4 + 6*h6v*x**5
    H2 = 2*h2v + 6*h3v*x + 12*h4v*x**2 + 20*h5v*x**3 + 30*h6v*x**4
    H3 = 6*h3v + 24*h4v*x + 60*h5v*x**2 + 120*h6v*x**3
    H4 = 24*h4v + 120*h5v*x + 360*h6v*x**2
    H5 = 120*h5v + 720*h6v*x
    return np.array([F,F1,F2,F3,F4,F5,H,H1,H2,H3,H4,H5], dtype=float)

def rhs_fun(rho, y):
    return np.array(drv.rhs(rho, y[:,None], None, Lam=1.0)).reshape(12)

def integrate_outside_diag(pars, eps=1e-3, R=60.0, max_step=0.2, rtol=1e-5, atol=1e-7):
    y0 = horizon_state(eps, pars)
    def ev_blow(rho, y):
        vals = [abs(y[0]), abs(y[1]), abs(y[6]), abs(y[7]), abs(y[2]), abs(y[8])]
        return 1e8 - max(vals)
    ev_blow.terminal = True
    ev_blow.direction = -1

    sol = solve_ivp(
        rhs_fun, (eps, R), y0,
        method="BDF", rtol=rtol, atol=atol, max_step=max_step,
        events=[ev_blow]
    )

    rho = float(sol.t[-1])
    F, F1, F2 = map(float, sol.y[:3, -1])
    H, H1 = map(float, sol.y[6:8, -1])
    N = H * F1**2
    N1 = H1*F1**2 + 2*H*F1*F2
    m = F/2 * (1 - N)
    m1 = F1/2 * (1 - N) - F/2 * N1
    uH = rho * H1 / H
    uF = rho * F2 / F1
    score = abs(math.log10(abs(N))) + abs(uH) + abs(uF) + math.log10(1 + abs(rho*m1)/(abs(m)+1))

    return {
        "status": int(sol.status),
        "rho_end": rho,
        "F": F,
        "F1": F1,
        "H": H,
        "N": N,
        "m": m,
        "m1": m1,
        "uH": uH,
        "uF": uF,
        "score": score,
        "blow": len(sol.t_events[0]) > 0,
    }

def simplify(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating, float, int, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out

def main():
    slice_rf = []
    for r0v, f1v in itertools.product([0.8, 1.0, 1.2], [0.8, 1.0, 1.2]):
        rec = integrate_outside_diag((r0v, f1v, 0.0, 0.0, 0.0, 0.0))
        rec.update({"r0": r0v, "f1": f1v, "f2": 0.0, "f3": 0.0, "h2": 0.0, "h3": 0.0})
        slice_rf.append(rec)

    slice_h = []
    for h2v, h3v in itertools.product([-0.2, 0.0, 0.2], [-0.1, 0.0, 0.1]):
        rec = integrate_outside_diag((1.0, 1.0, 0.0, 0.0, h2v, h3v))
        rec.update({"r0": 1.0, "f1": 1.0, "f2": 0.0, "f3": 0.0, "h2": h2v, "h3": h3v})
        slice_h.append(rec)

    slice_f = []
    for f2v, f3v in itertools.product([-0.2, 0.0, 0.2], [-0.1, 0.0, 0.1]):
        rec = integrate_outside_diag((1.0, 1.0, f2v, f3v, 0.0, 0.0))
        rec.update({"r0": 1.0, "f1": 1.0, "f2": f2v, "f3": f3v, "h2": 0.0, "h3": 0.0})
        slice_f.append(rec)

    rf5 = []
    for r0v, f1v in itertools.product(np.linspace(0.7, 1.3, 5), np.linspace(0.7, 1.3, 5)):
        rec = integrate_outside_diag((float(r0v), float(f1v), 0.0, 0.0, 0.0, 0.0))
        rec.update({"r0": float(r0v), "f1": float(f1v), "f2": 0.0, "f3": 0.0, "h2": 0.0, "h3": 0.0})
        rf5.append(rec)

    best5 = min(rf5, key=lambda d: d["score"])
    best_pars = (best5["r0"], best5["f1"], 0.0, 0.0, 0.0, 0.0)
    rows = [integrate_outside_diag(best_pars, R=R, rtol=1e-6, atol=1e-8) for R in [40, 60, 80, 100, 120]]

    allrecs = slice_rf + slice_h + slice_f
    uHs = [r["uH"] for r in allrecs]
    uFs = [r["uF"] for r in allrecs]
    uNs = [r["uH"] + 2*r["uF"] for r in allrecs]
    Ns = [r["N"] for r in allrecs]
    scores = [r["score"] for r in allrecs]

    out = {
        "model": "Pure Weyl a6 seed, Lambda=1, alpha_R=0",
        "slice_statistics_R60": {
            "num_runs": len(allrecs),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
            "score_mean": float(statistics.mean(scores)),
            "uH_mean": float(statistics.mean(uHs)),
            "uH_std": float(statistics.pstdev(uHs)),
            "uF_mean": float(statistics.mean(uFs)),
            "uF_std": float(statistics.pstdev(uFs)),
            "uN_mean": float(statistics.mean(uNs)),
            "uN_std": float(statistics.pstdev(uNs)),
            "N_min": float(min(Ns)),
            "N_max": float(max(Ns)),
        },
        "best_on_5x5_r0_f1_slice": simplify(best5),
        "slices_R60": {
            "r0_f1_3x3_f2_f3_h2_h3_zero": [simplify(r) for r in slice_rf],
            "h2_h3_3x3_at_r0_f1_eq_1_f2_f3_zero": [simplify(r) for r in slice_h],
            "f2_f3_3x3_at_r0_f1_eq_1_h2_h3_zero": [simplify(r) for r in slice_f],
            "r0_f1_5x5_f2_f3_h2_h3_zero": [simplify(r) for r in rf5],
        },
        "best_candidate_trajectory": {
            "parameters": {
                "r0": best5["r0"], "f1": best5["f1"], "f2": 0.0, "f3": 0.0, "h2": 0.0, "h3": 0.0
            },
            "records": [simplify(r) for r in rows]
        },
    }

    with open("/mnt/data/sct_horizon_to_infinity_scan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
