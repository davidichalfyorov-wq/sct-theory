
from __future__ import annotations
import json, itertools, sys
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

def horizon_state_norm(eps: float, pars):
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

def integrate_inside(pars, eps=1e-3, L=2.0, max_step=5e-2, rtol=1e-5, atol=1e-7):
    y0 = horizon_state_norm(-eps, pars)
    def ev_H(rho, y): return y[6]
    ev_H.terminal = True
    ev_H.direction = 1
    def ev_Fsmall(rho, y): return y[0] - 1e-3
    ev_Fsmall.terminal = True
    ev_Fsmall.direction = -1
    def ev_blow(rho, y): return 1e6 - max(abs(y[0]), abs(y[1]), abs(y[6]), abs(y[7]))
    ev_blow.terminal = True
    ev_blow.direction = -1
    sol = solve_ivp(rhs_fun, (-eps, -L), y0, method="BDF", rtol=rtol, atol=atol,
                    max_step=max_step, events=[ev_H, ev_Fsmall, ev_blow])
    rec = {
        "status": int(sol.status),
        "message": sol.message,
        "t_end": float(sol.t[-1]),
        "H_end": float(sol.y[6,-1]),
        "F_end": float(sol.y[0,-1]),
        "F1_end": float(sol.y[1,-1]),
    }
    for i, name in enumerate(["Hzero","Fsmall","blow"]):
        if len(sol.t_events[i]) > 0:
            rec[name] = {
                "rho": float(sol.t_events[i][0]),
                "F": float(sol.y_events[i][0][0]),
                "H": float(sol.y_events[i][0][6]),
                "F1": float(sol.y_events[i][0][1]),
            }
    return rec

def main():
    sample_pars = [
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (1.3, 0.8, 0.1, -0.05, -0.2, 0.1),
        (0.9, 1.2, -0.15, 0.08, 0.25, -0.12),
    ]
    scan_a = list(itertools.product([0.8,1.0,1.2], [0.8,1.0,1.2]))
    scan_b = list(itertools.product([-0.2,0.0,0.2], [-0.1,0.0,0.1]))

    out = {
        "sample_local_horizon_to_inside_runs": [],
        "coarse_scan_r0_f1_with_f2_f3_h2_h3_zero": [],
        "coarse_scan_h2_h3_at_r0_f1_equal_1": [],
    }
    for pars in sample_pars:
        rec = {"r0":pars[0],"f1":pars[1],"f2":pars[2],"f3":pars[3],"h2":pars[4],"h3":pars[5]}
        rec.update(integrate_inside(pars))
        out["sample_local_horizon_to_inside_runs"].append(rec)
    for r0v, f1v in scan_a:
        rec = {"r0":r0v,"f1":f1v}
        rec.update(integrate_inside((r0v, f1v, 0.0, 0.0, 0.0, 0.0)))
        out["coarse_scan_r0_f1_with_f2_f3_h2_h3_zero"].append(rec)
    for h2v, h3v in scan_b:
        rec = {"r0":1.0,"f1":1.0,"h2":h2v,"h3":h3v}
        rec.update(integrate_inside((1.0,1.0,0.0,0.0,h2v,h3v)))
        out["coarse_scan_h2_h3_at_r0_f1_equal_1"].append(rec)

    with open("/mnt/data/sct_horizon_to_center_scan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
