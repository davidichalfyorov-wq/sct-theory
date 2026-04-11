"""
Numerical BVP driver for the a6-truncated SCT spherical seed in modified Schwarzschild coordinates.

This driver implements the pure-Weyl a6 seed (alpha_R = 0) with:
    ds^2 = -H(rho) dt^2 + drho^2/H(rho) + F(rho)^2 dOmega^2

State vector:
    y = (F, F1, F2, F3, F4, F5, H, H1, H2, H3, H4, H5)
where Fn = d^n F / d rho^n, Hn = d^n H / d rho^n.

Boundary-value problem:
    smooth even center at rho = eps,
    soft asymptotic conditions at rho = Rmax:
        F(Rmax)=Rmax, F'(Rmax)=1,
        H(Rmax)=1-2M/Rmax, H'(Rmax)=2M/Rmax^2.

The solver uses scipy.integrate.solve_bvp with four unknown center parameters:
    p = (h2, h4, f3, f5)
from the center series
    F = rho + f3 rho^3 + f5 rho^5 + ...
    H = 1 + h2 rho^2 + h4 rho^4 + ...
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
import warnings

def f6_eval(f0,f1,f2,f3,f4,f5,h0,h1,h2,h3,h4,h5,Lam):
    return (1/248670)*(-24570*Lam**2*f0**6*h0*h4 - 49140*Lam**2*f0**6*h1*h3 + 24570*Lam**2*f0**6*h2**2 - 680400*Lam**2*f0**6*h2 - 98280*Lam**2*f0**5*f1*h1*h2 - 1360800*Lam**2*f0**5*f1*h1 - 2041200*Lam**2*f0**5*f2*h0 + 196560*Lam**2*f0**5*f2*h1**2 + 245700*Lam**2*f0**5*f3*h0*h1 + 49140*Lam**2*f0**5*f4*h0**2 + 98280*Lam**2*f0**4*f1**2*h0*h2 + 98280*Lam**2*f0**4*f1**2*h1**2 - 49140*Lam**2*f0**4*f1*f2*h0*h1 - 98280*Lam**2*f0**4*f1*f3*h0**2 + 49140*Lam**2*f0**4*f2**2*h0**2 - 196560*Lam**2*f0**3*f1**3*h0*h1 - 98280*Lam**2*f0**3*f1**2*f2*h0**2 + 98280*Lam**2*f0**2*f1**4*h0**2 - 98280*Lam**2*f0**2 + 49734*f0**6*h0*h1*h5 + 72759*f0**6*h0*h2*h4 + 147360*f0**6*h0*h3**2 + 99468*f0**6*h1**2*h4 + 120651*f0**6*h1*h2*h3 + 614*f0**6*h2**3 - 198936*f0**5*f1*h0**2*h5 - 684303*f0**5*f1*h0*h1*h4 - 1020468*f0**5*f1*h0*h2*h3 - 407082*f0**5*f1*h1**2*h3 - 281826*f0**5*f1*h1*h2**2 - 1148487*f0**5*f2*h0**2*h4 - 4901562*f0**5*f2*h0*h1*h3 - 2157903*f0**5*f2*h0*h2**2 - 2538276*f0**5*f2*h1**2*h2 - 2893782*f0**5*f3*h0**2*h3 - 8892255*f0**5*f3*h0*h1*h2 - 1790424*f0**5*f3*h1**3 - 3237315*f0**5*f4*h0**2*h2 - 4774464*f0**5*f4*h0*h1**2 - 2287764*f0**5*f5*h0**2*h1 - 11973*f0**4*f1**2*h0**2*h4 + 1204668*f0**4*f1**2*h0*h1*h3 + 864819*f0**4*f1**2*h0*h2**2 + 688908*f0**4*f1**2*h1**2*h2 + 1950678*f0**4*f1*f2*h0**2*h3 + 7398393*f0**4*f1*f2*h0*h1*h2 + 2092512*f0**4*f1*f2*h1**3 + 3111138*f0**4*f1*f3*h0**2*h2 + 4157394*f0**4*f1*f3*h0*h1**2 + 1832790*f0**4*f1*f4*h0**2*h1 + 198936*f0**4*f1*f5*h0**3 + 2891019*f0**4*f2**2*h0**2*h2 + 8362680*f0**4*f2**2*h0*h1**2 + 11621178*f0**4*f2*f3*h0**2*h1 + 1451496*f0**4*f2*f4*h0**3 + 1749900*f0**4*f3**2*h0**3 - 186963*f0**4*h0*h4 - 489972*f0**4*h1*h3 - 184200*f0**3*f1**3*h0**2*h3 - 1057308*f0**3*f1**3*h0*h1*h2 - 255424*f0**3*f1**3*h1**3 - 364716*f0**3*f1**2*f2*h0**2*h2 - 3300864*f0**3*f1**2*f2*h0*h1**2 + 246828*f0**3*f1**2*f3*h0**2*h1 + 504708*f0**3*f1**2*f4*h0**3 - 6218592*f0**3*f1*f2**2*h0**2*h1 - 3190344*f0**3*f1*f2*f3*h0**3 + 979944*f0**3*f1*h0*h3 + 979944*f0**3*f1*h1*h2 - 122800*f0**3*f2**3*h0**3 + 1469916*f0**3*f2*h0*h2 + 1959888*f0**3*f2*h1**2 + 2449860*f0**3*f3*h0*h1 + 489972*f0**3*f4*h0**2 - 543390*f0**2*f1**4*h0**2*h2 + 114204*f0**2*f1**4*h0*h1**2 - 3643476*f0**2*f1**3*f2*h0**2*h1 - 1834632*f0**2*f1**3*f3*h0**3 - 1355712*f0**2*f1**2*f2**2*h0**3 - 2449860*f0**2*f1**2*h0*h2 - 1959888*f0**2*f1**2*h1**2 - 9309468*f0**2*f1*f2*h0*h1 - 2939832*f0**2*f1*f3*h0**2 - 1469916*f0**2*f2**2*h0**2 + 606018*f0**2*h2 + 571020*f0*f1**5*h0**2*h1 + 2779578*f0*f1**4*f2*h0**3 + 6859608*f0*f1**3*h0*h1 + 8819496*f0*f1**2*f2*h0**2 - 2656164*f0*f1*h1 - 2050146*f0*f2*h0 + 67540*f1**6*h0**3 - 4899720*f1**4*h0**2 + 4822356*f1**2*h0 + 9824)/(f0**5*h0**3)

def h6_eval(f0,f1,f2,f3,f4,f5,h0,h1,h2,h3,h4,h5,Lam):
    return (1/248670)*(98280*Lam**2*f0**6*h0*h4 - 49140*Lam**2*f0**6*h1*h3 + 24570*Lam**2*f0**6*h2**2 - 680400*Lam**2*f0**6*h2 + 491400*Lam**2*f0**5*f1*h0*h3 - 98280*Lam**2*f0**5*f1*h1*h2 - 1360800*Lam**2*f0**5*f1*h1 - 491400*Lam**2*f0**5*f2*h0*h2 - 5443200*Lam**2*f0**5*f2*h0 + 196560*Lam**2*f0**5*f2*h1**2 - 491400*Lam**2*f0**5*f3*h0*h1 - 196560*Lam**2*f0**5*f4*h0**2 + 98280*Lam**2*f0**4*f1**2*h0*h2 + 98280*Lam**2*f0**4*f1**2*h1**2 - 1277640*Lam**2*f0**4*f1*f2*h0*h1 - 589680*Lam**2*f0**4*f1*f3*h0**2 + 294840*Lam**2*f0**4*f2**2*h0**2 - 196560*Lam**2*f0**3*f1**3*h0*h1 + 393120*Lam**2*f0**3*f1**2*f2*h0**2 + 98280*Lam**2*f0**2*f1**4*h0**2 - 98280*Lam**2*f0**2 - 696276*f0**6*h0*h1*h5 - 664041*f0**6*h0*h2*h4 + 32235*f0**6*h0*h3**2 + 99468*f0**6*h1**2*h4 + 120651*f0**6*h1*h2*h3 + 614*f0**6*h2**3 - 1690956*f0**5*f1*h0**2*h5 - 3065088*f0**5*f1*h0*h1*h4 - 1269138*f0**5*f1*h0*h2*h3 - 407082*f0**5*f1*h1**2*h3 - 281826*f0**5*f1*h1*h2**2 - 1042572*f0**5*f2*h0**2*h4 - 1991202*f0**5*f2*h0*h1*h3 + 1153092*f0**5*f2*h0*h2**2 - 2538276*f0**5*f2*h1**2*h2 - 195252*f0**5*f3*h0**2*h3 - 1114410*f0**5*f3*h0*h1*h2 - 1790424*f0**5*f3*h1**3 + 349980*f0**5*f4*h0**2*h2 - 1790424*f0**5*f4*h0*h1**2 - 298404*f0**5*f5*h0**2*h1 - 1527018*f0**4*f1**2*h0**2*h4 + 873108*f0**4*f1**2*h0*h1*h3 + 1744374*f0**4*f1**2*h0*h2**2 + 688908*f0**4*f1**2*h1**2*h2 + 3562428*f0**4*f1*f2*h0**2*h3 + 6113598*f0**4*f1*f2*h0*h1*h2 + 2092512*f0**4*f1*f2*h1**3 + 8204268*f0**4*f1*f3*h0**2*h2 + 4019244*f0**4*f1*f3*h0*h1**2 + 5765460*f0**4*f1*f4*h0**2*h1 + 1193616*f0**4*f1*f5*h0**3 - 8294526*f0**4*f2**2*h0**2*h2 - 1105200*f0**4*f2**2*h0*h1**2 - 11910372*f0**4*f2*f3*h0**2*h1 - 3153504*f0**4*f2*f4*h0**3 - 1363080*f0**4*f3**2*h0**3 + 1328082*f0**4*h0*h4 - 489972*f0**4*h1*h3 - 184200*f0**3*f1**3*h0**2*h3 - 1591488*f0**3*f1**3*h0*h1*h2 - 255424*f0**3*f1**3*h1**3 + 4921824*f0**3*f1**2*f2*h0**2*h2 - 1458864*f0**3*f1**2*f2*h0*h1**2 + 5533368*f0**3*f1**2*f3*h0**2*h1 + 1959888*f0**3*f1**2*f4*h0**3 + 4354488*f0**3*f1*f2**2*h0**2*h1 + 309456*f0**3*f1*f2*f3*h0**3 + 979944*f0**3*f1*h0*h3 + 979944*f0**3*f1*h1*h2 + 6103160*f0**3*f2**3*h0**3 - 5879664*f0**3*f2*h0*h2 + 1959888*f0**3*f2*h1**2 - 4899720*f0**3*f3*h0*h1 - 1959888*f0**3*f4*h0**2 - 2993250*f0**2*f1**4*h0**2*h2 - 843636*f0**2*f1**4*h0*h1**2 - 18729456*f0**2*f1**3*f2*h0**2*h1 - 6734352*f0**2*f1**3*f3*h0**3 - 10381512*f0**2*f1**2*f2**2*h0**3 - 1959888*f0**2*f1**2*h1**2 + 2939832*f0**2*f1*f2*h0*h1 + 1959888*f0**2*f1*f3*h0**2 + 5879664*f0**2*f2**2*h0**2 + 606018*f0**2*h2 + 5470740*f0*f1**5*h0**2*h1 + 14448648*f0*f1**4*f2*h0**3 + 1959888*f0*f1**3*h0*h1 - 5879664*f0*f1**2*f2*h0**2 - 2656164*f0*f1*h1 + 979944*f0*f2*h0 - 2382320*f1**6*h0**3 + 2372496*f1**2*h0 + 9824)/(f0**6*h0**2)

def center_series(eps, p):
    h2, h4, f3, f5 = p
    return np.array([
        eps + f3*eps**3 + f5*eps**5,
        1 + 3*f3*eps**2 + 5*f5*eps**4,
        6*f3*eps + 20*f5*eps**3,
        6*f3 + 60*f5*eps**2,
        120*f5*eps,
        120*f5,
        1 + h2*eps**2 + h4*eps**4,
        2*h2*eps + 4*h4*eps**3,
        2*h2 + 12*h4*eps**2,
        24*h4*eps,
        24*h4,
        0.0,
    ], dtype=float)

def initial_guess(rho, p, M, l=2.0):
    h2, h4, f3, f5 = p
    expf = np.exp(-rho**2 / l**2)

    F = rho + f3 * rho**3 * expf
    F1 = 1 + f3 * expf * (3*rho**2 - 2*rho**4 / l**2)
    F2 = f3 * expf * (6*rho - 14*rho**3 / l**2 + 4*rho**5 / l**4)
    F3 = f3 * expf * (6 - 54*rho**2 / l**2 + 48*rho**4 / l**4 - 8*rho**6 / l**6)
    F4 = f3 * expf * (-120*rho / l**2 + 220*rho**3 / l**4 - 88*rho**5 / l**6 + 16*rho**7 / l**8)
    F5 = f3 * expf * (-120 / l**2 + 900*rho**2 / l**4 - 920*rho**4 / l**6 + 240*rho**6 / l**8 - 32*rho**8 / l**10)

    H = 1 - 2*M*rho**2 / (rho**3 + l**3)
    H1 = np.gradient(H, rho, edge_order=2)
    H2 = np.gradient(H1, rho, edge_order=2)
    H3 = np.gradient(H2, rho, edge_order=2)
    H4 = np.gradient(H3, rho, edge_order=2)
    H5 = np.gradient(H4, rho, edge_order=2)

    return np.vstack([F,F1,F2,F3,F4,F5,H,H1,H2,H3,H4,H5])

def rhs(rho, y, p, Lam=1.0):
    F0,F1,F2,F3,F4,F5,H0,H1,H2,H3,H4,H5 = y
    F6 = f6_eval(F0,F1,F2,F3,F4,F5,H0,H1,H2,H3,H4,H5,Lam)
    H6 = h6_eval(F0,F1,F2,F3,F4,F5,H0,H1,H2,H3,H4,H5,Lam)
    return np.vstack([F1,F2,F3,F4,F5,F6,H1,H2,H3,H4,H5,H6])

def bc(ya, yb, p, eps, Rmax, M):
    left = ya - center_series(eps, p)
    right = np.array([
        yb[0] - Rmax,
        yb[1] - 1.0,
        yb[6] - (1.0 - 2.0*M/Rmax),
        yb[7] - (2.0*M/Rmax**2),
    ], dtype=float)
    return np.concatenate([left, right])

def solve_soft(M, Rmax=40.0, eps=0.02, Lam=1.0, p0=None, l=2.0, nodes=160, tol=2e-3, max_nodes=20000, verbose=0):
    if p0 is None:
        p0 = np.array([-2*M/l**3, 0.0, 0.0, 0.0], dtype=float)
    rho = np.geomspace(eps, Rmax, nodes)
    y0 = initial_guess(rho, p0, M, l=l)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_bvp(
            lambda r, y, p: rhs(r, y, p, Lam=Lam),
            lambda ya, yb, p: bc(ya, yb, p, eps, Rmax, M),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=verbose
        )
    return sol

def continue_soft(M, R_values, eps=0.02, Lam=1.0, l=2.0, tol=2e-3, max_nodes=30000, verbose=0):
    sols = []
    prev = None
    for i, Rmax in enumerate(R_values):
        if prev is None or prev.status != 0:
            sol = solve_soft(M, Rmax=Rmax, eps=eps, Lam=Lam, l=l, tol=tol, max_nodes=max_nodes, verbose=verbose)
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
                y_guess[0, mask] = rho[mask]
                y_guess[1, mask] = 1.0
                y_guess[2:6, mask] = 0.0
                y_guess[6, mask] = 1.0 - 2.0*M/rho[mask]
                y_guess[7, mask] = 2.0*M/rho[mask]**2
                y_guess[8:, mask] = 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_bvp(
                    lambda r, y, p: rhs(r, y, p, Lam=Lam),
                    lambda ya, yb, p: bc(ya, yb, p, eps, Rmax, M),
                    rho, y_guess, p=prev.p,
                    tol=tol, max_nodes=max_nodes, verbose=verbose
                )
        sols.append(sol)
        if sol.status == 0:
            prev = sol
    return sols

def summarize_solution(sol, M):
    if sol.status != 0:
        return {
            "status": int(sol.status),
            "message": sol.message,
        }
    rho = np.geomspace(sol.x[0], sol.x[-1], 600)
    y = sol.sol(rho)
    H = y[6]
    F = y[0]
    return {
        "status": int(sol.status),
        "message": sol.message,
        "Rmax": float(sol.x[-1]),
        "parameters": {
            "h2": float(sol.p[0]),
            "h4": float(sol.p[1]),
            "f3": float(sol.p[2]),
            "f5": float(sol.p[3]),
        },
        "H_min": float(H.min()),
        "H_at_Rmax": float(H[-1]),
        "F_minus_r_at_Rmax": float(F[-1] - sol.x[-1]),
        "M": float(M),
    }

if __name__ == "__main__":
    # Example continuation run
    M = 1.0
    R_values = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0]
    sols = continue_soft(M, R_values, verbose=1)
    for sol in sols:
        print(summarize_solution(sol, M))
