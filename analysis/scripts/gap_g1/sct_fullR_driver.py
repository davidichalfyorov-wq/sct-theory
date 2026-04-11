
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
import warnings

def f6_eval(f0,f1,f2,f3,f4,f5,h0,h1,h2,h3,h4,h5,Lam,alpha_R):
    return (1/248670)*(3402000*Lam**2*alpha_R*f0**6*h0*h4 + 2721600*Lam**2*alpha_R*f0**6*h1*h3 + 680400*Lam**2*alpha_R*f0**6*h2**2 + 16329600*Lam**2*alpha_R*f0**5*f1*h0*h3 + 13608000*Lam**2*alpha_R*f0**5*f1*h1*h2 + 44906400*Lam**2*alpha_R*f0**5*f2*h0*h2 + 21772800*Lam**2*alpha_R*f0**5*f2*h1**2 + 51710400*Lam**2*alpha_R*f0**5*f3*h0*h1 + 13608000*Lam**2*alpha_R*f0**5*f4*h0**2 - 9525600*Lam**2*alpha_R*f0**4*f1**2*h0*h2 - 5443200*Lam**2*alpha_R*f0**4*f1**2*h1**2 - 13608000*Lam**2*alpha_R*f0**4*f1*f2*h0*h1 - 2721600*Lam**2*alpha_R*f0**4*f1*f3*h0**2 + 5443200*Lam**2*alpha_R*f0**4*f2**2*h0**2 - 21772800*Lam**2*alpha_R*f0**3*f1**3*h0*h1 - 43545600*Lam**2*alpha_R*f0**3*f1**2*f2*h0**2 + 16329600*Lam**2*alpha_R*f0**3*f1*h1 + 16329600*Lam**2*alpha_R*f0**3*f2*h0 + 27216000*Lam**2*alpha_R*f0**2*f1**4*h0**2 - 24494400*Lam**2*alpha_R*f0**2*f1**2*h0 - 2721600*Lam**2*alpha_R*f0**2 - 24570*Lam**2*f0**6*h0*h4 - 49140*Lam**2*f0**6*h1*h3 + 24570*Lam**2*f0**6*h2**2 - 680400*Lam**2*f0**6*h2 - 98280*Lam**2*f0**5*f1*h1*h2 - 1360800*Lam**2*f0**5*f1*h1 - 2041200*Lam**2*f0**5*f2*h0 + 196560*Lam**2*f0**5*f2*h1**2 + 245700*Lam**2*f0**5*f3*h0*h1 + 49140*Lam**2*f0**5*f4*h0**2 + 98280*Lam**2*f0**4*f1**2*h0*h2 + 98280*Lam**2*f0**4*f1**2*h1**2 - 49140*Lam**2*f0**4*f1*f2*h0*h1 - 98280*Lam**2*f0**4*f1*f3*h0**2 + 49140*Lam**2*f0**4*f2**2*h0**2 - 196560*Lam**2*f0**3*f1**3*h0*h1 - 98280*Lam**2*f0**3*f1**2*f2*h0**2 + 98280*Lam**2*f0**2*f1**4*h0**2 - 98280*Lam**2*f0**2 + 49734*f0**6*h0*h1*h5 + 72759*f0**6*h0*h2*h4 + 147360*f0**6*h0*h3**2 + 99468*f0**6*h1**2*h4 + 120651*f0**6*h1*h2*h3 + 614*f0**6*h2**3 - 198936*f0**5*f1*h0**2*h5 - 684303*f0**5*f1*h0*h1*h4 - 1020468*f0**5*f1*h0*h2*h3 - 407082*f0**5*f1*h1**2*h3 - 281826*f0**5*f1*h1*h2**2 - 1148487*f0**5*f2*h0**2*h4 - 4901562*f0**5*f2*h0*h1*h3 - 2157903*f0**5*f2*h0*h2**2 - 2538276*f0**5*f2*h1**2*h2 - 2893782*f0**5*f3*h0**2*h3 - 8892255*f0**5*f3*h0*h1*h2 - 1790424*f0**5*f3*h1**3 - 3237315*f0**5*f4*h0**2*h2 - 4774464*f0**5*f4*h0*h1**2 - 2287764*f0**5*f5*h0**2*h1 - 11973*f0**4*f1**2*h0**2*h4 + 1204668*f0**4*f1**2*h0*h1*h3 + 864819*f0**4*f1**2*h0*h2**2 + 688908*f0**4*f1**2*h1**2*h2 + 1950678*f0**4*f1*f2*h0**2*h3 + 7398393*f0**4*f1*f2*h0*h1*h2 + 2092512*f0**4*f1*f2*h1**3 + 3111138*f0**4*f1*f3*h0**2*h2 + 4157394*f0**4*f1*f3*h0*h1**2 + 1832790*f0**4*f1*f4*h0**2*h1 + 198936*f0**4*f1*f5*h0**3 + 2891019*f0**4*f2**2*h0**2*h2 + 8362680*f0**4*f2**2*h0*h1**2 + 11621178*f0**4*f2*f3*h0**2*h1 + 1451496*f0**4*f2*f4*h0**3 + 1749900*f0**4*f3**2*h0**3 - 186963*f0**4*h0*h4 - 489972*f0**4*h1*h3 - 184200*f0**3*f1**3*h0**2*h3 - 1057308*f0**3*f1**3*h0*h1*h2 - 255424*f0**3*f1**3*h1**3 - 364716*f0**3*f1**2*f2*h0**2*h2 - 3300864*f0**3*f1**2*f2*h0*h1**2 + 246828*f0**3*f1**2*f3*h0**2*h1 + 504708*f0**3*f1**2*f4*h0**3 - 6218592*f0**3*f1*f2**2*h0**2*h1 - 3190344*f0**3*f1*f2*f3*h0**3 + 979944*f0**3*f1*h0*h3 + 979944*f0**3*f1*h1*h2 - 122800*f0**3*f2**3*h0**3 + 1469916*f0**3*f2*h0*h2 + 1959888*f0**3*f2*h1**2 + 2449860*f0**3*f3*h0*h1 + 489972*f0**3*f4*h0**2 - 543390*f0**2*f1**4*h0**2*h2 + 114204*f0**2*f1**4*h0*h1**2 - 3643476*f0**2*f1**3*f2*h0**2*h1 - 1834632*f0**2*f1**3*f3*h0**3 - 1355712*f0**2*f1**2*f2**2*h0**3 - 2449860*f0**2*f1**2*h0*h2 - 1959888*f0**2*f1**2*h1**2 - 9309468*f0**2*f1*f2*h0*h1 - 2939832*f0**2*f1*f3*h0**2 - 1469916*f0**2*f2**2*h0**2 + 606018*f0**2*h2 + 571020*f0*f1**5*h0**2*h1 + 2779578*f0*f1**4*f2*h0**3 + 6859608*f0*f1**3*h0*h1 + 8819496*f0*f1**2*f2*h0**2 - 2656164*f0*f1*h1 - 2050146*f0*f2*h0 + 67540*f1**6*h0**3 - 4899720*f1**4*h0**2 + 4822356*f1**2*h0 + 9824)/(f0**5*h0**3)

def h6_eval(f0,f1,f2,f3,f4,f5,h0,h1,h2,h3,h4,h5,Lam,alpha_R):
    return (1/248670)*(6804000*Lam**2*alpha_R*f0**6*h0*h4 + 2721600*Lam**2*alpha_R*f0**6*h1*h3 + 680400*Lam**2*alpha_R*f0**6*h2**2 + 29937600*Lam**2*alpha_R*f0**5*f1*h0*h3 + 13608000*Lam**2*alpha_R*f0**5*f1*h1*h2 + 92534400*Lam**2*alpha_R*f0**5*f2*h0*h2 + 21772800*Lam**2*alpha_R*f0**5*f2*h1**2 + 92534400*Lam**2*alpha_R*f0**5*f3*h0*h1 + 27216000*Lam**2*alpha_R*f0**5*f4*h0**2 - 29937600*Lam**2*alpha_R*f0**4*f1**2*h0*h2 - 5443200*Lam**2*alpha_R*f0**4*f1**2*h1**2 - 27216000*Lam**2*alpha_R*f0**4*f1*f2*h0*h1 - 16329600*Lam**2*alpha_R*f0**4*f1*f3*h0**2 + 32659200*Lam**2*alpha_R*f0**4*f2**2*h0**2 - 21772800*Lam**2*alpha_R*f0**3*f1**3*h0*h1 - 70761600*Lam**2*alpha_R*f0**3*f1**2*f2*h0**2 + 16329600*Lam**2*alpha_R*f0**3*f1*h1 + 16329600*Lam**2*alpha_R*f0**3*f2*h0 + 68040000*Lam**2*alpha_R*f0**2*f1**4*h0**2 - 65318400*Lam**2*alpha_R*f0**2*f1**2*h0 - 2721600*Lam**2*alpha_R*f0**2 + 98280*Lam**2*f0**6*h0*h4 - 49140*Lam**2*f0**6*h1*h3 + 24570*Lam**2*f0**6*h2**2 - 680400*Lam**2*f0**6*h2 + 491400*Lam**2*f0**5*f1*h0*h3 - 98280*Lam**2*f0**5*f1*h1*h2 - 1360800*Lam**2*f0**5*f1*h1 - 491400*Lam**2*f0**5*f2*h0*h2 - 5443200*Lam**2*f0**5*f2*h0 + 196560*Lam**2*f0**5*f2*h1**2 - 491400*Lam**2*f0**5*f3*h0*h1 - 196560*Lam**2*f0**5*f4*h0**2 + 98280*Lam**2*f0**4*f1**2*h0*h2 + 98280*Lam**2*f0**4*f1**2*h1**2 - 1277640*Lam**2*f0**4*f1*f2*h0*h1 - 589680*Lam**2*f0**4*f1*f3*h0**2 + 294840*Lam**2*f0**4*f2**2*h0**2 - 196560*Lam**2*f0**3*f1**3*h0*h1 + 393120*Lam**2*f0**3*f1**2*f2*h0**2 + 98280*Lam**2*f0**2*f1**4*h0**2 - 98280*Lam**2*f0**2 - 696276*f0**6*h0*h1*h5 - 664041*f0**6*h0*h2*h4 + 32235*f0**6*h0*h3**2 + 99468*f0**6*h1**2*h4 + 120651*f0**6*h1*h2*h3 + 614*f0**6*h2**3 - 1690956*f0**5*f1*h0**2*h5 - 3065088*f0**5*f1*h0*h1*h4 - 1269138*f0**5*f1*h0*h2*h3 - 407082*f0**5*f1*h1**2*h3 - 281826*f0**5*f1*h1*h2**2 - 1042572*f0**5*f2*h0**2*h4 - 1991202*f0**5*f2*h0*h1*h3 + 1153092*f0**5*f2*h0*h2**2 - 2538276*f0**5*f2*h1**2*h2 - 195252*f0**5*f3*h0**2*h3 - 1114410*f0**5*f3*h0*h1*h2 - 1790424*f0**5*f3*h1**3 + 349980*f0**5*f4*h0**2*h2 - 1790424*f0**5*f4*h0*h1**2 - 298404*f0**5*f5*h0**2*h1 - 1527018*f0**4*f1**2*h0**2*h4 + 873108*f0**4*f1**2*h0*h1*h3 + 1744374*f0**4*f1**2*h0*h2**2 + 688908*f0**4*f1**2*h1**2*h2 + 3562428*f0**4*f1*f2*h0**2*h3 + 6113598*f0**4*f1*f2*h0*h1*h2 + 2092512*f0**4*f1*f2*h1**3 + 8204268*f0**4*f1*f3*h0**2*h2 + 4019244*f0**4*f1*f3*h0*h1**2 + 5765460*f0**4*f1*f4*h0**2*h1 + 1193616*f0**4*f1*f5*h0**3 - 8294526*f0**4*f2**2*h0**2*h2 - 1105200*f0**4*f2**2*h0*h1**2 - 11910372*f0**4*f2*f3*h0**2*h1 - 3153504*f0**4*f2*f4*h0**3 - 1363080*f0**4*f3**2*h0**3 + 1328082*f0**4*h0*h4 - 489972*f0**4*h1*h3 - 184200*f0**3*f1**3*h0**2*h3 - 1591488*f0**3*f1**3*h0*h1*h2 - 255424*f0**3*f1**3*h1**3 + 4921824*f0**3*f1**2*f2*h0**2*h2 - 1458864*f0**3*f1**2*f2*h0*h1**2 + 5533368*f0**3*f1**2*f3*h0**2*h1 + 1959888*f0**3*f1**2*f4*h0**3 + 4354488*f0**3*f1*f2**2*h0**2*h1 + 309456*f0**3*f1*f2*f3*h0**3 + 979944*f0**3*f1*h0*h3 + 979944*f0**3*f1*h1*h2 + 6103160*f0**3*f2**3*h0**3 - 5879664*f0**3*f2*h0*h2 + 1959888*f0**3*f2*h1**2 - 4899720*f0**3*f3*h0*h1 - 1959888*f0**3*f4*h0**2 - 2993250*f0**2*f1**4*h0**2*h2 - 843636*f0**2*f1**4*h0*h1**2 - 18729456*f0**2*f1**3*f2*h0**2*h1 - 6734352*f0**2*f1**3*f3*h0**3 - 10381512*f0**2*f1**2*f2**2*h0**3 - 1959888*f0**2*f1**2*h1**2 + 2939832*f0**2*f1*f2*h0*h1 + 1959888*f0**2*f1*f3*h0**2 + 5879664*f0**2*f2**2*h0**2 + 606018*f0**2*h2 + 5470740*f0*f1**5*h0**2*h1 + 14448648*f0*f1**4*f2*h0**3 + 1959888*f0*f1**3*h0*h1 - 5879664*f0*f1**2*f2*h0**2 - 2656164*f0*f1*h1 + 979944*f0*f2*h0 - 2382320*f1**6*h0**3 + 2372496*f1**2*h0 + 9824)/(f0**6*h0**2)

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

def center_recurrence(p, alpha_R):
    h2c, h4c, f3c, f5c = p
    eqF = -155131200*alpha_R*f3c**2 + 106142400*alpha_R*f3c*h2c + 272160000*alpha_R*f5c + 81648000*alpha_R*h4c - 93997260*f3c**3 + 161884170*f3c**2*h2c - 737100*f3c**2 + 355100760*f3c*f5c - 28199178*f3c*h2c**2 + 1474200*f3c*h2c - 80071740*f3c*h4c - 2041200*f3c - 256461660*f5c*h2c + 2457000*f5c + 256959*h2c**3 + 1740690*h2c*h4c - 1020600*h2c - 737100*h4c
    eqH = 21092400*alpha_R*f3c**2 + 13608000*alpha_R*f3c*h2c + 11340000*alpha_R*f5c + 3402000*alpha_R*h4c + 15666210*f3c**3 - 26806626*f3c**2*h2c + 245700*f3c**2 - 59183460*f3c*f5c + 5396139*f3c*h2c**2 - 491400*f3c*h2c + 7542990*f3c*h4c - 340200*f3c + 43710660*f5c*h2c - 819000*f5c - 3564270*h2c*h4c + 245700*h4c
    return np.array([eqF, eqH], dtype=float)

def initial_guess(rho, p, M, l=2.0):
    h2, h4, f3, f5 = p
    expf = np.exp(-rho**2 / l**2)
    F = rho + f3 * rho**3 * expf + f5 * rho**5 * expf
    F1 = np.gradient(F, rho, edge_order=2)
    F2 = np.gradient(F1, rho, edge_order=2)
    F3 = np.gradient(F2, rho, edge_order=2)
    F4 = np.gradient(F3, rho, edge_order=2)
    F5 = np.gradient(F4, rho, edge_order=2)
    Hc = 1 + h2 * rho**2 * expf + h4 * rho**4 * expf
    Hasy = 1 - 2*M*rho**2 / (rho**3 + l**3)
    w = np.exp(-(rho/l)**2)
    H = w*Hc + (1-w)*Hasy
    H1 = np.gradient(H, rho, edge_order=2)
    H2 = np.gradient(H1, rho, edge_order=2)
    H3 = np.gradient(H2, rho, edge_order=2)
    H4 = np.gradient(H3, rho, edge_order=2)
    H5 = np.gradient(H4, rho, edge_order=2)
    return np.vstack([F,F1,F2,F3,F4,F5,H,H1,H2,H3,H4,H5])

def rhs(rho, y, p_unused, Lam=1.0, alpha_R=0.0):
    F0,F1,F2,F3,F4,F5,H0,H1,H2,H3,H4,H5 = y
    F6 = f6_eval(F0,F1,F2,F3,F4,F5,H0,H1,H2,H3,H4,H5,Lam,alpha_R)
    H6 = h6_eval(F0,F1,F2,F3,F4,F5,H0,H1,H2,H3,H4,H5,Lam,alpha_R)
    return np.vstack([F1,F2,F3,F4,F5,F6,H1,H2,H3,H4,H5,H6])

CF = 4.5416507936508
CH = 27.0939682539683

def asymp_strict(rho, M):
    a5 = -CF * M**2
    b6 = CH * M**2
    F = rho + a5 / np.power(rho, 5)
    F1 = 1.0 - 5.0 * a5 / np.power(rho, 6)
    H = 1.0 - 2.0 * M / rho + b6 / np.power(rho, 6)
    H1 = 2.0 * M / np.power(rho, 2) - 6.0 * b6 / np.power(rho, 7)
    return F, F1, H, H1

LEFT_IDX = np.array([0,1,2,3,4,6,7,8,9,10], dtype=int)

def bc_soft(ya, yb, p, eps, Rmax, M, alpha_R):
    left = ya[LEFT_IDX] - center_series(eps, p)[LEFT_IDX]
    rec = center_recurrence(p, alpha_R)
    right = np.array([
        yb[0] - Rmax,
        yb[1] - 1.0,
        yb[6] - (1.0 - 2.0*M/Rmax),
        yb[7] - (2.0*M/Rmax**2),
    ], dtype=float)
    return np.concatenate([left, rec, right])

def bc_strict(ya, yb, p, eps, Rmax, M, alpha_R):
    left = ya[LEFT_IDX] - center_series(eps, p)[LEFT_IDX]
    rec = center_recurrence(p, alpha_R)
    F, F1, H, H1 = asymp_strict(Rmax, M)
    right = np.array([yb[0] - F, yb[1] - F1, yb[6] - H, yb[7] - H1], dtype=float)
    return np.concatenate([left, rec, right])

def solve_soft(M, alpha_R, Rmax=40.0, eps=0.02, Lam=1.0, p0=None, l=2.0,
               nodes=180, tol=2e-3, max_nodes=50000, verbose=0):
    if p0 is None:
        p0 = np.array([-2*M/l**3, 0.0, 0.0, 0.0], dtype=float)
    rho = np.geomspace(eps, Rmax, nodes)
    y0 = initial_guess(rho, p0, M, l=l)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_bvp(
            lambda r, y, p: rhs(r, y, p, Lam=Lam, alpha_R=alpha_R),
            lambda ya, yb, p: bc_soft(ya, yb, p, eps, Rmax, M, alpha_R),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=verbose
        )
    return sol

def solve_strict(M, alpha_R, Rmax=40.0, eps=0.02, Lam=1.0, p0=None, l=2.0,
                 nodes=180, tol=2e-3, max_nodes=50000, verbose=0):
    if p0 is None:
        p0 = np.array([-2*M/l**3, 0.0, 0.0, 0.0], dtype=float)
    rho = np.geomspace(eps, Rmax, nodes)
    y0 = initial_guess(rho, p0, M, l=l)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_bvp(
            lambda r, y, p: rhs(r, y, p, Lam=Lam, alpha_R=alpha_R),
            lambda ya, yb, p: bc_strict(ya, yb, p, eps, Rmax, M, alpha_R),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=verbose
        )
    return sol

def continue_bvp(kind, M, alpha_R, R_values, eps=0.02, Lam=1.0, tol=2e-3,
                 max_nodes=50000, verbose=0):
    solve_one = solve_soft if kind == "soft" else solve_strict
    bc_fun = bc_soft if kind == "soft" else bc_strict
    sols = []
    prev = None
    for Rmax in R_values:
        if prev is None or prev.status != 0:
            sol = solve_one(M, alpha_R, Rmax=Rmax, eps=eps, Lam=Lam, tol=tol,
                            max_nodes=max_nodes, verbose=verbose)
        else:
            rho = np.geomspace(eps, Rmax, 240)
            y_guess = np.zeros((12, rho.size))
            r_prev_max = prev.x[-1]
            rho_common = np.minimum(rho, r_prev_max)
            y_prev = prev.sol(rho_common)
            y_guess[:] = y_prev
            mask = rho > r_prev_max
            if np.any(mask):
                y_guess[:, mask] = y_prev[:, -1, None]
                if kind == "soft":
                    y_guess[0, mask] = rho[mask]
                    y_guess[1, mask] = 1.0
                    y_guess[2:6, mask] = 0.0
                    y_guess[6, mask] = 1.0 - 2.0*M/rho[mask]
                    y_guess[7, mask] = 2.0*M/rho[mask]**2
                    y_guess[8:, mask] = 0.0
                else:
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
                    lambda r, y, p: rhs(r, y, p, Lam=Lam, alpha_R=alpha_R),
                    lambda ya, yb, p: bc_fun(ya, yb, p, eps, Rmax, M, alpha_R),
                    rho, y_guess, p=prev.p,
                    tol=tol, max_nodes=max_nodes, verbose=verbose
                )
        sols.append(sol)
        if sol.status == 0:
            prev = sol
    return sols

def misner_sharp_profile(sol, n=1000):
    rho = np.geomspace(sol.x[0], sol.x[-1], n)
    y = sol.sol(rho)
    F = y[0]
    F1 = y[1]
    H = y[6]
    N = H * F1**2
    m = 0.5 * F * (1.0 - N)
    return rho, F, H, N, m

def mass_profile_stats(sol, frac_list=(0.1,0.5,0.9), n=1000):
    rho, F, H, N, m = misner_sharp_profile(sol, n=n)
    stats = {}
    for frac in frac_list:
        inds = np.where(m >= frac)[0]
        stats[f"rho{int(100*frac):02d}"] = float(rho[inds[0]]) if len(inds) else None
    return stats, rho, m

def summary_record(sol, M, alpha_R):
    rec = {
        "Rmax": float(sol.x[-1]),
        "status": int(sol.status),
        "message": sol.message,
        "M": float(M),
        "alpha_R": float(alpha_R),
    }
    if sol.status == 0:
        stats, _, _ = mass_profile_stats(sol)
        rho = np.geomspace(sol.x[0], sol.x[-1], 600)
        y = sol.sol(rho)
        rec.update({
            "h2": float(sol.p[0]),
            "h4": float(sol.p[1]),
            "f3": float(sol.p[2]),
            "f5": float(sol.p[3]),
            "center_eqF": float(center_recurrence(sol.p, alpha_R)[0]),
            "center_eqH": float(center_recurrence(sol.p, alpha_R)[1]),
            "Hmin": float(y[6].min()),
            **stats
        })
    return rec
