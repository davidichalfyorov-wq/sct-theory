"""
P12 soft BVP: R_max convergence study (v2).

Extends the existing P12 full-formfactor soft BVP to large R_max values
(300, 500, 1000) using continuation from previous solutions.
Saves results JSON and convergence plots.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp

# ---------- paths ----------------------------------------------------------
PROJECT = Path(r"F:\Black Mesa Research Facility\Main Facility\Physics department\SCT Theory")
OUTDIR = PROJECT / "analysis" / "results" / "gap_g1"
FIGDIR = PROJECT / "analysis" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------- quadrature (copied from p12) -----------------------------------

def alpha_tau_quadrature_unique(K: int = 6):
    x, w = leggauss(K)
    alpha = 0.25 * (x + 1.0)
    w = 0.25 * w * 2.0
    tau = alpha * (1.0 - alpha)
    P = -89.0 / 24.0 + (43.0 / 6.0) * tau + (236.0 / 3.0) * tau ** 2
    idx = np.argsort(tau)
    return {
        'alpha': alpha[idx], 'tau': tau[idx], 'w': w[idx],
        'P': P[idx], 'c': w[idx] * P[idx],
        'dtau': np.diff(np.concatenate(([0.0], tau[idx]))),
    }


# ---------- initial guess --------------------------------------------------

def initial_guess(rho, M, coeffs, l=2.0):
    K = len(coeffs['tau'])
    h2 = -2.0 * M / l ** 3
    f3 = 0.0
    F = rho.copy(); Fp = np.ones_like(rho)
    H = 1.0 - 2.0 * M * rho ** 2 / (rho ** 3 + l ** 3)
    Hp = np.gradient(H, rho, edge_order=2)
    Fpp = np.gradient(Fp, rho, edge_order=2)
    Hpp = np.gradient(Hp, rho, edge_order=2)
    Wamp = (F**2*Hpp - 2*F*H*Fpp - 2*F*Fp*Hp + 2*H*Fp**2 - 2) / (6*F**2)
    Y = [F, Fp, H, Hp]
    for tau_k in coeffs['tau']:
        uk = Wamp / (1.0 + 4.0 * tau_k)
        vk = np.gradient(uk, rho, edge_order=2)
        Y.extend([uk, vk])
    y0 = np.vstack(Y)
    p0 = np.concatenate(([h2, f3], np.zeros(K)))
    return y0, p0


# ---------- ODE system (copied from p12) -----------------------------------

def local_second_derivs(ycol, coeffs, Lam=1.0):
    K = len(coeffs['tau'])
    F, Fp, H, Hp = ycol[0], ycol[1], ycol[2], ycol[3]
    u = np.array([ycol[4 + 2*i] for i in range(K)])
    v = np.array([ycol[5 + 2*i] for i in range(K)])
    c = coeffs['c']; dtau = coeffs['dtau']
    J = float(np.dot(c, u)); Jp = float(np.dot(c, v))
    n = K + 2
    A = np.zeros((n, n)); b = np.zeros(n)
    A[0, 0] = 4*F*J - 2*F;  A[0, 2:] = 2*F**2*c
    b[0] = -12*F*Fp*Jp - 12*J*Fp**2
    A[1, 0] = -16*H*J - 4*H;  A[1, 1] = 4*F*J - 2*F;  A[1, 2:] = -4*F*H*c
    b[1] = 4*F*Hp*Jp + 16*H*Fp*Jp + 16*J*Fp*Hp + 4*Fp*Hp
    Acoef = Hp + 2*H*Fp/F;  Scoef = 6*H*(Fp/F)**2
    aF = -H/(3*F);  aH = 1/6
    a0 = -(Fp*Hp)/(3*F) + H*Fp**2/(3*F**2) - 1/(3*F**2)
    for k in range(K):
        row = 2 + k; lam = Lam**2 / dtau[k]
        if k == 0:
            A[row, 0] = -lam*aF;  A[row, 1] = -lam*aH;  A[row, 2+k] = H
            b[row] = -Acoef*v[k] + Scoef*u[k] - lam*u[k] + lam*a0
        else:
            A[row, 2+k] = H
            b[row] = -Acoef*v[k] + Scoef*u[k] - lam*(u[k] - u[k-1])
    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(A, b, rcond=None)[0]
    Fpp = float(s[0]); Hpp = float(s[1]); upp = np.array(s[2:])
    u0 = aF*Fpp + aH*Hpp + a0
    Jpp = float(np.dot(c, upp))
    return Fpp, Hpp, u0, upp, J, Jp, Jpp


def rhs(rho, y, p, coeffs, Lam=1.0):
    out = np.zeros_like(y)
    K = len(coeffs['tau'])
    for j in range(y.shape[1]):
        Fpp, Hpp, _u0, upp, *_ = local_second_derivs(y[:, j], coeffs, Lam=Lam)
        out[0, j] = y[1, j]; out[1, j] = Fpp
        out[2, j] = y[3, j]; out[3, j] = Hpp
        for k in range(K):
            out[4+2*k, j] = y[5+2*k, j]; out[5+2*k, j] = upp[k]
    return out


def bc(ya, yb, p, eps, Rmax, M, coeffs, Lam=1.0):
    K = len(coeffs['tau']); h2 = p[0]; f3 = p[1]; a2 = p[2:]
    res = [
        ya[0] - (eps + f3*eps**3), ya[1] - (1 + 3*f3*eps**2),
        ya[2] - (1 + h2*eps**2), ya[3] - 2*h2*eps,
    ]
    for k in range(K):
        res.append(ya[4+2*k] - a2[k]*eps**2)
        res.append(ya[5+2*k] - 2*a2[k]*eps)
    res.extend([yb[0] - Rmax, yb[2] - (1 - 2*M/Rmax)])
    for k, tau_k in enumerate(coeffs['tau']):
        uR = -2*M/Rmax**3 - 12*M**2*tau_k/(Lam**2*Rmax**6)
        res.append(yb[4+2*k] - uR)
    return np.array(res)


# ---------- diagnostics ----------------------------------------------------

@dataclass
class Record:
    Rmax: float; status: int; message: str
    h2: float | None; f3: float | None
    Hmin: float | None; Hmax: float | None
    Fmin: float | None; Fp_outer: float | None
    Hp_outer: float | None; Jabs_max: float | None
    center_u_slopes: list[float] | None
    solve_time_s: float | None = None


def diagnostics(sol, coeffs, Lam=1.0):
    rho = np.geomspace(sol.x[0], sol.x[-1], 600)
    y = sol.sol(rho)
    F, Fp, H, Hp = y[0], y[1], y[2], y[3]
    K = len(coeffs['tau'])
    J = []
    for j in range(rho.size):
        *_, Jj, _Jpj, _Jppj = local_second_derivs(y[:, j], coeffs, Lam=Lam)
        J.append(Jj)
    J = np.array(J)
    i1 = min(20, rho.size); rfit = rho[:i1]; slopes = []
    for k in range(K):
        uk = y[4+2*k, :i1]; vals = np.abs(uk) + 1e-300
        coeff = np.polyfit(np.log(rfit), np.log(vals), 1)
        slopes.append(float(coeff[0]))
    return {
        'Hmin': float(np.min(H)), 'Hmax': float(np.max(H)),
        'Fmin': float(np.min(F)), 'Fp_outer': float(Fp[-1]),
        'Hp_outer': float(Hp[-1]), 'Jabs_max': float(np.max(np.abs(J))),
        'center_u_slopes': slopes,
    }


# ---------- continuation solver --------------------------------------------

def continue_soft(R_values, eps=0.02, M=1.0, K=6, Lam=1.0, l=2.0,
                  tol=1e-2, max_nodes=30000):
    coeffs = alpha_tau_quadrature_unique(K)
    records = []
    prev = None
    for Rmax in R_values:
        t0 = time.time()
        if prev is None:
            rho = np.geomspace(eps, Rmax, 140)
            y0, p0 = initial_guess(rho, M=M, coeffs=coeffs, l=l)
        else:
            n_pts = max(200, prev.x.size, int(Rmax / 2))
            n_pts = min(n_pts, 2000)  # cap mesh size for initial guess
            rho = np.geomspace(eps, Rmax, n_pts)
            rho_common = np.minimum(rho, prev.x[-1])
            y0 = prev.sol(rho_common)
            mask = rho > prev.x[-1]
            if np.any(mask):
                last_idx = max(0, np.searchsorted(rho, prev.x[-1]) - 1)
                y0[:, mask] = y0[:, last_idx][:, None]
                y0[0, mask] = rho[mask]
                y0[1, mask] = 1.0
                y0[2, mask] = 1 - 2*M/rho[mask]
                y0[3, mask] = 2*M/rho[mask]**2
                for k, tau_k in enumerate(coeffs['tau']):
                    y0[4+2*k, mask] = -2*M/rho[mask]**3 - 12*M**2*tau_k/(Lam**2*rho[mask]**6)
                    y0[5+2*k, mask] = 6*M/rho[mask]**4 + 72*M**2*tau_k/(Lam**2*rho[mask]**7)
            p0 = prev.p

        print(f"\n--- Solving R_max={Rmax:.0f} (mesh={rho.size}, max_nodes={max_nodes}) ---")
        sol = solve_bvp(
            lambda r, y, p: rhs(r, y, p, coeffs, Lam=Lam),
            lambda ya, yb, p: bc(ya, yb, p, eps, Rmax, M, coeffs, Lam=Lam),
            rho, y0, p=np.array(p0, dtype=float),
            tol=tol, max_nodes=max_nodes, verbose=2,
        )
        dt = time.time() - t0

        if sol.status == 0:
            diag = diagnostics(sol, coeffs, Lam=Lam)
            prev = sol
            rec = Record(
                Rmax=float(Rmax), status=int(sol.status), message=sol.message,
                h2=float(sol.p[0]), f3=float(sol.p[1]),
                Hmin=diag['Hmin'], Hmax=diag['Hmax'], Fmin=diag['Fmin'],
                Fp_outer=diag['Fp_outer'], Hp_outer=diag['Hp_outer'],
                Jabs_max=diag['Jabs_max'], center_u_slopes=diag['center_u_slopes'],
                solve_time_s=round(dt, 1),
            )
        else:
            rec = Record(
                Rmax=float(Rmax), status=int(sol.status), message=sol.message,
                h2=None, f3=None, Hmin=None, Hmax=None, Fmin=None,
                Fp_outer=None, Hp_outer=None, Jabs_max=None,
                center_u_slopes=None, solve_time_s=round(dt, 1),
            )
        records.append(rec)
        print(f"  status={sol.status} h2={rec.h2} Hmin={rec.Hmin} "
              f"nodes={sol.x.size} time={dt:.1f}s")
    return coeffs, records


# ---------- plotting -------------------------------------------------------

def make_plots(all_records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    R = [r['Rmax'] for r in all_records if r['h2'] is not None]
    h2 = [r['h2'] for r in all_records if r['h2'] is not None]
    Hmin = [r['Hmin'] for r in all_records if r['Hmin'] is not None]
    Hmax = [r['Hmax'] for r in all_records if r['Hmax'] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # h2 vs R_max
    ax = axes[0]
    ax.plot(R, h2, 'o-', color='#2166ac', markersize=6)
    ax.set_xlabel(r'$R_{\max}$', fontsize=12)
    ax.set_ylabel(r'$h_2$', fontsize=12)
    ax.set_title(r'$h_2$ vs $R_{\max}$ (P12 soft BVP)', fontsize=13)
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # H_min, H_max vs R_max
    ax = axes[1]
    ax.plot(R, Hmin, 's-', color='#d6604d', markersize=6, label=r'$H_{\min}$')
    ax.plot(R, Hmax, '^-', color='#4393c3', markersize=6, label=r'$H_{\max}$')
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5, label='Schwarzschild (flat)')
    ax.axhline(0.0, ls=':', color='black', alpha=0.5, label='horizon')
    ax.set_xlabel(r'$R_{\max}$', fontsize=12)
    ax.set_ylabel(r'$H$', fontsize=12)
    ax.set_title(r'$H_{\min}, H_{\max}$ vs $R_{\max}$', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGDIR / 'v2_rmax_convergence.png'
    plt.savefig(fig_path, dpi=180)
    print(f"\nFigure saved: {fig_path}")
    plt.close()


# ---------- main -----------------------------------------------------------

def main():
    # Full continuation chain: start from small R to build up solution,
    # then push to 300, 500, 1000
    R_warmup = [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    R_new = [300.0, 500.0, 1000.0]

    # Also try intermediate steps to help continuation
    R_all = R_warmup + [250.0, 300.0, 400.0, 500.0, 700.0, 1000.0]

    print("=" * 60)
    print("P12 soft BVP: R_max convergence study (v2)")
    print(f"R_values: {R_all}")
    print(f"Parameters: M=1, K=6, Lam=1, eps=0.02, tol=1e-2, max_nodes=30000")
    print("=" * 60)

    coeffs, records = continue_soft(
        R_all, eps=0.02, M=1.0, K=6, Lam=1.0, l=2.0,
        tol=1e-2, max_nodes=30000,
    )

    # Load previous records for combined plot
    prev_path = OUTDIR / 'p12_soft_bvp_records.json'
    if prev_path.exists():
        with open(prev_path, encoding='utf-8') as f:
            prev_records = json.load(f)
    else:
        prev_records = []

    # Merge: use new results (they re-solve the warmup too)
    new_dicts = [asdict(r) for r in records]

    # Build combined set keyed by Rmax (new results override old)
    combined = {}
    for r in prev_records:
        combined[r['Rmax']] = r
    for r in new_dicts:
        combined[r['Rmax']] = r
    all_records = sorted(combined.values(), key=lambda x: x['Rmax'])

    # Save
    out_path = OUTDIR / 'v2_rmax_convergence.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Rmax':>8} {'status':>7} {'h2':>12} {'f3':>12} {'Hmin':>10} {'Hmax':>10} {'time_s':>8}")
    print("-" * 80)
    for r in all_records:
        h2s = f"{r['h2']:.4f}" if r['h2'] is not None else "FAIL"
        f3s = f"{r['f3']:.4f}" if r['f3'] is not None else "FAIL"
        hms = f"{r['Hmin']:.6f}" if r['Hmin'] is not None else "FAIL"
        hxs = f"{r['Hmax']:.6f}" if r['Hmax'] is not None else "FAIL"
        ts = f"{r.get('solve_time_s', '?')}"
        print(f"{r['Rmax']:8.0f} {r['status']:7d} {h2s:>12} {f3s:>12} {hms:>10} {hxs:>10} {ts:>8}")
    print("=" * 80)

    # Plot
    make_plots(all_records)

    # Analysis
    converged = [r for r in all_records if r['h2'] is not None]
    if len(converged) >= 3:
        last3_hmin = [r['Hmin'] for r in converged[-3:]]
        last3_h2 = [r['h2'] for r in converged[-3:]]
        print(f"\nLast 3 H_min values: {last3_hmin}")
        print(f"Last 3 h2 values: {last3_h2}")
        hmin_spread = max(last3_hmin) - min(last3_hmin)
        h2_spread = max(last3_h2) - min(last3_h2)
        print(f"H_min spread: {hmin_spread:.6f}")
        print(f"h2 spread: {h2_spread:.4f}")
        if last3_hmin[-1] > 0.5:
            print("\n>>> H_min stays well above 0 => HORIZONLESS solution persists")
        elif last3_hmin[-1] < 0.1:
            print("\n>>> H_min approaching 0 => horizon may form")
        else:
            print(f"\n>>> H_min = {last3_hmin[-1]:.4f}, intermediate range, convergence unclear")


if __name__ == '__main__':
    main()
