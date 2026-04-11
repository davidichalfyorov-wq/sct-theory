from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp

OUTDIR = Path('/mnt/data')


def alpha_tau_quadrature_unique(K: int = 6):
    """Unique tau-layers from the symmetric alpha-kernel.

    We integrate over alpha in [0, 1/2] and double the weights because
    P(alpha) and u(tau(alpha)) are symmetric under alpha -> 1-alpha.
    """
    x, w = leggauss(K)
    alpha = 0.25 * (x + 1.0)       # [-1,1] -> [0,1/2]
    w = 0.25 * w * 2.0             # Jacobian * symmetry factor
    tau = alpha * (1.0 - alpha)
    P = -89.0 / 24.0 + (43.0 / 6.0) * tau + (236.0 / 3.0) * tau ** 2
    idx = np.argsort(tau)
    alpha = alpha[idx]
    tau = tau[idx]
    w = w[idx]
    P = P[idx]
    c = w * P
    dtau = np.diff(np.concatenate(([0.0], tau)))
    return {
        'alpha': alpha,
        'tau': tau,
        'w': w,
        'P': P,
        'c': c,
        'dtau': dtau,
    }


def initial_guess(rho: np.ndarray, M: float, coeffs: dict, l: float = 2.0):
    K = len(coeffs['tau'])
    h2 = -2.0 * M / l ** 3
    f3 = 0.0

    F = rho.copy()
    Fp = np.ones_like(rho)
    H = 1.0 - 2.0 * M * rho ** 2 / (rho ** 3 + l ** 3)
    Hp = np.gradient(H, rho, edge_order=2)
    Fpp = np.gradient(Fp, rho, edge_order=2)
    Hpp = np.gradient(Hp, rho, edge_order=2)
    Wamp = (F ** 2 * Hpp - 2.0 * F * H * Fpp - 2.0 * F * Fp * Hp + 2.0 * H * Fp ** 2 - 2.0) / (6.0 * F ** 2)

    Y = [F, Fp, H, Hp]
    for tau_k in coeffs['tau']:
        uk = Wamp / (1.0 + 4.0 * tau_k)
        vk = np.gradient(uk, rho, edge_order=2)
        Y.extend([uk, vk])
    y0 = np.vstack(Y)

    p0 = np.concatenate(([h2, f3], np.zeros(K)))
    return y0, p0


def local_second_derivs(ycol: np.ndarray, coeffs: dict, Lam: float = 1.0):
    K = len(coeffs['tau'])
    F = ycol[0]
    Fp = ycol[1]
    H = ycol[2]
    Hp = ycol[3]
    u = np.array([ycol[4 + 2 * i] for i in range(K)], dtype=float)
    v = np.array([ycol[5 + 2 * i] for i in range(K)], dtype=float)

    c = coeffs['c']
    dtau = coeffs['dtau']

    J = float(np.dot(c, u))
    Jp = float(np.dot(c, v))

    # Unknown second derivatives s = [F'', H'', u1'', ..., uK'']
    n = K + 2
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    # Metric equations from the frozen-source reduced Lagrangian
    A[0, 0] = 4.0 * F * J - 2.0 * F
    A[0, 2:] = 2.0 * F ** 2 * c
    b[0] = -12.0 * F * Fp * Jp - 12.0 * J * Fp ** 2

    A[1, 0] = -16.0 * H * J - 4.0 * H
    A[1, 1] = 4.0 * F * J - 2.0 * F
    A[1, 2:] = -4.0 * F * H * c
    b[1] = 4.0 * F * Hp * Jp + 16.0 * H * Fp * Jp + 16.0 * J * Fp * Hp + 4.0 * Fp * Hp

    Acoef = Hp + 2.0 * H * Fp / F
    Scoef = 6.0 * H * (Fp / F) ** 2

    # u0 = Wamp(F, H, F', H', F'', H'')
    aF = -H / (3.0 * F)
    aH = 1.0 / 6.0
    a0 = -(Fp * Hp) / (3.0 * F) + H * Fp ** 2 / (3.0 * F ** 2) - 1.0 / (3.0 * F ** 2)

    for k in range(K):
        row = 2 + k
        lam = Lam ** 2 / dtau[k]
        if k == 0:
            A[row, 0] = -lam * aF
            A[row, 1] = -lam * aH
            A[row, 2 + k] = H
            b[row] = -Acoef * v[k] + Scoef * u[k] - lam * u[k] + lam * a0
        else:
            A[row, 2 + k] = H
            b[row] = -Acoef * v[k] + Scoef * u[k] - lam * (u[k] - u[k - 1])

    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(A, b, rcond=None)[0]

    Fpp = float(s[0])
    Hpp = float(s[1])
    upp = np.array(s[2:], dtype=float)
    u0 = aF * Fpp + aH * Hpp + a0
    Jpp = float(np.dot(c, upp))
    return Fpp, Hpp, u0, upp, J, Jp, Jpp


def rhs(rho: np.ndarray, y: np.ndarray, p: np.ndarray, coeffs: dict, Lam: float = 1.0):
    out = np.zeros_like(y)
    for j in range(y.shape[1]):
        Fpp, Hpp, _u0, upp, _J, _Jp, _Jpp = local_second_derivs(y[:, j], coeffs, Lam=Lam)
        out[0, j] = y[1, j]
        out[1, j] = Fpp
        out[2, j] = y[3, j]
        out[3, j] = Hpp
        K = len(coeffs['tau'])
        for k in range(K):
            out[4 + 2 * k, j] = y[5 + 2 * k, j]
            out[5 + 2 * k, j] = upp[k]
    return out


def bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray, eps: float, Rmax: float, M: float, coeffs: dict, Lam: float = 1.0):
    K = len(coeffs['tau'])
    h2 = p[0]
    f3 = p[1]
    a2 = p[2:]

    res = []
    # Center series with free parameters h2, f3, a2_k
    res.extend([
        ya[0] - (eps + f3 * eps ** 3),
        ya[1] - (1.0 + 3.0 * f3 * eps ** 2),
        ya[2] - (1.0 + h2 * eps ** 2),
        ya[3] - (2.0 * h2 * eps),
    ])
    for k in range(K):
        res.append(ya[4 + 2 * k] - a2[k] * eps ** 2)
        res.append(ya[5 + 2 * k] - 2.0 * a2[k] * eps)

    # Soft Schwarzschild data at the outer boundary
    res.extend([
        yb[0] - Rmax,
        yb[2] - (1.0 - 2.0 * M / Rmax),
    ])
    for k, tau_k in enumerate(coeffs['tau']):
        uR = -2.0 * M / Rmax ** 3 - 12.0 * M ** 2 * tau_k / (Lam ** 2 * Rmax ** 6)
        res.append(yb[4 + 2 * k] - uR)

    return np.array(res, dtype=float)


@dataclass
class Record:
    Rmax: float
    status: int
    message: str
    h2: float | None
    f3: float | None
    Hmin: float | None
    Hmax: float | None
    Fmin: float | None
    Fp_outer: float | None
    Hp_outer: float | None
    Jabs_max: float | None
    center_u_slopes: list[float] | None


def diagnostics(sol, coeffs: dict, Lam: float = 1.0):
    rho = np.geomspace(sol.x[0], sol.x[-1], 600)
    y = sol.sol(rho)
    F = y[0]
    Fp = y[1]
    H = y[2]
    Hp = y[3]
    K = len(coeffs['tau'])

    J = []
    for j in range(rho.size):
        _Fpp, _Hpp, _u0, _upp, Jj, _Jpj, _Jppj = local_second_derivs(y[:, j], coeffs, Lam=Lam)
        J.append(Jj)
    J = np.array(J)

    # Estimate center powers u_k ~ rho^s on the first few points.
    i1 = min(20, rho.size)
    slopes = []
    rfit = rho[:i1]
    for k in range(K):
        uk = y[4 + 2 * k, :i1]
        vals = np.abs(uk) + 1.0e-300
        coeff = np.polyfit(np.log(rfit), np.log(vals), 1)
        slopes.append(float(coeff[0]))

    return {
        'Hmin': float(np.min(H)),
        'Hmax': float(np.max(H)),
        'Fmin': float(np.min(F)),
        'Fp_outer': float(Fp[-1]),
        'Hp_outer': float(Hp[-1]),
        'Jabs_max': float(np.max(np.abs(J))),
        'center_u_slopes': slopes,
    }


def continue_soft(R_values, eps: float = 0.02, M: float = 1.0, K: int = 6, Lam: float = 1.0, l: float = 2.0, tol: float = 1e-2, max_nodes: int = 20000):
    coeffs = alpha_tau_quadrature_unique(K)
    records: list[Record] = []
    prev = None

    for Rmax in R_values:
        if prev is None:
            rho = np.geomspace(eps, Rmax, 140)
            y0, p0 = initial_guess(rho, M=M, coeffs=coeffs, l=l)
        else:
            rho = np.geomspace(eps, Rmax, max(180, prev.x.size))
            rho_common = np.minimum(rho, prev.x[-1])
            y0 = prev.sol(rho_common)
            mask = rho > prev.x[-1]
            if np.any(mask):
                y0[:, mask] = y0[:, np.searchsorted(rho, prev.x[-1]) - 1][:, None]
                y0[0, mask] = rho[mask]
                y0[1, mask] = 1.0
                y0[2, mask] = 1.0 - 2.0 * M / rho[mask]
                y0[3, mask] = 2.0 * M / rho[mask] ** 2
                for k, tau_k in enumerate(coeffs['tau']):
                    y0[4 + 2 * k, mask] = -2.0 * M / rho[mask] ** 3 - 12.0 * M ** 2 * tau_k / (Lam ** 2 * rho[mask] ** 6)
                    y0[5 + 2 * k, mask] = 6.0 * M / rho[mask] ** 4 + 72.0 * M ** 2 * tau_k / (Lam ** 2 * rho[mask] ** 7)
            p0 = prev.p

        sol = solve_bvp(
            lambda r, y, p: rhs(r, y, p, coeffs, Lam=Lam),
            lambda ya, yb, p: bc(ya, yb, p, eps, Rmax, M, coeffs, Lam=Lam),
            rho,
            y0,
            p=np.array(p0, dtype=float),
            tol=tol,
            max_nodes=max_nodes,
            verbose=0,
        )

        if sol.status == 0:
            diag = diagnostics(sol, coeffs, Lam=Lam)
            prev = sol
            rec = Record(
                Rmax=float(Rmax),
                status=int(sol.status),
                message=sol.message,
                h2=float(sol.p[0]),
                f3=float(sol.p[1]),
                Hmin=diag['Hmin'],
                Hmax=diag['Hmax'],
                Fmin=diag['Fmin'],
                Fp_outer=diag['Fp_outer'],
                Hp_outer=diag['Hp_outer'],
                Jabs_max=diag['Jabs_max'],
                center_u_slopes=diag['center_u_slopes'],
            )
        else:
            rec = Record(
                Rmax=float(Rmax),
                status=int(sol.status),
                message=sol.message,
                h2=None,
                f3=None,
                Hmin=None,
                Hmax=None,
                Fmin=None,
                Fp_outer=None,
                Hp_outer=None,
                Jabs_max=None,
                center_u_slopes=None,
            )
        records.append(rec)
        print(f"R={Rmax:6.1f} status={sol.status} h2={rec.h2} f3={rec.f3} Hmin={rec.Hmin}")

    return coeffs, records


def main():
    R_values = [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    coeffs, records = continue_soft(R_values)

    # Save machine-readable outputs
    coeffs_out = {k: np.asarray(v).tolist() for k, v in coeffs.items()}
    with open(OUTDIR / 'p12_soft_bvp_coeffs.json', 'w', encoding='utf-8') as f:
        json.dump(coeffs_out, f, indent=2)

    with open(OUTDIR / 'p12_soft_bvp_records.json', 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    # CSV summary
    header = 'Rmax,status,h2,f3,Hmin,Hmax,Fmin,Fp_outer,Hp_outer,Jabs_max\n'
    lines = [header]
    for r in records:
        lines.append(','.join([
            f'{r.Rmax}',
            f'{r.status}',
            '' if r.h2 is None else f'{r.h2}',
            '' if r.f3 is None else f'{r.f3}',
            '' if r.Hmin is None else f'{r.Hmin}',
            '' if r.Hmax is None else f'{r.Hmax}',
            '' if r.Fmin is None else f'{r.Fmin}',
            '' if r.Fp_outer is None else f'{r.Fp_outer}',
            '' if r.Hp_outer is None else f'{r.Hp_outer}',
            '' if r.Jabs_max is None else f'{r.Jabs_max}',
        ]) + '\n')
    (OUTDIR / 'p12_soft_bvp_summary.csv').write_text(''.join(lines), encoding='utf-8')

    # Simple plot
    import matplotlib.pyplot as plt
    R = [r.Rmax for r in records if r.h2 is not None]
    h2 = [r.h2 for r in records if r.h2 is not None]
    plt.figure(figsize=(6, 4))
    plt.plot(R, h2, marker='o')
    plt.xlabel('R_max')
    plt.ylabel('h2')
    plt.title('P12 soft full-formfactor BVP: h2 vs R_max')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'p12_soft_bvp_h2_vs_R.png', dpi=160)


if __name__ == '__main__':
    main()
