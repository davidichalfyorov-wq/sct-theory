from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import p12_full_formfactor_soft_bvp as p12

OUT = Path('/mnt/data')


def initial_guess_fixed_h2(rho, h2_fixed, M0=1.0, coeffs=None, f3=0.0):
    if coeffs is None:
        coeffs = p12.alpha_tau_quadrature_unique(6)
    if h2_fixed < 0 and M0 > 0:
        l = (2.0 * M0 / (-h2_fixed)) ** (1.0 / 3.0)
    else:
        l = 2.0
    l = float(np.clip(l, 0.15, 30.0))
    F = rho + f3 * rho**3 * np.exp(-(rho / l) ** 2)
    Fp = np.gradient(F, rho, edge_order=2)
    H = 1.0 - 2.0 * M0 * rho**2 / (rho**3 + l**3)
    h2_guess = -2.0 * M0 / l**3
    if abs(h2_guess - h2_fixed) > 1e-8:
        w = np.exp(-(rho / max(2 * l, 1e-6)) ** 2)
        H += (h2_fixed - h2_guess) * rho**2 * w
    Hp = np.gradient(H, rho, edge_order=2)
    Fpp = np.gradient(Fp, rho, edge_order=2)
    Hpp = np.gradient(Hp, rho, edge_order=2)
    Wamp = (F**2 * Hpp - 2.0 * F * H * Fpp - 2.0 * F * Fp * Hp + 2.0 * H * Fp**2 - 2.0) / (6.0 * F**2)
    Y = [F, Fp, H, Hp]
    a2 = []
    eps = rho[0]
    for tau_k in coeffs['tau']:
        uk = Wamp / (1.0 + 4.0 * tau_k)
        vk = np.gradient(uk, rho, edge_order=2)
        Y.extend([uk, vk])
        a2.append(float(uk[0] / eps**2))
    y0 = np.vstack(Y)
    return y0, np.array(a2, dtype=float)


def bc_fixed_h2_fixedM_relaxedF(ya, yb, p, eps, Rmax, h2_fixed, M_fixed, coeffs, Lam=1.0):
    K = len(coeffs['tau'])
    f3 = p[0]
    a2 = p[1:]
    res = []
    res.extend([
        ya[0] - (eps + f3 * eps ** 3),
        ya[1] - (1.0 + 3.0 * f3 * eps ** 2),
        ya[2] - (1.0 + h2_fixed * eps ** 2),
        ya[3] - (2.0 * h2_fixed * eps),
    ])
    for k in range(K):
        res.append(ya[4 + 2 * k] - a2[k] * eps**2)
        res.append(ya[5 + 2 * k] - 2.0 * a2[k] * eps)
    res.append(yb[2] - (1.0 - 2.0 * M_fixed / Rmax))
    for k, tau_k in enumerate(coeffs['tau']):
        uR = -2.0 * M_fixed / Rmax**3 - 12.0 * M_fixed**2 * tau_k / (Lam**2 * Rmax**6)
        res.append(yb[4 + 2 * k] - uR)
    return np.array(res, dtype=float)


def solve_fixed_h2_fixedM_relaxedF(h2_fixed, M_fixed=1.0, Rmax=50.0, eps=0.02, K=6, Lam=1.0, nodes=180, tol=3e-2, max_nodes=30000, prev=None):
    coeffs = p12.alpha_tau_quadrature_unique(K)
    if prev is None:
        rho = np.geomspace(eps, Rmax, nodes)
        y0, a2 = initial_guess_fixed_h2(rho, h2_fixed, M0=M_fixed, coeffs=coeffs)
        p0 = np.concatenate(([0.0], a2))
    else:
        rho = np.geomspace(eps, Rmax, max(nodes, prev.x.size))
        y0 = prev.sol(np.minimum(rho, prev.x[-1]))
        if np.any(rho > prev.x[-1]):
            mask = rho > prev.x[-1]
            y0[:, mask] = y0[:, np.searchsorted(rho, prev.x[-1]) - 1][:, None]
            y0[2, mask] = 1.0 - 2.0 * M_fixed / rho[mask]
            y0[3, mask] = 2.0 * M_fixed / rho[mask] ** 2
            for k, tau_k in enumerate(coeffs['tau']):
                y0[4 + 2 * k, mask] = -2.0 * M_fixed / rho[mask] ** 3 - 12.0 * M_fixed**2 * tau_k / (Lam**2 * rho[mask] ** 6)
                y0[5 + 2 * k, mask] = 6.0 * M_fixed / rho[mask] ** 4 + 72.0 * M_fixed**2 * tau_k / (Lam**2 * rho[mask] ** 7)
        p0 = prev.p.copy()
    sol = solve_bvp(
        lambda r, y, p: p12.rhs(r, y, p, coeffs, Lam=Lam),
        lambda ya, yb, p: bc_fixed_h2_fixedM_relaxedF(ya, yb, p, eps, Rmax, h2_fixed, M_fixed, coeffs, Lam=Lam),
        rho, y0, p=np.array(p0, dtype=float), tol=tol, max_nodes=max_nodes, verbose=0,
    )
    return sol, coeffs


def bc_fixed_h2_free_logM(ya, yb, p, eps, Rmax, h2_fixed, coeffs, Lam=1.0):
    K = len(coeffs['tau'])
    q = p[0]
    M = float(np.exp(q))
    f3 = p[1]
    a2 = p[2:]
    res = []
    res.extend([
        ya[0] - (eps + f3 * eps ** 3),
        ya[1] - (1.0 + 3.0 * f3 * eps ** 2),
        ya[2] - (1.0 + h2_fixed * eps ** 2),
        ya[3] - (2.0 * h2_fixed * eps),
    ])
    for k in range(K):
        res.append(ya[4 + 2 * k] - a2[k] * eps**2)
        res.append(ya[5 + 2 * k] - 2.0 * a2[k] * eps)
    res.extend([
        yb[0] - Rmax,
        yb[2] - (1.0 - 2.0 * M / Rmax),
    ])
    for k, tau_k in enumerate(coeffs['tau']):
        uR = -2.0 * M / Rmax**3 - 12.0 * M**2 * tau_k / (Lam**2 * Rmax**6)
        res.append(yb[4 + 2 * k] - uR)
    return np.array(res, dtype=float)


def solve_fixed_h2_free_logM(h2_fixed, M0=0.1, Rmax=50.0, eps=0.02, K=6, Lam=1.0, nodes=180, tol=2e-2, max_nodes=30000):
    coeffs = p12.alpha_tau_quadrature_unique(K)
    rho = np.geomspace(eps, Rmax, nodes)
    y0, a2 = initial_guess_fixed_h2(rho, h2_fixed, M0=M0, coeffs=coeffs)
    p0 = np.concatenate(([np.log(max(M0, 1e-6)), 0.0], a2))
    sol = solve_bvp(
        lambda r, y, p: p12.rhs(r, y, p, coeffs, Lam=Lam),
        lambda ya, yb, p: bc_fixed_h2_free_logM(ya, yb, p, eps, Rmax, h2_fixed, coeffs, Lam=Lam),
        rho, y0, p=np.array(p0, dtype=float), tol=tol, max_nodes=max_nodes, verbose=0,
    )
    return sol, coeffs


def horizon_data(rho, H):
    roots = []
    for i in range(len(rho) - 1):
        if H[i] == 0:
            roots.append(float(rho[i]))
        elif H[i] * H[i + 1] < 0:
            r = rho[i] - H[i] * (rho[i + 1] - rho[i]) / (H[i + 1] - H[i])
            roots.append(float(r))
    return roots


def diag_fixedM(sol, coeffs, M_fixed=1.0, Lam=1.0, dense=4000):
    rho = np.geomspace(sol.x[0], sol.x[-1], dense)
    y = sol.sol(rho)
    F, Fp, H, Hp = y[0], y[1], y[2], y[3]
    roots = horizon_data(rho, H)
    d = p12.diagnostics(sol, coeffs, Lam=Lam)
    return {
        'status': int(sol.status),
        'message': sol.message,
        'f3': float(sol.p[0]),
        'Hmin': float(np.min(H)),
        'Hmax': float(np.max(H)),
        'num_horizons': len(roots),
        'horizon_radii': roots,
        'F_outer': float(F[-1]),
        'F_outer_mismatch': float(F[-1] - rho[-1]),
        'Fp_outer': float(Fp[-1]),
        'Hp_outer': float(Hp[-1]),
        'Hp_outer_target': float(2.0 * M_fixed / rho[-1]**2),
        'center_u_slopes': d['center_u_slopes'],
    }


def diag_freeM(sol, coeffs, Lam=1.0, dense=4000):
    rho = np.geomspace(sol.x[0], sol.x[-1], dense)
    y = sol.sol(rho)
    F, Fp, H, Hp = y[0], y[1], y[2], y[3]
    roots = horizon_data(rho, H)
    d = p12.diagnostics(sol, coeffs, Lam=Lam)
    M = float(np.exp(sol.p[0]))
    return {
        'status': int(sol.status),
        'message': sol.message,
        'M': M,
        'f3': float(sol.p[1]),
        'Hmin': float(np.min(H)),
        'Hmax': float(np.max(H)),
        'num_horizons': len(roots),
        'horizon_radii': roots,
        'F_outer': float(F[-1]),
        'Fp_outer': float(Fp[-1]),
        'Hp_outer': float(Hp[-1]),
        'Hp_outer_target': float(2.0 * M / rho[-1]**2),
        'center_u_slopes': d['center_u_slopes'],
    }


def run_fixedM_scan():
    h2_values = [-0.01, -0.1, -0.5, -1.0, -5.0, -10.0, -50.0, -100.0]
    rows = []
    prev = None
    for h2 in h2_values:
        sol, coeffs = solve_fixed_h2_fixedM_relaxedF(h2, M_fixed=1.0, prev=prev)
        if sol.status == 0:
            prev = sol
            d = diag_fixedM(sol, coeffs)
        else:
            prev = None
            d = {'status': int(sol.status), 'message': sol.message}
        d['h2_fixed'] = h2
        rows.append(d)
        print('fixedM', h2, d['status'], d.get('Hmin'), d.get('num_horizons'))
    return rows


def run_freeM_scan():
    # Seed map chosen from successful exploratory runs.
    h2_values = [-0.01, -0.1, -0.5, -1.0, -5.0, -10.0, -50.0, -100.0]
    seed_map = {
        -0.01: 0.1,
        -0.1: 0.1,
        -0.5: 0.1,
        -1.0: 0.1,
        -5.0: 0.1,
        -10.0: 0.1,
        -50.0: 0.1,
        -100.0: 0.1,
    }
    rows = []
    for h2 in h2_values:
        try:
            sol, coeffs = solve_fixed_h2_free_logM(h2, M0=seed_map[h2])
            if sol.status == 0:
                d = diag_freeM(sol, coeffs)
            else:
                d = {'status': int(sol.status), 'message': sol.message}
        except Exception as e:
            d = {'status': -1, 'message': str(e)}
        d['h2_fixed'] = h2
        d['M0_seed'] = seed_map[h2]
        rows.append(d)
        print('freeM', h2, d['status'], d.get('M'), d.get('Hmin'), d.get('num_horizons'))
    return rows


def save_csv(rows, filename, fields):
    lines = [','.join(fields) + '\n']
    for r in rows:
        vals = []
        for f in fields:
            v = r.get(f, '')
            if isinstance(v, list):
                vals.append('"' + ';'.join(f'{x:.12g}' for x in v) + '"')
            else:
                vals.append(str(v))
        lines.append(','.join(vals) + '\n')
    (OUT / filename).write_text(''.join(lines), encoding='utf-8')


def make_plots(fixed_rows, free_rows):
    # Hmin vs |h2|
    plt.figure(figsize=(6, 4))
    x = [abs(r['h2_fixed']) for r in fixed_rows if r.get('status') == 0]
    y = [r['Hmin'] for r in fixed_rows if r.get('status') == 0]
    plt.semilogx(x, y, marker='o', label='M=1, relaxed F')
    x2 = [abs(r['h2_fixed']) for r in free_rows if r.get('status') == 0]
    y2 = [r['Hmin'] for r in free_rows if r.get('status') == 0]
    if x2:
        plt.semilogx(x2, y2, marker='s', label='M free (>0)')
    plt.axhline(0.0, color='k', lw=1, ls='--')
    plt.xlabel(r'$|h_2|$')
    plt.ylabel(r'$H_{\min}$')
    plt.title('Negative-$h_2$ scan at $R_{max}=50$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'p12_negative_h2_Hmin_scan.png', dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    x = [abs(r['h2_fixed']) for r in fixed_rows if r.get('status') == 0]
    y = [r['F_outer_mismatch'] for r in fixed_rows if r.get('status') == 0]
    plt.semilogx(x, y, marker='o')
    plt.xlabel(r'$|h_2|$')
    plt.ylabel(r'$F(R_{max})-R_{max}$')
    plt.title('AF mismatch in fixed-M relaxed-F scan')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'p12_negative_h2_Fmismatch.png', dpi=160)
    plt.close()


def main():
    fixed_rows = run_fixedM_scan()
    free_rows = run_freeM_scan()

    with open(OUT / 'p12_negative_h2_fixedM_relaxedF.json', 'w', encoding='utf-8') as f:
        json.dump(fixed_rows, f, indent=2)
    with open(OUT / 'p12_negative_h2_freeM.json', 'w', encoding='utf-8') as f:
        json.dump(free_rows, f, indent=2)

    save_csv(fixed_rows, 'p12_negative_h2_fixedM_relaxedF.csv', [
        'h2_fixed', 'status', 'Hmin', 'Hmax', 'num_horizons', 'F_outer', 'F_outer_mismatch',
        'Fp_outer', 'Hp_outer', 'Hp_outer_target', 'f3', 'message'
    ])
    save_csv(free_rows, 'p12_negative_h2_freeM.csv', [
        'h2_fixed', 'M0_seed', 'status', 'M', 'Hmin', 'Hmax', 'num_horizons',
        'F_outer', 'Fp_outer', 'Hp_outer', 'Hp_outer_target', 'f3', 'message'
    ])

    make_plots(fixed_rows, free_rows)

if __name__ == '__main__':
    main()
