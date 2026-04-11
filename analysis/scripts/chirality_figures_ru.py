# ruff: noqa: E402, I001
"""
Russian-language figures for Paper 3 (chirality theorem).

Figure 1: pq cross-term convergence (fig_pq_convergence_ru.pdf)
Figure 2: Crossed chirality scatter (fig_crossed_chirality_ru.pdf)

Author: David Alfyorov
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy import einsum

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

D = 4

# ---------- gamma matrices (Euclidean chiral basis) ----------

I2 = np.eye(2, dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)
sigma_pauli = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
], dtype=complex)


def block(A, B, C, DD):
    return np.block([[A, B], [C, DD]])


gam = np.zeros((D, 4, 4), dtype=complex)
for j in range(3):
    gam[j] = block(Z2, -1j * sigma_pauli[j], 1j * sigma_pauli[j], Z2)
gam[3] = block(Z2, I2, I2, Z2)

g5 = gam[0] @ gam[1] @ gam[2] @ gam[3]
P_L = 0.5 * (np.eye(4) + g5)
P_R = 0.5 * (np.eye(4) - g5)

# sigma = (1/4)[gamma^r, gamma^s]  (standard, no i)
sig = np.zeros((D, D, 4, 4), dtype=complex)
for a in range(D):
    for b in range(D):
        sig[a, b] = 0.25 * (gam[a] @ gam[b] - gam[b] @ gam[a])


def thooft_symbols():
    eta = np.zeros((3, D, D))
    eb = np.zeros((3, D, D))
    eta[0, 0, 1] = 1; eta[0, 1, 0] = -1; eta[0, 2, 3] = 1; eta[0, 3, 2] = -1
    eta[1, 0, 2] = 1; eta[1, 2, 0] = -1; eta[1, 3, 1] = 1; eta[1, 1, 3] = -1
    eta[2, 0, 3] = 1; eta[2, 3, 0] = -1; eta[2, 1, 2] = 1; eta[2, 2, 1] = -1
    eb[0, 0, 1] = 1; eb[0, 1, 0] = -1; eb[0, 2, 3] = -1; eb[0, 3, 2] = 1
    eb[1, 0, 2] = 1; eb[1, 2, 0] = -1; eb[1, 3, 1] = -1; eb[1, 1, 3] = 1
    eb[2, 0, 3] = 1; eb[2, 3, 0] = -1; eb[2, 1, 2] = -1; eb[2, 2, 1] = 1
    return eta, eb


eta, eb = thooft_symbols()


def rnd_ts3(rng):
    A = rng.standard_normal((3, 3))
    A = (A + A.T) / 2
    A -= np.trace(A) / 3 * np.eye(3)
    return A


def mk_weyl(Wp, Wm):
    C = np.zeros((D, D, D, D))
    for i in range(3):
        for j in range(3):
            C += Wp[i, j] * einsum('ab,cd->abcd', eta[i], eta[j])
            C += Wm[i, j] * einsum('ab,cd->abcd', eb[i], eb[j])
    return C


def build_eps():
    from itertools import product as iproduct
    e = np.zeros((D, D, D, D))
    for a, b, c, d in iproduct(range(D), repeat=4):
        if len({a, b, c, d}) == 4:
            p = [a, b, c, d]
            s = 1
            for i in range(4):
                for j in range(i + 1, 4):
                    if p[i] > p[j]:
                        s *= -1
            e[a, b, c, d] = s
    return e


eps = build_eps()


def sd_decompose(C):
    sC = 0.5 * einsum('abef,efcd->abcd', eps, C)
    return 0.5 * (C + sC), 0.5 * (C - sC)


def compute_pq(Cp, Cm):
    p = float(einsum('abcd,abcd->', Cp, Cp))
    q = float(einsum('abcd,abcd->', Cm, Cm))
    return p, q


def mk_omega(C):
    O = np.zeros((D, D, 4, 4), dtype=complex)
    for m in range(D):
        for n in range(D):
            for r in range(D):
                for s in range(D):
                    O[m, n] += 0.25 * C[m, n, r, s] * sig[r, s]
    return O


def compute_tr_osq_sq(C):
    """Compute tr((Omega^2)^2) for a given Weyl tensor."""
    O = mk_omega(C)
    Osq = sum(O[a, b] @ O[a, b] for a in range(D) for b in range(D))
    S = Osq @ Osq
    return np.trace(S).real


OUTDIR = ANALYSIS_DIR / "figures" / "chirality"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# FIGURE 1: pq cross-term convergence (Russian)
# =====================================================================

def figure1_pq_convergence():
    ensemble_sizes = [10, 25, 50, 100, 200, 500]
    betas = []
    beta_errors = []

    master_rng = np.random.default_rng(314159)

    for N in ensemble_sizes:
        data_p2q2 = []
        data_pq = []
        data_tr = []

        for _ in range(N):
            Wp = rnd_ts3(master_rng)
            Wm = rnd_ts3(master_rng)
            C = mk_weyl(Wp, Wm)
            Cp, Cm = sd_decompose(C)
            p, q = compute_pq(Cp, Cm)
            tr_val = compute_tr_osq_sq(C)

            data_p2q2.append(p**2 + q**2)
            data_pq.append(p * q)
            data_tr.append(tr_val)

        # Fit: tr = alpha*(p^2+q^2) + beta*pq
        A_mat = np.column_stack([data_p2q2, data_pq])
        b_vec = np.array(data_tr)

        result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        coeffs = result[0]

        residuals = b_vec - A_mat @ coeffs
        dof = N - 2
        if dof > 0:
            sigma2 = np.sum(residuals**2) / dof
            cov = sigma2 * np.linalg.inv(A_mat.T @ A_mat)
            se = np.sqrt(np.diag(cov))
        else:
            se = np.array([0.0, 0.0])

        beta_val = coeffs[1]
        beta_se = se[1]

        betas.append(beta_val)
        beta_errors.append(beta_se)

        print(f"N={N:4d}: alpha={coeffs[0]:.8f}, beta={beta_val:.2e} +/- {beta_se:.2e}")

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.errorbar(
        ensemble_sizes, betas, yerr=beta_errors,
        fmt='o-', color='#2166ac', markersize=5, capsize=4, linewidth=1.2,
        label=r'Подгоночный $\beta$ (перекрёстный коэфф.)'
    )
    ax.axhline(y=0, color='#b2182b', linestyle='--', linewidth=1.5,
               label=r'$\beta = 0$ (предсказание киральности)')

    ax.set_xlabel(r'Число случайных тензоров Вейля $N$')
    ax.set_ylabel(r'Подгоночный коэффициент $\beta$ при $pq$')
    ax.set_title(r'Сходимость перекрёстного члена $pq$ к нулю')
    ax.legend(loc='upper right', frameon=True, edgecolor='gray')
    ax.set_xscale('log')

    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.1)

    fig.tight_layout()

    outpath = OUTDIR / "fig_pq_convergence_ru.pdf"
    fig.savefig(str(outpath), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close(fig)


# =====================================================================
# FIGURE 2: Crossed chirality scatter (Russian)
# =====================================================================

def figure2_crossed_chirality():
    rng = np.random.default_rng(271828)
    N = 200

    ps, qs = [], []
    tr_Ls, tr_Rs = [], []

    for _ in range(N):
        Wp = rnd_ts3(rng)
        Wm = rnd_ts3(rng)
        C = mk_weyl(Wp, Wm)
        Cp, Cm = sd_decompose(C)
        p, q = compute_pq(Cp, Cm)

        O = mk_omega(C)
        Osq = sum(O[a, b] @ O[a, b] for a in range(D) for b in range(D))

        tr_L = np.trace(P_L @ Osq).real
        tr_R = np.trace(P_R @ Osq).real

        ps.append(p)
        qs.append(q)
        tr_Ls.append(tr_L)
        tr_Rs.append(tr_R)

    ps = np.array(ps)
    qs = np.array(qs)
    tr_Ls = np.array(tr_Ls)
    tr_Rs = np.array(tr_Rs)

    # ---------- plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Left panel: tr_R(Omega^2) vs p
    ax1.scatter(ps, tr_Rs, s=12, alpha=0.6, color='#2166ac', edgecolors='none',
                label=r'$\mathrm{tr}_R(\Omega^2)$')

    slope_R = np.sum(ps * tr_Rs) / np.sum(ps**2)
    p_fit = np.linspace(0, np.max(ps) * 1.05, 100)
    ax1.plot(p_fit, slope_R * p_fit, 'r-', linewidth=1.5,
             label=rf'$-\frac{{1}}{{8}}p$ (наклон $= {slope_R:.4f}$)')

    ax1.set_xlabel(r'$p = |C^+|^2$')
    ax1.set_ylabel(r'$\mathrm{tr}_R(\Omega^2)$')
    ax1.set_title(r'Правое $\leftrightarrow$ самодуальное')
    ax1.legend(loc='lower left', frameon=True, edgecolor='gray')
    ax1.grid(True, alpha=0.3)

    # Right panel: tr_L(Omega^2) vs q
    ax2.scatter(qs, tr_Ls, s=12, alpha=0.6, color='#b2182b', edgecolors='none',
                label=r'$\mathrm{tr}_L(\Omega^2)$')

    slope_L = np.sum(qs * tr_Ls) / np.sum(qs**2)
    q_fit = np.linspace(0, np.max(qs) * 1.05, 100)
    ax2.plot(q_fit, slope_L * q_fit, 'b-', linewidth=1.5,
             label=rf'$-\frac{{1}}{{8}}q$ (наклон $= {slope_L:.4f}$)')

    ax2.set_xlabel(r'$q = |C^-|^2$')
    ax2.set_ylabel(r'$\mathrm{tr}_L(\Omega^2)$')
    ax2.set_title(r'Левое $\leftrightarrow$ антисамодуальное')
    ax2.legend(loc='lower left', frameon=True, edgecolor='gray')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Перекрёстное киральное соответствие', fontsize=12, y=1.01)
    fig.tight_layout()

    outpath = OUTDIR / "fig_crossed_chirality_ru.pdf"
    fig.savefig(str(outpath), dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    print(f"Slope R = {slope_R:.6f} (expected -0.5)")
    print(f"Slope L = {slope_L:.6f} (expected -0.5)")
    plt.close(fig)


# =====================================================================

if __name__ == "__main__":
    figure1_pq_convergence()
    figure2_crossed_chirality()
    print("\nAll Russian figures generated.")
