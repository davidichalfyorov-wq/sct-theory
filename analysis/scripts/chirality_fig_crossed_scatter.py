# ruff: noqa: E402, I001
"""
Figure: Crossed chirality scatter plot.

Left panel:  tr_R(Omega^2) vs p -- should be perfectly linear, no q dependence
Right panel: tr_L(Omega^2) vs q -- should be perfectly linear, no p dependence

Scatter of 200 random Weyl tensors showing the crossed assignment.

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

# sigma = (1/4)[gamma^r, gamma^s] (standard, no i)
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


# ---------- main computation ----------

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

# Fit line: tr_R = slope * p
slope_R = np.sum(ps * tr_Rs) / np.sum(ps**2)
p_fit = np.linspace(0, np.max(ps) * 1.05, 100)
ax1.plot(p_fit, slope_R * p_fit, 'r-', linewidth=1.5,
         label=rf'$-\frac{{1}}{{8}}p$ (slope $= {slope_R:.4f}$)')

ax1.set_xlabel(r'$p = |C^+|^2$')
ax1.set_ylabel(r'$\mathrm{tr}_R(\Omega^2)$')
ax1.set_title(r'Right-handed $\leftrightarrow$ self-dual')
ax1.legend(loc='lower left', frameon=True, edgecolor='gray')
ax1.grid(True, alpha=0.3)

# Right panel: tr_L(Omega^2) vs q
ax2.scatter(qs, tr_Ls, s=12, alpha=0.6, color='#b2182b', edgecolors='none',
            label=r'$\mathrm{tr}_L(\Omega^2)$')

# Fit line: tr_L = slope * q
slope_L = np.sum(qs * tr_Ls) / np.sum(qs**2)
q_fit = np.linspace(0, np.max(qs) * 1.05, 100)
ax2.plot(q_fit, slope_L * q_fit, 'b-', linewidth=1.5,
         label=rf'$-\frac{{1}}{{8}}q$ (slope $= {slope_L:.4f}$)')

ax2.set_xlabel(r'$q = |C^-|^2$')
ax2.set_ylabel(r'$\mathrm{tr}_L(\Omega^2)$')
ax2.set_title(r'Left-handed $\leftrightarrow$ anti-self-dual')
ax2.legend(loc='lower left', frameon=True, edgecolor='gray')
ax2.grid(True, alpha=0.3)

fig.suptitle('Crossed chirality assignment', fontsize=12, y=1.01)
fig.tight_layout()

outpath = Path(__file__).resolve().parent.parent / "figures" / "chirality" / "fig_crossed_chirality.pdf"
fig.savefig(str(outpath), dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
print(f"Slope R = {slope_R:.6f} (expected -0.5)")
print(f"Slope L = {slope_L:.6f} (expected -0.5)")
plt.close(fig)
