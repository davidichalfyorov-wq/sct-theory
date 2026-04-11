# ruff: noqa: E402, I001
"""
Generate BLACK-AND-WHITE QNM figures for TMF submission.
TMF requires: black and white, shades of gray allowed, vector format.

Author: David Alfyorov
"""

import numpy as np
from scipy import constants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

G_N = constants.G; c_light = constants.c; hbar = constants.hbar
M_sun = 1.989e30; eV_to_J = constants.eV
LAMBDA_EV = 2.38e-3
M2_OVER_LAMBDA = np.sqrt(60.0 / 13.0)
M0_OVER_LAMBDA = np.sqrt(6.0)
M2_INV_M = M2_OVER_LAMBDA * LAMBDA_EV * eV_to_J / (hbar * c_light)
M0_INV_M = M0_OVER_LAMBDA * LAMBDA_EV * eV_to_J / (hbar * c_light)
M_CRIT_SOLAR = hbar * c_light**3 / (8 * np.pi * G_N * np.sqrt(1.2807) * LAMBDA_EV * eV_to_J) / M_sun

FIGDIR = Path(__file__).parent.parent.parent / "papers" / "drafts" / "figures"

def r_s_from_M_solar(M_solar):
    return 2 * G_N * M_solar * M_sun / c_light**2

def h_yukawa(r, m2, m0):
    return 1.0 - (4.0/3.0)*np.exp(-m2*r) + (1.0/3.0)*np.exp(-m0*r)

def f_SCT(r, r_s, m2, m0):
    return 1.0 - (r_s/r)*h_yukawa(r, m2, m0)

def V_RW_GR(r, r_s, l=2):
    f = 1.0 - r_s/r
    M = r_s/2
    return f*(l*(l+1)/r**2 - 6*M/r**3)

def V_RW_SCT(r, r_s, m2, m0, l=2):
    f = f_SCT(r, r_s, m2, m0)
    one_minus_f = (r_s/r)*h_yukawa(r, m2, m0)
    return f*(l*(l+1)/r**2 - 3*one_minus_f/r**2)

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

m2 = M2_INV_M; m0 = M0_INV_M

# ============================================================
# Figure 1: Effective potentials (B&W, Russian)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

masses_solar = [M_CRIT_SOLAR * 10, 1.0, 10.0, 1e6]
titles = [
    f'$M = 10\\,M_{{\\rm crit}}$ ({M_CRIT_SOLAR*10:.1e} $M_\\odot$)',
    '$M = 1\\,M_\\odot$',
    '$M = 10\\,M_\\odot$',
    '$M = 10^6\\,M_\\odot$',
]

for ax, M_sol, title in zip(axes.flat, masses_solar, titles):
    r_s = r_s_from_M_solar(M_sol)
    r_arr = np.linspace(r_s*1.01, r_s*6, 500)
    V_gr = [V_RW_GR(r, r_s, 2) for r in r_arr]
    V_sct = [V_RW_SCT(r, r_s, m2, m0, 2) for r in r_arr]

    ax.plot(r_arr/r_s, np.array(V_gr)*r_s**2, 'k-', lw=1.5)
    ax.plot(r_arr/r_s, np.array(V_sct)*r_s**2, 'k--', lw=1.5)
    ax.set_xlabel('$r / r_s$')
    ax.set_ylabel('$V(r) \\cdot r_s^2$')
    ax.set_title(title, fontsize=9)
    ax.legend(['GR', 'SCT'], fontsize=8)
    ax.grid(True, alpha=0.3, color='gray')

fig.tight_layout()
path = FIGDIR / "fig_qnm_potentials_bw.pdf"
fig.savefig(path, dpi=200)
print(f"Saved {path}")
plt.close()

# ============================================================
# Figure 2: Mass scan (B&W, Russian)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

N_pts = 150
M_arr = np.logspace(-10, 11, N_pts)
shifts = []
for M_sol in M_arr:
    r_s = r_s_from_M_solar(M_sol)
    r_peak = r_s * 1.64
    m2_rp = m2 * r_peak
    log10_metric = -m2_rp / np.log(10)
    omega_M = 0.3737
    omega_phys = omega_M * c_light**3 / (G_N * M_sol * M_sun)
    omega_eV = hbar * omega_phys / eV_to_J
    perturb = (13.0/60.0) * (omega_eV / LAMBDA_EV)**2
    log10_perturb = np.log10(perturb) if perturb > 0 else -300
    shifts.append(max(log10_metric, log10_perturb))

ax.plot(M_arr, shifts, 'k-', lw=2)
ax.axvline(M_CRIT_SOLAR, color='gray', ls='--', lw=1)
ax.text(M_CRIT_SOLAR*3, -5, '$M_{\\rm crit}$', fontsize=9, color='gray')

# LIGO band - gray fill
ax.axhspan(-1, 0, alpha=0.15, color='gray')
ax.text(30, -0.5, 'LIGO O4', fontsize=8, color='gray', ha='center')

# Astrophysical region
ax.axvspan(3, 1e11, alpha=0.05, color='gray')
ax.text(1e5, -15, 'Астрофизические ЧД', fontsize=8, color='black', ha='center')

ax.set_xscale('log')
ax.set_xlabel('$M / M_\\odot$')
ax.set_ylabel('$\\log_{10}(\\delta\\omega / \\omega)$')
ax.set_title('Сдвиг КНМ-частоты в СКТ ($l=2$, $n=0$)')
ax.set_xlim(1e-10, 1e11)
finite = [s for s in shifts if np.isfinite(s)]
ax.set_ylim(min(finite)*1.1, 2)
ax.grid(True, alpha=0.3, color='gray')

fig.tight_layout()
path = FIGDIR / "fig_qnm_mass_scan_bw.pdf"
fig.savefig(path, dpi=200)
print(f"Saved {path}")
plt.close()

print("Done. Both figures in black-and-white vector PDF.")
