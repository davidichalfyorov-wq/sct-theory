"""
Publication figures for the No-Scalaron Theorem paper.

Generates 5 figures:
  Fig 1: Pi_s(z, xi) > 1 for several xi values
  Fig 2: alpha_R_min(x) decomposition: D(x) vs |S(x)|
  Fig 3: Robustness region in (N_f/N_s, N_v/N_s) plane
  Fig 4: Pi_TT(z) with the exact zero at z_0 = 2.4148
  Fig 5: Comparison of effective masses: exact vs local
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 12,
    "figure.figsize": (3.5, 2.8), "figure.dpi": 300,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

from sct_tools.form_factors import hR_scalar_mp, hR_dirac_mp, hR_vector_mp

# Setup
mp.mp.dps = 50
fig_dir = Path(__file__).resolve().parent.parent / "figures" / "no_scalaron"
fig_dir.mkdir(parents=True, exist_ok=True)

NS, ND, NV = 4, 22.5, 12


def phi_mp(x):
    if x == 0:
        return mp.mpf(1)
    x = mp.mpf(x)
    return mp.exp(-x / 4) * mp.sqrt(mp.pi / x) * mp.erfi(mp.sqrt(x) / 2)


def scalar_cz(x):
    x = mp.mpf(x)
    p = phi_mp(float(x))
    f_Ric = mp.mpf(1) / (6 * x) + (p - 1) / x**2
    f_R = p / 32 + p / (8 * x) - mp.mpf(7) / (48 * x) - (p - 1) / (8 * x**2)
    f_RU = -p / 4 - (p - 1) / (2 * x)
    f_U = p / 2
    f_Rbis = f_Ric / 3 + f_R
    return f_Rbis, f_RU, f_U


# ======================================================================
# Fig 1: Pi_s(z, xi) for several xi
# ======================================================================
print("Generating Fig 1: Pi_s(z, xi)...")
z_arr = np.linspace(0.01, 20, 500)
xi_values = [0, 0.1, 1 / 6, 0.3, 0.5, 1.0]
xi_labels = ["$\\xi = 0$", "$\\xi = 0.1$", "$\\xi = 1/6$",
             "$\\xi = 0.3$", "$\\xi = 0.5$", "$\\xi = 1.0$"]

fig, ax = plt.subplots(figsize=(4.5, 3.2))
for xi_val, label in zip(xi_values, xi_labels):
    Pi_s = []
    for z in z_arr:
        aR = (NS * float(hR_scalar_mp(z, xi=xi_val, dps=50))
              + ND * float(hR_dirac_mp(z, dps=50))
              + NV * float(hR_vector_mp(z, dps=50)))
        Pi_s.append(1 + 3 * z * aR)
    ax.plot(z_arr, Pi_s, label=label)

ax.axhline(y=1, color="gray", ls="--", lw=0.5)
ax.set_xlabel(r"$z = k^2/\Lambda^2$")
ax.set_ylabel(r"$\Pi_s(z, \xi)$")
ax.set_title("Scalar propagator denominator (SM spectrum)")
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(0.9, None)
ax.set_xlim(0, 20)
fig.tight_layout()
fig.savefig(fig_dir / "fig1_pi_s_vs_z.pdf")
fig.savefig(fig_dir / "fig1_pi_s_vs_z.png", dpi=200)
plt.close(fig)
print(f"  Saved: {fig_dir / 'fig1_pi_s_vs_z.pdf'}")

# ======================================================================
# Fig 2: D(x) and |S(x)| decomposition
# ======================================================================
print("Generating Fig 2: D vs |S| decomposition...")
x_arr = np.logspace(-1.5, 2.5, 600)
D_vals, S_vals, aR_min_vals = [], [], []

for x_val in x_arr:
    f_Rbis, f_RU, f_U = scalar_cz(x_val)
    hR12 = hR_dirac_mp(x_val, dps=50)
    hR1 = hR_vector_mp(x_val, dps=50)
    D = float(ND * hR12 + NV * hR1)
    S = float(4 * f_Rbis - f_RU**2 / f_U)
    D_vals.append(D)
    S_vals.append(S)
    aR_min_vals.append(D + S)

D_vals = np.array(D_vals)
S_vals = np.array(S_vals)
aR_min_vals = np.array(aR_min_vals)

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.loglog(x_arr, D_vals, label=r"$D(x)$ (Dirac + vector)", color="C0")
ax.loglog(x_arr, np.abs(S_vals), label=r"$|S(x)|$ (scalar)", color="C3", ls="--")
ax.loglog(x_arr, aR_min_vals, label=r"$\alpha_{R,\min}(x) = D + S$", color="C2", lw=2)
ax.axhline(y=0, color="gray", ls=":", lw=0.5)
ax.set_xlabel(r"$x$")
ax.set_ylabel("Form factor combination")
ax.set_title(r"Proof: $D(x) \gg |S(x)|$ $\Rightarrow$ $\alpha_{R,\min} > 0$")
ax.legend(fontsize=7)
ax.set_xlim(x_arr[0], x_arr[-1])
fig.tight_layout()
fig.savefig(fig_dir / "fig2_D_vs_S.pdf")
fig.savefig(fig_dir / "fig2_D_vs_S.png", dpi=200)
plt.close(fig)
print(f"  Saved: {fig_dir / 'fig2_D_vs_S.pdf'}")

# ======================================================================
# Fig 3: Robustness region
# ======================================================================
print("Generating Fig 3: Robustness region...")

# For each (N_f/N_s, N_v/N_s) point, check if no-scalaron holds
# Using the precomputed S_1(x) per scalar and D per (N_f/2, N_v)
x_scan = np.linspace(0.02, 200, 2000)
s1_arr = []
hR12_arr = []
hR1_arr = []

for x_val in x_scan:
    f_Rbis, f_RU, f_U = scalar_cz(x_val)
    s1_arr.append(float(f_Rbis - f_RU**2 / (4 * f_U)))
    hR12_arr.append(float(hR_dirac_mp(x_val, dps=50)))
    hR1_arr.append(float(hR_vector_mp(x_val, dps=50)))

s1_arr = np.array(s1_arr)
hR12_arr = np.array(hR12_arr)
hR1_arr = np.array(hR1_arr)

# No-scalaron iff: (N_f/2)*hR^(1/2) + N_v*hR^(1) > N_s*|S_1| for all x
# => (nf/ns/2)*hR^(1/2) + (nv/ns)*hR^(1) > |S_1| for all x (with N_s=1)
nf_range = np.linspace(0, 15, 150)
nv_range = np.linspace(0, 5, 100)

safe_map = np.zeros((len(nv_range), len(nf_range)))
for i, nv_r in enumerate(nv_range):
    for j, nf_r in enumerate(nf_range):
        D_test = (nf_r / 2) * hR12_arr + nv_r * hR1_arr
        aR_min_test = s1_arr + D_test  # s1 is per-scalar, negative
        safe_map[i, j] = 1 if np.all(aR_min_test > 0) else 0

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.contourf(nf_range, nv_range, safe_map, levels=[0.5, 1.5],
            colors=["#d4edda"], alpha=0.8)
ax.contour(nf_range, nv_range, safe_map, levels=[0.5],
           colors=["darkgreen"], linewidths=1.5)
ax.plot(45 / 4, 12 / 4, "r*", markersize=12, label="SM", zorder=5)
ax.plot(48 / 4, 12 / 4, "bs", markersize=6, label="SM + 3$\\nu_R$", zorder=5)
ax.set_xlabel(r"$N_f / N_s$")
ax.set_ylabel(r"$N_v / N_s$")
ax.set_title("No-scalaron safe region (green)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(fig_dir / "fig3_robustness.pdf")
fig.savefig(fig_dir / "fig3_robustness.png", dpi=200)
plt.close(fig)
print(f"  Saved: {fig_dir / 'fig3_robustness.pdf'}")

# ======================================================================
# Fig 4: Pi_TT(z) with exact zero
# ======================================================================
print("Generating Fig 4: Pi_TT(z)...")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nt4a_propagator import Pi_TT

z_tt = np.linspace(0.01, 8, 500)
pi_tt = [float(Pi_TT(z, xi=0, dps=50).real) for z in z_tt]

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(z_tt, pi_tt, "C0", lw=1.5)
ax.axhline(y=0, color="gray", ls="--", lw=0.5)
ax.axvline(x=2.4148, color="C3", ls=":", lw=1, label=r"$z_0 = 2.4148$ (spin-2 fakeon)")

# Local approximation zero
z_local = 60 / 13
ax.axvline(x=z_local, color="C1", ls=":", lw=1, label=r"$z_{\rm loc} = 60/13 \approx 4.615$")

ax.set_xlabel(r"$z = k^2/\Lambda^2$")
ax.set_ylabel(r"$\Pi_{\rm TT}(z)$")
ax.set_title("Spin-2 propagator denominator")
ax.legend(fontsize=7)
ax.set_ylim(-5, 2)
fig.tight_layout()
fig.savefig(fig_dir / "fig4_pi_tt.pdf")
fig.savefig(fig_dir / "fig4_pi_tt.png", dpi=200)
plt.close(fig)
print(f"  Saved: {fig_dir / 'fig4_pi_tt.pdf'}")

# ======================================================================
# Fig 5: Per-spin hR contributions
# ======================================================================
print("Generating Fig 5: Per-spin hR contributions...")
x_plot = np.logspace(-1, 3, 400)
hr0_vals = [float(hR_scalar_mp(x, xi=0, dps=50)) for x in x_plot]
hr12_vals = [float(hR_dirac_mp(x, dps=50)) for x in x_plot]
hr1_vals = [float(hR_vector_mp(x, dps=50)) for x in x_plot]

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.semilogx(x_plot, [NS * h for h in hr0_vals], label=r"$N_s \cdot h_R^{(0)}$ (scalar)", ls="--")
ax.semilogx(x_plot, [ND * h for h in hr12_vals], label=r"$N_D \cdot h_R^{(1/2)}$ (Dirac)")
ax.semilogx(x_plot, [NV * h for h in hr1_vals], label=r"$N_v \cdot h_R^{(1)}$ (vector)")
ax.semilogx(x_plot,
            [NS * h0 + ND * h12 + NV * h1 for h0, h12, h1 in zip(hr0_vals, hr12_vals, hr1_vals)],
            label=r"$\alpha_R^{\rm total}(x, \xi=0)$", color="k", lw=2)
ax.axhline(y=0, color="gray", ls=":", lw=0.5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Contribution to $\alpha_R$")
ax.set_title("Per-spin contributions to $R^2$ form factor")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(fig_dir / "fig5_per_spin_hR.pdf")
fig.savefig(fig_dir / "fig5_per_spin_hR.png", dpi=200)
plt.close(fig)
print(f"  Saved: {fig_dir / 'fig5_per_spin_hR.pdf'}")

print(f"\nAll 5 figures saved to {fig_dir}")
