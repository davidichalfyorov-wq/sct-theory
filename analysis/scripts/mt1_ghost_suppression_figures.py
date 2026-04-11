#!/usr/bin/env python3
"""Generate 8 publication figures for the MT-1 Ghost Suppression Theorem."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "figure.dpi": 150,
    "text.usetex": False, "font.family": "serif",
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "mt1")
os.makedirs(FIGDIR, exist_ok=True)

DATA = os.path.join(os.path.dirname(__file__), "..", "fnd1_data", "mt1_ghost_suppression.json")
with open(DATA) as f:
    d = json.load(f)

grid = d["grid_results"]
kerr = d["kerr_temperature"]
rn = d["rn_temperature"]
kn = d["kn_temperature_2d"]
obs = d["observed_bhs"]

M_arr = np.array([r["M_Msun"] for r in grid])
bz_arr = np.array([r["boltzmann_exp"] for r in grid])
sw_arr = np.array([r["schwinger_exp"] for r in grid])
yk_arr = np.array([r["yukawa_exp"] for r in grid])
vb_arr = np.array([r["violation_log10"] for r in grid])

Mcrit = d["M_crit_Msun"]

# ---- Fig 1: Triple suppression ----
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(M_arr, bz_arr, "o-", ms=3, label="I: Boltzmann  m/T", color="C0")
ax.semilogy(M_arr, sw_arr, "s-", ms=3, label="II: Schwinger  $4\\pi m^2 GM$", color="C1")
ax.semilogy(M_arr, yk_arr, "^-", ms=3, label="III: Yukawa  $m \\cdot r_H$", color="C2")
ax.axhline(1, color="red", ls=":", lw=0.8, alpha=0.5)
ax.axvline(Mcrit, color="gray", ls="--", lw=0.8, alpha=0.7)
ax.text(Mcrit * 2, 1e2, "$M_{\\rm crit}$", fontsize=8, color="gray")
ax.set_xscale("log")
ax.set_xlabel("$M / M_\\odot$")
ax.set_ylabel("Suppression exponent")
ax.set_title("Three independent suppression mechanisms")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(1e-2, 1e20)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_triple_suppression.pdf"))
print("Saved mt1_triple_suppression.pdf")
plt.close()

# ---- Fig 2: Ghost ratio m/T_H ----
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(M_arr, bz_arr, "o-", ms=4, color="C0")
ax.axhline(1, color="red", ls=":", lw=1, label="m = T (ghost becomes active)")
ax.axvline(Mcrit, color="gray", ls="--", lw=0.8)
ax.axvline(3, color="green", ls="-.", lw=0.8, alpha=0.7, label="Lightest observed BH (3 $M_\\odot$)")
ax.fill_between([1e-8, Mcrit], [1e-2, 1e-2], [1e20, 1e20], alpha=0.1, color="red", label="Ghost-active")
ax.fill_between([Mcrit, 1e10], [1e-2, 1e-2], [1e20, 1e20], alpha=0.1, color="green", label="Ghost-safe")
ax.set_xlabel("$M / M_\\odot$")
ax.set_ylabel("$m_{\\rm ghost} / T_H$")
ax.set_title("Ghost thermal ratio")
ax.legend(fontsize=7)
ax.set_ylim(1e-1, 1e20)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_ghost_ratio.pdf"))
print("Saved mt1_ghost_ratio.pdf")
plt.close()

# ---- Fig 3: Exclusion plot (M, Lambda) ----
Lambda_arr = np.logspace(-4, 20, 100)
M_crit_arr = []
from mt1_ghost_suppression import M_crit_Msun, M_GHOST_OVER_LAMBDA
import mpmath as mp
for lam in Lambda_arr:
    mc = float(M_crit_Msun(mp.mpf(str(lam))))
    M_crit_arr.append(mc)
M_crit_arr = np.array(M_crit_arr)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(Lambda_arr, M_crit_arr, "k-", lw=2, label="$M_{\\rm crit}(\\Lambda)$")
ax.fill_between(Lambda_arr, M_crit_arr, 1e15, alpha=0.15, color="green", label="Ghost-safe")
ax.fill_between(Lambda_arr, 1e-20, M_crit_arr, alpha=0.15, color="red", label="Ghost-active")
ax.axhline(3, color="blue", ls=":", lw=0.8, label="Lightest BH (3 $M_\\odot$)")
ax.axvline(2.38e-3, color="orange", ls="--", lw=0.8, label="$\\Lambda_{\\rm PPN}$")
# Mark observed BHs
for o in obs:
    ax.plot(2.38e-3, o["M_Msun"], "D", ms=5, color="navy", zorder=5)
    ax.annotate(o["name"], (2.38e-3 * 1.5, o["M_Msun"]), fontsize=5, color="navy")
ax.set_xlabel("$\\Lambda$ [eV]")
ax.set_ylabel("$M_{\\rm crit} / M_\\odot$")
ax.set_title("Ghost suppression: exclusion plot")
ax.legend(fontsize=6, loc="upper right")
ax.set_xlim(1e-4, 1e20)
ax.set_ylim(1e-35, 1e15)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_exclusion_plot.pdf"))
print("Saved mt1_exclusion_plot.pdf")
plt.close()

# ---- Fig 4: Critical mass vs Lambda ----
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(Lambda_arr, M_crit_arr, "C0-", lw=2)
ax.axhline(3, color="red", ls=":", lw=1, label="3 $M_\\odot$ (lightest BH)")
ax.axhline(1e10, color="blue", ls=":", lw=0.8, label="$10^{10} M_\\odot$ (TON 618)")
ax.axvline(2.38e-3, color="orange", ls="--", lw=1, label="$\\Lambda_{\\rm PPN} = 2.38$ meV")
ax.axvline(1.22e28, color="purple", ls="--", lw=0.8, label="$\\Lambda = M_{\\rm Pl}$")
ax.set_xlabel("$\\Lambda$ [eV]")
ax.set_ylabel("$M_{\\rm crit} / M_\\odot$")
ax.set_title("Critical mass vs cutoff scale")
ax.legend(fontsize=7)
ax.set_xlim(1e-4, 1e30)
ax.set_ylim(1e-40, 1e5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_critical_mass.pdf"))
print("Saved mt1_critical_mass.pdf")
plt.close()

# ---- Fig 5: Violation bound ----
fig, ax = plt.subplots(figsize=(5, 3.5))
sel = M_arr > 1e-6
ax.plot(M_arr[sel], vb_arr[sel], "o-", ms=4, color="C3")
ax.set_xscale("log")
ax.set_xlabel("$M / M_\\odot$")
ax.set_ylabel("$\\log_{10}(|dS_{\\rm ghost}/dt| / |dS_{\\rm matter}/dt|)$")
ax.set_title("Second law violation bound (safety exponent)")
ax.axvline(3, color="green", ls="-.", lw=0.8, label="Lightest observed BH")
for o in obs:
    if o["M_Msun"] > 1e-2:
        ax.annotate(o["name"], (o["M_Msun"], o["violation_log10"]),
                    fontsize=6, rotation=45, color="navy")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_violation_bound.pdf"))
print("Saved mt1_violation_bound.pdf")
plt.close()

# ---- Fig 6: Ghost partition function (placeholder — all zeros at astrophysical scale) ----
fig, ax = plt.subplots(figsize=(5, 3.5))
# Plot log10(m/T) which is the relevant quantity
ax.semilogy(M_arr, bz_arr, "o-", ms=4, color="C0", label="$m/T_H$ (determines all thermodynamics)")
ax.axhline(1, color="red", ls=":", lw=1)
ax.axhline(30, color="orange", ls=":", lw=0.8, label="$m/T = 30$ (machine underflow)")
ax.set_xscale("log")
ax.set_xlabel("$M / M_\\odot$")
ax.set_ylabel("$m_{\\rm ghost} / T_H$")
ax.set_title("Ghost thermodynamics: all quantities $\\propto e^{-m/T}$")
ax.text(1e4, 50, "$n_{\\rm ghost}, \\rho_{\\rm ghost}, S_{\\rm ghost}$\nall identically zero\nat float64 precision",
        fontsize=7, ha="center", bbox=dict(boxstyle="round", fc="lightyellow"))
ax.legend(fontsize=7)
ax.set_ylim(0.1, 1e20)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_partition_function.pdf"))
print("Saved mt1_partition_function.pdf")
plt.close()

# ---- Fig 7: Kerr temperature ratio ----
a_arr = np.array([k["a_over_M"] for k in kerr])
T_ratio_arr = np.array([k["T_ratio"] for k in kerr])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(a_arr, T_ratio_arr, "o-", ms=4, color="C0", label="Kerr")
# RN curve
Q_arr = np.array([r["Q_over_M"] for r in rn])
T_rn_arr = np.array([r["T_ratio"] for r in rn])
ax.plot(Q_arr, T_rn_arr, "s-", ms=4, color="C1", label="RN")
ax.axhline(1, color="red", ls=":", lw=1, label="Schwarzschild (a=Q=0)")
ax.set_xlabel("$a/M$ (Kerr)  or  $Q/M$ (RN)")
ax.set_ylabel("$T_H / T_H^{\\rm Sch}$")
ax.set_title("Schwarzschild is the hottest BH")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_kerr_temperature.pdf"))
print("Saved mt1_kerr_temperature.pdf")
plt.close()

# ---- Fig 8: KN 2D heatmap ----
a_kn = np.array([k["a_over_M"] for k in kn])
Q_kn = np.array([k["Q_over_M"] for k in kn])
T_kn = np.array([k["T_ratio"] for k in kn])

fig, ax = plt.subplots(figsize=(5, 4))
sc = ax.scatter(a_kn, Q_kn, c=T_kn, cmap="RdYlGn_r", vmin=0, vmax=1, s=60, edgecolors="gray", lw=0.3)
plt.colorbar(sc, ax=ax, label="$T_H / T_H^{\\rm Sch}$")
# Extremal boundary
theta = np.linspace(0, np.pi/2, 100)
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1, label="Extremal: $a^2+Q^2=M^2$")
ax.set_xlabel("$a / M$")
ax.set_ylabel("$Q / M$")
ax.set_title("Kerr-Newman: $T_H \\leq T_H^{\\rm Sch}$ everywhere")
ax.legend(fontsize=7)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mt1_kerr_rn_2d.pdf"))
print("Saved mt1_kerr_rn_2d.pdf")
plt.close()

print("\nAll 8 figures generated.")
