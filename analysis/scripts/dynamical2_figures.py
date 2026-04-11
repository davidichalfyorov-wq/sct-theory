#!/usr/bin/env python3
"""
DYN-2: Generate publication figures for the dynamical normalization analysis.

Figures:
  1. M_ss_norm vs N with M=2 baseline
  2. rho_eff vs N with convergence fit
  3. Spatial rho(zeta) profile at N=10000 (20 bins)
  4. M_1leg_norm and rho_eff evolution showing L*(1+R)=2 constraint
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "text.usetex": False,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "dyn2")
os.makedirs(FIGDIR, exist_ok=True)

# Load Phase 1 data
p1_path = os.path.join(os.path.dirname(__file__), "..", "fnd1_data", "dynamical2_highN.json")
with open(p1_path) as f:
    p1 = json.load(f)

# Load Phase 2 data
p2_path = os.path.join(os.path.dirname(__file__), "..", "fnd1_data", "dynamical2_rho_diagnostic.json")
with open(p2_path) as f:
    p2 = json.load(f)

# ---- Extract Phase 1 data ----
p1_results = p1["results"]
N_vals_p1 = sorted([int(k) for k in p1_results.keys()])
# Skip N=1000 (too noisy)
N_vals_p1 = [n for n in N_vals_p1 if n >= 3000]

Mss_means = [p1_results[str(n)]["M_ss_norm_mean"] for n in N_vals_p1]
Mss_se = [p1_results[str(n)]["M_ss_norm_se"] for n in N_vals_p1]
rho_means = [p1_results[str(n)]["rho_eff_mean"] for n in N_vals_p1]
rho_se = [p1_results[str(n)]["rho_eff_se"] for n in N_vals_p1]
corr_means = [p1_results[str(n)]["corr_gm_gp_pointwise_mean"] for n in N_vals_p1]

# ---- Figure 1: M_ss_norm vs N ----
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.errorbar(N_vals_p1, Mss_means, yerr=Mss_se, fmt="o-", color="C0",
            markersize=5, capsize=3, label=r"$M_{\rm ss}^{\rm norm}$")
ax.axhline(2.0, color="red", linestyle="--", linewidth=1, label=r"Target = 2")
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$M_{\rm ss} / (C_{\rm AN}\, N^{8/9}\, \mathcal{E}^2\, T^4)$")
ax.set_title(r"Dynamical normalization: $M_{\rm ss}^{\rm norm} \to 2$")
ax.legend(loc="lower right")
ax.set_ylim(1.5, 2.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "dyn2_Mss_convergence.pdf"))
print("Saved dyn2_Mss_convergence.pdf")
plt.close()

# ---- Figure 2: rho_eff and corr(g-,g+) vs N ----
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.errorbar(N_vals_p1, rho_means, yerr=rho_se, fmt="s-", color="C1",
            markersize=5, capsize=3, label=r"$\rho_{\rm eff} = M_{-+}/\sqrt{M_{--}M_{++}}$")
ax.plot(N_vals_p1, corr_means, "^--", color="C2",
        markersize=5, label=r"$\mathrm{corr}(g_-, g_+)$ (pointwise)")
ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"Correlation")
ax.set_title(r"Past-future correlation growth")
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "dyn2_rho_convergence.pdf"))
print("Saved dyn2_rho_convergence.pdf")
plt.close()

# ---- Figure 3: Spatial rho profile at N=10000 ----
p2_results = p2["results"]
n_bins = p2["n_zeta_bins"]  # 20

if "10000" in p2_results:
    rho_profile = p2_results["10000"]["rho_profile_mean"]
    asym_profile = p2_results["10000"]["asymmetry_profile_mean"]

    zeta_centers = [(i + 0.5) / n_bins for i in range(n_bins)]
    valid = [(z, r, a) for z, r, a in zip(zeta_centers, rho_profile, asym_profile)
             if r is not None and a is not None]
    if valid:
        z_v, r_v, a_v = zip(*valid)

        fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
        ax1.plot(z_v, r_v, "o-", color="C0", markersize=4,
                 label=r"$\rho(g_-, g_+)$")
        ax1.set_xlabel(r"Normalized position $\zeta$")
        ax1.set_ylabel(r"Pearson $\rho(g_-, g_+)$", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.set_ylim(0, 0.8)
        ax1.axvline(0.5, color="gray", linestyle=":", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(z_v, a_v, "s--", color="C3", markersize=4,
                 label=r"Asymmetry $\langle|g_- - g_+|\rangle / \langle|g_-|+|g_+|\rangle$")
        ax2.set_ylabel("Asymmetry", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax2.set_ylim(0.3, 1.0)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        ax1.set_title(r"Spatial profile: $N = 10{,}000$, 40 trials")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "dyn2_rho_profile.pdf"))
        print("Saved dyn2_rho_profile.pdf")
        plt.close()

# ---- Figure 4: L and R evolution ----
# Use Phase 2 data (has M_1leg_norm directly)
N_vals_p2 = sorted([int(k) for k in p2_results.keys()])
L_vals = [p2_results[str(n)]["M_1leg_norm_mean"] for n in N_vals_p2]
L_se = [p2_results[str(n)]["M_1leg_norm_se"] for n in N_vals_p2]
R_vals = [p2_results[str(n)]["rho_eff_mean"] for n in N_vals_p2]
R_se = [p2_results[str(n)]["rho_eff_se"] for n in N_vals_p2]
prod_vals = [p2_results[str(n)]["constraint_product_mean"] for n in N_vals_p2]
prod_se = [p2_results[str(n)]["constraint_product_se"] for n in N_vals_p2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

ax1.errorbar(N_vals_p2, L_vals, yerr=L_se, fmt="o-", color="C0",
             markersize=5, capsize=3, label=r"$L = M_{\rm 1leg}^{\rm norm}$")
ax1.errorbar(N_vals_p2, [1+r for r in R_vals],
             yerr=R_se, fmt="s-", color="C1",
             markersize=5, capsize=3, label=r"$1 + \rho_{\rm eff}$")
ax1.set_xlabel(r"$N$")
ax1.set_ylabel("Value")
ax1.set_title(r"Individual factors: $L$ and $1+R$")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.errorbar(N_vals_p2, prod_vals, yerr=prod_se, fmt="D-", color="C4",
             markersize=5, capsize=3, label=r"$L \cdot (1+R)$")
ax2.axhline(2.0, color="red", linestyle="--", linewidth=1, label=r"Target = 2")
ax2.set_xlabel(r"$N$")
ax2.set_ylabel(r"$L \cdot (1+R)$")
ax2.set_title(r"Constraint: $M_{\rm 1leg}^{\rm norm} \cdot (1 + \rho_{\rm eff}) = 2$")
ax2.legend(fontsize=8)
ax2.set_ylim(1.5, 2.5)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "dyn2_perbin_decomposition.pdf"))
print("Saved dyn2_perbin_decomposition.pdf")
plt.close()

print("\nAll figures generated.")

# ---- Summary table ----
print("\n" + "=" * 80)
print("SUMMARY TABLE FOR LATEX")
print("=" * 80)
print(f"{'N':>8} {'M_ss_norm':>12} {'rho_eff':>10} {'corr(g-,g+)':>14} {'M_aa_norm':>12}")
for n in N_vals_p1:
    d = p1_results[str(n)]
    print(f"{n:8d} {d['M_ss_norm_mean']:12.4f} {d['rho_eff_mean']:10.4f} "
          f"{d['corr_gm_gp_pointwise_mean']:14.4f} {d['M_aa_norm_mean']:12.4f}")
