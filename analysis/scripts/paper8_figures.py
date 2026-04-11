"""Paper 8: Generate publication figures for prediction bands and comparison."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ANALYSIS_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "papers" / "drafts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import mpmath as mp
mp.mp.dps = 30

# Try SciencePlots, fall back to clean defaults
try:
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })

# --- Colors ---
COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}
LABELS = {1: r"$e^{-u}$", 2: r"$e^{-u^2}$", 3: r"$e^{-u^3}$",
          4: r"$e^{-u^4}$", 5: r"$e^{-u^5}$"}


# === Canonical form factor imports ===
from scripts.nt2_entire_function import F1_total_complex

ALPHA_C = mp.mpf(13) / 120
C2 = 2 * ALPHA_C  # 13/60
F1_0_CANON = F1_total_complex(0, dps=30)


def phi_n(x, n):
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(1)
    return mp.quad(lambda a: mp.exp(-(a * (1 - a) * x) ** n), [0, 1])


def phi_canonical(x):
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(1)
    return mp.exp(-x / 4) * mp.sqrt(mp.pi / x) * mp.erfi(mp.sqrt(x) / 2)


def _hC_s(x, p):
    x = mp.mpf(x)
    if abs(x) < 1e-10:
        return mp.mpf(1) / 120
    return 1 / (12 * x) + (p - 1) / (2 * x ** 2)


def _hC_d(x, p):
    x = mp.mpf(x)
    if abs(x) < 1e-10:
        return mp.mpf(-1) / 20
    return (3 * p - 1) / (6 * x) + 2 * (p - 1) / x ** 2


def _hC_v(x, p):
    x = mp.mpf(x)
    if abs(x) < 1e-10:
        return mp.mpf(1) / 10
    return p / 4 + (6 * p - 5) / (6 * x) + (p - 1) / x ** 2


def F1_n(z, n):
    p = phi_n(z, n) if n != 1 else phi_canonical(z)
    return (4 * _hC_s(z, p) + mp.mpf(22.5) * _hC_d(z, p) + 12 * _hC_v(z, p)) / (16 * mp.pi ** 2)


def PiTT_n(z, n):
    z = mp.mpf(z)
    if abs(z) < 1e-12:
        return mp.mpf(1)
    F1_0 = F1_n(0, n)
    return 1 + C2 * z * F1_n(z, n) / F1_0


# Verified zeros from our computation
Z0 = {1: 2.4148, 2: 5.1735, 3: 5.2479, 4: 5.2287, 5: 5.2084}
M2 = {n: float(mp.sqrt(z)) for n, z in Z0.items()}


# ==============================================================
# FIGURE 1: Pi_TT(z) for different cutoff functions
# ==============================================================
print("Generating Figure 1: Pi_TT(z)...")

fig, ax = plt.subplots(figsize=(6, 4))
z_arr = np.linspace(0.01, 7.0, 200)

for n in [1, 2, 3, 4, 5]:
    Pi_vals = []
    for z in z_arr:
        try:
            v = float(PiTT_n(z, n))
        except Exception:
            v = np.nan
        Pi_vals.append(v)
    ax.plot(z_arr, Pi_vals, color=COLORS[n], label=LABELS[n], linewidth=1.3)

ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
ax.axhline(1, color="gray", linewidth=0.4, linestyle="--")

# Mark zeros
for n in [1, 2, 3, 4, 5]:
    ax.plot(Z0[n], 0, "o", color=COLORS[n], markersize=5, zorder=5)

ax.set_xlabel(r"$z = k^2/\Lambda^2$")
ax.set_ylabel(r"$\Pi_{\mathrm{TT}}(z)$")
ax.set_xlim(0, 7)
ax.set_ylim(-5, 35)
ax.legend(loc="upper left", fontsize=9)

fig.tight_layout()
out1 = FIGURES_DIR / "fig_predictions_PiTT.pdf"
fig.savefig(out1, dpi=300)
plt.close(fig)
print(f"  Saved: {out1}")


# ==============================================================
# FIGURE 2: Modified Newtonian potential V(r)/V_N
# ==============================================================
print("Generating Figure 2: V(r)/V_N...")

fig, ax = plt.subplots(figsize=(6, 4))
rL = np.linspace(0.01, 5.0, 300)

for n in [1, 2, 3, 5]:  # skip 4 (nearly overlaps 3 and 5)
    m2 = M2[n]
    V = 1.0 - (4.0 / 3.0) * np.exp(-m2 * rL)
    ax.plot(rL, V, color=COLORS[n], label=LABELS[n], linewidth=1.3)

# GR reference
ax.axhline(1, color="gray", linewidth=0.5, linestyle="--", label="GR")

# Mark V(0) = -1/3
ax.axhline(-1.0 / 3.0, color="black", linewidth=0.4, linestyle=":",
           label=r"$V(0)/V_N = -1/3$")

ax.set_xlabel(r"$r \cdot \Lambda$")
ax.set_ylabel(r"$V(r) / V_N(r)$")
ax.set_xlim(0, 5)
ax.set_ylim(-0.5, 1.1)
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
out2 = FIGURES_DIR / "fig_predictions_potential.pdf"
fig.savefig(out2, dpi=300)
plt.close(fig)
print(f"  Saved: {out2}")


# ==============================================================
# FIGURE 3: Master functions phi_n(x) comparison
# ==============================================================
print("Generating Figure 3: phi_n(x)...")

fig, ax = plt.subplots(figsize=(6, 4))
x_arr = np.linspace(0.01, 20.0, 200)

for n in [1, 2, 3, 5]:
    phi_vals = []
    for x in x_arr:
        if n == 1:
            v = float(phi_canonical(x))
        else:
            v = float(phi_n(x, n))
        phi_vals.append(v)
    ax.plot(x_arr, phi_vals, color=COLORS[n], label=LABELS[n], linewidth=1.3)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\varphi_n(x)$")
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", fontsize=9)

fig.tight_layout()
out3 = FIGURES_DIR / "fig_predictions_phi.pdf"
fig.savefig(out3, dpi=300)
plt.close(fig)
print(f"  Saved: {out3}")


# ==============================================================
# FIGURE 4: Comparison heatmap (6 programs x 9 axes)
# ==============================================================
print("Generating Figure 4: Comparison heatmap...")

programs = ["SCT", "LQG", "AS", "CDT", "String", "IDG"]
axes_names = [r"$d_S$(UV)", r"$c_{\log}$", "Singularity",
              "Inflation", "Dispersion", r"$\gamma_{\rm PPN}$",
              "UV prop.", r"$\Lambda_{\rm cc}$", "Matter"]

# Status matrix: 2 = computed/unique, 1 = partial/conditional, 0 = not computed/landscape
status = np.array([
    [2, 1, 1, 2, 2, 2, 2, 0, 2],  # SCT
    [2, 2, 2, 1, 2, 2, 1, 0, 0],  # LQG
    [2, 1, 2, 2, 0, 2, 2, 0, 1],  # AS
    [1, 0, 0, 0, 0, 0, 1, 1, 0],  # CDT
    [1, 1, 1, 0, 1, 2, 1, 0, 0],  # String
    [2, 2, 2, 1, 2, 2, 2, 0, 0],  # IDG
])

cmap = matplotlib.colors.ListedColormap(["#d9534f", "#f0ad4e", "#5cb85c"])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(8, 3.5))
im = ax.imshow(status, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(9))
ax.set_xticklabels(axes_names, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(6))
ax.set_yticklabels(programs, fontsize=9)

# Add text labels
status_text = {0: "n.c.", 1: "cond.", 2: "yes"}
for i in range(6):
    for j in range(9):
        txt = status_text[status[i, j]]
        color = "white" if status[i, j] == 0 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#5cb85c", label="Computed / unique"),
    Patch(facecolor="#f0ad4e", label="Conditional / partial"),
    Patch(facecolor="#d9534f", label="Not computed / landscape"),
]
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1),
          fontsize=8, frameon=True)

fig.tight_layout()
out4 = FIGURES_DIR / "fig_predictions_heatmap.pdf"
fig.savefig(out4, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out4}")


print("\nAll figures generated.")
