"""Generate the SCT Theory research roadmap progress chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Try SciencePlots for nicer defaults
try:
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
    })

# === Data: (label, status, group) ===
# status: "complete", "closed", "conditional", "negative", "pending"
tasks = [
    # --- Core Form Factors ---
    ("NT-1: Dirac form factors",       "complete",    "Core"),
    ("NT-1b: Scalar form factors",     "complete",    "Core"),
    ("NT-1b: Vector form factors",     "complete",    "Core"),
    ("NT-1b: Combined SM",             "complete",    "Core"),
    ("NT-2: Entire-function proof",    "complete",    "Core"),
    # --- Field Equations & Cosmology ---
    ("NT-4a: Linearized field eqs",    "complete",    "Field Eqs"),
    ("NT-4b: Nonlinear EOM",           "complete",    "Field Eqs"),
    ("NT-4c: FLRW cosmology",          "complete",    "Field Eqs"),
    ("INF-1: Spectral inflation",      "negative",    "Field Eqs"),
    ("INF-2: Dilaton inflation",       "negative",    "Field Eqs"),
    ("MT-2: Modified cosmology",       "negative",    "Field Eqs"),
    # --- Solar System & Lab ---
    ("PPN-1: Solar system tests",      "complete",    "Solar"),
    ("LT-3d: Laboratory tests",        "complete",    "Solar"),
    # --- UV Consistency ---
    ("MR-1: Lorentzian formulation",   "complete",    "UV"),
    ("MR-2: Unitarity (D\u00b2-quant)", "closed",    "UV"),
    ("OT: Optical theorem",            "closed",      "UV"),
    ("MR-3: Causality",                "conditional", "UV"),
    ("MR-4: Two-loop structure",       "complete",    "UV"),
    ("MR-5: Finiteness (all-orders)",  "conditional", "UV"),
    ("MR-5b: Two-loop D=0",           "complete",    "UV"),
    ("MR-7: Graviton scattering",      "complete",    "UV"),
    ("LT-1: All-orders UV",            "pending",     "UV"),
    # --- Auxiliary & Structural ---
    ("CL: Commutativity",              "complete",    "Aux"),
    ("GZ: Entire part",                "complete",    "Aux"),
    ("SS: Scalar sector",              "complete",    "Aux"),
    ("KK: Kubo\u2013Kugo resolution",  "complete",    "Aux"),
    ("A3: Ghost width",                "complete",    "Aux"),
    ("GP: Dressed propagator",         "complete",    "Aux"),
    ("MR-6: Convergence analysis",     "complete",    "Aux"),
    ("NT-3: Spectral dimension",       "conditional", "Aux"),
    ("MT-1: Black hole entropy",       "conditional", "Aux"),
    ("MR-9: Singularity",              "conditional", "Aux"),
    # --- Foundation ---
    ("FND-1: Finite-nerve route",      "negative",    "Foundation"),
]

# Colors
COLORS = {
    "complete":    "#27ae60",
    "closed":      "#2980b9",
    "conditional": "#f39c12",
    "negative":    "#c0392b",
    "pending":     "#bdc3c7",
}
LABELS = {
    "complete":    "Complete / Certified",
    "closed":      "Closed (verified)",
    "conditional": "Conditional",
    "negative":    "Negative result",
    "pending":     "Pending",
}

# Group separators
GROUP_ORDER = ["Core", "Field Eqs", "Solar", "UV", "Aux", "Foundation"]
GROUP_NAMES = {
    "Core":       "Core Form Factors",
    "Field Eqs":  "Field Equations & Cosmology",
    "Solar":      "Solar System & Laboratory",
    "UV":         "UV Consistency Path",
    "Aux":        "Auxiliary & Structural",
    "Foundation": "Foundational",
}

# Build ordered list with group headers
rows = []
for g in GROUP_ORDER:
    rows.append(("__GROUP__", g))
    for label, status, group in tasks:
        if group == g:
            rows.append((label, status))

# Reverse for bottom-to-top plotting
rows = rows[::-1]

fig, ax = plt.subplots(figsize=(10, 12.5))

y_pos = 0
y_positions = []
y_labels = []
bar_colors = []
group_label_positions = []

for label, status in rows:
    if label == "__GROUP__":
        group_label_positions.append((y_pos + 0.3, GROUP_NAMES[status]))
        y_pos += 0.9
        continue
    y_positions.append(y_pos)
    y_labels.append(label)
    bar_colors.append(COLORS[status])
    y_pos += 1

# Draw bars
bar_height = 0.7
bars = ax.barh(y_positions, [1.0]*len(y_positions), height=bar_height,
               color=bar_colors, edgecolor="white", linewidth=0.5)

# Group header labels
for yp, gname in group_label_positions:
    ax.text(-0.02, yp, gname, transform=ax.get_yaxis_transform(),
            fontsize=8.5, fontweight="bold", color="#2c3e50",
            va="bottom", ha="right")
    ax.axhline(y=yp - 0.3, color="#ecf0f1", linewidth=0.8, zorder=0)

# Axes
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlim(0, 1.08)
ax.set_xticks([])
ax.set_xlabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(left=False)

# Title
ax.set_title("SCT Theory \u2014 Research Roadmap",
             fontsize=14, fontweight="bold", pad=15, color="#2c3e50")

# Legend
patches = [mpatches.Patch(color=COLORS[s], label=LABELS[s])
           for s in ["complete", "closed", "conditional", "negative", "pending"]]
ax.legend(handles=patches, loc="lower right", fontsize=7.5,
          framealpha=0.9, edgecolor="#bdc3c7")

# Status count annotation
n_complete = sum(1 for _, s, _ in tasks if s in ("complete", "closed"))
n_cond = sum(1 for _, s, _ in tasks if s == "conditional")
n_neg = sum(1 for _, s, _ in tasks if s == "negative")
n_pend = sum(1 for _, s, _ in tasks if s == "pending")
total = len(tasks)
ax.text(0.98, 0.01,
        f"{n_complete}/{total} complete  \u00b7  {n_cond} conditional  "
        f"\u00b7  {n_neg} negative  \u00b7  {n_pend} pending",
        transform=ax.transAxes, fontsize=7, color="#7f8c8d",
        ha="right", va="bottom")

plt.tight_layout()

out = Path(__file__).resolve().parent.parent.parent / "docs" / "figures" / "roadmap_progress.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
print(f"Tasks: {total} ({n_complete} complete, {n_cond} conditional, {n_neg} negative, {n_pend} pending)")
