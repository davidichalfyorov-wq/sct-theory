"""Generate the SCT Theory research roadmap progress chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0,
})

# ── Task data ────────────────────────────────────────────────
# (label, status, group)
tasks = [
    ("NT-1  Dirac form factors",       "complete",    "Core"),
    ("NT-1b  Scalar form factors",     "complete",    "Core"),
    ("NT-1b  Vector form factors",     "complete",    "Core"),
    ("NT-1b  Combined SM totals",      "complete",    "Core"),
    ("NT-2  Entire-function proof",    "complete",    "Core"),

    ("NT-4a  Linearized field eqs",    "complete",    "Eqs"),
    ("NT-4b  Nonlinear EOM",           "complete",    "Eqs"),
    ("NT-4c  FLRW cosmology",          "complete",    "Eqs"),
    ("INF-1  Spectral inflation",      "negative",    "Eqs"),
    ("INF-2  Dilaton inflation",       "negative",    "Eqs"),
    ("MT-2  Late-time cosmology",      "negative",    "Eqs"),

    ("PPN-1  Solar system tests",      "complete",    "Solar"),
    ("LT-3d  Laboratory bounds",       "complete",    "Solar"),

    ("MR-1  Lorentzian formulation",   "complete",    "UV"),
    ("MR-2  Unitarity (D\u00b2-quant)","closed",      "UV"),
    ("OT  Optical theorem",            "closed",      "UV"),
    ("MR-3  Causality",                "conditional", "UV"),
    ("MR-4  Two-loop structure",       "complete",    "UV"),
    ("MR-5  All-orders finiteness",    "conditional", "UV"),
    ("MR-5b  Two-loop D=0",           "complete",    "UV"),
    ("MR-7  Graviton scattering",      "complete",    "UV"),
    ("LT-1  All-orders UV proof",      "pending",     "UV"),

    ("CL  Commutativity",              "complete",    "Aux"),
    ("GZ  Entire part",                "complete",    "Aux"),
    ("SS  Scalar sector",              "complete",    "Aux"),
    ("KK  Kubo\u2013Kugo resolution",  "complete",    "Aux"),
    ("A3  Ghost width",                "complete",    "Aux"),
    ("GP  Dressed propagator",         "complete",    "Aux"),
    ("MR-6  Convergence",              "complete",    "Aux"),
    ("NT-3  Spectral dimension",       "conditional", "Aux"),
    ("MT-1  Black hole entropy",       "conditional", "Aux"),
    ("MR-9  Singularity",              "conditional", "Aux"),

    ("FND-1  Finite-nerve (Paper 6)",   "negative",    "Fnd"),
    ("FND-1  Route 1: Ensemble BD",     "negative",    "Fnd"),
    ("FND-1  Route 3: DW bridge",       "negative",    "Fnd"),
    ("FND-1  BD-comm CRN (5 metrics)",   "in_progress", "Fnd"),
    ("FND-1  Route 2: Link-graph",      "in_progress", "Fnd"),
    ("FND-1  Analytical (pp-wave)",      "complete",    "Fnd"),

    ("FUND  Fundamental program",       "complete",    "Meta"),
    ("CHIRAL-Q  UV-finiteness",         "complete",    "Meta"),
    ("COMP-1  Roadmap completion",      "pending",     "Meta"),
]

# ── Palette & labels ─────────────────────────────────────────
PAL = {
    "complete":    "#1a9850",
    "closed":      "#4575b4",
    "conditional": "#e6a817",
    "negative":    "#d73027",
    "pending":     "#bababa",
    "in_progress": "#9b59b6",
}
STATUS_TEXT = {
    "complete":    "COMPLETE",
    "closed":      "CLOSED",
    "conditional": "CONDITIONAL",
    "negative":    "NEGATIVE",
    "pending":     "PENDING",
    "in_progress": "IN PROGRESS",
}
GROUP_ORDER = ["Core", "Eqs", "Solar", "UV", "Aux", "Fnd", "Meta"]
GROUP_TITLE = {
    "Core":  "CORE FORM FACTORS",
    "Eqs":   "FIELD EQUATIONS & COSMOLOGY",
    "Solar": "SOLAR SYSTEM & LABORATORY",
    "UV":    "UV CONSISTENCY",
    "Aux":   "AUXILIARY & STRUCTURAL",
    "Fnd":   "FOUNDATIONAL SYNTHESIS (FND-1)",
    "Meta":  "META & PROGRAMS",
}

# ── Build row list (top to bottom) ───────────────────────────
rows = []  # (kind, payload)   kind: "header" | "task"
for g in GROUP_ORDER:
    rows.append(("header", g))
    for label, status, group in tasks:
        if group == g:
            rows.append(("task", (label, status)))

# ── Layout constants ─────────────────────────────────────────
ROW_H   = 0.55          # row height for tasks
HDR_H   = 0.50          # header text area
HDR_GAP = 0.45          # extra gap ABOVE each header (separates groups)
BAR_W   = 0.60          # bar width (fraction of row)
DOT_R   = 0.11          # status dot radius

fig_w = 11
total_h = sum((HDR_H + HDR_GAP) if k == "header" else ROW_H for k, _ in rows) + 1.8
fig, ax = plt.subplots(figsize=(fig_w, total_h * 0.38 + 1.0))

y = total_h  # start from top

for kind, payload in rows:
    if kind == "header":
        y -= HDR_GAP  # gap above header
        # separator line at top of gap
        ax.plot([0, fig_w - 0.5], [y + HDR_GAP * 0.5, y + HDR_GAP * 0.5],
                color="#d0d0d0", linewidth=0.6, zorder=0)
        y -= HDR_H    # header text area
        # group title centered in its area
        ax.text(0.15, y + HDR_H * 0.5, GROUP_TITLE[payload],
                fontsize=7.5, fontweight="bold", color="#555555",
                va="center", family="sans-serif")
    else:
        y -= ROW_H
        label, status = payload
        col = PAL[status]

        # task name
        ax.text(0.25, y + ROW_H / 2, label,
                fontsize=8.2, va="center", color="#2c3e50")

        # colored bar
        bar_x = 5.8
        bar_len = 3.6
        bar = FancyBboxPatch(
            (bar_x, y + ROW_H * 0.18), bar_len, ROW_H * 0.64,
            boxstyle="round,pad=0.04", facecolor=col, edgecolor="none",
            alpha=0.88, zorder=2)
        ax.add_patch(bar)

        # status text inside bar
        ax.text(bar_x + bar_len / 2, y + ROW_H / 2,
                STATUS_TEXT[status],
                fontsize=6.8, fontweight="bold", color="white",
                va="center", ha="center", zorder=3)

# ── Title ────────────────────────────────────────────────────
ax.text(fig_w / 2, total_h + 0.6,
        "SCT Theory \u2014 Research Roadmap",
        fontsize=15, fontweight="bold", ha="center", color="#2c3e50")

# ── Summary line ─────────────────────────────────────────────
n_comp = sum(1 for _, s, _ in tasks if s in ("complete", "closed"))
n_cond = sum(1 for _, s, _ in tasks if s == "conditional")
n_neg  = sum(1 for _, s, _ in tasks if s == "negative")
n_pend = sum(1 for _, s, _ in tasks if s == "pending")
tot    = len(tasks)

summary = (f"{n_comp}/{tot} complete  \u00b7  {n_cond} conditional  "
           f"\u00b7  {n_neg} negative  \u00b7  {n_pend} pending")
ax.text(fig_w / 2, -0.3, summary,
        fontsize=7.5, ha="center", color="#888888")

# ── Axes cleanup ─────────────────────────────────────────────
ax.set_xlim(-0.2, fig_w + 0.3)
ax.set_ylim(-0.7, total_h + 1.1)
ax.set_aspect("equal")
ax.axis("off")

out = Path(__file__).resolve().parent.parent.parent / "docs" / "figures" / "roadmap_progress.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
print(f"Saved: {out}  ({out.stat().st_size / 1024:.0f} KiB)")
print(f"Tasks: {tot}  ({n_comp} complete, {n_cond} conditional, {n_neg} negative, {n_pend} pending)")
