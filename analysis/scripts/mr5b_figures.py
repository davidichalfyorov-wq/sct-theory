# ruff: noqa: E402, I001
"""
MR-5b Figures: Ratio comparison chart for two-loop D=0 analysis.

Fig 1: SM a_6 ratio comparison (spectral action a_6 invariant ratios
       normalized to CCC coefficient).

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.mr5b_two_loop import (
    INVARIANT_LABELS,
    ratio_comparison,
    sm_a6_total,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr5b"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Use default style (SciencePlots with tex unavailable in this environment)
plt.style.use("default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
})


def fig1_ratio_comparison() -> Path:
    """Fig 1: SM a_6 ratio comparison chart.

    Shows the ratios of each invariant's coefficient to the CCC coefficient
    in the SM total a_6. This visualizes the tensor structure that the
    two-loop counterterm must match for absorption to work.
    """
    ratios_data = ratio_comparison(dps=30)
    ratios = ratios_data["sm_a6_ratios_normalized_to_CCC"]

    # Prepare data
    labels = list(INVARIANT_LABELS)
    values = [ratios[key] for key in INVARIANT_LABELS]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
    # Highlight CCC
    colors[5] = "#3498db"

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Ratio to CCC coefficient", fontsize=12)
    ax.set_title("SM a_6 invariant ratios (normalized to CCC)", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=1, color="#3498db", linewidth=0.8, linestyle="--",
               label="CCC reference")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.3 if x_pos >= 0 else -0.3
        ax.text(x_pos + offset, i, f"{val:.2f}", va="center", ha=ha,
                fontsize=8, fontweight="bold")

    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(min(values) * 1.3, max(values) * 1.15)
    fig.tight_layout()

    outpath = FIGURES_DIR / "mr5b_a6_ratio_comparison.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 1 saved to {outpath}")
    return outpath


def fig2_per_spin_breakdown() -> Path:
    """Fig 2: Per-spin a_6 CCC coefficient breakdown.

    Shows how scalar, Dirac, and vector fields contribute to the
    total SM a_6 CCC coefficient.
    """
    sm = sm_a6_total(dps=30)

    ccc_scalar = float(sm["scalar"]["coefficients"]["Riem^3 (CCC)"])
    ccc_dirac = float(sm["dirac"]["coefficients"]["Riem^3 (CCC)"])
    ccc_vector = float(sm["vector"]["coefficients"]["Riem^3 (CCC)"])

    contributions = [
        4 * ccc_scalar,
        22.5 * ccc_dirac,
        12 * ccc_vector,
    ]
    labels_spin = [
        f"Scalar (N_s=4)\n{contributions[0]:.1f}",
        f"Dirac (N_D=22.5)\n{contributions[1]:.1f}",
        f"Vector (N_v=12)\n{contributions[2]:.1f}",
    ]
    colors_spin = ["#f39c12", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(3), contributions, color=colors_spin,
           edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_spin, fontsize=10)
    ax.set_ylabel("N_s * a_6[CCC]", fontsize=12)
    ax.set_title("CCC coefficient: per-spin SM contributions", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.8)

    total = sum(contributions)
    ax.axhline(y=total, color="#3498db", linewidth=1.5, linestyle="--",
               label=f"SM total = {total:.1f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    outpath = FIGURES_DIR / "mr5b_ccc_per_spin.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2 saved to {outpath}")
    return outpath


if __name__ == "__main__":
    fig1_ratio_comparison()
    fig2_per_spin_breakdown()
    print("All figures generated.")
