# ruff: noqa: E402, I001
"""
GZ-D: Figures for the Entire Part g(z) Analysis.

Generates two publication-quality figures:
  1. gz_constancy.pdf -- g_A(z) vs z showing it is flat at -13/60
  2. gz_sum_rule.pdf  -- partial sums of Sum R_n/z_n converging to 13/60

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.gz_entire_part import (
    LOCAL_C2,
    compute_g_A,
    get_full_catalogue,
    verify_sum_rule,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "gz"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colors
SCT_BLUE = "#2E5B88"
SCT_RED = "#B8352A"
SCT_GREEN = "#357A38"
SCT_GOLD = "#C4960C"
SCT_GRAY = "#666666"


def figure_constancy(catalogue: list[dict], dps: int = 60) -> Path:
    """
    Figure 1: g_A(z) vs z for real z in [0.3, 100].

    Shows that g_A is constant at -13/60 (horizontal line)
    within the truncation error of the finite pole sum.
    """
    mp.mp.dps = dps

    z_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0]

    g_A_re = []
    for z in z_values:
        g = compute_g_A(mp.mpc(z, 0), catalogue, dps=dps)
        g_A_re.append(float(mp.re(g)))

    target = float(-LOCAL_C2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1])

    # Upper panel: g_A(z) vs z
    ax1.semilogx(z_values, g_A_re, "o-", color=SCT_BLUE, markersize=5,
                 linewidth=1.2, label=r"$g_A(z)$ (computed)")
    ax1.axhline(y=target, color=SCT_RED, linestyle="--", linewidth=1.0,
                label=r"$-c_2 = -13/60$")
    ax1.set_ylabel(r"$g_A(z)$", fontsize=13)
    ax1.set_title(r"Entire part $g_A(z)$ of $1/(z\,\Pi_{\mathrm{TT}})$", fontsize=14)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.set_xlim(0.2, 150)
    y_span = max(abs(max(g_A_re) - target), abs(min(g_A_re) - target))
    ax1.set_ylim(target - max(y_span * 3, 0.005), target + max(y_span * 3, 0.005))
    ax1.grid(True, alpha=0.3)

    # Lower panel: deviation from -13/60
    deviations = [g - target for g in g_A_re]
    ax2.semilogx(z_values, deviations, "s-", color=SCT_GREEN, markersize=4,
                 linewidth=1.0)
    ax2.axhline(y=0, color=SCT_GRAY, linestyle="-", linewidth=0.5)
    ax2.set_xlabel(r"$z$", fontsize=13)
    ax2.set_ylabel(r"$g_A(z) - (-13/60)$", fontsize=12)
    ax2.set_xlim(0.2, 150)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Deviation from constant (truncation error)", fontsize=11)

    fig.tight_layout()
    outpath = FIGURES_DIR / "gz_constancy.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def figure_sum_rule(catalogue: list[dict], dps: int = 60) -> Path:
    """
    Figure 2: Partial sums of Sum R_n/z_n converging to c_2 = 13/60.

    Shows the cumulative sum as poles are added.
    """
    mp.mp.dps = dps

    sr = verify_sum_rule(catalogue, dps=dps)
    partial_sums = sr["partial_sums"]

    n_vals = list(range(1, len(partial_sums) + 1))
    ps_re = [p["partial_sum_re"] for p in partial_sums]
    labels = [p["label"] for p in partial_sums]

    target = float(LOCAL_C2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1])

    # Upper panel: partial sums
    ax1.plot(n_vals, ps_re, "o-", color=SCT_BLUE, markersize=5,
             linewidth=1.2, label=r"$\sum_{n \leq N} R_n / z_n$")
    ax1.axhline(y=target, color=SCT_RED, linestyle="--", linewidth=1.0,
                label=r"$c_2 = 13/60$")
    ax1.set_ylabel(r"Partial sum", fontsize=13)
    ax1.set_title(r"Sum rule: $\sum_n R_n/z_n = c_2 = 13/60$", fontsize=14)
    ax1.legend(fontsize=11, loc="center right")
    ax1.grid(True, alpha=0.3)

    # Annotate first few points
    for i, lbl in enumerate(labels[:3]):
        short = lbl.split("(")[0].strip()
        ax1.annotate(short, (n_vals[i], ps_re[i]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=8, color=SCT_GRAY)

    # Lower panel: deficit from target
    deficits = [p - target for p in ps_re]
    ax2.plot(n_vals, deficits, "s-", color=SCT_GREEN, markersize=4,
             linewidth=1.0)
    ax2.axhline(y=0, color=SCT_GRAY, linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Number of poles included", fontsize=13)
    ax2.set_ylabel(r"Deficit from $c_2$", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Convergence of sum rule", fontsize=11)

    fig.tight_layout()
    outpath = FIGURES_DIR / "gz_sum_rule.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def main():
    """Generate all GZ figures."""
    print("=" * 60)
    print("GZ-D: Generating figures")
    print("=" * 60)

    print("\n  Loading ghost catalogue...")
    catalogue = get_full_catalogue(dps=60)
    print(f"  {len(catalogue)} poles loaded.")

    print("\n  Figure 1: g_A constancy")
    figure_constancy(catalogue, dps=60)

    print("\n  Figure 2: Sum rule convergence")
    figure_sum_rule(catalogue, dps=60)

    print("\n  All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
