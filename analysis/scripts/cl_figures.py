# ruff: noqa: E402, I001
"""
CL Commutativity of Limits -- Publication Figures.

Generates 2 figures as PDF in analysis/figures/cl/:
  1. cl_convergence.pdf  -- T_FK(N,s) vs N for several s values
  2. cl_M_bounds.pdf     -- M_n vs n, Weierstrass bound decay

Execute:  python analysis/scripts/cl_figures.py

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sct_tools.plotting import SCT_COLORS

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = ANALYSIS_DIR / "figures" / "cl"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ANALYSIS_DIR / "results" / "cl"
RESULTS_FILE = RESULTS_DIR / "cl_commutativity_results.json"

# Try to set SciencePlots style, fall back gracefully
try:
    plt.style.use(["science", "ieee"])
except Exception:
    pass

# Always disable usetex and set clean defaults
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
def load_results() -> dict:
    """Load the CL commutativity results JSON."""
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ===================================================================
# Figure 1: Convergence of T_FK(N,s) vs N
# ===================================================================

def figure_convergence(data: dict):
    """
    Plot T_FK(N,s) vs N for multiple s values, demonstrating convergence.

    Shows that the N-pole truncated amplitude converges as N increases,
    with convergence rate consistent with the Weierstrass M-test bound.
    """
    conv_data = data["convergence_vs_N"]["data"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Color cycle
    colors = [
        SCT_COLORS.get("spin2", "#E91E63"),
        SCT_COLORS.get("scalar", "#2196F3"),
        SCT_COLORS.get("vector", "#FF9800"),
        SCT_COLORS.get("dirac", "#4CAF50"),
        SCT_COLORS.get("combined", "#9C27B0"),
    ]

    # Left panel: T_FK(N,s) vs N
    for i, cd in enumerate(conv_data):
        s_val = cd["s"]
        N_vals = cd["N_values"]
        T_vals = cd["T_FK_values"]
        color = colors[i % len(colors)]
        ax1.plot(N_vals, T_vals, "o-", color=color, markersize=4,
                 label=f"s = {s_val}")

    ax1.set_xlabel("N (number of poles)")
    ax1.set_ylabel(r"$T_{\rm FK}(N, s)$")
    ax1.set_title("Convergence of N-pole amplitude")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Right panel: successive differences (log scale)
    for i, cd in enumerate(conv_data):
        s_val = cd["s"]
        diffs = cd["successive_differences"]
        if not diffs:
            continue
        N_mid = cd["N_values"][1:len(diffs) + 1]
        color = colors[i % len(colors)]
        ax2.semilogy(N_mid, diffs, "s-", color=color, markersize=4,
                     label=f"s = {s_val}")

    # Add 1/N^2 reference line
    N_ref = np.array([4, 6, 8, 10, 12])
    # Normalize to match the scale of actual differences
    all_first_diffs = [
        cd["successive_differences"][0]
        for cd in conv_data
        if cd["successive_differences"]
    ]
    if all_first_diffs:
        ref_scale = max(all_first_diffs) * 2
        ref_line = ref_scale * (4.0 / N_ref) ** 2
        ax2.semilogy(N_ref, ref_line, "k--", alpha=0.5,
                     label=r"$\sim 1/N^2$ reference")

    ax2.set_xlabel("N (number of poles)")
    ax2.set_ylabel(r"$|T_{\rm FK}(N+2) - T_{\rm FK}(N)|$")
    ax2.set_title("Successive differences (convergence rate)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    outpath = FIG_DIR / "cl_convergence.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ===================================================================
# Figure 2: Weierstrass M-test bounds
# ===================================================================

def figure_M_bounds(data: dict):
    """
    Plot M_n = |R_n|/Im(z_n) vs pole index n, showing the decay
    that guarantees convergence via the Weierstrass M-test.
    """
    m_data = data["weierstrass_M_test"]
    M_values = m_data["M_values"]
    partial_sums = m_data["partial_sums"]
    total_est = m_data["total_estimate"]

    n_vals = list(range(1, len(M_values) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: M_n vs n (log scale)
    ax1.semilogy(n_vals, M_values, "o-",
                 color=SCT_COLORS.get("spin2", "#E91E63"),
                 markersize=6, label=r"$M_n = |R_n|/{\rm Im}(z_n)$")

    # Add 1/n^2 reference
    n_ref = np.array(n_vals, dtype=float)
    if M_values:
        # Normalize: M_1 ~ C_R / (z_1 * b_1), use first point as anchor
        ref = M_values[0] * (1.0 / n_ref) ** 2
        ax1.semilogy(n_ref, ref, "k--", alpha=0.5,
                     label=r"$\sim C / n^2$ reference")

    ax1.set_xlabel("Complex pole pair index n")
    ax1.set_ylabel(r"$M_n$")
    ax1.set_title(r"Weierstrass bound: $M_n = |R_n| / {\rm Im}(z_n)$")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Right panel: partial sums of M_n
    ax2.plot(n_vals, partial_sums, "s-",
             color=SCT_COLORS.get("scalar", "#2196F3"),
             markersize=6, label=r"$\sum_{k=1}^{n} M_k$")

    # Add total estimate line
    ax2.axhline(y=total_est, color="red", linestyle=":", alpha=0.7,
                label=f"Estimated total = {total_est:.4e}")

    ax2.set_xlabel("Number of complex pole pairs included")
    ax2.set_ylabel(r"$\sum_{k=1}^{n} M_k$")
    ax2.set_title("Partial sums (convergence of Weierstrass series)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    outpath = FIG_DIR / "cl_M_bounds.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Generate all CL figures."""
    print("=" * 60)
    print("CL Commutativity of Limits -- Figure Generation")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run cl_commutativity.py first to generate results.")
        sys.exit(1)

    data = load_results()
    print(f"\nLoaded results: verdict = {data['verdict']['status']}")

    print("\nGenerating figures:")
    figure_convergence(data)
    figure_M_bounds(data)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
