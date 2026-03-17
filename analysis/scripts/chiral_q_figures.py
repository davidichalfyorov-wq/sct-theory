# ruff: noqa: E402, I001
"""
Generate publication-quality figures for the D^2-quantization chirality paper.

Figures:
  1. Block-diagonal structure of delta(D^2) in chiral basis (N=16)
  2. Chirality violation correlation: ||{D,gamma5}|| vs ||[delta(D^2),gamma5]||
  3. Loop-order independence: max |pq coefficient| at L=1..8
  5. Spectral truncation convergence: N=4..64

Figure 4 is TikZ (created inline in the LaTeX source).

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

PROJECT_ROOT = ANALYSIS_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "chiral_q"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import proof functions
# ---------------------------------------------------------------------------
from scripts.chiral_q_proof import (  # noqa: E402
    make_gamma5,
    make_random_dirac,
    compute_delta_D2,
    extract_blocks,
    verify_anticommutes_gamma5,
    verify_commutes_gamma5,
)

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "high-vis", "no-latex"])
except ImportError:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ===================================================================
# FIGURE 1: Block-diagonal structure heatmap
# ===================================================================
def figure_block_structure(n: int = 16, seed: int = 42):
    """Heatmap of |delta(D^2)| in the chiral basis showing block-diagonal structure."""
    rng = np.random.default_rng(seed)
    D0 = make_random_dirac(n, rng=rng)
    dD = make_random_dirac(n, rng=rng)
    terms = compute_delta_D2(D0, dD)
    delta_D2 = terms["full"]
    half = n // 2

    # Absolute values for heatmap
    abs_matrix = np.abs(delta_D2)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Panel (a): Full delta(D^2) with block structure visible
    im0 = axes[0].imshow(abs_matrix, cmap="inferno", interpolation="nearest",
                          aspect="equal")
    axes[0].set_title(r"$|\delta(D^2)|$ in chiral basis", fontsize=9)
    axes[0].set_xlabel("Column index", fontsize=8)
    axes[0].set_ylabel("Row index", fontsize=8)
    # Draw block boundaries
    axes[0].axhline(y=half - 0.5, color="white", linewidth=1.0, linestyle="--")
    axes[0].axvline(x=half - 0.5, color="white", linewidth=1.0, linestyle="--")
    # Label blocks
    axes[0].text(half / 2 - 0.5, half / 2 - 0.5, "LL", color="white",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    axes[0].text(half + half / 2 - 0.5, half / 2 - 0.5, "LR", color="white",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    axes[0].text(half / 2 - 0.5, half + half / 2 - 0.5, "RL", color="white",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    axes[0].text(half + half / 2 - 0.5, half + half / 2 - 0.5, "RR", color="white",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label=r"$|M_{ij}|$")

    # Panel (b): Log scale to show off-diagonal is truly zero
    # Add small epsilon to avoid log(0)
    eps = 1e-16
    log_matrix = np.log10(abs_matrix + eps)

    im1 = axes[1].imshow(log_matrix, cmap="inferno", interpolation="nearest",
                          aspect="equal", vmin=-16, vmax=np.max(log_matrix))
    axes[1].set_title(r"$\log_{10}|\delta(D^2)|$ (off-diag $\sim 10^{-16}$)",
                       fontsize=9)
    axes[1].set_xlabel("Column index", fontsize=8)
    axes[1].set_ylabel("Row index", fontsize=8)
    axes[1].axhline(y=half - 0.5, color="white", linewidth=1.0, linestyle="--")
    axes[1].axvline(x=half - 0.5, color="white", linewidth=1.0, linestyle="--")
    axes[1].text(half / 2 - 0.5, half / 2 - 0.5, "LL", color="white",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    axes[1].text(half + half / 2 - 0.5, half / 2 - 0.5, "LR", color="cyan",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    axes[1].text(half / 2 - 0.5, half + half / 2 - 0.5, "RL", color="cyan",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    axes[1].text(half + half / 2 - 0.5, half + half / 2 - 0.5, "RR", color="white",
                  ha="center", va="center", fontsize=9, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], shrink=0.8,
                  label=r"$\log_{10}|M_{ij}|$")

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_block_structure.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Figure 1 saved: {outpath}")
    return outpath


# ===================================================================
# FIGURE 2: Chirality violation correlation
# ===================================================================
def figure_violation_correlation(n: int = 16, n_points: int = 200, seed: int = 123):
    """Plot ||[delta(D^2), gamma5]|| vs ||{D, gamma5}||/||D||."""
    rng = np.random.default_rng(seed)
    gamma5 = make_gamma5(n)
    half = n // 2

    x_vals = []  # chirality violation of D
    y_vals = []  # commutator violation of delta(D^2)

    for i in range(n_points):
        # Interpolate between chiral D and non-chiral D
        t = i / (n_points - 1)  # t in [0, 1]

        # Chiral part: block anti-diagonal
        A_chiral = rng.standard_normal((half, half)) + 1j * rng.standard_normal((half, half))
        D_chiral = np.zeros((n, n), dtype=complex)
        D_chiral[:half, half:] = A_chiral
        D_chiral[half:, :half] = A_chiral.conj().T

        # Non-chiral part: arbitrary self-adjoint
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        D_nonchiral = (M + M.conj().T) / 2.0

        # Interpolate
        D_test = (1.0 - t) * D_chiral + t * D_nonchiral

        # Same for perturbation
        B_chiral = rng.standard_normal((half, half)) + 1j * rng.standard_normal((half, half))
        dD_chiral = np.zeros((n, n), dtype=complex)
        dD_chiral[:half, half:] = B_chiral
        dD_chiral[half:, :half] = B_chiral.conj().T

        N = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        dD_nonchiral = (N + N.conj().T) / 2.0

        dD_test = (1.0 - t) * dD_chiral + t * dD_nonchiral

        # Compute chirality violation of D
        anticomm_D = D_test @ gamma5 + gamma5 @ D_test
        x_val = norm(anticomm_D) / max(norm(D_test), 1e-15)

        # Compute delta(D^2) = {D, dD} + (dD)^2
        delta_D2 = D_test @ dD_test + dD_test @ D_test + dD_test @ dD_test
        comm_delta = delta_D2 @ gamma5 - gamma5 @ delta_D2
        y_val = norm(comm_delta) / max(norm(delta_D2), 1e-15)

        x_vals.append(x_val)
        y_vals.append(y_val)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Color by chirality violation magnitude
    scatter = ax.scatter(x_vals, y_vals, c=x_vals, cmap="coolwarm",
                          s=12, alpha=0.7, edgecolors="none", zorder=2)

    # Highlight the chiral points (near origin)
    mask_chiral = x_vals < 1e-10
    if np.any(mask_chiral):
        ax.scatter(x_vals[mask_chiral], y_vals[mask_chiral],
                    c="green", s=30, marker="*", zorder=3,
                    label=r"$\{D,\gamma_5\}=0$ (exact)")

    # Linear fit for the non-chiral region
    mask_fit = x_vals > 0.1
    if np.sum(mask_fit) > 5:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(x_vals[mask_fit], y_vals[mask_fit], 1)
        x_fit = np.linspace(0.1, max(x_vals), 100)
        y_fit = P.polyval(x_fit, coeffs)
        ax.plot(x_fit, y_fit, "k--", linewidth=1.0, alpha=0.6,
                 label=f"Linear fit (slope $\\approx {coeffs[1]:.2f}$)")

    ax.set_xlabel(r"$\|\{D, \gamma_5\}\| / \|D\|$", fontsize=10)
    ax.set_ylabel(r"$\|[\delta(D^2), \gamma_5]\| / \|\delta(D^2)\|$", fontsize=10)
    ax.set_title("Chirality violation correlation", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    plt.colorbar(scatter, ax=ax, label=r"$\|\{D,\gamma_5\}\|/\|D\|$",
                  shrink=0.8)

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_violation_correlation.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Figure 2 saved: {outpath}")
    return outpath


# ===================================================================
# FIGURE 3: Loop-order independence
# ===================================================================
def figure_loop_independence(n: int = 8, max_L: int = 8,
                             n_trials: int = 50, seed: int = 456):
    """Plot max |pq coefficient| vs loop order L=1..8."""
    from scipy.linalg import funm
    import math

    rng = np.random.default_rng(seed)
    gamma5 = make_gamma5(n)

    loop_orders = list(range(1, max_L + 1))
    max_violations = []

    for L in loop_orders:
        max_viol = 0.0
        for _ in range(n_trials):
            D0 = make_random_dirac(n, rng=rng)
            dD = make_random_dirac(n, rng=rng)

            D0_sq = D0 @ D0
            delta_D2 = D0 @ dD + dD @ D0 + dD @ dD

            # L-th order counterterm: f^(L)(D0^2) * [delta(D^2)]^L / L!
            sign = (-1) ** L
            f_L = funm(D0_sq, lambda x, s=sign: s * np.exp(-x))

            power = np.eye(n, dtype=complex)
            for _ in range(L):
                power = power @ delta_D2

            CT_L = f_L @ power / math.factorial(L)

            # Check off-diagonal blocks (pq cross-terms)
            _, CT_LR, CT_RL, _ = extract_blocks(CT_L, n)
            scale = max(norm(CT_L), 1e-15)
            viol = (norm(CT_LR) + norm(CT_RL)) / scale
            max_viol = max(max_viol, viol)

        max_violations.append(max_viol)

    # Replace exact zeros with a small sub-epsilon value for log plotting
    eps_machine = np.finfo(float).eps  # ~2.2e-16
    plot_violations = [max(v, 1e-17) for v in max_violations]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.semilogy(loop_orders, plot_violations, "ko-", markersize=6,
                 linewidth=1.2, label="Max $|pq|$ coefficient")

    # Machine epsilon line
    ax.axhline(y=eps_machine, color="red", linestyle="--",
                linewidth=0.8, alpha=0.7, label=r"Machine $\varepsilon$")

    # Shade the "numerically zero" region
    ax.axhspan(1e-18, 1e-10, color="green", alpha=0.08)
    ax.text(4.5, 2e-14, "Numerically zero", fontsize=8, ha="center",
             color="green", fontstyle="italic")

    ax.set_xlabel("Loop order $L$", fontsize=10)
    ax.set_ylabel(r"Max $\|pq$ cross-term$\| / \|$CT$_L\|$",
                   fontsize=10)
    ax.set_title(r"Loop-order independence ($N=8$, 50 trials/order)", fontsize=10)
    ax.set_xticks(loop_orders)
    ax.set_ylim(1e-18, 1e-5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_loop_independence.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Figure 3 saved: {outpath}")
    return outpath


# ===================================================================
# FIGURE 5: Spectral truncation convergence
# ===================================================================
def figure_spectral_truncation(n_values=None, n_trials: int = 50, seed: int = 789):
    """Plot max chirality violation vs matrix size N for chiral and non-chiral D."""
    if n_values is None:
        n_values = [4, 8, 16, 32, 64]

    chiral_violations = []
    nonchiral_violations = []

    for n in n_values:
        rng = np.random.default_rng(seed + n)
        gamma5 = make_gamma5(n)
        half = n // 2

        max_chiral_viol = 0.0
        max_nonchiral_viol = 0.0

        for _ in range(n_trials):
            # CHIRAL case: {D, gamma5} = 0
            D0 = make_random_dirac(n, rng=rng)
            dD = make_random_dirac(n, rng=rng)
            delta_D2 = D0 @ dD + dD @ D0 + dD @ dD
            comm = delta_D2 @ gamma5 - gamma5 @ delta_D2
            scale = max(norm(delta_D2), 1e-15)
            max_chiral_viol = max(max_chiral_viol, norm(comm) / scale)

            # NON-CHIRAL case: D arbitrary self-adjoint
            M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            D_nc = (M + M.conj().T) / 2.0
            N_mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            dD_nc = (N_mat + N_mat.conj().T) / 2.0
            delta_D2_nc = D_nc @ dD_nc + dD_nc @ D_nc + dD_nc @ dD_nc
            comm_nc = delta_D2_nc @ gamma5 - gamma5 @ delta_D2_nc
            scale_nc = max(norm(delta_D2_nc), 1e-15)
            max_nonchiral_viol = max(max_nonchiral_viol,
                                      norm(comm_nc) / scale_nc)

        chiral_violations.append(max_chiral_viol)
        nonchiral_violations.append(max_nonchiral_viol)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.semilogy(n_values, chiral_violations, "go-", markersize=7,
                 linewidth=1.2, label=r"$\{D,\gamma_5\}=0$ (chiral)")
    ax.semilogy(n_values, nonchiral_violations, "rs-", markersize=7,
                 linewidth=1.2, label=r"$\{D,\gamma_5\}\neq 0$ (non-chiral)")

    # Machine epsilon
    ax.axhline(y=np.finfo(float).eps, color="gray", linestyle=":",
                linewidth=0.8, label=r"Machine $\varepsilon$")

    ax.set_xlabel("Matrix size $N$", fontsize=10)
    ax.set_ylabel(r"Max $\|[\delta(D^2), \gamma_5]\| / \|\delta(D^2)\|$",
                   fontsize=10)
    ax.set_title("Spectral truncation convergence (50 trials)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])

    fig.tight_layout()
    outpath = FIGURES_DIR / "fig_spectral_truncation.pdf"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Figure 5 saved: {outpath}")
    return outpath


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("Generating figures for D^2-quantization chirality paper...")
    print("=" * 60)

    paths = []
    print("\nFigure 1: Block-diagonal structure (N=16)")
    paths.append(figure_block_structure(n=16))

    print("\nFigure 2: Chirality violation correlation")
    paths.append(figure_violation_correlation(n=16, n_points=200))

    print("\nFigure 3: Loop-order independence (L=1..8)")
    paths.append(figure_loop_independence(n=8, max_L=8, n_trials=50))

    print("\nFigure 5: Spectral truncation convergence (N=4..64)")
    paths.append(figure_spectral_truncation(n_values=[4, 8, 16, 32, 64]))

    print("\n" + "=" * 60)
    print(f"All {len(paths)} figures generated successfully.")
    for p in paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
