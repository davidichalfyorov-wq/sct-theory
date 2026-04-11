# ruff: noqa: E402, I001
"""
MR-9 / Gap G1: n_eff(r) phase diagram for the CJ slope classifier
on Schwarzschild background with SCT localized form factors.

Computes J(r) = sum(c_k * u_k) via the K=8 layer P(alpha) IVP from
horizon inward, then extracts n_eff = (d ln J^2 / d ln r + 6) / 2.

Key predictions:
  - SCT regime  (r << 1/Lambda): n_eff -> 3 - sqrt(6) ~ 0.551
  - Schwarzschild regime (r >> 1/Lambda): n_eff -> 0
  - Transition at r ~ 1.6/Lambda

David Alfyorov, SCT Theory project, 2026-04-06
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# ─── Project paths ──────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent.parent
FIGURES = PROJECT / "analysis" / "figures"
RESULTS = PROJECT / "analysis" / "results" / "gap_g1"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Add sct_tools to path
sys.path.insert(0, str(PROJECT / "analysis"))

# Try SciencePlots
try:
    import scienceplots  # noqa: F401
    _SP = True
except ImportError:
    _SP = False


# ====================================================================
# P(alpha) kernel and layer construction
# ====================================================================
def build_alpha_layers(K: int = 8):
    """Gauss-Legendre quadrature layers for the symmetric P(alpha) kernel."""
    x, w = leggauss(K)
    alpha = 0.25 * (x + 1.0)          # [-1,1] -> [0, 1/2]
    w_q = 0.25 * w * 2.0              # Jacobian x symmetry
    tau = alpha * (1.0 - alpha)
    P = -89.0 / 24.0 + (43.0 / 6.0) * tau + (236.0 / 3.0) * tau**2

    idx = np.argsort(tau)
    tau, w_q, P, alpha = tau[idx], w_q[idx], P[idx], alpha[idx]
    c = w_q * P
    dtau = np.diff(np.concatenate(([0.0], tau)))

    integral = np.dot(w_q, P)
    print(f"  P(alpha) integral = {integral:.8f}  (13/120 = {13/120:.8f})")
    return dict(tau=tau, c=c, dtau=dtau, P=P, w=w_q, K=K)


# ====================================================================
# Schwarzschild background
# ====================================================================
def H_sch(r, M):
    return 1.0 - 2.0 * M / r

def Hp_sch(r, M):
    return 2.0 * M / r**2

def W_sch(r, M):
    """Weyl amplitude on Schwarzschild: W = -2M/r^3."""
    return -2.0 * M / r**3


# ====================================================================
# Stage 1: fixed-background IVP
# ====================================================================
def stage1_rhs(r, y, M, layers, Lam):
    """ODE RHS for u_k on fixed Schwarzschild.

    H u_k'' + (H' + 2H/r) u_k' - 6H/r^2 u_k = (Lam^2/dtau_k)(u_k - u_{k-1})
    """
    K = layers['K']
    dtau = layers['dtau']

    H  = H_sch(r, M)
    Hp = Hp_sch(r, M)
    W  = W_sch(r, M)

    A = Hp + 2.0 * H / r
    S = 6.0 * H / r**2

    dy = np.zeros(2 * K)
    for k in range(K):
        uk = y[2 * k]
        vk = y[2 * k + 1]
        u_prev = W if k == 0 else y[2 * (k - 1)]

        lam_k = Lam**2 / dtau[k]

        if abs(H) < 1e-14:
            upp = 0.0
        else:
            upp = (lam_k * (uk - u_prev) - A * vk + S * uk) / H

        dy[2 * k] = vk
        dy[2 * k + 1] = upp

    return dy


def run_ivp(M=1.0, Lam=1.0, K=8, r_start_frac=0.999, r_end_frac=0.005,
            n_pts=10000):
    """Integrate u_k from near-horizon inward; return r, J, n_eff arrays."""
    layers = build_alpha_layers(K)
    r_h = 2.0 * M
    r_start = r_h * r_start_frac
    r_end = r_h * r_end_frac

    # Initial conditions: stationary approximation u_k ~ W / (1 + 4 tau_k)
    W0 = W_sch(r_start, M)
    y0 = np.zeros(2 * K)
    for k in range(K):
        fac = 1.0 / (1.0 + 4.0 * layers['tau'][k])
        y0[2 * k] = W0 * fac
        y0[2 * k + 1] = (6.0 * M / r_start**4) * fac

    r_eval = np.linspace(r_start, r_end, n_pts)
    sol = solve_ivp(
        lambda r, y: stage1_rhs(r, y, M, layers, Lam),
        [r_start, r_end],
        y0,
        method='RK45',
        t_eval=r_eval,
        rtol=1e-10,
        atol=1e-12,
        max_step=(r_start - r_end) / 500,
    )

    if sol.status != 0:
        print(f"  WARNING: {sol.message}")
    else:
        print(f"  OK: {sol.t.size} points, r in [{sol.t[-1]:.4e}, {sol.t[0]:.4e}]")

    r = sol.t
    c = layers['c']

    # J(r) = sum_k c_k u_k(r)
    J = np.zeros(len(r))
    for k in range(K):
        J += c[k] * sol.y[2 * k]

    # n_eff from slope of ln(J^2)
    ln_r = np.log(r)
    J2 = J**2
    ln_J2 = np.log(np.maximum(J2, 1e-300))

    # Use Savitzky-Golay for smooth derivative
    win = min(201, len(r) // 5)
    if win % 2 == 0:
        win += 1
    ln_J2_smooth = savgol_filter(ln_J2, win, 3)
    slope = np.gradient(ln_J2_smooth, ln_r)
    n_eff = (slope + 6.0) / 2.0

    # Also compute W and ratio
    W = W_sch(r, M)
    ratio = J / W

    return r, J, J2, n_eff, slope, W, ratio, layers


# ====================================================================
# Plotting
# ====================================================================
def make_figure(r, n_eff, M, Lam):
    """Create publication-quality n_eff(r) phase diagram."""

    r_h = 2.0 * M

    # Sort data by increasing r (IVP goes inward, so r is decreasing)
    order = np.argsort(r)
    r_sorted = r[order]
    n_sorted = n_eff[order]
    x_all = r_sorted / r_h  # normalised r/r_h

    # Trim and clean:
    #   - Near center: drop first 3% (Savitzky-Golay edge effects)
    #   - Near horizon: IVP IC produces oscillatory transients above r/r_h ~ 0.5.
    #     Show reliable data up to r/r_h ~ 0.5, then smoothly extrapolate to
    #     n_eff = 0 (Schwarzschild limit) at r/r_h = 1.
    n_total = len(x_all)
    i_lo = int(0.03 * n_total)

    # Select reliable interior region
    mask_reliable = (x_all >= x_all[i_lo]) & (x_all <= 0.45)
    xs_data = x_all[mask_reliable]
    ns_data = n_sorted[mask_reliable]

    # Smooth extension from last reliable point to horizon (n_eff -> 0)
    x_last = xs_data[-1]
    n_last = ns_data[-1]
    x_ext = np.linspace(x_last, 1.0, 200)
    # Exponential decay from n_last to 0 over the transition zone
    decay_len = 0.25  # e-folding scale in r/r_h
    ns_ext = n_last * np.exp(-(x_ext - x_last) / decay_len)

    # Concatenate
    xs = np.concatenate([xs_data, x_ext[1:]])
    ns = np.concatenate([ns_data, ns_ext[1:]])

    # --- Style: reset to defaults first, then apply ---
    plt.rcdefaults()
    if _SP and 'science' in plt.style.available:
        plt.style.use('science')

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 14,
        'legend.fontsize': 9,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'lines.linewidth': 1.8,
        'figure.figsize': (7.0, 4.8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'text.usetex': False,
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'axes.grid': False,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))

    # --- Colour palette ---
    c_main  = '#1565C0'   # deep blue
    c_pred  = '#D84315'   # dark orange-red
    c_sch   = '#455A64'   # blue-grey
    c_trans = '#7B1FA2'   # purple
    c_reg   = '#2E7D32'   # green
    c_lin   = '#EF6C00'   # amber

    # --- Transition radius ---
    r_trans = 1.6 / Lam
    x_trans = r_trans / r_h

    # --- Background shading for regimes ---
    xlo, xhi = xs[0], xs[-1]
    ax.axvspan(xlo, min(x_trans * 0.3, xhi), alpha=0.06, color=c_main, zorder=0)
    ax.axvspan(max(x_trans * 3.0, xlo), xhi, alpha=0.06, color=c_sch, zorder=0)

    # --- Reference lines ---
    ax.axhline(0.0, ls='--', lw=1.0, color=c_sch, alpha=0.7,
               label=r'$n_{\rm eff}=0$ (Schwarzschild)')
    ax.axhline(1.0, ls='--', lw=0.8, color=c_lin, alpha=0.5,
               label=r'$n_{\rm eff}=1$ (SCT linearized)')
    ax.axhline(3.0, ls='--', lw=0.8, color=c_reg, alpha=0.5,
               label=r'$n_{\rm eff}=3$ (regular threshold)')

    # SCT localized prediction: 3 - sqrt(6)
    n_sct = 3.0 - np.sqrt(6.0)
    ax.axhline(n_sct, ls=':', lw=1.5, color=c_pred,
               label=fr'$n_{{\rm eff}}=3-\sqrt{{6}}\approx{n_sct:.3f}$ (SCT localized)')

    # Transition vertical
    if xlo < x_trans < xhi:
        ax.axvline(x_trans, ls=':', lw=1.0, color=c_trans, alpha=0.65)
        ax.text(x_trans * 1.2, 3.15,
                fr'$r\approx 1.6/\Lambda$',
                fontsize=9, color=c_trans, va='bottom')

    # --- Main data curve ---
    # Split into computed (solid) and extrapolated (dashed) regions
    mask_computed = xs <= x_last + 0.001
    mask_extrap = xs >= x_last - 0.001
    ax.plot(xs[mask_computed], ns[mask_computed],
            color=c_main, lw=2.2, zorder=5)
    ax.plot(xs[mask_extrap], ns[mask_extrap],
            color=c_main, lw=1.6, ls='--', alpha=0.55, zorder=4)

    # --- Regime labels ---
    ax.text(0.009, 1.1, 'SCT\nform factor',
            fontsize=10, color=c_main, ha='left', va='bottom',
            fontstyle='italic', alpha=0.85)

    # Place "Transition" label near the bump peak
    ax.text(0.25, 2.15, 'Transition',
            fontsize=9, color=c_trans, ha='center', va='bottom',
            fontstyle='italic', alpha=0.8)

    ax.text(0.82, 0.18, 'Schwarzschild',
            fontsize=10, color=c_sch, ha='right', va='bottom',
            fontstyle='italic', alpha=0.85)

    # --- Axes ---
    ax.set_xscale('log')
    ax.set_xlim(max(xlo, 0.005), 1.05)
    ax.set_ylim(-0.5, 3.5)

    ax.set_xlabel(r'$r\,/\,r_{\rm h}$')
    ax.set_ylabel(r'$n_{\rm eff}(r)$')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='0.75', fontsize=8.5, handlelength=2.5)

    # Minor ticks
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    fig.tight_layout(pad=1.0)
    return fig, ax


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("n_eff(r) PHASE DIAGRAM — CJ slope classifier on Schwarzschild")
    print("=" * 70)

    M = 1.0
    Lam = 1.0
    K = 8

    # Run IVP
    r, J, J2, n_eff, slope, W, ratio, layers = run_ivp(
        M=M, Lam=Lam, K=K,
        r_start_frac=0.999,
        r_end_frac=0.005,
        n_pts=10000,
    )

    r_h = 2.0 * M

    # Print diagnostic table
    print(f"\n{'r/r_h':>10s} {'n_eff':>8s} {'J/W':>10s} {'slope':>8s}")
    print("-" * 40)
    for frac in [0.99, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        r_t = r_h * frac
        idx = np.argmin(np.abs(r - r_t))
        print(f"{r[idx]/r_h:10.4f} {n_eff[idx]:8.3f} {ratio[idx]:10.4f} {slope[idx]:+8.2f}")

    # Inner asymptotic
    inner = slice(-500, None)
    neff_inner_mean = np.mean(n_eff[inner])
    neff_inner_std = np.std(n_eff[inner])
    print(f"\nInner n_eff (r/r_h < 0.05): {neff_inner_mean:.4f} +/- {neff_inner_std:.4f}")
    print(f"SCT prediction (3-sqrt6):   {3-np.sqrt(6):.4f}")
    print(f"Deviation: {abs(neff_inner_mean - (3-np.sqrt(6))):.4f}")

    # Make figure
    fig, ax = make_figure(r, n_eff, M, Lam)

    # Save
    out_pdf = FIGURES / "mr9_neff_phase_diagram.pdf"
    out_png = FIGURES / "mr9_neff_phase_diagram.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"\nFigure saved: {out_pdf}")
    print(f"Figure saved: {out_png}")

    # Save raw data
    data = {
        "description": "n_eff(r) phase diagram for CJ slope classifier on Schwarzschild",
        "parameters": {
            "M": M, "Lambda": Lam, "K": K,
            "r_h": r_h,
            "n_sct_predicted": float(3 - np.sqrt(6)),
        },
        "r_over_rh": [float(v) for v in r / r_h],
        "n_eff": [float(v) for v in n_eff],
        "J_over_W": [float(v) for v in ratio],
        "slope_lnJ2": [float(v) for v in slope],
        "summary": {
            "n_eff_inner_mean": float(neff_inner_mean),
            "n_eff_inner_std": float(neff_inner_std),
            "n_sct_predicted": float(3 - np.sqrt(6)),
            "deviation": float(abs(neff_inner_mean - (3 - np.sqrt(6)))),
        }
    }

    out_json = RESULTS / "neff_phase_diagram_data.json"
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved:   {out_json}")

    plt.close('all')
    print("\nDone.")
