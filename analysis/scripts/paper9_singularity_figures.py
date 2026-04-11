# ruff: noqa: E402, I001
"""
Generate 4 publication-quality figures for the paper:
"Black hole singularity softening in SCT: exact exponent sqrt(6)."

Figure 1: n_eff(r) phase diagram with M*Lambda dependence (multiple curves)
Figure 2: Kretschner scalar ratio K_SCT/K_GR
Figure 3: Pi_TT(z) structure on the real axis
Figure 4: Connection matrix |M12/M11| vs Lambda

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
from scipy.integrate import solve_ivp, quad
from scipy.signal import savgol_filter

# ─── Project paths ──────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent.parent
FIGURES = PROJECT / "analysis" / "figures"
RESULTS = PROJECT / "analysis" / "results" / "gap_g1"
FIGURES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "analysis"))

# ─── Style setup ────────────────────────────────────────────────────
try:
    import scienceplots  # noqa: F401
    _SP = True
except ImportError:
    _SP = False


def apply_style():
    """Apply clean academic style."""
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
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'text.usetex': False,
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'axes.grid': False,
    })


# ====================================================================
# COMMON: P(alpha) kernel and Schwarzschild helpers
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
    return dict(tau=tau, c=c, dtau=dtau, P=P, w=w_q, K=K)


def H_sch(r, M):
    return 1.0 - 2.0 * M / r


def Hp_sch(r, M):
    return 2.0 * M / r**2


def W_sch(r, M):
    return -2.0 * M / r**3


# ====================================================================
# FIGURE 1: n_eff(r) with M*Lambda dependence
# ====================================================================

def stage1_rhs(r, y, M, layers, Lam):
    """ODE RHS for the localized IVP."""
    K = layers['K']
    dtau = layers['dtau']
    H = H_sch(r, M)
    Hp = Hp_sch(r, M)
    W = W_sch(r, M)
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
    """Integrate u_k from near-horizon inward; return r, n_eff."""
    layers = build_alpha_layers(K)
    r_h = 2.0 * M
    r_start = r_h * r_start_frac
    r_end = r_h * r_end_frac

    W0 = W_sch(r_start, M)
    y0 = np.zeros(2 * K)
    for k in range(K):
        fac = 1.0 / (1.0 + 4.0 * layers['tau'][k])
        y0[2 * k] = W0 * fac
        y0[2 * k + 1] = (6.0 * M / r_start**4) * fac

    r_eval = np.linspace(r_start, r_end, n_pts)
    sol = solve_ivp(
        lambda r, y: stage1_rhs(r, y, M, layers, Lam),
        [r_start, r_end], y0,
        method='RK45', t_eval=r_eval,
        rtol=1e-10, atol=1e-12,
        max_step=(r_start - r_end) / 500,
    )
    if sol.status != 0:
        print(f"  WARNING: IVP failed for M*Lam={M*Lam:.2f}: {sol.message}")

    r = sol.t
    c = layers['c']
    J = np.zeros(len(r))
    for k in range(K):
        J += c[k] * sol.y[2 * k]

    # n_eff from slope of ln(J^2)
    ln_r = np.log(r)
    J2 = J**2
    ln_J2 = np.log(np.maximum(J2, 1e-300))

    win = min(51, len(r) // 5)
    if win % 2 == 0:
        win += 1
    if win < 5:
        win = 5
    ln_J2_smooth = savgol_filter(ln_J2, win, 3)
    slope = np.gradient(ln_J2_smooth, ln_r)
    n_eff = (slope + 6.0) / 2.0

    W = W_sch(r, M)
    return r, J, n_eff, W


def figure1():
    """n_eff(r/r_h) for multiple M*Lambda values."""
    print("=" * 60)
    print("FIGURE 1: n_eff(r) phase diagram with M*Lambda dependence")
    print("=" * 60)

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    # M*Lambda values and visual styles
    # We fix M=1 and vary Lambda to get different M*Lambda products
    M = 1.0
    configs = [
        (0.5, '#1565C0', '-',   1.8),   # M*Lam=0.5
        (1.0, '#D84315', '-',   2.2),   # M*Lam=1.0
        (2.0, '#2E7D32', '--',  1.8),   # M*Lam=2.0
        (5.0, '#7B1FA2', '-.',  1.8),   # M*Lam=5.0
    ]

    for Lam, color, ls, lw in configs:
        MLam = M * Lam
        print(f"\n  Running M*Lambda = {MLam:.1f}  (M={M}, Lambda={Lam})...")
        r, J, n_eff, W = run_ivp(M=M, Lam=Lam, K=8, n_pts=10000)
        r_h = 2.0 * M

        # Sort by increasing r
        order = np.argsort(r)
        r_s = r[order]
        n_s = n_eff[order]
        x = r_s / r_h

        # Trim edge effects (3% on each end)
        n_total = len(x)
        i_lo = int(0.03 * n_total)
        i_hi = int(0.90 * n_total)  # stop well before horizon IC artifacts
        x_plot = x[i_lo:i_hi]
        n_plot = n_s[i_lo:i_hi]

        # Extra smoothing for visual clarity
        if len(n_plot) > 51:
            n_plot = savgol_filter(n_plot, 51, 3)

        ax.plot(x_plot, n_plot, color=color, ls=ls, lw=lw,
                label=fr'$M\Lambda = {MLam:.1f}$', zorder=5)

        # Report inner value
        inner_mask = x_plot < 0.03
        if np.any(inner_mask):
            n_inner = np.mean(n_plot[inner_mask])
            print(f"    Inner n_eff ~ {n_inner:.3f}")

    # Reference lines
    n_sct = 3.0 - np.sqrt(6.0)
    ax.axhline(n_sct, ls=':', lw=1.5, color='#D84315', alpha=0.6,
               label=fr'$3-\sqrt{{6}}\approx{n_sct:.3f}$')
    ax.axhline(0.0, ls='--', lw=1.0, color='#455A64', alpha=0.5,
               label=r'$n_{\rm eff}=0$ (Schwarzschild)')
    ax.axhline(3.0, ls='--', lw=0.8, color='#2E7D32', alpha=0.4,
               label=r'$n_{\rm eff}=3$ (regular)')

    ax.set_xscale('log')
    ax.set_xlim(0.005, 0.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel(r'$r\,/\,r_{\rm h}$')
    ax.set_ylabel(r'$n_{\rm eff}(r)$')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='0.75', fontsize=8.5, handlelength=2.5, ncol=1)

    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    fig.tight_layout(pad=1.0)

    out_pdf = FIGURES / "paper_fig1_neff_MLam.pdf"
    out_png = FIGURES / "paper_fig1_neff_MLam.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"\n  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ====================================================================
# FIGURE 2: Kretschner scalar comparison K_SCT/K_GR
# ====================================================================

def figure2():
    """Kretschner scalar ratio K_SCT/K_GR vs r/r_h."""
    print("\n" + "=" * 60)
    print("FIGURE 2: Kretschner scalar ratio K_SCT/K_GR")
    print("=" * 60)

    M = 1.0
    Lam = 1.0
    K = 8

    r, J, n_eff, W = run_ivp(M=M, Lam=Lam, K=K, n_pts=10000)
    r_h = 2.0 * M

    # Sort by increasing r
    order = np.argsort(r)
    r_s = r[order]
    J_s = J[order]
    W_s = W[order]

    # Kretschner: K_GR = 48 M^2 / r^6
    # K_SCT ~ 48 M^2 * (J/W)^2 / r^6  (from form-factor dressing)
    # Ratio = (J/W)^2
    ratio = (J_s / W_s)**2
    x = r_s / r_h

    # Trim edges
    n_total = len(x)
    i_lo = int(0.03 * n_total)
    i_hi = int(0.90 * n_total)
    x_plot = x[i_lo:i_hi]
    ratio_plot = ratio[i_lo:i_hi]

    # Smooth
    if len(ratio_plot) > 51:
        ln_ratio_smooth = savgol_filter(np.log(np.maximum(ratio_plot, 1e-300)), 51, 3)
        ratio_plot = np.exp(ln_ratio_smooth)

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    ax.plot(x_plot, ratio_plot, color='#1565C0', lw=2.2, label='SCT form-factor dressed')
    ax.axhline(1.0, ls='--', lw=1.2, color='#455A64', alpha=0.6,
               label=r'GR Schwarzschild ($K_{\rm SCT}/K_{\rm GR}=1$)')

    # Annotate the power-law behavior
    # At small r: ratio ~ r^{6-2*sqrt(6)} ~ r^{1.101}
    r_ref = np.logspace(np.log10(x_plot[0]), np.log10(0.05), 100)
    power_law = (r_ref / 0.05)**(6.0 - 2.0 * np.sqrt(6.0))
    # Normalize to match data at x=0.05
    idx_05 = np.argmin(np.abs(x_plot - 0.05))
    if idx_05 > 0:
        norm_val = ratio_plot[idx_05]
        power_law *= norm_val
        ax.plot(r_ref, power_law, ls=':', lw=1.5, color='#D84315', alpha=0.7,
                label=fr'$\propto r^{{6-2\sqrt{{6}}}}\approx r^{{{6-2*np.sqrt(6):.3f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.005, 0.5)
    ax.set_xlabel(r'$r\,/\,r_{\rm h}$')
    ax.set_ylabel(r'$K_{\rm SCT}\,/\,K_{\rm GR}$')

    ax.legend(loc='lower right', frameon=True, framealpha=0.9,
              edgecolor='0.75', fontsize=9, handlelength=2.5)

    fig.tight_layout(pad=1.0)

    out_pdf = FIGURES / "paper_fig2_kretschner.pdf"
    out_png = FIGURES / "paper_fig2_kretschner.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ====================================================================
# FIGURE 3: Pi_TT(z) structure
# ====================================================================

def h_C_total_real(z):
    """Total SM h_C(z) = alpha_C(z) for REAL z (positive or negative).

    h_C(z) = N_s * hC_scalar(z) + N_D * hC_dirac(z) + N_v * hC_vector(z)

    Uses integral definition: h_C_spin(z) involves integral_0^1 P_spin(alpha) * exp(-z*tau) dalpha
    where tau = alpha(1-alpha).

    For the combined Weyl coefficient alpha_C(z):
    alpha_C(z) = integral_0^1 P_total(alpha) * exp(-z*alpha*(1-alpha)) dalpha
    with P_total = -89/24 + (43/6)*tau + (236/3)*tau^2.

    Wait -- alpha_C(z) = sum_spins N_s * hC^(s)(z).
    The P(alpha) kernel above is alpha_C = integral P(alpha) exp(-z*tau) dalpha.
    This gives alpha_C(0) = integral P dalpha = 13/120.

    Then Pi_TT(z) = 1 + (13/60) * z * F_hat_1(z) where F_hat_1 = alpha_C(z)/alpha_C(0).
    But it's simpler to compute:
    Pi_TT(z) = 1 + 2 * z * (alpha_C(z) / (16*pi^2)) * (16*pi^2) * (1/?) ...

    Actually, Pi_TT(z) = 1 + (13/60) z F_hat_1(z) from NT-4a.
    F_hat_1(z) = alpha_C(z) / alpha_C(0) = alpha_C(z) / (13/120).
    So Pi_TT(z) = 1 + (13/60) * z * [120/13 * alpha_C(z)]
                = 1 + 2 z alpha_C(z).

    Let's compute alpha_C(z) = integral_0^1 P(alpha) exp(-z alpha(1-alpha)) dalpha.
    """
    # For real z (both positive and negative), the integral is well-defined
    def integrand(alpha):
        tau = alpha * (1.0 - alpha)
        P = -89.0 / 24.0 + (43.0 / 6.0) * tau + (236.0 / 3.0) * tau**2
        return P * np.exp(-z * tau)

    result, _ = quad(integrand, 0, 1, limit=200)
    return result


def figure3():
    """Pi_TT(z) = 1 + 2*z*alpha_C(z) on the real axis."""
    print("\n" + "=" * 60)
    print("FIGURE 3: Pi_TT(z) structure")
    print("=" * 60)

    # Compute Pi_TT on a dense grid
    z_neg = np.linspace(-3.0, -0.01, 200)
    z_pos = np.linspace(0.01, 10.0, 400)
    z_all = np.concatenate([z_neg, [0.0], z_pos])
    z_all = np.sort(z_all)

    Pi_TT = np.zeros(len(z_all))
    for i, z in enumerate(z_all):
        aC = h_C_total_real(z)
        Pi_TT[i] = 1.0 + 2.0 * z * aC

    print(f"  Pi_TT(0) = {1.0 + 2.0 * 0.0 * h_C_total_real(0.0):.6f} (should be 1)")
    print(f"  alpha_C(0) = {h_C_total_real(0.0):.8f}  (13/120 = {13/120:.8f})")

    # Find zero crossing
    for i in range(len(z_all) - 1):
        if z_all[i] > 0 and Pi_TT[i] * Pi_TT[i + 1] < 0:
            # Linear interpolation
            z_zero = z_all[i] - Pi_TT[i] * (z_all[i + 1] - z_all[i]) / (Pi_TT[i + 1] - Pi_TT[i])
            print(f"  Pi_TT zero crossing at z = {z_zero:.4f}")
            break

    # UV asymptote: Pi_TT -> 1 + 2 * z * alpha_C(z -> inf)
    # alpha_C(z -> inf) ~ (P integral with tau=0) / z ... actually
    # For large z: alpha_C(z) -> P(0)/sqrt(pi*z/4) * exp(-z/4) * 2/sqrt(z)
    # Let's just compute numerically at z=10
    print(f"  Pi_TT(10) = {Pi_TT[-1]:.4f}")

    # Theoretical UV limit: as z -> +inf, z*alpha_C(z) -> -89/12 + ...
    # So Pi_TT -> 1 + 2*(-89/12) = 1 - 89/6 = -13.833...
    uv_limit = 1.0 - 89.0 / 6.0
    print(f"  UV asymptote Pi_TT -> 1 - 89/6 = {uv_limit:.4f}")

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    ax.plot(z_all, Pi_TT, color='#1565C0', lw=2.2, zorder=5)

    # Zero line
    ax.axhline(0.0, ls='-', lw=0.8, color='#455A64', alpha=0.4)

    # Mark Pi_TT(0) = 1
    ax.plot(0, 1, 'o', color='#1565C0', ms=6, zorder=6)
    ax.annotate(r'$\Pi_{\rm TT}(0)=1$', xy=(0, 1), xytext=(1.2, 1.5),
                fontsize=10, color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.0))

    # Mark zero crossing
    if 'z_zero' in dir():
        pass
    # Find zero from data
    z_zero_val = None
    for i in range(len(z_all) - 1):
        if z_all[i] > 0 and Pi_TT[i] * Pi_TT[i + 1] < 0:
            z_zero_val = z_all[i] - Pi_TT[i] * (z_all[i + 1] - z_all[i]) / (Pi_TT[i + 1] - Pi_TT[i])
            break

    if z_zero_val is not None:
        ax.axvline(z_zero_val, ls=':', lw=1.2, color='#D84315', alpha=0.6)
        ax.annotate(fr'$z_2 \approx {z_zero_val:.2f}$',
                    xy=(z_zero_val, 0), xytext=(z_zero_val + 1.5, 2.0),
                    fontsize=10, color='#D84315',
                    arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.0))

    # UV asymptote annotation
    ax.axhline(uv_limit, ls='--', lw=1.0, color='#7B1FA2', alpha=0.5)
    ax.text(8.5, uv_limit + 0.8, fr'UV: $1-89/6\approx{uv_limit:.1f}$',
            fontsize=9, color='#7B1FA2', ha='right')

    # Local approximation zero
    z_local = -60.0 / 13.0
    ax.text(-2.8, -3.0,
            fr'Local zero at $z=-60/13\approx{z_local:.2f}$' + '\n(outside plot range)',
            fontsize=8, color='#455A64', ha='left', style='italic')

    ax.set_xlim(-3, 10)
    ax.set_ylim(-16, 4)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\Pi_{\rm TT}(z)$')

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))

    fig.tight_layout(pad=1.0)

    out_pdf = FIGURES / "paper_fig3_pi_tt.pdf"
    out_png = FIGURES / "paper_fig3_pi_tt.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ====================================================================
# FIGURE 4: Connection matrix |M12/M11| vs Lambda
# ====================================================================

def figure4():
    """Connection matrix ratio |M12/M11| vs Lambda from stored data."""
    print("\n" + "=" * 60)
    print("FIGURE 4: Connection matrix |M12/M11| vs Lambda")
    print("=" * 60)

    # Load existing data
    json_path = RESULTS / "connection_matrix.json"
    if not json_path.exists():
        print(f"  ERROR: {json_path} not found, computing from scratch...")
        figure4_compute()
        return

    with open(json_path) as f:
        data = json.load(f)

    scan = data['Lambda_scan']
    lambdas = []
    ratios = []

    for key, val in scan.items():
        lam = val['Lambda']
        r = abs(val['ratio_M12_M11'])
        lambdas.append(lam)
        ratios.append(r)
        print(f"  Lambda={lam:.1f}: |M12/M11| = {r:.6f}")

    lambdas = np.array(lambdas)
    ratios = np.array(ratios)

    # Sort
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    ratios = ratios[order]

    # Power-law fit: |M12/M11| ~ a * Lambda^b
    # Use log-log linear fit
    mask = lambdas > 0.3  # skip very small Lambda where numerics differ
    log_lam = np.log(lambdas[mask])
    log_rat = np.log(ratios[mask])
    coeffs = np.polyfit(log_lam, log_rat, 1)
    b_fit = coeffs[0]
    a_fit = np.exp(coeffs[1])
    print(f"\n  Power-law fit: |M12/M11| ~ {a_fit:.4f} * Lambda^({b_fit:.4f})")

    # Compare with stored fit
    if 'power_law_fit' in data:
        print(f"  Stored fit: a={data['power_law_fit']['a']:.4f}, b={data['power_law_fit']['b']:.4f}")

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    ax.plot(lambdas, ratios, 'o', color='#1565C0', ms=8, zorder=6,
            label='Numerical data')

    # Fit line
    lam_fit = np.logspace(np.log10(lambdas[0] * 0.8), np.log10(lambdas[-1] * 1.2), 100)
    ax.plot(lam_fit, a_fit * lam_fit**b_fit, '--', color='#D84315', lw=1.5,
            label=fr'Fit: $|M_{{12}}/M_{{11}}| \propto \Lambda^{{{b_fit:.2f}}}$')

    # Mark ratio = 0.5 level (approximate GR mixing)
    ax.axhline(0.5, ls=':', lw=1.0, color='#455A64', alpha=0.5)
    ax.text(0.12, 0.55, r'$|M_{12}/M_{11}|=0.5$', fontsize=8, color='#455A64')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Lambda$')
    ax.set_ylabel(r'$|M_{12}/M_{11}|$')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='0.75', fontsize=9)

    fig.tight_layout(pad=1.0)

    out_pdf = FIGURES / "paper_fig4_connection.pdf"
    out_png = FIGURES / "paper_fig4_connection.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")
    plt.close(fig)


def figure4_compute():
    """Compute connection matrix from scratch if JSON not available."""
    M = 1.0
    m2sq_coeff = 1.2807  # m2^2 = 1.2807 * Lambda^2

    lambdas_list = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    ratios_list = []

    for Lam in lambdas_list:
        m2sq = m2sq_coeff * Lam**2

        def massive_rhs(r, y):
            """H u'' + (H'+2H/r)u' + (m2^2 - 6H/r^2)u = 0"""
            u, v = y
            H = H_sch(r, M)
            Hp = Hp_sch(r, M)
            A = Hp + 2.0 * H / r
            S = 6.0 * H / r**2
            if abs(H) < 1e-14:
                return [v, 0.0]
            upp = (-A * v + (S - m2sq) * u) / H
            return [v, upp]

        eps = 1e-4
        r_h = 2.0 * M

        # Regular solution: u ~ r^{sqrt(6)} near r=0
        rho = np.sqrt(6.0)
        u1_0 = eps**rho
        v1_0 = rho * eps**(rho - 1.0)

        # Irregular solution: u ~ r^{-sqrt(6)} near r=0
        u2_0 = eps**(-rho)
        v2_0 = -rho * eps**(-rho - 1.0)

        r_end = r_h - 1e-4

        sol1 = solve_ivp(massive_rhs, [eps, r_end], [u1_0, v1_0],
                         method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
        sol2 = solve_ivp(massive_rhs, [eps, r_end], [u2_0, v2_0],
                         method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)

        if sol1.success and sol2.success:
            u1_end = sol1.y[0, -1]
            u2_end = sol2.y[0, -1]
            ratio = abs(u2_end / u1_end) if abs(u1_end) > 1e-30 else float('inf')
            ratios_list.append(ratio)
            print(f"  Lambda={Lam:.1f}: |M12/M11| ~ {ratio:.6f}")
        else:
            ratios_list.append(float('nan'))
            print(f"  Lambda={Lam:.1f}: FAILED")

    lambdas_arr = np.array(lambdas_list)
    ratios_arr = np.array(ratios_list)

    valid = np.isfinite(ratios_arr)
    if np.sum(valid) < 2:
        print("  ERROR: Not enough valid data points.")
        return

    log_lam = np.log(lambdas_arr[valid])
    log_rat = np.log(ratios_arr[valid])
    coeffs = np.polyfit(log_lam, log_rat, 1)
    b_fit = coeffs[0]
    a_fit = np.exp(coeffs[1])

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    ax.plot(lambdas_arr[valid], ratios_arr[valid], 'o', color='#1565C0', ms=8, zorder=6,
            label='Numerical data')

    lam_fit = np.logspace(np.log10(0.2), np.log10(12), 100)
    ax.plot(lam_fit, a_fit * lam_fit**b_fit, '--', color='#D84315', lw=1.5,
            label=fr'Fit: $\propto \Lambda^{{{b_fit:.2f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Lambda$')
    ax.set_ylabel(r'$|M_{12}/M_{11}|$')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='0.75')

    fig.tight_layout(pad=1.0)

    out_pdf = FIGURES / "paper_fig4_connection.pdf"
    out_png = FIGURES / "paper_fig4_connection.png"
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")
    plt.close(fig)


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PAPER FIGURES: Black hole singularity softening in SCT")
    print("=" * 70)

    figure1()
    figure2()
    figure3()
    figure4()

    print("\n" + "=" * 70)
    print("ALL 4 FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
