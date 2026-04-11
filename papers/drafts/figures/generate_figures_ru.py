# ruff: noqa: E402, I001
"""
Публикационные графики для «Нелокальные однопетлевые формфакторы спектрального
действия с содержимым Стандартной модели» (русскоязычная версия).

Генерирует 5 PDF-файлов в papers/drafts/figures/:
  1. fig_phi_ru.pdf          -- Мастер-функция phi(x)
  2. fig_hC_ru.pdf           -- Формфакторы Вейля h_C^{(s)}(x), s = 0, 1/2, 1
  3. fig_hR_ru.pdf           -- Формфакторы R^2 h_R^{(s)}(x)
  4. fig_alpha_total_ru.pdf   -- Полные SM-суммы alpha_C(x) и alpha_R(x, xi)
  5. fig_potential_ru.pdf     -- Модифицированный ньютонов потенциал V(r)/V_N(r)

Запуск: python papers/drafts/figures/generate_figures_ru.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Font configuration for Russian labels ────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# ── Path setup ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
# Force absolute paths at position 0 to override any CWD issues
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(1, str(ANALYSIS_DIR / "scripts"))

from sct_tools.plotting import init_style, SCT_COLORS
from sct_tools import form_factors as ff

FIGURES_DIR = Path(__file__).resolve().parent
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── SM constants ─────────────────────────────────────────────────────
N_s = 4
N_f = 45
N_D = N_f / 2  # 22.5 Dirac
N_v = 12


# ── Accuracy verification ───────────────────────────────────────────
def verify_key_values():
    """Cross-check critical values before plotting."""
    from mpmath import mpf, mp
    old_dps = mp.dps
    mp.dps = 50

    checks = []

    # phi(0) = 1
    checks.append(("phi(0)", abs(ff.phi(0) - 1.0) < 1e-14))

    # Local limits
    checks.append(("hC_scalar(0) = 1/120",
                    abs(ff.hC_scalar(0) - 1/120) < 1e-14))
    checks.append(("hC_dirac(0) = -1/20",
                    abs(ff.hC_dirac(0) - (-1/20)) < 1e-14))
    checks.append(("hC_vector(0) = 1/10",
                    abs(ff.hC_vector(0) - 1/10) < 1e-12))
    checks.append(("hR_scalar(0, xi=0) = 1/72",
                    abs(ff.hR_scalar(0, xi=0) - 1/72) < 1e-14))
    checks.append(("hR_dirac(0) = 0",
                    abs(ff.hR_dirac(0)) < 1e-14))
    checks.append(("hR_vector(0) = 0",
                    abs(ff.hR_vector(0)) < 1e-12))

    # SM totals
    alpha_C = N_s * ff.hC_scalar(0) + N_D * ff.hC_dirac(0) + N_v * ff.hC_vector(0)
    checks.append(("alpha_C(0) = 13/120",
                    abs(alpha_C - 13/120) < 1e-12))

    # UV asymptotics
    x_uv = 5000.0
    checks.append(("x*hC_scalar ~ 1/12",
                    abs(x_uv * ff.hC_scalar(x_uv) - 1/12) / (1/12) < 0.01))
    checks.append(("x*hC_dirac ~ -1/6",
                    abs(x_uv * ff.hC_dirac(x_uv) - (-1/6)) / (1/6) < 0.01))
    checks.append(("x*hC_vector ~ -1/3",
                    abs(x_uv * ff.hC_vector(x_uv) - (-1/3)) / (1/3) < 0.01))

    # mpmath cross-check at x=1
    mp_val = float(ff.hC_scalar_mp(1.0, dps=50))
    np_val = ff.hC_scalar(1.0)
    checks.append(("hC_scalar(1) mp vs np",
                    abs(mp_val - np_val) / abs(mp_val) < 1e-12))

    mp.dps = old_dps

    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    if not all_pass:
        raise RuntimeError("Accuracy verification FAILED — aborting figure generation")
    print(f"  All {len(checks)} checks passed.\n")


# ── Figure 1: Master function phi(x) ────────────────────────────────
def figure_phi():
    """phi(x) = int_0^1 exp(-alpha(1-alpha)*x) dalpha."""
    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    x = np.linspace(0, 30, 500)
    y = ff.phi_vec(x)

    ax.plot(x, y, color=SCT_COLORS['total'], lw=1.8)

    # Mark key points
    ax.plot(0, 1, 'o', color=SCT_COLORS['total'], ms=4, zorder=5)
    ax.annotate(r'$\varphi(0) = 1$', xy=(0, 1), xytext=(3, 0.92),
                fontsize=8, ha='left')

    # Asymptotic 2/x
    x_asymp = np.linspace(2, 30, 200)
    ax.plot(x_asymp, 2.0 / x_asymp, '--', color=SCT_COLORS['reference'],
            lw=1.0, label=r'$2/x$')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\varphi(x)$')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    path = FIGURES_DIR / "fig_phi_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Figure 2: Weyl form factors h_C^{(s)}(x) ───────────────────────
def figure_hC():
    """h_C for spin 0, 1/2, 1 on one panel."""
    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    x = np.linspace(0.001, 50, 2000)
    y_s = ff.scan_hC_scalar(x)
    y_d = ff.scan_hC_dirac(x)
    y_v = ff.scan_hC_vector(x)

    ax.plot(x, y_s, color=SCT_COLORS['scalar'], lw=1.8,
            label=r'$h_C^{(0)}$  (скаляр)')
    ax.plot(x, y_d, color=SCT_COLORS['dirac'], lw=1.8,
            label=r'$h_C^{(1/2)}$  (Дирак)')
    ax.plot(x, y_v, color=SCT_COLORS['vector'], lw=1.8,
            label=r'$h_C^{(1)}$  (вектор)')

    ax.axhline(0, color='k', lw=0.5, ls='-')

    # Mark local limits
    ax.plot(0, 1/120, 'o', color=SCT_COLORS['scalar'], ms=3.5, zorder=5)
    ax.plot(0, -1/20, 'o', color=SCT_COLORS['dirac'], ms=3.5, zorder=5)
    ax.plot(0, 1/10, 'o', color=SCT_COLORS['vector'], ms=3.5, zorder=5)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$h_C^{(s)}(x)$')
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.08, 0.12)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7)
    ax.grid(True, alpha=0.3)

    path = FIGURES_DIR / "fig_hC_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Figure 3: R^2 form factors h_R^{(s)}(x) ────────────────────────
def figure_hR():
    """h_R for spin 0 (multiple xi), 1/2, 1."""
    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    x = np.linspace(0.001, 50, 2000)
    y_d = ff.scan_hR_dirac(x)
    y_v = ff.scan_hR_vector(x)

    # Scalar at three xi values
    xi_vals = [0.0, 1/6, 0.5]
    xi_labels = [r'$\xi = 0$', r'$\xi = 1/6$', r'$\xi = 1/2$']
    xi_styles = ['-', '--', ':']

    for xi, label, ls in zip(xi_vals, xi_labels, xi_styles):
        y_s = ff.scan_hR_scalar(x, xi=xi)
        ax.plot(x, y_s, color=SCT_COLORS['scalar'], lw=1.5, ls=ls,
                label=r'$h_R^{(0)}$, ' + label)

    ax.plot(x, y_d, color=SCT_COLORS['dirac'], lw=1.8,
            label=r'$h_R^{(1/2)}$')
    ax.plot(x, y_v, color=SCT_COLORS['vector'], lw=1.8,
            label=r'$h_R^{(1)}$')

    ax.axhline(0, color='k', lw=0.5, ls='-')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$h_R^{(s)}(x)$')
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.02, 0.025)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=6.5,
              ncol=1)
    ax.grid(True, alpha=0.3)

    path = FIGURES_DIR / "fig_hR_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Figure 4: SM totals alpha_C(x) and alpha_R(x, xi) ──────────────
def figure_alpha_total():
    """Two-panel figure: alpha_C(x) and alpha_R(x, xi)."""
    init_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    x = np.linspace(0.001, 80, 3000)

    # ── Left panel: alpha_C(x) ──
    y_s = ff.scan_hC_scalar(x)
    y_d = ff.scan_hC_dirac(x)
    y_v = ff.scan_hC_vector(x)
    alpha_C = N_s * y_s + N_D * y_d + N_v * y_v

    ax1.plot(x, alpha_C, color=SCT_COLORS['total'], lw=1.8,
             label=r'$\alpha_C(x)$')

    # Asymptotic -89/(12x)
    x_asymp = np.linspace(5, 80, 200)
    ax1.plot(x_asymp, -89.0 / (12.0 * x_asymp), '--',
             color=SCT_COLORS['reference'], lw=1.0,
             label=r'$-89/(12x)$')

    ax1.axhline(0, color='k', lw=0.5, ls='-')

    # Mark alpha_C(0) = 13/120
    ax1.plot(0, 13/120, 'o', color=SCT_COLORS['total'], ms=4, zorder=5)
    ax1.annotate(r'$13/120$', xy=(0, 13/120), xytext=(5, 0.13),
                 fontsize=7.5, ha='left')

    # Find and mark zero crossing
    for i in range(len(alpha_C) - 1):
        if alpha_C[i] > 0 and alpha_C[i + 1] < 0:
            x0 = x[i] - alpha_C[i] * (x[i + 1] - x[i]) / (alpha_C[i + 1] - alpha_C[i])
            ax1.plot(x0, 0, 'x', color=SCT_COLORS['prediction'], ms=6, mew=1.5,
                     zorder=5)
            ax1.annotate(f'$x_0 \\approx {x0:.1f}$', xy=(x0, 0),
                         xytext=(x0 + 3, 0.03), fontsize=7.5, ha='left')
            break

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\alpha_C(x)$')
    ax1.set_xlim(0, 80)
    ax1.set_ylim(-0.12, 0.16)
    ax1.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(r'(a) Коэффициент Вейля', fontsize=9)

    # ── Right panel: alpha_R(x, xi) ──
    xi_vals = [0.0, 0.1, 1/6, 0.3]
    xi_labels = [r'$\xi = 0$', r'$\xi = 0.1$', r'$\xi = 1/6$', r'$\xi = 0.3$']
    xi_colors = [SCT_COLORS['total'], SCT_COLORS['dirac'],
                 SCT_COLORS['vector'], SCT_COLORS['scalar']]

    for xi, label, clr in zip(xi_vals, xi_labels, xi_colors):
        y_sr = ff.scan_hR_scalar(x, xi=xi)
        y_dr = ff.scan_hR_dirac(x)
        y_vr = ff.scan_hR_vector(x)
        alpha_R = N_s * y_sr + N_D * y_dr + N_v * y_vr
        lw = 1.8 if xi == 1/6 else 1.3
        ls = '--' if xi == 1/6 else '-'
        ax2.plot(x, alpha_R, color=clr, lw=lw, ls=ls, label=label)

    ax2.axhline(0, color='k', lw=0.5, ls='-')

    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\alpha_R(x, \xi)$')
    ax2.set_xlim(0, 80)
    ax2.set_ylim(-0.01, 0.07)
    ax2.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r'(b) Коэффициент при $R^2$', fontsize=9)

    fig.tight_layout(w_pad=2.5)

    path = FIGURES_DIR / "fig_alpha_total_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Figure 5: Modified Newtonian potential ───────────────────────────
def figure_potential():
    """V(r)/V_N(r) = 1 - (4/3)exp(-m2*r) + (1/3)exp(-m0*r)."""
    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    # Effective masses (in units of Lambda)
    m2 = np.sqrt(60.0 / 13.0)  # ~ 2.148

    xi_vals = [0.0, 0.1, 0.3]
    xi_labels = [r'$\xi = 0$', r'$\xi = 0.1$', r'$\xi = 0.3$']
    xi_colors = [SCT_COLORS['total'], SCT_COLORS['dirac'], SCT_COLORS['scalar']]

    rL = np.linspace(0, 5, 1000)  # r in units of 1/Lambda

    for xi, label, clr in zip(xi_vals, xi_labels, xi_colors):
        xi_shifted = xi - 1.0 / 6.0
        if abs(xi_shifted) < 1e-15:
            # Scalar decoupled: V/V_N = 1 - (4/3)exp(-m2*r)
            ratio = 1.0 - (4.0 / 3.0) * np.exp(-m2 * rL)
        else:
            m0 = 1.0 / np.sqrt(6.0 * xi_shifted**2)  # in units of Lambda
            ratio = (1.0 - (4.0 / 3.0) * np.exp(-m2 * rL)
                     + (1.0 / 3.0) * np.exp(-m0 * rL))
        ax.plot(rL, ratio, color=clr, lw=1.8, label=label)

    # Conformal coupling: scalar decouples
    ratio_conf = 1.0 - (4.0 / 3.0) * np.exp(-m2 * rL)
    ax.plot(rL, ratio_conf, '--', color=SCT_COLORS['vector'], lw=1.5,
            label=r'$\xi = 1/6$')

    ax.axhline(1, color='k', lw=0.5, ls='-')
    ax.axhline(0, color='k', lw=0.5, ls=':')

    ax.set_xlabel(r'$r \cdot \Lambda$')
    ax.set_ylabel(r'$V(r) / V_{\mathrm{N}}(r)$')
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.4, 1.1)
    ax.legend(loc='lower right', frameon=True, fancybox=False, fontsize=7)
    ax.grid(True, alpha=0.3)

    path = FIGURES_DIR / "fig_potential_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Формфакторы SCT — Публикационные графики (RU)")
    print("=" * 60)

    print("\nПроверка точности:")
    verify_key_values()

    print("Генерация графиков:")
    paths = []
    paths.append(figure_phi())
    paths.append(figure_hC())
    paths.append(figure_hR())
    paths.append(figure_alpha_total())
    paths.append(figure_potential())

    print(f"\nСгенерировано {len(paths)} графиков в {FIGURES_DIR}/")
    for p in paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
