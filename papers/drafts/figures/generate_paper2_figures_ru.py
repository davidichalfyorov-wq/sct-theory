# ruff: noqa: E402, I001
"""
Публикационные графики для Статьи 2 «Тесты Солнечной системы и лабораторные
ограничения спектрального действия» (русскоязычная версия).

Генерирует 4 PDF-файла в papers/drafts/figures/:
  1. fig_potential_ratio_ru.pdf      -- V(r)/V_N(r) при различных xi
  2. fig_exclusion_unified_ru.pdf    -- Единый график исключения
  3. fig_lambda_min_ru.pdf           -- Lambda_min как функция xi
  4. fig_potential_deviation_ru.pdf   -- Отклонение потенциала при различных Lambda

Запуск: python papers/drafts/figures/generate_paper2_figures_ru.py
"""

from __future__ import annotations

import math
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
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(1, str(ANALYSIS_DIR / "scripts"))

from sct_tools.plotting import init_style, save_figure, SCT_COLORS

# Import computational functions from lt3d_laboratory
from lt3d_laboratory import (
    ALPHA_1,
    ALPHA_2,
    M2_OVER_LAMBDA,
    V_ratio,
    composite_exclusion_data,
    Lambda_min_eotwash,
    lambda_1,
    lambda_2,
)

FIGURES_DIR = SCRIPT_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _apply_ru_font():
    """Apply Times New Roman font for Russian text after init_style."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']


# =====================================================================
# Figure 1: V(r)/V_N(r) — potential ratio at several xi
# =====================================================================
def figure_potential_ratio_ru() -> Path:
    """V(r)/V_N(r) as a function of r*Lambda for several xi values."""
    init_style()
    _apply_ru_font()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    m2 = np.sqrt(60.0 / 13.0)  # ~ 2.148

    xi_vals = [0.0, 0.1, 0.3]
    xi_labels = [r'$\xi = 0$', r'$\xi = 0{,}1$', r'$\xi = 0{,}3$']
    xi_colors = [SCT_COLORS['total'], SCT_COLORS['dirac'], SCT_COLORS['scalar']]

    rL = np.linspace(0, 5, 1000)

    for xi, label, clr in zip(xi_vals, xi_labels, xi_colors):
        xi_shifted = xi - 1.0 / 6.0
        if abs(xi_shifted) < 1e-15:
            ratio = 1.0 - (4.0 / 3.0) * np.exp(-m2 * rL)
        else:
            m0 = 1.0 / np.sqrt(6.0 * xi_shifted**2)
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

    path = FIGURES_DIR / "fig_potential_ratio_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# =====================================================================
# Figure 2: Unified exclusion plot
# =====================================================================
def figure_exclusion_unified_ru() -> Path:
    """Unified alpha-lambda exclusion plot with Russian labels."""
    init_style()
    _apply_ru_font()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    data = composite_exclusion_data()

    # --- Experimental exclusion curves ---
    # Casimir (Chen 2016)
    d = data["casimir_chen2016"]
    ax.plot(d[:, 0], d[:, 1], 'o-', color='#1565C0', ms=3, lw=1.2,
            label='Казимир (Chen+ 2016)')
    ax.fill_between(d[:, 0], d[:, 1], 1e16, alpha=0.08, color='#1565C0')

    # Geraci 2008 (Stanford)
    d = data["geraci_2008"]
    ax.plot(d[:, 0], d[:, 1], 's-', color='#00838F', ms=3, lw=1.2,
            label='Стэнфорд (Geraci+ 2008)')

    # Lee 2020 (Eot-Wash)
    d = data["lee_2020"]
    ax.plot(d[:, 0], d[:, 1], '^-', color='#C62828', ms=3, lw=1.5,
            label='Eot-Wash (Lee+ 2020)')
    ax.fill_between(d[:, 0], d[:, 1], 1e16, alpha=0.08, color='#C62828')

    # Kapner 2007
    d = data["kapner_2007"]
    ax.plot(d[:, 0], d[:, 1], 'v-', color='#E65100', ms=3, lw=1.2,
            label='Eot-Wash (Kapner+ 2007)')

    # Atom interferometry
    d = data["atom_interf"]
    ax.plot(d[:, 0], d[:, 1], 'D-', color='#6A1B9A', ms=3, lw=1.0,
            alpha=0.7, label='Атомн. интерф. (прибл.)')

    # --- SCT prediction: horizontal lines ---
    ax.axhline(y=abs(ALPHA_1), color=SCT_COLORS["prediction"], lw=2.5,
               ls='--', label=r'СКТ $|\alpha_1| = 4/3$ (спин-2)')
    ax.axhline(y=abs(ALPHA_2), color=SCT_COLORS["scalar"], lw=2.0,
               ls=':', label=r'СКТ $|\alpha_2| = 1/3$ (скаляр)')

    # Mark the SCT boundary crossing
    xi = 0.0
    bound = Lambda_min_eotwash(xi)
    lam_cross = bound["spin2_lambda_cross_m"]
    Lambda_min_val = bound["Lambda_min_eV"]
    ax.plot(lam_cross, abs(ALPHA_1), '*', color=SCT_COLORS["prediction"],
            ms=15, zorder=10)
    ax.annotate(
        rf'$\Lambda_{{\min}} = {Lambda_min_val:.2e}$ эВ'
        f'\n$\\lambda_1 = {lam_cross*1e6:.1f}$ мкм',
        xy=(lam_cross, abs(ALPHA_1)),
        xytext=(lam_cross * 5, abs(ALPHA_1) * 30),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color=SCT_COLORS["prediction"]),
        color=SCT_COLORS["prediction"],
    )

    # Mark Lambda values along the SCT horizontal line
    for Lambda_val, label_txt in [
        (1e-2, r'$10^{-2}$'),
        (1e-1, r'$10^{-1}$'),
        (1.0,  r'$1$'),
    ]:
        lam = lambda_1(Lambda_val)
        if 1e-9 < lam < 1:
            ax.plot(lam, abs(ALPHA_1), '|', color=SCT_COLORS["prediction"],
                    ms=10, mew=1.5)
            ax.text(lam, abs(ALPHA_1) * 0.3, label_txt + ' эВ',
                    fontsize=6, ha='center', color=SCT_COLORS["prediction"])

    # Shade excluded region
    ax.text(3e-7, 2e14, 'ИСКЛЮЧЕНО', fontsize=14, color='gray', alpha=0.3,
            ha='center', rotation=0)

    # Axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-9, 1e1)
    ax.set_ylim(1e-5, 1e16)
    ax.set_xlabel(r'Юкавский радиус $\lambda$ (м)')
    ax.set_ylabel(r'Юкавская связь $|\alpha|$')
    ax.set_title('Лабораторные ограничения СКТ: единый график исключения')
    ax.legend(loc='upper right', fontsize=7, ncol=1)
    ax.grid(True, which='major', alpha=0.2)
    ax.grid(True, which='minor', alpha=0.05)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_exclusion_unified_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# =====================================================================
# Figure 3: Lambda_min vs xi
# =====================================================================
def figure_lambda_min_ru() -> Path:
    """Lambda_min as a function of xi, with Russian labels."""
    init_style()
    _apply_ru_font()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    xi_vals = np.linspace(0, 0.5, 50)
    Lambda_mins = []
    for xi in xi_vals:
        bound = Lambda_min_eotwash(xi)
        Lambda_mins.append(bound["Lambda_min_eV"])

    ax.plot(xi_vals, Lambda_mins, color=SCT_COLORS["prediction"], lw=2)
    ax.axvline(x=1 / 6, color='gray', ls='--', lw=0.8, alpha=0.5,
               label=r'$\xi = 1/6$ (конформная)')

    # Mark specific xi values
    for xi_mark in [0.0, 1 / 6, 0.25]:
        bound = Lambda_min_eotwash(xi_mark)
        ax.plot(xi_mark, bound["Lambda_min_eV"], 'o',
                color=SCT_COLORS["prediction"], ms=5, zorder=5)
        ax.annotate(f'{bound["Lambda_min_eV"]:.3e}',
                    xy=(xi_mark, bound["Lambda_min_eV"]),
                    xytext=(xi_mark + 0.03, bound["Lambda_min_eV"] * 1.1),
                    fontsize=6)

    ax.set_xlabel(r'$\xi$ (неминимальная связь Хиггса)')
    ax.set_ylabel(r'$\Lambda_{\min}$ (эВ)')
    ax.set_title(r'Ограничение Eot-Wash на $\Lambda$ от $\xi$')
    ax.legend(fontsize=7)
    ax.set_xlim(-0.02, 0.52)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_lambda_min_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# =====================================================================
# Figure 4: Potential deviation V(r)/V_N(r) for several Lambda
# =====================================================================
def figure_potential_deviation_ru() -> Path:
    """V(r)/V_N(r) vs r/lambda_1 for several Lambda, with Russian labels."""
    init_style()
    _apply_ru_font()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    from scipy import constants as sc
    HBAR_C_EV_M = sc.hbar * sc.c / sc.eV

    Lambda_values = [1e-3, 1e-2, 1e-1, 1.0]
    colors = ['#1565C0', '#C62828', '#2E7D32', '#E65100']

    for Lambda_eV, col in zip(Lambda_values, colors):
        lam1 = lambda_1(Lambda_eV)
        r_min = 0.01 * lam1
        r_max = 20 * lam1
        r = np.logspace(np.log10(r_min), np.log10(r_max), 300)
        vr = V_ratio(r, Lambda_eV, xi=0.0)
        ax.plot(r / lam1, vr, color=col, lw=1.5,
                label=rf'$\Lambda = {Lambda_eV:.0e}$ эВ')

    ax.axhline(y=1, color='gray', ls=':', lw=0.5,
               label=r'$V/V_N = 1$ (Ньютон)')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5, alpha=0.5)

    ax.set_xlabel(r'$r / \lambda_1$')
    ax.set_ylabel(r'$V(r) / V_N(r)$')
    ax.set_title(r'Модифицированный ньютоновский потенциал ($\xi = 0$)')
    ax.set_xscale('log')
    ax.set_xlim(0.01, 20)
    ax.set_ylim(-0.5, 1.2)
    ax.legend(fontsize=6, loc='lower right')

    fig.tight_layout()
    path = FIGURES_DIR / "fig_potential_deviation_ru.pdf"
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Статья 2 (СКТ тесты Солнечной системы) — графики (RU)")
    print("=" * 60)

    print("\nГенерация графиков:")
    paths = []
    paths.append(figure_potential_ratio_ru())
    paths.append(figure_exclusion_unified_ru())
    paths.append(figure_lambda_min_ru())
    paths.append(figure_potential_deviation_ru())

    print(f"\nСгенерировано {len(paths)} графиков в {FIGURES_DIR}/")
    for p in paths:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
