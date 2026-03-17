"""Generate illustrative figures for the SCT Theory README."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfi

OUT = Path(__file__).parent

# ---------- style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 180,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

SCT_BLUE = '#1a5276'
SCT_ORANGE = '#e67e22'
SCT_GREEN = '#27ae60'
SCT_RED = '#c0392b'
SCT_PURPLE = '#8e44ad'
SCT_GRAY = '#7f8c8d'

MASTER_FUNCTION_XLABEL = r'$x = -k^2 / \Lambda^2$'
MASTER_FUNCTION_LEFT_LABEL = (
    'Left branch\n'
    + r'($x < 0$, Euclidean continuation)'
)
MASTER_FUNCTION_RIGHT_LABEL = (
    'Right branch\n'
    + r'($x > 0$, Lorentzian $k^2 < 0$)'
)
MASTER_FUNCTION_NEGATIVE_ASYMPTOTIC = (
    'Negative-$x$ branch:\n'
    + r'$x \to -\infty,\ \varphi(x) \sim e^{|x|/4}\sqrt{\pi/|x|}$'
)
MASTER_FUNCTION_POSITIVE_ASYMPTOTIC = (
    'Positive-$x$ branch:\n'
    + r'$x \to +\infty,\ \varphi(x) \sim 2/x$'
)


# ===== Helper: master function =====
def phi_real(x):
    """SCT master function for real x (positive and negative).

    phi(x) = e^{-x/4} sqrt(pi/x) erfi(sqrt(x)/2)  for x > 0
    phi(0) = 1
    phi(-y) = e^{y/4} sqrt(pi/y) erf(sqrt(y)/2)    for y > 0
    """
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)

    # Positive x: use erfi
    mask_pos = x > 1e-10
    xp = x[mask_pos]
    result[mask_pos] = np.exp(-xp / 4) * np.sqrt(np.pi / xp) * erfi(np.sqrt(xp) / 2)

    # Negative x: phi(-y) = e^{y/4} sqrt(pi/y) erf(sqrt(y)/2)
    mask_neg = x < -1e-10
    y = -x[mask_neg]
    result[mask_neg] = np.exp(y / 4) * np.sqrt(np.pi / y) * erf(np.sqrt(y) / 2)

    return result


def build_master_function_figure():
    """Build the README figure for the SCT master function."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    x = np.linspace(-8, 20, 600)
    y = phi_real(x)

    ax.plot(x, y, color=SCT_BLUE, linewidth=2.2, label=r'$\varphi(x)$')
    ax.axhline(1, color=SCT_GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=SCT_GRAY, linestyle=':', linewidth=0.8, alpha=0.3)
    ax.set_xlabel(MASTER_FUNCTION_XLABEL)
    ax.set_ylabel(r'$\varphi(x)$')
    ax.set_title('SCT Master Function on the Real Axis')
    ax.set_ylim(-0.3, 8)
    ax.legend(loc='upper right', framealpha=0.9)

    ax.annotate(
        r'$\varphi(0) = 1$',
        xy=(0.2, 1),
        fontsize=10,
        color=SCT_GRAY,
        xytext=(4, 2.5),
        arrowprops=dict(arrowstyle='->', color=SCT_GRAY, lw=1.2),
    )

    ax.annotate(
        MASTER_FUNCTION_NEGATIVE_ASYMPTOTIC,
        xy=(-7, phi_real(-7)),
        fontsize=9,
        color=SCT_RED,
        xytext=(-5.9, 7.0),
        arrowprops=dict(arrowstyle='->', color=SCT_RED, lw=1.0),
    )

    ax.annotate(
        MASTER_FUNCTION_POSITIVE_ASYMPTOTIC,
        xy=(12, phi_real(12)),
        fontsize=9,
        color=SCT_BLUE,
        xytext=(7.8, 2.05),
        arrowprops=dict(arrowstyle='->', color=SCT_BLUE, lw=1.0),
    )

    ax.axvspan(-8, 0, alpha=0.04, color=SCT_RED)
    ax.axvspan(0, 20, alpha=0.04, color=SCT_BLUE)
    ax.text(-6.7, 0.22, MASTER_FUNCTION_LEFT_LABEL, fontsize=8, color=SCT_RED, alpha=0.75)
    ax.text(13.6, 0.22, MASTER_FUNCTION_RIGHT_LABEL, fontsize=8, color=SCT_BLUE, alpha=0.75)

    return fig, ax


def save_master_function_figure():
    fig, _ = build_master_function_figure()
    fig.savefig(OUT / 'master_function.png')
    plt.close(fig)
    print('[1/4] master_function.png')


def main():
    save_master_function_figure()

    # ===== 2. Modified Newtonian potential =====
    fig, ax = plt.subplots(figsize=(7, 4))

    r = np.linspace(0.01, 8, 500)
    # m_2 = Lambda * sqrt(60/13)  (spin-2, xi-independent)
    # m_0 = Lambda * sqrt(6)      (spin-0, at xi=0)
    m2 = np.sqrt(60 / 13)
    m0 = np.sqrt(6)
    ratio = 1 - (4 / 3) * np.exp(-m2 * r) + (1 / 3) * np.exp(-m0 * r)

    ax.plot(r, ratio, color=SCT_BLUE, linewidth=2.2,
            label=r'SCT ($\xi = 0$):  $V/V_{\rm N} = 1 - \frac{4}{3}e^{-m_2 r}'
                  r' + \frac{1}{3}e^{-m_0 r}$')
    ax.axhline(1, color=SCT_GRAY, linestyle='--', linewidth=1,
               label=r'Newton: $V/V_{\rm N} = 1$')
    ax.axhline(0, color=SCT_RED, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r'$r \cdot \Lambda$  (distance in units of $1/\Lambda$)')
    ax.set_ylabel(r'$V(r) / V_{\rm Newton}(r)$')
    ax.set_title(r'Modified Newtonian Potential ($\xi = 0$)')
    ax.set_ylim(-0.15, 1.35)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    ax.annotate(r'$V(0) = 0$  (finite)', xy=(0.05, 0), fontsize=10, color=SCT_GREEN,
                xytext=(1.5, -0.1),
                arrowprops=dict(arrowstyle='->', color=SCT_GREEN, lw=1.2))

    fig.savefig(OUT / 'newtonian_potential.png')
    plt.close(fig)
    print('[2/4] newtonian_potential.png')

    # ===== 3. SM sector contributions =====
    # Left: bar chart of beta_W per degree of freedom.
    # Right: form factors h_C^(s)(x) directly, showing momentum-dependent running.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2),
                                   gridspec_kw={'width_ratios': [1, 1.3]})

    sectors = ['Scalars\n(Higgs)', 'Fermions\n(quarks +\nleptons)', 'Gauge bosons\n(W, Z, g, ' + r'$\gamma$)']
    beta_W = [1/120, 1/20, 1/10]
    bar_colors = [SCT_GREEN, SCT_ORANGE, SCT_BLUE]

    bars = ax1.bar(sectors, beta_W, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
    ax1.set_ylabel(r'$\beta_W^{(s)}$  (per degree of freedom)')
    ax1.set_title(r'Weyl$^2$ Heat Kernel Coefficients')
    ax1.set_ylim(0, 0.13)

    for bar, val in zip(bars, beta_W):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'1/{int(round(1 / val))}', ha='center', fontsize=11, fontweight='bold')

    # Right: h_C(x) for each spin.
    # Use x >= 0.3 to avoid the 1/x^2 singularity near x=0
    # (the singularities cancel in the full expression, but numerically they dominate at small x).
    x = np.linspace(0.3, 15, 300)
    p = phi_real(x)

    # h_C^(0)(x) = 1/(12x) + (phi-1)/(2x^2)
    hC0 = 1 / (12 * x) + (p - 1) / (2 * x**2)
    # h_C^(1/2)(x) = (3phi-1)/(6x) + 2(phi-1)/x^2
    hC12 = (3 * p - 1) / (6 * x) + 2 * (p - 1) / x**2
    # h_C^(1)(x) = phi/4 + (6phi-5)/(6x) + (phi-1)/x^2
    hC1 = p / 4 + (6 * p - 5) / (6 * x) + (p - 1) / x**2

    ax2.plot(x, hC0, color=SCT_GREEN, linewidth=2, label=r'Scalar $h_C^{(0)}$')
    ax2.plot(x, hC12, color=SCT_ORANGE, linewidth=2, label=r'Dirac $h_C^{(1/2)}$')
    ax2.plot(x, hC1, color=SCT_BLUE, linewidth=2, label=r'Vector $h_C^{(1)}$')
    ax2.axhline(0, color=SCT_GRAY, linestyle=':', linewidth=0.8, alpha=0.5)

    ax2.set_xlabel(r'$x = -k^2/\Lambda^2$  (momentum scale)')
    ax2.set_ylabel(r'$h_C^{(s)}(x)$')
    ax2.set_title('Weyl$^2$ Form Factors by Spin')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.set_ylim(-0.15, 0.35)

    fig.tight_layout()
    fig.savefig(OUT / 'sm_contributions.png')
    plt.close(fig)
    print('[3/4] sm_contributions.png')

    # ===== 4. Roadmap overview (status chart) =====
    tasks = [
        ('NT-1: Dirac form factors', 1.0, 'complete'),
        ('NT-1b: Scalar form factors', 1.0, 'complete'),
        ('NT-1b: Vector form factors', 1.0, 'complete'),
        ('NT-1b: Combined SM', 1.0, 'complete'),
        ('NT-2: Entire-function proof', 1.0, 'complete'),
        ('NT-4a: Linearized field eqs', 1.0, 'complete'),
        ('NT-4b: Nonlinear EOM', 1.0, 'complete'),
        ('NT-4c: FLRW cosmology', 1.0, 'complete'),
        ('PPN-1: Solar system tests', 1.0, 'complete'),
        ('LT-3d: Laboratory tests', 1.0, 'complete'),
        ('MR-1: Lorentzian formulation', 1.0, 'complete'),
        ('MR-2: Unitarity (D\u00b2-quant)', 1.0, 'complete'),
        ('MR-3: Causality analysis', 0.85, 'conditional'),
        ('MR-4: Two-loop structure', 1.0, 'complete'),
        ('MR-5: Finiteness (all-orders)', 0.95, 'conditional'),
        ('MR-5b: Two-loop D=0', 1.0, 'complete'),
        ('MR-6: Convergence analysis', 1.0, 'complete'),
        ('MR-7: Graviton scattering', 1.0, 'complete'),
        ('NT-3: Spectral dimension', 0.85, 'conditional'),
        ('INF-1: Spectral inflation', 0.50, 'conditional'),
        ('MT-2: Modified cosmology', 1.0, 'complete'),
        ('MT-1: Black hole entropy', 0.0, 'pending'),
        ('LT-1: All-orders UV', 0.0, 'pending'),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    y_pos = np.arange(len(tasks))[::-1]

    colors = []
    for _, _, status in tasks:
        if status == 'complete':
            colors.append(SCT_GREEN)
        elif status == 'conditional':
            colors.append(SCT_ORANGE)
        else:
            colors.append(SCT_GRAY)

    ax.barh(y_pos, [t[1] for t in tasks], color=colors, height=0.65,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Completion')
    ax.set_title('SCT Theory \u2014 Research Roadmap Progress', fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SCT_GREEN, label='Complete'),
        Patch(facecolor=SCT_ORANGE, label='Conditional / partial'),
        Patch(facecolor=SCT_GRAY, label='Pending'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    ax.grid(axis='x', alpha=0.2)
    ax.grid(axis='y', alpha=0)

    fig.savefig(OUT / 'roadmap_progress.png')
    plt.close(fig)
    print('[4/4] roadmap_progress.png')

    print('\nAll figures generated in', OUT)


if __name__ == '__main__':
    main()
