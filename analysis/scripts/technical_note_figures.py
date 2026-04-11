"""
Generate all 6 figures for the path_kurtosis technical note.
Uses SciencePlots style. Output to docs/figures/technical_note/
"""
import sys, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'analysis')

# Try SciencePlots
try:
    plt.style.use(['science', 'ieee'])
except Exception:
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

OUTDIR = 'docs/figures/technical_note'
os.makedirs(OUTDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DATA (all from verified computations in this session)
# ═══════════════════════════════════════════════════════════════

# A_eff table: (N, eps, A_eff, A_se, method)
aeff_data = [
    # midpoint, M=30
    (500,  2.0, 0.060, 0.048, 'mid'), (500,  3.0, 0.030, 0.025, 'mid'),
    (1000, 2.0, 0.051, 0.026, 'mid'), (1000, 3.0, 0.015, 0.018, 'mid'),
    (2000, 2.0, 0.069, 0.019, 'mid'), (2000, 3.0, 0.074, 0.012, 'mid'),
    (3000, 2.0, 0.052, 0.016, 'mid'), (3000, 3.0, 0.048, 0.011, 'mid'),
    (5000, 2.0, 0.043, 0.013, 'mid'), (5000, 3.0, 0.057, 0.008, 'mid'),
    (10000,2.0, 0.076, 0.008, 'mid'), (10000,3.0, 0.075, 0.005, 'mid'),
    # exact, M=15
    (5000, 2.0, 0.051, 0.015, 'exact'), (5000, 3.0, 0.068, 0.008, 'exact'),
]

# Edge deletion data (N=5000, eps=5, M=8)
edge_del_frac = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]
edge_del_ratio = [1.000, 0.988, 0.965, 0.892, 0.802, 0.562]
edge_del_d = [15.79, 14.54, 12.43, 17.16, 11.37, 8.27]

# Boundary collapse (N=5000, M=8, fixed C2)
bc_alpha = [1.0, 0.80, 0.60]
bc_Afix = [0.0759, 0.2815, 0.3350]
bc_dk = [0.043, 0.143, 0.147]

# Companion statistics ladder (N=5000, eps=3, M=10)
stat_names = ['Median', 'IQR', 'MAD', 'Skewness', 'Kurtosis']
stat_d = [3.50, -2.24, -2.47, -5.93, 5.86]
stat_type = ['Location', 'Scale', 'Scale', 'Shape', 'Shape']

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: A_eff vs N
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)

for ax, eps_target, title in [(ax1, 2.0, r'$\varepsilon = 2$'), (ax2, 3.0, r'$\varepsilon = 3$')]:
    for method, marker, color, label in [('mid', 'o', '#1f77b4', 'Midpoint surrogate'),
                                          ('exact', 's', '#d62728', 'Exact geodesic')]:
        Ns = [d[0] for d in aeff_data if d[1] == eps_target and d[4] == method]
        As = [d[2] for d in aeff_data if d[1] == eps_target and d[4] == method]
        Es = [d[3] for d in aeff_data if d[1] == eps_target and d[4] == method]
        if Ns:
            ax.errorbar(Ns, As, yerr=Es, fmt=marker, color=color, ms=5,
                       capsize=3, label=label, zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('$N$')
    ax.set_title(title)
    ax.set_xlim(400, 15000)
    ax.set_ylim(-0.02, 0.12)
    ax.axhline(0, color='gray', ls=':', lw=0.5)

ax1.set_ylabel('$A_{\\mathrm{eff}}$')
ax2.legend(fontsize=7, loc='upper left')
fig.suptitle('$A_{\\mathrm{eff}}$ ensemble (M = 15\u201330 seeds per point)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig1_aeff_vs_N.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/fig1_aeff_vs_N.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 1: A_eff vs N — saved', flush=True)

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: eps^2 scaling
# Use midpoint N=5000 M=30 data across eps
# ═══════════════════════════════════════════════════════════════
eps_vals_mid = [0.5, 1.0, 2.0, 3.0, 5.0]
# dk_mean at N=5000 midpoint (from aeff_ensemble.json)
# A_eff * eps^2 * sqrt(N) * C2 = dk
C2 = 1.0/1120
sqrtN = np.sqrt(5000)
aeff_5k_mid = {0.5: 0.059, 1.0: 0.073, 2.0: 0.049, 3.0: 0.065, 5.0: 0.073}
dk_5k = {e: aeff_5k_mid[e] * e**2 * sqrtN * C2 for e in eps_vals_mid}
dk_se_5k = {0.5: 0.072*0.25*sqrtN*C2, 1.0: 0.025*1.0*sqrtN*C2,
            2.0: 0.012*4*sqrtN*C2, 3.0: 0.007*9*sqrtN*C2, 5.0: 0.004*25*sqrtN*C2}

eps2 = np.array([e**2 for e in eps_vals_mid])
dks = np.array([dk_5k[e] for e in eps_vals_mid])
dkse = np.array([dk_se_5k[e] for e in eps_vals_mid])

fig, ax = plt.subplots(figsize=(4, 3.2))
ax.errorbar(eps2, dks, yerr=dkse, fmt='o', color='#1f77b4', ms=5, capsize=3)
# Fit line through perturbative points (eps=2,3)
mask_pert = np.array([False, False, True, True, False])
slope = np.sum(dks[mask_pert] * eps2[mask_pert]) / np.sum(eps2[mask_pert]**2)
eps2_fit = np.linspace(0, 30, 100)
ax.plot(eps2_fit, slope * eps2_fit, 'k--', lw=0.8, label=f'Linear fit ($\\varepsilon=2,3$)')
ax.set_xlabel(r'$\varepsilon^2$')
ax.set_ylabel(r'$\Delta\kappa$')
ax.set_title(r'$\varepsilon^2$-scaling at $N = 5000$ (midpoint, $M = 30$)')
ax.set_xlim(-1, 28)
ax.set_ylim(-0.01, 0.14)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig2_eps2_scaling.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/fig2_eps2_scaling.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 2: eps^2 scaling — saved', flush=True)

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Robustness panel (edge deletion curve)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4, 3.2))
ax.plot(np.array(edge_del_frac)*100, edge_del_ratio, 'o-', color='#2ca02c', ms=5)
ax.axhline(1.0, color='gray', ls=':', lw=0.5)
ax.axhline(0.5, color='red', ls='--', lw=0.5, label='50% threshold')
ax.set_xlabel('Edges deleted (%)')
ax.set_ylabel('Signal retention ratio')
ax.set_title('Q1: Edge deletion robustness ($N=5000$, $\\varepsilon=5$)')
ax.set_xlim(-2, 55)
ax.set_ylim(0.3, 1.1)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig3_edge_deletion.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/fig3_edge_deletion.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 3: Edge deletion — saved', flush=True)

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Boundary collapse (fixed C2)
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

# Panel A: A_eff vs alpha
ax1.plot(bc_alpha, bc_Afix, 'o-', color='#9467bd', ms=6)
ax1.set_xlabel(r'Window $\alpha$ (interior fraction)')
ax1.set_ylabel(r'$A_{\mathrm{eff}}$ (fixed $C_2$)')
ax1.set_title('Q2: Boundary windowing ($N=5000$)')
ax1.set_xlim(0.55, 1.05)

# Panel B: raw dk vs alpha
ax2.plot(bc_alpha, bc_dk, 's-', color='#e377c2', ms=6)
ax2.set_xlabel(r'Window $\alpha$ (interior fraction)')
ax2.set_ylabel(r'Raw $\Delta\kappa$')
ax2.set_title(r'$\Delta\kappa$ amplification: $3.4\times$ at $\alpha=0.80$')
ax2.set_xlim(0.55, 1.05)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig4_boundary_collapse.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/fig4_boundary_collapse.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 4: Boundary collapse — saved', flush=True)

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Per-element scatter (δY vs f)
# Generate fresh data for one seed
# ═══════════════════════════════════════════════════════════════
from sct_tools.hasse import sprinkle_diamond, build_hasse_bitset, path_counts

N = 5000; EPS = 3.0; SEED = 5003000 + 0
pts = sprinkle_diamond(N, seed=SEED)
p0, c0 = build_hasse_bitset(pts, eps=None)
pE, cE = build_hasse_bitset(pts, eps=EPS)
pd0, pu0 = path_counts(p0, c0)
pdE, puE = path_counts(pE, cE)
Y0 = np.log2(pd0 * pu0 + 1.0)
YE = np.log2(pdE * puE + 1.0)
dY = YE - Y0
f_val = 0.5 * (pts[:, 1]**2 - pts[:, 2]**2)
r_coord = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
b_slack = 0.5 - (np.abs(pts[:, 0]) + r_coord)

from scipy.stats import pearsonr
rho, _ = pearsonr(dY, f_val)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
sc = ax.scatter(f_val, dY, c=b_slack, s=0.5, alpha=0.4, cmap='coolwarm',
                vmin=0, vmax=0.25, rasterized=True)
ax.set_xlabel(r'$f(x,y) = (x^2 - y^2)/2$')
ax.set_ylabel(r'$\delta Y = Y_\varepsilon - Y_0$')
ax.set_title(f'Per-element response ($N=5000$, $\\varepsilon=3$, $r = {rho:.2f}$)')
cb = plt.colorbar(sc, ax=ax, label='Boundary slack', shrink=0.85)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig5_scatter_dY_vs_f.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/fig5_scatter_dY_vs_f.png', bbox_inches='tight', dpi=200)
plt.close()
print(f'Fig 5: Scatter dY vs f (r={rho:.3f}) — saved', flush=True)

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Companion statistics ladder (Cohen's d panel)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 3))
colors = ['#ff7f0e' if t == 'Location' else '#2ca02c' if t == 'Scale' else '#1f77b4'
          for t in stat_type]
bars = ax.barh(range(len(stat_names)), stat_d, color=colors, edgecolor='k', lw=0.5)
ax.set_yticks(range(len(stat_names)))
ax.set_yticklabels(stat_names)
ax.set_xlabel("Cohen's $d$")
ax.set_title('Q3: Sensitivity by statistic ($N=5000$, $\\varepsilon=3$)')
ax.axvline(0, color='k', lw=0.5)
ax.axvline(3, color='gray', ls='--', lw=0.5)
ax.axvline(-3, color='gray', ls='--', lw=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#ff7f0e', label='Location'),
                   Patch(facecolor='#2ca02c', label='Scale'),
                   Patch(facecolor='#1f77b4', label='Shape')]
ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig6_statistics_ladder.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/fig6_statistics_ladder.png', bbox_inches='tight', dpi=200)
plt.close()
print('Fig 6: Statistics ladder — saved', flush=True)

print('\nAll 6 figures generated.', flush=True)
