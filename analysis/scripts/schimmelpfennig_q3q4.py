"""
Computations for Schimmelpfennig Q3 (full distribution) and Q4 (mesoscopic localization).
N=5000, eps=5, M=10 CRN seeds. Uses bitset Hasse.
"""
import sys, time, json
import numpy as np
from scipy import stats
sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_diamond, build_hasse_bitset, path_counts,
    path_kurtosis_from_lists
)

N = 5000
EPS = 5.0
M = 10
T = 1.0

print("=" * 70, flush=True)
print(f"Q3/Q4 ANALYSIS: N={N}, eps={EPS}, M={M}", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# Collect per-element data across seeds
# ═══════════════════════════════════════════════════════════════
all_Y0 = []          # flat Y values
all_Ye = []          # pp-wave Y values
all_dY = []          # Ye - Y0 per element
all_t_frac = []      # normalized time coordinate
all_r_frac = []      # normalized radial coordinate
all_f_val = []       # f(x,y) = (x^2 - y^2)/2
all_dk_contrib = []  # per-element contribution to kurtosis shift

t_total = time.time()

for m in range(M):
    seed = 1000 * N + 100 * int(EPS * 10) + m
    t0 = time.time()

    pts = sprinkle_diamond(N, T=T, seed=seed)
    t_coord = pts[:, 0]
    x_coord = pts[:, 1]
    y_coord = pts[:, 2]
    z_coord = pts[:, 3]

    # Flat
    par0, ch0 = build_hasse_bitset(pts, eps=None)
    pd0, pu0 = path_counts(par0, ch0)
    Y0 = np.log2(pd0 * pu0 + 1.0)

    # pp-wave
    parE, chE = build_hasse_bitset(pts, eps=EPS)
    pdE, puE = path_counts(parE, chE)
    Ye = np.log2(pdE * puE + 1.0)

    dY = Ye - Y0

    # Coordinates (normalized)
    r = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
    t_frac = (t_coord - t_coord.min()) / (t_coord.max() - t_coord.min())  # 0..1
    r_max = r.max()
    r_frac = r / r_max if r_max > 0 else r

    f_val = (x_coord**2 - y_coord**2) / 2.0

    # Per-element kurtosis contribution: (X^4/sigma^4 - 3) where X = Y - mean
    mu0 = Y0.mean()
    s0 = Y0.std()
    X0 = (Y0 - mu0) / s0 if s0 > 1e-10 else Y0 * 0

    muE = Ye.mean()
    sE = Ye.std()
    XE = (Ye - muE) / sE if sE > 1e-10 else Ye * 0

    dk_contrib = XE**4 - X0**4  # per-element shift in 4th moment (unnormalized)

    all_Y0.extend(Y0.tolist())
    all_Ye.extend(Ye.tolist())
    all_dY.extend(dY.tolist())
    all_t_frac.extend(t_frac.tolist())
    all_r_frac.extend(r_frac.tolist())
    all_f_val.extend(f_val.tolist())
    all_dk_contrib.extend(dk_contrib.tolist())

    dt = time.time() - t0
    print(f"  seed {m}: dk={Ye.mean() - Y0.mean():.4f} (mean dY), "
          f"kurtosis flat={stats.kurtosis(Y0):.4f} ppw={stats.kurtosis(Ye):.4f}  ({dt:.1f}s)",
          flush=True)

Y0 = np.array(all_Y0)
Ye = np.array(all_Ye)
dY = np.array(all_dY)
t_frac = np.array(all_t_frac)
r_frac = np.array(all_r_frac)
f_val = np.array(all_f_val)
dk_contrib = np.array(all_dk_contrib)

n_total = len(Y0)
print(f"\nTotal elements: {n_total} ({M} seeds x {N})", flush=True)

# ═══════════════════════════════════════════════════════════════
# Q3: FULL DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q3: FULL DISTRIBUTION — flat vs pp-wave")
print("=" * 70)

# Basic moments
for label, arr in [("flat Y0", Y0), ("ppwave Ye", Ye), ("delta dY", dY)]:
    print(f"\n  {label}:")
    print(f"    mean={arr.mean():.4f}, std={arr.std():.4f}")
    print(f"    skew={stats.skew(arr):.4f}, kurtosis={stats.kurtosis(arr):.4f}")
    print(f"    min={arr.min():.2f}, Q1={np.percentile(arr,25):.2f}, "
          f"median={np.median(arr):.2f}, Q3={np.percentile(arr,75):.2f}, max={arr.max():.2f}")

# Percentile-by-percentile comparison
print("\n  Percentile shift (Ye - Y0):")
print(f"  {'pct':>5s}  {'Y0':>8s}  {'Ye':>8s}  {'shift':>8s}  {'rel%':>8s}")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    q0 = np.percentile(Y0, p)
    qe = np.percentile(Ye, p)
    shift = qe - q0
    rel = 100 * shift / q0 if abs(q0) > 0.01 else float('nan')
    print(f"  {p:5d}  {q0:8.3f}  {qe:8.3f}  {shift:+8.4f}  {rel:+8.2f}%")

# Tail analysis: what fraction of kurtosis shift comes from elements beyond 2σ
X0_std = (Y0 - Y0.mean()) / Y0.std()
Xe_std = (Ye - Ye.mean()) / Ye.std()
for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
    mask0 = np.abs(X0_std) > threshold
    maskE = np.abs(Xe_std) > threshold
    frac0 = mask0.sum() / n_total
    fracE = maskE.sum() / n_total
    # Contribution to 4th moment from tails
    m4_tail0 = np.mean(X0_std[mask0]**4) * frac0 if mask0.sum() > 0 else 0
    m4_tailE = np.mean(Xe_std[maskE]**4) * fracE if maskE.sum() > 0 else 0
    m4_total0 = np.mean(X0_std**4)
    m4_totalE = np.mean(Xe_std**4)
    print(f"  |z|>{threshold:.1f}: flat {frac0:.4f} ({100*m4_tail0/m4_total0:.1f}% of m4) | "
          f"ppw {fracE:.4f} ({100*m4_tailE/m4_totalE:.1f}% of m4)")

# KS test
ks_stat, ks_p = stats.ks_2samp(Y0, Ye)
print(f"\n  KS test (Y0 vs Ye): stat={ks_stat:.4f}, p={ks_p:.2e}")

# ═══════════════════════════════════════════════════════════════
# Q4: MESOSCOPIC LOCALIZATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q4: MESOSCOPIC LOCALIZATION")
print("=" * 70)

# Stratify by time (5 strata)
print("\n  A. Time stratification (5 equal bins):")
print(f"  {'stratum':>8s}  {'n':>6s}  {'mean_dY':>10s}  {'std_dY':>10s}  {'mean_dk4':>10s}  {'|f|_mean':>10s}")
t_bins = np.linspace(0, 1, 6)
for k in range(5):
    mask = (t_frac >= t_bins[k]) & (t_frac < t_bins[k+1])
    if k == 4:
        mask = (t_frac >= t_bins[k]) & (t_frac <= t_bins[k+1])
    n_k = mask.sum()
    if n_k == 0:
        continue
    md = dY[mask].mean()
    sd = dY[mask].std()
    mk = dk_contrib[mask].mean()
    mf = np.abs(f_val[mask]).mean()
    label = f"[{t_bins[k]:.1f},{t_bins[k+1]:.1f})"
    print(f"  {label:>8s}  {n_k:6d}  {md:+10.5f}  {sd:10.5f}  {mk:+10.5f}  {mf:10.5f}")

# Stratify by radial distance (5 strata)
print("\n  B. Radial stratification (5 equal bins):")
print(f"  {'stratum':>8s}  {'n':>6s}  {'mean_dY':>10s}  {'std_dY':>10s}  {'mean_dk4':>10s}  {'|f|_mean':>10s}")
r_bins = np.linspace(0, 1, 6)
for k in range(5):
    mask = (r_frac >= r_bins[k]) & (r_frac < r_bins[k+1])
    if k == 4:
        mask = (r_frac >= r_bins[k]) & (r_frac <= r_bins[k+1])
    n_k = mask.sum()
    if n_k == 0:
        continue
    md = dY[mask].mean()
    sd = dY[mask].std()
    mk = dk_contrib[mask].mean()
    mf = np.abs(f_val[mask]).mean()
    label = f"[{r_bins[k]:.1f},{r_bins[k+1]:.1f})"
    print(f"  {label:>8s}  {n_k:6d}  {md:+10.5f}  {sd:10.5f}  {mk:+10.5f}  {mf:10.5f}")

# Stratify by |f(x,y)| (quartiles)
print("\n  C. Profile |f| stratification (quartiles):")
f_abs = np.abs(f_val)
f_quartiles = np.percentile(f_abs, [0, 25, 50, 75, 100])
print(f"  {'stratum':>8s}  {'n':>6s}  {'mean_dY':>10s}  {'std_dY':>10s}  {'mean_dk4':>10s}  {'|f|_mean':>10s}")
labels = ["Q1low", "Q2", "Q3", "Q4high"]
for k in range(4):
    mask = (f_abs >= f_quartiles[k]) & (f_abs < f_quartiles[k+1])
    if k == 3:
        mask = (f_abs >= f_quartiles[k]) & (f_abs <= f_quartiles[k+1])
    n_k = mask.sum()
    if n_k == 0:
        continue
    md = dY[mask].mean()
    sd = dY[mask].std()
    mk = dk_contrib[mask].mean()
    mf = f_abs[mask].mean()
    print(f"  {labels[k]:>8s}  {n_k:6d}  {md:+10.5f}  {sd:10.5f}  {mk:+10.5f}  {mf:10.5f}")

# Correlation matrix: dY vs coordinates
print("\n  D. Correlations with dY:")
for name, arr in [("t_frac", t_frac), ("r_frac", r_frac),
                   ("|f|", f_abs), ("f", f_val), ("Y0", Y0)]:
    r_corr, p_corr = stats.pearsonr(arr, dY)
    print(f"    corr(dY, {name:>6s}) = {r_corr:+.4f}  (p={p_corr:.2e})")

# Top/bottom contributors
print("\n  E. Extreme contributors to kurtosis shift:")
idx_sort = np.argsort(dk_contrib)
print("    5 LARGEST positive dk_contrib:")
for i in idx_sort[-5:][::-1]:
    print(f"      dk4={dk_contrib[i]:+.4f}, Y0={Y0[i]:.2f}, Ye={Ye[i]:.2f}, "
          f"dY={dY[i]:+.3f}, t={t_frac[i]:.2f}, r={r_frac[i]:.2f}, f={f_val[i]:+.5f}")
print("    5 LARGEST negative dk_contrib:")
for i in idx_sort[:5]:
    print(f"      dk4={dk_contrib[i]:+.4f}, Y0={Y0[i]:.2f}, Ye={Ye[i]:.2f}, "
          f"dY={dY[i]:+.3f}, t={t_frac[i]:.2f}, r={r_frac[i]:.2f}, f={f_val[i]:+.5f}")

# Fraction of signal from top 10% contributors
top10_mask = dk_contrib > np.percentile(dk_contrib, 90)
bot10_mask = dk_contrib < np.percentile(dk_contrib, 10)
print(f"\n  F. Signal concentration:")
print(f"    Top 10% of dk_contrib: sum={dk_contrib[top10_mask].sum():.4f} "
      f"({100*dk_contrib[top10_mask].sum()/dk_contrib.sum():.1f}% of total)")
print(f"    Bottom 10%: sum={dk_contrib[bot10_mask].sum():.4f}")
print(f"    Middle 80%: sum={dk_contrib[~top10_mask & ~bot10_mask].sum():.4f}")
total_dk = dk_contrib.sum()
print(f"    Total sum(dk_contrib) = {total_dk:.4f}")

elapsed = time.time() - t_total
print(f"\nTotal time: {elapsed:.0f}s")
print("\nDONE.")
