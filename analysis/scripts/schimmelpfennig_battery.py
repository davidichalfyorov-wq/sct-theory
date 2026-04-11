"""
Schimmelpfennig battery — TOP-3 tests from independent analysis experimental design.
  Test 1: Poisson Thinning & Rebuild (Q1)
  Test 5: Transport Support / Location-Scale Residual (Q3)
  Test 7: Conditional Residual Localization (Q4)

N=5000, eps=3.0, M=10, exact pp-wave causal relation.
"""
import sys, time, numpy as np
from scipy import stats as sp_stats
sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_diamond, build_hasse_bitset, path_counts,
    path_kurtosis_from_lists
)

N = 5000
EPS = 3.0
M = 10
T = 1.0

def Y_from_graph(parents, children):
    pd, pu = path_counts(parents, children)
    return np.log2(pd * pu + 1.0)

def excess_kurtosis(Y):
    X = Y - Y.mean()
    s2 = np.var(Y)
    return 0.0 if s2 < 1e-12 else float(np.mean(X**4) / (s2**2) - 3.0)

def empirical_C2(pts, mask=None):
    if mask is None:
        mask = np.ones(len(pts), dtype=bool)
    f = 0.5 * (pts[mask, 1]**2 - pts[mask, 2]**2)
    return np.mean(f * f)

def A_mask(pts, Y0, Ye, eps, mask=None):
    if mask is None:
        mask = np.ones(len(pts), dtype=bool)
    dk = excess_kurtosis(Ye[mask]) - excess_kurtosis(Y0[mask])
    denom = eps**2 * np.sqrt(mask.sum()) * empirical_C2(pts, mask)
    return dk / denom if abs(denom) > 1e-15 else 0.0

def quantile_bins(x, nq):
    q = np.quantile(x, np.linspace(0, 1, nq + 1))
    q[0] -= 1e-12
    q[-1] += 1e-12
    return np.digitize(x, q[1:-1], right=False)

def longest_depth(parents):
    n = len(parents)
    depth = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = parents[i]
        depth[i] = 0 if len(p) == 0 else 1 + int(np.max(depth[p]))
    return depth

def total_degree(parents, children):
    n = len(parents)
    deg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        deg[i] = len(parents[i]) + len(children[i])
    return deg

def bootstrap_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    boots = np.array([np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(B)])
    return np.quantile(boots, [0.025, 0.975])

# ═══════════════════════════════════════════════════════════════
# PRE-BUILD all graphs (shared across tests)
# ═══════════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print(f"SCHIMMELPFENNIG BATTERY: N={N}, eps={EPS}, M={M}, exact=True", flush=True)
print("=" * 70, flush=True)

all_pts = []
all_p0 = []
all_c0 = []
all_pE = []
all_cE = []
all_Y0 = []
all_YE = []
all_Abase = []

t_build = time.time()
for m in range(M):
    seed = 1000 * N + 100 * int(EPS * 10) + m
    pts = sprinkle_diamond(N, T=T, seed=seed)
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pE, cE = build_hasse_bitset(pts, eps=EPS)
    Y0 = Y_from_graph(p0, c0)
    YE = Y_from_graph(pE, cE)
    Ab = A_mask(pts, Y0, YE, EPS)

    all_pts.append(pts)
    all_p0.append(p0)
    all_c0.append(c0)
    all_pE.append(pE)
    all_cE.append(cE)
    all_Y0.append(Y0)
    all_YE.append(YE)
    all_Abase.append(Ab)
    print(f"  Built seed {m}: A_base={Ab:.4f} ({time.time()-t_build:.0f}s)", flush=True)

print(f"Build time: {time.time()-t_build:.0f}s\n", flush=True)

# ═══════════════════════════════════════════════════════════════
# TEST 1: POISSON THINNING & REBUILD (Q1)
# ═══════════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print("TEST 1: POISSON THINNING & REBUILD (Q1)", flush=True)
print("  Pre-registered: GOOD if lower 95% CI of R_0.70 > 0.50", flush=True)
print("                  BAD  if upper 95% CI of R_0.70 < 0.25", flush=True)
print("=" * 70, flush=True)

t0 = time.time()
keep_fracs = [0.70, 0.50]
thin_results = {q: [] for q in keep_fracs}

for m in range(M):
    pts = all_pts[m]
    Ab = all_Abase[m]
    seed = 1000 * N + 100 * int(EPS * 10) + m

    for q in keep_fracs:
        rng = np.random.default_rng(seed + int(1000 * q))
        keep = rng.random(N) < q
        subpts = pts[keep]

        p0s, c0s = build_hasse_bitset(subpts, eps=None)
        pEs, cEs = build_hasse_bitset(subpts, eps=EPS)
        Y0s = Y_from_graph(p0s, c0s)
        YEs = Y_from_graph(pEs, cEs)
        Aq = A_mask(subpts, Y0s, YEs, EPS)
        R = Aq / Ab if abs(Ab) > 1e-15 else 0.0
        thin_results[q].append(R)

    print(f"  seed {m}: R_70%={thin_results[0.70][-1]:.3f}  R_50%={thin_results[0.50][-1]:.3f}", flush=True)

for q in keep_fracs:
    arr = np.array(thin_results[q])
    ci = bootstrap_ci(arr)
    print(f"\n  q={q:.0%}: mean R = {arr.mean():.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
    if q == 0.70:
        if ci[0] > 0.50:
            print(f"  >>> GOOD: lower CI {ci[0]:.3f} > 0.50", flush=True)
        elif ci[1] < 0.25:
            print(f"  >>> BAD: upper CI {ci[1]:.3f} < 0.25", flush=True)
        else:
            print(f"  >>> INDETERMINATE", flush=True)

print(f"  Time: {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════════════════════════════════════════════════════
# TEST 5: TRANSPORT SUPPORT / LOCATION-SCALE RESIDUAL (Q3)
# ═══════════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print("TEST 5: TRANSPORT SUPPORT / LOCATION-SCALE RESIDUAL (Q3)", flush=True)
print("  Pre-registered: Shape change real if D_LS > 95th pct of flat-flat null", flush=True)
print("  Broad if S_eff > 0.50. Tail-driven if S_eff < 0.25.", flush=True)
print("=" * 70, flush=True)

t0 = time.time()
p_grid = np.linspace(0.01, 0.99, 99)

D_ls_ppw = []
S_eff_ppw = []

# Flat-flat null calibration
D_ls_null = []
for m in range(M):
    # Two independent flat sprinklings
    pts1 = sprinkle_diamond(N, seed=50000 + m)
    pts2 = sprinkle_diamond(N, seed=60000 + m)
    p1, c1 = build_hasse_bitset(pts1, eps=None)
    p2, c2 = build_hasse_bitset(pts2, eps=None)
    Y1 = Y_from_graph(p1, c1)
    Y2 = Y_from_graph(p2, c2)

    q1 = np.quantile(Y1, p_grid)
    q2 = np.quantile(Y2, p_grid)
    iqr1 = np.quantile(Y1, 0.75) - np.quantile(Y1, 0.25)
    iqr2 = np.quantile(Y2, 0.75) - np.quantile(Y2, 0.25)
    b_ls = iqr2 / iqr1 if iqr1 > 1e-10 else 1.0
    a_ls = np.median(Y2) - b_ls * np.median(Y1)
    R = q2 - (a_ls + b_ls * q1)
    D_ls_null.append(np.mean(R * R))

D_ls_null = np.array(D_ls_null)
null_95 = np.percentile(D_ls_null, 95)
print(f"  Flat-flat null D_LS: mean={D_ls_null.mean():.6f}, 95th pct={null_95:.6f}", flush=True)

# PP-wave signal
for m in range(M):
    Y0 = all_Y0[m]
    YE = all_YE[m]

    q0 = np.quantile(Y0, p_grid)
    qE = np.quantile(YE, p_grid)
    iqr0 = np.quantile(Y0, 0.75) - np.quantile(Y0, 0.25)
    iqrE = np.quantile(YE, 0.75) - np.quantile(YE, 0.25)
    b_ls = iqrE / iqr0 if iqr0 > 1e-10 else 1.0
    a_ls = np.median(YE) - b_ls * np.median(Y0)
    R = qE - (a_ls + b_ls * q0)

    D_ls_ppw.append(np.mean(R * R))
    S_eff_ppw.append((np.sum(np.abs(R))**2) / (len(R) * np.sum(R * R)) if np.sum(R*R) > 1e-15 else 0)

D_ls_ppw = np.array(D_ls_ppw)
S_eff_ppw = np.array(S_eff_ppw)

n_exceed = np.sum(D_ls_ppw > null_95)
print(f"  PP-wave D_LS: mean={D_ls_ppw.mean():.6f}, seeds exceeding null 95th: {n_exceed}/{M}", flush=True)
S_ci = bootstrap_ci(S_eff_ppw)
print(f"  S_eff: mean={S_eff_ppw.mean():.3f}, 95% CI = [{S_ci[0]:.3f}, {S_ci[1]:.3f}]", flush=True)

if n_exceed >= M // 2:
    print(f"  >>> Shape change IS REAL (majority of seeds exceed null)", flush=True)
    if S_ci[0] > 0.50:
        print(f"  >>> BROAD structural shift (S_eff lower CI > 0.50)", flush=True)
    elif S_ci[1] < 0.25:
        print(f"  >>> TAIL-DRIVEN shape change (S_eff upper CI < 0.25)", flush=True)
    else:
        print(f"  >>> MIXED / intermediate support", flush=True)
else:
    print(f"  >>> Shape change NOT detected beyond location-scale", flush=True)

print(f"  Time: {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════════════════════════════════════════════════════
# TEST 7: CONDITIONAL RESIDUAL LOCALIZATION (Q4)
# ═══════════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print("TEST 7: CONDITIONAL RESIDUAL LOCALIZATION (Q4)", flush=True)
print("  Pre-registered: No residual if all Holm-corrected p > 0.05 AND E < 0.10", flush=True)
print("                  Residual present if any corrected p < 0.01 AND E > 0.10", flush=True)
print("=" * 70, flush=True)

t0 = time.time()
NPERM = 200

def stratified_perm_stat(resid, fbin, gbin):
    S = 0.0
    for b in np.unique(fbin):
        idxb = (fbin == b)
        if idxb.sum() < 10:
            continue
        grand = resid[idxb].mean()
        for q_val in np.unique(gbin[idxb]):
            idx = idxb & (gbin == q_val)
            if idx.sum() < 5:
                continue
            S += idx.sum() * (resid[idx].mean() - grand)**2
    return S

def permute_within_fbins(gbin, fbin, rng):
    gnew = gbin.copy()
    for b in np.unique(fbin):
        idx = np.where(fbin == b)[0]
        gnew[idx] = rng.permutation(gnew[idx])
    return gnew

feat_pvals = {'degree': [], 'depth': [], 'Y0': []}
feat_effs = {'degree': [], 'depth': [], 'Y0': []}

for m in range(M):
    pts = all_pts[m]
    Y0 = all_Y0[m]
    YE = all_YE[m]
    p0 = all_p0[m]
    c0 = all_c0[m]
    dY = YE - Y0
    f = 0.5 * (pts[:, 1]**2 - pts[:, 2]**2)
    seed = 1000 * N + 100 * int(EPS * 10) + m
    rng = np.random.default_rng(seed + 777)

    feats = {
        'degree': total_degree(p0, c0).astype(float),
        'depth': longest_depth(p0).astype(float),
        'Y0': Y0.copy(),
    }

    fbin = quantile_bins(f, 10)

    resid = np.empty_like(dY)
    for b_val in np.unique(fbin):
        idx = (fbin == b_val)
        resid[idx] = dY[idx] - np.median(dY[idx])

    total_var = np.sum((resid - resid.mean())**2)

    for name, g in feats.items():
        gbin = quantile_bins(g, 5)
        Sobs = stratified_perm_stat(resid, fbin, gbin)

        Sperm = np.empty(NPERM)
        for r in range(NPERM):
            gperm = permute_within_fbins(gbin, fbin, rng)
            Sperm[r] = stratified_perm_stat(resid, fbin, gperm)

        pval = (1 + np.sum(Sperm >= Sobs)) / (NPERM + 1)
        eff = Sobs / total_var if total_var > 1e-15 else 0.0
        feat_pvals[name].append(pval)
        feat_effs[name].append(eff)

    print(f"  seed {m}: p_degree={feat_pvals['degree'][-1]:.3f} "
          f"p_depth={feat_pvals['depth'][-1]:.3f} "
          f"p_Y0={feat_pvals['Y0'][-1]:.3f}", flush=True)

print(f"\n  SUMMARY:", flush=True)
for name in ['degree', 'depth', 'Y0']:
    ps = np.array(feat_pvals[name])
    es = np.array(feat_effs[name])
    median_p = np.median(ps)
    mean_e = es.mean()
    frac_sig = np.mean(ps < 0.05)
    print(f"    {name:>8s}: median p = {median_p:.3f}, mean E = {mean_e:.4f}, "
          f"frac p<0.05 = {frac_sig:.1%}", flush=True)
    if median_p > 0.05 and mean_e < 0.10:
        print(f"             >>> NO residual localization by {name}", flush=True)
    elif median_p < 0.01 and mean_e > 0.10:
        print(f"             >>> RESIDUAL localization by {name} PRESENT", flush=True)
    else:
        print(f"             >>> INDETERMINATE", flush=True)

print(f"  Time: {time.time()-t0:.0f}s", flush=True)
print(f"\nALL TESTS COMPLETE.", flush=True)
