"""
Schimmelpfennig battery — remaining tests (#5, #7, #9).
  Test 5: Boundary Collapse Curve (Q2) — multi-N
  Test 7: Two-Step Thickening (Q1)
  Test 9: Layer-Residue Decimation (Q1/Q4)

Reuses exact pp-wave causal relation.
"""
import sys, time, numpy as np
from scipy import stats as sp_stats
sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_diamond, build_hasse_bitset, path_counts
)

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
    n_m = mask.sum()
    if n_m < 50:
        return 0.0
    dk = excess_kurtosis(Ye[mask]) - excess_kurtosis(Y0[mask])
    denom = eps**2 * np.sqrt(n_m) * empirical_C2(pts, mask)
    return dk / denom if abs(denom) > 1e-15 else 0.0

def longest_depth(parents):
    n = len(parents)
    depth = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = parents[i]
        depth[i] = 0 if len(p) == 0 else 1 + int(np.max(depth[p]))
    return depth

def boundary_slack(pts, T=1.0):
    r = np.sqrt(np.sum(pts[:, 1:]**2, axis=1))
    return T / 2 - (np.abs(pts[:, 0]) + r)

def bootstrap_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    boots = np.array([np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(B)])
    return np.quantile(boots, [0.025, 0.975])

EPS = 3.0
T = 1.0
M = 8

# ═══════════════════════════════════════════════════════════════
# TEST 5: BOUNDARY COLLAPSE CURVE (Q2)
# Multi-N: does A_alpha(N) - A_1.0(N) → 0 as N→∞?
# ═══════════════════════════════════════════════════════════════
print("=" * 70, flush=True)
print("TEST 5: BOUNDARY COLLAPSE CURVE (Q2)", flush=True)
print("  H0: boundary dependence is finite-size artifact (c_alpha = 0)", flush=True)
print("  H1: intrinsic boundary feature (c_alpha != 0)", flush=True)
print("=" * 70, flush=True)

N_VALUES = [2000, 3000, 5000]
ALPHAS = [1.0, 0.90, 0.80, 0.70, 0.60]

t0 = time.time()
bc_data = {N: {a: [] for a in ALPHAS} for N in N_VALUES}

for N in N_VALUES:
    print(f"\n  N={N}:", flush=True)
    for m in range(M):
        seed = 1000 * N + 100 * int(EPS * 10) + m
        pts = sprinkle_diamond(N, T=T, seed=seed)
        p0, c0 = build_hasse_bitset(pts, eps=None)
        pE, cE = build_hasse_bitset(pts, eps=EPS)
        Y0 = Y_from_graph(p0, c0)
        YE = Y_from_graph(pE, cE)
        b = boundary_slack(pts, T)

        for a in ALPHAS:
            if a == 1.0:
                mask = np.ones(N, dtype=bool)
            else:
                thr = np.quantile(b, 1.0 - a)
                mask = b >= thr
            bc_data[N][a].append(A_mask(pts, Y0, YE, EPS, mask))

        print(f"    m={m}: A_1.0={bc_data[N][1.0][-1]:.4f}  "
              f"A_0.80={bc_data[N][0.80][-1]:.4f}  "
              f"A_0.60={bc_data[N][0.60][-1]:.4f}", flush=True)

print(f"\n  --- A_alpha table (mean ± SE) ---", flush=True)
print(f"  {'N':>6s}", end="", flush=True)
for a in ALPHAS:
    print(f"  {'α='+str(a):>14s}", end="", flush=True)
print(flush=True)
for N in N_VALUES:
    print(f"  {N:6d}", end="", flush=True)
    for a in ALPHAS:
        arr = np.array(bc_data[N][a])
        print(f"  {arr.mean():+7.4f}±{arr.std()/np.sqrt(M):.4f}", end="", flush=True)
    print(flush=True)

# Delta_alpha = A_alpha - A_1.0 for each seed
print(f"\n  --- Delta_alpha = A_alpha - A_1.0 (mean) ---", flush=True)
print(f"  {'N':>6s}  {'N^-1/4':>8s}", end="", flush=True)
for a in [0.90, 0.80, 0.70, 0.60]:
    print(f"  {'Δ_'+str(a):>10s}", end="", flush=True)
print(flush=True)
for N in N_VALUES:
    deltas = {}
    for a in [0.90, 0.80, 0.70, 0.60]:
        d = np.array(bc_data[N][a]) - np.array(bc_data[N][1.0])
        deltas[a] = d.mean()
    print(f"  {N:6d}  {N**(-0.25):8.4f}", end="", flush=True)
    for a in [0.90, 0.80, 0.70, 0.60]:
        print(f"  {deltas[a]:+10.4f}", end="", flush=True)
    print(flush=True)

# Simple trend: does |Delta_alpha| decrease with N?
print(f"\n  Trend check: |Δ_0.80| across N:", flush=True)
for N in N_VALUES:
    d = np.array(bc_data[N][0.80]) - np.array(bc_data[N][1.0])
    print(f"    N={N}: mean Δ={d.mean():+.4f} ± {d.std()/np.sqrt(M):.4f}", flush=True)

print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════════
# TEST 7: TWO-STEP THICKENING (Q1)
# Add "almost-direct" links (one intermediate) to both graphs
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("TEST 7: TWO-STEP THICKENING (Q1)", flush=True)
print("  Pre-registered: GOOD if lower 95% CI of R_5% > 0.50", flush=True)
print("                  BAD  if upper 95% CI of R_5% < 0.25", flush=True)
print("=" * 70, flush=True)

N = 5000
t0 = time.time()

def find_one_intermediate_candidates(parents):
    """Find pairs (j, i) where j is NOT a direct parent of i,
    but there exists exactly one k with j->k->i (both Hasse links)."""
    n = len(parents)
    candidates = [[] for _ in range(n)]
    for i in range(n):
        parent_set = set(int(j) for j in parents[i])
        grandparent_count = {}
        for k in parents[i]:
            for j in parents[k]:
                jj = int(j)
                if jj not in parent_set:
                    grandparent_count[jj] = grandparent_count.get(jj, 0) + 1
        # Only keep those with exactly 1 intermediate
        candidates[i] = [j for j, c in grandparent_count.items() if c == 1]
    return candidates

def thicken_graph(parents, children, frac, seed_key):
    """Add frac * n_existing_edges from one-intermediate candidates."""
    n = len(parents)
    n_edges = sum(len(p) for p in parents)
    target = int(round(frac * n_edges))
    if target == 0:
        return parents, children

    cands = find_one_intermediate_candidates(parents)
    # Collect all candidates with deterministic scoring
    rng = np.random.default_rng(seed_key)
    all_cands = []
    for i in range(n):
        for j in cands[i]:
            all_cands.append((rng.random(), j, i))
    all_cands.sort(key=lambda x: x[0])
    chosen = set((j, i) for _, j, i in all_cands[:target])

    new_parents = []
    new_children = [[] for _ in range(n)]
    for i in range(n):
        pars = list(int(j) for j in parents[i])
        for j, ii in chosen:
            if ii == i:
                pars.append(j)
        pars = sorted(set(pars))
        arr = np.array(pars, dtype=np.int32)
        new_parents.append(arr)
        for j in arr:
            new_children[j].append(i)
    return new_parents, new_children

THICKEN_FRACS = [0.02, 0.05]
thick_results = {q: [] for q in THICKEN_FRACS}

for m in range(M):
    seed = 1000 * N + 100 * int(EPS * 10) + m
    pts = sprinkle_diamond(N, T=T, seed=seed)
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pE, cE = build_hasse_bitset(pts, eps=EPS)
    Y0 = Y_from_graph(p0, c0)
    YE = Y_from_graph(pE, cE)
    Ab = A_mask(pts, Y0, YE, EPS)

    for q in THICKEN_FRACS:
        key = seed + int(q * 10000)
        p0t, c0t = thicken_graph(p0, c0, q, key)
        pEt, cEt = thicken_graph(pE, cE, q, key)
        Y0t = Y_from_graph(p0t, c0t)
        YEt = Y_from_graph(pEt, cEt)
        At = A_mask(pts, Y0t, YEt, EPS)
        R = At / Ab if abs(Ab) > 1e-10 else 0.0
        thick_results[q].append(R)

    print(f"  seed {m}: R_2%={thick_results[0.02][-1]:.3f}  "
          f"R_5%={thick_results[0.05][-1]:.3f}", flush=True)

for q in THICKEN_FRACS:
    arr = np.array(thick_results[q])
    ci = bootstrap_ci(arr)
    print(f"\n  q={q:.0%}: mean R = {arr.mean():.3f}, "
          f"median = {np.median(arr):.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
    if q == 0.05:
        if ci[0] > 0.50:
            print(f"  >>> GOOD: lower CI {ci[0]:.3f} > 0.50", flush=True)
        elif ci[1] < 0.25:
            print(f"  >>> BAD: upper CI {ci[1]:.3f} < 0.25", flush=True)
        else:
            print(f"  >>> INDETERMINATE", flush=True)

print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════════
# TEST 9: LAYER-RESIDUE DECIMATION (Q1/Q4)
# Remove one depth-residue class mod 5, rebuild
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("TEST 9: LAYER-RESIDUE DECIMATION (Q1/Q4)", flush=True)
print("  Pre-registered: GOOD if mean retention > 0.50 AND worst > 0.30", flush=True)
print("                  BAD  if mean retention < 0.25", flush=True)
print("=" * 70, flush=True)

t0 = time.time()
K = 5
layer_results = {r: [] for r in range(K)}

for m in range(M):
    seed = 1000 * N + 100 * int(EPS * 10) + m
    pts = sprinkle_diamond(N, T=T, seed=seed)
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pE, cE = build_hasse_bitset(pts, eps=EPS)
    Y0 = Y_from_graph(p0, c0)
    YE = Y_from_graph(pE, cE)
    Ab = A_mask(pts, Y0, YE, EPS)

    depth = longest_depth(p0)

    rs = []
    for r in range(K):
        keep = (depth % K) != r
        subpts = pts[keep]
        p0s, c0s = build_hasse_bitset(subpts, eps=None)
        pEs, cEs = build_hasse_bitset(subpts, eps=EPS)
        Y0s = Y_from_graph(p0s, c0s)
        YEs = Y_from_graph(pEs, cEs)
        As = A_mask(subpts, Y0s, YEs, EPS)
        R = As / Ab if abs(Ab) > 1e-10 else 0.0
        layer_results[r].append(R)
        rs.append(R)

    print(f"  seed {m}: R = [{', '.join(f'{v:.2f}' for v in rs)}]  "
          f"mean={np.mean(rs):.3f}", flush=True)

print(f"\n  Per-residue summary:", flush=True)
all_means = []
for r in range(K):
    arr = np.array(layer_results[r])
    ci = bootstrap_ci(arr)
    all_means.append(arr.mean())
    print(f"    r={r}: mean R = {arr.mean():.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)

mean_ret = np.mean(all_means)
worst_ret = np.min(all_means)
print(f"\n  Overall: mean retention = {mean_ret:.3f}, worst residue = {worst_ret:.3f}", flush=True)
if mean_ret > 0.50 and worst_ret > 0.30:
    print(f"  >>> GOOD: signal robust across all depth residues", flush=True)
elif mean_ret < 0.25:
    print(f"  >>> BAD: signal fragile under structured layer removal", flush=True)
else:
    print(f"  >>> INTERMEDIATE", flush=True)

print(f"  Time: {time.time()-t0:.0f}s", flush=True)
print(f"\nALL REMAINING TESTS COMPLETE.", flush=True)
