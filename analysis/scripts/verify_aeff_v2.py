"""Verify A_eff = 0.065: Dpk = A_eff * eps^2 * sqrt(N) * C2
Also: boundary exclusion test + graph adversary test."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c): os.add_dll_directory(_c)

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy import sparse as sp
from scripts.discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad, build_link_graph


def path_counts(link_adj, n):
    """Compute p_down, p_up from sparse link adjacency (CSR).
    link_adj[i,j]=1 means i->j is a link (i < j in topological order)."""
    # Elements already sorted by time (index = topological order)
    # p_down[i] = sum of p_down[j] for all j->i links, or 1 if source
    # link_adj is symmetric adjacency; directed version: j->i if j < i and link_adj[j,i]=1

    p_down = np.zeros(n, dtype=np.float64)
    p_up = np.zeros(n, dtype=np.float64)

    # Convert to COO for fast iteration
    L = sp.triu(link_adj).tocoo()  # upper triangle: i < j means i->j

    # Build predecessor/successor lists from COO
    preds = [[] for _ in range(n)]  # preds[j] = list of i where i->j
    succs = [[] for _ in range(n)]  # succs[i] = list of j where i->j
    for i, j in zip(L.row, L.col):
        preds[j].append(i)
        succs[i].append(j)

    # Forward pass: p_down
    for i in range(n):
        if not preds[i]:
            p_down[i] = 1.0
        else:
            p_down[i] = sum(p_down[j] for j in preds[i])

    # Backward pass: p_up
    for i in range(n - 1, -1, -1):
        if not succs[i]:
            p_up[i] = 1.0
        else:
            p_up[i] = sum(p_up[j] for j in succs[i])

    return p_down, p_up


def compute_pk(coords, causal_func, frac_interior=1.0, **kw):
    """Compute path_kurtosis with optional boundary exclusion."""
    n = len(coords)
    C = causal_func(coords, **kw)
    A = build_link_graph(C)
    pd, pu = path_counts(A, n)
    P = pd * pu
    Y = np.log2(P + 1)

    if frac_interior < 1.0:
        t = coords[:, 0]
        t_min, t_max = t.min(), t.max()
        margin = (1.0 - frac_interior) / 2
        t_lo = t_min + margin * (t_max - t_min)
        t_hi = t_max - margin * (t_max - t_min)
        mask = (t >= t_lo) & (t <= t_hi)
        Y = Y[mask]

    return scipy_kurtosis(Y, fisher=True)


# ═══════════════════════════════════════════════════════════════
# TEST 1: A_eff VERIFICATION
# ═══════════════════════════════════════════════════════════════
C2 = 1.0 / 1120

configs = [
    (2000, 5.0, 20),
    (2000, 10.0, 15),
    (5000, 5.0, 10),
]

print("=" * 70)
print("TEST 1: A_eff VERIFICATION — Dpk = A_eff * eps^2 * sqrt(N) * C2")
print(f"C2 = {C2:.6e}")
print("=" * 70)

results_aeff = []
for N, eps, M in configs:
    deltas = []
    for m in range(M):
        rng = np.random.default_rng(42000 + m)
        coords = sprinkle_4d(N, 1.0, rng)
        pk_flat = compute_pk(coords, causal_flat)
        pk_curved = compute_pk(coords, causal_ppwave_quad, eps=eps)
        deltas.append(pk_curved - pk_flat)

    mean_d = np.mean(deltas)
    se_d = np.std(deltas, ddof=1) / np.sqrt(M)
    predicted = 0.065 * eps**2 * np.sqrt(N) * C2
    A_ext = mean_d / (eps**2 * np.sqrt(N) * C2)
    ratio = mean_d / predicted

    print(f"\nN={N}, eps={eps}, M={M}")
    print(f"  Observed: {mean_d:+.6f} +/- {se_d:.6f}")
    print(f"  Predicted (A=0.065): {predicted:+.6f}")
    print(f"  Ratio: {ratio:.3f}, A_eff extracted: {A_ext:.4f}")
    results_aeff.append((N, eps, mean_d, se_d, predicted, ratio, A_ext))

print("\nSUMMARY:")
A_vals = [r[6] for r in results_aeff]
print(f"A_eff = {np.mean(A_vals):.4f} +/- {np.std(A_vals, ddof=1):.4f}")

# ═══════════════════════════════════════════════════════════════
# TEST 2: BOUNDARY EXCLUSION (analytical criterion #2)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: BOUNDARY EXCLUSION — N=2000, eps=5, M=20")
print("=" * 70)

N, eps, M = 2000, 5.0, 20
fracs = [1.0, 0.8, 0.6, 0.5]
baseline = None

for frac in fracs:
    deltas = []
    for m in range(M):
        rng = np.random.default_rng(42000 + m)
        coords = sprinkle_4d(N, 1.0, rng)
        pk_f = compute_pk(coords, causal_flat, frac_interior=frac)
        pk_c = compute_pk(coords, causal_ppwave_quad, frac_interior=frac, eps=eps)
        deltas.append(pk_c - pk_f)

    mean_d = np.mean(deltas)
    se_d = np.std(deltas, ddof=1) / np.sqrt(M)
    if baseline is None:
        baseline = mean_d

    amplification = mean_d / baseline if abs(baseline) > 1e-12 else 0
    print(f"\n{int(frac*100)}% interior: Dpk = {mean_d:+.6f} +/- {se_d:.6f}, "
          f"d={mean_d/se_d:.1f}, amplification={amplification:.2f}x")

# ═══════════════════════════════════════════════════════════════
# TEST 3: GRAPH ADVERSARY (analytical criterion #5)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: GRAPH ADVERSARY — degree-preserving rewiring")
print("N=2000, eps=5, M=10, rewired=5 per trial")
print("=" * 70)

import networkx as nx

N, eps, M, n_rewired = 2000, 5.0, 10, 5

def sparse_to_nx_dag(A_sp):
    """Convert sparse upper-triangular link adj to directed nx graph."""
    L = sp.triu(A_sp).tocoo()
    G = nx.DiGraph()
    G.add_nodes_from(range(A_sp.shape[0]))
    for i, j in zip(L.row, L.col):
        G.add_edge(i, j)
    return G

def nx_dag_pk(G):
    """Compute path_kurtosis from nx DAG."""
    n = G.number_of_nodes()
    topo = list(nx.topological_sort(G))
    pd = {}
    for v in topo:
        preds = list(G.predecessors(v))
        pd[v] = sum(pd[p] for p in preds) if preds else 1.0
    pu = {}
    for v in reversed(topo):
        succs = list(G.successors(v))
        pu[v] = sum(pu[s] for s in succs) if succs else 1.0
    nodes = sorted(G.nodes())
    P = np.array([pd[v] * pu[v] for v in nodes])
    Y = np.log2(P + 1)
    return scipy_kurtosis(Y, fisher=True)

def degree_preserving_rewire(G, n_swaps=None, seed=999):
    G2 = G.copy()
    edges = list(G2.edges())
    if n_swaps is None:
        n_swaps = len(edges) * 3
    rng = np.random.default_rng(seed)
    swapped = 0
    for _ in range(n_swaps * 20):
        if swapped >= n_swaps:
            break
        i1, i2 = rng.choice(len(edges), 2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue
        if G2.has_edge(a, d) or G2.has_edge(c, b):
            continue
        G2.remove_edge(a, b)
        G2.remove_edge(c, d)
        G2.add_edge(a, d)
        G2.add_edge(c, b)
        if nx.is_directed_acyclic_graph(G2):
            edges[i1] = (a, d)
            edges[i2] = (c, b)
            swapped += 1
        else:
            G2.remove_edge(a, d)
            G2.remove_edge(c, b)
            G2.add_edge(a, b)
            G2.add_edge(c, d)
    return G2, swapped

geo_deltas = []
rew_deltas = []

for m in range(M):
    rng = np.random.default_rng(42000 + m)
    coords = sprinkle_4d(N, 1.0, rng)

    A_flat = build_link_graph(causal_flat(coords))
    G_flat = sparse_to_nx_dag(A_flat)
    pk_flat = nx_dag_pk(G_flat)

    A_curved = build_link_graph(causal_ppwave_quad(coords, eps=eps))
    G_curved = sparse_to_nx_dag(A_curved)
    pk_curved = nx_dag_pk(G_curved)
    geo_deltas.append(pk_curved - pk_flat)

    for r in range(n_rewired):
        G_rew, nsw = degree_preserving_rewire(G_curved, seed=m * 100 + r)
        pk_rew = nx_dag_pk(G_rew)
        rew_deltas.append(pk_rew - pk_flat)

    print(f"  Trial {m+1}/{M}: geo_delta={pk_curved - pk_flat:+.4f}, swaps={nsw}")

geo_mean = np.mean(geo_deltas)
geo_se = np.std(geo_deltas, ddof=1) / np.sqrt(M)
rew_mean = np.mean(rew_deltas)
rew_se = np.std(rew_deltas, ddof=1) / np.sqrt(len(rew_deltas))

print(f"\nGeometric (true pp-wave): {geo_mean:+.6f} +/- {geo_se:.6f}, d={geo_mean/geo_se:.1f}")
print(f"Rewired (degree-preserved): {rew_mean:+.6f} +/- {rew_se:.6f}, d={rew_mean/rew_se:.1f}")
ratio = rew_mean / geo_mean if abs(geo_mean) > 1e-10 else float('inf')
print(f"Ratio rewired/geometric: {ratio:.3f}")

if abs(rew_mean) < 2 * rew_se:
    print("\n>>> PASS: Rewiring kills signal. path_kurtosis is GEOMETRY-SENSITIVE.")
elif abs(ratio) < 0.3:
    print(f"\n>>> PASS: Rewiring reduces signal to {abs(ratio):.1%}. Mostly geometry-sensitive.")
else:
    print(f"\n>>> AMBIGUOUS: Rewiring preserves {abs(ratio):.1%} of signal.")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
