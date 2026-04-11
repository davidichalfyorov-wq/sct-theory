"""analytical criterion #5: Graph-theoretic adversary.
Preserve degree sequence but randomize geometry.
If path_kurtosis survives ONLY geometric ensemble, it's manifold-sensitive."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import kurtosis
from scripts.discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad as causal_ppwave, build_link_graph
import networkx as nx

def path_counts_from_graph(G, n):
    """Compute p_down, p_up from a DAG."""
    topo = list(nx.topological_sort(G))
    
    p_down = {}
    for v in topo:
        preds = list(G.predecessors(v))
        p_down[v] = sum(p_down[p] for p in preds) if preds else 1.0
    
    p_up = {}
    for v in reversed(topo):
        succs = list(G.successors(v))
        p_up[v] = sum(p_up[s] for s in succs) if succs else 1.0
    
    nodes = sorted(G.nodes())
    P = np.array([p_down[v] * p_up[v] for v in nodes])
    Y = np.log2(P + 1)
    return kurtosis(Y, fisher=True)

def degree_preserving_rewire(G, n_swaps=None):
    """Rewire DAG edges preserving in/out degree sequence.
    Swap: if a->b and c->d exist, and a->d, c->b don't exist,
    and the result is still a DAG, replace with a->d, c->b."""
    G2 = G.copy()
    edges = list(G2.edges())
    if n_swaps is None:
        n_swaps = len(edges) * 5
    
    rng = np.random.default_rng(12345)
    swapped = 0
    attempts = 0
    max_attempts = n_swaps * 20
    
    while swapped < n_swaps and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = rng.choice(len(edges), 2, replace=False)
        a, b = edges[idx1]
        c, d = edges[idx2]
        
        if a == c or b == d or a == d or c == b:
            continue
        if G2.has_edge(a, d) or G2.has_edge(c, b):
            continue
        
        # Try swap
        G2.remove_edge(a, b)
        G2.remove_edge(c, d)
        G2.add_edge(a, d)
        G2.add_edge(c, b)
        
        # Check still DAG
        if nx.is_directed_acyclic_graph(G2):
            edges[idx1] = (a, d)
            edges[idx2] = (c, b)
            swapped += 1
        else:
            # Revert
            G2.remove_edge(a, d)
            G2.remove_edge(c, b)
            G2.add_edge(a, b)
            G2.add_edge(c, d)
    
    return G2, swapped

N = 2000
eps = 5.0
M = 15
n_rewired = 10  # rewired versions per sprinkling

print("=" * 70)
print(f"GRAPH ADVERSARY TEST: N={N}, eps={eps}, M={M}, rewired={n_rewired}")
print("Preserve degree sequence, randomize geometry")
print("=" * 70)

geo_deltas = []
rewired_deltas = []

for m in range(M):
    rng = np.random.default_rng(42000 + m)
    coords = sprinkle_4d(N, 1.0, rng)
    
    C_flat = causal_flat(coords)
    G_flat = build_link_graph(C_flat)
    pk_flat = path_counts_from_graph(G_flat, N)
    
    C_curved = causal_ppwave(coords, eps=eps)
    G_curved = build_link_graph(C_curved)
    pk_curved = path_counts_from_graph(G_curved, N)
    
    geo_delta = pk_curved - pk_flat
    geo_deltas.append(geo_delta)
    
    # Now rewire the CURVED graph, keeping degree sequence
    for r in range(n_rewired):
        G_rewired, n_sw = degree_preserving_rewire(G_curved)
        pk_rewired = path_counts_from_graph(G_rewired, N)
        rewired_deltas.append(pk_rewired - pk_flat)
    
    if (m + 1) % 5 == 0:
        print(f"  Trial {m+1}/{M} done (last swap count: {n_sw})")

geo_mean = np.mean(geo_deltas)
geo_se = np.std(geo_deltas, ddof=1) / np.sqrt(M)
rew_mean = np.mean(rewired_deltas)
rew_se = np.std(rewired_deltas, ddof=1) / np.sqrt(len(rewired_deltas))

print(f"\nGeometric (true pp-wave):")
print(f"  Δpk = {geo_mean:+.6f} ± {geo_se:.6f}, d = {geo_mean/geo_se:.2f}")
print(f"\nRewired (degree-preserved, geometry destroyed):")
print(f"  Δpk = {rew_mean:+.6f} ± {rew_se:.6f}, d = {rew_mean/rew_se:.2f}")
print(f"\nRatio rewired/geometric: {rew_mean/geo_mean:.3f}" if abs(geo_mean) > 1e-10 else "")

if abs(rew_mean) < 2 * rew_se:
    print("\n>>> PASS: Rewiring kills signal. path_kurtosis is GEOMETRY-SENSITIVE.")
else:
    ratio = abs(rew_mean / geo_mean)
    if ratio < 0.3:
        print(f"\n>>> PASS: Rewiring reduces signal to {ratio:.1%}. Mostly geometry-sensitive.")
    else:
        print(f"\n>>> FAIL: Rewiring preserves {ratio:.1%} of signal. Degree-driven, not geometry.")

print("=" * 70)
