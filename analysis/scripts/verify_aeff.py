"""Verify A_eff = 0.065 from independent analysis derivation: Dpk = A_eff * eps^2 * sqrt(N) * C2"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# GPU
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c): os.add_dll_directory(_c)

import numpy as np
from scripts.discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad as causal_ppwave, build_link_graph

def path_kurtosis(coords, causal_func, **kw):
    n = len(coords)
    C = causal_func(coords, **kw)
    G = build_link_graph(C)
    # p_down
    topo = list(range(n))  # already sorted by time
    p_down = np.zeros(n, dtype=np.float64)
    for i in topo:
        preds = list(G.predecessors(i))
        if not preds:
            p_down[i] = 1.0
        else:
            p_down[i] = sum(p_down[j] for j in preds)
    # p_up
    p_up = np.zeros(n, dtype=np.float64)
    for i in reversed(topo):
        succs = list(G.successors(i))
        if not succs:
            p_up[i] = 1.0
        else:
            p_up[i] = sum(p_up[j] for j in succs)
    P = p_down * p_up
    Y = np.log2(P + 1)
    from scipy.stats import kurtosis
    return kurtosis(Y, fisher=True)

C2 = 1.0 / 1120  # T=1

# Test at multiple (N, eps) with M trials
configs = [
    (2000, 5.0, 20),
    (2000, 10.0, 15),
    (5000, 5.0, 10),
    (5000, 10.0, 8),
]

print("=" * 70)
print("A_eff VERIFICATION: Dpk = A_eff * eps^2 * sqrt(N) * C2")
print(f"C2 = {C2:.6e}")
print("=" * 70)

results = []
for N, eps, M in configs:
    deltas = []
    for m in range(M):
        rng = np.random.default_rng(42000 + m)
        coords = sprinkle_4d(N, 1.0, rng)
        pk_flat = path_kurtosis(coords, causal_flat)
        pk_curved = path_kurtosis(coords, causal_ppwave, eps=eps)
        deltas.append(pk_curved - pk_flat)
    
    mean_d = np.mean(deltas)
    se_d = np.std(deltas, ddof=1) / np.sqrt(M)
    
    predicted = 0.065 * eps**2 * np.sqrt(N) * C2
    A_extracted = mean_d / (eps**2 * np.sqrt(N) * C2) if abs(eps**2 * np.sqrt(N) * C2) > 0 else 0
    ratio = mean_d / predicted if abs(predicted) > 0 else 0
    
    print(f"\nN={N}, eps={eps}, M={M}")
    print(f"  Observed: {mean_d:+.6f} +/- {se_d:.6f}")
    print(f"  Predicted (A=0.065): {predicted:+.6f}")
    print(f"  Ratio obs/pred: {ratio:.3f}")
    print(f"  A_eff extracted: {A_extracted:.4f}")
    results.append((N, eps, mean_d, se_d, predicted, ratio, A_extracted))

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print(f"{'N':>6} {'eps':>5} {'obs':>10} {'pred':>10} {'ratio':>7} {'A_eff':>7}")
for N, eps, obs, se, pred, ratio, A in results:
    print(f"{N:>6} {eps:>5.1f} {obs:>+10.6f} {pred:>+10.6f} {ratio:>7.3f} {A:>7.4f}")

A_vals = [r[6] for r in results]
print(f"\nA_eff mean: {np.mean(A_vals):.4f} +/- {np.std(A_vals, ddof=1):.4f}")
print("=" * 70)
