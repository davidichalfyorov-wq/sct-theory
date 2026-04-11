"""analytical criterion #2: Test boundary exclusion effect on path_kurtosis.
Compare NO exclusion vs 50%/80% interior-only."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import kurtosis
from scripts.discovery_common import sprinkle_4d, causal_flat, causal_ppwave_quad as causal_ppwave, build_link_graph

def path_kurtosis_with_exclusion(coords, causal_func, frac_interior=1.0, **kw):
    """Compute path_kurtosis, optionally excluding boundary elements."""
    n = len(coords)
    C = causal_func(coords, **kw)
    G = build_link_graph(C)
    
    topo = list(range(n))
    p_down = np.zeros(n, dtype=np.float64)
    for i in topo:
        preds = list(G.predecessors(i))
        p_down[i] = sum(p_down[j] for j in preds) if preds else 1.0
    
    p_up = np.zeros(n, dtype=np.float64)
    for i in reversed(topo):
        succs = list(G.successors(i))
        p_up[i] = sum(p_up[j] for j in succs) if succs else 1.0
    
    P = p_down * p_up
    Y = np.log2(P + 1)
    
    if frac_interior < 1.0:
        # Exclude boundary: keep elements with t in middle fraction
        t = coords[:, 0]
        t_min, t_max = t.min(), t.max()
        t_range = t_max - t_min
        margin = (1.0 - frac_interior) / 2
        t_lo = t_min + margin * t_range
        t_hi = t_max - margin * t_range
        mask = (t >= t_lo) & (t <= t_hi)
        Y = Y[mask]
    
    return kurtosis(Y, fisher=True)

N = 2000
eps = 5.0
M = 20
fracs = [1.0, 0.8, 0.6, 0.5]

print("=" * 70)
print(f"BOUNDARY EXCLUSION TEST: N={N}, eps={eps}, M={M}")
print("=" * 70)

C2 = 1.0 / 1120

for frac in fracs:
    deltas = []
    for m in range(M):
        rng = np.random.default_rng(42000 + m)
        coords = sprinkle_4d(N, 1.0, rng)
        pk_flat = path_kurtosis_with_exclusion(coords, causal_flat, frac_interior=frac)
        pk_curved = path_kurtosis_with_exclusion(coords, causal_ppwave, frac_interior=frac, eps=eps)
        deltas.append(pk_curved - pk_flat)
    
    mean_d = np.mean(deltas)
    se_d = np.std(deltas, ddof=1) / np.sqrt(M)
    
    # Extract A_eff for this exclusion
    N_eff = int(N * frac)  # approximate effective N
    A_ext = mean_d / (eps**2 * np.sqrt(N) * C2)
    
    print(f"\nfrac_interior={frac:.1f} ({int(frac*100)}% kept):")
    print(f"  Δpk = {mean_d:+.6f} ± {se_d:.6f}")
    print(f"  d = {mean_d / se_d:.2f}")
    print(f"  A_eff(vs full N): {A_ext:.4f}")
    print(f"  Amplification vs full: {mean_d / 0.060:.2f}x" if frac < 1.0 else "  [baseline]")

print("\n" + "=" * 70)
