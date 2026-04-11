"""
A_eff ensemble measurement — CRN, multiple N, ε→0 extrapolation.

Protocol (per independent analysis recommendation):
1. For each N: CRN ensemble with M≥30 seeds
2. Multiple ε values (small enough for plateau)
3. Extract A_eff = Δκ / (ε² × N^{1/2} × C₂)
4. For each N: extrapolate ε→0
5. Then study N→∞

Uses GPU for N≥2000 causal matrix construction.

Author: David Alfyorov
"""

import numpy as np
import time
import json
import os
import sys
sys.path.insert(0, 'analysis')

# GPU disabled for background stability — CPU-only
cp = np
print("Mode: CPU-only (GPU disabled for background stability)")

# ── Configuration ──
N_VALUES = [500, 1000, 2000, 3000, 5000]
EPS_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
M_SEEDS = 30  # ensemble size
USE_GPU = False  # Force CPU — GPU hangs on background tasks
T = 1.0
C2_EXACT = T**4 / 1120

OUT_DIR = "analysis/discovery_runs/run_001"
os.makedirs(OUT_DIR, exist_ok=True)

def sprinkle_diamond(n, T=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pts = []
    while len(pts) < n:
        batch = rng.uniform(-T/2, T/2, (n*10, 4))
        r = np.sqrt(batch[:,1]**2 + batch[:,2]**2 + batch[:,3]**2)
        mask = np.abs(batch[:,0]) + r < T/2
        pts.extend(batch[mask].tolist())
    return np.array(pts[:n])

def build_causal_gpu(coords, eps=0.0):
    """Build causal matrix on GPU. Returns numpy array."""
    n = len(coords)
    if USE_GPU and n >= 500:
        c = cp.asarray(coords, dtype=cp.float32)
        dt = c[:,0:1] - c[:,0:1].T
        ds2 = dt**2
        for k in range(1, 4):
            ds2 = ds2 - (c[:,k:k+1] - c[:,k:k+1].T)**2
        if abs(eps) > 1e-12:
            xm = (c[:,1:2] + c[:,1:2].T) / 2
            ym = (c[:,2:3] + c[:,2:3].T) / 2
            du = dt + (c[:,3:4] - c[:,3:4].T)
            correction = eps/2 * (xm**2 - ym**2) * du**2
            ds2 = ds2 - correction
        C = ((dt > 0) & (ds2 > 0)).astype(cp.int8)
        return cp.asnumpy(C)
    else:
        dt = coords[:,0:1] - coords[:,0:1].T
        ds2 = dt**2
        for k in range(1, 4):
            ds2 -= (coords[:,k:k+1] - coords[:,k:k+1].T)**2
        if abs(eps) > 1e-12:
            xm = (coords[:,1:2] + coords[:,1:2].T) / 2
            ym = (coords[:,2:3] + coords[:,2:3].T) / 2
            du = dt + (coords[:,3:4] - coords[:,3:4].T)
            correction = eps/2 * (xm**2 - ym**2) * du**2
            ds2 -= correction
        return ((dt > 0) & (ds2 > 0)).astype(np.int8)

def hasse_fast(C):
    C2 = (C @ C > 0).astype(np.int8)
    return C * (1 - C2)

def compute_path_counts(L):
    from scipy.sparse import csr_matrix
    n = L.shape[0]
    p_down = np.zeros(n, dtype=np.float64)
    p_up = np.zeros(n, dtype=np.float64)
    Ls = csr_matrix(L)
    Lt = csr_matrix(L.T)
    for i in range(n):
        preds = Ls[i].indices
        p_down[i] = sum(p_down[j] for j in preds) if len(preds) > 0 else 1.0
    for i in range(n-1, -1, -1):
        succs = Lt[i].indices
        p_up[i] = sum(p_up[j] for j in succs) if len(succs) > 0 else 1.0
    return p_down, p_up

def path_kurtosis(coords, L):
    p_d, p_u = compute_path_counts(L)
    Y = np.log2(p_d * p_u + 1)
    X = Y - Y.mean()
    s2 = np.var(Y)
    if s2 < 1e-12:
        return 0.0
    return float(np.mean(X**4) / s2**2 - 3)

def run_one_trial(N, eps, seed):
    """Single CRN trial: same points, flat vs pp-wave."""
    rng = np.random.default_rng(seed)
    coords = sprinkle_diamond(N, T, rng)
    order = np.argsort(coords[:, 0])
    coords = coords[order]

    C_flat = build_causal_gpu(coords, 0.0)
    L_flat = hasse_fast(C_flat)
    pk_flat = path_kurtosis(coords, L_flat)

    C_pp = build_causal_gpu(coords, eps)
    L_pp = hasse_fast(C_pp)
    pk_pp = path_kurtosis(coords, L_pp)

    dk = pk_pp - pk_flat
    return dk, pk_flat, pk_pp

# ── Main loop ──
print("=" * 70)
print(f"A_eff ENSEMBLE — N={N_VALUES}, ε={EPS_VALUES}, M={M_SEEDS}")
print(f"C₂(exact) = {C2_EXACT:.8f}")
print("=" * 70)

results = {}

for N in N_VALUES:
    t_start = time.time()
    results[N] = {}
    print(f"\n{'─'*60}")
    print(f"N = {N}")
    print(f"{'─'*60}")

    for eps in EPS_VALUES:
        dks = []
        t0 = time.time()
        for m in range(M_SEEDS):
            seed = 1000 * N + 100 * int(eps * 10) + m
            dk, pk_f, pk_p = run_one_trial(N, eps, seed)
            dks.append(dk)

        dks = np.array(dks)
        dk_mean = dks.mean()
        dk_se = dks.std() / np.sqrt(M_SEEDS)

        A_eff = dk_mean / (eps**2 * N**0.5 * C2_EXACT)
        A_eff_se = dk_se / (eps**2 * N**0.5 * C2_EXACT)

        elapsed = time.time() - t0
        d_stat = dk_mean / dk_se if dk_se > 1e-15 else 0

        results[N][eps] = {
            'dk_mean': float(dk_mean),
            'dk_se': float(dk_se),
            'A_eff': float(A_eff),
            'A_eff_se': float(A_eff_se),
            'd': float(d_stat),
            'M': M_SEEDS
        }

        sig = "***" if abs(d_stat) > 3 else "**" if abs(d_stat) > 2 else "*" if abs(d_stat) > 1 else ""
        print(f"  ε={eps:4.1f}: Δκ={dk_mean:+.6f}±{dk_se:.6f}  "
              f"A_eff={A_eff:.4f}±{A_eff_se:.4f}  d={d_stat:+.1f}{sig}  "
              f"({elapsed:.1f}s)")

    total = time.time() - t_start
    print(f"  Total N={N}: {total:.0f}s")

# ── Summary table ──
print(f"\n{'='*70}")
print("SUMMARY: A_eff(N, ε)")
print(f"{'='*70}")
header = f"{'N':>6} |"
for eps in EPS_VALUES:
    header += f" ε={eps:4.1f}        |"
print(header)
print("-" * len(header))

for N in N_VALUES:
    row = f"{N:6d} |"
    for eps in EPS_VALUES:
        r = results[N][eps]
        row += f" {r['A_eff']:6.4f}±{r['A_eff_se']:.4f} |"
    print(row)

# ── ε→0 extrapolation for each N ──
print(f"\n{'='*70}")
print("ε→0 EXTRAPOLATION (using smallest 2-3 ε values)")
print(f"{'='*70}")
for N in N_VALUES:
    # Use ε=0.5, 1.0, 2.0 for extrapolation
    small_eps = [e for e in [0.5, 1.0, 2.0] if e in results[N]]
    aeffs = [results[N][e]['A_eff'] for e in small_eps]
    ses = [results[N][e]['A_eff_se'] for e in small_eps]
    # Weighted mean of small-ε values
    if len(aeffs) > 0 and all(s > 0 for s in ses):
        w = [1/s**2 for s in ses]
        A_extrap = sum(a*ww for a, ww in zip(aeffs, w)) / sum(w)
        se_extrap = 1 / np.sqrt(sum(w))
    else:
        A_extrap = np.mean(aeffs) if aeffs else 0
        se_extrap = 0
    print(f"  N={N:5d}: A_eff(ε→0) = {A_extrap:.4f} ± {se_extrap:.4f}")

# Save results
out_path = os.path.join(OUT_DIR, "aeff_ensemble.json")
# Convert for JSON
json_results = {}
for N in results:
    json_results[str(N)] = {}
    for eps in results[N]:
        json_results[str(N)][str(eps)] = results[N][eps]

with open(out_path, 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"\nSaved to {out_path}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
