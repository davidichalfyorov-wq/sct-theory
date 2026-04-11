"""
A_eff ensemble at N=10000 — confirms continuum limit convergence.

Uses bitset Hasse from sct_tools.hasse (27x faster than dense).
M=25 seeds, eps=2.0 and 3.0 (perturbative window).
Seed formula: 1000*N + 100*int(eps*10) + m (same as N=500-5000 runs).

Expected: A_eff consistent with A_inf = 0.060 +/- 0.006
"""
import sys, time, json
import numpy as np
sys.path.insert(0, 'analysis')
from sct_tools.hasse import crn_trial_bitset

N = 10000
EPS_VALUES = [2.0, 3.0]
M = 25
T = 1.0
C2 = T**4 / 1120

results = {}
t_start = time.time()

for eps in EPS_VALUES:
    print(f"\n=== N={N}, eps={eps}, M={M} ===", flush=True)
    dks = []
    for m in range(M):
        seed = 1000 * N + 100 * int(eps * 10) + m
        t0 = time.time()
        dk = crn_trial_bitset(N, eps, seed, T)
        dt = time.time() - t0
        dks.append(dk)
        print(f"  m={m:2d}  dk={dk:+.6f}  ({dt:.1f}s)", flush=True)

    dks = np.array(dks)
    mn = dks.mean()
    se = dks.std(ddof=0) / np.sqrt(M)
    A_eff = mn / (eps**2 * np.sqrt(N) * C2)
    A_se = se / (eps**2 * np.sqrt(N) * C2)
    d = mn / se if se > 1e-15 else 0.0

    results[str(eps)] = {
        'mean': float(mn),
        'se': float(se),
        'A_eff': float(A_eff),
        'A_se': float(A_se),
        'd': float(d),
        'dks': dks.tolist(),
    }

    print(f"  RESULT: dk_mean={mn:+.6f} +/- {se:.6f}")
    print(f"  A_eff = {A_eff:.4f} +/- {A_se:.4f}")
    print(f"  Cohen's d = {d:.2f}")

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"Total time: {elapsed:.0f}s = {elapsed/60:.1f}min")
print(f"\n=== SUMMARY (N={N}) ===")
for eps_s, r in results.items():
    eps = float(eps_s)
    print(f"  eps={eps}: A_eff = {r['A_eff']:.4f} +/- {r['A_se']:.4f}, d={r['d']:.2f}")

# Compare with A_inf = 0.060 +/- 0.006
print(f"\nA_inf (N=2000-5000) = 0.060 +/- 0.006")
for eps_s, r in results.items():
    z = abs(r['A_eff'] - 0.060) / np.sqrt(r['A_se']**2 + 0.006**2)
    print(f"  eps={eps_s}: |A_eff - A_inf|/sigma_combined = {z:.2f} ({'CONSISTENT' if z < 2 else 'TENSION'})")

# Save
out = {
    'N': N, 'M': M, 'T': T, 'C2': C2,
    'eps_values': EPS_VALUES,
    'results': results,
    'elapsed_s': elapsed,
    'A_inf_reference': {'value': 0.060, 'se': 0.006, 'source': 'N=2000-5000 ensemble'},
}
outpath = 'analysis/discovery_runs/run_001/aeff_n10000.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}")
