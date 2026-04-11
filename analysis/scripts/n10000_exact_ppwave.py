"""N=10000 exact pp-wave ensemble. Replaces midpoint data."""
import sys, time, json, numpy as np
sys.path.insert(0, 'analysis')
from sct_tools.hasse import crn_trial_bitset

N = 10000
EPS_VALUES = [2.0, 3.0]
M = 20
T = 1.0
C2 = T**4 / 1120

print(f"N={N}, M={M}, eps={EPS_VALUES}, exact=True", flush=True)
results = {}

for eps in EPS_VALUES:
    print(f"\n=== eps={eps} ===", flush=True)
    dks = []
    for m in range(M):
        seed = 1000 * N + 100 * int(eps * 10) + m
        t0 = time.time()
        dk = crn_trial_bitset(N, eps, seed, T)
        dt = time.time() - t0
        dks.append(dk)
        print(f"  m={m:2d}: dk={dk:+.6f} ({dt:.1f}s)", flush=True)

    dks = np.array(dks)
    mn = dks.mean()
    se = dks.std(ddof=0) / np.sqrt(M)
    A_eff = mn / (eps**2 * np.sqrt(N) * C2)
    A_se = se / (eps**2 * np.sqrt(N) * C2)
    d = mn / se if se > 1e-15 else 0.0

    results[str(eps)] = {
        'mean': float(mn), 'se': float(se),
        'A_eff': float(A_eff), 'A_se': float(A_se),
        'd': float(d), 'dks': dks.tolist(),
    }
    print(f"  A_eff = {A_eff:.4f} +/- {A_se:.4f}, d = {d:.2f}", flush=True)

print(f"\n=== SUMMARY ===", flush=True)
for e, r in results.items():
    print(f"  eps={e}: A_eff={r['A_eff']:.4f}+/-{r['A_se']:.4f}, d={r['d']:.2f}", flush=True)

out = {'N': N, 'M': M, 'T': T, 'C2': C2, 'exact': True, 'results': results}
with open('analysis/discovery_runs/run_001/aeff_n10000_exact.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved.", flush=True)
