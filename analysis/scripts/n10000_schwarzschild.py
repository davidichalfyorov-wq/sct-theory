"""N=10000 Schwarzschild confirmatory. Confirms B_eff and linear scaling."""
import sys, time, json, numpy as np
sys.path.insert(0, 'analysis')
from sct_tools.hasse import crn_trial_schwarzschild

N = 10000
M_VALUES = [0.005, 0.01]
M_SEEDS = 15
T = 1.0
R_MIN = 0.10

a = R_MIN
C1_rms = np.sqrt(8) * (T - 2*a) / (T**2 + 4*T*a + 12*a**2)
print(f"N={N}, M_seeds={M_SEEDS}, r_min={R_MIN}, C1_rms={C1_rms:.4f}", flush=True)

results = {}

for M_sch in M_VALUES:
    print(f"\n=== M = {M_sch} ===", flush=True)
    dks = []
    for m in range(M_SEEDS):
        seed = 3000000 + int(M_sch * 10000) + m
        t0 = time.time()
        dk = crn_trial_schwarzschild(N, M_sch, seed, T=T, r_min=R_MIN)
        dt = time.time() - t0
        dks.append(dk)
        print(f"  m={m:2d}: dk={dk:+.6f} ({dt:.1f}s)", flush=True)

    dks = np.array(dks)
    mn = dks.mean()
    se = dks.std(ddof=0) / np.sqrt(M_SEEDS)
    d = mn / se if se > 1e-15 else 0.0
    B_eff = mn / (M_sch * N**0.25 * C1_rms)
    B_se = se / (M_sch * N**0.25 * C1_rms)

    results[str(M_sch)] = {
        'mean': float(mn), 'se': float(se), 'd': float(d),
        'B_eff': float(B_eff), 'B_se': float(B_se),
        'dks': dks.tolist(),
    }
    print(f"  dk={mn:+.6f}+/-{se:.6f}, d={d:.2f}", flush=True)
    print(f"  B_eff={B_eff:.4f}+/-{B_se:.4f}", flush=True)

# N-scaling test: compare with N=5000 pilot
print(f"\n=== N-SCALING: B_eff at N=5000 vs N=10000 ===", flush=True)
B5k = {'0.005': -0.7506, '0.01': -0.7422}  # from pilot
for M_s in ['0.005', '0.01']:
    if M_s in results:
        print(f"  M={M_s}: B_eff(5k)={B5k[M_s]:.4f}, B_eff(10k)={results[M_s]['B_eff']:.4f}", flush=True)

out = {'N': N, 'M_seeds': M_SEEDS, 'T': T, 'r_min': R_MIN,
       'C1_rms': float(C1_rms), 'results': results}
with open('analysis/discovery_runs/run_001/schwarzschild_n10000.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved.", flush=True)
