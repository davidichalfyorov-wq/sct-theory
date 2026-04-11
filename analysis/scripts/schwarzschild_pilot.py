"""
Schwarzschild pilot: criterion #4 for E4.
N=5000, M_sch = {0.005, 0.01, 0.02}, r_min=0.10, M_seeds=10.
Shapiro first-order causal relation, shell sprinkling.

Expected: Δpk ~ M (linear), NOT M² (quadratic like pp-wave).
"""
import sys, time, json, numpy as np
sys.path.insert(0, 'analysis')
from sct_tools.hasse import crn_trial_schwarzschild

N = 5000
M_VALUES = [0.005, 0.01, 0.02]
M_SEEDS = 10
T = 1.0
R_MIN = 0.10

# C_1,rms from independent analysis formula: sqrt(8)*(T-2a)/(T²+4Ta+12a²) at a=0.10, T=1
a = R_MIN
C1_rms = np.sqrt(8) * (T - 2*a) / (T**2 + 4*T*a + 12*a**2)
print(f"C_1,rms(T={T}, r_min={R_MIN}) = {C1_rms:.4f}", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"SCHWARZSCHILD PILOT: N={N}, r_min={R_MIN}, M_seeds={M_SEEDS}", flush=True)
print(f"{'='*70}", flush=True)

results = {}
t_start = time.time()

for M in M_VALUES:
    print(f"\n--- M = {M} ---", flush=True)
    dks = []
    for m in range(M_SEEDS):
        seed = 2000000 + int(M * 10000) + m
        t0 = time.time()
        dk = crn_trial_schwarzschild(N, M, seed, T=T, r_min=R_MIN)
        dt = time.time() - t0
        dks.append(dk)
        print(f"  m={m:2d}: dk={dk:+.6f} ({dt:.1f}s)", flush=True)

    dks = np.array(dks)
    mn = dks.mean()
    se = dks.std(ddof=0) / np.sqrt(M_SEEDS)
    d = mn / se if se > 1e-15 else 0.0

    # B_eff = dk / (M * N^{1/4} * C_1,rms)
    B_eff = mn / (M * N**0.25 * C1_rms)
    B_se = se / (M * N**0.25 * C1_rms)

    # Also A_eff (quadratic normalization for comparison)
    C2_raw = 24.0 / (T**2 + 4*T*a + 12*a**2)
    A_eff = mn / (M**2 * np.sqrt(N) * C2_raw)
    A_se = se / (M**2 * np.sqrt(N) * C2_raw)

    results[str(M)] = {
        'mean': float(mn), 'se': float(se), 'd': float(d),
        'B_eff': float(B_eff), 'B_se': float(B_se),
        'A_eff': float(A_eff), 'A_se': float(A_se),
        'dks': dks.tolist(),
    }

    print(f"  dk_mean = {mn:+.6f} ± {se:.6f}, d = {d:.2f}", flush=True)
    print(f"  B_eff (linear M) = {B_eff:.4f} ± {B_se:.4f}", flush=True)
    print(f"  A_eff (quadratic M²) = {A_eff:.4f} ± {A_se:.4f}", flush=True)

elapsed = time.time() - t_start
print(f"\n{'='*70}", flush=True)
print(f"Total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

# Test: linear vs quadratic
print(f"\n=== LINEAR vs QUADRATIC TEST ===", flush=True)
Ms = np.array(M_VALUES)
dks_mean = np.array([results[str(M)]['mean'] for M in M_VALUES])
dks_se = np.array([results[str(M)]['se'] for M in M_VALUES])

# If linear: dk/M should be constant
print(f"\ndk/M (should be constant if linear):", flush=True)
for M_val in M_VALUES:
    r = results[str(M_val)]
    ratio = r['mean'] / M_val
    print(f"  M={M_val}: dk/M = {ratio:.4f}", flush=True)

# If quadratic: dk/M² should be constant
print(f"\ndk/M² (should be constant if quadratic):", flush=True)
for M_val in M_VALUES:
    r = results[str(M_val)]
    ratio = r['mean'] / M_val**2
    print(f"  M={M_val}: dk/M² = {ratio:.2f}", flush=True)

# B_eff consistency (linear normalization)
print(f"\nB_eff consistency:", flush=True)
for M_val in M_VALUES:
    r = results[str(M_val)]
    print(f"  M={M_val}: B_eff = {r['B_eff']:.4f} ± {r['B_se']:.4f}", flush=True)

# Save
out = {
    'N': N, 'T': T, 'r_min': R_MIN, 'M_seeds': M_SEEDS,
    'C1_rms': float(C1_rms),
    'M_values': M_VALUES,
    'results': results,
    'elapsed_s': elapsed,
}
outpath = 'analysis/discovery_runs/run_001/schwarzschild_pilot.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}", flush=True)
