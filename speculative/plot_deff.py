import numpy as np

def generate_causal_set_2d(N):
    np.random.seed(42)
    u = np.random.rand(N)
    v = np.random.rand(N)
    t = u + v
    idx = np.argsort(t)
    u = u[idx]
    v = v[idx]
    U_diff = u[:, np.newaxis] < u[np.newaxis, :]
    V_diff = v[:, np.newaxis] < v[np.newaxis, :]
    return (U_diff & V_diff).astype(np.float32)

N = 2000
C = generate_causal_set_2d(N)
A = 0.5 * (C - C.T)
A_sq = np.dot(A.T, A)
lam_sq = np.linalg.eigvalsh(A_sq)
omega = np.sqrt(np.maximum(lam_sq, 0))
omega = omega[omega > 1e-8]
omega_sorted = np.sort(omega)[::-1]
lambda_sq = 1.0 / omega_sorted

t_vals = np.logspace(-6, 1, 500)
trace_vals = np.array([np.sum(np.exp(-t * lambda_sq)) for t in t_vals])

log_t = np.log(t_vals)
log_tr = np.log(trace_vals)
d_eff = -2.0 * np.gradient(log_tr, log_t)

# Find where it plateaus around 2.0
plateau = np.where((d_eff > 1.9) & (d_eff < 2.1))[0]
if len(plateau) > 0:
    print(f"Plateau at 2.0 found! Indices: {plateau[0]} to {plateau[-1]}")
    print(f"t range: {t_vals[plateau[0]]:.2e} to {t_vals[plateau[-1]]:.2e}")
else:
    print("No plateau at 2.0 found.")

print("\nSample values of d_eff:")
for i in range(0, 500, 50):
    print(f"t={t_vals[i]:.2e} | d_eff={d_eff[i]:.4f} | Trace={trace_vals[i]:.1f}")

