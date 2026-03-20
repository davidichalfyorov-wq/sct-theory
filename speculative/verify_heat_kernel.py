import numpy as np
from scipy.optimize import curve_fit

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
print(f"Generating Causal Set N={N}...")
C = generate_causal_set_2d(N)
A = 0.5 * (C - C.T)
A_sq = np.dot(A.T, A)
lam_sq = np.linalg.eigvalsh(A_sq)
omega = np.sqrt(np.maximum(lam_sq, 0))
omega = omega[omega > 1e-8]

# D_SJ^2 is roughly proportional to A^-1.
# So the eigenvalues of D^2 are lambda_n^2 = 1 / omega_n
lambda_sq = 1.0 / omega

# Heat kernel Trace: Tr(e^{-t D^2}) = sum_n exp(-t lambda_n^2)
t_vals = np.logspace(-4, 1, 50)
trace_vals = []

for t in t_vals:
    trace = np.sum(np.exp(-t * lambda_sq))
    trace_vals.append(trace)

trace_vals = np.array(trace_vals)

# In 2D, Seeley-DeWitt says Tr(e^{-t D^2}) ~ a_0 t^{-1} + a_2 t^0 + ...
# Let's fit Tr(t) = C * t^p to find the leading power p.
# For continuum 2D manifold, p should be -1.0.

def power_law(t, C, p):
    return C * (t ** p)

# Fit small t (UV regime)
popt_uv, _ = curve_fit(power_law, t_vals[:15], trace_vals[:15], p0=[100, -1])
# Fit large t (IR regime)
popt_ir, _ = curve_fit(power_law, t_vals[-15:], trace_vals[-15:], p0=[100, -1])

print(f"\nHeat Kernel Asymptotics Tr(e^{{-t D^2}}) ~ t^p")
print(f"Continuum 2D Prediction (Weyl's Law): p = -1.0000")
print(f"Causal Set UV Regime (Small t): p = {popt_uv[1]:.4f}")
print(f"Causal Set IR Regime (Large t): p = {popt_ir[1]:.4f}")

if abs(popt_uv[1] - (-1.0)) > 0.1:
    print("\nCONCLUSION: FATAL FAILURE.")
    print("The UV regime completely violates the heat kernel expansion.")
    print("The Seeley-DeWitt coefficients (and General Relativity) CANNOT be extracted.")
else:
    print("\nCONCLUSION: SUCCESS.")
    print("The UV regime matches the heat kernel expansion.")

