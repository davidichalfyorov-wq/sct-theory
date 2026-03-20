import numpy as np

def generate_causal_set_diamond(N, seed=42):
    np.random.seed(seed)
    u = np.random.rand(N)
    v = np.random.rand(N)
    t = (u + v) / np.sqrt(2)
    x = (u - v) / np.sqrt(2)
    
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    u = u[idx]
    v = v[idx]
    
    U_diff = u[:, np.newaxis] < u[np.newaxis, :]
    V_diff = v[:, np.newaxis] < v[np.newaxis, :]
    C = (U_diff & V_diff).astype(np.int32)
    
    return t, x, C

N = 200
t, x, C = generate_causal_set_diamond(N)

D = np.zeros((2*N, 2*N), dtype=np.complex128)
sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) 
    
for i in range(N):
    for j in range(i+1, N):
        if C[i, j]:
            dt = t[j] - t[i]
            dx = x[j] - x[i]
            dist = np.sqrt(dt**2 + dx**2)
            dirac_elem = (sigma1 * dt + sigma2 * dx) / dist
            D[2*i:2*i+2, 2*j:2*j+2] = dirac_elem
            D[2*j:2*j+2, 2*i:2*i+2] = dirac_elem.conj().T

D = D / np.sqrt(N)
evals = np.linalg.eigvalsh(D)

abs_evals = np.abs(evals)
sorted_idx = np.argsort(abs_evals)

print("Lowest 20 signed eigenvalues:")
for idx in sorted_idx[:20]:
    print(f"{evals[idx]:.8f}")

