import numpy as np

def build_4d_causal_set(N, seed=42):
    np.random.seed(seed)
    pts = []
    while len(pts) < N:
        p = np.random.uniform(-1, 1, 4)
        if abs(p[0]) + np.sqrt(p[1]**2 + p[2]**2 + p[3]**2) < 1.0:
            pts.append(p)
    pts = np.array(pts)
    pts = pts[np.argsort(pts[:, 0])]
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(i+1, N):
            dt = t[j] - t[i]
            dr = np.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2 + (z[j]-z[i])**2)
            if dt > dr:
                C[i, j] = 1
    return pts, C

I2 = np.eye(2, dtype=np.complex128)
Z2 = np.zeros((2,2), dtype=np.complex128)
s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

g0 = np.block([[Z2, I2], [I2, Z2]])
g1 = np.block([[Z2, -1j*s1], [1j*s1, Z2]])
g2 = np.block([[Z2, -1j*s2], [1j*s2, Z2]])
g3 = np.block([[Z2, -1j*s3], [1j*s3, Z2]])

N = 200
pts, C = build_4d_causal_set(N)

D = np.zeros((4*N, 4*N), dtype=np.complex128)
for i in range(N):
    for j in range(i+1, N):
        if C[i, j]:
            dp = pts[j] - pts[i]
            dist = np.linalg.norm(dp)
            dirac_elem = (g0 * dp[0] + g1 * dp[1] + g2 * dp[2] + g3 * dp[3]) / dist
            D[4*i:4*i+4, 4*j:4*j+4] = dirac_elem
            D[4*j:4*j+4, 4*i:4*i+4] = dirac_elem.conj().T

evals = np.linalg.eigvalsh(D)
abs_evals = np.abs(evals)

# Filter zeros
tol = 1e-8
nz_evals = evals[abs_evals > tol]
nz_abs = np.abs(nz_evals)
sorted_idx = np.argsort(nz_abs)

print(f"Total zero eigenvalues: {len(evals) - len(nz_evals)}")
print("Lowest 20 non-zero signed eigenvalues:")
for idx in sorted_idx[:20]:
    print(f"{nz_evals[idx]:.8f}")

# Check exact degeneracy
diff = np.abs(nz_evals[sorted_idx[1:]] - nz_evals[sorted_idx[:-1]])
print(f"Minimum diff between consecutive absolute evals: {np.min(diff):.4e}")

