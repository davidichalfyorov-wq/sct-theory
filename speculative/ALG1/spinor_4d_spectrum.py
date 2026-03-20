import numpy as np

def generate_causal_set_4d_diamond(N, seed=42):
    np.random.seed(seed)
    u = np.random.rand(N)
    v = np.random.rand(N)
    # Actually, generating uniformly in a 4D diamond is trickier.
    # Let's just generate N points in a 4D hypercube [-1, 1]^4
    # and restrict to the causal diamond |t| + |x| + |y| + |z| < 1
    # or just use Euclidean coordinates and a random graph.
    # To keep the "causal set" spirit, let's use a 4D Minkowski space sprinkling.
    # A simple way to get a causal set is sprinkling in a box and keeping those with |t| > sqrt(x^2+y^2+z^2)
    
    pts = []
    while len(pts) < N:
        p = np.random.uniform(-1, 1, 4)
        # diamond condition |t| + sqrt(x^2+y^2+z^2) < 1
        if abs(p[0]) + np.sqrt(p[1]**2 + p[2]**2 + p[3]**2) < 1.0:
            pts.append(p)
    pts = np.array(pts)
    
    # Sort by time
    pts = pts[np.argsort(pts[:, 0])]
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    
    # Causal relations: t_j - t_i > sqrt(dx^2 + dy^2 + dz^2)
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(i+1, N):
            dt = t[j] - t[i]
            dr = np.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2 + (z[j]-z[i])**2)
            if dt > dr:
                C[i, j] = 1
                
    return pts, C

# 4D Gamma Matrices (Euclidean, Hermitian)
I2 = np.eye(2, dtype=np.complex128)
Z2 = np.zeros((2,2), dtype=np.complex128)
s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

g0 = np.block([[Z2, I2], [I2, Z2]])
g1 = np.block([[Z2, -1j*s1], [1j*s1, Z2]])
g2 = np.block([[Z2, -1j*s2], [1j*s2, Z2]])
g3 = np.block([[Z2, -1j*s3], [1j*s3, Z2]])

def build_dirac_4d(pts, C):
    N = len(pts)
    D = np.zeros((4*N, 4*N), dtype=np.complex128)
    
    for i in range(N):
        for j in range(i+1, N):
            if C[i, j]:
                dp = pts[j] - pts[i]
                # Euclidean distance
                dist = np.linalg.norm(dp)
                dirac_elem = (g0 * dp[0] + g1 * dp[1] + g2 * dp[2] + g3 * dp[3]) / dist
                D[4*i:4*i+4, 4*j:4*j+4] = dirac_elem
                D[4*j:4*j+4, 4*i:4*i+4] = dirac_elem.conj().T
                
    return D / np.sqrt(N)

N = 100
print(f"Generating 4D Causal Set N={N}...")
pts, C = generate_causal_set_4d_diamond(N)
print("Building Dirac operator...")
D = build_dirac_4d(pts, C)

print("Diagonalizing...")
evals = np.linalg.eigvalsh(D)

abs_evals = np.abs(evals)
sorted_idx = np.argsort(abs_evals)

print("Lowest 20 signed eigenvalues:")
for idx in sorted_idx[:20]:
    print(f"{evals[idx]:.8f}")

