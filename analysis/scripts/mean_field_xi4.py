"""
Mean-field computation of Ξ₄ — the flat-space path-response constant.

Goal: Verify analytical kernel representation by direct computation of
  D₂ = β² ⟨f, K_C f⟩
and extract Ξ₄ = A_eff / β².

Method:
1. Build flat causal set (N=2000), compute Hasse diagram
2. Compute path-occupation kernel G_x(v) = p_down(v) × R_{vx} / p_down(x)
3. Compute centered moments a, b, c, d exactly
4. Compute full D₂ from centered quadratic formula
5. Perturb to pp-wave, compute observed Δκ
6. Compare: does D₂ × ε² match Δκ_obs?
7. Test mean-field factorization: B(u,v) ≈ B_MF(u,v)?
8. Extract Ξ₄ = D₂ / (N^{1/2} × C₂)

Author: David Alfyorov
"""

import numpy as np
import time
from scipy.sparse import csr_matrix, lil_matrix
import sys
sys.path.insert(0, 'analysis')

# ── Configuration ──
N = 2000
T = 1.0
EPS_LIST = [2.0, 5.0, 10.0]
M_TRIALS = 5  # CRN trials for ensemble stats
SEED = 42

np.random.seed(SEED)

def sprinkle_diamond(n, T=1.0):
    """Poisson sprinkling into 4D causal diamond |t|+r < T/2."""
    pts = []
    while len(pts) < n:
        batch = np.random.uniform(-T/2, T/2, (n*10, 4))
        r = np.sqrt(batch[:,1]**2 + batch[:,2]**2 + batch[:,3]**2)
        mask = np.abs(batch[:,0]) + r < T/2
        pts.extend(batch[mask].tolist())
    return np.array(pts[:n])

def causal_matrix_flat(coords):
    """Build causal matrix for flat Minkowski spacetime."""
    n = len(coords)
    dt = coords[:,0:1] - coords[:,0:1].T
    dx = coords[:,1:2] - coords[:,1:2].T
    dy = coords[:,2:3] - coords[:,2:3].T
    dz = coords[:,3:4] - coords[:,3:4].T
    s2 = dt**2 - dx**2 - dy**2 - dz**2
    C = (s2 > 0) & (dt > 0)
    np.fill_diagonal(C, False)
    return C.astype(np.int8)

def causal_matrix_ppwave(coords, eps):
    """Build causal matrix for pp-wave with midpoint surrogate."""
    n = len(coords)
    dt = coords[:,0:1] - coords[:,0:1].T
    dx = coords[:,1:2] - coords[:,1:2].T
    dy = coords[:,2:3] - coords[:,2:3].T
    dz = coords[:,3:4] - coords[:,3:4].T

    xm = (coords[:,1:2] + coords[:,1:2].T) / 2
    ym = (coords[:,2:3] + coords[:,2:3].T) / 2
    du = dt + dz

    s2_flat = dt**2 - dx**2 - dy**2 - dz**2
    correction = (eps / 2) * (xm**2 - ym**2) * du**2
    s2_curved = s2_flat - correction

    C = (s2_curved > 0) & (dt > 0)
    np.fill_diagonal(C, False)
    return C.astype(np.int8)

def transitive_reduction(C):
    """Compute Hasse diagram (transitive reduction) of causal matrix."""
    n = C.shape[0]
    # Sort by time coordinate is implicit in causal ordering
    L = C.copy().astype(bool)
    for k in range(n):
        for i in range(n):
            if not L[i, k]:
                continue
            for j in range(n):
                if L[k, j]:
                    L[i, j] = False
    return L.astype(np.int8)

def hasse_fast(C_dense):
    """Fast Hasse diagram via transitive reduction using topological layers."""
    n = C_dense.shape[0]
    C = C_dense.astype(bool)
    # Use matrix chain: if C[i,k] and C[k,j] then remove C[i,j]
    # Efficient: L = C but not (C @ C > 0)
    C2 = C.astype(np.float32) @ C.astype(np.float32)
    L = C & (C2 == 0)
    return L.astype(np.int8)

def compute_path_counts(L):
    """Compute p_down, p_up from Hasse diagram L.
    CONVENTION: C[i,j]=1 means j≺i (j precedes i), so L[i,j]=1 means j→i is a link.
    Therefore: predecessors of x = {j : L[x,j]=1} = ROW x.
               successors of x  = {i : L[i,x]=1} = COLUMN x.
    """
    n = L.shape[0]

    p_down = np.zeros(n, dtype=np.float64)

    # Forward sweep (increasing time = increasing index)
    for x in range(n):
        predecessors = np.where(L[x, :] > 0)[0]  # ROW x = predecessors
        if len(predecessors) == 0:
            p_down[x] = 1.0  # source
        else:
            p_down[x] = sum(p_down[y] for y in predecessors)

    # p_up: backward sweep
    p_up = np.zeros(n, dtype=np.float64)
    for x in range(n-1, -1, -1):
        successors = np.where(L[:, x] > 0)[0]  # COLUMN x = successors
        if len(successors) == 0:
            p_up[x] = 1.0  # sink
        else:
            p_up[x] = sum(p_up[z] for z in successors)

    return p_down, p_up

def compute_resolvent_column(L_sparse, x, n):
    """Compute column x of R = (I-L)^{-1} by backward sweep.
    R[v,x] = number of directed paths from v to x.
    """
    col = np.zeros(n, dtype=np.float64)
    col[x] = 1.0
    # Process in reverse topological order (from x backwards)
    # For each v < x: R[v,x] = sum_{w: v→w link} R[w,x]
    for v in range(x-1, -1, -1):
        successors = L_sparse[v].nonzero()[1]
        for w in successors:
            if w <= x:
                col[v] += col[w]
    return col

def compute_Gx_all(L_sparse, p_down, n, sample_x=None):
    """Compute G_x(v) for all x and v (or a sample of x).
    G_x(v) = p_down(v) * R[v,x] / p_down(x)
    """
    if sample_x is None:
        sample_x = range(n)

    G = {}  # G[x] = array of G_x(v) for all v
    for x in sample_x:
        R_col = compute_resolvent_column(L_sparse, x, n)
        Gx = np.zeros(n, dtype=np.float64)
        for v in range(n):
            if R_col[v] > 0 and p_down[x] > 0:
                Gx[v] = p_down[v] * R_col[v] / p_down[x]
        G[x] = Gx
    return G

def compute_path_kurtosis(coords, L):
    """Compute path_kurtosis for given coords and Hasse diagram."""
    p_down, p_up = compute_path_counts(L)
    P = p_down * p_up
    Y = np.log2(P + 1)
    return float(np.mean((Y - Y.mean())**4) / np.var(Y)**2 - 3)

print("=" * 70)
print("MEAN-FIELD Ξ₄ COMPUTATION")
print(f"N={N}, T={T}, eps={EPS_LIST}, M={M_TRIALS}")
print("=" * 70)

# ── Step 1: Sprinkle and build flat causal set ──
t0 = time.time()
coords = sprinkle_diamond(N, T)
# Sort by time for topological ordering
order = np.argsort(coords[:, 0])
coords = coords[order]

print(f"\n[1] Sprinkled {N} points into diamond ({time.time()-t0:.1f}s)")

# ── Step 2: Build causal + Hasse ──
t0 = time.time()
C_flat = causal_matrix_flat(coords)
L_flat = hasse_fast(C_flat)
n_links = L_flat.sum()
print(f"[2] Causal matrix + Hasse: {n_links} links ({time.time()-t0:.1f}s)")

# ── Step 3: Path counts ──
t0 = time.time()
p_down, p_up = compute_path_counts(L_flat)
P_flat = p_down * p_up
Y_flat = np.log2(P_flat + 1)
X = Y_flat - Y_flat.mean()
sigma0_sq = np.var(Y_flat)
kappa0 = float(np.mean(X**4) / sigma0_sq**2 - 3)
print(f"[3] Path counts: σ²={sigma0_sq:.4f}, κ₀={kappa0:.4f} ({time.time()-t0:.1f}s)")

# ── Step 4: Profile function ──
f_profile = (coords[:, 1]**2 - coords[:, 2]**2) / 2
C2 = np.mean(f_profile**2)
C2_exact = T**4 / 1120
print(f"[4] C₂ measured={C2:.6f}, exact={C2_exact:.6f}, ratio={C2/C2_exact:.4f}")

# ── Step 5: CRN — compute perturbed path_kurtosis ──
print(f"\n[5] CRN perturbation (same coords, different causal relation):")
for eps in EPS_LIST:
    C_pp = causal_matrix_ppwave(coords, eps)
    L_pp = hasse_fast(C_pp)
    p_down_pp, p_up_pp = compute_path_counts(L_pp)
    P_pp = p_down_pp * p_up_pp
    Y_pp = np.log2(P_pp + 1)
    kappa_pp = float(np.mean((Y_pp - Y_pp.mean())**4) / np.var(Y_pp)**2 - 3)
    dk = kappa_pp - kappa0

    # Compute perturbation field
    delta_Y = Y_pp - Y_flat
    eta_tilde = delta_Y - delta_Y.mean()

    # Centered moments
    a_tilde = np.mean(X * eta_tilde)
    b_tilde = np.mean(eta_tilde**2)
    c_tilde = np.mean(X**3 * eta_tilde)
    d_tilde = np.mean(X**2 * eta_tilde**2)

    # Full centered quadratic formula (analytical eq. 3.48)
    D2_full = (4*c_tilde/sigma0_sq**2
               - 4*(kappa0+3)*a_tilde/sigma0_sq
               + 6*d_tilde/sigma0_sq**2
               - 2*(kappa0+3)*b_tilde/sigma0_sq
               - 16*a_tilde*c_tilde/sigma0_sq**3
               + 12*(kappa0+3)*a_tilde**2/sigma0_sq**2)

    # A_eff
    A_eff = dk / (eps**2 * N**0.5 * C2_exact)

    print(f"  ε={eps:5.1f}: Δκ_obs={dk:+.6f}, D₂_full={D2_full:+.6f}, "
          f"ratio={dk/D2_full if abs(D2_full)>1e-10 else float('nan'):.3f}, "
          f"A_eff={A_eff:.4f}")

# ── Step 6: Compute G_x(v) kernel (EXPENSIVE but critical) ──
print(f"\n[6] Computing path-occupation kernel G_x(v)...")
t0 = time.time()

L_sparse = csr_matrix(L_flat)       # row access: predecessors of i = L_sparse[i]
L_sparse_T = csr_matrix(L_flat.T)   # row access on transpose: successors of v = L_sparse_T[v]
# Precompute successor lists for resolvent column computation
L_sparse_col = {}
for v in range(N):
    L_sparse_col[v] = L_sparse_T[v].indices.tolist()

# Sample interior elements (exclude boundary 10%)
heights = np.zeros(N)
for i in range(N):
    preds = np.where(L_flat[i, :] > 0)[0]  # ROW i = predecessors of i
    if len(preds) == 0:
        heights[i] = 0
    else:
        heights[i] = max(heights[j] for j in preds) + 1

max_h = heights.max()
interior_mask = (heights > max_h * 0.1) & (heights < max_h * 0.9)
interior_idx = np.where(interior_mask)[0]
n_interior = len(interior_idx)
print(f"  Interior elements: {n_interior} / {N} (height range [{max_h*0.1:.0f}, {max_h*0.9:.0f}])")

# Compute G for sampled interior elements (limit to ~200 for speed)
n_sample = min(200, n_interior)
sample_idx = np.random.choice(interior_idx, n_sample, replace=False)

# For each sampled x, compute G_x via resolvent column
G_matrix = np.zeros((n_sample, N), dtype=np.float64)

for i, x in enumerate(sample_idx):
    if i % 50 == 0:
        print(f"  Computing G for element {i}/{n_sample}... ({time.time()-t0:.1f}s)")
    # R[v,x] = #paths from v to x. For v < x, R[v,x] = Σ_{w: w is successor of v, w≤x} R[w,x]
    # Successors of v: all i with L[i,v]=1, i.e., COLUMN v of L
    R_col = np.zeros(N, dtype=np.float64)
    R_col[x] = 1.0
    for v in range(x-1, -1, -1):
        # successors of v = column v of L = all i with L[i,v]>0
        succs = L_sparse_col[v]
        for w in succs:
            if w <= x:
                R_col[v] += R_col[w]
    # G_x(v) = p_down(v) * R[v,x] / p_down(x)
    if p_down[x] > 0:
        G_matrix[i] = p_down * R_col / p_down[x]

print(f"  G computed for {n_sample} elements ({time.time()-t0:.1f}s)")

# ── Step 7: Compute kernels B, D ──
print(f"\n[7] Computing kernels...")

# B(u,v) = (1/n_sample) Σ_x G_x(u) G_x(v)
# We need ⟨f, B f⟩ = (1/n_sample) Σ_x [Σ_u G_x(u) f(u)]²
Gf = G_matrix @ f_profile  # shape: (n_sample,)
X_sampled = X[sample_idx]

# ⟨f, B f⟩
fBf = np.mean(Gf**2)

# ⟨f, D f⟩ = (1/n_sample) Σ_x X²(x) [Σ_u G_x(u) f(u)]²
fDf = np.mean(X_sampled**2 * Gf**2)

# a-type: ⟨f, A f⟩ = [Σ_x X(x) Σ_u G_x(u) f(u)]² / n_sample²
sum_XGf = np.sum(X_sampled * Gf)
fAf = (sum_XGf / n_sample)**2 * n_sample  # rescaled

# c-type: ⟨f, C f⟩ (symmetric)
sum_X3Gf = np.sum(X_sampled**3 * Gf)
fCf = sum_XGf * sum_X3Gf / n_sample**2 * n_sample  # rescaled

print(f"  ⟨f,Bf⟩ = {fBf:.6f}")
print(f"  ⟨f,Df⟩ = {fDf:.6f}")
print(f"  ⟨f,Af⟩ = {fAf:.6f}")
print(f"  ⟨f,Cf⟩ = {fCf:.6f}")

# ── Step 8: Extract β and Ξ₄ ──
# From linear response at small ε
eps_small = 2.0
C_pp_small = causal_matrix_ppwave(coords, eps_small)
L_pp_small = hasse_fast(C_pp_small)
p_d_pp, p_u_pp = compute_path_counts(L_pp_small)
Y_pp_small = np.log2(p_d_pp * p_u_pp + 1)
delta_Y_small = Y_pp_small - Y_flat
eta_small = delta_Y_small - delta_Y_small.mean()

# β from degree regression
deg_flat = np.array(L_flat.sum(axis=0) + L_flat.sum(axis=1)).flatten()  # total degree
L_pp_small_tmp = hasse_fast(C_pp_small)
deg_pp = np.array(L_pp_small_tmp.sum(axis=0) + L_pp_small_tmp.sum(axis=1)).flatten()
delta_deg = (deg_pp - deg_flat).astype(float)
# β = regression of δk/k₀ on ε×f
k0_mean = deg_flat.mean()
from numpy.linalg import lstsq
A_reg = (eps_small * f_profile).reshape(-1, 1)
beta_fit = lstsq(A_reg, delta_deg / k0_mean, rcond=None)[0][0]
print(f"\n[8] β = {beta_fit:.4f} (from ε={eps_small})")

# Ξ₄ from kernel
# D₂_kernel = β² × [6/σ⁴ × fDf - 2(κ₀+3)/σ² × fBf - 16/σ⁶ × fCf + 12(κ₀+3)/σ⁴ × fAf]
D2_kernel = beta_fit**2 * (
    6 * fDf / sigma0_sq**2
    - 2 * (kappa0 + 3) * fBf / sigma0_sq
    - 16 * fCf / sigma0_sq**3
    + 12 * (kappa0 + 3) * fAf / sigma0_sq**2
)

Xi4 = D2_kernel / (N**0.5 * C2_exact)
A_eff_kernel = D2_kernel * EPS_LIST[1]**2  # predicted Δκ at ε=5 (just D₂ × ε²)

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"β = {beta_fit:.4f}")
print(f"β² = {beta_fit**2:.6f}")
print(f"D₂(kernel) = {D2_kernel:.8f}")
print(f"Ξ₄ = A_eff/β² = {Xi4:.4f}")
print(f"A_eff(kernel) = {beta_fit**2 * Xi4:.6f}")
print(f"A_eff(data, ε=5) = {float('nan')}")  # will fill from step 5 output

# ── Step 9: Mean-field test ──
print(f"\n[9] Mean-field factorization test:")
# Exact: B(u,v) for sampled x
# Mean occupation: Ḡ(v) = (1/n_sample) Σ_x G_x(v)
G_bar = G_matrix.mean(axis=0)  # shape: (N,)
# Mean-field B: B_MF(u,v) = Ḡ(u) × Ḡ(v)
# ⟨f, B_MF f⟩ = [Σ_v Ḡ(v) f(v)]²
Gbar_f = np.dot(G_bar, f_profile)
fBf_MF = Gbar_f**2
print(f"  ⟨f,B_exact f⟩ = {fBf:.6f}")
print(f"  ⟨f,B_MF f⟩   = {fBf_MF:.6f}")
print(f"  MF/exact ratio = {fBf_MF/fBf if fBf > 0 else float('nan'):.4f}")

# ── Step 10: Verify identity Σ G_x(v) = E[|γ|] ──
print(f"\n[10] Identity check: Σ G_x(v) = E[|γ|]")
sum_G = G_matrix.sum(axis=1)  # for each sampled x
print(f"  Mean Σ G_x(v) = {sum_G.mean():.2f} ± {sum_G.std():.2f}")
print(f"  Should be ≈ average path length to x (H/2 ~ {max_h/2:.1f})")
print(f"  Max height H = {max_h:.0f}")

# ── Step 11: Overlap statistics ──
print(f"\n[11] Path overlap statistics (two-replica):")
# For sampled elements, compute overlap: Σ_v G_x(v)²
overlap = np.sum(G_matrix**2, axis=1)
eff_length = sum_G**2 / overlap
print(f"  Mean overlap Σ G²_x = {overlap.mean():.4f}")
print(f"  Effective length l_eff = (ΣG)²/(ΣG²) = {eff_length.mean():.2f} ± {eff_length.std():.2f}")

print(f"\n{'='*70}")
print("DONE")
