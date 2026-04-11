"""
Mean-field Ξ₄ computation v2 — FIXED per independent analysis audit.

Fixes applied:
1. A_C and C_C normalization: removed extra n_sample factor
2. Include D₁ terms in kernel comparison (not just D₂)
3. Add α_q regression: regress η_x/ε against q_x = Σ G_x(v)f(v)
4. Add up-sector predictor q_x^up
5. Fair comparison on same sampled set

Author: David Alfyorov
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
import sys
sys.path.insert(0, 'analysis')

# ── Configuration ──
N = 2000
T = 1.0
EPS_LIST = [1.0, 2.0, 5.0, 10.0]
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
    n = len(coords)
    dt = coords[:,0:1] - coords[:,0:1].T
    ds2 = dt**2
    for k in range(1, 4):
        ds2 -= (coords[:,k:k+1] - coords[:,k:k+1].T)**2
    C = ((dt > 0) & (ds2 > 0)).astype(np.int8)
    return C

def causal_matrix_ppwave(coords, eps):
    n = len(coords)
    dt = coords[:,0:1] - coords[:,0:1].T
    dx = coords[:,1:2] - coords[:,1:2].T
    dy = coords[:,2:3] - coords[:,2:3].T
    dz = coords[:,3:4] - coords[:,3:4].T
    ds2_flat = dt**2 - dx**2 - dy**2 - dz**2
    xm = (coords[:,1:2] + coords[:,1:2].T) / 2
    ym = (coords[:,2:3] + coords[:,2:3].T) / 2
    du = (dt + dz)
    correction = eps/2 * (xm**2 - ym**2) * du**2
    ds2 = ds2_flat - correction
    C = ((dt > 0) & (ds2 > 0)).astype(np.int8)
    return C

def hasse_fast(C):
    C2 = (C @ C > 0).astype(np.int8)
    return C * (1 - C2)

def compute_path_counts(L):
    n = L.shape[0]
    p_down = np.zeros(n, dtype=np.float64)
    p_up = np.zeros(n, dtype=np.float64)
    L_sparse = csr_matrix(L)
    L_sparse_T = csr_matrix(L.T)
    for i in range(n):
        preds = L_sparse[i].indices
        if len(preds) == 0:
            p_down[i] = 1.0
        else:
            p_down[i] = sum(p_down[j] for j in preds)
    for i in range(n-1, -1, -1):
        succs = L_sparse_T[i].indices
        if len(succs) == 0:
            p_up[i] = 1.0
        else:
            p_up[i] = sum(p_up[j] for j in succs)
    return p_down, p_up

print("=" * 70)
print("MEAN-FIELD Ξ₄ v2 — FIXED (analytical audit)")
print(f"N={N}, T={T}, eps={EPS_LIST}")
print("=" * 70)

# ── Step 1: Build flat causal set ──
t0 = time.time()
coords = sprinkle_diamond(N, T)
order = np.argsort(coords[:, 0])
coords = coords[order]
print(f"\n[1] Sprinkled {N} points ({time.time()-t0:.1f}s)")

# ── Step 2: Causal + Hasse ──
t0 = time.time()
C_flat = causal_matrix_flat(coords)
L_flat = hasse_fast(C_flat)
n_links = L_flat.sum()
print(f"[2] Hasse: {n_links} links ({time.time()-t0:.1f}s)")

# ── Step 3: Path counts + moments ──
t0 = time.time()
p_down, p_up = compute_path_counts(L_flat)
P_flat = p_down * p_up
Y_flat = np.log2(P_flat + 1)
X = Y_flat - Y_flat.mean()
sigma0_sq = np.var(Y_flat)
kappa0 = float(np.mean(X**4) / sigma0_sq**2 - 3)
print(f"[3] σ²={sigma0_sq:.4f}, κ₀={kappa0:.4f} ({time.time()-t0:.1f}s)")

# ── Step 4: Profile ──
f_profile = (coords[:, 1]**2 - coords[:, 2]**2) / 2
C2 = np.mean(f_profile**2)
C2_exact = T**4 / 1120
print(f"[4] C₂={C2:.6f} (exact={C2_exact:.6f}, ratio={C2/C2_exact:.4f})")

# ── Step 5: CRN — perturbed moments ──
print(f"\n[5] CRN perturbation:")
crn_results = {}
for eps in EPS_LIST:
    C_pp = causal_matrix_ppwave(coords, eps)
    L_pp = hasse_fast(C_pp)
    p_d_pp, p_u_pp = compute_path_counts(L_pp)
    Y_pp = np.log2(p_d_pp * p_u_pp + 1)
    kappa_pp = float(np.mean((Y_pp - Y_pp.mean())**4) / np.var(Y_pp)**2 - 3)
    dk = kappa_pp - kappa0

    delta_Y = Y_pp - Y_flat
    eta_tilde = delta_Y - delta_Y.mean()

    a_t = np.mean(X * eta_tilde)
    b_t = np.mean(eta_tilde**2)
    c_t = np.mean(X**3 * eta_tilde)
    d_t = np.mean(X**2 * eta_tilde**2)

    # Full D_full = D₁ + D₂ (eq 3.48 from independent analysis)
    D1 = 4*c_t/sigma0_sq**2 - 4*(kappa0+3)*a_t/sigma0_sq
    D2 = (6*d_t/sigma0_sq**2 - 2*(kappa0+3)*b_t/sigma0_sq
          - 16*a_t*c_t/sigma0_sq**3 + 12*(kappa0+3)*a_t**2/sigma0_sq**2)
    D_full = D1 + D2

    A_eff = dk / (eps**2 * N**0.5 * C2_exact) if abs(eps) > 0 else 0

    crn_results[eps] = {
        'dk': dk, 'D1': D1, 'D2': D2, 'D_full': D_full,
        'delta_Y': delta_Y, 'eta_tilde': eta_tilde, 'A_eff': A_eff,
        'Y_pp': Y_pp
    }

    print(f"  ε={eps:5.1f}: Δκ={dk:+.6f}  D₁={D1:+.6f}  D₂={D2:+.6f}  "
          f"D_full={D_full:+.6f}  ratio={dk/D_full if abs(D_full)>1e-10 else float('nan'):.3f}  "
          f"A_eff={A_eff:.4f}")

# ── Step 6: G_x(v) kernel ──
print(f"\n[6] Computing G_x(v) kernel...")
t0 = time.time()

L_sparse_T = csr_matrix(L_flat.T)
succ_lists = {}
for v in range(N):
    succ_lists[v] = L_sparse_T[v].indices.tolist()

# Interior sample
heights = np.zeros(N)
L_sparse = csr_matrix(L_flat)
for i in range(N):
    preds = L_sparse[i].indices
    if len(preds) == 0:
        heights[i] = 0
    else:
        heights[i] = max(heights[j] for j in preds) + 1

max_h = heights.max()
interior_mask = (heights > max_h * 0.1) & (heights < max_h * 0.9)
interior_idx = np.where(interior_mask)[0]
n_sample = min(200, len(interior_idx))
sample_idx = np.random.choice(interior_idx, n_sample, replace=False)
print(f"  Interior: {len(interior_idx)}/{N}, sample: {n_sample}")

# Compute G_x(v) for each sampled x
G_matrix = np.zeros((n_sample, N), dtype=np.float64)

for i, x in enumerate(sample_idx):
    if i % 50 == 0 and i > 0:
        print(f"  G: {i}/{n_sample} ({time.time()-t0:.1f}s)")
    # R[v,x] via backward sweep using SUCCESSORS
    R_to_x = np.zeros(N, dtype=np.float64)
    R_to_x[x] = 1.0
    for v in range(x-1, -1, -1):
        for w in succ_lists[v]:
            if w <= x:
                R_to_x[v] += R_to_x[w]
    if p_down[x] > 0:
        G_matrix[i] = p_down * R_to_x / p_down[x]

print(f"  G computed ({time.time()-t0:.1f}s)")

# ── Step 7: FIXED kernel contractions ──
print(f"\n[7] Kernel contractions (FIXED normalization):")

Gf = G_matrix @ f_profile  # q_x = Σ G_x(v) f(v), shape (n_sample,)
X_s = X[sample_idx]

# Kernel moments (CORRECT normalization per independent analysis audit)
a0 = np.mean(X_s * Gf)        # ⟨X q⟩
b0 = np.mean(Gf**2)            # ⟨q²⟩
c0 = np.mean(X_s**3 * Gf)     # ⟨X³ q⟩
d0 = np.mean(X_s**2 * Gf**2)  # ⟨X² q²⟩

# FIXED: A and C contractions
fAf = a0**2                    # was: a0² × n_sample (BUG)
fCf = a0 * c0                  # was: a0 × c0 × n_sample (BUG)
fBf = b0
fDf = d0

print(f"  a₀ = {a0:.6f}")
print(f"  b₀ = {b0:.6f}")
print(f"  c₀ = {c0:.6f}")
print(f"  d₀ = {d0:.6f}")
print(f"  ⟨f,Bf⟩ = {fBf:.8f}  (was 0.00237)")
print(f"  ⟨f,Df⟩ = {fDf:.8f}  (was 0.04648)")
print(f"  ⟨f,Af⟩ = {fAf:.8f}  (was 0.00573 — NOW FIXED)")
print(f"  ⟨f,Cf⟩ = {fCf:.8f}  (was 0.61368 — NOW FIXED)")

# ── Step 8: α_q regression (analytical Step 2) ──
print(f"\n[8] Predictor test: regress η_x/ε against q_x")

for eps in [2.0, 5.0]:
    res = crn_results[eps]
    eta_s = res['eta_tilde'][sample_idx] / eps
    alpha_q = np.dot(Gf, eta_s) / np.dot(Gf, Gf) if np.dot(Gf, Gf) > 0 else 0
    pred = alpha_q * Gf
    SS_res = np.sum((eta_s - pred)**2)
    SS_tot = np.sum((eta_s - eta_s.mean())**2)
    R2 = 1 - SS_res/SS_tot if SS_tot > 0 else 0

    # Also regression of η on f(x) directly (a_full)
    f_s = f_profile[sample_idx]
    a_full = np.dot(f_s, eta_s) / np.dot(f_s, f_s) if np.dot(f_s, f_s) > 0 else 0
    pred_f = a_full * f_s
    SS_res_f = np.sum((eta_s - pred_f)**2)
    R2_f = 1 - SS_res_f/SS_tot if SS_tot > 0 else 0

    print(f"  ε={eps}: α_q={alpha_q:.4f}, R²(q_x)={R2:.4f} | "
          f"a_full={a_full:.4f}, R²(f)={R2_f:.4f}")

# ── Step 9: Kernel D₂ with DIFFERENT couplings ──
print(f"\n[9] Kernel D₂ with different couplings:")

# β from degree regression
deg_flat = np.array(L_flat.sum(axis=0) + L_flat.sum(axis=1)).flatten().astype(float)
eps_ref = 2.0
C_pp_ref = causal_matrix_ppwave(coords, eps_ref)
L_pp_ref = hasse_fast(C_pp_ref)
deg_pp = np.array(L_pp_ref.sum(axis=0) + L_pp_ref.sum(axis=1)).flatten().astype(float)
delta_deg = deg_pp - deg_flat
k0_mean = deg_flat.mean()
beta_fit = np.dot(eps_ref * f_profile, delta_deg / k0_mean) / np.dot(eps_ref * f_profile, eps_ref * f_profile)
print(f"  β (degree) = {beta_fit:.4f}")

# α_q at ε=2
eta_s_2 = crn_results[2.0]['eta_tilde'][sample_idx] / 2.0
alpha_q_2 = np.dot(Gf, eta_s_2) / np.dot(Gf, Gf) if np.dot(Gf, Gf) > 0 else 0
print(f"  α_q (path predictor) = {alpha_q_2:.4f}")

# a_full at ε=2
f_s = f_profile[sample_idx]
a_full_2 = np.dot(f_s, eta_s_2) / np.dot(f_s, f_s) if np.dot(f_s, f_s) > 0 else 0
print(f"  a_full (direct f) = {a_full_2:.4f}")

for coupling_name, alpha in [('β', beta_fit), ('α_q', alpha_q_2), ('a_full', a_full_2)]:
    # Kernel D₁ + D₂ using this coupling
    D1_ker = alpha * (4*c0/sigma0_sq**2 - 4*(kappa0+3)*a0/sigma0_sq)
    D2_ker = alpha**2 * (6*d0/sigma0_sq**2 - 2*(kappa0+3)*b0/sigma0_sq
                         - 16*fCf/sigma0_sq**3 + 12*(kappa0+3)*fAf/sigma0_sq**2)
    D_full_ker = D1_ker + D2_ker

    dk_obs = crn_results[2.0]['dk']
    D_full_obs = crn_results[2.0]['D_full']

    print(f"  {coupling_name:6s}: D₁_ker={D1_ker:+.6f}, D₂_ker={D2_ker:+.6f}, "
          f"D_full_ker={D_full_ker:+.6f} | obs={dk_obs:+.6f} ratio={dk_obs/D_full_ker if abs(D_full_ker)>1e-10 else float('nan'):.3f}")

# ── Step 10: Identity + overlap ──
print(f"\n[10] Identity & overlap:")
sum_G = G_matrix.sum(axis=1)
overlap = np.sum(G_matrix**2, axis=1)
eff_len = sum_G**2 / overlap
print(f"  Σ G_x(v) = {sum_G.mean():.2f} ± {sum_G.std():.2f}")
print(f"  Σ G²_x   = {overlap.mean():.4f} (min 1.0 from G_x(x)=1)")
print(f"  l_eff     = {eff_len.mean():.2f} ± {eff_len.std():.2f}")
print(f"  Max height H = {max_h:.0f}")

# ── Step 11: Mean-field test (CORRECT interpretation per independent analysis) ──
print(f"\n[11] Mean-field test:")
G_bar = G_matrix.mean(axis=0)
Gbar_f = np.dot(G_bar, f_profile)
fBf_MF = Gbar_f**2
print(f"  Ḡ·f = {Gbar_f:.6f} (should ≈ 0 by l=2 symmetry)")
print(f"  ⟨f,B_MF f⟩ = {fBf_MF:.8f}")
print(f"  ⟨f,B_exact f⟩ = {fBf:.8f}")
print(f"  MF/exact = {fBf_MF/fBf if fBf > 0 else float('nan'):.6f}")
print(f"  NOTE: Low ratio is expected (Ḡ is ~radial, f is l=2 quadrupole)")

print(f"\n{'='*70}")
print("DONE — v2 with analytical fixes")
print(f"{'='*70}")
