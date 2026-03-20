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

def build_dirac(t, x, C):
    N = len(t)
    D = np.zeros((2*N, 2*N), dtype=np.complex128)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) 
    
    # We use sigma1 and sigma2 for the 2D Euclidean Dirac operator
    # to maintain strict Hermitian properties.
    for i in range(N):
        for j in range(i+1, N):
            if C[i, j]:
                dt = t[j] - t[i]
                dx = x[j] - x[i]
                dist = np.sqrt(dt**2 + dx**2)
                dirac_elem = (sigma1 * dt + sigma2 * dx) / dist
                D[2*i:2*i+2, 2*j:2*j+2] = dirac_elem
                D[2*j:2*j+2, 2*i:2*i+2] = dirac_elem.conj().T
                
    # Normalize by 1/sqrt(N) or similar to keep spectrum stable? 
    # For now, just raw.
    return D / np.sqrt(N)

def analyze_degeneracy(evals, tol=1e-5):
    # Sort eigenvalues
    evals = np.sort(evals)
    
    # Find gaps
    diffs = evals[1:] - evals[:-1]
    
    # Group degenerate values
    degeneracies = []
    current_deg = 1
    
    for i in range(len(diffs)):
        if diffs[i] < tol:
            current_deg += 1
        else:
            if current_deg > 1:
                degeneracies.append((evals[i], current_deg))
            current_deg = 1
            
    # Check last
    if current_deg > 1:
        degeneracies.append((evals[-1], current_deg))
        
    return degeneracies, evals

for N in [100, 200, 400]:
    print(f"\n--- Running for N={N} ---")
    t, x, C = generate_causal_set_diamond(N)
    D = build_dirac(t, x, C)
    evals = np.linalg.eigvalsh(D)
    
    # We apply Spectral Truncation: we only care about the lowest magnitude eigenvalues
    # Let's look at the modes closest to 0 (the macroscopic modes)
    abs_evals = np.abs(evals)
    sorted_idx = np.argsort(abs_evals)
    evals_trunc = evals[sorted_idx][:int(0.2 * len(evals))] # take 20% lowest modes
    
    degs, _ = analyze_degeneracy(evals_trunc, tol=1e-4)
    
    if len(degs) == 0:
        print("No exact degeneracies found in the truncated spectrum.")
    else:
        print(f"Found {len(degs)} degenerate multiplets in truncated spectrum:")
        # Group by multiplicity
        deg_counts = {}
        for val, mult in degs:
            deg_counts[mult] = deg_counts.get(mult, 0) + 1
        
        for mult, count in deg_counts.items():
            print(f"  Multiplicity {mult}: occurs {count} times")

    # Let's print the first 10 smallest magnitude eigenvalues to see their spacing
    print("Lowest 10 absolute eigenvalues:")
    print(np.sort(abs_evals)[:10])

