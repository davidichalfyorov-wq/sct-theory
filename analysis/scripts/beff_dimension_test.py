#!/usr/bin/env python3
"""
Test hypothesis: m_scr(d) = d + 1.

Measure β_pair / β_Hasse in dimensions d = 2, 3, 4 on flat causal diamonds,
extract m_scr from (1 - e^{-m}) / m = β_Hasse / β_pair.

If m_scr(2) ≈ 3, m_scr(3) ≈ 4, m_scr(4) ≈ 5 → hypothesis confirmed.

Method: CRN with small perturbation (eps → 0 limit).
For each d, sprinkle into d-dim causal diamond, build flat + perturbed Hasse,
measure degree response β = δk / (k₀ · ε).
β_pair from all causal pairs, β_Hasse from Hasse links only.
"""
import sys, os, time, json, math
import numpy as np

# ============================================================
# PARAMETERS
# ============================================================
M_SEEDS = 30
EPS = 0.5  # small perturbation for linear response

# N per dimension (adjusted for computational cost)
N_PER_DIM = {2: 2000, 3: 3000, 4: 5000}


# ============================================================
# SPRINKLING INTO d-DIMENSIONAL CAUSAL DIAMOND
# ============================================================

def sprinkle_diamond_2d(N, T, rng):
    """Sprinkle N points into 2D causal diamond |t| + |x| < T/2."""
    pts = []
    while len(pts) < N:
        batch = rng.uniform(-T/2, T/2, size=(8*N, 2))
        mask = np.abs(batch[:, 0]) + np.abs(batch[:, 1]) < T/2
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:N])
    return arr[np.argsort(arr[:, 0])]


def sprinkle_diamond_3d(N, T, rng):
    """Sprinkle N points into 3D causal diamond |t| + r < T/2, r = |x,y|."""
    pts = []
    while len(pts) < N:
        batch = rng.uniform(-T/2, T/2, size=(16*N, 3))
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2)
        mask = np.abs(batch[:, 0]) + r < T/2
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:N])
    return arr[np.argsort(arr[:, 0])]


def sprinkle_diamond_4d(N, T, rng):
    """Sprinkle N points into 4D causal diamond |t| + r < T/2, r = |x,y,z|."""
    pts = []
    while len(pts) < N:
        batch = rng.uniform(-T/2, T/2, size=(32*N, 4))
        r = np.linalg.norm(batch[:, 1:], axis=1)
        mask = np.abs(batch[:, 0]) + r < T/2
        pts.extend(batch[mask].tolist())
    arr = np.array(pts[:N])
    return arr[np.argsort(arr[:, 0])]


SPRINKLERS = {2: sprinkle_diamond_2d, 3: sprinkle_diamond_3d, 4: sprinkle_diamond_4d}


# ============================================================
# CAUSAL PREDICATES (flat + pp-wave perturbation in any d)
# ============================================================

def flat_pred(pts, i, d):
    """Flat Minkowski: s² = -dt² + Σdx² < 0 and dt > 0."""
    x = pts[i]
    y = pts[:i]
    dt = x[0] - y[:, 0]
    dx2 = np.sum((x[1:] - y[:, 1:])**2, axis=1)
    return (dt > 1e-12) & (dt**2 > dx2)


def perturbed_pred(pts, i, d, eps):
    """Perturbed metric: add eps * x1^2 * dt^2 correction (simple quadrupole).
    ds² = -(1 - eps*x1²)dt² + dx²
    Causal if (1 - eps*x1_mid²)*dt² > dx²
    """
    x = pts[i]
    y = pts[:i]
    dt = x[0] - y[:, 0]
    dx2 = np.sum((x[1:] - y[:, 1:])**2, axis=1)

    # Midpoint x1
    x1_mid = 0.5 * (x[1] + y[:, 1])

    # Perturbed interval
    s2_pert = (1.0 + eps * x1_mid**2) * dt**2 - dx2
    return (dt > 1e-12) & (s2_pert > 0)


# ============================================================
# HASSE BUILDER (bitset, same as production)
# ============================================================

def build_hasse(pts, pred_fn):
    """Build Hasse diagram using bitset transitive reduction."""
    n = len(pts)
    parents = [np.empty(0, dtype=np.int32) for _ in range(n)]
    children_lists = [[] for _ in range(n)]
    past_bits = [0] * n

    for i in range(n):
        rel_mask = pred_fn(pts, i)
        if rel_mask.size != i:
            continue
        rel_preds = np.flatnonzero(rel_mask)
        if rel_preds.size == 0:
            continue

        covered = 0
        direct = []
        for j in rel_preds[::-1]:
            bit = 1 << int(j)
            if covered & bit:
                continue
            direct.append(int(j))
            covered |= past_bits[int(j)] | bit

        direct.sort()
        parents[i] = np.array(direct, dtype=np.int32)

        pb = 0
        for jj in direct:
            children_lists[jj].append(i)
            pb |= past_bits[jj] | (1 << jj)
        past_bits[i] = pb

    children = [np.array(ch, dtype=np.int32) for ch in children_lists]
    return parents, children


# ============================================================
# DEGREE MEASUREMENT
# ============================================================

def measure_degrees(pts, parents_flat, parents_pert, pred_flat_fn, pred_pert_fn):
    """Measure mean degree (links) and mean causal predecessors for flat and perturbed.
    Returns β_Hasse and β_pair estimates.
    """
    n = len(pts)

    # Hasse degrees (links)
    k_flat_hasse = np.array([len(parents_flat[i]) for i in range(n)], dtype=np.float64)
    k_pert_hasse = np.array([len(parents_pert[i]) for i in range(n)], dtype=np.float64)

    # Causal predecessor counts (all pairs)
    k_flat_causal = np.zeros(n)
    k_pert_causal = np.zeros(n)
    for i in range(n):
        k_flat_causal[i] = np.sum(pred_flat_fn(pts, i))
        k_pert_causal[i] = np.sum(pred_pert_fn(pts, i))

    # Use bulk elements only (middle 50% by time)
    t = pts[:, 0]
    t_range = t.max() - t.min()
    bulk = (t > t.min() + 0.25 * t_range) & (t < t.max() - 0.25 * t_range)

    if bulk.sum() < 10:
        return 0, 0, 0, 0

    mean_k0_hasse = np.mean(k_flat_hasse[bulk])
    mean_k0_causal = np.mean(k_flat_causal[bulk])

    # Relative degree change
    dk_hasse = np.mean(k_pert_hasse[bulk] - k_flat_hasse[bulk])
    dk_causal = np.mean(k_pert_causal[bulk] - k_flat_causal[bulk])

    beta_hasse = dk_hasse / (mean_k0_hasse + 1e-30)
    beta_causal = dk_causal / (mean_k0_causal + 1e-30)

    return beta_hasse, beta_causal, mean_k0_hasse, mean_k0_causal


# ============================================================
# EXTRACT m_scr FROM RATIO
# ============================================================

def extract_m(ratio):
    """Solve (1 - e^{-m}) / m = ratio for m > 0.
    Uses Newton's method.
    """
    if ratio <= 0 or ratio >= 1:
        return float('nan')

    # f(m) = (1 - e^{-m}) / m - ratio = 0
    m = 1.0  # initial guess
    for _ in range(100):
        em = math.exp(-m)
        f = (1 - em) / m - ratio
        fp = (em * m - (1 - em)) / (m * m)  # derivative
        if abs(fp) < 1e-30:
            break
        m_new = m - f / fp
        if m_new <= 0:
            m_new = m / 2
        if abs(m_new - m) < 1e-12:
            m = m_new
            break
        m = m_new
    return m


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    T = 1.0
    print("=" * 72)
    print("B_EFF DIMENSION TEST: m_scr(d) = d + 1 ?")
    print(f"Dimensions: 2, 3, 4")
    print(f"M={M_SEEDS}, eps={EPS}")
    print("=" * 72, flush=True)

    results = {}

    for d in [2, 3, 4]:
        N = N_PER_DIM[d]
        sprinkle = SPRINKLERS[d]

        print(f"\n{'='*60}")
        print(f"DIMENSION d = {d}, N = {N}")
        print(f"{'='*60}", flush=True)

        beta_h_list = []
        beta_c_list = []
        k0_h_list = []
        k0_c_list = []

        for si in range(M_SEEDS):
            t0 = time.time()
            seed = 7700000 + d * 1000 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle(N, T, rng)

            pred_flat = lambda P, i, _d=d: flat_pred(P, i, _d)
            pred_pert = lambda P, i, _d=d, _e=EPS: perturbed_pred(P, i, _d, _e)

            par_flat, ch_flat = build_hasse(pts, pred_flat)
            par_pert, ch_pert = build_hasse(pts, pred_pert)

            bh, bc, k0h, k0c = measure_degrees(pts, par_flat, par_pert, pred_flat, pred_pert)

            beta_h_list.append(bh)
            beta_c_list.append(bc)
            k0_h_list.append(k0h)
            k0_c_list.append(k0c)

            elapsed = time.time() - t0
            if (si + 1) % 10 == 0 or si == 0:
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"β_H={bh:+.4f}  β_C={bc:+.4f}  "
                      f"<k_H>={k0h:.1f}  <k_C>={k0c:.0f}  "
                      f"({elapsed:.1f}s)", flush=True)

        bh_arr = np.array(beta_h_list)
        bc_arr = np.array(beta_c_list)

        bh_mean = float(bh_arr.mean())
        bc_mean = float(bc_arr.mean())
        bh_se = float(bh_arr.std(ddof=1) / math.sqrt(M_SEEDS))
        bc_se = float(bc_arr.std(ddof=1) / math.sqrt(M_SEEDS))

        if abs(bc_mean) > 1e-10:
            ratio = bh_mean / bc_mean
            m_scr = extract_m(abs(ratio))
        else:
            ratio = float('nan')
            m_scr = float('nan')

        results[str(d)] = {
            'd': d, 'N': N,
            'beta_hasse_mean': bh_mean,
            'beta_hasse_se': bh_se,
            'beta_causal_mean': bc_mean,
            'beta_causal_se': bc_se,
            'ratio': ratio,
            'm_scr': m_scr,
            'target_m': d + 1,
            'mean_k_hasse': float(np.mean(k0_h_list)),
            'mean_k_causal': float(np.mean(k0_c_list)),
        }

        print(f"\n  SUMMARY d={d}:")
        print(f"    β_Hasse  = {bh_mean:+.5f} ± {bh_se:.5f}")
        print(f"    β_causal = {bc_mean:+.5f} ± {bc_se:.5f}")
        print(f"    ratio    = {ratio:.4f}")
        print(f"    m_scr    = {m_scr:.3f}  (target: d+1 = {d+1})")
        print(f"    <k_Hasse>  = {np.mean(k0_h_list):.2f}")
        print(f"    <k_causal> = {np.mean(k0_c_list):.0f}")
        print(flush=True)

    # ============================================================
    # FINAL TABLE
    # ============================================================
    print(f"\n{'='*72}")
    print("FINAL TABLE: m_scr(d) vs d+1")
    print(f"{'='*72}")
    print(f"  {'d':<4} {'m_scr':<10} {'d+1':<6} {'match?':<10} {'ratio':<10} {'<k_H>':<8}")
    print(f"  {'-'*48}")
    for d in [2, 3, 4]:
        r = results[str(d)]
        match = abs(r['m_scr'] - (d+1)) < 0.5
        print(f"  {d:<4} {r['m_scr']:<10.3f} {d+1:<6} {'YES' if match else 'NO':<10} "
              f"{r['ratio']:<10.4f} {r['mean_k_hasse']:<8.2f}")

    print(f"\n  Hypothesis m_scr = d + 1:", end=" ")
    all_match = all(abs(results[str(d)]['m_scr'] - (d+1)) < 1.0 for d in [2, 3, 4])
    print("SUPPORTED" if all_match else "NOT SUPPORTED")

    outfile = 'analysis/fnd1_data/beff_dimension_test.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}", flush=True)
