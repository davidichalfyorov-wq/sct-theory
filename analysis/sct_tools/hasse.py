"""
SCT Theory — Bitset-packed Hasse diagram (transitive reduction) for causal sets.

Provides O(N²) memory, O(N² × N/64) time transitive reduction using
uint64 ancestor bitsets. This is 20-30× faster than the naive C @ C > 0
dense method for N ≥ 2000.

Algorithm (independent analysis, 2026-03-27, validated against dense method):
  For each element i (time-sorted ascending):
    1. Compute all causal predecessors j < i from geometry.
    2. Build ancestor bitset: anc[i] = OR of (anc[j] | bit(j)) for all predecessors.
    3. Iterate predecessors in REVERSE time order.
       If j is NOT covered (bit j not in covered set), mark j as direct parent.
       Update covered |= anc[j].
    4. Result: parents[i] = direct (Hasse) predecessors of i.

Key functions:
    build_hasse_bitset(points, eps=None) -> parents, children lists
    hasse_to_sparse(parents, children, n) -> scipy.sparse.csr_matrix
    path_kurtosis_from_lists(parents, children) -> float
    crn_trial_bitset(N, eps, seed, T=1.0) -> float (Δκ)

Performance (single core, Intel i9-12900KS):
    N=2000:  ~0.9s per trial
    N=3000:  ~1.8s per trial
    N=5000:  ~4.4s per trial
    N=10000: ~18s per trial (estimated)

Compared to dense C @ C method:
    N=2000:  0.9s vs 9s    (10×)
    N=5000:  4.4s vs 120s  (27×)
"""

import numpy as np
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────
# DIAMOND SPRINKLING
# ─────────────────────────────────────────────────────────────

def sprinkle_diamond(n, T=1.0, rng=None, seed=None):
    """Sprinkle n points uniformly into a 4D causal diamond of duration T.

    The diamond is {(t, x, y, z) : |t| + r < T/2}, where r = sqrt(x²+y²+z²).

    Parameters:
        n: number of points
        T: total proper time of the diamond (default 1.0)
        rng: numpy random Generator (if None, created from seed)
        seed: random seed (used only if rng is None)

    Returns:
        (n, 4) array of coordinates [t, x, y, z], sorted by t ascending.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        batch = rng.uniform(-T / 2, T / 2, (n * 10, 4))
        r = np.sqrt(batch[:, 1] ** 2 + batch[:, 2] ** 2 + batch[:, 3] ** 2)
        mask = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[mask].tolist())
    c = np.array(pts[:n], dtype=np.float64)
    return c[np.argsort(c[:, 0])]


def sprinkle_shell(n, T=1.0, r_min=0.10, rng=None, seed=None):
    """Sprinkle n points into a 4D causal diamond shell: |t|+r < T/2 AND r >= r_min.

    Parameters:
        n: number of points
        T: total proper time of the diamond (default 1.0)
        r_min: minimum radial distance from origin (default 0.10)
        rng: numpy random Generator (if None, created from seed)
        seed: random seed (used only if rng is None)

    Returns:
        (n, 4) array of coordinates [t, x, y, z], sorted by t ascending.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        batch = rng.uniform(-T / 2, T / 2, (n * 10, 4))
        r = np.sqrt(batch[:, 1] ** 2 + batch[:, 2] ** 2 + batch[:, 3] ** 2)
        mask = (np.abs(batch[:, 0]) + r < T / 2) & (r >= r_min)
        pts.extend(batch[mask].tolist())
    c = np.array(pts[:n], dtype=np.float64)
    return c[np.argsort(c[:, 0])]


# ─────────────────────────────────────────────────────────────
# BITSET UTILITIES
# ─────────────────────────────────────────────────────────────

def _set_bits(nwords, indices):
    """Create a uint64 bitset with given indices set."""
    row = np.zeros(nwords, dtype=np.uint64)
    if len(indices) > 0:
        words = indices >> 6
        bits = (indices & 63).astype(np.uint64)
        masks = np.left_shift(np.uint64(1), bits)
        np.bitwise_or.at(row, words, masks)
    return row


def _test_bit(bitset, j):
    """Test whether bit j is set in a uint64 bitset array."""
    w = j >> 6
    mask = np.uint64(1) << np.uint64(j & 63)
    return bool(bitset[w] & mask)


# ─────────────────────────────────────────────────────────────
# BITSET HASSE BUILDER
# ─────────────────────────────────────────────────────────────

def _ppwave_exact_preds(t, x, y, z, i, eps, tol=1e-12):
    """Exact pp-wave causal predecessors of element i among j < i.

    Uses the exact geodesic-derived V_needed formula (independent analysis, 2026-03-27).
    For eps > 0, omega = sqrt(eps/2):
      V_needed = omega * [(xA²+xB²)cosh(omega*U) - 2*xA*xB] / sinh(omega*U)
               + omega * [(yA²+yB²)cos(omega*U) - 2*yA*yB] / sin(omega*U)
    Condition: A prec B iff U > 0 and V >= V_needed.
    Valid when omega*U < pi (always true for T=1, |eps| <= 5).

    Returns: bool array of shape (i,)
    """
    dt = t[i] - t[:i]
    dz = z[i] - z[:i]
    U = dt + dz  # retarded null coordinate difference
    V = dt - dz  # advanced null coordinate difference

    xA = x[:i]
    xB = x[i]
    yA = y[:i]
    yB = y[i]

    mask_U = U > tol
    result = np.zeros(i, dtype=bool)
    if not np.any(mask_U):
        return result

    Um = U[mask_U]
    Vm = V[mask_U]
    xa = xA[mask_U]
    ya = yA[mask_U]

    w = np.sqrt(abs(eps) / 2.0)
    eta = w * Um

    if eps > 0:
        # x-sector: hyperbolic, y-sector: oscillatory
        sh = np.sinh(eta)
        ch = np.cosh(eta)
        sn = np.sin(eta)
        cs = np.cos(eta)
        Sx = w * ((xa * xa + xB * xB) * ch - 2.0 * xa * xB) / sh
        Sy = w * ((ya * ya + yB * yB) * cs - 2.0 * ya * yB) / sn
    else:
        # x-sector: oscillatory, y-sector: hyperbolic
        sn = np.sin(eta)
        cs = np.cos(eta)
        sh = np.sinh(eta)
        ch = np.cosh(eta)
        Sx = w * ((xa * xa + xB * xB) * cs - 2.0 * xa * xB) / sn
        Sy = w * ((ya * ya + yB * yB) * ch - 2.0 * ya * yB) / sh

    Vneed = Sx + Sy
    result[mask_U] = Vm >= (Vneed - tol)
    return result


def _schwarzschild_shapiro_preds(t, x, y, z, i, M, tol=1e-12):
    """Exact first-order Schwarzschild causal predecessors using Shapiro time delay.

    Condition: A prec B iff dt > 0 and dt >= R_AB + 2M * ln[(rA+rB+R)/(rA+rB-R)].
    Valid in weak field (M/r << 1). No ad-hoc softening.

    Returns: bool array of shape (i,)
    """
    dt = t[i] - t[:i]  # positive (sorted)
    dx = x[i] - x[:i]
    dy = y[i] - y[:i]
    dz = z[i] - z[:i]
    R = np.sqrt(dx * dx + dy * dy + dz * dz)
    rA = np.sqrt(x[:i] ** 2 + y[:i] ** 2 + z[:i] ** 2)
    rB = np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)

    result = np.zeros(i, dtype=bool)
    mask_dt = dt > tol
    if not np.any(mask_dt):
        return result

    dtm = dt[mask_dt]
    Rm = R[mask_dt]
    rAm = rA[mask_dt]

    sum_r = rAm + rB
    # Avoid log singularity: sum_r - R can be very small for near-lightcone pairs
    denom = sum_r - Rm
    safe = denom > tol
    dt_need = np.full_like(dtm, np.inf)

    if np.any(safe):
        arg = (sum_r[safe] + Rm[safe]) / denom[safe]
        # Clamp arg to avoid numerical issues
        arg = np.maximum(arg, 1.0)
        shapiro = 2.0 * M * np.log(arg)
        dt_need[safe] = Rm[safe] + shapiro

    result[mask_dt] = dtm >= (dt_need - tol)
    return result


def build_hasse_bitset(points, eps=None, exact=True, metric='ppwave', M_sch=None):
    """Build Hasse diagram using bitset ancestor tracking.

    Parameters:
        points: (N, 4) array [t, x, y, z], MUST be sorted by t ascending.
        eps: pp-wave amplitude (used when metric='ppwave').
             If None AND M_sch is None, use flat Minkowski causal relation.
        exact: if True (default), use exact causal relation (pp-wave or Schwarzschild).
        metric: 'ppwave' (default) or 'schwarzschild'.
        M_sch: Schwarzschild mass parameter (used when metric='schwarzschild').
               Overrides eps. Uses Shapiro first-order causal relation.

    Returns:
        parents: list of N int32 arrays (direct predecessors of each element)
        children: list of N lists (direct successors of each element)
    """
    n = len(points)
    nwords = (n + 63) // 64

    anc = np.zeros((n, nwords), dtype=np.uint64)
    parents = [None] * n
    children = [[] for _ in range(n)]

    t = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]
    z = points[:, 3]

    use_ppwave = eps is not None and eps != 0.0
    use_sch = M_sch is not None and M_sch != 0.0

    for i in range(n):
        if i == 0:
            parents[i] = np.empty(0, dtype=np.int32)
            continue

        if use_sch:
            # Schwarzschild Shapiro first-order causal relation
            mask = _schwarzschild_shapiro_preds(t, x, y, z, i, M_sch)
            rel_preds = np.nonzero(mask)[0].astype(np.int32)
        elif use_ppwave and exact:
            # Exact geodesic-derived causal relation
            mask = _ppwave_exact_preds(t, x, y, z, i, eps)
            rel_preds = np.nonzero(mask)[0].astype(np.int32)
        elif use_ppwave:
            # Legacy midpoint surrogate
            dt = t[i] - t[:i]
            dx = x[i] - x[:i]
            dy = y[i] - y[:i]
            dz = z[i] - z[:i]
            s2 = dt * dt - dx * dx - dy * dy - dz * dz
            xm = 0.5 * (x[i] + x[:i])
            ym = 0.5 * (y[i] + y[:i])
            du = dt + dz
            corr = eps / 2 * (xm * xm - ym * ym) * (du * du)
            rel_preds = np.nonzero(s2 - corr > 0)[0].astype(np.int32)
        else:
            dt = t[i] - t[:i]
            dx = x[i] - x[:i]
            dy = y[i] - y[:i]
            dz = z[i] - z[:i]
            s2 = dt * dt - dx * dx - dy * dy - dz * dz
            rel_preds = np.nonzero(s2 > 0)[0].astype(np.int32)

        # Build ancestor bitset for element i
        row = _set_bits(nwords, rel_preds)
        anc[i] = row

        # Transitive reduction: iterate predecessors in REVERSE time order
        # (latest first). If j is not yet covered, it's a direct parent.
        covered = np.zeros(nwords, dtype=np.uint64)
        pars = []

        for j in rel_preds[::-1]:
            if _test_bit(covered, j):
                continue
            pars.append(int(j))
            covered |= anc[j]
            # Also set bit j in covered (j itself is now accounted for)
            w = j >> 6
            covered[w] |= np.uint64(1) << np.uint64(j & 63)

        arr = np.array(pars, dtype=np.int32)
        parents[i] = arr
        for j in arr:
            children[j].append(i)

    return parents, children


# ─────────────────────────────────────────────────────────────
# GENERIC PREDICATE BITSET HASSE
# ─────────────────────────────────────────────────────────────

def build_hasse_bitset_generic(points, pred_func):
    """Build Hasse diagram using bitset TR with an arbitrary causal predicate.

    Parameters:
        points: (N, 4) array [t, x, y, z], MUST be sorted by t ascending.
        pred_func: callable(points, i) -> bool array of shape (i,)
                   Returns True for each j < i that is a causal predecessor of i.

    Returns:
        parents: list of N int32 arrays
        children: list of N lists
    """
    n = len(points)
    nwords = (n + 63) // 64

    anc = np.zeros((n, nwords), dtype=np.uint64)
    parents = [None] * n
    children = [[] for _ in range(n)]

    for i in range(n):
        if i == 0:
            parents[i] = np.empty(0, dtype=np.int32)
            continue

        mask = pred_func(points, i)
        rel_preds = np.nonzero(mask)[0].astype(np.int32)

        row = _set_bits(nwords, rel_preds)
        anc[i] = row

        covered = np.zeros(nwords, dtype=np.uint64)
        pars = []

        for j in rel_preds[::-1]:
            if _test_bit(covered, j):
                continue
            pars.append(int(j))
            covered |= anc[j]
            w = j >> 6
            covered[w] |= np.uint64(1) << np.uint64(j & 63)

        arr = np.array(pars, dtype=np.int32)
        parents[i] = arr
        for j in arr:
            children[j].append(i)

    return parents, children


# ─────────────────────────────────────────────────────────────
# CONVERSION TO SPARSE MATRIX
# ─────────────────────────────────────────────────────────────

def hasse_to_sparse(parents, children, n=None):
    """Convert parent/children lists to sparse CSR matrix.

    Returns L where L[i, j] = 1 iff j is a direct predecessor of i
    (j -> i is a Hasse link).
    """
    if n is None:
        n = len(parents)
    rows, cols = [], []
    for i in range(n):
        for j in parents[i]:
            rows.append(i)
            cols.append(int(j))
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


# ─────────────────────────────────────────────────────────────
# PATH COUNTING AND KURTOSIS
# ─────────────────────────────────────────────────────────────

def path_counts(parents, children):
    """Compute forward and backward path counts via DP on Hasse DAG.

    Returns:
        pd: array of shape (N,), pd[i] = number of source-to-i Hasse paths
        pu: array of shape (N,), pu[i] = number of i-to-sink Hasse paths
    """
    n = len(parents)
    pd = np.zeros(n, dtype=np.float64)
    pu = np.zeros(n, dtype=np.float64)

    # Forward: sources have pd = 1
    for i in range(n):
        p = parents[i]
        if len(p) > 0:
            pd[i] = pd[p].sum()
        else:
            pd[i] = 1.0

    # Backward: sinks have pu = 1
    for i in range(n - 1, -1, -1):
        c = children[i]
        if c:
            pu[i] = sum(pu[j] for j in c)
        else:
            pu[i] = 1.0

    return pd, pu


def path_kurtosis_from_lists(parents, children):
    """Compute excess kurtosis of Y = log₂(p_down × p_up + 1).

    Parameters:
        parents, children: output of build_hasse_bitset()

    Returns:
        float: excess kurtosis κ = μ₄/σ⁴ - 3
    """
    pd, pu = path_counts(parents, children)
    Y = np.log2(pd * pu + 1.0)
    X = Y - Y.mean()
    s2 = np.var(Y)
    if s2 < 1e-12:
        return 0.0
    return float(np.mean(X ** 4) / (s2 * s2) - 3.0)


# ─────────────────────────────────────────────────────────────
# CRN TRIAL (single seed, single eps)
# ─────────────────────────────────────────────────────────────

def crn_trial_bitset(N, eps, seed, T=1.0):
    """Run one CRN trial: Δκ = κ(pp-wave) - κ(flat).

    Uses the same point set for both flat and pp-wave causal relations.

    Parameters:
        N: number of sprinkled points
        eps: pp-wave amplitude
        seed: random seed for sprinkling
        T: diamond duration (default 1.0)

    Returns:
        float: Δκ = path_kurtosis(pp-wave) - path_kurtosis(flat)
    """
    c = sprinkle_diamond(N, T=T, seed=seed)

    parents0, children0 = build_hasse_bitset(c, eps=None)
    parentsE, childrenE = build_hasse_bitset(c, eps=eps)

    k0 = path_kurtosis_from_lists(parents0, children0)
    ke = path_kurtosis_from_lists(parentsE, childrenE)

    return ke - k0


def crn_trial_schwarzschild(N, M_sch, seed, T=1.0, r_min=0.10):
    """Run one CRN trial for Schwarzschild: Δκ = κ(Schwarzschild) - κ(flat).

    Sprinkles into a shell (r >= r_min) to avoid the singularity.
    Uses Shapiro first-order causal relation.

    Parameters:
        N: number of sprinkled points
        M_sch: Schwarzschild mass parameter (r_s = 2*M_sch)
        seed: random seed for sprinkling
        T: diamond duration (default 1.0)
        r_min: minimum radial distance (default 0.10)

    Returns:
        float: Δκ = path_kurtosis(Schwarzschild) - path_kurtosis(flat)
    """
    c = sprinkle_shell(N, T=T, r_min=r_min, seed=seed)

    parents0, children0 = build_hasse_bitset(c, eps=None)
    parentsS, childrenS = build_hasse_bitset(c, M_sch=M_sch)

    k0 = path_kurtosis_from_lists(parents0, children0)
    ks = path_kurtosis_from_lists(parentsS, childrenS)

    return ks - k0


# ─────────────────────────────────────────────────────────────
# ENSEMBLE RUNNER
# ─────────────────────────────────────────────────────────────

def run_aeff_ensemble(N_values, eps_values, M=30, T=1.0, seed_formula=None):
    """Run full A_eff ensemble measurement.

    Parameters:
        N_values: list of N values (e.g., [500, 1000, 2000, 5000])
        eps_values: list of epsilon values (e.g., [0.5, 1.0, 2.0, 3.0, 5.0])
        M: number of seeds per (N, eps) pair
        T: diamond duration
        seed_formula: function(N, eps, m) -> seed. Default: 1000*N + 100*int(eps*10) + m

    Returns:
        dict: results[N][eps] = {'mean', 'se', 'A_eff', 'A_se', 'd', 'dks'}
    """
    import time
    C2 = T ** 4 / 1120

    if seed_formula is None:
        def seed_formula(N, eps, m):
            return 1000 * N + 100 * int(eps * 10) + m

    results = {}
    for N in N_values:
        results[N] = {}
        for eps in eps_values:
            t0 = time.time()
            dks = []
            for m in range(M):
                seed = seed_formula(N, eps, m)
                dk = crn_trial_bitset(N, eps, seed, T)
                dks.append(dk)
            dks = np.array(dks)
            mn = dks.mean()
            se = dks.std(ddof=0) / np.sqrt(M)
            A_eff = mn / (eps ** 2 * np.sqrt(N) * C2)
            A_se = se / (eps ** 2 * np.sqrt(N) * C2)
            d = mn / se if se > 1e-15 else 0.0
            elapsed = time.time() - t0

            results[N][eps] = {
                'mean': float(mn),
                'se': float(se),
                'A_eff': float(A_eff),
                'A_se': float(A_se),
                'd': float(d),
                'dks': dks.tolist(),
                'elapsed': float(elapsed),
            }
    return results
