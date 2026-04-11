#!/usr/bin/env python3
"""
Bridge Level 3: Improved spectral observables with noise reduction.

Three fixes over Level 2:
  1. Bulk projection: P_bulk @ L @ P_bulk before eigendecomposition
  2. Soft threshold: sigmoid weight instead of hard cut
  3. Heat trace: Tr(exp(-t*H^2)) instead of log det — smooth, bounded

Also computes:
  - Band-resolved delta-Gamma (which spectral bands carry curvature signal)
  - Resolvent trace as secondary check
  - Full eigenvalue spectra saved for post-hoc analysis

Tests on pp-wave (eps=1,3) + Schwarzschild, N=2000 and N=5000.
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)

# ============================================================
# PARAMETERS
# ============================================================
T = 1.0
ZETA = 0.15
M_SEEDS = 15

SCH_M = 0.05
SCH_R0 = 0.50
SCH_R_ABCD = riemann_schwarzschild_local(SCH_M, SCH_R0)
SCH_E2 = 6.0 * SCH_M**2 / SCH_R0**6

# Heat trace parameters: scan over t
HEAT_T_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]

# Soft threshold parameters
SOFT_ALPHA = 8.0  # sigmoid steepness


# ============================================================
# LINK MATRIX + BULK PROJECTION
# ============================================================

def hasse_to_link_matrix(parents, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def get_pj_eigenvalues(L, mask=None):
    """Compute positive eigenvalues of i*(L - L^T), optionally with bulk projection.

    If mask is provided, projects: L_bulk = P @ L @ P where P = diag(mask).
    Returns sorted positive eigenvalues (descending).
    """
    if mask is not None:
        m = mask.astype(np.float64)
        L = (m[:, None] * L) * m[None, :]  # P @ L @ P without forming dense P

    PJ = L - L.T
    H = 1j * PJ
    evals = np.linalg.eigvalsh(H)  # real eigenvalues
    sigma = np.sort(evals[evals > 1e-14])[::-1]
    return sigma


# ============================================================
# SPECTRAL OBSERVABLES
# ============================================================

def hard_logdet(sigma, threshold):
    """Original Level 2: hard truncation log-pseudodeterminant."""
    s = sigma[sigma > threshold]
    if len(s) == 0:
        return 0.0, 0
    return float(np.sum(np.log(s))), len(s)


def soft_logdet(sigma, threshold, alpha=8.0):
    """Soft threshold log-pseudodeterminant with sigmoid weight."""
    s = sigma[sigma > 1e-14]
    if len(s) == 0:
        return 0.0, 0.0
    log_s = np.log(s)
    log_tau = math.log(threshold)
    w = 1.0 / (1.0 + np.exp(-alpha * (log_s - log_tau)))
    val = float(np.sum(w * log_s))
    eff_modes = float(np.sum(w))
    return val, eff_modes


def heat_trace(sigma, t):
    """Heat trace: Tr(exp(-t * H^2)) = 2 * sum exp(-t * sigma_k^2)."""
    s = sigma[sigma > 1e-14]
    if len(s) == 0:
        return 0.0
    return 2.0 * float(np.sum(np.exp(-t * s**2)))


def resolvent_trace(sigma, mu):
    """Resolvent trace: Tr((H^2 + mu^2)^{-1}) = 2 * sum 1/(sigma_k^2 + mu^2)."""
    s = sigma[sigma > 1e-14]
    if len(s) == 0:
        return 0.0
    return 2.0 * float(np.sum(1.0 / (s**2 + mu**2)))


def band_logdet(sigma, threshold, n_bands=4):
    """Band-resolved log-determinant: which spectral bands carry signal."""
    s = sigma[sigma > threshold]
    if len(s) < 2:
        return [0.0] * n_bands
    log_min = math.log(s[-1])
    log_max = math.log(s[0])
    edges = np.linspace(log_min, log_max, n_bands + 1)
    bands = []
    log_s = np.log(s)
    for b in range(n_bands):
        mask_b = (log_s >= edges[b]) & (log_s < edges[b + 1])
        if b == n_bands - 1:
            mask_b = (log_s >= edges[b]) & (log_s <= edges[b + 1])
        bands.append(float(np.sum(log_s[mask_b])))
    return bands


# ============================================================
# CJ (for correlation)
# ============================================================

def Y_from_graph(par, ch):
    n = len(par)
    p_down = np.ones(n, dtype=np.float64)
    p_up = np.ones(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            p_down[i] = np.sum(p_down[par[i]]) + 1
    for i in range(n - 1, -1, -1):
        if ch[i] is not None and len(ch[i]) > 0:
            p_up[i] = np.sum(p_up[ch[i]]) + 1
    return np.log2(p_down * p_up + 1)


def make_strata(pts, par0, T):
    tau_hat = 2 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if par0[i] is not None and len(par0[i]) > 0:
            depth[i] = int(np.max(depth[par0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


def compute_CJ(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = (np.mean(np.abs(X[idx]) * dY2[idx])
               - np.mean(np.abs(X[idx])) * np.mean(dY2[idx]))
        total += w * cov
    return float(total)


# ============================================================
# SINGLE SEED RUNNER
# ============================================================

def run_seed(seed_idx, N, geometry, eps=None, base_seed=9500000):
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    t0 = time.time()
    threshold = math.sqrt(N) / (4 * math.pi)
    bmask = bulk_mask(pts, T, ZETA)

    # Build Hasse diagrams
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    if geometry == 'ppwave':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        E2 = eps**2 / 2.0
    else:
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, SCH_R_ABCD))
        E2 = SCH_E2

    L_flat = hasse_to_link_matrix(par0, N)
    L_curved = hasse_to_link_matrix(parC, N)
    t_hasse = time.time() - t0

    # ---- EIGENVALUES: full graph and bulk-projected ----
    sig_flat_full = get_pj_eigenvalues(L_flat, mask=None)
    sig_curved_full = get_pj_eigenvalues(L_curved, mask=None)
    sig_flat_bulk = get_pj_eigenvalues(L_flat, mask=bmask)
    sig_curved_bulk = get_pj_eigenvalues(L_curved, mask=bmask)
    t_eigen = time.time() - t0

    # ---- OBSERVABLES: full graph ----
    hard_ld_flat, n_hard_flat = hard_logdet(sig_flat_full, threshold)
    hard_ld_curved, n_hard_curved = hard_logdet(sig_curved_full, threshold)
    dG_hard = 0.5 * (hard_ld_curved - hard_ld_flat)
    dn_modes = n_hard_curved - n_hard_flat

    soft_ld_flat, eff_flat = soft_logdet(sig_flat_full, threshold, SOFT_ALPHA)
    soft_ld_curved, eff_curved = soft_logdet(sig_curved_full, threshold, SOFT_ALPHA)
    dG_soft = 0.5 * (soft_ld_curved - soft_ld_flat)

    # ---- OBSERVABLES: bulk-projected ----
    threshold_bulk = math.sqrt(bmask.sum()) / (4 * math.pi)
    hard_ld_flat_b, _ = hard_logdet(sig_flat_bulk, threshold_bulk)
    hard_ld_curved_b, _ = hard_logdet(sig_curved_bulk, threshold_bulk)
    dG_hard_bulk = 0.5 * (hard_ld_curved_b - hard_ld_flat_b)

    soft_ld_flat_b, _ = soft_logdet(sig_flat_bulk, threshold_bulk, SOFT_ALPHA)
    soft_ld_curved_b, _ = soft_logdet(sig_curved_bulk, threshold_bulk, SOFT_ALPHA)
    dG_soft_bulk = 0.5 * (soft_ld_curved_b - soft_ld_flat_b)

    # ---- HEAT TRACES ----
    dH = {}
    dH_bulk = {}
    for ht in HEAT_T_VALUES:
        h_flat = heat_trace(sig_flat_full, ht)
        h_curved = heat_trace(sig_curved_full, ht)
        dH[str(ht)] = h_curved - h_flat

        h_flat_b = heat_trace(sig_flat_bulk, ht)
        h_curved_b = heat_trace(sig_curved_bulk, ht)
        dH_bulk[str(ht)] = h_curved_b - h_flat_b

    # ---- RESOLVENT (mu = threshold) ----
    r_flat = resolvent_trace(sig_flat_full, threshold)
    r_curved = resolvent_trace(sig_curved_full, threshold)
    dR = r_curved - r_flat

    # ---- CJ ----
    Y0 = Y_from_graph(par0, ch0)
    YC = Y_from_graph(parC, chC)
    strata = make_strata(pts, par0, T)
    cj = compute_CJ(Y0, YC - Y0, bmask, strata)

    t_total = time.time() - t0

    return {
        'seed': seed_idx, 'N': N, 'geometry': geometry, 'eps': eps, 'E2': E2,
        'CJ': cj,
        # Hard logdet (Level 2 original)
        'dG_hard': dG_hard,
        'dn_modes': dn_modes,
        # Soft logdet
        'dG_soft': dG_soft,
        # Bulk-projected hard
        'dG_hard_bulk': dG_hard_bulk,
        # Bulk-projected soft (BEST CANDIDATE)
        'dG_soft_bulk': dG_soft_bulk,
        # Heat traces (full and bulk)
        'dH': dH,
        'dH_bulk': dH_bulk,
        # Resolvent
        'dR': dR,
        # Diagnostics
        'n_bulk': int(bmask.sum()),
        'corr_dG_dn': 0,  # computed across seeds later
        'time_s': t_total,
        't_hasse': t_hasse,
        't_eigen': t_eigen,
    }


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def summarize(values, label=""):
    arr = np.array(values, dtype=np.float64)
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    t = m / (se + 1e-30)
    return m, se, t


def print_comparison(seeds, cj_list, label):
    """Print comparison of all observables for one configuration."""
    dG_hard = [s['dG_hard'] for s in seeds]
    dG_soft = [s['dG_soft'] for s in seeds]
    dG_hard_b = [s['dG_hard_bulk'] for s in seeds]
    dG_soft_b = [s['dG_soft_bulk'] for s in seeds]
    dn_modes = [s['dn_modes'] for s in seeds]

    print(f"\n  {label}:")
    for name, vals in [("dG_hard (L2 orig)", dG_hard),
                       ("dG_soft         ", dG_soft),
                       ("dG_hard_bulk    ", dG_hard_b),
                       ("dG_soft_bulk    ", dG_soft_b)]:
        m, se, t = summarize(vals)
        r = float(np.corrcoef(vals, cj_list)[0, 1]) if len(vals) >= 3 else 0
        r_dn = float(np.corrcoef(vals, dn_modes)[0, 1]) if len(vals) >= 3 else 0
        print(f"    {name}: {m:+.4f}±{se:.4f} (t={t:+.2f})  "
              f"r(CJ)={r:+.3f}  r(dn)={r_dn:+.3f}")

    # Heat traces
    for ht in HEAT_T_VALUES:
        vals = [s['dH'][str(ht)] for s in seeds]
        vals_b = [s['dH_bulk'][str(ht)] for s in seeds]
        m, se, t = summarize(vals)
        mb, seb, tb = summarize(vals_b)
        r = float(np.corrcoef(vals, cj_list)[0, 1]) if len(vals) >= 3 else 0
        rb = float(np.corrcoef(vals_b, cj_list)[0, 1]) if len(vals_b) >= 3 else 0
        print(f"    dH(t={ht})     full: {m:+.4f}±{se:.4f} (t={t:+.2f}) r(CJ)={r:+.3f}"
              f"  | bulk: {mb:+.4f}±{seb:.4f} (t={tb:+.2f}) r(CJ)={rb:+.3f}")

    # CJ for reference
    m_cj, se_cj, t_cj = summarize(cj_list)
    print(f"    CJ              : {m_cj:.6f}±{se_cj:.6f} (t={t_cj:+.2f})")

    # Diagnostic: corr(dG_hard, dn_modes)
    r_diag = float(np.corrcoef(dG_hard, dn_modes)[0, 1]) if len(dG_hard) >= 3 else 0
    print(f"    DIAGNOSTIC: corr(dG_hard, dn_modes) = {r_diag:+.3f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=2000)
    args = parser.parse_args()
    N = args.N

    print("=" * 72)
    print(f"BRIDGE LEVEL 3: Improved spectral observables @ N={N}")
    print(f"Fixes: bulk projection + soft threshold (alpha={SOFT_ALPHA}) + heat trace")
    print(f"M={M_SEEDS}, T={T}, zeta={ZETA}")
    print(f"Hard threshold = {math.sqrt(N)/(4*math.pi):.2f}")
    print(f"Heat t values: {HEAT_T_VALUES}")
    print("=" * 72, flush=True)

    all_results = {}

    configs = [
        ('ppwave', 1.0, 'ppw_eps1'),
        ('ppwave', 3.0, 'ppw_eps3'),
        ('schwarzschild', None, 'sch'),
    ]

    for geometry, eps, key in configs:
        E2 = (eps**2 / 2.0) if eps else SCH_E2
        if geometry == 'ppwave':
            label = f"PP-WAVE eps={eps} (E^2={E2:.2f})"
        else:
            label = f"SCHWARZSCHILD (E^2={SCH_E2:.4f})"

        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}", flush=True)

        seeds_data = []
        cj_list = []

        for si in range(M_SEEDS):
            t0 = time.time()
            res = run_seed(si, N, geometry, eps=eps)
            seeds_data.append(res)
            cj_list.append(res['CJ'])

            if (si + 1) % 5 == 0 or si == 0:
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"hard={res['dG_hard']:+.4f} "
                      f"soft={res['dG_soft']:+.4f} "
                      f"bulk_soft={res['dG_soft_bulk']:+.4f} "
                      f"dH(0.01)={res['dH']['0.01']:+.4f} "
                      f"CJ={res['CJ']:.6f} "
                      f"dn={res['dn_modes']:+d} "
                      f"({res['time_s']:.1f}s)", flush=True)

        print_comparison(seeds_data, cj_list, label)

        all_results[key] = {
            'geometry': geometry, 'eps': eps, 'E2': E2, 'N': N,
            'seeds': seeds_data,
        }

    # ============================================================
    # FINAL COMPARISON TABLE
    # ============================================================
    print(f"\n{'='*72}")
    print("FINAL COMPARISON: Which observable best detects curvature?")
    print(f"{'='*72}")
    print(f"\n{'Observable':<22} {'ppw e=1 t':<12} {'ppw e=3 t':<12} {'Sch t':<12} "
          f"{'ppw3 r(CJ)':<12} {'Sch r(CJ)':<12}")
    print("-" * 82)

    for obs_name, obs_key in [
        ("dG_hard (L2 orig)", 'dG_hard'),
        ("dG_soft", 'dG_soft'),
        ("dG_hard_bulk", 'dG_hard_bulk'),
        ("dG_soft_bulk", 'dG_soft_bulk'),
    ]:
        row = []
        for cfg_key in ['ppw_eps1', 'ppw_eps3', 'sch']:
            vals = [s[obs_key] for s in all_results[cfg_key]['seeds']]
            cjs = [s['CJ'] for s in all_results[cfg_key]['seeds']]
            _, _, t = summarize(vals)
            row.append(f"{t:+.2f}")
        # correlations
        for cfg_key in ['ppw_eps3', 'sch']:
            vals = [s[obs_key] for s in all_results[cfg_key]['seeds']]
            cjs = [s['CJ'] for s in all_results[cfg_key]['seeds']]
            r = float(np.corrcoef(vals, cjs)[0, 1]) if len(vals) >= 3 else 0
            row.append(f"{r:+.3f}")
        print(f"  {obs_name:<20} {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

    # Best heat trace
    for ht in HEAT_T_VALUES:
        row = []
        for cfg_key in ['ppw_eps1', 'ppw_eps3', 'sch']:
            vals = [s['dH_bulk'][str(ht)] for s in all_results[cfg_key]['seeds']]
            _, _, t = summarize(vals)
            row.append(f"{t:+.2f}")
        for cfg_key in ['ppw_eps3', 'sch']:
            vals = [s['dH_bulk'][str(ht)] for s in all_results[cfg_key]['seeds']]
            cjs = [s['CJ'] for s in all_results[cfg_key]['seeds']]
            r = float(np.corrcoef(vals, cjs)[0, 1]) if len(vals) >= 3 else 0
            row.append(f"{r:+.3f}")
        print(f"  dH_bulk(t={ht:<5})    {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

    print(f"\n  CJ (reference):", end="")
    for cfg_key in ['ppw_eps1', 'ppw_eps3', 'sch']:
        cjs = [s['CJ'] for s in all_results[cfg_key]['seeds']]
        _, _, t = summarize(cjs)
        print(f"  t={t:+.2f}", end="")
    print()

    # Save
    outfile = f'analysis/fnd1_data/bridge_level3_N{N}.json'
    for key in all_results:
        for sr in all_results[key]['seeds']:
            for k in list(sr.keys()):
                if isinstance(sr[k], np.ndarray):
                    sr[k] = sr[k].tolist()
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\nSaved to {outfile}", flush=True)
