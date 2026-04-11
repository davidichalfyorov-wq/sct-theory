#!/usr/bin/env python3
"""
Bridge CJ_φ test: Field-dressed CJ using SJ vacuum correlator.

CJ_φ = Σ_B w_B Cov_B(|X|, I_i · δY_i²)

where I_i = Σ_{j ∈ flat_neighbors(i)} |W_ij|² is the SJ field intensity
at element i, and W is the SJ Wightman function (positive spectral part
of the physical Pauli-Jordan i(L₀ - L₀^T)).

Tests on pp-wave (eps=1,3) + Schwarzschild + dS (null test).
Compares plain CJ vs CJ_φ.
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local, riemann_ds,
    build_hasse_from_predicate, bulk_mask,
)

# ============================================================
# PARAMETERS
# ============================================================
N = 2000
T = 1.0
ZETA = 0.15
M_SEEDS = 20

SCH_M = 0.05
SCH_R0 = 0.50
SCH_R_ABCD = riemann_schwarzschild_local(SCH_M, SCH_R0)
SCH_E2 = 6.0 * SCH_M**2 / SCH_R0**6

DS_H = 0.50
DS_R_ABCD = riemann_ds(DS_H)


# ============================================================
# LINK MATRIX + SJ VACUUM
# ============================================================

def hasse_to_link_matrix(parents, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def compute_sj_wightman(L):
    """Compute SJ Wightman function W from link matrix L.

    Physical PJ = i(L - L^T). Eigendecompose, take positive-frequency part.
    W = positive spectral projection of i*PJ.

    Returns W (N×N complex matrix) and diagnostics.
    """
    n = L.shape[0]
    PJ = L - L.T
    H = 1j * PJ  # Hermitian

    # Full eigendecomposition (need eigenvectors for W)
    evals, evecs = np.linalg.eigh(H)

    # Positive eigenvalues = positive frequency modes
    pos_mask = evals > 1e-14

    # Sorkin-Yazdi truncation: keep only sigma > sqrt(N)/(4*pi)
    threshold = math.sqrt(n) / (4 * math.pi)
    trunc_mask = evals > threshold

    # W = sum over positive (truncated) modes: W = V_+ @ diag(sigma_+) @ V_+†
    # But we want W as the positive-frequency part of PJ, not H.
    # H = i*PJ, so PJ = -i*H = -i * V @ diag(evals) @ V†
    # Positive freq part of PJ: modes with evals > 0 of H
    # W_SJ = (1/2)(PJ_+ + i*Delta_+) where PJ_+ is positive part
    # Simpler: W = V_+ @ diag(sigma_+) @ V_+† (in H basis, then convert)
    # Actually: W(x,y) = <φ(x)φ(y)> = positive spectral part of i*Delta

    # Use truncated positive modes
    use_mask = trunc_mask  # only truncated positive modes
    n_modes = use_mask.sum()

    if n_modes == 0:
        return np.zeros((n, n)), 0

    V_pos = evecs[:, use_mask]  # N × n_modes
    sigma_pos = evals[use_mask]  # n_modes

    # W = V_pos @ diag(sigma_pos) @ V_pos†
    W = (V_pos * sigma_pos[None, :]) @ V_pos.conj().T

    return W, int(n_modes)


def compute_field_intensity(W, parents, n):
    """Compute per-element field intensity I_i = Σ_{j ∈ flat_neighbors(i)} |W_ij|².

    Uses flat Hasse parents as local neighborhood (preserves bulk-locality).
    """
    I = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                I[i] += abs(W[i, int(j)])**2
    # Also include children as neighbors (symmetric neighborhood)
    children_lists = [[] for _ in range(n)]
    for i in range(n):
        if parents[i] is not None:
            for j in parents[i]:
                children_lists[int(j)].append(i)
    for i in range(n):
        for j in children_lists[i]:
            I[i] += abs(W[i, j])**2

    return I


# ============================================================
# CJ AND CJ_PHI
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
    """Plain CJ = Σ_B w_B Cov_B(|X|, δY²)."""
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


def compute_CJ_phi(Y0, delta, mask, strata, I_field):
    """CJ_φ = Σ_B w_B Cov_B(|X|, I_i · δY_i²)."""
    X = Y0[mask] - np.mean(Y0[mask])
    dY2 = delta[mask] ** 2
    I_m = I_field[mask]
    weighted = I_m * dY2  # field-dressed response
    strata_m = strata[mask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = (np.mean(np.abs(X[idx]) * weighted[idx])
               - np.mean(np.abs(X[idx])) * np.mean(weighted[idx]))
        total += w * cov
    return float(total)


# ============================================================
# SINGLE SEED RUNNER
# ============================================================

def run_seed(seed_idx, geometry, eps=None, base_seed=9600000):
    seed = base_seed + seed_idx
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    t0 = time.time()
    bmask = bulk_mask(pts, T, ZETA)

    # Flat Hasse
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    L_flat = hasse_to_link_matrix(par0, N)

    # Curved Hasse
    if geometry == 'ppwave':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
        E2 = eps**2 / 2.0
        label = f'ppw_eps{eps}'
    elif geometry == 'schwarzschild':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, SCH_R_ABCD))
        E2 = SCH_E2
        label = 'sch'
    elif geometry == 'ds':
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, DS_R_ABCD))
        E2 = 3 * DS_H**4  # rough Ricci measure
        label = 'ds'

    # SJ Wightman function from FLAT PJ
    W, n_sj_modes = compute_sj_wightman(L_flat)

    # Field intensity from flat SJ vacuum
    I_field = compute_field_intensity(W, par0, N)

    # Path counts
    Y0 = Y_from_graph(par0, ch0)
    YC = Y_from_graph(parC, chC)
    delta_Y = YC - Y0
    strata = make_strata(pts, par0, T)

    # Plain CJ
    cj = compute_CJ(Y0, delta_Y, bmask, strata)

    # CJ_φ (field-dressed)
    cj_phi = compute_CJ_phi(Y0, delta_Y, bmask, strata, I_field)

    # Diagnostics
    I_bulk = I_field[bmask]
    I_mean = float(np.mean(I_bulk)) if I_bulk.size > 0 else 0
    I_std = float(np.std(I_bulk)) if I_bulk.size > 0 else 0
    sigma0 = float(np.std(Y0[bmask]))

    t_total = time.time() - t0

    return {
        'seed': seed_idx, 'geometry': geometry, 'eps': eps, 'E2': E2,
        'CJ': cj,
        'CJ_phi': cj_phi,
        'ratio_phi_cj': cj_phi / (cj + 1e-30),
        'I_mean': I_mean, 'I_std': I_std,
        'n_sj_modes': n_sj_modes,
        'sigma0': sigma0,
        'time_s': t_total,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 72)
    print("BRIDGE CJ_φ TEST: Field-dressed CJ via SJ vacuum")
    print(f"N={N}, T={T}, zeta={ZETA}, M={M_SEEDS}")
    print(f"CJ_φ = Σ_B w_B Cov_B(|X|, I_i · δY²)")
    print(f"I_i = Σ_{{neighbors}} |W_ij|²  (SJ field intensity)")
    print("=" * 72, flush=True)

    configs = [
        ('ppwave', 1.0, 'ppw_eps1'),
        ('ppwave', 3.0, 'ppw_eps3'),
        ('schwarzschild', None, 'sch'),
        ('ds', None, 'ds'),
    ]

    all_results = {}

    for geometry, eps, key in configs:
        if geometry == 'ppwave':
            E2 = eps**2 / 2.0
            label = f"PP-WAVE eps={eps} (E²={E2:.2f})"
        elif geometry == 'schwarzschild':
            label = f"SCHWARZSCHILD (E²={SCH_E2:.4f})"
        else:
            label = f"de SITTER H={DS_H} (Ricci only, null test)"

        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}", flush=True)

        cj_list, cj_phi_list, ratio_list = [], [], []
        seeds_data = []

        for si in range(M_SEEDS):
            res = run_seed(si, geometry, eps=eps)
            cj_list.append(res['CJ'])
            cj_phi_list.append(res['CJ_phi'])
            ratio_list.append(res['ratio_phi_cj'])
            seeds_data.append(res)

            if (si + 1) % 5 == 0 or si == 0:
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"CJ={res['CJ']:.6f}  "
                      f"CJ_φ={res['CJ_phi']:.6f}  "
                      f"ratio={res['ratio_phi_cj']:.4f}  "
                      f"I_mean={res['I_mean']:.2f}  "
                      f"SJ_modes={res['n_sj_modes']}  "
                      f"({res['time_s']:.1f}s)", flush=True)

        cj_arr = np.array(cj_list)
        cj_phi_arr = np.array(cj_phi_list)
        ratio_arr = np.array(ratio_list)

        cj_m, cj_se = float(cj_arr.mean()), float(cj_arr.std(ddof=1) / math.sqrt(M_SEEDS))
        cj_phi_m, cj_phi_se = float(cj_phi_arr.mean()), float(cj_phi_arr.std(ddof=1) / math.sqrt(M_SEEDS))
        t_cj = cj_m / (cj_se + 1e-30)
        t_phi = cj_phi_m / (cj_phi_se + 1e-30)
        corr = float(np.corrcoef(cj_arr, cj_phi_arr)[0, 1]) if len(cj_arr) >= 3 else 0

        summary = {
            'geometry': geometry, 'eps': eps, 'key': key,
            'CJ_mean': cj_m, 'CJ_se': cj_se, 'CJ_t': t_cj,
            'CJ_phi_mean': cj_phi_m, 'CJ_phi_se': cj_phi_se, 'CJ_phi_t': t_phi,
            'ratio_mean': float(ratio_arr.mean()),
            'ratio_std': float(ratio_arr.std(ddof=1)),
            'corr_CJ_CJphi': corr,
            'seeds': seeds_data,
        }
        all_results[key] = summary

        print(f"\n  SUMMARY {key}:")
        print(f"    CJ     = {cj_m:.6f} ± {cj_se:.6f} (t={t_cj:+.2f})")
        print(f"    CJ_φ   = {cj_phi_m:.6f} ± {cj_phi_se:.6f} (t={t_phi:+.2f})")
        print(f"    ratio CJ_φ/CJ = {ratio_arr.mean():.4f} ± {ratio_arr.std(ddof=1):.4f}")
        print(f"    Corr(CJ, CJ_φ) = {corr:+.4f}")
        print(flush=True)

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*72}")
    print("FINAL COMPARISON: CJ vs CJ_φ")
    print(f"{'='*72}")
    print(f"  {'Config':<20} {'CJ t':<10} {'CJ_φ t':<10} {'ratio':<12} {'corr':<10}")
    print(f"  {'-'*62}")
    for key in ['ppw_eps1', 'ppw_eps3', 'sch', 'ds']:
        r = all_results[key]
        print(f"  {key:<20} {r['CJ_t']:+.2f}     {r['CJ_phi_t']:+.2f}     "
              f"{r['ratio_mean']:.4f}±{r['ratio_std']:.4f}  {r['corr_CJ_CJphi']:+.4f}")

    # Key questions
    print(f"\n  KEY QUESTIONS:")
    ppw3 = all_results['ppw_eps3']
    sch = all_results['sch']
    ds = all_results['ds']
    print(f"    1. Does CJ_φ detect Weyl?       ppw3 t={ppw3['CJ_phi_t']:+.2f}, sch t={sch['CJ_phi_t']:+.2f}")
    print(f"    2. Does CJ_φ reject Ricci?       ds t={ds['CJ_phi_t']:+.2f} (should be ~0)")
    print(f"    3. Is CJ_φ different from CJ?    ratio = {ppw3['ratio_mean']:.4f}")
    print(f"    4. Are CJ and CJ_φ correlated?   ppw3 r={ppw3['corr_CJ_CJphi']:+.3f}, sch r={sch['corr_CJ_CJphi']:+.3f}")

    # Save
    outfile = 'analysis/fnd1_data/bridge_cj_phi_test.json'
    for key in all_results:
        if 'seeds' in all_results[key]:
            for sr in all_results[key]['seeds']:
                for k in list(sr.keys()):
                    if isinstance(sr[k], np.ndarray):
                        sr[k] = sr[k].tolist()
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"\nSaved to {outfile}", flush=True)
