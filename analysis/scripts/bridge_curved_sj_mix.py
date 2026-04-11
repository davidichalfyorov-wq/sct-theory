#!/usr/bin/env python3
"""
Bridge: Curved-SJ mixed local-response test.

CJ_mix = Σ_B w_B Cov_B(|X|, δY · δI)

where δI_i = Σ_{j∈N₀(i)} [|W_curved(i,j)|² - |W_flat(i,j)|²]
is the LOCAL field-response (curved minus flat SJ vacuum intensity).

Both δY = O(ε) and δI = O(ε), so the product is O(ε²) — correct
quadratic Weyl channel.

This is the LAST untested bridge route. If it fails, CJ is genuinely
new — not a standard QFT coefficient.
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, riemann_schwarzschild_local,
    build_hasse_from_predicate, bulk_mask,
)

N = 2000
T = 1.0
ZETA = 0.15
M_SEEDS = 20

SCH_M = 0.05
SCH_R0 = 0.50
SCH_R_ABCD = riemann_schwarzschild_local(SCH_M, SCH_R0)
SCH_E2 = 6.0 * SCH_M**2 / SCH_R0**6


def hasse_to_link_matrix(parents, n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                L[int(j), i] = 1.0
    return L


def compute_sj_wightman(L):
    """SJ Wightman from link matrix. Returns W and n_modes."""
    n = L.shape[0]
    PJ = L - L.T
    H = 1j * PJ
    threshold = math.sqrt(n) / (4 * math.pi)

    evals, evecs = np.linalg.eigh(H)
    use_mask = evals > threshold
    n_modes = int(use_mask.sum())

    if n_modes == 0:
        return np.zeros((n, n)), 0

    V_pos = evecs[:, use_mask]
    sigma_pos = evals[use_mask]
    W = (V_pos * sigma_pos[None, :]) @ V_pos.conj().T
    return W, n_modes


def compute_local_field_intensity(W, parents, children, n):
    """Per-element field intensity I_i = Σ_{j ∈ flat_neighbors(i)} |W_ij|².
    Neighbors = flat Hasse parents + children (symmetric local neighborhood).
    """
    I = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if parents[i] is not None and len(parents[i]) > 0:
            for j in parents[i]:
                I[i] += abs(W[i, int(j)])**2
        if children[i] is not None and len(children[i]) > 0:
            for j in children[i]:
                I[i] += abs(W[i, int(j)])**2
    return I


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
    """Plain CJ."""
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


def compute_CJ_mix(Y0, deltaY, deltaI, mask, strata):
    """CJ_mix = Σ_B w_B Cov_B(|X|, δY · δI).
    δY and δI are both O(ε), product is O(ε²).
    """
    X = Y0[mask] - np.mean(Y0[mask])
    product = deltaY[mask] * deltaI[mask]  # O(ε²)
    strata_m = strata[mask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = (np.mean(np.abs(X[idx]) * product[idx])
               - np.mean(np.abs(X[idx])) * np.mean(product[idx]))
        total += w * cov
    return float(total)


def compute_CJ_mix_abs(Y0, deltaY, deltaI, mask, strata):
    """CJ_mix_abs = Σ_B w_B Cov_B(|X|, |δY| · |δI|).
    Absolute values to avoid sign cancellation.
    """
    X = Y0[mask] - np.mean(Y0[mask])
    product = np.abs(deltaY[mask]) * np.abs(deltaI[mask])
    strata_m = strata[mask]
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = (np.mean(np.abs(X[idx]) * product[idx])
               - np.mean(np.abs(X[idx])) * np.mean(product[idx]))
        total += w * cov
    return float(total)


def run_seed(seed_idx, geometry, eps=None, base_seed=9700000):
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
    else:
        parC, chC = build_hasse_from_predicate(
            pts, lambda P, i: jet_preds(P, i, SCH_R_ABCD))
        E2 = SCH_E2

    L_curved = hasse_to_link_matrix(parC, N)

    # SJ Wightman: BOTH flat and curved
    W_flat, n_modes_flat = compute_sj_wightman(L_flat)
    W_curved, n_modes_curved = compute_sj_wightman(L_curved)

    # Local field intensity: flat and curved
    I_flat = compute_local_field_intensity(W_flat, par0, ch0, N)
    I_curved = compute_local_field_intensity(W_curved, par0, ch0, N)
    # Note: using FLAT neighborhood for both (same N₀)

    # δI = curved field intensity - flat field intensity
    deltaI = I_curved - I_flat

    # Path counts and δY
    Y0 = Y_from_graph(par0, ch0)
    YC = Y_from_graph(parC, chC)
    deltaY = YC - Y0
    strata = make_strata(pts, par0, T)

    # Plain CJ
    cj = compute_CJ(Y0, deltaY, bmask, strata)

    # CJ_mix: Cov(|X|, δY · δI)
    cj_mix = compute_CJ_mix(Y0, deltaY, deltaI, bmask, strata)

    # CJ_mix_abs: Cov(|X|, |δY| · |δI|) — no sign cancellation variant
    cj_mix_abs = compute_CJ_mix_abs(Y0, deltaY, deltaI, bmask, strata)

    # Diagnostics
    dI_bulk = deltaI[bmask]
    dI_mean = float(np.mean(dI_bulk))
    dI_std = float(np.std(dI_bulk))
    dI_frac_pos = float(np.mean(dI_bulk > 0))

    # Correlation between δY and δI in bulk
    dY_bulk = deltaY[bmask]
    corr_dY_dI = float(np.corrcoef(dY_bulk, dI_bulk)[0, 1]) if len(dY_bulk) > 3 else 0.0

    t_total = time.time() - t0

    return {
        'seed': seed_idx, 'geometry': geometry, 'eps': eps, 'E2': E2,
        'CJ': cj,
        'CJ_mix': cj_mix,
        'CJ_mix_abs': cj_mix_abs,
        'dI_mean': dI_mean,
        'dI_std': dI_std,
        'dI_frac_pos': dI_frac_pos,
        'corr_dY_dI': corr_dY_dI,
        'n_modes_flat': n_modes_flat,
        'n_modes_curved': n_modes_curved,
        'time_s': t_total,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("BRIDGE: Curved-SJ Mixed Local-Response Test")
    print(f"N={N}, T={T}, zeta={ZETA}, M={M_SEEDS}")
    print(f"CJ_mix = Σ_B w_B Cov_B(|X|, δY · δI)")
    print(f"δI_i = Σ_{{N₀}} [|W_curved|² - |W_flat|²]  (curved-flat field response)")
    print("=" * 72, flush=True)

    configs = [
        ('ppwave', 1.0, 'ppw_eps1'),
        ('ppwave', 3.0, 'ppw_eps3'),
        ('schwarzschild', None, 'sch'),
    ]

    all_results = {}

    for geometry, eps, key in configs:
        if geometry == 'ppwave':
            E2 = eps**2 / 2.0
            label = f"PP-WAVE eps={eps} (E²={E2:.2f})"
        else:
            label = f"SCHWARZSCHILD (E²={SCH_E2:.4f})"

        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}", flush=True)

        cj_list, mix_list, mix_abs_list = [], [], []
        corr_dydI_list = []
        seeds_data = []

        for si in range(M_SEEDS):
            res = run_seed(si, geometry, eps=eps)
            cj_list.append(res['CJ'])
            mix_list.append(res['CJ_mix'])
            mix_abs_list.append(res['CJ_mix_abs'])
            corr_dydI_list.append(res['corr_dY_dI'])
            seeds_data.append(res)

            if (si + 1) % 5 == 0 or si == 0:
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"CJ={res['CJ']:.6f}  "
                      f"mix={res['CJ_mix']:+.6f}  "
                      f"mix_abs={res['CJ_mix_abs']:.6f}  "
                      f"r(δY,δI)={res['corr_dY_dI']:+.3f}  "
                      f"δI_mean={res['dI_mean']:+.3f}  "
                      f"({res['time_s']:.1f}s)", flush=True)

        cj_a = np.array(cj_list)
        mix_a = np.array(mix_list)
        mix_abs_a = np.array(mix_abs_list)

        def stats(a):
            m = float(a.mean())
            se = float(a.std(ddof=1) / math.sqrt(len(a)))
            t = m / (se + 1e-30)
            return m, se, t

        cj_m, cj_se, cj_t = stats(cj_a)
        mix_m, mix_se, mix_t = stats(mix_a)
        mixabs_m, mixabs_se, mixabs_t = stats(mix_abs_a)

        corr_cj_mix = float(np.corrcoef(cj_a, mix_a)[0, 1]) if len(cj_a) >= 3 else 0
        corr_cj_mixabs = float(np.corrcoef(cj_a, mix_abs_a)[0, 1]) if len(cj_a) >= 3 else 0
        mean_corr_dydI = float(np.mean(corr_dydI_list))

        summary = {
            'geometry': geometry, 'eps': eps, 'key': key, 'E2': E2 if geometry == 'ppwave' else SCH_E2,
            'CJ_mean': cj_m, 'CJ_se': cj_se, 'CJ_t': cj_t,
            'mix_mean': mix_m, 'mix_se': mix_se, 'mix_t': mix_t,
            'mixabs_mean': mixabs_m, 'mixabs_se': mixabs_se, 'mixabs_t': mixabs_t,
            'corr_CJ_mix': corr_cj_mix,
            'corr_CJ_mixabs': corr_cj_mixabs,
            'mean_corr_dY_dI': mean_corr_dydI,
            'seeds': seeds_data,
        }
        all_results[key] = summary

        print(f"\n  SUMMARY {key}:")
        print(f"    CJ       = {cj_m:.6f} ± {cj_se:.6f} (t={cj_t:+.2f})")
        print(f"    CJ_mix   = {mix_m:+.6f} ± {mix_se:.6f} (t={mix_t:+.2f})")
        print(f"    CJ_mix_a = {mixabs_m:.6f} ± {mixabs_se:.6f} (t={mixabs_t:+.2f})")
        print(f"    Corr(CJ, CJ_mix)     = {corr_cj_mix:+.4f}")
        print(f"    Corr(CJ, CJ_mix_abs) = {corr_cj_mixabs:+.4f}")
        print(f"    Mean corr(δY, δI)    = {mean_corr_dydI:+.4f}")
        print(flush=True)

    # FINAL TABLE
    print(f"\n{'='*72}")
    print("FINAL TABLE: Plain CJ vs Curved-SJ Mixed")
    print(f"{'='*72}")
    print(f"  {'Config':<15} {'CJ t':<8} {'mix t':<8} {'mix_abs t':<10} "
          f"{'r(CJ,mix)':<12} {'r(δY,δI)':<10}")
    print(f"  {'-'*65}")
    for key in ['ppw_eps1', 'ppw_eps3', 'sch']:
        r = all_results[key]
        print(f"  {key:<15} {r['CJ_t']:+.2f}   {r['mix_t']:+.2f}   {r['mixabs_t']:+.2f}      "
              f"{r['corr_CJ_mix']:+.4f}     {r['mean_corr_dY_dI']:+.4f}")

    # KEY VERDICT
    print(f"\n  VERDICT:")
    ppw3 = all_results['ppw_eps3']
    sch = all_results['sch']
    print(f"    CJ_mix nontrivial?  ppw3 t={ppw3['mix_t']:+.2f}, sch t={sch['mix_t']:+.2f}")
    print(f"    CJ_mix ≠ CJ?       corr = {ppw3['corr_CJ_mix']:+.3f} (ppw3), {sch['corr_CJ_mix']:+.3f} (sch)")
    print(f"    δY correlated with δI?  {ppw3['mean_corr_dY_dI']:+.3f} (ppw3), {sch['mean_corr_dY_dI']:+.3f} (sch)")

    nontrivial = abs(ppw3['mix_t']) > 2.0 and abs(ppw3['corr_CJ_mix']) < 0.9
    if nontrivial:
        print(f"    → CJ_mix is NONTRIVIAL and DIFFERENT from CJ. Bridge candidate ALIVE.")
    else:
        if abs(ppw3['mix_t']) < 2.0:
            print(f"    → CJ_mix NOT SIGNIFICANT. Last bridge route DEAD.")
        else:
            print(f"    → CJ_mix ≈ CJ (corr > 0.9). Trivial dressing. Bridge DEAD.")
    print(f"    → CJ is {'likely a genuinely new observable' if not nontrivial else 'possibly connected to field response'}.")

    outfile = 'analysis/fnd1_data/bridge_curved_sj_mix.json'
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
