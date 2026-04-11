import argparse, json, math
import numpy as np
from run_universal import sprinkle_local_diamond, build_hasse_from_predicate, Y_from_graph, bulk_mask, ppwave_exact_preds, geometry_riemann, jet_preds, excess_kurtosis
from schwarzschild_exact_local_tools import map_rnc_to_schwarzschild_expmap, schwarzschild_exact_midpoint_preds_from_mapped


def a_align(Y0, Y1, pts, T, zeta, n_tau=5, n_rho=3, n_depth=3):
    mask = bulk_mask(pts, T, zeta)
    if not np.any(mask):
        return 0.0
    X = Y0 - Y0[mask].mean()
    dY = Y1 - Y0
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 0.5 * n_tau).astype(int), 0, n_tau - 1)
    rho_bin = np.clip(np.floor(rho_hat * n_rho).astype(int), 0, n_rho - 1)
    # flat depth
    parents0, _ = build_hasse_from_predicate(pts, lambda P, i: (ppwave_exact_preds(P, i, eps=0.0) if False else None))
    raise RuntimeError('This helper is not intended for standalone use without precomputed depth.')


def compute_depth(parents):
    n = len(parents)
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        p = parents[i]
        if len(p) > 0:
            depth[i] = max(depth[j] for j in p) + 1
    return depth


def a_align_from_graphs(pts, T, zeta, parents0, children0, Y0, Y1, n_tau=5, n_rho=3, n_depth=3):
    mask = bulk_mask(pts, T, zeta)
    X = Y0 - Y0[mask].mean()
    dY = Y1 - Y0
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 0.5 * n_tau).astype(int), 0, n_tau - 1)
    rho_bin = np.clip(np.floor(rho_hat * n_rho).astype(int), 0, n_rho - 1)
    depth = compute_depth(parents0)
    depth_tercile = np.clip((depth * n_depth) // (depth.max() + 1), 0, n_depth - 1)
    strata = tau_bin * (n_rho * n_depth) + rho_bin * n_depth + depth_tercile
    idx = np.where(mask)[0]
    labels = np.unique(strata[idx])
    total = float(idx.size)
    acc = 0.0
    for lab in labels:
        j = idx[strata[idx] == lab]
        if j.size < 2:
            continue
        X2 = X[j] ** 2
        d2 = dY[j] ** 2
        cov = float(np.mean((X2 - X2.mean()) * (d2 - d2.mean())))
        acc += (j.size / total) * cov
    return acc


def run_one(N=1000, T=1.0, eps=3.0, M=0.05, r0=0.5, zeta=0.15, seed=1234):
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)

    # flat
    parents0, children0 = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, np.zeros((4,4,4,4))))
    Y0 = Y_from_graph(parents0, children0)

    # ppw exact
    parentsP, childrenP = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
    YP = Y_from_graph(parentsP, childrenP)

    # sch quadratic jet
    Rsch = geometry_riemann('schwarzschild_local', {'M': M, 'r0': r0})
    parentsQ, childrenQ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, Rsch))
    YQ = Y_from_graph(parentsQ, childrenQ)

    # sch exact midpoint after expmap
    mapped = map_rnc_to_schwarzschild_expmap(pts, M=M, r0=r0)
    parentsX, childrenX = build_hasse_from_predicate(pts, lambda P, i: schwarzschild_exact_midpoint_preds_from_mapped(mapped, i, M=M))
    YX = Y_from_graph(parentsX, childrenX)

    mask = bulk_mask(pts, T, zeta)
    k0 = excess_kurtosis(Y0[mask])
    out = {
        'dk_ppw_exact': excess_kurtosis(YP[mask]) - k0,
        'dk_sch_quad': excess_kurtosis(YQ[mask]) - k0,
        'dk_sch_exact_mid': excess_kurtosis(YX[mask]) - k0,
        'aalign_ppw_exact': a_align_from_graphs(pts, T, zeta, parents0, children0, Y0, YP),
        'aalign_sch_quad': a_align_from_graphs(pts, T, zeta, parents0, children0, Y0, YQ),
        'aalign_sch_exact_mid': a_align_from_graphs(pts, T, zeta, parents0, children0, Y0, YX),
        'links_flat': int(sum(len(p) for p in parents0)),
        'links_sch_quad': int(sum(len(p) for p in parentsQ)),
        'links_sch_exact_mid': int(sum(len(p) for p in parentsX)),
    }
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=1000)
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=3.0)
    ap.add_argument('--M', type=float, default=0.05)
    ap.add_argument('--r0', type=float, default=0.5)
    ap.add_argument('--zeta', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = run_one(args.N, args.T, args.eps, args.M, args.r0, args.zeta, args.seed)
    txt = json.dumps(res, indent=2)
    print(txt, flush=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(txt)
