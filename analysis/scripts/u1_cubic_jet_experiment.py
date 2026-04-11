
#!/usr/bin/env python3
"""
U1 cubic-jet universality experiment.

Self-contained script with:
- cubic_jet_preds
- nabla_riemann_schwarzschild
- generic uint64 Hasse builder
- exact pp-wave predicate
- U1 experiment runner

Only dependencies: numpy, scipy
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.special import logsumexp
from scipy.stats import kurtosis as scipy_kurtosis

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return 0.0
    v = np.var(x)
    if v < 1e-15:
        return 0.0
    return float(scipy_kurtosis(x, fisher=True, bias=True))

def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(type(obj).__name__)

def sprinkle_local_diamond(N: int, T: float, rng: np.random.Generator) -> np.ndarray:
    pts = []
    tmin, tmax = -T/2.0, T/2.0
    batch = max(4096, 8*N)
    while len(pts) < N:
        arr = rng.uniform(tmin, tmax, size=(batch, 4))
        r = np.linalg.norm(arr[:, 1:], axis=1)
        mask = (np.abs(arr[:, 0]) + r) < (T/2.0)
        if np.any(mask):
            pts.extend(arr[mask].tolist())
    arr = np.asarray(pts[:N], dtype=np.float64)
    order = np.argsort(arr[:, 0], kind="mergesort")
    return arr[order]

def bulk_mask(pts: np.ndarray, T: float, zeta: float) -> np.ndarray:
    tau = pts[:, 0]
    r = np.linalg.norm(pts[:, 1:], axis=1)
    slack = T/2.0 - (np.abs(tau) + r)
    return slack >= zeta*T

def rotate_points_spatial(pts: np.ndarray, Q: np.ndarray) -> np.ndarray:
    out = np.array(pts, copy=True)
    out[:, 1:] = pts[:, 1:] @ Q.T
    return out

def riemann_vacuum_from_E(E: np.ndarray) -> np.ndarray:
    R = np.zeros((4,4,4,4), dtype=np.float64)
    delta = np.eye(3)
    # R_{0i0j}=E_{ij}
    for i in range(3):
        for j in range(3):
            v = float(E[i,j])
            if abs(v) < 1e-15:
                continue
            set_riemann_component(R, 0, i+1, 0, j+1, v)
    # purely spatial
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    v = -(
                        delta[i,k]*E[j,l] + delta[j,l]*E[i,k]
                        - delta[i,l]*E[j,k] - delta[j,k]*E[i,l]
                    )
                    if abs(v) < 1e-15:
                        continue
                    set_riemann_component(R, i+1, j+1, k+1, l+1, float(v))
    return R

def set_riemann_component(R: np.ndarray, a: int, b: int, c: int, d: int, val: float) -> None:
    entries = [
        (a,b,c,d,+val),
        (b,a,c,d,-val),
        (a,b,d,c,-val),
        (b,a,d,c,+val),
        (c,d,a,b,+val),
        (d,c,a,b,-val),
        (c,d,b,a,-val),
        (d,c,b,a,+val),
    ]
    for i,j,k,l,v in entries:
        R[i,j,k,l] = v

def riemann_ppwave_canonical(eps: float) -> np.ndarray:
    R = np.zeros((4,4,4,4), dtype=np.float64)
    ex = -0.5*eps
    ey = +0.5*eps
    set_riemann_component(R, 0,1,0,1, ex)
    set_riemann_component(R, 0,1,3,1, ex)
    set_riemann_component(R, 3,1,3,1, ex)
    set_riemann_component(R, 0,2,0,2, ey)
    set_riemann_component(R, 0,2,3,2, ey)
    set_riemann_component(R, 3,2,3,2, ey)
    return R

def riemann_schwarzschild_local(M: float, r0: float) -> np.ndarray:
    q = M / (r0**3)
    E = np.diag([-q, -q, 2.0*q]).astype(np.float64)
    return riemann_vacuum_from_E(E)

def nabla_riemann_ppwave(eps: float) -> np.ndarray:
    # Homogeneous quadratic pp-wave = Cahen-Wallach symmetric plane wave; ∇R = 0.
    return np.zeros((4,4,4,4,4), dtype=np.float64)

def nabla_riemann_schwarzschild(M: float, r0: float) -> np.ndarray:
    """
    Covariant derivative of the vacuum Riemann tensor in the static orthonormal frame
    with radial axis aligned with +z and E = diag(-q,-q,2q), q = M/r0^3.

    Model:
        ∇_0 R = 0
        ∇_k E_ij = alpha * (δ_ki n_j + δ_kj n_i + δ_ij n_k - 5 n_i n_j n_k),
        alpha = 3 * sqrt(1 - 2M/r0) * q / r0
    and ∇_k R is obtained by linearity from ∇_k E.
    """
    nab = np.zeros((4,4,4,4,4), dtype=np.float64)
    q = M / (r0**3)
    f = max(1.0 - 2.0*M/r0, 0.0)
    alpha = 3.0 * math.sqrt(f) * q / r0
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    delta = np.eye(3)
    gradE = np.zeros((3,3,3), dtype=np.float64)  # k,i,j
    for k in range(3):
        nk = n[k]
        for i in range(3):
            ni = n[i]
            for j in range(3):
                nj = n[j]
                gradE[k,i,j] = alpha * (
                    delta[k,i]*nj + delta[k,j]*ni + delta[i,j]*nk - 5.0*ni*nj*nk
                )
    # sanity: k=z gives radial derivative of diag(-q,-q,2q)
    # Fill nabla_k R from gradE[k]
    for k in range(3):
        dR = riemann_vacuum_from_E(gradE[k])
        nab[k+1, :, :, :, :] = dR
    return nab

def minkowski_preds(pts: np.ndarray, i: int, tol: float = 1e-12) -> np.ndarray:
    x = pts[i]
    y = pts[:i]
    d = x[None, :] - y
    s2 = -d[:,0]**2 + np.sum(d[:,1:]**2, axis=1)
    return (d[:,0] > tol) & (s2 <= tol)

def jet_preds(pts: np.ndarray, i: int, R_abcd: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    x = pts[i]
    y = pts[:i]
    d = x[None, :] - y
    m = 0.5*(x[None, :] + y)
    gmid = ETA[None,:,:] - (1.0/3.0) * np.einsum('aubv,nu,nv->nab', R_abcd, m, m, optimize=True)
    s2 = np.einsum('na,nab,nb->n', d, gmid, d, optimize=True)
    return (d[:,0] > tol) & (s2 <= tol)

def cubic_jet_preds(pts: np.ndarray, i: int, R_abcd: np.ndarray, nabla_R: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Cubic RNC jet causal predicate using midpoint metric:
        g_ab(m) = eta_ab
                - (1/3) R_aμbν m^μ m^ν
                - (1/6) ∇_γ R_aμbν m^γ m^μ m^ν
    """
    x = pts[i]
    y = pts[:i]
    d = x[None, :] - y
    m = 0.5*(x[None, :] + y)
    quad = (1.0/3.0) * np.einsum('aubv,nu,nv->nab', R_abcd, m, m, optimize=True)
    cubic = (1.0/6.0) * np.einsum('gaubv,ng,nu,nv->nab', nabla_R, m, m, m, optimize=True)
    gmid = ETA[None,:,:] - quad - cubic
    s2 = np.einsum('na,nab,nb->n', d, gmid, d, optimize=True)
    return (d[:,0] > tol) & (s2 <= tol)

def vneeded_ppwave_exact(U: np.ndarray, xA: np.ndarray, xB: np.ndarray,
                         yA: np.ndarray, yB: np.ndarray, eps: float,
                         tol: float = 1e-12, series_threshold: float = 1e-4) -> np.ndarray:
    U, xA, xB, yA, yB = np.broadcast_arrays(U, xA, xB, yA, yB)
    out = np.full(U.shape, np.inf, dtype=np.float64)
    maskU = U > tol
    if not np.any(maskU):
        return out

    Um = U[maskU]
    x0 = xA[maskU]
    x1 = xB[maskU]
    y0 = yA[maskU]
    y1 = yB[maskU]

    if abs(eps) < tol:
        out[maskU] = ((x1 - x0) ** 2 + (y1 - y0) ** 2) / Um
        return out

    def series2(Um_, x0_, x1_, y0_, y1_, eps_):
        term0 = ((x1_ - x0_) ** 2 + (y1_ - y0_) ** 2) / Um_
        term1 = (eps_ * Um_ / 6.0) * (
            (x0_ * x0_ + x0_ * x1_ + x1_ * x1_)
            - (y0_ * y0_ + y0_ * y1_ + y1_ * y1_)
        )
        term2 = -(eps_ * eps_) * (Um_ ** 3) / 720.0 * (
            4.0 * (x0_ * x0_ + x1_ * x1_ + y0_ * y0_ + y1_ * y1_)
            + 7.0 * (x0_ * x1_ + y0_ * y1_)
        )
        return term0 + term1 + term2

    w = math.sqrt(abs(eps) / 2.0)
    eta = w * Um
    small = np.abs(eta) < series_threshold
    conj = eta >= (math.pi - tol)
    vals = np.empty_like(Um)
    vals[conj] = -np.inf

    reg = ~conj
    if np.any(reg & small):
        idx = reg & small
        vals[idx] = series2(Um[idx], x0[idx], x1[idx], y0[idx], y1[idx], eps)

    if np.any(reg & ~small):
        idx = reg & ~small
        et = eta[idx]
        xa = x0[idx]
        xb = x1[idx]
        ya = y0[idx]
        yb = y1[idx]
        if eps > 0:
            Sx = w * (((xa * xa + xb * xb) * np.cosh(et) - 2.0 * xa * xb) / np.sinh(et))
            Sy = w * (((ya * ya + yb * yb) * np.cos(et) - 2.0 * ya * yb) / np.sin(et))
        else:
            Sx = w * (((xa * xa + xb * xb) * np.cos(et) - 2.0 * xa * xb) / np.sin(et))
            Sy = w * (((ya * ya + yb * yb) * np.cosh(et) - 2.0 * ya * yb) / np.sinh(et))
        vals[idx] = Sx + Sy

    out[maskU] = vals
    return out

def ppwave_exact_preds(pts: np.ndarray, i: int, eps: float, tol: float = 1e-12) -> np.ndarray:
    t = pts[:,0]; x = pts[:,1]; y = pts[:,2]; z = pts[:,3]
    U = (t[i] + z[i]) - (t[:i] + z[:i])
    V = (t[i] - z[i]) - (t[:i] - z[:i])
    Vneed = vneeded_ppwave_exact(U, x[:i], np.full(i,x[i]), y[:i], np.full(i,y[i]), eps, tol=tol)
    return (U > tol) & (V >= Vneed - tol)

def build_hasse_uint64(pts: np.ndarray, pred_fn) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generic uint64 transitive-reduction builder.
    pred_fn(pts, i) -> bool mask over j < i.
    """
    n = len(pts)
    n_words = (n + 63) // 64
    one = np.uint64(1)

    past = np.zeros((n, n_words), dtype=np.uint64)
    parents: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n)]
    children_lists: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        rel_mask = np.asarray(pred_fn(pts, i), dtype=bool)
        if rel_mask.size != i:
            raise ValueError(f"Predicate for node {i} returned mask length {rel_mask.size}, expected {i}")
        rel_preds = np.flatnonzero(rel_mask)
        if rel_preds.size == 0:
            continue

        covered = np.zeros(n_words, dtype=np.uint64)
        direct = []

        for j in rel_preds[::-1]:
            jj = int(j)
            w = jj >> 6
            bit = one << np.uint64(jj & 63)
            if (covered[w] & bit) != 0:
                continue
            direct.append(jj)
            covered |= past[jj]
            covered[w] |= bit

        direct.sort()
        arr = np.asarray(direct, dtype=np.int32)
        parents[i] = arr

        row = np.zeros(n_words, dtype=np.uint64)
        for jj in arr:
            children_lists[jj].append(i)
            row |= past[jj]
            row[jj >> 6] |= one << np.uint64(jj & 63)
        past[i] = row

    children = [np.asarray(ch, dtype=np.int32) for ch in children_lists]
    return parents, children

def log_path_counts(parents: List[np.ndarray], children: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(parents)
    log_pd = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = parents[i]
        if p.size == 0:
            log_pd[i] = 0.0
        else:
            log_pd[i] = float(logsumexp(log_pd[p]))
    log_pu = np.zeros(n, dtype=np.float64)
    for i in range(n-1, -1, -1):
        ch = children[i]
        if ch.size == 0:
            log_pu[i] = 0.0
        else:
            log_pu[i] = float(logsumexp(log_pu[ch]))
    return log_pd, log_pu

def Y_from_graph(parents: List[np.ndarray], children: List[np.ndarray]) -> np.ndarray:
    log_pd, log_pu = log_path_counts(parents, children)
    L = log_pd + log_pu
    return (np.maximum(L, 0.0) + np.log1p(np.exp(-np.abs(L)))) / math.log(2.0)

def ppwave_E2(eps: float) -> float:
    return 0.5 * eps * eps

def sch_E2(M: float, r0: float) -> float:
    q = M / (r0**3)
    return 6.0 * q * q

def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se = float(std / math.sqrt(x.size)) if x.size > 1 else 0.0
    tstat = float(mean / se) if se > 1e-15 else 0.0
    return {"mean": mean, "std": std, "se": se, "t_stat": tstat, "n": int(x.size)}

def run_one_condition(geometry: str, predicate: str, T: float, q: float, N: int, n_seeds: int,
                      zeta: float = 0.15, r0: float = 0.50, seed_base: int = 1234500) -> Dict[str, Any]:
    per_seed = []
    tol = 1e-12

    if geometry == "ppwave":
        eps = q / (T*T)
        R = riemann_ppwave_canonical(eps)
        nabla_R = nabla_riemann_ppwave(eps)
        E2 = ppwave_E2(eps)
    elif geometry == "schwarzschild_local":
        M = q * (r0**3) / (T*T)
        R = riemann_schwarzschild_local(M, r0)
        nabla_R = nabla_riemann_schwarzschild(M, r0)
        E2 = sch_E2(M, r0)
    else:
        raise ValueError("geometry must be 'ppwave' or 'schwarzschild_local'")

    for s in range(n_seeds):
        rng = np.random.default_rng(seed_base + s)
        pts = sprinkle_local_diamond(N, T, rng)

        parents0, children0 = build_hasse_uint64(pts, lambda P, i: minkowski_preds(P, i, tol=tol))
        Y0 = Y_from_graph(parents0, children0)

        if geometry == "ppwave" and predicate == "exact":
            parents1, children1 = build_hasse_uint64(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps, tol=tol))
        elif predicate == "quadratic":
            parents1, children1 = build_hasse_uint64(pts, lambda P, i: jet_preds(P, i, R, tol=tol))
        elif predicate == "cubic":
            parents1, children1 = build_hasse_uint64(pts, lambda P, i: cubic_jet_preds(P, i, R, nabla_R, tol=tol))
        else:
            raise ValueError("predicate must be exact/quadratic/cubic (exact only for ppwave)")

        Y1 = Y_from_graph(parents1, children1)
        mask = bulk_mask(pts, T, zeta)
        dk = excess_kurtosis(Y1[mask]) - excess_kurtosis(Y0[mask])
        per_seed.append(dk)

    stats = summarize(np.asarray(per_seed))
    out = {
        "geometry": geometry,
        "predicate": predicate,
        "T": T,
        "q": q,
        "N": N,
        "n_seeds": n_seeds,
        "zeta": zeta,
        "E_squared": E2,
        "per_seed": per_seed,
        "dk_mean": stats["mean"],
        "dk_std": stats["std"],
        "dk_se": stats["se"],
        "t_stat": stats["t_stat"],
        "A_E": stats["mean"] / max((T**4) * E2, 1e-15),
    }
    if geometry == "ppwave":
        out["eps"] = eps
    else:
        out["M"] = M
        out["r0"] = r0
    return out

def default_u1_conditions() -> List[Dict[str, Any]]:
    conds = []
    for T in (0.70, 0.50):
        for q in (0.10, 0.20):
            conds.append({"geometry": "ppwave", "predicate": "cubic", "T": T, "q": q, "N": 10000, "n_seeds": 50, "zeta": 0.15})
            conds.append({"geometry": "ppwave", "predicate": "exact", "T": T, "q": q, "N": 10000, "n_seeds": 50, "zeta": 0.15})
            conds.append({"geometry": "schwarzschild_local", "predicate": "cubic", "T": T, "q": q, "N": 10000, "n_seeds": 50, "zeta": 0.15, "r0": 0.50})
    return conds

def summarize_u1(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    idx = {(r["geometry"], r["predicate"], r["T"], r["q"]): r for r in results}
    summary = {"pairs": []}
    ratios = []
    calib = []
    for T in (0.70, 0.50):
        for q in (0.10, 0.20):
            pp_c = idx[("ppwave", "cubic", T, q)]
            pp_e = idx[("ppwave", "exact", T, q)]
            sch_c = idx[("schwarzschild_local", "cubic", T, q)]
            Jpp = pp_e["dk_mean"] / pp_c["dk_mean"] if abs(pp_c["dk_mean"]) > 1e-15 else float("nan")
            ratio = sch_c["A_E"] / pp_c["A_E"] if abs(pp_c["A_E"]) > 1e-15 else float("nan")
            summary["pairs"].append({
                "T": T, "q": q,
                "A_pp_cubic": pp_c["A_E"],
                "A_pp_exact": pp_e["A_E"],
                "A_sch_cubic": sch_c["A_E"],
                "J_pp_exact_over_cubic": Jpp,
                "ratio_sch_over_pp_cubic": ratio,
            })
            if np.isfinite(ratio):
                ratios.append(ratio)
            if np.isfinite(Jpp):
                calib.append(Jpp)
    if ratios:
        rmean = float(np.mean(ratios))
        if 0.80 <= rmean <= 1.25:
            verdict = "PASS"
        elif 0.65 <= rmean <= 1.50:
            verdict = "BORDERLINE"
        else:
            verdict = "FAIL"
        summary["universality_ratio_mean"] = rmean
        summary["verdict"] = verdict
    if calib:
        summary["J_pp_mean"] = float(np.mean(calib))
    return summary

def smoke_test(out_dir: str) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tests = [
        {"geometry": "ppwave", "predicate": "cubic", "T": 0.70, "q": 0.10, "N": 500, "n_seeds": 5, "zeta": 0.15},
        {"geometry": "ppwave", "predicate": "exact", "T": 0.70, "q": 0.10, "N": 500, "n_seeds": 5, "zeta": 0.15},
        {"geometry": "schwarzschild_local", "predicate": "quadratic", "T": 1.00, "q": 0.40, "N": 500, "n_seeds": 5, "zeta": 0.15, "r0": 0.50},
        {"geometry": "schwarzschild_local", "predicate": "cubic", "T": 1.00, "q": 0.40, "N": 500, "n_seeds": 5, "zeta": 0.15, "r0": 0.50},
    ]
    out = []
    for c in tests:
        out.append(run_one_condition(**c))
    p = Path(out_dir) / "smoke_test.json"
    p.write_text(json.dumps(out, indent=2, default=json_default), encoding="utf-8")
    return {"n_conditions": len(out), "path": str(p)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["u1", "smoke"], default="u1")
    ap.add_argument("--out", default="u1_runs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        res = smoke_test(str(out_dir))
        print(json.dumps(res, indent=2, default=json_default))
        return

    conds = default_u1_conditions()
    results = []
    for c in conds:
        print(f"Running {c['geometry']} {c['predicate']} T={c['T']} q={c['q']} ...", flush=True)
        res = run_one_condition(**c)
        results.append(res)
        name = f"{c['geometry']}_{c['predicate']}_T{str(c['T']).replace('.','p')}_q{str(c['q']).replace('.','p')}.json"
        (out_dir / name).write_text(json.dumps(res, indent=2, default=json_default), encoding="utf-8")

    summary = summarize_u1(results)
    (out_dir / "u1_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=json_default))

if __name__ == "__main__":
    main()
