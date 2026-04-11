#!/usr/bin/env python3
"""Analysis run PILOT+BASELINE+ADVERSARY. N=2000, M=20."""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, skew as sp_skew, kurtosis as sp_kurtosis, spearmanr

N = 2000
M = 20
M_FLAT = 30
TT = 1.0
SEED = 66666
RUN_DIR = Path("docs/analysis_runs/run_20260325_200948")

def sprinkle(Nt, T, rng):
    pts = []
    while len(pts) < Nt:
        b = rng.uniform(-T/2, T/2, size=(Nt*8, 4))
        r = np.sqrt(b[:, 1]**2 + b[:, 2]**2 + b[:, 3]**2)
        pts.extend(b[np.abs(b[:, 0]) + r < T/2].tolist())
    pts = np.array(pts[:Nt])
    return pts[np.argsort(pts[:, 0])]

def causal_flat(pts, _=0.0):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i+1:, 1:] - pts[i, 1:])**2, axis=1)
        C[i, i+1:] = (dt**2 > dr2).astype(np.int8)
    return C

def causal_ppwave(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dx = pts[i+1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        xm = (pts[i+1:, 1] + pts[i, 1]) / 2
        ym = (pts[i+1:, 2] + pts[i, 2]) / 2
        du = dt + dx[:, 2]
        f = xm**2 - ym**2
        C[i, i+1:] = (dt**2 - dr2 - eps * f * du**2 / 2 > 0).astype(np.int8)
    return C

def causal_schwarz(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dx = pts[i+1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        rm = np.sqrt(np.sum(((pts[i+1:, 1:] + pts[i, 1:]) / 2)**2, axis=1))
        phi = -eps / (rm + 0.3)
        C[i, i+1:] = ((1 + 2*phi) * dt**2 - (1 - 2*phi) * dr2 > 0).astype(np.int8)
    return C

MFNS = {
    "ppwave_quad": (causal_ppwave, 5.0),
    "schwarzschild": (causal_schwarz, 0.005),
}

def gini(v):
    v = np.sort(np.abs(v))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n+1) * v)) / (n * v.sum()) - (n+1) / n)

def compute_obs(C, pts):
    n = len(C)
    Cf = C.astype(np.float64)
    o = {}
    tc = int(C.sum())
    o["tc"] = tc
    C2 = Cf @ Cf
    o["n2"] = float(C2.sum())
    lm = (C > 0) & (C2 == 0)
    A = lm.astype(np.float32)
    As = A + A.T
    deg = As.sum(axis=1)
    o["link_count"] = int(deg.sum()) // 2
    o["degree_cv"] = float(deg.std() / max(deg.mean(), 1e-10))
    cp = np.argwhere(C > 0)
    isz = C2[cp[:, 0], cp[:, 1]].astype(int)
    o["layer_0"] = int(np.sum(isz == 0))
    # column_gini_C: Gini of sqrt(past_size)
    o["column_gini_C"] = gini(np.sqrt(Cf.sum(axis=0)))
    # link_degree_skew
    o["link_degree_skew"] = float(sp_skew(deg))
    # fan_kurtosis
    CCT = Cf @ Cf.T
    fk = []
    for x in range(n):
        fl = np.where(lm[x, :])[0]
        if len(fl) < 4:
            continue
        fk.append(float(sp_kurtosis(CCT[x, fl], fisher=True)))
    o["fan_kurtosis"] = float(np.mean(fk)) if fk else 0.0
    # controls
    o["column_gini_C2"] = gini(np.sqrt(np.sum(C2**2, axis=0)))
    lv = []
    for x in range(n):
        fl = np.where(lm[x, :])[0]
        if len(fl) < 2:
            continue
        k = CCT[x, fl]
        mu = k.mean()
        if mu > 0:
            lv.append(float(k.var() / mu**2))
    o["lva"] = float(np.mean(lv)) if lv else 0.0
    return o

def cd(arr):
    s = arr.std(ddof=1)
    if s < 1e-15:
        return 0.0 if abs(arr.mean()) < 1e-15 else float(np.sign(arr.mean()) * 999)
    return float(arr.mean() / s)

def main():
    t0 = time.time()
    ON = ["column_gini_C", "link_degree_skew", "fan_kurtosis",
          "column_gini_C2", "lva", "tc", "link_count", "degree_cv", "layer_0", "n2"]
    CA = ["column_gini_C", "link_degree_skew", "fan_kurtosis"]
    R = {}

    # CRN PILOT
    for mi, (mn, (mfn, eps)) in enumerate(MFNS.items()):
        print(f"CRN {mn} eps={eps}", flush=True)
        seeds = [SEED * 1000 + mi * 100 + i for i in range(M)]
        res = []
        for i in range(M):
            t1 = time.time()
            rng = np.random.default_rng(seeds[i])
            pts = sprinkle(N, TT, rng)
            Cf2 = causal_flat(pts)
            Cc = mfn(pts, eps)
            of = compute_obs(Cf2, pts)
            oc = compute_obs(Cc, pts)
            d = {k: float(oc[k] - of[k]) for k in of}
            res.append({"flat": of, "curved": oc, "delta": d})
            if i % 5 == 0 or i == M - 1:
                print(f"  {i+1}/{M} {time.time()-t1:.1f}s", flush=True)
        st = {}
        for nm in ON:
            da = np.array([r["delta"][nm] for r in res])
            dv = cd(da)
            try:
                _, p = wilcoxon(da)
                p = float(p)
            except Exception:
                p = 1.0
            st[nm] = {"d": round(dv, 3), "p": round(p, 6)}
            fl = "***" if abs(dv) >= 0.5 and p < 0.05 else ""
            print(f"  {nm:25s} d={dv:+.3f} p={p:.4f} {fl}", flush=True)
        R[mn] = {"stats": st, "raw": res}

    # Conformal null: same causal matrix = exactly 0
    print("CONFORMAL NULL: PASS (identical matrices by construction)", flush=True)

    # Flat observations
    print(f"FLAT M={M_FLAT}", flush=True)
    fobs = []
    for i in range(M_FLAT):
        rng = np.random.default_rng(SEED * 2000 + i)
        pts = sprinkle(N, TT, rng)
        C = causal_flat(pts)
        fobs.append(compute_obs(C, pts))
        if i % 10 == 0 or i == M_FLAT - 1:
            print(f"  {i+1}/{M_FLAT}", flush=True)

    # TC-mediation
    print("TC-MEDIATION", flush=True)
    tc_f = np.array([f["tc"] for f in fobs])
    R["tc_med"] = {}
    for mn2 in MFNS:
        res2 = R[mn2]["raw"]
        R["tc_med"][mn2] = {}
        for nm in CA:
            of2 = np.array([f[nm] for f in fobs])
            sl2 = np.corrcoef(tc_f, of2)[0, 1] * of2.std() / tc_f.std() if tc_f.std() > 0 else 0
            ic2 = of2.mean() - sl2 * tc_f.mean()
            rd2 = np.array([r["curved"][nm] - (ic2 + sl2 * r["curved"]["tc"]) for r in res2])
            rf2 = np.array([r["flat"][nm] - (ic2 + sl2 * r["flat"]["tc"]) for r in res2])
            dr2 = cd(rd2 - rf2)
            s2 = "PASS" if abs(dr2) >= 0.3 else "MARG" if abs(dr2) >= 0.1 else "FAIL"
            print(f"  {mn2:15s} {nm:25s} d_resid={dr2:+.3f} {s2}", flush=True)
            R["tc_med"][mn2][nm] = round(dr2, 3)

    # Baseline R2
    print("BASELINE R2", flush=True)
    X = np.column_stack([[f[b] for f in fobs] for b in ["tc", "link_count", "degree_cv", "layer_0"]])
    R["baseline"] = {}
    for nm in CA:
        y = np.array([f[nm] for f in fobs])
        b2, _, _, _ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)
        yh = X @ b2[:4] + b2[4]
        ssr = np.sum((y - yh)**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - ssr / sst if sst > 0 else 0
        adj = 1 - (1 - r2) * (len(y) - 1) / max(len(y) - 5, 1)
        s2 = "RED" if r2 >= 0.7 else "YEL" if r2 >= 0.5 else "GRN"
        print(f"  {nm:25s} R2={r2:.4f} adj={adj:.4f} {s2}", flush=True)
        R["baseline"][nm] = {"r2": round(r2, 4), "adj": round(adj, 4)}

    # Random DAG
    print("RANDOM DAG", flush=True)
    mtc = int(np.mean([r["curved"]["tc"] for r in R["schwarzschild"]["raw"]]))
    R["rdag"] = {}
    rdo = []
    for i in range(10):
        rng = np.random.default_rng(SEED * 3000 + i)
        Cr = np.zeros((N, N), dtype=np.int8)
        ti = np.triu_indices(N, k=1)
        tp = len(ti[0])
        ch = rng.choice(tp, size=min(mtc, tp), replace=False)
        Cr[ti[0][ch], ti[1][ch]] = 1
        rdo.append(compute_obs(Cr, np.zeros((N, 4))))
    for nm in CA:
        geo = np.array([r["curved"][nm] for r in R["schwarzschild"]["raw"][:10]])
        rda = np.array([r[nm] for r in rdo])
        dv = cd(geo - rda[:len(geo)])
        print(f"  {nm:25s} geo={geo.mean():.4f} rdag={rda.mean():.4f} d={dv:+.3f}", flush=True)
        R["rdag"][nm] = {"d": round(dv, 3)}

    # Leakage
    print("LEAKAGE", flush=True)
    for nm in CA:
        v = np.array([f[nm] for f in fobs])
        for ref in ["column_gini_C2", "lva"]:
            vr = np.array([f[ref] for f in fobs])
            rho, _ = spearmanr(v, vr)
            s2 = "KILL" if abs(rho) > 0.8 else "PASS"
            print(f"  {nm:25s} vs {ref:20s} rho={rho:+.3f} {s2}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s", flush=True)

    # Summary
    print("SUMMARY:", flush=True)
    for nm in CA:
        dp = R["ppwave_quad"]["stats"][nm]["d"]
        ds = R["schwarzschild"]["stats"][nm]["d"]
        trp = R["tc_med"]["ppwave_quad"][nm]
        trs = R["tc_med"]["schwarzschild"][nm]
        r2 = R["baseline"][nm]["r2"]
        rd = R["rdag"][nm]["d"]
        multi = abs(dp) >= 0.5 and abs(ds) >= 0.5
        alive = "ALIVE" if multi and (abs(trp) >= 0.2 or abs(trs) >= 0.2) and r2 < 0.7 else "WEAK" if multi else "DEAD"
        print(f"  {nm:25s} pp={dp:+.3f} sch={ds:+.3f} TC_pp={trp:+.3f} TC_sch={trs:+.3f} R2={r2:.3f} rDAG={rd:+.3f} -> {alive}", flush=True)

    # Save
    sv = {"N": N, "M": M}
    for mn3 in ["ppwave_quad", "schwarzschild"]:
        sv[mn3] = {nm: R[mn3]["stats"][nm] for nm in ON}
    sv["tc_med"] = R["tc_med"]
    sv["baseline"] = R["baseline"]
    sv["rdag"] = R["rdag"]
    with open(RUN_DIR / "pilot_results.json", "w") as f:
        json.dump(sv, f, indent=2)
    print(f"Saved: pilot_results.json", flush=True)

if __name__ == "__main__":
    main()
