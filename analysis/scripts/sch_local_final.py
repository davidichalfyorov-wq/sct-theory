#!/usr/bin/env python3
"""Schwarzschild local T-scaling — FINAL version.

Uses numpy uint64 bitset Hasse (from build_hasse_from_predicate rewritten)
instead of Python bigint which crashes at N=10000.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, jet_preds, bulk_mask,
    excess_kurtosis, riemann_schwarzschild_local
)
from scipy.special import logsumexp
import math


# ── Numpy uint64 bitset Hasse builder ────────────────────────────
def build_hasse_uint64(pts, pred_fn):
    """Build Hasse diagram using numpy uint64 bitset arrays.

    Much faster and more memory-stable than Python bigint version.
    pts must be sorted by time (pts[:,0] ascending).
    pred_fn(pts, i) returns bool mask of length i for predecessors of i.
    """
    n = len(pts)
    n_words = (n + 63) // 64

    # Ancestor bitsets: past[i] = set of all ancestors of i (not including i)
    past = np.zeros((n, n_words), dtype=np.uint64)

    parents_list = [[] for _ in range(n)]
    children_list = [[] for _ in range(n)]

    for i in range(n):
        if i == 0:
            continue

        rel_mask = np.asarray(pred_fn(pts, i), dtype=bool)
        rel_preds = np.flatnonzero(rel_mask)

        if rel_preds.size == 0:
            continue

        # Find direct parents (Hasse links) by transitive reduction
        direct = []
        # Check newest first
        for j in reversed(rel_preds):
            j = int(j)
            word_j = j // 64
            bit_j = np.uint64(1) << np.uint64(j % 64)

            # Check if j is already covered by ancestors of accepted parents
            covered = False
            for p in direct:
                if past[p, word_j] & bit_j:
                    covered = True
                    break

            if not covered:
                direct.append(j)

        direct.sort()
        parents_list[i] = direct

        # Build past[i] = union of (past[j] | {j}) for all direct parents j
        for j in direct:
            children_list[j].append(i)
            past[i] |= past[j]
            word_j = j // 64
            bit_j = np.uint64(1) << np.uint64(j % 64)
            past[i, word_j] |= bit_j

    parents = [np.array(p, dtype=np.int32) for p in parents_list]
    children = [np.array(c, dtype=np.int32) for c in children_list]
    return parents, children


def Y_from_graph(parents, children):
    """Compute Y = log2(p_down * p_up + 1) using log-domain DP."""
    n = len(parents)
    log_pd = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = parents[i]
        if p.size == 0:
            log_pd[i] = 0.0
        else:
            log_pd[i] = float(logsumexp(log_pd[p]))

    log_pu = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        ch = children[i]
        if ch.size == 0:
            log_pu[i] = 0.0
        else:
            log_pu[i] = float(logsumexp(log_pu[ch]))

    L = log_pd + log_pu
    return (np.maximum(L, 0.0) + np.log1p(np.exp(-np.abs(L)))) / math.log(2.0)


# ── Main experiment ──────────────────────────────────────────────
R0 = 0.50
N = 10000
ZETA = 0.15
M_SEEDS = 50

LINES = [
    {"label": "strong", "M_sch": 0.10, "T_vals": [1.00, 0.70, 0.50]},
    {"label": "weak",   "M_sch": 0.05, "T_vals": [1.00, 0.70]},
]


if __name__ == "__main__":
    print(f"=== SCH LOCAL FINAL (uint64 bitset) ===", flush=True)
    print(f"r0={R0}, N={N}, zeta={ZETA}, M_seeds={M_SEEDS}", flush=True)

    all_results = {}
    t_total = time.time()

    for line in LINES:
        label = line["label"]
        M_sch = line["M_sch"]
        R_sch = riemann_schwarzschild_local(M_sch, R0)
        q_base = M_sch / R0**3

        print(f"\n--- {label}: M_sch={M_sch}, q_base={q_base:.3f} ---", flush=True)

        for T in line["T_vals"]:
            q_W = q_base * T**2
            pred = 0.031 * q_W**2
            dks = []
            t0 = time.time()

            for si in range(M_SEEDS):
                seed = 200000 + int(M_sch * 10000) + int(T * 1000) * 100 + si
                rng = np.random.default_rng(seed)
                pts = sprinkle_local_diamond(N, T, rng)

                par0, ch0 = build_hasse_uint64(
                    pts, lambda P, i: minkowski_preds(P, i))
                Y0 = Y_from_graph(par0, ch0)

                parF, chF = build_hasse_uint64(
                    pts, lambda P, i: jet_preds(P, i, R_abcd=R_sch))
                YF = Y_from_graph(parF, chF)

                mask = bulk_mask(pts, T, ZETA)
                dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
                dks.append(dk)

                if (si + 1) % 5 == 0:
                    elapsed = time.time() - t0
                    rate = elapsed / (si + 1)
                    eta = rate * (M_SEEDS - si - 1)
                    print(f"  {label} T={T:.2f}: {si+1}/{M_SEEDS} "
                          f"({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

            elapsed = time.time() - t0
            arr = np.array(dks)
            m = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            std = float(np.std(arr, ddof=1))
            d = float(m / std) if std > 0 else 0.0
            t_stat = float(m / se) if se > 0 else 0.0
            ratio = m / pred if abs(pred) > 1e-15 else 0.0

            sig = "***" if abs(t_stat) > 3 else "**" if abs(t_stat) > 2 else "*" if abs(t_stat) > 1 else ""
            print(f"\n  RESULT T={T:.2f} q_W={q_W:.3f}:", flush=True)
            print(f"    predicted: dk={pred:+.6f}", flush=True)
            print(f"    observed:  dk={m:+.6f} +/- {se:.6f}", flush=True)
            print(f"    d_cohen={d:+.3f}, t_stat={t_stat:+.3f} {sig}", flush=True)
            print(f"    ratio obs/pred: {ratio:.3f}", flush=True)
            print(f"    ({elapsed:.0f}s)\n", flush=True)

            all_results[f"{label}_T{T}"] = {
                "label": label, "M_sch": M_sch, "T": T, "q_W": q_W,
                "predicted_dk": pred, "observed_dk": m,
                "se": se, "std": std, "d_cohen": d, "t_stat": t_stat,
                "ratio": ratio, "n": len(dks),
                "per_seed": dks, "elapsed_s": elapsed,
            }

    # T-scaling
    print("\n=== T-SCALING (strong M=0.10) ===", flush=True)
    dk1 = all_results.get("strong_T1.0", {}).get("observed_dk", 0)
    if abs(dk1) > 1e-15:
        for key in sorted(k for k in all_results if k.startswith("strong")):
            r = all_results[key]
            T = r["T"]
            dk = r["observed_dk"]
            rat = dk / dk1
            print(f"T={T:.2f}: dk/dk(1)={rat:+.4f}, "
                  f"ratio/T^2={rat/T**2:+.4f}, ratio/T^4={rat/T**4:+.4f}",
                  flush=True)

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sch_local_final.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
