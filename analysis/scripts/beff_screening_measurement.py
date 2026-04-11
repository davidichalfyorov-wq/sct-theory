#!/usr/bin/env python3
"""
Direct measurement of b_eff screening multiplicity.

For each link (j → i) in the flat Hasse diagram, count how many
causal predecessors of i are "screened" by j (i.e., lie in j's
causal past and thus are transitively reduced away).

m_scr = mean screening count per link = expected number of
competitors per angular tube in the thin-shell model.

If m_scr ≈ 5, this confirms b_eff = 5.
Also checks: position dependence (bulk vs boundary), N-scaling.
"""
import sys, os, time, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds,
    build_hasse_from_predicate, bulk_mask,
)

T = 1.0
ZETA = 0.15
N_VALUES = [2000, 5000, 10000]
M_SEEDS = 10


def measure_screening(pts, par, ch, bmask):
    """For each link (j → i), count how many causal predecessors of i
    are screened by j.

    A predecessor k of i is "screened" by link j if k ≺ j (k is in j's past).
    This means the relation k → i is transitively reduced away because k → j → i exists.

    Returns per-link screening counts, split by bulk/boundary status of i.
    """
    N = len(pts)

    # Build ancestor sets for each element (using Hasse parents, BFS)
    # ancestor[i] = set of all elements in causal past of i (transitive closure)
    ancestor_sets = [set() for _ in range(N)]
    for i in range(N):
        if par[i] is not None and len(par[i]) > 0:
            for j in par[i]:
                ancestor_sets[i].add(int(j))
                ancestor_sets[i].update(ancestor_sets[int(j)])

    # For each element i, count causal predecessors (full) vs Hasse parents (links)
    screening_counts = []  # per link: how many predecessors does this parent screen?
    screening_bulk = []
    screening_boundary = []

    n_causal_predecessors = []
    n_links_list = []

    for i in range(N):
        if par[i] is None or len(par[i]) == 0:
            continue

        parents_i = set(int(j) for j in par[i])
        all_ancestors_i = ancestor_sets[i]
        n_ancestors = len(all_ancestors_i)
        n_links = len(parents_i)

        n_causal_predecessors.append(n_ancestors)
        n_links_list.append(n_links)

        # For each Hasse parent j of i:
        # screened_by_j = {k in ancestors(i) : k in ancestors(j)} - {j}
        # These are predecessors of i that go through j
        for j in parents_i:
            # Elements screened by j: ancestors of j that are also ancestors of i
            # (which they always are, since j is parent of i)
            screened = len(ancestor_sets[int(j)])  # all ancestors of j are screened
            screening_counts.append(screened)

            if bmask[i]:
                screening_bulk.append(screened)
            else:
                screening_boundary.append(screened)

    # TR ratio: total causal pairs → links
    # For each element: n_ancestors causal predecessors, n_links survive TR
    # TR factor per element = n_links / n_ancestors
    tr_ratios = []
    for na, nl in zip(n_causal_predecessors, n_links_list):
        if na > 0:
            tr_ratios.append(nl / na)

    return {
        'screening_all': screening_counts,
        'screening_bulk': screening_bulk,
        'screening_boundary': screening_boundary,
        'n_causal_predecessors': n_causal_predecessors,
        'n_links': n_links_list,
        'tr_ratios': tr_ratios,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("B_EFF SCREENING MULTIPLICITY MEASUREMENT")
    print(f"N values: {N_VALUES}, M={M_SEEDS}, T={T}")
    print("=" * 72, flush=True)

    results = {}

    for N in N_VALUES:
        print(f"\n{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}", flush=True)

        all_screening = []
        all_screening_bulk = []
        all_screening_boundary = []
        all_mean_k = []
        all_mean_ancestors = []
        all_tr_ratio = []

        for si in range(M_SEEDS):
            t0 = time.time()
            seed = 8800000 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            bmask = bulk_mask(pts, T, ZETA)

            par, ch = build_hasse_from_predicate(
                pts, lambda P, i: minkowski_preds(P, i))

            data = measure_screening(pts, par, ch, bmask)

            m_all = np.mean(data['screening_all']) if data['screening_all'] else 0
            m_bulk = np.mean(data['screening_bulk']) if data['screening_bulk'] else 0
            m_bnd = np.mean(data['screening_boundary']) if data['screening_boundary'] else 0
            mean_k = np.mean(data['n_links']) if data['n_links'] else 0
            mean_anc = np.mean(data['n_causal_predecessors']) if data['n_causal_predecessors'] else 0
            mean_tr = np.mean(data['tr_ratios']) if data['tr_ratios'] else 0

            all_screening.append(m_all)
            all_screening_bulk.append(m_bulk)
            all_screening_boundary.append(m_bnd)
            all_mean_k.append(mean_k)
            all_mean_ancestors.append(mean_anc)
            all_tr_ratio.append(mean_tr)

            elapsed = time.time() - t0
            if (si + 1) % 5 == 0 or si == 0:
                print(f"  seed {si+1:2d}/{M_SEEDS}: "
                      f"m_scr={m_all:.2f}  "
                      f"m_bulk={m_bulk:.2f}  "
                      f"m_bnd={m_bnd:.2f}  "
                      f"<k>={mean_k:.2f}  "
                      f"<ancestors>={mean_anc:.0f}  "
                      f"TR_ratio={mean_tr:.4f}  "
                      f"({elapsed:.1f}s)", flush=True)

        m_arr = np.array(all_screening)
        mb_arr = np.array(all_screening_bulk)
        mbd_arr = np.array(all_screening_boundary)
        k_arr = np.array(all_mean_k)
        anc_arr = np.array(all_mean_ancestors)
        tr_arr = np.array(all_tr_ratio)

        results[str(N)] = {
            'N': N,
            'm_scr_mean': float(m_arr.mean()),
            'm_scr_se': float(m_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'm_bulk_mean': float(mb_arr.mean()),
            'm_bulk_se': float(mb_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'm_boundary_mean': float(mbd_arr.mean()),
            'm_boundary_se': float(mbd_arr.std(ddof=1) / math.sqrt(M_SEEDS)),
            'mean_k': float(k_arr.mean()),
            'mean_ancestors': float(anc_arr.mean()),
            'mean_tr_ratio': float(tr_arr.mean()),
        }

        print(f"\n  SUMMARY N={N}:")
        print(f"    m_scr (all)      = {m_arr.mean():.3f} ± {m_arr.std(ddof=1)/math.sqrt(M_SEEDS):.3f}")
        print(f"    m_scr (bulk)     = {mb_arr.mean():.3f} ± {mb_arr.std(ddof=1)/math.sqrt(M_SEEDS):.3f}")
        print(f"    m_scr (boundary) = {mbd_arr.mean():.3f} ± {mbd_arr.std(ddof=1)/math.sqrt(M_SEEDS):.3f}")
        print(f"    <k> (links/elem) = {k_arr.mean():.3f}")
        print(f"    <ancestors>      = {anc_arr.mean():.1f}")
        print(f"    TR ratio k/anc   = {tr_arr.mean():.5f}")
        print(flush=True)

    # N-SCALING TABLE
    print(f"\n{'='*72}")
    print("N-SCALING OF m_scr")
    print(f"{'='*72}")
    for N in N_VALUES:
        r = results[str(N)]
        print(f"  N={N:5d}: m_scr = {r['m_scr_mean']:.3f}±{r['m_scr_se']:.3f}  "
              f"(bulk={r['m_bulk_mean']:.3f}, bnd={r['m_boundary_mean']:.3f})  "
              f"<k>={r['mean_k']:.2f}  TR={r['mean_tr_ratio']:.5f}")

    # Check: does m_scr converge to 5?
    print(f"\n  Target: m_scr → 5 (from β_pair/β_Hasse = (1-e^{{-m}})/m)")
    print(f"  (1-e^{{-5}})/5 = {(1-math.exp(-5))/5:.4f} ≈ 0.199 → β_Hasse/β_pair ≈ 1/5")

    outfile = 'analysis/fnd1_data/beff_screening_measurement.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}", flush=True)
