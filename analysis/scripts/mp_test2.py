"""Test multiprocessing with run_universal import."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import concurrent.futures as cf
import numpy as np

# This import triggers OGRePy warning in each worker
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, jet_preds, bulk_mask,
    riemann_schwarzschild_local
)

R_SCH = riemann_schwarzschild_local(0.05, 0.50)

def worker(seed):
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(500, 1.0, rng)
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    Y0 = Y_from_graph(par0, ch0)
    parF, chF = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
    YF = Y_from_graph(parF, chF)
    mask = bulk_mask(pts, 1.0, 0.15)
    dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
    return float(dk)

if __name__ == "__main__":
    print("Starting 4 workers...")
    with cf.ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, range(4)))
    print(f"Results: {[f'{r:+.6f}' for r in results]}")
    print("PASS")
