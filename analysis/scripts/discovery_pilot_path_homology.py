"""
Discovery Run 001 - Pilot Step 3: Path Homology of Causal Sets
===============================================================

Tests whether GLMY path homology of the Hasse diagram (link graph)
produces nontrivial Betti numbers, and whether they differ between
flat and curved spacetimes.

Path homology (Grigor'yan-Lin-Muranov-Yau, 2012+) is defined for
directed graphs. The path complex consists of:
  - 0-chains: vertices
  - 1-chains: directed edges (allowed paths of length 1)
  - n-chains: allowed paths of length n (sequences v0->v1->...->vn
    where each vi->v_{i+1} is an edge AND all intermediate vertices
    are "between" v0 and vn in the graph)

The boundary operator removes intermediate vertices:
  d_n(v0,...,vn) = sum_{i=1}^{n-1} (-1)^i (v0,...,v_{i-1},v_{i+1},...,vn)

Path Betti numbers: beta_k = dim H_k(path complex)

CRITICAL: We apply this to the DIRECTED Hasse diagram (link graph),
NOT to the transitive closure (full causal matrix). The transitive
closure would be homotopically trivial.

For this pilot, we use small N (50-200) because:
1. Path homology computation is expensive (exponential in path length)
2. We first need to check if beta_1 > 0 at all
3. If trivial at N=50, likely trivial at all N

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
from itertools import combinations
import json
import time
import gc
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")


# ---------------------------------------------------------------------------
# Sprinkling + Causal + Link graph (from previous pilots)
# ---------------------------------------------------------------------------
def sprinkle_2d(N, T, rng):
    """Sprinkle into 2D causal diamond (easier to get nontrivial topology)."""
    pts = np.empty((N, 2))
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 2))
        v = c[np.abs(c[:, 0]) + np.abs(c[:, 1]) < half]
        n = min(len(v), N - count)
        pts[count:count + n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


def sprinkle_4d(N, T, rng):
    pts = np.empty((N, 4))
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count + n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


def causal_flat_2d(pts):
    t, x = pts[:, 0], pts[:, 1]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    return ((dt**2 > dx**2) & (dt > 0)).astype(np.int8)


def causal_flat_4d(pts):
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    return ((dt**2 > dr2) & (dt > 0)).astype(np.int8)


def build_directed_link_graph(C):
    """Build DIRECTED link (Hasse) adjacency: L[i,j]=1 iff i->j is a link.

    A link i->j means: i precedes j and no element k with i < k < j.
    """
    N = C.shape[0]
    C_sp = sp.csr_matrix(C.astype(np.float64))
    C2 = C_sp @ C_sp
    has_intervening = (C2 != 0).astype(np.float64)
    link = C_sp - C_sp.multiply(has_intervening)
    link.eliminate_zeros()
    return link.tocsr()


# ---------------------------------------------------------------------------
# Path homology computation (GLMY)
# ---------------------------------------------------------------------------
def compute_path_homology(L_directed, max_dim=3):
    """Compute path Betti numbers of directed graph.

    L_directed: sparse CSR matrix where L[i,j]=1 means directed edge i->j.
    max_dim: maximum homological dimension to compute.

    Returns dict with beta_0, beta_1, ..., beta_{max_dim},
    plus chain counts at each dimension.

    For the GLMY path complex:
    - 0-paths = vertices
    - 1-paths = directed edges
    - 2-paths = allowed paths v0->v1->v2 where v0->v2 is NOT an edge
      (if v0->v2 IS an edge, it's a "shortcut" and the path is not allowed)
      Actually in GLMY: 2-paths are v0->v1->v2 (any directed 2-path).
      The boundary is: d(v0,v1,v2) = (v1,v2) - (v0,v2) + (v0,v1)
      where (v0,v2) is included only if v0->v2 is an edge.

    Simplified implementation for small graphs.
    """
    rows, cols = L_directed.nonzero()
    N = L_directed.shape[0]

    # Edge set for fast lookup
    edge_set = set(zip(rows.tolist(), cols.tolist()))
    edges = list(edge_set)
    n_edges = len(edges)

    # 0-chains: vertices
    n_0 = N

    # 1-chains: directed edges
    n_1 = n_edges

    # Compute beta_0 via connected components of the undirected version
    A_sym = L_directed + L_directed.T
    A_sym = (A_sym > 0).astype(np.float64)
    n_components = sp.csgraph.connected_components(A_sym, directed=False)[0]
    beta_0 = n_components

    if max_dim < 1:
        return {"beta_0": beta_0, "n_0_chains": n_0, "n_1_chains": n_1}

    # ----- beta_1 computation -----
    # We need:
    # 1. The boundary map d_1: C_1 -> C_0
    #    d_1(u->v) = v - u
    # 2. The boundary map d_2: C_2 -> C_1
    #    For a 2-path (u,v,w) meaning u->v->w:
    #    d_2(u,v,w) = (v,w) - (u,w) + (u,v)
    #    where (u,w) is included only if u->w is an edge
    #    and (v,w) and (u,v) must be edges (they are by construction)

    # Build edge index map
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    # Build d_1 matrix (n_0 x n_1)
    d1 = np.zeros((n_0, n_1), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        d1[v, idx] += 1.0   # +v
        d1[u, idx] -= 1.0   # -u

    # Find all 2-paths: u->v->w where (u,v) and (v,w) are edges
    two_paths = []
    # For each edge (u,v), find all w such that (v,w) is an edge
    L_dense = L_directed.toarray()
    for u, v in edges:
        successors_v = np.where(L_dense[v] > 0)[0]
        for w in successors_v:
            two_paths.append((u, v, w))

    n_2 = len(two_paths)

    if n_2 == 0:
        # No 2-paths => ker(d_1) / im(d_2) = ker(d_1)
        # beta_1 = dim(ker(d_1)) - beta_0 + n_0... actually:
        # beta_1 = dim(ker(d_1)) - dim(im(d_2))
        # With no d_2, beta_1 = dim(ker(d_1))
        rank_d1 = np.linalg.matrix_rank(d1)
        beta_1 = n_1 - rank_d1  # dim(ker(d1))
        return {
            "beta_0": beta_0, "beta_1": beta_1,
            "n_0_chains": n_0, "n_1_chains": n_1, "n_2_chains": 0,
            "rank_d1": rank_d1,
        }

    # Build d_2 matrix (n_1 x n_2)
    d2 = np.zeros((n_1, n_2), dtype=np.float64)
    for idx, (u, v, w) in enumerate(two_paths):
        # d_2(u,v,w) = (v,w) - (u,w) + (u,v)
        # (v,w) is always an edge
        d2[edge_to_idx[(v, w)], idx] += 1.0
        # (u,v) is always an edge
        d2[edge_to_idx[(u, v)], idx] += 1.0
        # (u,w) only if it's an edge
        if (u, w) in edge_set:
            d2[edge_to_idx[(u, w)], idx] -= 1.0

    # beta_1 = dim(ker(d_1)) - dim(im(d_2))
    rank_d1 = np.linalg.matrix_rank(d1)
    rank_d2 = np.linalg.matrix_rank(d2)
    ker_d1 = n_1 - rank_d1
    im_d2 = rank_d2
    beta_1 = ker_d1 - im_d2

    result = {
        "beta_0": int(beta_0),
        "beta_1": int(beta_1),
        "n_0_chains": n_0,
        "n_1_chains": n_1,
        "n_2_chains": n_2,
        "rank_d1": int(rank_d1),
        "rank_d2": int(rank_d2),
        "ker_d1": int(ker_d1),
        "im_d2": int(im_d2),
    }

    if max_dim < 2:
        return result

    # ----- beta_2 computation (if requested and feasible) -----
    # Find all 3-paths: (u,v,w,x) where u->v->w->x
    three_paths = []
    if n_2 < 50000:  # only if feasible
        for u, v, w in two_paths:
            successors_w = np.where(L_dense[w] > 0)[0]
            for x in successors_w:
                three_paths.append((u, v, w, x))

    n_3 = len(three_paths)
    result["n_3_chains"] = n_3

    if n_3 == 0 or n_3 > 100000:
        # beta_2 = dim(ker(d_2)) with no d_3
        ker_d2 = n_2 - rank_d2
        result["beta_2"] = int(ker_d2)
        return result

    # Build 2-path index
    twopath_to_idx = {tp: i for i, tp in enumerate(two_paths)}

    # Build d_3 matrix (n_2 x n_3)
    d3 = np.zeros((n_2, n_3), dtype=np.float64)
    for idx, (u, v, w, x) in enumerate(three_paths):
        # d_3(u,v,w,x) = (v,w,x) - (u,w,x) + (u,v,x) - (u,v,w)
        # Each term included only if it's a valid 2-path (i.e., consecutive edges exist)
        if (v, w, x) in twopath_to_idx:
            d3[twopath_to_idx[(v, w, x)], idx] += 1.0
        if (u, w, x) in twopath_to_idx:
            d3[twopath_to_idx[(u, w, x)], idx] -= 1.0
        if (u, v, x) in twopath_to_idx:
            d3[twopath_to_idx[(u, v, x)], idx] += 1.0
        if (u, v, w) in twopath_to_idx:
            d3[twopath_to_idx[(u, v, w)], idx] -= 1.0

    rank_d3 = np.linalg.matrix_rank(d3)
    ker_d2 = n_2 - rank_d2
    im_d3 = rank_d3
    beta_2 = ker_d2 - im_d3

    result["beta_2"] = int(beta_2)
    result["rank_d3"] = int(rank_d3)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 70)
    print("Discovery Run 001 - Pilot Step 3: Path Homology of Causal Sets")
    print("=" * 70)

    results = {}

    # Test 1: 2D flat, varying N
    print("\n--- Test 1: 2D Minkowski, varying N ---")
    for N in [20, 30, 50, 75, 100]:
        t0 = time.time()
        betas = []
        for trial in range(10):
            rng = np.random.default_rng(trial * 100 + 42)
            pts = sprinkle_2d(N, 1.0, rng)
            C = causal_flat_2d(pts)
            L = build_directed_link_graph(C)
            h = compute_path_homology(L, max_dim=2)
            betas.append(h)
        elapsed = time.time() - t0

        b0s = [b["beta_0"] for b in betas]
        b1s = [b["beta_1"] for b in betas]
        b2s = [b.get("beta_2", -1) for b in betas]
        n1s = [b["n_1_chains"] for b in betas]
        n2s = [b["n_2_chains"] for b in betas]

        print(f"  N={N:4d}: beta_0={np.mean(b0s):.1f}, beta_1={np.mean(b1s):.1f} "
              f"(range {min(b1s)}-{max(b1s)}), beta_2={np.mean(b2s):.1f}, "
              f"edges={np.mean(n1s):.0f}, 2-paths={np.mean(n2s):.0f}  [{elapsed:.1f}s]")

        results[f"2d_flat_N{N}"] = {
            "d": 2, "N": N, "geometry": "flat",
            "beta_0": b0s, "beta_1": b1s, "beta_2": b2s,
            "n_edges": [int(x) for x in n1s], "n_2paths": [int(x) for x in n2s],
        }

    # Test 2: 4D flat, varying N
    print("\n--- Test 2: 4D Minkowski, varying N ---")
    for N in [20, 30, 50, 75, 100]:
        t0 = time.time()
        betas = []
        for trial in range(10):
            rng = np.random.default_rng(trial * 100 + 42)
            pts = sprinkle_4d(N, 1.0, rng)
            C = causal_flat_4d(pts)
            L = build_directed_link_graph(C)
            h = compute_path_homology(L, max_dim=2)
            betas.append(h)
        elapsed = time.time() - t0

        b0s = [b["beta_0"] for b in betas]
        b1s = [b["beta_1"] for b in betas]
        b2s = [b.get("beta_2", -1) for b in betas]
        n1s = [b["n_1_chains"] for b in betas]
        n2s = [b["n_2_chains"] for b in betas]

        print(f"  N={N:4d}: beta_0={np.mean(b0s):.1f}, beta_1={np.mean(b1s):.1f} "
              f"(range {min(b1s)}-{max(b1s)}), beta_2={np.mean(b2s):.1f}, "
              f"edges={np.mean(n1s):.0f}, 2-paths={np.mean(n2s):.0f}  [{elapsed:.1f}s]")

        results[f"4d_flat_N{N}"] = {
            "d": 4, "N": N, "geometry": "flat",
            "beta_0": b0s, "beta_1": b1s, "beta_2": b2s,
            "n_edges": [int(x) for x in n1s], "n_2paths": [int(x) for x in n2s],
        }

    # Save
    outpath = os.path.join(OUTDIR, "pilot_path_homology.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    any_nontrivial = False
    for key, res in results.items():
        b1s = res["beta_1"]
        if max(b1s) > 0:
            any_nontrivial = True
            print(f"  {key}: beta_1 NONTRIVIAL (max={max(b1s)}, mean={np.mean(b1s):.1f})")
        else:
            print(f"  {key}: beta_1 = 0 (trivial)")

    if any_nontrivial:
        print("\n  => Path homology is NONTRIVIAL on causal sets. Candidate ALIVE.")
        print("     Next: CRN comparison flat vs curved.")
    else:
        print("\n  => Path homology is TRIVIAL on causal sets. Candidate DEAD.")
        print("     The Hasse diagram may be too tree-like for homological cycles.")


if __name__ == "__main__":
    main()
