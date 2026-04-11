"""
FND-1 Ray Distributed Worker
==============================

PURPOSE
-------
Distribute independent CRN paired sprinklings across a Ray cluster
(multiple machines connected via Tailscale VPN).  Each task is ONE
call to crn_one() from fnd1_gcp_pipeline.py — completely independent,
no inter-task communication, no shared state.

ARCHITECTURE DECISIONS
----------------------
Why Ray (not MPI, Dask, multiprocessing):
  - Each task is independent (embarrassingly parallel) → no communication.
  - Ray auto-balances: faster machines get more tasks.  No manual splitting.
  - Workers can join/leave dynamically (friend goes to sleep → tasks
    redistribute; friend wakes up → reconnects, gets new tasks).
  - One-line setup: `ray start --address=...`
  - MPI would require shared filesystem + job scheduler.  Overkill.
  - Dask is heavier and designed for dataframes, not raw function calls.
  - multiprocessing is single-machine only.

Why num_cpus=1 per task:
  - Each crn_one() calls eigvalsh which uses LAPACK internally.
  - LAPACK uses MKL/OpenBLAS threads.  We set MKL_NUM_THREADS=1 inside
    each task so that Ray controls parallelism, not MKL.
  - With --num-cpus=16 on a machine: Ray runs 16 tasks concurrently,
    each on 1 thread.  Total: 16 cores utilised, no thread contention.
  - Alternative (MKL_NUM_THREADS=4, num_cpus=4): fewer concurrent tasks,
    each faster.  Benchmarks show 1-thread × many-tasks is slightly better
    for our workload because eigvalsh MKL scaling is sublinear.

Why results saved on head node (not distributed):
  - Results are small (1 dict per task, ~1 KB).
  - Checkpointing on head node is simpler and more reliable.
  - Workers are stateless — can die and restart without data loss.

SETUP (3 machines example)
--------------------------
  Machine A (head, 24 cores):
    ray start --head --port=6379 --num-cpus=16
    python fnd1_ray_worker.py --all --N 10000 --M 80

  Machine B (worker, 32 cores, via Tailscale):
    ray start --address='100.71.68.111:6379' --num-cpus=24

  Machine C (worker, 12 cores):
    ray start --address='100.71.68.111:6379' --num-cpus=8

  Workers can connect/disconnect at any time.  Tasks auto-redistribute.

DEPENDENCIES
------------
  pip install ray numpy scipy
  + fnd1_gcp_pipeline.py in the same directory (or PYTHONPATH)

USAGE
-----
    python fnd1_ray_worker.py --metric ppwave_quad --N 10000 --M 80
    python fnd1_ray_worker.py --all --N 10000 --M 80
    python fnd1_ray_worker.py --metric ppwave_quad --N 30000 --M 40 --T 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ray
from scipy import stats

# Import pipeline functions (must be on PYTHONPATH or same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd1_gcp_pipeline import (
    METRICS, METRIC_FNS, METRIC_SEED_OFFSETS, MASTER_SEED,
    crn_one, causal_ppwave_quad, analyze_experiment,
)

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Ray remote task: one CRN pair
# ---------------------------------------------------------------------------
@ray.remote(num_cpus=1)
def ray_crn_one(seed: int, N: int, T: float, eps: float,
                metric_name: str, mode: str = "dense",
                include_random: bool = False,
                mkl_threads: int = 4) -> dict:
    """One CRN pair as a Ray task. Runs on any node in the cluster.

    num_cpus=1 means Ray schedules one task per declared CPU.
    mkl_threads controls how many threads LAPACK uses internally.
    With --num-cpus=4 on a node, Ray runs 4 concurrent tasks,
    each using mkl_threads=4 → 16 threads total on a 16-core machine.
    Adjust mkl_threads to match: cores / num-cpus.
    """
    import os
    os.environ["MKL_NUM_THREADS"] = str(mkl_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(mkl_threads)
    os.environ["OMP_NUM_THREADS"] = str(mkl_threads)

    from fnd1_gcp_pipeline import crn_one, METRIC_FNS
    metric_fn = METRIC_FNS[metric_name]
    return crn_one(seed, N, T, eps, metric_fn, mode,
                   include_random_operator=include_random)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_distributed(metric: str, N: int, M: int, T: float, mode: str):
    """Run one metric fully distributed via Ray."""
    cfg = METRICS[metric]
    eps_values = cfg["eps"]

    print(f"\n{'='*60}")
    print(f"RAY DISTRIBUTED: {cfg['label']}  N={N:,}  M={M}  T={T}")
    print(f"Cluster: {len(ray.nodes())} nodes, "
          f"{ray.cluster_resources().get('CPU', 0):.0f} total CPUs")
    print(f"{'='*60}")

    # Generate seeds (same logic as fnd1_gcp_pipeline.py)
    ss = np.random.SeedSequence(MASTER_SEED + METRIC_SEED_OFFSETS.get(metric, 999))
    all_seeds = ss.spawn(len(eps_values) * M)
    seed_idx = 0

    # Check for existing results
    result_path = RESULTS_DIR / f"crn_{metric}_N{N}_T{T}.json"
    existing_results = {}
    if result_path.exists():
        with open(result_path) as f:
            existing_results = json.load(f).get("results", {})

    all_results = {}
    total_submitted = 0
    total_skipped = 0

    for eps in eps_values:
        M_this = 5 if abs(eps) < 1e-12 else M
        existing = existing_results.get(str(eps), [])
        done_seeds = {r["seed"] for r in existing}

        seeds = []
        for i in range(M):
            s = int(all_seeds[seed_idx + i].generate_state(1)[0])
            if i < M_this:
                seeds.append(s)
        seed_idx += M

        remaining = [s for s in seeds if s not in done_seeds]
        total_skipped += len(seeds) - len(remaining)

        # Determine which get random operator tests
        nonzero_eps = sorted([e for e in eps_values if abs(e) > 1e-12])
        top_two = set(nonzero_eps[-2:]) if len(nonzero_eps) >= 2 else set(nonzero_eps)

        # Submit Ray tasks
        futures = []
        for j, seed in enumerate(remaining):
            do_random = (eps in top_two) and (j < 20) and (mode == "dense")
            f = ray_crn_one.remote(seed, N, T, eps, metric, mode, do_random)
            futures.append((f, eps, seed))
            total_submitted += 1

        all_results[str(eps)] = list(existing)

    print(f"  Submitted: {total_submitted} tasks, skipped: {total_skipped} (already done)")

    # Collect results as they complete
    if total_submitted == 0:
        print("  Nothing to do — all results exist.")
    else:
        pending = {f: (eps, seed) for f, eps, seed in
                   [(fut, e, s) for fut, e, s in
                    sum([[(f, e, s)] for f, e, s in
                         [(fut, eps_str, sd) for fut, eps_str, sd in
                          # Flatten — just collect all futures
                          []]], [])]}

        # Simpler: collect all futures grouped by eps
        all_futures = []
        eps_map = {}
        for eps in eps_values:
            M_this = 5 if abs(eps) < 1e-12 else M
            existing = existing_results.get(str(eps), [])
            done_seeds = {r["seed"] for r in existing}

            ss2 = np.random.SeedSequence(MASTER_SEED + METRIC_SEED_OFFSETS.get(metric, 999))
            all_seeds2 = ss2.spawn(len(eps_values) * M)
            # Recalculate seed index for this eps
            eps_idx = eps_values.index(eps)
            seed_offset = eps_idx * M
            seeds2 = []
            for i in range(M_this):
                s = int(all_seeds2[seed_offset + i].generate_state(1)[0])
                seeds2.append(s)

            remaining2 = [s for s in seeds2 if s not in done_seeds]

            nonzero_eps2 = sorted([e for e in eps_values if abs(e) > 1e-12])
            top_two2 = set(nonzero_eps2[-2:]) if len(nonzero_eps2) >= 2 else set(nonzero_eps2)

            for j, seed in enumerate(remaining2):
                do_random = (eps in top_two2) and (j < 20) and (mode == "dense")
                f = ray_crn_one.remote(seed, N, T, eps, metric, mode, do_random)
                all_futures.append(f)
                eps_map[f] = str(eps)

        # Wait and collect
        t0 = time.perf_counter()
        done_count = 0
        while all_futures:
            ready, all_futures = ray.wait(all_futures, num_returns=1, timeout=None)
            for f in ready:
                result = ray.get(f)
                eps_key = eps_map[f]
                all_results[eps_key].append(result)
                done_count += 1

                if done_count % 10 == 0 or not all_futures:
                    elapsed = time.perf_counter() - t0
                    rate = done_count / elapsed * 60
                    remaining_count = len(all_futures)
                    eta = remaining_count / rate if rate > 0 else 0
                    print(f"  {done_count} done, {remaining_count} pending  "
                          f"{rate:.1f}/min  ETA {eta:.0f} min")

                    # Save intermediate results
                    _save_results(all_results, metric, cfg, N, M, T, mode)

    # Final save
    _save_results(all_results, metric, cfg, N, M, T, mode)

    # Post-experiment analysis
    analyze_experiment(all_results, cfg["label"], N)

    return all_results


def _save_results(all_results, metric, cfg, N, M, T, mode):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "metric": metric, "label": cfg["label"],
        "N": N, "M": M, "T": T, "mode": mode,
        "eps_values": cfg["eps"],
        "results": all_results,
    }
    p = RESULTS_DIR / f"crn_{metric}_N{N}_T{T}.json"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    tmp.replace(p)
    print(f"  Saved: {p.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FND-1 Ray Distributed Runner")
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--mode", choices=["dense", "matfree"], default="dense")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--address", type=str, default="auto",
                        help="Ray head address (default: auto-detect)")
    args = parser.parse_args()

    # Connect to Ray cluster
    ray.init(address=args.address, ignore_reinit_error=True)
    res = ray.cluster_resources()
    total_cpus = int(res.get("CPU", 1))
    print(f"Ray connected: {total_cpus} CPUs across {len(ray.nodes())} nodes")
    print(f"Resources: {res}")

    if args.all:
        metrics = list(METRICS.keys())
    elif args.metric:
        metrics = [args.metric]
    else:
        metrics = ["ppwave_quad"]

    for metric in metrics:
        run_distributed(metric, args.N, args.M, args.T, args.mode)

    print("\nAll done.")
    ray.shutdown()


if __name__ == "__main__":
    main()
