"""
FND-1 Dask Distributed Worker
===============================

Distributes CRN paired sprinklings across machines using Dask.
Works on Windows + Tailscale without WSL or port forwarding.

SETUP
-----
  All machines:
    pip install numpy scipy dask distributed

  You (scheduler + local workers):
    python fnd1_dask_worker.py --metric ppwave_quad --N 10000 --M 80

  Friend (connect from anywhere via Tailscale):
    dask worker tcp://100.71.68.111:8786 --nworkers 4 --nthreads 1

  Friends can connect/disconnect at any time.  Scheduler redistributes
  unfinished tasks automatically.

WHY DASK (NOT RAY)
------------------
  Ray on Windows crashes with "access violation" under compute load.
  Dask on Windows is stable and binds to 0.0.0.0 by default, making it
  reachable via Tailscale without WSL or port forwarding.

USAGE
-----
    python fnd1_dask_worker.py --metric ppwave_quad --N 10000 --M 80
    python fnd1_dask_worker.py --all --N 10000 --M 80
    python fnd1_dask_worker.py --metric ppwave_quad --N 10000 --workers 8
"""
from __future__ import annotations

import os
# CRITICAL: limit BLAS threads BEFORE numpy import.
# Without this, each Dask worker spawns N_CPU threads for eigvalsh,
# and 3 workers × 24 threads = 72 threads on 24 cores = deadlock.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd1_gcp_pipeline import (
    METRICS, METRIC_FNS, METRIC_SEED_OFFSETS, MASTER_SEED,
    crn_one, analyze_experiment,
)

RESULTS_DIR = Path("results")


def run_distributed(metric: str, N: int, M: int, T: float, mode: str,
                    client, n_local_workers: int):
    """Run one metric distributed via Dask."""
    cfg = METRICS[metric]
    eps_values = cfg["eps"]

    print(f"\n{'='*60}")
    print(f"DASK: {cfg['label']}  N={N:,}  M={M}  T={T}")
    n_workers = len(client.scheduler_info()["workers"])
    print(f"Cluster: {n_workers} workers")
    print(f"{'='*60}")

    # Generate seeds
    ss = np.random.SeedSequence(MASTER_SEED + METRIC_SEED_OFFSETS.get(metric, 999))
    all_seeds = ss.spawn(len(eps_values) * M)
    seed_idx = 0

    # Load existing results
    result_path = RESULTS_DIR / f"crn_{metric}_N{N}_T{T}.json"
    existing_by_eps = {}
    if result_path.exists():
        with open(result_path) as f:
            existing_by_eps = json.load(f).get("results", {})

    all_results = {}

    for eps in eps_values:
        M_this = 5 if abs(eps) < 1e-12 else M
        existing = existing_by_eps.get(str(eps), [])
        done_seeds = {r["seed"] for r in existing}
        results = list(existing)

        seeds = []
        for i in range(M):
            s = int(all_seeds[seed_idx + i].generate_state(1)[0])
            if i < M_this:
                seeds.append(s)
        seed_idx += M

        remaining = [s for s in seeds if s not in done_seeds]
        print(f"  eps={eps}: {len(existing)}/{M_this} done, {len(remaining)} remaining")

        if not remaining:
            all_results[str(eps)] = results
            continue

        # Determine which get random operator tests
        nonzero_eps = sorted([e for e in eps_values if abs(e) > 1e-12])
        top_two = set(nonzero_eps[-2:]) if len(nonzero_eps) >= 2 else set(nonzero_eps)

        # Submit all tasks to Dask
        futures = []
        for j, seed in enumerate(remaining):
            do_random = (eps in top_two) and (j < 20) and (mode == "dense")
            f = client.submit(
                crn_one, seed, N, T, eps,
                METRIC_FNS[metric], mode, do_random,
                key=f"crn_{metric}_{eps}_{seed}",
                pure=False,
            )
            futures.append(f)

        # Collect as they complete
        t0 = time.perf_counter()
        done_count = 0
        from dask.distributed import as_completed
        for batch in as_completed(futures, with_results=True):
            future, result = batch
            results.append(result)
            done_count += 1

            if done_count % 5 == 0 or done_count == len(remaining):
                elapsed = time.perf_counter() - t0
                rate = done_count / elapsed * 60 if elapsed > 0 else 0
                eta = (len(remaining) - done_count) / rate if rate > 0 else 0
                print(f"    {done_count}/{len(remaining)}  "
                      f"{rate:.1f}/min  ETA {eta:.0f} min")

                # Checkpoint
                _save(all_results | {str(eps): results},
                      metric, cfg, N, M, T, mode)

        # Print t-test for this eps
        if abs(eps) > 1e-12:
            deltas = [r["frobenius_delta"] for r in results]
            if len(deltas) >= 2:
                t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
                d_mean = float(np.mean(deltas))
                d_std = float(np.std(deltas, ddof=1))
                cohen_d = d_mean / (d_std + 1e-20)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"    frobenius: d={cohen_d:+.3f} p={p_val:.2e} {sig}")

        all_results[str(eps)] = results

    # Final save
    _save(all_results, metric, cfg, N, M, T, mode)

    # Post-experiment analysis
    analyze_experiment(all_results, cfg["label"], N)


def _save(all_results, metric, cfg, N, M, T, mode):
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


def main():
    parser = argparse.ArgumentParser(description="FND-1 Dask Distributed")
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--mode", choices=["dense", "matfree"], default="dense")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of local workers (default 8)")
    parser.add_argument("--scheduler", type=str, default=None,
                        help="Connect to existing scheduler (e.g. tcp://ip:8786)")
    args = parser.parse_args()

    from dask.distributed import Client, LocalCluster

    if args.scheduler:
        # Connect to existing scheduler (friend started it)
        client = Client(args.scheduler)
        print(f"Connected to scheduler: {args.scheduler}")
    else:
        # Start local scheduler + workers, open to network
        cluster = LocalCluster(
            scheduler_port=8786,
            host="0.0.0.0",
            n_workers=args.workers,
            threads_per_worker=1,
            memory_limit="16GiB",  # prevent pause/resume thrashing
        )
        client = Client(cluster)
        print(f"Scheduler: {cluster.scheduler_address}")
        print(f"Dashboard: http://100.71.68.111:8787/status")
        print(f"Friends connect: dask worker tcp://100.71.68.111:8786 "
              f"--nworkers 4 --nthreads 1")

    print(f"Workers: {len(client.scheduler_info()['workers'])}")

    if args.all:
        metrics = list(METRICS.keys())
    elif args.metric:
        metrics = [args.metric]
    else:
        metrics = ["ppwave_quad"]

    for metric in metrics:
        run_distributed(metric, args.N, args.M, args.T, args.mode,
                        client, args.workers)

    print("\nDone.")
    client.close()


if __name__ == "__main__":
    main()
