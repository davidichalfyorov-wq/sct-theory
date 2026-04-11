"""
FND-1: Sequential experiment launcher.

Runs experiments one at a time so each gets FULL access to all CPU cores.
This avoids the problem of 9 concurrent experiments fighting over CPUs.

Order: fastest first (get results early), slowest last.

Usage:
    python analysis/scripts/fnd1_run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent.parent / "speculative" / "numerics" / "ensemble_results"

# Ordered fast → slow based on benchmarks.
# Each tuple: (script_name, expected_json, estimated_minutes_at_80w)
# Split into two batches for spot instance safety.
# Batch A: fast experiments (~40 min total). Download JSONs, then start Batch B.
# Batch B: heavy experiments (~1.5 h total).

BATCH_A = [
    ("fnd1_exp8_d2_hasse_ricci.py",     "exp8_d2_hasse_ricci.json",       3),
    ("fnd1_exp11_d3_intermediate.py",    "exp11_d3_intermediate.json",     5),
    ("fnd1_exp3_d4_commutator.py",       "exp3_d4_commutator.json",        8),
    ("fnd1_exp5_d4_spectral_action.py",  "exp5_d4_spectral_action.json",  10),
    ("fnd1_exp9_d2_large_ensemble.py",   "exp9_d2_large_ensemble.json",   15),
    ("fnd1_exp4_d4_magnetic_phase.py",   "exp4_d4_magnetic_phase.json",   15),
]

BATCH_B = [
    ("fnd1_exp1_d4_link_verification.py","exp1_d4_link_verification.json", 20),
    ("fnd1_exp6_d4_higher_N.py",         "exp6_d4_higher_N.json",         25),
    ("fnd1_exp2_d4_sj_verification.py",  "exp2_d4_sj_verification.json",  30),
]

# Select batch via command line: python fnd1_run_all.py [a|b|all]
EXPERIMENTS = BATCH_A + BATCH_B  # default: all


def main():
    # Select batch
    batch_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if batch_arg == "a":
        experiments = BATCH_A
        batch_name = "BATCH A (fast, ~40 min)"
    elif batch_arg == "b":
        experiments = BATCH_B
        batch_name = "BATCH B (heavy, ~1.5 h)"
    else:
        experiments = EXPERIMENTS
        batch_name = "ALL (A + B)"

    t_total = time.perf_counter()

    print("=" * 70)
    print(f"FND-1 SEQUENTIAL LAUNCHER — {batch_name}")
    print("=" * 70)
    print(f"Experiments: {len(experiments)}")
    est_total = sum(e[2] for e in experiments)
    print(f"Estimated total: ~{est_total} min ({est_total / 60:.1f} h)")
    print()

    # Show system info
    from multiprocessing import cpu_count
    import os
    print(f"CPU cores: {cpu_count()}")
    try:
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
        print(f"RAM: {ram_gb:.0f} GB")
    except (AttributeError, ValueError):
        pass

    from fnd1_parallel import N_WORKERS, OPENBLAS_THREADS_PER_WORKER, N_GPUS
    print(f"N_WORKERS: {N_WORKERS}, BLAS threads/worker: {OPENBLAS_THREADS_PER_WORKER}")
    print(f"GPUs detected: {N_GPUS}")
    if N_GPUS > 0:
        try:
            import cupy as cp
            print(f"CuPy version: {cp.__version__}")
            for i in range(N_GPUS):
                with cp.cuda.Device(i):
                    mem = cp.cuda.Device(i).mem_info
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    print(f"  GPU {i}: {str(props['name'], 'utf-8')}, "
                          f"{mem[1] / 1e9:.0f} GB total, {mem[0] / 1e9:.0f} GB free")
        except ImportError:
            print("  WARNING: CuPy not installed — falling back to CPU!")
    print()

    completed = 0
    failed = []

    for script, expected_json, est_min in experiments:
        script_path = SCRIPTS_DIR / script
        json_path = RESULTS_DIR / expected_json

        print(f"{'=' * 70}")
        print(f"[{completed + 1}/{len(experiments)}] {script}")
        print(f"  Estimated: ~{est_min} min")
        print(f"{'=' * 70}", flush=True)

        timeout_sec = max(7200, est_min * 180)  # 3x estimated time, min 2h

        t0 = time.perf_counter()
        try:
            # -u for unbuffered output; start_new_session for clean process group kill
            proc = subprocess.Popen(
                [sys.executable, "-u", str(script_path)],
                cwd=str(SCRIPTS_DIR.parent.parent),
                start_new_session=True,
            )
            proc.wait(timeout=timeout_sec)
            elapsed = time.perf_counter() - t0

            class _R:
                returncode = proc.returncode
            result = _R()

            if result.returncode == 0 and json_path.exists():
                completed += 1
                print(f"\n  DONE: {script} ({elapsed / 60:.1f} min)")
            elif result.returncode == 0:
                completed += 1
                print(f"\n  DONE (no JSON?): {script} ({elapsed / 60:.1f} min)")
            else:
                failed.append((script, f"exit code {result.returncode}"))
                print(f"\n  FAILED: {script} (exit={result.returncode}, {elapsed / 60:.1f} min)")
        except subprocess.TimeoutExpired:
            # Kill entire process group (workers are grandchildren)
            import os, signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            elapsed = time.perf_counter() - t0
            failed.append((script, f"TIMEOUT ({timeout_sec}s)"))
            print(f"\n  TIMEOUT: {script} ({elapsed / 60:.1f} min)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            failed.append((script, str(e)))
            print(f"\n  ERROR: {script}: {e} ({elapsed / 60:.1f} min)")

        print(flush=True)

    total_time = time.perf_counter() - t_total

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed: {completed}/{len(experiments)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
    print(f"Total wall time: {total_time / 60:.1f} min ({total_time / 3600:.1f} h)")

    # List produced JSONs
    print(f"\nJSON results:")
    for _, expected_json, _ in experiments:
        jp = RESULTS_DIR / expected_json
        if jp.exists():
            sz = jp.stat().st_size / 1024
            print(f"  [OK] {expected_json} ({sz:.0f} KB)")
        else:
            print(f"  [--] {expected_json}")

    print("=" * 70)


if __name__ == "__main__":
    main()
