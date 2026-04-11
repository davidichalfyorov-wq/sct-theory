"""
FND-1: GPU-accelerated linear algebra via CuPy.

Transparent fallback to numpy when CuPy / CUDA is unavailable.
Designed for multi-GPU nodes (e.g. 8x H100 80GB).

All functions are safe to call from multiprocessing workers:
- CuPy import is lazy (happens on first call, AFTER fork)
- GPU assignment via CUDA_VISIBLE_DEVICES in worker init

Usage in workers:
    from fnd1_gpu import gpu_eigvalsh, gpu_eigh, gpu_matmul

    eigenvalues = gpu_eigvalsh(M)              # symmetric/Hermitian
    eigenvalues, eigenvectors = gpu_eigh(M)    # with eigenvectors
    C2 = gpu_matmul(C, C)                      # dense matmul
"""

from __future__ import annotations

import os
import numpy as np

# Lazy state — initialized on first GPU call (must be AFTER fork)
_initialized = False
_use_gpu = False
_gpu_id = -1

# Safety flag: only enable GPU in Pool workers (set by _init_worker).
# Prevents CUDA init in main process before fork, which would poison children.
_worker_mode = False


def enable_worker_mode():
    """Call from Pool initializer to allow GPU in this process."""
    global _worker_mode, _initialized
    _worker_mode = True
    _initialized = False  # force re-init so GPU is detected fresh


def _lazy_init():
    """Initialize CuPy on first use. Only enables GPU if in worker mode."""
    global _initialized, _use_gpu, _gpu_id
    if _initialized:
        return
    _initialized = True
    if not _worker_mode:
        # Main process — never touch CUDA (would break fork)
        _use_gpu = False
        return
    try:
        import cupy as cp
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices > 0:
            _gpu_id = cp.cuda.runtime.getDevice()
            _use_gpu = True
        else:
            _use_gpu = False
    except Exception:
        _use_gpu = False


def detect_gpus() -> int:
    """Detect number of GPUs WITHOUT initializing CUDA.
    Safe to call in main process before fork.
    """
    try:
        # Use nvidia-smi to count GPUs without CUDA init
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except Exception:
        pass
    return 0


def assign_gpu_to_worker(worker_global_id: int, n_gpus: int):
    """Set CUDA_VISIBLE_DEVICES for this worker. Call in Pool initializer."""
    if n_gpus > 0:
        gpu_id = worker_global_id % n_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# ---------------------------------------------------------------------------
# GPU-accelerated operations
# ---------------------------------------------------------------------------

def _gpu_op(cpu_func, *args):
    """Try GPU, fall back to CPU on ANY error (cusolver missing, OOM, etc.)."""
    global _use_gpu
    _lazy_init()
    if _use_gpu:
        try:
            import cupy as cp
            gpu_args = [cp.asarray(a) for a in args]
            result = cpu_func.__name__  # just for the error message
            if cpu_func is np.linalg.eigvalsh:
                r = cp.linalg.eigvalsh(gpu_args[0])
                return cp.asnumpy(r)
            elif cpu_func is np.linalg.eigh:
                eigs, evecs = cp.linalg.eigh(gpu_args[0])
                return cp.asnumpy(eigs), cp.asnumpy(evecs)
            elif cpu_func is _matmul_cpu:
                r = gpu_args[0] @ gpu_args[1]
                return cp.asnumpy(r)
            elif cpu_func is _svd_cpu:
                r = cp.linalg.svd(gpu_args[0], compute_uv=False)
                return cp.asnumpy(r)
        except Exception as e:
            # GPU failed — disable for this process and fall back
            import sys
            print(f"  [GPU fallback] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            _use_gpu = False
    return cpu_func(*args)


def _matmul_cpu(a, b):
    return a @ b


def _svd_cpu(m):
    return np.linalg.svd(m, compute_uv=False)


def gpu_eigvalsh(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of real symmetric or complex Hermitian matrix."""
    return _gpu_op(np.linalg.eigvalsh, matrix)


def gpu_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues + eigenvectors of symmetric/Hermitian matrix."""
    return _gpu_op(np.linalg.eigh, matrix)


def gpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense matrix multiplication on GPU."""
    return _gpu_op(_matmul_cpu, a, b)


def gpu_svd_values(matrix: np.ndarray) -> np.ndarray:
    """Singular values only (no U, V). GPU-accelerated."""
    return _gpu_op(_svd_cpu, matrix)


def gpu_info() -> dict:
    """Return GPU info dict (safe for JSON). Call from worker only."""
    _lazy_init()
    if _use_gpu:
        import cupy as cp
        dev = cp.cuda.Device()
        mem = dev.mem_info
        return {
            "gpu_available": True,
            "gpu_id": int(dev.id),
            "gpu_name": str(cp.cuda.runtime.getDeviceProperties(dev.id)["name"], "utf-8"),
            "gpu_mem_total_gb": round(mem[1] / 1e9, 1),
            "gpu_mem_free_gb": round(mem[0] / 1e9, 1),
            "cupy_version": cp.__version__,
        }
    return {"gpu_available": False}
