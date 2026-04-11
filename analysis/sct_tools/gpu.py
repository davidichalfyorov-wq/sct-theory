"""
sct_tools.gpu — GPU initialization helper
===========================================

Usage:
    from sct_tools.gpu import xp, USE_GPU

    # xp is either cupy (if GPU available) or numpy (fallback)
    a = xp.random.rand(1000, 1000, dtype=xp.float32)
    b = a @ a

The module handles:
1. Windows DLL path fix for CUDA Toolkit
2. GPU detection and fallback to CPU
3. Provides unified `xp` namespace (cupy or numpy)
"""
import os

import numpy as np

# Windows: add CUDA toolkit DLLs to search path before importing CuPy
_cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
]
for _p in _cuda_paths:
    if os.path.isdir(_p):
        try:
            os.add_dll_directory(_p)
        except (OSError, AttributeError):
            pass
        break

USE_GPU = False
xp = np  # default: numpy

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        # Test that matmul actually works (cuBLAS loaded)
        _test = cp.ones((2, 2), dtype=cp.float32)
        _ = _test @ _test
        cp.cuda.Stream.null.synchronize()
        del _test

        USE_GPU = True
        xp = cp
except (ImportError, OSError, Exception):
    pass


def gpu_info():
    """Print GPU info if available."""
    if USE_GPU:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"GPU: {props['name'].decode()}")
        print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
        print(f"CuPy: {cp.__version__}")
    else:
        print("GPU: not available, using CPU (NumPy)")


def to_gpu(arr):
    """Move numpy array to GPU if available."""
    if USE_GPU:
        import cupy as cp
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Move GPU array to CPU (numpy). No-op if already numpy."""
    if USE_GPU:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


def free_gpu():
    """Free GPU memory pool."""
    if USE_GPU:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
