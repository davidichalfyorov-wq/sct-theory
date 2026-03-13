"""
SCT Theory — Parallel and high-performance computation utilities.

Provides:
    - parallel_scan(): joblib-based parameter scans
    - progress_compute(): tqdm-wrapped batch computation
    - cached(): joblib.Memory disk-cache decorator for expensive evaluations
    - vegas_integrate(): adaptive Monte Carlo for multi-dim integrals
    - symengine_convert(): convert sympy expressions to symengine for speed
    - precision_context(): mpmath context manager for setting dps
    - jax_grad(), jax_hessian(): automatic differentiation via JAX
    - jax_jit(): JIT compilation of numerical functions via JAX
    - wsl_run(): execute Python code in WSL2 venv (healpy, CLASS, cadabra2, pySecDec)
    - wsl_run_script(): run a .py file in WSL2 venv
    - wsl_check(): verify WSL2 packages availability
"""

import os

import numpy as np
from joblib import Memory, Parallel, delayed
from tqdm import tqdm

# =============================================================================
# DISK CACHE (joblib.Memory)
# =============================================================================
# Cache directory: analysis/.sct_cache/ (auto-created)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.sct_cache')
_memory = Memory(_CACHE_DIR, verbose=0)


def cached(func):
    """Decorator: cache function results to disk via joblib.Memory.

    Cached results persist across sessions in analysis/.sct_cache/.
    Use clear_cache() to invalidate.

    Example:
        @cached
        def expensive_hC(x, xi=0):
            return hC_scalar_mp(x, xi=xi, dps=200)
    """
    return _memory.cache(func)


def clear_cache():
    """Clear all cached computation results."""
    _memory.clear(warn=False)


def parallel_scan(func, param_grid, n_jobs=-1, verbose=0, desc=None):
    """Run func over a parameter grid in parallel.

    Parameters:
        func: callable, takes **params and returns scalar or dict
        param_grid: list of dicts, each dict is a set of kwargs for func
        n_jobs: number of parallel jobs (-1 = all cores)
        verbose: joblib verbosity (0=silent, 10=max)
        desc: tqdm description string (None = no progress bar)

    Returns:
        List of results in same order as param_grid.

    Example:
        grid = [{'x': x, 'xi': xi} for x in np.linspace(0,10,100)
                                     for xi in [0, 1/6, 1]]
        results = parallel_scan(hR_scalar, grid)
    """
    if desc is not None:
        param_grid = tqdm(param_grid, desc=desc, unit="pt")
    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(func)(**p) for p in param_grid
    )


def progress_compute(func, x_values, desc="Computing", **kwargs):
    """Evaluate func(x, **kwargs) over x_values with progress bar.

    Parameters:
        func: callable f(x, **kwargs) -> scalar
        x_values: iterable of x values
        desc: progress bar description
        **kwargs: extra arguments passed to func

    Returns:
        numpy array of results.
    """
    results = []
    for x in tqdm(x_values, desc=desc, unit="pt"):
        results.append(func(x, **kwargs))
    arr = np.array(results)
    if arr.dtype == object:
        import warnings
        warnings.warn(
            f"progress_compute: result array has dtype=object (func may return "
            f"None or mixed types). First elements: {results[:3]}",
            stacklevel=2,
        )
    return arr


def progress_compute_mp(func, x_values, dps=100, desc="Computing (mpmath)"):
    """Evaluate mpmath func(x, dps=dps) over x_values with progress bar.

    Parameters:
        func: callable f(x, dps=...) -> mpmath number
        x_values: iterable of x values
        dps: decimal places
        desc: progress bar description

    Returns:
        List of mpmath values (not numpy — mpmath numbers don't fit in arrays).
    """
    if not isinstance(dps, int) or dps <= 0:
        raise ValueError(
            f"progress_compute_mp: dps must be a positive integer, got {dps}"
        )
    results = []
    for x in tqdm(x_values, desc=desc, unit="pt"):
        results.append(func(x, dps=dps))
    if any(r is None for r in results):
        import warnings
        warnings.warn(
            f"progress_compute_mp: {sum(1 for r in results if r is None)} of "
            f"{len(results)} results are None (func may have failed silently).",
            stacklevel=2,
        )
    return results


def vegas_integrate(integrand, limits, nitn=10, neval=10000, adapt=True):
    """Adaptive Monte Carlo integration using vegas.

    Superior to grid-based methods for d >= 2 dimensions.
    Uses importance sampling that adapts to the integrand shape.

    Parameters:
        integrand: callable f(x) -> scalar, where x is array of shape (d,)
                   or vegas.BatchIntegrand for vectorized evaluation
        limits: list of (lo, hi) tuples, one per dimension
        nitn: number of iterations for adaptation + evaluation
        neval: number of integrand evaluations per iteration
        adapt: if True, first run adaptation iterations

    Returns:
        vegas.RAvg result with .mean, .sdev, .Q (quality factor).
        Q > 0.1 indicates good convergence.

    Example:
        # 3D integral of exp(-x^2 - y^2 - z^2) over [-1,1]^3
        def f(x):
            return np.exp(-x[0]**2 - x[1]**2 - x[2]**2)
        result = vegas_integrate(f, [(-1,1), (-1,1), (-1,1)])
        print(f"{result.mean:.6f} +/- {result.sdev:.6f}, Q={result.Q:.2f}")
    """
    if not limits:
        raise ValueError("vegas_integrate: limits must be a non-empty list of (lo, hi) tuples")
    for i, lim in enumerate(limits):
        if not isinstance(lim, (list, tuple)) or len(lim) != 2:
            raise ValueError(f"vegas_integrate: limits[{i}] must be a (lo, hi) tuple, got {lim!r}")
        if lim[0] >= lim[1]:
            raise ValueError(f"vegas_integrate: limits[{i}] has lo >= hi: ({lim[0]}, {lim[1]})")
    import vegas
    integ = vegas.Integrator(limits)
    if adapt:
        integ(integrand, nitn=nitn, neval=neval)  # adaptation
    result = integ(integrand, nitn=nitn, neval=neval)  # evaluation
    return result


def symengine_simplify(sympy_expr):
    """Convert a SymPy expression to SymEngine, simplify, convert back.

    SymEngine is 10-100x faster for large expressions.
    Falls back to SymPy if conversion fails.

    Parameters:
        sympy_expr: sympy.Expr

    Returns:
        Simplified sympy.Expr
    """
    try:
        import symengine as se
        import sympy
        se_expr = se.sympify(sympy_expr)
        se_simplified = se.expand(se_expr)
        return sympy.sympify(str(se_simplified))
    except Exception as e:
        import warnings
        warnings.warn(
            f"SymEngine simplify failed ({e}), falling back to SymPy",
            stacklevel=2,
        )
        import sympy
        return sympy.simplify(sympy_expr)


def symengine_diff(sympy_expr, *symbols):
    """Differentiate a SymPy expression using SymEngine backend.

    10-100x faster than sympy.diff for large expressions.

    Parameters:
        sympy_expr: sympy expression
        *symbols: sympy symbols to differentiate with respect to

    Returns:
        Derivative as sympy.Expr
    """
    try:
        import symengine as se
        import sympy
        se_expr = se.sympify(sympy_expr)
        for sym in symbols:
            se_expr = se.diff(se_expr, se.Symbol(str(sym)))
        return sympy.sympify(str(se_expr))
    except Exception as e:
        import warnings
        warnings.warn(
            f"SymEngine diff failed ({e}), falling back to SymPy",
            stacklevel=2,
        )
        import sympy
        result = sympy_expr
        for sym in symbols:
            result = sympy.diff(result, sym)
        return result


# =============================================================================
# JAX AUTOMATIC DIFFERENTIATION & JIT
# =============================================================================

def jax_grad(func, argnums=0):
    """Create a gradient function using JAX automatic differentiation.

    Exact gradients (not finite-difference) for any differentiable function.
    Critical for: variational calculus (NT-4 field equations),
    gradient-based optimization, sensitivity analysis.

    Parameters:
        func: callable f(*args) -> scalar (must use jax.numpy ops)
        argnums: which argument(s) to differentiate w.r.t. (int or tuple)

    Returns:
        Gradient function with same signature as func.

    Example:
        import jax.numpy as jnp
        def action(g_mu_nu, phi):
            return jnp.sum(g_mu_nu * phi**2)
        dS_dg = jax_grad(action, argnums=0)
    """
    import jax
    return jax.grad(func, argnums=argnums)


def jax_hessian(func, argnums=0):
    """Create a Hessian function using JAX automatic differentiation.

    Second derivatives via forward-over-reverse mode.
    Useful for: stability analysis, mass matrices, curvature of
    the effective potential.

    Parameters:
        func: callable f(*args) -> scalar
        argnums: which argument to differentiate w.r.t.

    Returns:
        Hessian function returning (n, n) array.
    """
    import jax
    return jax.hessian(func, argnums=argnums)


def jax_jacobian(func, argnums=0):
    """Create a Jacobian function using JAX.

    For vector-valued functions f: R^n -> R^m.

    Parameters:
        func: callable f(*args) -> array
        argnums: which argument to differentiate w.r.t.

    Returns:
        Jacobian function returning (m, n) array.
    """
    import jax
    return jax.jacfwd(func, argnums=argnums)


def jax_jit(func):
    """JIT-compile a function using JAX XLA compiler.

    Traces the function and compiles to optimized machine code.
    First call is slow (compilation), subsequent calls are fast.
    Function must use jax.numpy, not regular numpy.

    Parameters:
        func: callable using jax.numpy operations

    Returns:
        JIT-compiled version of func.

    Example:
        import jax.numpy as jnp
        @jax_jit
        def fast_phi(x):
            return jnp.where(x < 2.0, 1.0 - x/6, 2*dawsn(jnp.sqrt(x)/2)/jnp.sqrt(x))
    """
    import jax
    return jax.jit(func)


def jax_vmap(func, in_axes=0):
    """Vectorize a function over a batch dimension using JAX.

    Automatically maps a scalar function over arrays without
    explicit loops. Composes with jit for maximum speed.

    Parameters:
        func: callable operating on single elements
        in_axes: which axes to vectorize over (0 = first axis)

    Returns:
        Vectorized version of func.

    Example:
        scalar_hC = lambda x: ...  # scalar function
        batch_hC = jax_vmap(scalar_hC)  # now works on arrays
    """
    import jax
    return jax.vmap(func, in_axes=in_axes)


class precision_context:
    """Context manager for mpmath precision.

    WARNING: mpmath.mp.dps is process-global. Not thread-safe.

    Usage:
        with precision_context(200):
            result = phi_mp(x)  # uses 200-digit precision
    """
    def __init__(self, dps):
        if not isinstance(dps, int) or dps < 1:
            raise ValueError(
                f"precision_context: dps must be a positive integer, got {dps!r}"
            )
        self.dps = dps
        self.old_dps = None

    def __enter__(self):
        import mpmath
        self.old_dps = mpmath.mp.dps
        mpmath.mp.dps = self.dps
        return self

    def __exit__(self, *args):
        import mpmath
        mpmath.mp.dps = self.old_dps


# =============================================================================
# BENCHMARK TIMER
# =============================================================================

class benchmark:
    """Context manager for timing computations.

    Usage:
        with benchmark("matrix diag") as b:
            eigenvalues = np.linalg.eigvalsh(big_matrix)
        print(f"Took {b.elapsed:.3f}s")

    Also works as decorator:
        @benchmark("form factor scan")
        def run_scan():
            ...
    """
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[BENCH] {self.label}: {self.elapsed:.4f}s")

    def __call__(self, func):
        """Use as decorator."""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def cache_info():
    """Return cache directory path and approximate size in MB."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(_CACHE_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except FileNotFoundError:
                continue
    return {
        'path': _CACHE_DIR,
        'size_mb': total / (1024 * 1024),
        'exists': os.path.isdir(_CACHE_DIR),
    }


# =============================================================================
# WSL2 INTEGRATION (healpy, CLASS, cadabra2, pySecDec)
# =============================================================================

_WSL_VENV = "~/sct-wsl"
_WSL_DISTRO = "Ubuntu"


def wsl_run(python_code, timeout=120, return_json=True):
    """Execute Python code in the WSL2 venv and return the result.

    Runs the given code in the WSL2 Ubuntu venv (~/sct-wsl) where
    healpy, CLASS, cadabra2, and pySecDec are installed.

    Parameters:
        python_code: str, Python code to execute. If return_json=True,
                     the code must print a JSON-serializable result via
                     print(json.dumps(result)).
        timeout: max seconds to wait (default 120)
        return_json: if True, parse stdout as JSON and return the object.
                     If False, return raw stdout string.

    Returns:
        Parsed JSON object (if return_json=True) or stdout string.

    Raises:
        RuntimeError: if WSL command fails or times out.
        json.JSONDecodeError: if return_json=True but output isn't valid JSON.

    Example:
        # Get CMB angular power spectrum from CLASS
        result = wsl_run('''
        import json
        from classy import Class
        cosmo = Class()
        cosmo.set({"output": "tCl", "l_max_scalars": 2500})
        cosmo.compute()
        cls = cosmo.raw_cl(2500)
        json.dumps({"ell": cls["ell"][:10].tolist(),
                     "tt": cls["tt"][:10].tolist()})
        cosmo.clean()
        print(json.dumps(result))
        ''')
    """
    if not isinstance(python_code, str):
        raise TypeError(
            f"wsl_run: python_code must be a string, got {type(python_code).__name__}"
        )
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"wsl_run: timeout must be positive, got {timeout}")

    import json
    import shlex
    import subprocess

    activate = f"source {_WSL_VENV}/bin/activate"
    escaped = shlex.quote(python_code)
    cmd = [
        "wsl", "-d", _WSL_DISTRO, "--",
        "bash", "-c",
        f"{activate} && python3 -c {escaped}"
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"WSL command timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"WSL Python failed (rc={proc.returncode}):\n{proc.stderr.strip()}"
        )

    stdout = proc.stdout.strip()
    if return_json:
        # Parse only the last non-empty line (preceding lines may be
        # library banners, deprecation warnings, or debug prints).
        lines = [ln for ln in stdout.split('\n') if ln.strip()]
        json_line = lines[-1] if lines else ""
        if not json_line:
            raise RuntimeError(
                "WSL command produced no stdout output. When return_json=True, "
                "the Python code must print a JSON-serializable string via "
                "print(json.dumps(result))."
            )
        return json.loads(json_line)
    return stdout


def wsl_run_script(script_path, args=None, timeout=300):
    """Execute a Python script in the WSL2 venv.

    Parameters:
        script_path: Windows path to the .py file (auto-converted to WSL path)
        args: optional list of CLI arguments
        timeout: max seconds to wait

    Returns:
        stdout string from the script.

    Raises:
        RuntimeError: if script fails or times out.
    """
    import subprocess

    # Convert Windows path to WSL path
    # F:\foo\bar -> /mnt/f/foo/bar
    script_path = str(script_path)
    if not os.path.isabs(script_path):
        raise ValueError(
            f"wsl_run_script: requires an absolute Windows path, got: {script_path!r}"
        )
    wsl_path = script_path.replace("\\", "/")
    if len(wsl_path) >= 2 and wsl_path[1] == ":":
        drive = wsl_path[0].lower()
        wsl_path = f"/mnt/{drive}{wsl_path[2:]}"

    import shlex

    activate = f"source {_WSL_VENV}/bin/activate"
    cmd_parts = [f"{activate} && python3 {shlex.quote(wsl_path)}"]
    if args:
        arg_str = " ".join(shlex.quote(a) for a in args)
        cmd_parts[0] += " " + arg_str

    cmd = [
        "wsl", "-d", _WSL_DISTRO, "--",
        "bash", "-c", cmd_parts[0]
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"WSL script timed out after {timeout}s"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"WSL script failed (rc={proc.returncode}):\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def wsl_check():
    """Verify WSL2 venv and key packages are available.

    Returns:
        dict with keys: 'available' (bool), 'packages' (dict of name->version),
        'errors' (list of error strings).
    """
    code = """
import json, importlib, sys
packages = {
    'healpy': 'healpy',
    'classy': 'classy',
    'cadabra2': 'cadabra2',
    'pySecDec': 'pySecDec',
    'numpy': 'numpy',
    'scipy': 'scipy',
}
result = {'versions': {}, 'errors': []}
for name, module in packages.items():
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, '__version__', getattr(mod, 'version', 'unknown'))
        result['versions'][name] = str(ver)
    except ImportError as e:
        result['errors'].append(f"{name}: {e}")
print(json.dumps(result))
"""
    try:
        data = wsl_run(code, timeout=30, return_json=True)
        return {
            'available': True,
            'packages': data.get('versions', {}),
            'errors': data.get('errors', []),
        }
    except Exception as e:
        return {
            'available': False,
            'packages': {},
            'errors': [str(e)],
        }
