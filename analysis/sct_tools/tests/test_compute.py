"""
Pytest tests for sct_tools.compute utilities.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import compute
from sct_tools import form_factors as ff


class TestParallelScan:
    def test_basic_scan(self):
        grid = [{'x': x} for x in [0, 1, 5, 10]]
        results = compute.parallel_scan(ff.hC_scalar, grid, n_jobs=1)
        assert len(results) == 4
        assert results[0] == pytest.approx(1 / 120, rel=1e-10)

    def test_scan_with_kwargs(self):
        grid = [{'x': 0, 'xi': xi} for xi in [0, 1 / 6, 1]]
        results = compute.parallel_scan(ff.hR_scalar, grid, n_jobs=1)
        assert results[1] == pytest.approx(0, abs=1e-14)


class TestProgressCompute:
    def test_basic(self):
        x_vals = np.linspace(0.1, 10, 5)
        results = compute.progress_compute(ff.hC_scalar, x_vals, desc=None)
        assert len(results) == 5
        # Verify results match direct function calls
        direct = [ff.hC_scalar(x) for x in x_vals]
        for r, d in zip(results, direct):
            assert r == pytest.approx(d, rel=1e-14)


class TestSymengineFunctions:
    def test_simplify(self):
        import sympy as sp
        x = sp.Symbol('x')
        expr = (x**2 + 2 * x + 1) - (x + 1)**2
        result = compute.symengine_simplify(expr)
        assert result == 0

    def test_diff(self):
        import sympy as sp
        x = sp.Symbol('x')
        expr = x**3 + 2 * x
        result = compute.symengine_diff(expr, x)
        assert result == 3 * x**2 + 2


class TestPrecisionContext:
    def test_context_restores(self):
        from mpmath import mp
        original = mp.dps
        with compute.precision_context(200):
            assert mp.dps == 200
        assert mp.dps == original
