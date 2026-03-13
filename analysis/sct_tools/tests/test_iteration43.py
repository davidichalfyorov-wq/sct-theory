"""
Tests for Iteration 43: scan_hC_vector/scan_hR_vector coverage,
error message prefix fixes in tensors.py, compute.py, data_io.py.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from sct_tools import form_factors as ff

# ============================================================================
# scan_hC_vector: basic properties
# ============================================================================


class TestScanHCVector:
    """Tests for scan_hC_vector (vectorized h_C^(1) over array)."""

    def test_matches_hC_vector_fast_pointwise(self):
        """scan_hC_vector must equal hC_vector_fast at each point."""
        x_arr = np.array([0.01, 0.1, 1.0, 5.0, 20.0, 100.0])
        scan_result = ff.scan_hC_vector(x_arr)
        for i, x in enumerate(x_arr):
            assert scan_result[i] == pytest.approx(
                ff.hC_vector_fast(x), rel=1e-12
            ), f"scan_hC_vector[{i}] at x={x}"

    def test_matches_mpmath(self):
        """scan_hC_vector must match mpmath reference to 10 digits."""
        x_arr = np.array([0.1, 1.0, 10.0, 50.0])
        scan_result = ff.scan_hC_vector(x_arr)
        for i, x in enumerate(x_arr):
            mp_val = float(ff.hC_vector_mp(x, dps=50))
            assert scan_result[i] == pytest.approx(
                mp_val, rel=1e-10
            ), f"scan_hC_vector[{i}] at x={x}: got {scan_result[i]}, mpmath={mp_val}"

    def test_empty_array(self):
        result = ff.scan_hC_vector(np.array([]))
        assert len(result) == 0

    def test_single_element(self):
        result = ff.scan_hC_vector(np.array([1.0]))
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_returns_ndarray(self):
        result = ff.scan_hC_vector(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_beta_W_vector_limit(self):
        """h_C^(1)(x->0) = 1/10 (physical, ghost-subtracted)."""
        # beta_W^(1) = 1/10 = h_C^(1)(0) from Phase 2
        result = ff.scan_hC_vector(np.array([1e-8]))
        assert result[0] == pytest.approx(0.1, rel=1e-3)

    def test_large_x_decay(self):
        """h_C^(1)(x) should decay for large x."""
        x_arr = np.array([100.0, 1000.0])
        result = ff.scan_hC_vector(x_arr)
        assert abs(result[1]) < abs(result[0])


# ============================================================================
# scan_hR_vector: basic properties
# ============================================================================


class TestScanHRVector:
    """Tests for scan_hR_vector (vectorized h_R^(1) over array)."""

    def test_matches_hR_vector_fast_pointwise(self):
        """scan_hR_vector must equal hR_vector_fast at each point."""
        x_arr = np.array([0.01, 0.1, 1.0, 5.0, 20.0, 100.0])
        scan_result = ff.scan_hR_vector(x_arr)
        for i, x in enumerate(x_arr):
            assert scan_result[i] == pytest.approx(
                ff.hR_vector_fast(x), rel=1e-12
            ), f"scan_hR_vector[{i}] at x={x}"

    def test_matches_mpmath(self):
        """scan_hR_vector must match mpmath reference to 10 digits."""
        x_arr = np.array([0.1, 1.0, 10.0, 50.0])
        scan_result = ff.scan_hR_vector(x_arr)
        for i, x in enumerate(x_arr):
            mp_val = float(ff.hR_vector_mp(x, dps=50))
            assert scan_result[i] == pytest.approx(
                mp_val, rel=1e-10
            ), f"scan_hR_vector[{i}] at x={x}: got {scan_result[i]}, mpmath={mp_val}"

    def test_empty_array(self):
        result = ff.scan_hR_vector(np.array([]))
        assert len(result) == 0

    def test_single_element(self):
        result = ff.scan_hR_vector(np.array([1.0]))
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_returns_ndarray(self):
        result = ff.scan_hR_vector(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_beta_R_vector_zero(self):
        """h_R^(1)(0) = 0 (conformal invariance), so beta_R^(1) = 0."""
        # h_R^(1)(x->0) = 0 from Taylor expansion
        result = ff.scan_hR_vector(np.array([1e-8]))
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_large_x_decay(self):
        """h_R^(1)(x) should decay for large x."""
        x_arr = np.array([100.0, 1000.0])
        result = ff.scan_hR_vector(x_arr)
        assert abs(result[1]) < abs(result[0])


# ============================================================================
# scan_hC_vector + scan_hR_vector: combined length/shape tests
# ============================================================================


class TestScanVectorShape:
    """Shape and length consistency for vector scan functions."""

    def test_consistent_lengths(self):
        n = 7
        x_arr = np.linspace(0.01, 50, n)
        assert len(ff.scan_hC_vector(x_arr)) == n
        assert len(ff.scan_hR_vector(x_arr)) == n

    def test_list_input(self):
        """Should accept plain list as input."""
        result_c = ff.scan_hC_vector([1.0, 2.0, 3.0])
        result_r = ff.scan_hR_vector([1.0, 2.0, 3.0])
        assert len(result_c) == 3
        assert len(result_r) == 3


# ============================================================================
# Error message prefix tests: tensors.py
# ============================================================================


class TestGeodesicEquationsErrorPrefix:
    """geodesic_equations error must follow 'geodesic_equations: ...' format."""

    def test_unknown_method_prefix(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors

        metric, _ = tensors.schwarzschild()
        with pytest.raises(ValueError, match="^geodesic_equations: unknown method"):
            tensors.geodesic_equations(metric, method="bogus")


class TestWeylTensorErrorPrefix:
    """weyl_tensor error must follow 'weyl_tensor: ...' format."""

    def test_dim_lt_3_prefix(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors

        coords, syms = tensors.cartesian_coords(dim=2)

        import OGRePy as gp

        eta = gp.Metric(
            coords=coords,
            components=[[-1, 0], [0, 1]],
        )
        with pytest.raises(ValueError, match="^weyl_tensor: only defined"):
            tensors.weyl_tensor(eta)


# ============================================================================
# Error message prefix tests: compute.py
# ============================================================================


class TestWslRunScriptErrorPrefix:
    """wsl_run_script error must follow 'wsl_run_script: ...' format."""

    def test_relative_path_prefix(self):
        from sct_tools import compute

        with pytest.raises(ValueError, match="^wsl_run_script: requires"):
            compute.wsl_run_script("relative/path/script.py")


# ============================================================================
# Error message prefix tests: data_io.py
# ============================================================================


class TestLoadResultsErrorPrefix:
    """load_results error must follow 'load_results: ...' format."""

    def test_missing_results_key_prefix(self):
        from sct_tools import data_io

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"metadata": {}}, f)
            f.flush()
            tmppath = f.name
        try:
            with pytest.raises(ValueError, match="^load_results: file"):
                data_io.load_results(tmppath)
        finally:
            os.unlink(tmppath)
