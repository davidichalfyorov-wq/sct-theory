"""
Tests for Iteration 34: cross-module hardening.

Covers:
    D-01: _serialize handles plain Python float NaN/Inf (not just np.floating)
    G-09: spectral_dimension_graph validates t_values for NaN/Inf/empty
    L-01: prove_local / prove_scilean reject empty/None code
    G-07: causal_set_dimension warns on brentq failure
"""

import json
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import data_io, graphs, lean

# ============================================================================
# D-01: _serialize plain Python float NaN/Inf
# ============================================================================


class TestSerializePlainFloat:
    """_serialize must handle plain Python float NaN/Inf like np.floating."""

    def test_python_nan_serialized_as_null(self, tmp_path):
        f = tmp_path / "nan_test.json"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_io.save_results(str(f), {'value': float('nan')})
            assert any("non-finite" in str(x.message) for x in w)
        with open(f) as fh:
            content = json.load(fh)
        assert content['results']['value'] is None

    def test_python_inf_serialized_as_null(self, tmp_path):
        f = tmp_path / "inf_test.json"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_io.save_results(str(f), {'value': float('inf')})
            assert any("non-finite" in str(x.message) for x in w)
        with open(f) as fh:
            content = json.load(fh)
        assert content['results']['value'] is None

    def test_python_neg_inf_serialized_as_null(self, tmp_path):
        f = tmp_path / "neginf_test.json"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_io.save_results(str(f), {'value': float('-inf')})
            assert any("non-finite" in str(x.message) for x in w)
        with open(f) as fh:
            content = json.load(fh)
        assert content['results']['value'] is None

    def test_python_finite_float_unchanged(self, tmp_path):
        f = tmp_path / "ok_test.json"
        data_io.save_results(str(f), {'pi': 3.14159})
        results, _ = data_io.load_results(str(f))
        assert results['pi'] == pytest.approx(3.14159)


# ============================================================================
# G-09: spectral_dimension_graph t_values validation
# ============================================================================


class TestSpectralDimensionTValuesGuard:
    """spectral_dimension_graph must validate t_values for NaN/Inf/empty."""

    def _make_adjacency(self):
        """Small complete graph adjacency matrix."""
        A = np.ones((4, 4)) - np.eye(4)
        return A

    def test_nan_in_t_values_raises(self):
        A = self._make_adjacency()
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.spectral_dimension_graph(A, t_values=[1.0, float('nan'), 3.0])

    def test_inf_in_t_values_raises(self):
        A = self._make_adjacency()
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.spectral_dimension_graph(A, t_values=[1.0, float('inf')])

    def test_empty_t_values_raises(self):
        A = self._make_adjacency()
        with pytest.raises(ValueError, match="non-empty"):
            graphs.spectral_dimension_graph(A, t_values=[])

    def test_valid_t_values_succeed(self):
        A = self._make_adjacency()
        t, d_S = graphs.spectral_dimension_graph(A, t_values=[0.1, 1.0, 10.0])
        assert len(t) == 3
        assert len(d_S) == 3


# ============================================================================
# L-01: prove_local / prove_scilean empty code validation
# ============================================================================


class TestProveLocalCodeValidation:
    """prove_local and prove_scilean must reject empty/None code."""

    def test_prove_local_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove_local("")

    def test_prove_local_whitespace_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove_local("   ")

    def test_prove_local_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            lean.prove_local(None)

    def test_prove_scilean_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove_scilean("")

    def test_prove_scilean_whitespace_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove_scilean("   ")

    def test_prove_scilean_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            lean.prove_scilean(None)


# ============================================================================
# G-07: causal_set_dimension warns on brentq failure
# ============================================================================


class TestCausalSetDimensionWarning:
    """causal_set_dimension must warn (not silently return NaN) on failure."""

    def test_extreme_ordering_fraction_returns_nan(self):
        """All pairs related (f=1) hits f>=1 guard -> NaN (no brentq call)."""
        N = 5
        C = np.ones((N, N), dtype=bool)
        np.fill_diagonal(C, False)
        # f=1 hits the f>=1 guard, returns NaN without calling brentq
        result = graphs.causal_set_dimension(C)
        assert np.isnan(result)

    def test_sparse_ordering_warns(self):
        """Very few relations -> f close to 0 -> dimension > 10 -> warn."""
        # ordering_fraction(d=10) ≈ 0.0005, so need f < 0.0005.
        # N=100, 1 relation -> f = 1/4950 ≈ 0.0002 -> brentq fails.
        N = 100
        C = np.zeros((N, N), dtype=bool)
        C[0, 1] = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = graphs.causal_set_dimension(C)
            brentq_warns = [
                x for x in w if "brentq" in str(x.message)
            ]
            assert len(brentq_warns) >= 1
        assert np.isnan(result)

    def test_normal_diamond_no_warning(self):
        """Normal 2D diamond causal set should not warn."""
        # causal_set_sprinkle returns (points, causal_matrix)
        _pts, C = graphs.causal_set_sprinkle(100, dim=2, region='diamond', seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = graphs.causal_set_dimension(C)
            brentq_warns = [x for x in w if "brentq" in str(x.message)]
            assert len(brentq_warns) == 0
        # Should give d ≈ 2 (within reasonable tolerance for N=100)
        assert 1.0 < d < 4.0
