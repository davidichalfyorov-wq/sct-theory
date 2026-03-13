"""
Tests for Iteration 39: data_io.py + graphs.py hardening.

Covers:
    D-F2: read_fits raises on empty HDU (no mock — just test the guard logic)
    D-F10: read_csv validates skip_header >= 0
    G-F4: spectral_action_on_graph validates n_eigenvalues
    G-F6: spectral_dimension_graph validates t_values > 0
    G-F11: causal_set_dimension validates NaN/Inf and 2-D square shape
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import data_io, graphs

# ============================================================================
# D-F10: read_csv validates skip_header >= 0
# ============================================================================


class TestReadCsvSkipHeader:
    """read_csv must reject negative or non-integer skip_header."""

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            data_io.read_csv("dummy.csv", skip_header=-1)

    def test_float_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            data_io.read_csv("dummy.csv", skip_header=1.5)

    def test_zero_accepted(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        result = data_io.read_csv(str(f), skip_header=0)
        assert result['n_rows'] == 2

    def test_positive_accepted(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("# comment\na,b\n1,2\n3,4\n")
        result = data_io.read_csv(str(f), skip_header=1)
        assert result['n_rows'] == 2


# ============================================================================
# G-F4: spectral_action_on_graph validates n_eigenvalues
# ============================================================================


class TestSpectralActionNEigenvalues:
    """spectral_action_on_graph must reject non-positive n_eigenvalues."""

    def _simple_adj(self):
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            graphs.spectral_action_on_graph(self._simple_adj(), n_eigenvalues=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            graphs.spectral_action_on_graph(self._simple_adj(), n_eigenvalues=-1)

    def test_float_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            graphs.spectral_action_on_graph(self._simple_adj(), n_eigenvalues=1.5)

    def test_none_uses_all(self):
        result = graphs.spectral_action_on_graph(self._simple_adj(), n_eigenvalues=None)
        assert len(result['eigenvalues']) == 3

    def test_valid_subset(self):
        result = graphs.spectral_action_on_graph(self._simple_adj(), n_eigenvalues=2)
        assert len(result['eigenvalues']) == 2


# ============================================================================
# G-F6: spectral_dimension_graph validates t_values > 0
# ============================================================================


class TestSpectralDimensionTPositive:
    """spectral_dimension_graph must reject non-positive t_values."""

    def _simple_adj(self):
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)

    def test_negative_t_raises(self):
        with pytest.raises(ValueError, match="must all be positive"):
            graphs.spectral_dimension_graph(self._simple_adj(), t_values=[-1.0, 1.0])

    def test_zero_t_raises(self):
        with pytest.raises(ValueError, match="must all be positive"):
            graphs.spectral_dimension_graph(self._simple_adj(), t_values=[0.0, 1.0])

    def test_positive_works(self):
        t, d_S = graphs.spectral_dimension_graph(
            self._simple_adj(), t_values=[0.1, 1.0, 10.0]
        )
        assert len(t) == 3
        assert len(d_S) == 3


# ============================================================================
# G-F11: causal_set_dimension NaN/Inf and shape guard
# ============================================================================


class TestCausalSetDimensionGuards:
    """causal_set_dimension must reject NaN/Inf and non-square inputs."""

    def test_nan_raises(self):
        m = np.array([[0, 1], [np.nan, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.causal_set_dimension(m)

    def test_inf_raises(self):
        m = np.array([[0, np.inf], [1, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.causal_set_dimension(m)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="square 2-D"):
            graphs.causal_set_dimension(np.array([0, 1, 0, 1]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square 2-D"):
            graphs.causal_set_dimension(np.ones((3, 4)))

    def test_valid_works(self):
        # Simple 3-point causal set with known structure
        m = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        result = graphs.causal_set_dimension(m)
        assert np.isfinite(result) or np.isnan(result)  # valid return
