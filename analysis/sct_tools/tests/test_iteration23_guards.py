"""
Tests for Iteration 23 input validation guards in graphs.py and tensors.py.

Covers:
    graphs.py:
        G1: graph_laplacian_spectrum adjacency validation (2D, square, non-empty)
        G2: heat_kernel_trace t_values > 0 validation
        G3: spectral_dimension_graph dt_frac boundary (reject >= 1)
        G4: causal_set_sprinkle diamond rejection max iterations
        G5: causal_set_dimension N < 2
    tensors.py:
        T1: cartesian_coords dim validation (2-7)
        T2: verify_vacuum seed parameter
        T3: linearized_metric _require_ogrepy guard
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import graphs

try:
    import OGRePy  # noqa: F401
    HAS_OGREPY = True
except ImportError:
    HAS_OGREPY = False


# ============================================================================
# graphs.py guards
# ============================================================================


class TestGraphLaplacianSpectrumGuards:
    """Test adjacency validation in graph_laplacian_spectrum."""

    def test_1d_array_rejected(self):
        """G1: Rejects 1D array."""
        with pytest.raises(ValueError, match="adjacency must be 2D"):
            graphs.graph_laplacian_spectrum(np.array([1, 2, 3]))

    def test_3d_array_rejected(self):
        """G1: Rejects 3D array."""
        with pytest.raises(ValueError, match="adjacency must be 2D"):
            graphs.graph_laplacian_spectrum(np.zeros((2, 2, 2)))

    def test_non_square_rejected(self):
        """G1: Rejects non-square matrix."""
        with pytest.raises(ValueError, match="adjacency must be square"):
            graphs.graph_laplacian_spectrum(np.zeros((2, 3)))

    def test_empty_rejected(self):
        """G1: Rejects empty matrix."""
        with pytest.raises(ValueError, match="non-empty"):
            graphs.graph_laplacian_spectrum(np.zeros((0, 0)))

    def test_valid_adjacency(self):
        """graph_laplacian_spectrum works with valid adjacency."""
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        eigs = graphs.graph_laplacian_spectrum(A)
        assert len(eigs) == 3
        assert eigs[0] == pytest.approx(0.0, abs=1e-12)  # connected graph has 0 eigenvalue


class TestHeatKernelTraceGuards:
    """Test t_values validation in heat_kernel_trace."""

    def test_negative_t_rejected(self):
        """G2: Rejects negative t values."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="t_values must be > 0"):
            graphs.heat_kernel_trace(A, [-1.0, 0.5])

    def test_zero_t_rejected(self):
        """G2: Rejects zero t values."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="t_values must be > 0"):
            graphs.heat_kernel_trace(A, [0.0, 0.5])

    def test_valid_t(self):
        """heat_kernel_trace works with valid positive t."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graphs.heat_kernel_trace(A, [0.1, 1.0])
        assert len(result) == 2
        assert np.all(result > 0)  # trace of e^{-tL} is always positive


class TestSpectralDimensionGraphGuards:
    """Test dt_frac boundary in spectral_dimension_graph."""

    def test_dt_frac_one_rejected(self):
        """G3: dt_frac=1.0 rejected (t - dt = 0)."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="dt_frac must be in"):
            graphs.spectral_dimension_graph(A, t_values=np.array([1.0]), dt_frac=1.0)

    def test_dt_frac_above_one_rejected(self):
        """G3: dt_frac > 1 rejected."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="dt_frac must be in"):
            graphs.spectral_dimension_graph(A, t_values=np.array([1.0]), dt_frac=1.5)

    def test_dt_frac_valid(self):
        """spectral_dimension_graph works with valid dt_frac."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        t_vals, d_s = graphs.spectral_dimension_graph(
            A, t_values=np.array([0.5, 1.0, 2.0]), dt_frac=0.01,
        )
        assert len(t_vals) == 3
        assert len(d_s) == 3


class TestCausalSetDimensionGuards:
    """Test N < 2 guard in causal_set_dimension."""

    def test_single_point_rejected(self):
        """G5: Rejects N=1 causal matrix (division by zero)."""
        C = np.zeros((1, 1), dtype=bool)
        with pytest.raises(ValueError, match="N >= 2"):
            graphs.causal_set_dimension(C)

    def test_empty_matrix_rejected(self):
        """G5: Rejects empty causal matrix."""
        C = np.zeros((0, 0), dtype=bool)
        with pytest.raises(ValueError, match="N >= 2"):
            graphs.causal_set_dimension(C)

    def test_valid_causal_set(self):
        """causal_set_dimension works with valid diamond sprinkling."""
        points, C = graphs.causal_set_sprinkle(20, dim=2, region='diamond', seed=42)
        d = graphs.causal_set_dimension(C)
        assert np.isfinite(d)
        assert 1.0 < d < 4.0  # should estimate ~2 for 2D diamond


class TestCausalSetSprinkleDiamondGuard:
    """Test diamond rejection sampling max iteration guard."""

    def test_diamond_normal_succeeds(self):
        """G4: Diamond sprinkling with reasonable n_points succeeds."""
        points, C = graphs.causal_set_sprinkle(10, dim=2, region='diamond', seed=42)
        assert points.shape == (10, 2)
        assert C.shape == (10, 10)

    def test_diamond_4d_succeeds(self):
        """G4: Diamond sprinkling in 4D succeeds for small n_points."""
        points, C = graphs.causal_set_sprinkle(5, dim=4, region='diamond', seed=42)
        assert points.shape == (5, 4)


# ============================================================================
# tensors.py guards
# ============================================================================


@pytest.mark.skipif(not HAS_OGREPY, reason="OGRePy not installed")
class TestCartesianCoordsGuards:
    """Test dim validation in cartesian_coords."""

    def test_dim_1_rejected(self):
        """T1: dim < 2 rejected."""
        from sct_tools import tensors
        with pytest.raises(ValueError, match="dim must be >= 2"):
            tensors.cartesian_coords(dim=1)

    def test_dim_0_rejected(self):
        """T1: dim=0 rejected."""
        from sct_tools import tensors
        with pytest.raises(ValueError, match="dim must be >= 2"):
            tensors.cartesian_coords(dim=0)

    def test_dim_8_rejected(self):
        """T1: dim > 7 rejected (only 7 named coordinates available)."""
        from sct_tools import tensors
        with pytest.raises(ValueError, match="dim must be <= 7"):
            tensors.cartesian_coords(dim=8)

    def test_dim_4_valid(self):
        """T1: dim=4 works (standard spacetime)."""
        from sct_tools import tensors
        coords, syms = tensors.cartesian_coords(dim=4)
        assert len(syms) == 4


@pytest.mark.skipif(not HAS_OGREPY, reason="OGRePy not installed")
class TestVerifyVacuumSeed:
    """Test seed parameter in verify_vacuum."""

    def test_seed_reproducible(self):
        """T2: Same seed gives same result."""
        from sct_tools import tensors
        metric, _ = tensors.schwarzschild()
        result1 = tensors.verify_vacuum(metric, seed=42)
        result2 = tensors.verify_vacuum(metric, seed=42)
        assert result1 == result2


@pytest.mark.skipif(not HAS_OGREPY, reason="OGRePy not installed")
class TestLinearizedMetricGuard:
    """Test _require_ogrepy guard in linearized_metric."""

    def test_linearized_metric_works(self):
        """T3: linearized_metric works when OGRePy is available."""
        from sct_tools import tensors
        metric, params = tensors.minkowski(dim=4)
        h, epsilon = tensors.linearized_metric(metric)
        assert h.shape == (4, 4)
