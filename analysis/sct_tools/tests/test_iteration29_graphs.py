"""
Tests for Iteration 29: graphs.py NaN/Inf guards + validation hardening.

Covers:
    G1: graph_laplacian_spectrum NaN/Inf adjacency rejection
    G2: heat_kernel_trace NaN/Inf t_values rejection
    G3: spectral_action_on_graph empty coefficients rejection
    G4: feynman_graph edge tuple format validation
    G5: NaN propagation prevention (adjacency → eigenvalues → all downstream)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import graphs

# ============================================================================
# G1: graph_laplacian_spectrum NaN/Inf guard
# ============================================================================


class TestLaplacianNaNGuard:
    """graph_laplacian_spectrum must reject NaN/Inf adjacency matrices."""

    def test_nan_adjacency_raises(self):
        A = np.array([[0, 1], [1, np.nan]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.graph_laplacian_spectrum(A)

    def test_inf_adjacency_raises(self):
        A = np.array([[0, np.inf], [np.inf, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.graph_laplacian_spectrum(A)

    def test_neginf_adjacency_raises(self):
        A = np.array([[0, -np.inf], [-np.inf, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.graph_laplacian_spectrum(A)

    def test_valid_adjacency_passes(self):
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        eigs = graphs.graph_laplacian_spectrum(A)
        assert len(eigs) == 3
        assert np.all(np.isfinite(eigs))
        # Path graph eigenvalues are known: 0, 1, 3
        assert eigs[0] == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# G2: heat_kernel_trace NaN/Inf t_values guard
# ============================================================================


class TestHeatKernelNaNGuard:
    """heat_kernel_trace must reject NaN/Inf t_values."""

    def test_nan_t_raises(self):
        A = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.heat_kernel_trace(A, [1.0, np.nan])

    def test_inf_t_raises(self):
        A = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.heat_kernel_trace(A, [np.inf])

    def test_valid_t_passes(self):
        A = np.array([[0, 1], [1, 0]])
        result = graphs.heat_kernel_trace(A, [0.1, 1.0])
        assert len(result) == 2
        assert np.all(np.isfinite(result))
        # At t=0+, Tr(e^{-tL}) → N (number of vertices)
        assert result[0] == pytest.approx(2.0, rel=0.1)


# ============================================================================
# G3: spectral_action_on_graph empty coefficients
# ============================================================================


class TestSpectralActionCoefficients:
    """spectral_action_on_graph must reject empty coefficients."""

    def test_empty_coefficients_raises(self):
        A = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="coefficients must be non-empty"):
            graphs.spectral_action_on_graph(A, coefficients=[])

    def test_default_coefficients_works(self):
        A = np.array([[0, 1], [1, 0]])
        result = graphs.spectral_action_on_graph(A)
        assert np.isfinite(result['action'])
        assert len(result['eigenvalues']) == 2

    def test_custom_coefficients_works(self):
        # f(x) = 1 + x, coefficients = [1, 1]
        A = np.array([[0, 1], [1, 0]])
        result = graphs.spectral_action_on_graph(A, coefficients=[1, 1])
        # Eigenvalues of K_2 Laplacian: 0, 2
        # f(0) + f(2) = 1 + 3 = 4
        assert result['action'] == pytest.approx(4.0, rel=1e-10)

    def test_custom_function_works(self):
        A = np.array([[0, 1], [1, 0]])
        result = graphs.spectral_action_on_graph(A, f=lambda x: np.exp(-x))
        # e^0 + e^{-2} = 1 + e^{-2}
        expected = 1.0 + np.exp(-2.0)
        assert result['action'] == pytest.approx(expected, rel=1e-10)


# ============================================================================
# G4: feynman_graph edge validation
# ============================================================================


class TestFeynmanGraphEdgeValidation:
    """feynman_graph must reject invalid edge tuples."""

    def test_2tuple_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            graphs.feynman_graph([1, 2], [(1, 2)])

    def test_4tuple_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            graphs.feynman_graph([1, 2], [(1, 2, 'scalar', 'extra')])

    def test_string_edge_raises(self):
        with pytest.raises(ValueError, match="3-tuple"):
            graphs.feynman_graph([1, 2], ["not_a_tuple"])

    def test_valid_edges_pass(self):
        G = graphs.feynman_graph([1, 2], [(1, 2, 'scalar'), (1, 2, 'fermion')])
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 2

    def test_empty_edges_pass(self):
        G = graphs.feynman_graph([1, 2, 3], [])
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 0

    def test_loop_number_after_validation(self):
        """Validated graph should give correct loop number."""
        # One-loop self-energy: 2 vertices, 2 edges → L = 2 - 2 + 1 = 1
        G = graphs.feynman_graph([1, 2], [(1, 2, 'scalar'), (1, 2, 'scalar')])
        assert graphs.loop_number(G) == 1


# ============================================================================
# G5: NaN propagation prevention (downstream functions)
# ============================================================================


class TestNaNPropagationPrevention:
    """NaN in adjacency must be caught before reaching downstream functions."""

    def test_spectral_dimension_nan_adjacency(self):
        A = np.array([[0, np.nan], [np.nan, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.spectral_dimension_graph(A)

    def test_spectral_action_nan_adjacency(self):
        A = np.array([[0, np.nan], [np.nan, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.spectral_action_on_graph(A)

    def test_zeta_function_nan_adjacency(self):
        A = np.array([[0, np.nan], [np.nan, 0]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            graphs.zeta_function_graph(A, [2.0])
