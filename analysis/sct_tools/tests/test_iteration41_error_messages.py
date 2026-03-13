"""
Tests for Iteration 41: Error message format consistency.

Verifies that error messages follow the "function_name: message" pattern
(Pattern A) after standardization in fitting.py, graphs.py, entanglement.py.

Covers:
    chi2: 5 error messages now prefixed
    weighted_least_squares: 4 error messages now prefixed
    likelihood_ratio_test: 1 error message now prefixed
    graph_laplacian_spectrum: 4 error messages now prefixed
    negativity: 2 error messages now prefixed
"""

import numpy as np
import pytest

from sct_tools import entanglement, fitting, graphs

# ============================================================================
# chi2: error messages must start with "chi2: "
# ============================================================================


class TestChi2ErrorPrefix:
    """chi2 error messages must follow 'chi2: ...' format."""

    def test_empty_arrays_prefix(self):
        with pytest.raises(ValueError, match="^chi2: "):
            fitting.chi2([], [], [])

    def test_observed_nan_prefix(self):
        with pytest.raises(ValueError, match="^chi2: observed"):
            fitting.chi2([float("nan")], [1.0], [0.1])

    def test_expected_nan_prefix(self):
        with pytest.raises(ValueError, match="^chi2: expected"):
            fitting.chi2([1.0], [float("nan")], [0.1])

    def test_errors_nan_prefix(self):
        with pytest.raises(ValueError, match="^chi2: errors"):
            fitting.chi2([1.0], [1.0], [float("nan")])

    def test_errors_zero_prefix(self):
        with pytest.raises(ValueError, match="^chi2: errors"):
            fitting.chi2([1.0], [1.0], [0.0])


# ============================================================================
# weighted_least_squares: error messages must start with
#   "weighted_least_squares: "
# ============================================================================


class TestWlsErrorPrefix:
    """weighted_least_squares errors must follow 'weighted_least_squares: ...'."""

    def test_x_nan_prefix(self):
        with pytest.raises(ValueError, match="^weighted_least_squares: x"):
            fitting.weighted_least_squares(
                [float("nan"), 2.0], [1.0, 2.0], [0.1, 0.1], degree=1
            )

    def test_y_nan_prefix(self):
        with pytest.raises(ValueError, match="^weighted_least_squares: y "):
            fitting.weighted_least_squares(
                [1.0, 2.0], [float("nan"), 2.0], [0.1, 0.1], degree=1
            )

    def test_yerr_nan_prefix(self):
        with pytest.raises(ValueError, match="^weighted_least_squares: y_err.*NaN"):
            fitting.weighted_least_squares(
                [1.0, 2.0], [1.0, 2.0], [float("nan"), 0.1], degree=1
            )

    def test_yerr_zero_prefix(self):
        with pytest.raises(ValueError, match="^weighted_least_squares: y_err.*zero"):
            fitting.weighted_least_squares(
                [1.0, 2.0], [1.0, 2.0], [0.0, 0.1], degree=1
            )


# ============================================================================
# likelihood_ratio_test: df_diff error must start with
#   "likelihood_ratio_test: "
# ============================================================================


class TestLrtErrorPrefix:
    """likelihood_ratio_test df_diff error must be prefixed."""

    def test_negative_df_diff_prefix(self):
        with pytest.raises(ValueError, match="^likelihood_ratio_test: df_diff"):
            fitting.likelihood_ratio_test(-100.0, -95.0, df_diff=-1)


# ============================================================================
# graph_laplacian_spectrum: errors must start with
#   "graph_laplacian_spectrum: "
# ============================================================================


class TestGraphLaplacianErrorPrefix:
    """graph_laplacian_spectrum errors must be prefixed."""

    def test_1d_prefix(self):
        with pytest.raises(ValueError, match="^graph_laplacian_spectrum: adjacency must be 2D"):
            graphs.graph_laplacian_spectrum(np.array([1, 2, 3]))

    def test_non_square_prefix(self):
        with pytest.raises(
            ValueError, match="^graph_laplacian_spectrum: adjacency must be square"
        ):
            graphs.graph_laplacian_spectrum(np.ones((2, 3)))

    def test_empty_prefix(self):
        with pytest.raises(
            ValueError, match="^graph_laplacian_spectrum: adjacency matrix must be non-empty"
        ):
            graphs.graph_laplacian_spectrum(np.ones((0, 0)))

    def test_nan_prefix(self):
        mat = np.array([[0, 1], [1, float("nan")]])
        with pytest.raises(
            ValueError, match="^graph_laplacian_spectrum: adjacency matrix contains NaN"
        ):
            graphs.graph_laplacian_spectrum(mat)


# ============================================================================
# negativity: errors must start with "negativity: "
# ============================================================================


class TestNegativityErrorPrefix:
    """negativity error messages must be prefixed."""

    def test_non_bipartite_prefix(self):
        rho = np.eye(8) / 8.0
        with pytest.raises(ValueError, match="^negativity: requires bipartite"):
            entanglement.negativity(rho, dims=[2, 2, 2])

    def test_nan_state_prefix(self):
        rho = np.array([[1.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(ValueError, match="^negativity: received NaN"):
            entanglement.negativity(rho, dims=[1, 2])
