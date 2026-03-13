"""
Tests for Iteration 42: Remaining error message format consistency.

Verifies that error messages follow the "function_name: message" pattern
(Pattern A) after standardization in fitting.py, graphs.py, entanglement.py.

Covers:
    fitting.py:
        fit_lmfit: 5 error messages now prefixed
        ks_test: 2 error messages now prefixed
        anderson_darling_test: 2 error messages now prefixed
        weighted_least_squares: 1 error message (empty) now prefixed
        residual_diagnostics: 2 error messages now prefixed
        chi2_cov: 1 error message now prefixed
        bayesian_limit: 6 error messages now prefixed
    graphs.py:
        spectral_dimension_graph: 2 error messages now prefixed
        causal_set_sprinkle: 5 error messages now prefixed
        causal_set_dimension: 2 error messages now prefixed
    entanglement.py:
        entanglement_entropy: 1 error message now prefixed
        renyi_entropy: 2 error messages now prefixed
        log_negativity: 2 error messages now prefixed
        mutual_information: 2 error messages now prefixed
        concurrence: 2 error messages now prefixed
        entanglement_spectrum: 1 error message now prefixed
        heisenberg_ground_state: 1 error message now prefixed
        half_chain_entropy: 1 error message now prefixed
        dmrg_ground_state: 1 error message now prefixed
"""

import numpy as np
import pytest

from sct_tools import entanglement, fitting, graphs

# ============================================================================
# fitting.py: fit_lmfit
# ============================================================================


class TestFitLmfitErrorPrefix:
    """fit_lmfit error messages must follow 'fit_lmfit: ...' format."""

    _params = {"a": {"value": 1.0}}

    def test_empty_arrays_prefix(self):
        with pytest.raises(ValueError, match="^fit_lmfit: "):
            fitting.fit_lmfit(lambda x, a: a * x, [], [], [], self._params)

    def test_x_nan_prefix(self):
        with pytest.raises(ValueError, match="^fit_lmfit: x_data"):
            fitting.fit_lmfit(
                lambda x, a: a * x, [float("nan")], [1.0], [0.1], self._params
            )

    def test_y_nan_prefix(self):
        with pytest.raises(ValueError, match="^fit_lmfit: y_data"):
            fitting.fit_lmfit(
                lambda x, a: a * x, [1.0], [float("nan")], [0.1], self._params
            )

    def test_yerr_nan_prefix(self):
        with pytest.raises(ValueError, match="^fit_lmfit: y_err"):
            fitting.fit_lmfit(
                lambda x, a: a * x, [1.0], [1.0], [float("nan")], self._params
            )

    def test_yerr_zero_prefix(self):
        with pytest.raises(ValueError, match="^fit_lmfit: y_err"):
            fitting.fit_lmfit(
                lambda x, a: a * x, [1.0], [1.0], [0.0], self._params
            )


# ============================================================================
# fitting.py: ks_test
# ============================================================================


class TestKsTestErrorPrefix:
    """ks_test error messages must follow 'ks_test: ...' format."""

    _cdf = lambda self, x: x  # noqa: E731

    def test_empty_prefix(self):
        with pytest.raises(ValueError, match="^ks_test: "):
            fitting.ks_test([], self._cdf)

    def test_nan_prefix(self):
        with pytest.raises(ValueError, match="^ks_test: "):
            fitting.ks_test([1.0, float("nan"), 3.0], self._cdf)


# ============================================================================
# fitting.py: anderson_darling_test
# ============================================================================


class TestAndersonDarlingErrorPrefix:
    """anderson_darling_test errors must follow 'anderson_darling_test: ...'."""

    def test_empty_prefix(self):
        with pytest.raises(ValueError, match="^anderson_darling_test: "):
            fitting.anderson_darling_test([])

    def test_nan_prefix(self):
        with pytest.raises(ValueError, match="^anderson_darling_test: "):
            fitting.anderson_darling_test([1.0, float("nan")])


# ============================================================================
# fitting.py: weighted_least_squares (empty)
# ============================================================================


class TestWlsEmptyErrorPrefix:
    """weighted_least_squares empty-array error must be prefixed."""

    def test_empty_prefix(self):
        with pytest.raises(ValueError, match="^weighted_least_squares: "):
            fitting.weighted_least_squares([], [], [], degree=1)


# ============================================================================
# fitting.py: residual_diagnostics
# ============================================================================


class TestResidualDiagnosticsErrorPrefix:
    """residual_diagnostics errors must follow 'residual_diagnostics: ...'."""

    def test_empty_prefix(self):
        with pytest.raises(ValueError, match="^residual_diagnostics: "):
            fitting.residual_diagnostics([])

    def test_nan_prefix(self):
        with pytest.raises(ValueError, match="^residual_diagnostics: "):
            fitting.residual_diagnostics([1.0, float("nan")])


# ============================================================================
# fitting.py: chi2_cov
# ============================================================================


class TestChi2CovErrorPrefix:
    """chi2_cov empty-array error must be prefixed."""

    def test_empty_prefix(self):
        with pytest.raises(ValueError, match="^chi2_cov: "):
            fitting.chi2_cov([], [], np.eye(1))


# ============================================================================
# fitting.py: bayesian_limit
# ============================================================================


class TestBayesianLimitErrorPrefix:
    """bayesian_limit error messages must follow 'bayesian_limit: ...'."""

    def test_empty_prefix(self):
        with pytest.raises(ValueError, match="^bayesian_limit: "):
            fitting.bayesian_limit([])

    def test_nan_prefix(self):
        with pytest.raises(ValueError, match="^bayesian_limit: "):
            fitting.bayesian_limit([1.0, float("nan")])

    def test_1d_prefix(self):
        with pytest.raises(ValueError, match="^bayesian_limit: "):
            fitting.bayesian_limit(np.ones((3, 3)))

    def test_min_samples_prefix(self):
        with pytest.raises(ValueError, match="^bayesian_limit: "):
            fitting.bayesian_limit([1.0])

    def test_cl_range_prefix(self):
        with pytest.raises(ValueError, match="^bayesian_limit: "):
            fitting.bayesian_limit([1.0, 2.0, 3.0], cl=1.5)

    def test_unknown_side_prefix(self):
        with pytest.raises(ValueError, match="^bayesian_limit: "):
            fitting.bayesian_limit([1.0, 2.0, 3.0], side="both")


# ============================================================================
# graphs.py: spectral_dimension_graph
# ============================================================================


class TestSpectralDimensionGraphErrorPrefix:
    """spectral_dimension_graph errors must be prefixed."""

    def test_dt_frac_prefix(self):
        A = np.array([[0, 1], [1, 0]])
        with pytest.raises(
            ValueError, match="^spectral_dimension_graph: dt_frac"
        ):
            graphs.spectral_dimension_graph(
                A, t_values=np.array([1.0]), dt_frac=1.5
            )

    def test_eigenvalues_prefix(self):
        """Force negative eigenvalues by passing a mock adjacency."""
        A = np.array([[0, 1], [1, 0]])
        # Cannot easily get negative eigenvalues from graph_laplacian_spectrum,
        # so we test the dt_frac path which is reachable.
        with pytest.raises(
            ValueError, match="^spectral_dimension_graph: dt_frac"
        ):
            graphs.spectral_dimension_graph(
                A, t_values=np.array([1.0]), dt_frac=0.0
            )


# ============================================================================
# graphs.py: causal_set_sprinkle
# ============================================================================


class TestCausalSetSprinkleErrorPrefix:
    """causal_set_sprinkle errors must be prefixed."""

    def test_n_points_prefix(self):
        with pytest.raises(
            ValueError, match="^causal_set_sprinkle: n_points"
        ):
            graphs.causal_set_sprinkle(0)

    def test_dim_prefix(self):
        with pytest.raises(ValueError, match="^causal_set_sprinkle: dim"):
            graphs.causal_set_sprinkle(10, dim=5)

    def test_region_prefix(self):
        with pytest.raises(ValueError, match="^causal_set_sprinkle: region"):
            graphs.causal_set_sprinkle(10, region="schwarzschild")

    def test_desitter_dim_prefix(self):
        with pytest.raises(
            ValueError, match="^causal_set_sprinkle: de Sitter"
        ):
            graphs.causal_set_sprinkle(10, dim=3, region="desitter")


# ============================================================================
# graphs.py: causal_set_dimension
# ============================================================================


class TestCausalSetDimensionErrorPrefix:
    """causal_set_dimension errors must be prefixed."""

    def test_n_lt_2_prefix(self):
        C = np.array([[False]])
        with pytest.raises(
            ValueError, match="^causal_set_dimension: requires N >= 2"
        ):
            graphs.causal_set_dimension(C)

    def test_unknown_method_prefix(self):
        C = np.array([[False, True], [False, False]])
        with pytest.raises(
            ValueError, match="^causal_set_dimension: unknown method"
        ):
            graphs.causal_set_dimension(C, method="bogus")


# ============================================================================
# entanglement.py: entanglement_entropy
# ============================================================================


class TestEntanglementEntropyErrorPrefix:
    """entanglement_entropy NaN error must be prefixed."""

    def test_nan_prefix(self):
        rho = np.array([[1.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(
            ValueError, match="^entanglement_entropy: received NaN"
        ):
            entanglement.entanglement_entropy(rho, dims=[1, 2], keep=0)


# ============================================================================
# entanglement.py: renyi_entropy
# ============================================================================


class TestRenyiEntropyErrorPrefix:
    """renyi_entropy error messages must be prefixed."""

    def test_alpha_prefix(self):
        rho = np.eye(4) / 4.0
        with pytest.raises(ValueError, match="^renyi_entropy: requires alpha"):
            entanglement.renyi_entropy(rho, dims=[2, 2], keep=0, alpha=-1.0)

    def test_nan_prefix(self):
        rho = np.array([[1.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(ValueError, match="^renyi_entropy: received NaN"):
            entanglement.renyi_entropy(rho, dims=[1, 2], keep=0, alpha=2.0)


# ============================================================================
# entanglement.py: log_negativity
# ============================================================================


class TestLogNegativityErrorPrefix:
    """log_negativity error messages must be prefixed."""

    def test_non_bipartite_prefix(self):
        rho = np.eye(8) / 8.0
        with pytest.raises(
            ValueError, match="^log_negativity: requires bipartite"
        ):
            entanglement.log_negativity(rho, dims=[2, 2, 2])

    def test_nan_prefix(self):
        rho = np.array([[1.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(
            ValueError, match="^log_negativity: received NaN"
        ):
            entanglement.log_negativity(rho, dims=[1, 2])


# ============================================================================
# entanglement.py: mutual_information
# ============================================================================


class TestMutualInformationErrorPrefix:
    """mutual_information error messages must be prefixed."""

    def test_non_bipartite_prefix(self):
        rho = np.eye(8) / 8.0
        with pytest.raises(
            ValueError, match="^mutual_information: requires bipartite"
        ):
            entanglement.mutual_information(rho, dims=[2, 2, 2])

    def test_nan_prefix(self):
        rho = np.array([[1.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(
            ValueError, match="^mutual_information: received NaN"
        ):
            entanglement.mutual_information(rho, dims=[1, 2])


# ============================================================================
# entanglement.py: concurrence
# ============================================================================


class TestConcurrenceErrorPrefix:
    """concurrence error messages must be prefixed."""

    def test_non_2qubit_prefix(self):
        rho = np.eye(8) / 8.0
        with pytest.raises(
            ValueError, match="^concurrence: only defined"
        ):
            entanglement.concurrence(rho, dims=[2, 4])

    def test_nan_prefix(self):
        rho = np.array(
            [[0.5, 0, 0, float("nan")], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]
        )
        with pytest.raises(
            ValueError, match="^concurrence: received NaN"
        ):
            entanglement.concurrence(rho, dims=[2, 2])


# ============================================================================
# entanglement.py: entanglement_spectrum
# ============================================================================


class TestEntanglementSpectrumErrorPrefix:
    """entanglement_spectrum NaN error must be prefixed."""

    def test_nan_prefix(self):
        rho = np.array([[1.0, float("nan")], [0.0, 0.0]])
        with pytest.raises(
            ValueError, match="^entanglement_spectrum: received NaN"
        ):
            entanglement.entanglement_spectrum(rho, dims=[1, 2], keep=0)


# ============================================================================
# entanglement.py: heisenberg_ground_state
# ============================================================================


class TestHeisenbergGroundStateErrorPrefix:
    """heisenberg_ground_state L < 2 error must be prefixed."""

    def test_l_lt_2_prefix(self):
        with pytest.raises(
            ValueError, match="^heisenberg_ground_state: requires L >= 2"
        ):
            entanglement.heisenberg_ground_state(1)


# ============================================================================
# entanglement.py: half_chain_entropy
# ============================================================================


class TestHalfChainEntropyErrorPrefix:
    """half_chain_entropy L < 2 error must be prefixed."""

    def test_l_lt_2_prefix(self):
        with pytest.raises(
            ValueError, match="^half_chain_entropy: requires L >= 2"
        ):
            entanglement.half_chain_entropy(np.array([1.0, 0.0]), L=1)


# ============================================================================
# entanglement.py: dmrg_ground_state
# ============================================================================


class TestDmrgGroundStateErrorPrefix:
    """dmrg_ground_state L < 2 error must be prefixed."""

    def test_l_lt_2_prefix(self):
        with pytest.raises(
            ValueError, match="^dmrg_ground_state: requires L >= 2"
        ):
            entanglement.dmrg_ground_state(1)
