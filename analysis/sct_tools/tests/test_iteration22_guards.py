"""
Tests for Iteration 22 input validation guards in entanglement.py and fitting.py.

Covers:
    - entanglement: E1-E8 (renyi empty eigs, base validation, dims checks, etc.)
    - fitting: F1-F9 (shape mismatch, ndof warning, negative params, etc.)
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting

try:
    import quimb  # noqa: F401
    HAS_QUIMB = True
except ImportError:
    HAS_QUIMB = False


# ============================================================================
# Entanglement guards
# ============================================================================


@pytest.mark.skipif(not HAS_QUIMB, reason="quimb not installed")
class TestEntanglementGuards:
    """Test input validation guards in entanglement.py."""

    def test_renyi_entropy_zero_state(self):
        """E1: renyi_entropy raises on all-zero density matrix."""
        from sct_tools.entanglement import renyi_entropy
        zero_state = np.zeros(4, dtype=complex)
        with pytest.raises(ValueError, match="all eigenvalues are zero"):
            renyi_entropy(zero_state, [2, 2], 0, alpha=2)

    def test_entanglement_entropy_base_zero(self):
        """E2: entanglement_entropy rejects base=0."""
        from sct_tools.entanglement import entanglement_entropy
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        with pytest.raises(ValueError, match="base must be > 0"):
            entanglement_entropy(bell, [2, 2], 0, base=0)

    def test_entanglement_entropy_base_one(self):
        """E2: entanglement_entropy rejects base=1 (division by zero in log)."""
        from sct_tools.entanglement import entanglement_entropy
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        with pytest.raises(ValueError, match="base must be > 0 and != 1"):
            entanglement_entropy(bell, [2, 2], 0, base=1)

    def test_entanglement_entropy_base_negative(self):
        """E2: entanglement_entropy rejects negative base."""
        from sct_tools.entanglement import entanglement_entropy
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        with pytest.raises(ValueError, match="base must be > 0"):
            entanglement_entropy(bell, [2, 2], 0, base=-2)

    def test_fit_cft_entropy_negative_L(self):
        """E3: fit_cft_entropy rejects negative L values."""
        from sct_tools.entanglement import fit_cft_entropy
        with pytest.raises(ValueError, match="L_values must all be positive"):
            fit_cft_entropy([-1, 4, 8], [0.5, 0.8, 1.0])

    def test_fit_cft_entropy_length_mismatch(self):
        """E4: fit_cft_entropy rejects length mismatch."""
        from sct_tools.entanglement import fit_cft_entropy
        with pytest.raises(ValueError, match="must have same length"):
            fit_cft_entropy([4, 8, 16], [0.5, 0.8])

    def test_fit_cft_entropy_too_few_points(self):
        """E4: fit_cft_entropy rejects fewer than 2 data points."""
        from sct_tools.entanglement import fit_cft_entropy
        with pytest.raises(ValueError, match="need at least 2"):
            fit_cft_entropy([4], [0.5])

    def test_negativity_wrong_dims(self):
        """E5: negativity rejects non-bipartite dims."""
        from sct_tools.entanglement import negativity
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0
        with pytest.raises(ValueError, match="bipartite dims"):
            negativity(state, [2, 2, 2])

    def test_log_negativity_wrong_dims(self):
        """E5: log_negativity rejects non-bipartite dims."""
        from sct_tools.entanglement import log_negativity
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0
        with pytest.raises(ValueError, match="bipartite dims"):
            log_negativity(state, [2, 2, 2])

    def test_concurrence_wrong_dims(self):
        """E6: concurrence rejects non-[2,2] dims."""
        from sct_tools.entanglement import concurrence
        state = np.zeros(6, dtype=complex)
        state[0] = 1.0
        with pytest.raises(ValueError, match="dims=\\[2,2\\]"):
            concurrence(state, [2, 3])

    def test_half_chain_entropy_L1(self):
        """E7: half_chain_entropy rejects L < 2."""
        from sct_tools.entanglement import half_chain_entropy
        state = np.array([1.0, 0.0], dtype=complex)
        with pytest.raises(ValueError, match="L >= 2"):
            half_chain_entropy(state, L=1)

    def test_half_chain_entropy_L0(self):
        """E7: half_chain_entropy rejects L = 0."""
        from sct_tools.entanglement import half_chain_entropy
        with pytest.raises(ValueError, match="L >= 2"):
            half_chain_entropy(np.array([1.0], dtype=complex), L=0)

    def test_entanglement_spectrum_clips_negative(self):
        """E8: entanglement_spectrum clips tiny negative eigenvalues to 0."""
        from sct_tools.entanglement import entanglement_spectrum
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        spectrum = entanglement_spectrum(bell, [2, 2], 0)
        # All eigenvalues must be >= 0 (no negatives from numerical noise)
        assert np.all(spectrum >= 0), f"Negative eigenvalue found: {spectrum}"
        # Bell state: eigenvalues should be [0.5, 0.5]
        assert spectrum[0] == pytest.approx(0.5, abs=1e-10)
        assert spectrum[1] == pytest.approx(0.5, abs=1e-10)

    def test_fit_cft_entropy_valid(self):
        """fit_cft_entropy works with valid inputs."""
        from sct_tools.entanglement import fit_cft_entropy
        result = fit_cft_entropy([4, 8, 16, 32], [0.5, 0.8, 1.1, 1.4])
        assert 'central_charge' in result
        assert result['central_charge'] > 0


# ============================================================================
# Fitting guards
# ============================================================================


class TestChi2Guards:
    """Test chi2 input validation guards."""

    def test_shape_mismatch_obs_exp(self):
        """F1: chi2 rejects mismatched observed/expected shapes."""
        with pytest.raises(ValueError, match="array shapes must match"):
            fitting.chi2([1, 2, 3], [1, 2], [0.1, 0.1, 0.1])

    def test_shape_mismatch_obs_err(self):
        """F1: chi2 rejects mismatched observed/errors shapes."""
        with pytest.raises(ValueError, match="array shapes must match"):
            fitting.chi2([1, 2], [1, 2], [0.1])

    def test_negative_n_params(self):
        """F9: chi2 rejects negative n_params."""
        with pytest.raises(ValueError, match="n_params must be non-negative"):
            fitting.chi2([1, 2], [1, 2], [0.1, 0.1], n_params=-1)

    def test_ndof_clamp_warning(self):
        """F2: chi2 warns when n_params >= n_data (ndof clamped to 1)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val, ndof, chi2_red, p_val = fitting.chi2(
                [1, 2], [1.1, 2.1], [0.1, 0.1], n_params=3
            )
            assert ndof == 1  # clamped
            assert len(w) == 1
            assert "n_params (3) >= n_data (2)" in str(w[0].message)

    def test_valid_chi2(self):
        """chi2 works with valid matching arrays."""
        val, ndof, chi2_red, p_val = fitting.chi2(
            [1, 2, 3], [1.1, 2.0, 2.9], [0.1, 0.1, 0.1]
        )
        assert val > 0
        assert ndof == 3
        assert 0 <= p_val <= 1


class TestChi2CovGuards:
    """Test chi2_cov input validation guards."""

    def test_shape_mismatch_obs_exp(self):
        """F1: chi2_cov rejects mismatched observed/expected shapes."""
        with pytest.raises(ValueError, match="observed and expected shapes must match"):
            fitting.chi2_cov([1, 2, 3], [1, 2], np.eye(3))

    def test_cov_shape_mismatch(self):
        """F1: chi2_cov rejects wrong covariance matrix shape."""
        with pytest.raises(ValueError, match="covariance matrix shape"):
            fitting.chi2_cov([1, 2], [1, 2], np.eye(3))

    def test_negative_n_params(self):
        """F9: chi2_cov rejects negative n_params."""
        with pytest.raises(ValueError, match="n_params must be non-negative"):
            fitting.chi2_cov([1, 2], [1, 2], np.eye(2), n_params=-1)

    def test_ndof_clamp_warning(self):
        """F2: chi2_cov warns when n_params >= n_data."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val, ndof, chi2_red, p_val = fitting.chi2_cov(
                [1, 2], [1.1, 2.1], np.eye(2) * 0.01, n_params=5
            )
            assert ndof == 1
            assert any("n_params (5) >= n_data (2)" in str(x.message) for x in w)


class TestModelComparisonGuards:
    """Test model_comparison input validation guards."""

    def test_negative_k1(self):
        """F3: model_comparison rejects negative parameter counts."""
        with pytest.raises(ValueError, match="non-negative parameter counts"):
            fitting.model_comparison(10.0, -1, 8.0, 2, 100)

    def test_negative_k2(self):
        """F3: model_comparison rejects negative k_2."""
        with pytest.raises(ValueError, match="non-negative parameter counts"):
            fitting.model_comparison(10.0, 1, 8.0, -2, 100)

    def test_valid_comparison(self):
        """model_comparison works with valid inputs."""
        result = fitting.model_comparison(10.0, 2, 8.0, 3, 50)
        assert 'AIC_1' in result
        assert 'BIC_1' in result
        assert result['favors'] in ('model_1', 'model_2')


class TestRunMcmcGuards:
    """Test run_mcmc input validation guards."""

    def test_1d_initial_pos(self):
        """F4: run_mcmc rejects 1D initial_pos."""
        with pytest.raises(ValueError, match="must be 2D"):
            fitting.run_mcmc(lambda theta: 0.0, np.array([1.0, 2.0]))


class TestFitLmfitGuards:
    """Test fit_lmfit input validation guards."""

    def test_shape_mismatch(self):
        """F5: fit_lmfit rejects mismatched array shapes."""
        with pytest.raises(ValueError, match="array shapes must match"):
            fitting.fit_lmfit(
                lambda x, a: a * x,
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),      # wrong length
                np.array([0.1, 0.1, 0.1]),
                {'a': 1.0},
            )


class TestWLSGuards:
    """Test weighted_least_squares input validation guards."""

    def test_shape_mismatch(self):
        """F6: weighted_least_squares rejects mismatched array shapes."""
        with pytest.raises(ValueError, match="array shapes must match"):
            fitting.weighted_least_squares(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),       # wrong length
                np.array([0.1, 0.1, 0.1]),
            )


class TestMcmcSummaryGuards:
    """Test mcmc_summary param_names validation."""

    def test_param_names_mismatch(self):
        """F7: mcmc_summary rejects param_names/chain dimension mismatch."""
        from unittest.mock import MagicMock
        mock_sampler = MagicMock()
        # Chain has 2 parameters
        mock_sampler.get_chain.return_value = np.random.randn(100, 2)
        # But we pass 3 param_names
        with pytest.raises(ValueError, match="param_names has 3 entries but chain has 2"):
            fitting.mcmc_summary(mock_sampler, ['a', 'b', 'c'], discard=0, thin=1)


class TestResidualDiagnosticsGuards:
    """Test residual_diagnostics skewness underflow guard."""

    def test_tiny_residuals_skewness(self):
        """F8: skewness should not overflow/underflow for tiny residuals."""
        tiny = np.array([1e-300, 1.1e-300, 0.9e-300, 1.05e-300, 0.95e-300])
        result = fitting.residual_diagnostics(tiny)
        assert np.isfinite(result['skewness'])

    def test_zero_std_skewness(self):
        """F8: skewness should be 0 for constant residuals (std=0)."""
        const = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = fitting.residual_diagnostics(const)
        assert result['skewness'] == 0.0

    def test_normal_residuals(self):
        """residual_diagnostics works on well-behaved residuals."""
        np.random.seed(42)
        res = np.random.normal(0, 1, 100)
        result = fitting.residual_diagnostics(res)
        assert 'shapiro_wilk' in result
        assert 'durbin_watson' in result
        assert np.isfinite(result['skewness'])
        assert abs(result['skewness']) < 1.0  # normal dist skewness ~ 0
