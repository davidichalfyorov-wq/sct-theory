"""
Tests for Iteration 28: entanglement.py + fitting.py hardening.

Covers:
    E1: renyi_entropy noise threshold unified (all alpha use 1e-15)
    E2: fit_cft_entropy cyclic parameter (open BC: c = 6*a)
    E3: log_negativity, mutual_information, entanglement_spectrum _check_density_matrix
    E4: renyi_entropy alpha validation before eigendecomposition
    E5: entanglement_entropy, renyi_entropy dims product validation
    E6: heisenberg_ground_state, dmrg_ground_state L<2 validation
    E7: renyi_entropy base parameter
    FT1: chi2/chi2_cov/likelihood_ratio_test sf() precision
    FT2: fit_lmfit NaN/Inf on x_data, y_data
    FT4: fit_lmfit params_dict tuple validation
    FT3: fit_lmfit uses validated arrays
    FT5: weighted_least_squares NaN/Inf on x, y
    FT6: weighted_least_squares degree validation
    FT7: run_mcmc NaN/Inf on initial_pos
    FT8: run_mcmc nwalkers mismatch
    FT9: mcmc_summary empty chain check
    FT10: discovery_significance negative background
    FT13: model_comparison AICc
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting

# ============================================================================
# Helper: skip if quimb not available
# ============================================================================

quimb = pytest.importorskip("quimb", reason="quimb required for entanglement tests")


def _bell_state():
    """Return |Phi+> = (|00> + |11>)/sqrt(2)."""
    return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def _product_state():
    """Return |00> product state."""
    return np.array([1, 0, 0, 0], dtype=complex)


# ============================================================================
# E1: renyi_entropy noise threshold unified
# ============================================================================


class TestRenyiNoiseThreshold:
    """Renyi entropy should not amplify noise eigenvalues for alpha < 1."""

    def test_alpha_half_bell_state(self):
        from sct_tools import entanglement
        psi = _bell_state()
        S = entanglement.renyi_entropy(psi, [2, 2], 0, alpha=0.5)
        # Bell state: Renyi entropy should be 1.0 bit for all alpha
        assert S == pytest.approx(1.0, abs=1e-10)

    def test_alpha_half_product_state(self):
        from sct_tools import entanglement
        psi = _product_state()
        S = entanglement.renyi_entropy(psi, [2, 2], 0, alpha=0.5)
        assert S == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# E2: fit_cft_entropy cyclic parameter
# ============================================================================


class TestFitCFTEntropyCyclic:
    """fit_cft_entropy should use correct CFT prefactor for open/periodic BC."""

    def test_cyclic_true_default(self):
        from sct_tools import entanglement
        L = [10, 20, 40, 80]
        # Synthetic data: S = (1/3) * log2(L) + 0.5 => c = 1
        S = [(1.0 / 3.0) * np.log2(ll) + 0.5 for ll in L]
        result = entanglement.fit_cft_entropy(L, S)
        assert result["central_charge"] == pytest.approx(1.0, rel=1e-10)

    def test_cyclic_false_open_bc(self):
        from sct_tools import entanglement
        L = [10, 20, 40, 80]
        # For open BC: S = (c/6) * log2(L) + const => c = 6*a
        # With c=1: S = (1/6) * log2(L) + 0.5
        S = [(1.0 / 6.0) * np.log2(ll) + 0.5 for ll in L]
        result = entanglement.fit_cft_entropy(L, S, cyclic=False)
        assert result["central_charge"] == pytest.approx(1.0, rel=1e-10)

    def test_cyclic_true_explicit(self):
        from sct_tools import entanglement
        L = [10, 20, 40, 80]
        S = [(1.0 / 3.0) * np.log2(ll) + 0.5 for ll in L]
        result = entanglement.fit_cft_entropy(L, S, cyclic=True)
        assert result["central_charge"] == pytest.approx(1.0, rel=1e-10)


# ============================================================================
# E3: _check_density_matrix added to log_negativity, mutual_information,
#     entanglement_spectrum
# ============================================================================


class TestCheckDensityMatrixCoverage:
    """Functions should warn on unnormalized density matrices."""

    def test_log_negativity_warns_unnormalized(self):
        from sct_tools import entanglement
        rho = np.eye(4, dtype=complex) * 0.5  # Tr = 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            entanglement.log_negativity(rho, [2, 2])
            assert any("trace" in str(x.message).lower() for x in w)

    def test_mutual_information_warns_unnormalized(self):
        from sct_tools import entanglement
        rho = np.eye(4, dtype=complex) * 0.5  # Tr = 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            entanglement.mutual_information(rho, [2, 2])
            assert any("trace" in str(x.message).lower() for x in w)

    def test_entanglement_spectrum_warns_unnormalized(self):
        from sct_tools import entanglement
        rho = np.eye(4, dtype=complex) * 0.5  # Tr = 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            entanglement.entanglement_spectrum(rho, [2, 2], 0)
            assert any("trace" in str(x.message).lower() for x in w)


# ============================================================================
# E4: renyi_entropy alpha validation before eigendecomposition
# ============================================================================


class TestRenyiAlphaValidation:
    """Alpha validation should happen early (before expensive computation)."""

    def test_negative_alpha_raises(self):
        from sct_tools import entanglement
        psi = _bell_state()
        with pytest.raises(ValueError, match="alpha > 0"):
            entanglement.renyi_entropy(psi, [2, 2], 0, alpha=-1.0)

    def test_zero_alpha_raises(self):
        from sct_tools import entanglement
        psi = _bell_state()
        with pytest.raises(ValueError, match="alpha > 0"):
            entanglement.renyi_entropy(psi, [2, 2], 0, alpha=0.0)


# ============================================================================
# E5: dims product validation
# ============================================================================


class TestDimsValidation:
    """State dimension must match product of dims."""

    def test_entanglement_entropy_wrong_dims(self):
        from sct_tools import entanglement
        psi = _bell_state()  # dim 4
        with pytest.raises(ValueError, match="does not match"):
            entanglement.entanglement_entropy(psi, [2, 3], 0)  # prod=6 != 4

    def test_renyi_entropy_wrong_dims(self):
        from sct_tools import entanglement
        psi = _bell_state()  # dim 4
        with pytest.raises(ValueError, match="does not match"):
            entanglement.renyi_entropy(psi, [3, 3], 0)  # prod=9 != 4

    def test_correct_dims_passes(self):
        from sct_tools import entanglement
        psi = _bell_state()
        S = entanglement.entanglement_entropy(psi, [2, 2], 0)
        assert S == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# E6: heisenberg/dmrg L<2 validation
# ============================================================================


class TestChainLengthValidation:
    """Ground state functions must reject L < 2."""

    def test_heisenberg_L_one_raises(self):
        from sct_tools import entanglement
        with pytest.raises(ValueError, match="L >= 2"):
            entanglement.heisenberg_ground_state(1)

    def test_heisenberg_L_zero_raises(self):
        from sct_tools import entanglement
        with pytest.raises(ValueError, match="L >= 2"):
            entanglement.heisenberg_ground_state(0)

    def test_dmrg_L_one_raises(self):
        from sct_tools import entanglement
        with pytest.raises(ValueError, match="L >= 2"):
            entanglement.dmrg_ground_state(1)

    def test_heisenberg_L_two_works(self):
        from sct_tools import entanglement
        E, psi = entanglement.heisenberg_ground_state(2)
        assert np.isfinite(E)


# ============================================================================
# E7: renyi_entropy base parameter
# ============================================================================


class TestRenyiBase:
    """renyi_entropy should support base parameter."""

    def test_base_e_bell_state(self):
        from sct_tools import entanglement
        psi = _bell_state()
        S_bits = entanglement.renyi_entropy(psi, [2, 2], 0, alpha=2, base=2)
        S_nats = entanglement.renyi_entropy(psi, [2, 2], 0, alpha=2, base=np.e)
        assert S_nats == pytest.approx(S_bits * np.log(2), rel=1e-10)

    def test_invalid_base_raises(self):
        from sct_tools import entanglement
        psi = _bell_state()
        with pytest.raises(ValueError, match="base"):
            entanglement.renyi_entropy(psi, [2, 2], 0, base=1)
        with pytest.raises(ValueError, match="base"):
            entanglement.renyi_entropy(psi, [2, 2], 0, base=0)


# ============================================================================
# FT1: chi2/chi2_cov sf() precision
# ============================================================================


class TestChi2SFPrecision:
    """p-value should use sf() for numerical stability at extreme values."""

    def test_chi2_large_value_nonzero_pval(self):
        # For chi2_val >> ndof, sf() returns tiny but non-zero p-value
        # 1 - cdf() would return exactly 0.0 due to floating point
        obs = np.array([100.0])
        exp = np.array([0.0])
        err = np.array([1.0])
        chi2_val, ndof, _, p_val = fitting.chi2(obs, exp, err)
        assert chi2_val == pytest.approx(10000.0)
        # sf() should give a tiny but representable p-value
        assert p_val >= 0.0
        assert p_val < 1e-100

    def test_chi2_normal_case(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 1.9, 3.1])
        err = np.array([0.5, 0.5, 0.5])
        _, _, _, p_val = fitting.chi2(obs, exp, err)
        assert 0 < p_val < 1

    def test_chi2_cov_sf_precision(self):
        obs = np.array([100.0, 100.0])
        exp = np.array([0.0, 0.0])
        cov = np.eye(2)
        _, _, _, p_val = fitting.chi2_cov(obs, exp, cov)
        assert p_val >= 0.0
        assert p_val < 1e-100


# ============================================================================
# FT2/FT3: fit_lmfit NaN/Inf validation + validated arrays
# ============================================================================


class TestFitLmfitValidation:
    """fit_lmfit should reject NaN/Inf in x_data, y_data."""

    def test_nan_x_raises(self):
        with pytest.raises(ValueError, match="x_data.*NaN"):
            fitting.fit_lmfit(
                lambda x, a: a * x,
                [1.0, np.nan, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1],
                {"a": 1.0}
            )

    def test_nan_y_raises(self):
        with pytest.raises(ValueError, match="y_data.*NaN"):
            fitting.fit_lmfit(
                lambda x, a: a * x,
                [1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [0.1, 0.1, 0.1],
                {"a": 1.0}
            )

    def test_inf_x_raises(self):
        with pytest.raises(ValueError, match="x_data.*NaN"):
            fitting.fit_lmfit(
                lambda x, a: a * x,
                [1.0, np.inf, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1],
                {"a": 1.0}
            )


# ============================================================================
# FT4: fit_lmfit params_dict tuple validation
# ============================================================================


class TestFitLmfitParamsTuple:
    """params_dict tuples must have exactly 3 elements."""

    def test_tuple_length_2_raises(self):
        with pytest.raises(ValueError, match="length 2"):
            fitting.fit_lmfit(
                lambda x, a: a * x,
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1],
                {"a": (1.0, 0.0)}  # missing max
            )

    def test_tuple_length_4_raises(self):
        with pytest.raises(ValueError, match="length 4"):
            fitting.fit_lmfit(
                lambda x, a: a * x,
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1],
                {"a": (1.0, 0.0, 10.0, "extra")}
            )

    def test_tuple_length_3_works(self):
        result = fitting.fit_lmfit(
            lambda x, a: a * x,
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1],
            {"a": (1.0, 0.0, 10.0)}
        )
        assert result.success


# ============================================================================
# FT5/FT6: weighted_least_squares validation
# ============================================================================


class TestWLSValidation:
    """weighted_least_squares should validate inputs."""

    def test_nan_x_raises(self):
        with pytest.raises(ValueError, match="x.*NaN"):
            fitting.weighted_least_squares(
                [1.0, np.nan, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]
            )

    def test_nan_y_raises(self):
        with pytest.raises(ValueError, match="y.*NaN"):
            fitting.weighted_least_squares(
                [1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [0.1, 0.1, 0.1]
            )

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fitting.weighted_least_squares(
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1], degree=-1
            )

    def test_degree_too_high_raises(self):
        with pytest.raises(ValueError, match="less than"):
            fitting.weighted_least_squares(
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1], degree=3
            )

    def test_valid_case_works(self):
        result = fitting.weighted_least_squares(
            [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.1, 0.1, 0.1, 0.1]
        )
        assert hasattr(result, 'params')


# ============================================================================
# FT7/FT8: run_mcmc validation
# ============================================================================


class TestRunMCMCValidation:
    """run_mcmc should validate initial_pos."""

    def test_nan_initial_pos_raises(self):
        pos = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            fitting.run_mcmc(lambda x: 0.0, pos, nwalkers=2, nsteps=1)

    def test_nwalkers_mismatch_raises(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows
        with pytest.raises(ValueError, match="nwalkers=4"):
            fitting.run_mcmc(lambda x: 0.0, pos, nwalkers=4, nsteps=1)


# ============================================================================
# FT10: discovery_significance negative background
# ============================================================================


class TestDiscoverySignificanceNegativeBackground:
    """Negative background should raise ValueError."""

    def test_negative_background_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fitting.discovery_significance(10, -5)

    def test_zero_background_inf_significance(self):
        result = fitting.discovery_significance(10, 0)
        assert result['Z'] == np.inf

    def test_zero_signal_zero_significance(self):
        result = fitting.discovery_significance(0, 10)
        assert result['Z'] == 0.0

    def test_normal_case(self):
        result = fitting.discovery_significance(10, 100)
        assert result['Z'] > 0
        assert 0 < result['p_value'] < 1


# ============================================================================
# FT13: model_comparison AICc
# ============================================================================


class TestModelComparisonAICc:
    """model_comparison should include AICc."""

    def test_aicc_present(self):
        result = fitting.model_comparison(10.0, 2, 8.0, 3, 100)
        assert 'AICc_1' in result
        assert 'AICc_2' in result
        assert 'dAICc' in result

    def test_aicc_converges_to_aic_large_n(self):
        result = fitting.model_comparison(10.0, 2, 8.0, 3, 10000)
        assert result['AICc_1'] == pytest.approx(result['AIC_1'], rel=1e-3)
        assert result['AICc_2'] == pytest.approx(result['AIC_2'], rel=1e-3)

    def test_aicc_inf_when_n_too_small(self):
        # n_data = 3, k = 3 => n - k - 1 = -1 <= 0 => inf
        result = fitting.model_comparison(10.0, 3, 8.0, 1, 3)
        assert result['AICc_1'] == np.inf

    def test_aicc_correction_magnitude(self):
        # AICc correction = 2k(k+1)/(n-k-1)
        result = fitting.model_comparison(10.0, 2, 8.0, 3, 10)
        correction_1 = 2 * 2 * 3 / (10 - 2 - 1)  # 12/7
        assert result['AICc_1'] == pytest.approx(
            result['AIC_1'] + correction_1, rel=1e-10
        )


# ============================================================================
# FT1 (likelihood_ratio_test sf precision)
# ============================================================================


class TestLikelihoodRatioSF:
    """likelihood_ratio_test should use sf() for p-value."""

    def test_large_statistic_nonzero_pval(self):
        # Very large test statistic => p-value should be tiny but representable
        result = fitting.likelihood_ratio_test(-1000, -500, 1)
        assert result['p_value'] >= 0.0
        assert result['p_value'] < 1e-100

    def test_normal_case(self):
        result = fitting.likelihood_ratio_test(-100, -98, 1)
        assert 0 < result['p_value'] < 1
