"""
Tests for Iteration 33: fitting.py hardening.

Covers:
    FIT-H1: chi2_cov rejects NaN/Inf in observed/expected
    FIT-H3: discovery_significance rejects negative signal
    FIT-M1: fit_lmfit rejects empty arrays
    FIT-H2: anderson_darling_test suppresses FutureWarning
    FIT-M2: discovery_significance zero signal returns Z=0
    FIT-M3: chi2_cov with valid inputs roundtrip
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting

# ============================================================================
# FIT-H1: chi2_cov NaN/Inf in observed/expected
# ============================================================================


class TestChi2CovNanInfGuards:
    """chi2_cov must reject NaN/Inf in observed and expected arrays."""

    def test_nan_in_observed_raises(self):
        cov = np.eye(3)
        with pytest.raises(ValueError, match="observed.*NaN"):
            fitting.chi2_cov([1.0, float('nan'), 3.0], [1.0, 2.0, 3.0], cov)

    def test_inf_in_observed_raises(self):
        cov = np.eye(3)
        with pytest.raises(ValueError, match="observed.*NaN"):
            fitting.chi2_cov([1.0, float('inf'), 3.0], [1.0, 2.0, 3.0], cov)

    def test_nan_in_expected_raises(self):
        cov = np.eye(3)
        with pytest.raises(ValueError, match="expected.*NaN"):
            fitting.chi2_cov([1.0, 2.0, 3.0], [1.0, float('nan'), 3.0], cov)

    def test_inf_in_expected_raises(self):
        cov = np.eye(3)
        with pytest.raises(ValueError, match="expected.*NaN"):
            fitting.chi2_cov([1.0, 2.0, 3.0], [1.0, float('-inf'), 3.0], cov)

    def test_valid_inputs_succeed(self):
        """Baseline: valid inputs should return finite chi2."""
        obs = [1.1, 2.2, 3.3]
        exp = [1.0, 2.0, 3.0]
        cov = np.diag([0.01, 0.01, 0.01])
        chi2_val, ndof, chi2_red, p_val = fitting.chi2_cov(obs, exp, cov)
        assert np.isfinite(chi2_val)
        assert chi2_val > 0
        assert ndof == 3
        assert 0 <= p_val <= 1


# ============================================================================
# FIT-H3: discovery_significance negative signal
# ============================================================================


class TestDiscoverySignificanceNegativeSignal:
    """discovery_significance must reject negative signal counts."""

    def test_negative_signal_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fitting.discovery_significance(-1.0, 10.0)

    def test_negative_small_signal_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fitting.discovery_significance(-0.001, 5.0)

    def test_zero_signal_returns_z_zero(self):
        """s=0 should return Z=0, p=1 (not raise)."""
        result = fitting.discovery_significance(0.0, 10.0)
        assert result['Z'] == 0.0
        assert result['p_value'] == 1.0

    def test_positive_signal_works(self):
        """s=10, b=5 should give finite positive Z."""
        result = fitting.discovery_significance(10.0, 5.0)
        assert result['Z'] > 0
        assert 0 < result['p_value'] < 1


# ============================================================================
# FIT-M1: fit_lmfit empty arrays
# ============================================================================


class TestFitLmfitEmptyArrays:
    """fit_lmfit must reject empty arrays."""

    def test_empty_arrays_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fitting.fit_lmfit(
                lambda x, a=1.0: a * x,
                [], [], [],
                {'a': 1.0},
            )

    def test_single_point_works(self):
        """Single data point should not crash."""
        result = fitting.fit_lmfit(
            lambda x, a=1.0: a * x,
            [1.0], [2.0], [0.1],
            {'a': 1.0},
        )
        assert hasattr(result, 'params')


# ============================================================================
# FIT-H2: anderson_darling_test FutureWarning suppression
# ============================================================================


class TestAndersonDarlingFutureWarning:
    """anderson_darling_test should not emit FutureWarning."""

    def test_no_future_warning(self):
        """Normal call should not propagate FutureWarning to caller."""
        data = np.random.default_rng(42).normal(0, 1, 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fitting.anderson_darling_test(data)
            future_warns = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warns) == 0, (
                f"FutureWarning leaked: {[str(x.message) for x in future_warns]}"
            )
        assert 'statistic' in result
        assert 'p_value' in result

    def test_returns_valid_result(self):
        """Result should have expected keys and reasonable values."""
        data = np.random.default_rng(42).normal(0, 1, 200)
        result = fitting.anderson_darling_test(data, dist='norm')
        assert result['statistic'] >= 0
        assert 0 <= result['p_value'] <= 1


# ============================================================================
# Additional: edge case tests for consistency
# ============================================================================


class TestFittingEdgeCases:
    """Additional edge case coverage."""

    def test_chi2_cov_nan_in_cov_raises(self):
        """NaN in covariance matrix should still raise."""
        cov = np.eye(3)
        cov[1, 1] = float('nan')
        with pytest.raises(ValueError, match="covariance.*NaN"):
            fitting.chi2_cov([1, 2, 3], [1, 2, 3], cov)

    def test_discovery_significance_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            fitting.discovery_significance(float('inf'), 10.0)

    def test_discovery_significance_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            fitting.discovery_significance(float('nan'), 10.0)

    def test_discovery_significance_zero_background(self):
        """s>0, b=0 should give infinite significance."""
        result = fitting.discovery_significance(5.0, 0.0)
        assert result['Z'] == np.inf
        assert result['p_value'] == 0.0

    def test_discovery_significance_negative_background_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fitting.discovery_significance(5.0, -1.0)
