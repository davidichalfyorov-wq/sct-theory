"""
Audit Round 6 — NaN / Inf / empty-input robustness tests.

Validates that every public function in form_factors and fitting raises
ValueError (not silent NaN propagation) when given non-finite or empty inputs.

Created: 2026-03-09
"""

import numpy as np
import pytest

from sct_tools.fitting import (
    chi2,
    chi2_cov,
    discovery_significance,
    ks_test,
    likelihood_ratio_test,
    model_comparison,
    residual_diagnostics,
    weighted_least_squares,
)
from sct_tools.form_factors import (
    dhC_dirac_dx,
    dhC_scalar_dx,
    dhR_dirac_dx,
    dhR_scalar_dx,
    dphi_dx,
    dphi_dx_fast,
    hC_dirac_fast,
    hC_scalar_fast,
    hR_dirac_fast,
    hR_scalar_fast,
    phi_fast,
    phi_vec,
)

# ============================================================================
# FORM FACTORS — NaN / Inf rejection
# ============================================================================


class TestPhiFastNaN:
    """phi_fast must reject NaN and Inf."""

    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            phi_fast(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            phi_fast(float("inf"))

    def test_neg_inf(self):
        with pytest.raises(ValueError, match="finite"):
            phi_fast(float("-inf"))


class TestPhiVecNaN:
    """phi_vec must reject arrays containing NaN."""

    def test_single_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            phi_vec([float("nan")])

    def test_nan_among_valid(self):
        with pytest.raises(ValueError, match="NaN"):
            phi_vec([0.5, float("nan"), 1.0])

    def test_all_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            phi_vec([float("nan"), float("nan")])

    def test_valid_still_works(self):
        """Sanity: valid inputs must still work after guard addition."""
        result = phi_vec([0.0, 0.5, 1.0, 5.0])
        assert len(result) == 4
        assert all(np.isfinite(result))


class TestHCScalarFastNaN:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            hC_scalar_fast(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            hC_scalar_fast(float("inf"))


class TestHRScalarFastNaN:
    def test_nan_x(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar_fast(float("nan"), xi=0.0)

    def test_inf_x(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar_fast(float("inf"), xi=0.0)

    def test_nan_xi(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar_fast(1.0, xi=float("nan"))

    def test_inf_xi(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar_fast(1.0, xi=float("inf"))


class TestHCDiracFastNaN:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            hC_dirac_fast(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            hC_dirac_fast(float("inf"))


class TestHRDiracFastNaN:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            hR_dirac_fast(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            hR_dirac_fast(float("inf"))


# ============================================================================
# DERIVATIVE FUNCTIONS — NaN / Inf / x<0 rejection
# ============================================================================


class TestDphiDxNaN:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            dphi_dx(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            dphi_dx(float("inf"))

    def test_negative_x(self):
        with pytest.raises(ValueError, match="x >= 0"):
            dphi_dx(-1.0)

    def test_zero_works(self):
        result = dphi_dx(0.0)
        assert np.isfinite(result)


class TestDphiDxFastNaN:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            dphi_dx_fast(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            dphi_dx_fast(float("inf"))


class TestDhCScalarDxGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            dhC_scalar_dx(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            dhC_scalar_dx(float("inf"))

    def test_negative_x(self):
        with pytest.raises(ValueError, match="x >= 0"):
            dhC_scalar_dx(-1.0)

    def test_zero_value(self):
        """dhC_scalar_dx(0) = Taylor[1] = -1/1680."""
        result = dhC_scalar_dx(0.0)
        assert result == pytest.approx(-1.0 / 1680.0, rel=1e-12)


class TestDhCDiracDxGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            dhC_dirac_dx(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            dhC_dirac_dx(float("inf"))

    def test_negative_x(self):
        with pytest.raises(ValueError, match="x >= 0"):
            dhC_dirac_dx(-1.0)

    def test_zero_value(self):
        """dhC_dirac_dx(0) = Taylor[1] = 1/168."""
        result = dhC_dirac_dx(0.0)
        assert result == pytest.approx(1.0 / 168.0, rel=1e-12)


class TestDhRDiracDxGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            dhR_dirac_dx(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            dhR_dirac_dx(float("inf"))

    def test_negative_x(self):
        with pytest.raises(ValueError, match="x >= 0"):
            dhR_dirac_dx(-1.0)

    def test_zero_value(self):
        """dhR_dirac_dx(0) = Taylor[1] = 1/2520."""
        result = dhR_dirac_dx(0.0)
        assert result == pytest.approx(1.0 / 2520.0, rel=1e-12)


class TestDhRScalarDxGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            dhR_scalar_dx(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            dhR_scalar_dx(float("inf"))

    def test_negative_x(self):
        """This guard existed before Round 6, verify it still works."""
        with pytest.raises(ValueError, match="x >= 0"):
            dhR_scalar_dx(-1.0)

    def test_nan_xi(self):
        """xi=NaN must also be caught (via hR_scalar_fast guard)."""
        with pytest.raises(ValueError, match="finite"):
            dhR_scalar_dx(3.0, xi=float("nan"))

    def test_zero_works(self):
        result = dhR_scalar_dx(0.0, xi=0.0)
        assert np.isfinite(result)


# ============================================================================
# FITTING — NaN / Inf / empty-input rejection
# ============================================================================

class TestChi2Guards:
    """chi2 must reject NaN errors, empty arrays, etc."""

    def test_nan_in_errors(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            chi2([1.0, 2.0], [1.1, 2.1], [0.1, float("nan")])

    def test_inf_in_errors(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            chi2([1.0, 2.0], [1.1, 2.1], [0.1, float("inf")])

    def test_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            chi2([], [], [])

    def test_zero_error_still_caught(self):
        """Existing guard: zero errors must still raise."""
        with pytest.raises(ValueError, match="zero or negative"):
            chi2([1.0], [1.1], [0.0])

    def test_negative_error_still_caught(self):
        with pytest.raises(ValueError, match="zero or negative"):
            chi2([1.0], [1.1], [-0.1])

    def test_valid_returns_finite(self):
        val, ndof, red, pval = chi2([1.0, 2.0], [1.1, 1.9], [0.1, 0.2])
        assert np.isfinite(val)
        assert np.isfinite(pval)


class TestChi2CovGuards:
    def test_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            chi2_cov([], [], np.array([]).reshape(0, 0))

    def test_nan_in_cov(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            chi2_cov([1.0], [1.1], [[float("nan")]])

    def test_inf_in_cov(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            chi2_cov([1.0], [1.1], [[float("inf")]])

    def test_valid_returns_finite(self):
        val, ndof, red, pval = chi2_cov([1.0, 2.0], [1.1, 1.9],
                                         [[0.01, 0.0], [0.0, 0.04]])
        assert np.isfinite(val)
        assert np.isfinite(pval)


class TestModelComparisonGuards:
    def test_n_data_zero(self):
        with pytest.raises(ValueError, match="n_data >= 2"):
            model_comparison(10.0, 2, 8.0, 3, n_data=0)

    def test_n_data_negative(self):
        with pytest.raises(ValueError, match="n_data >= 2"):
            model_comparison(10.0, 2, 8.0, 3, n_data=-5)

    def test_valid_returns_dict(self):
        result = model_comparison(10.0, 2, 8.0, 3, n_data=50)
        # AIC = chi2 + 2*k
        assert result['AIC_1'] == pytest.approx(10.0 + 2 * 2, rel=1e-12)
        assert result['AIC_2'] == pytest.approx(8.0 + 2 * 3, rel=1e-12)
        assert result['dAIC'] == pytest.approx(14.0 - 14.0, abs=1e-12)
        # BIC = chi2 + k*ln(n)
        import math
        ln50 = math.log(50)
        assert result['BIC_1'] == pytest.approx(10.0 + 2 * ln50, rel=1e-12)
        assert result['BIC_2'] == pytest.approx(8.0 + 3 * ln50, rel=1e-12)
        assert 'favors' in result


class TestLikelihoodRatioTestGuards:
    def test_nan_null(self):
        with pytest.raises(ValueError, match="finite"):
            likelihood_ratio_test(float("nan"), -100.0, 1)

    def test_nan_alt(self):
        with pytest.raises(ValueError, match="finite"):
            likelihood_ratio_test(-100.0, float("nan"), 1)

    def test_inf_null(self):
        with pytest.raises(ValueError, match="finite"):
            likelihood_ratio_test(float("-inf"), -100.0, 1)

    def test_df_zero(self):
        with pytest.raises(ValueError, match="df_diff must be positive"):
            likelihood_ratio_test(-110.0, -100.0, 0)

    def test_df_negative(self):
        with pytest.raises(ValueError, match="df_diff must be positive"):
            likelihood_ratio_test(-110.0, -100.0, -1)

    def test_valid_returns_all_keys(self):
        result = likelihood_ratio_test(-110.0, -100.0, 1)
        # stat = -2*(logL_null - logL_alt) = -2*(-110 - (-100)) = 20
        assert result['statistic'] == pytest.approx(20.0, rel=1e-12)
        assert result['df'] == 1
        assert result['reject_null_5pct'] == True  # 20 >> chi2(1) 5% = 3.841  # noqa: E712
        assert 0.0 < result['p_value'] < 0.001  # highly significant


class TestWeightedLeastSquaresGuards:
    def test_nan_in_yerr(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            weighted_least_squares([1, 2, 3], [1, 2, 3],
                                   [0.1, float("nan"), 0.1])

    def test_inf_in_yerr(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            weighted_least_squares([1, 2, 3], [1, 2, 3],
                                   [0.1, float("inf"), 0.1])

    def test_zero_yerr(self):
        with pytest.raises(ValueError, match="zero or negative"):
            weighted_least_squares([1, 2, 3], [1, 2, 3],
                                   [0.1, 0.0, 0.1])

    def test_negative_yerr(self):
        with pytest.raises(ValueError, match="zero or negative"):
            weighted_least_squares([1, 2, 3], [1, 2, 3],
                                   [0.1, -0.5, 0.1])

    def test_empty_yerr(self):
        with pytest.raises(ValueError, match="empty"):
            weighted_least_squares([], [], [])

    def test_valid_returns_result(self):
        result = weighted_least_squares([1, 2, 3, 4, 5],
                                         [2.1, 3.9, 6.2, 7.8, 10.1],
                                         [0.1, 0.1, 0.1, 0.1, 0.1])
        assert hasattr(result, 'params')


class TestDiscoverySignificanceGuards:
    def test_nan_signal(self):
        with pytest.raises(ValueError, match="finite"):
            discovery_significance(float("nan"), 10.0)

    def test_nan_background(self):
        with pytest.raises(ValueError, match="finite"):
            discovery_significance(5.0, float("nan"))

    def test_inf_signal(self):
        with pytest.raises(ValueError, match="finite"):
            discovery_significance(float("inf"), 10.0)

    def test_zero_signal_returns_zero(self):
        """s <= 0 should return Z=0, not raise."""
        result = discovery_significance(0.0, 10.0)
        assert result['Z'] == 0.0

    def test_zero_background_returns_inf(self):
        """s > 0, b = 0 should return Z=inf, not raise."""
        result = discovery_significance(5.0, 0.0)
        assert result['Z'] == np.inf

    def test_valid_returns_finite(self):
        result = discovery_significance(5.0, 10.0)
        assert np.isfinite(result['Z'])
        assert np.isfinite(result['p_value'])


class TestKsTestGuards:
    def test_nan_in_data(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            ks_test([1.0, float("nan"), 2.0], 'norm')

    def test_inf_in_data(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            ks_test([1.0, float("inf"), 2.0], 'norm')

    def test_empty_data(self):
        with pytest.raises(ValueError, match="empty"):
            ks_test([], 'norm')

    def test_valid_returns_dict(self):
        np.random.seed(42)
        result = ks_test(np.random.randn(100), 'norm')
        assert 'p_value' in result
        assert np.isfinite(result['p_value'])


class TestResidualDiagnosticsGuards:
    def test_nan_in_residuals(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            residual_diagnostics([0.1, float("nan"), -0.2])

    def test_inf_in_residuals(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            residual_diagnostics([0.1, float("inf"), -0.2])

    def test_empty_residuals(self):
        with pytest.raises(ValueError, match="empty"):
            residual_diagnostics([])

    def test_valid_returns_dict(self):
        np.random.seed(42)
        result = residual_diagnostics(np.random.randn(50))
        assert 'skewness' in result
        assert np.isfinite(result['skewness'])


# ============================================================================
# CROSS-MODULE: verify NaN doesn't silently propagate through pipelines
# ============================================================================

class TestPipelineNaNPropagation:
    """Ensure NaN at any pipeline stage is caught, not silently passed."""

    def test_phi_to_hC_pipeline(self):
        """If phi_fast raises on NaN, downstream hC_scalar_fast also raises."""
        with pytest.raises(ValueError):
            hC_scalar_fast(float("nan"))

    def test_hR_scalar_to_derivative_pipeline(self):
        """dhR_scalar_dx(NaN) must raise before calling hR_scalar_fast."""
        with pytest.raises(ValueError, match="finite"):
            dhR_scalar_dx(float("nan"), xi=0.0)

    def test_chi2_with_nan_observation(self):
        """NaN in observed values is now rejected with ValueError (not silent NaN)."""
        with pytest.raises(ValueError, match="observed.*NaN"):
            chi2([1.0, float("nan")], [1.0, 1.0], [0.1, 0.1])
