"""
Tests for previously untested functions in sct_tools.fitting.

Covers:
    - likelihood_ratio_test
    - ks_test
    - anderson_darling_test (requires scipy >= 1.13 for MonteCarloMethod)
    - weighted_least_squares
    - residual_diagnostics
    - fit_minuit_minos
    - fit_lmfit
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting

# ---------------------------------------------------------------------------
# Availability flags for optional packages
# ---------------------------------------------------------------------------

try:
    import iminuit  # noqa: F401
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False

try:
    import lmfit  # noqa: F401
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

try:
    import statsmodels  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy.stats import MonteCarloMethod  # noqa: F401
    HAS_MONTECARLO = True
except ImportError:
    HAS_MONTECARLO = False


# ============================================================================
# likelihood_ratio_test
# ============================================================================

class TestLikelihoodRatioTest:
    """Tests for fitting.likelihood_ratio_test."""

    def test_known_values(self):
        """null=-100, alt=-90, df=2 => statistic=20, very small p-value."""
        result = fitting.likelihood_ratio_test(-100.0, -90.0, df_diff=2)
        assert result['statistic'] == pytest.approx(20.0, rel=1e-14)
        assert result['df'] == 2
        # chi2(20, df=2): p ~ 4.5e-5; definitely reject at 5%
        assert result['p_value'] < 0.001
        assert bool(result['reject_null_5pct']) is True

    def test_identical_models(self):
        """Same log-likelihood => statistic=0, p_value=1."""
        result = fitting.likelihood_ratio_test(-100.0, -100.0, df_diff=1)
        assert result['statistic'] == pytest.approx(0.0, abs=1e-14)
        assert result['p_value'] == pytest.approx(1.0, abs=1e-14)
        assert bool(result['reject_null_5pct']) is False

    def test_statistic_nonnegative(self):
        """When alt >= null (better fit), statistic must be >= 0."""
        for null, alt in [(-100, -95), (-50, -50), (-200, -100)]:
            result = fitting.likelihood_ratio_test(null, alt, df_diff=1)
            assert result['statistic'] >= 0.0, (
                f"Negative statistic for null={null}, alt={alt}"
            )

    def test_large_df(self):
        """With many extra parameters and small improvement, should NOT reject."""
        # statistic = 2*(-100 - (-101)) = 2, df=10 => p ~ 0.996
        result = fitting.likelihood_ratio_test(-101.0, -100.0, df_diff=10)
        assert result['statistic'] == pytest.approx(2.0, rel=1e-14)
        assert result['p_value'] > 0.5
        assert bool(result['reject_null_5pct']) is False

    def test_df_is_preserved(self):
        """The returned df should match df_diff input."""
        for df in [1, 3, 7, 15]:
            result = fitting.likelihood_ratio_test(-100.0, -90.0, df_diff=df)
            assert result['df'] == df

    def test_marginal_rejection(self):
        """A case near the 5% boundary for df=1.
        chi2 critical value at 5% for df=1 is 3.841.
        statistic = 2*(100 - 98) = 4 => p ~ 0.046 => reject.
        """
        result = fitting.likelihood_ratio_test(-100.0, -98.0, df_diff=1)
        assert result['statistic'] == pytest.approx(4.0, rel=1e-14)
        assert result['p_value'] < 0.05
        assert bool(result['reject_null_5pct']) is True


# ============================================================================
# ks_test
# ============================================================================

class TestKSTest:
    """Tests for fitting.ks_test."""

    def test_normal_data_vs_norm(self):
        """Standard normal data tested against 'norm' should NOT reject."""
        np.random.seed(42)
        data = np.random.standard_normal(500)
        result = fitting.ks_test(data, 'norm')
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'reject_H0_5pct' in result
        assert result['p_value'] > 0.05
        assert bool(result['reject_H0_5pct']) is False

    def test_uniform_data_vs_norm(self):
        """Uniform(0,1) data tested against 'norm' should reject."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 500)
        result = fitting.ks_test(data, 'norm')
        assert result['p_value'] < 0.05
        assert bool(result['reject_H0_5pct']) is True

    def test_statistic_in_zero_one(self):
        """KS statistic is always in [0, 1]."""
        np.random.seed(42)
        data = np.random.standard_normal(100)
        result = fitting.ks_test(data, 'norm')
        assert 0.0 <= result['statistic'] <= 1.0

    def test_custom_cdf_callable(self):
        """Test with a callable CDF (standard uniform)."""
        from scipy.stats import uniform
        np.random.seed(42)
        data = np.random.uniform(0, 1, 300)
        result = fitting.ks_test(data, uniform.cdf)
        assert result['p_value'] > 0.05
        assert bool(result['reject_H0_5pct']) is False

    def test_small_sample(self):
        """KS test should still run on small samples (low power).
        Statistic must be in [0,1], p_value in [0,1], and for normal
        data from 'norm' the test should NOT reject (low power)."""
        np.random.seed(42)
        data = np.random.standard_normal(10)
        result = fitting.ks_test(data, 'norm')
        assert 0.0 <= result['statistic'] <= 1.0
        assert 0.0 <= result['p_value'] <= 1.0
        # With only 10 points from standard normal, should not reject
        assert result['p_value'] > 0.01


# ============================================================================
# anderson_darling_test
# ============================================================================

@pytest.mark.skipif(not HAS_MONTECARLO,
                    reason="scipy.stats.MonteCarloMethod not available (scipy < 1.13)")
class TestAndersonDarlingTest:
    """Tests for fitting.anderson_darling_test.

    Requires scipy >= 1.13 for MonteCarloMethod.
    """

    def test_normal_data(self):
        """Normal data should not be rejected as normal."""
        np.random.seed(42)
        data = np.random.standard_normal(200)
        result = fitting.anderson_darling_test(data, dist='norm')
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'reject_5pct' in result
        assert result['p_value'] > 0.05
        assert result['reject_5pct'] is False

    def test_exponential_vs_norm(self):
        """Exponential data should be rejected as normal."""
        np.random.seed(42)
        data = np.random.exponential(1.0, 200)
        result = fitting.anderson_darling_test(data, dist='norm')
        assert result['p_value'] < 0.05
        assert result['reject_5pct'] is True

    def test_statistic_positive_and_bounded(self):
        """Anderson-Darling statistic is always positive and bounded for
        well-behaved data.  For 100 standard normal samples, the AD
        statistic should typically be < 2 (critical value at 1% is ~1.09)."""
        np.random.seed(42)
        data = np.random.standard_normal(100)
        result = fitting.anderson_darling_test(data, dist='norm')
        assert result['statistic'] > 0.0
        # For genuine normal data, AD stat should be small (< 2 comfortably)
        assert result['statistic'] < 2.0
        assert 0.0 <= result['p_value'] <= 1.0


# ============================================================================
# weighted_least_squares
# ============================================================================

@pytest.mark.skipif(not HAS_STATSMODELS,
                    reason="statsmodels not available")
class TestWeightedLeastSquares:
    """Tests for fitting.weighted_least_squares."""

    def test_linear_recovery(self):
        """y = 2x + 1 + noise => params ~ [1, 2] (intercept, slope)."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        noise = 0.1 * np.random.standard_normal(50)
        y = 2.0 * x + 1.0 + noise
        y_err = 0.1 * np.ones(50)

        result = fitting.weighted_least_squares(x, y, y_err, degree=1)
        # params[0] = intercept, params[1] = slope (columns: x^0, x^1)
        assert result.params[0] == pytest.approx(1.0, abs=0.15)
        assert result.params[1] == pytest.approx(2.0, abs=0.05)

    def test_rsquared_clean_data(self):
        """Clean linear data should give R^2 very close to 1."""
        np.random.seed(42)
        x = np.linspace(1, 10, 30)
        y = 3.0 * x - 2.0 + 0.001 * np.random.standard_normal(30)
        y_err = 0.01 * np.ones(30)

        result = fitting.weighted_least_squares(x, y, y_err, degree=1)
        assert result.rsquared > 0.999

    def test_quadratic_fit(self):
        """y = x^2 + 0.5x + 1 => should recover params for degree=2."""
        np.random.seed(42)
        x = np.linspace(-3, 3, 40)
        y = 1.0 * x**2 + 0.5 * x + 1.0 + 0.05 * np.random.standard_normal(40)
        y_err = 0.1 * np.ones(40)

        result = fitting.weighted_least_squares(x, y, y_err, degree=2)
        # params: [intercept (x^0), coeff of x^1, coeff of x^2]
        assert result.params[0] == pytest.approx(1.0, abs=0.15)
        assert result.params[1] == pytest.approx(0.5, abs=0.15)
        assert result.params[2] == pytest.approx(1.0, abs=0.15)

    def test_has_standard_errors(self):
        """Result should have .bse (standard errors for parameters)."""
        np.random.seed(42)
        x = np.linspace(0, 5, 20)
        y = x + np.random.standard_normal(20) * 0.1
        y_err = 0.1 * np.ones(20)
        result = fitting.weighted_least_squares(x, y, y_err, degree=1)
        assert len(result.bse) == 2
        assert all(e > 0 for e in result.bse)

    def test_large_errors_low_rsquared(self):
        """Very noisy data should yield low R^2."""
        np.random.seed(42)
        x = np.linspace(0, 1, 20)
        y = x + 100.0 * np.random.standard_normal(20)
        y_err = np.ones(20)
        result = fitting.weighted_least_squares(x, y, y_err, degree=1)
        assert result.rsquared < 0.5


# ============================================================================
# residual_diagnostics
# ============================================================================

class TestResidualDiagnostics:
    """Tests for fitting.residual_diagnostics."""

    def test_normal_residuals(self):
        """Normal residuals: Shapiro-Wilk should not reject at 5%."""
        np.random.seed(42)
        residuals = np.random.standard_normal(100)
        result = fitting.residual_diagnostics(residuals)

        assert result['shapiro_wilk']['p_value'] > 0.05
        assert bool(result['shapiro_wilk']['normal_5pct']) is True

    def test_mean_correct(self):
        """Mean of shifted residuals should match the shift."""
        np.random.seed(42)
        residuals = np.random.standard_normal(50) + 0.5  # shifted by 0.5
        result = fitting.residual_diagnostics(residuals)
        # Mean should be close to 0.5 (the shift) for large enough sample
        assert result['mean'] == pytest.approx(0.5, abs=0.3)
        # Also check it's a float (not None or NaN)
        assert np.isfinite(result['mean'])

    def test_std_correct(self):
        """Std of standard normal residuals should be close to 1."""
        np.random.seed(42)
        residuals = np.random.standard_normal(500)  # larger sample for tighter bound
        result = fitting.residual_diagnostics(residuals)
        # Std should be close to 1.0 for standard normal
        assert result['std'] == pytest.approx(1.0, abs=0.1)
        assert result['std'] > 0

    def test_durbin_watson_no_autocorrelation(self):
        """Independent normal residuals -> DW close to 2."""
        np.random.seed(42)
        residuals = np.random.standard_normal(200)
        result = fitting.residual_diagnostics(residuals)
        dw = result['durbin_watson']['statistic']
        assert 1.5 < dw < 2.5
        assert result['durbin_watson']['interpretation'] == 'no autocorrelation'

    def test_durbin_watson_autocorrelated(self):
        """Strongly autocorrelated residuals -> DW deviates from 2."""
        np.random.seed(42)
        # AR(1) process with rho ~ 0.95
        n = 200
        residuals = np.zeros(n)
        residuals[0] = np.random.standard_normal()
        for i in range(1, n):
            residuals[i] = 0.95 * residuals[i - 1] + 0.1 * np.random.standard_normal()
        result = fitting.residual_diagnostics(residuals)
        dw = result['durbin_watson']['statistic']
        assert dw < 1.0  # strong positive autocorrelation => DW << 2

    def test_skewness_symmetric(self):
        """Symmetric residuals should have near-zero skewness."""
        np.random.seed(42)
        residuals = np.random.standard_normal(500)
        result = fitting.residual_diagnostics(residuals)
        assert abs(result['skewness']) < 0.3

    def test_non_normal_residuals_detected(self):
        """Highly skewed data should fail Shapiro-Wilk."""
        np.random.seed(42)
        residuals = np.random.exponential(1.0, 100)
        result = fitting.residual_diagnostics(residuals)
        assert result['shapiro_wilk']['p_value'] < 0.05
        assert bool(result['shapiro_wilk']['normal_5pct']) is False

    def test_small_sample(self):
        """With fewer than 3 residuals, Shapiro-Wilk returns NaN."""
        residuals = np.array([0.1, -0.1])
        result = fitting.residual_diagnostics(residuals)
        assert np.isnan(result['shapiro_wilk']['statistic'])
        assert result['shapiro_wilk']['normal_5pct'] is None

    def test_all_keys_present(self):
        """Result dict should contain all expected keys."""
        np.random.seed(42)
        result = fitting.residual_diagnostics(np.random.standard_normal(20))
        assert 'shapiro_wilk' in result
        assert 'durbin_watson' in result
        assert 'mean' in result
        assert 'std' in result
        assert 'skewness' in result


# ============================================================================
# fit_minuit_minos
# ============================================================================

@pytest.mark.skipif(not HAS_IMINUIT,
                    reason="iminuit not available")
class TestFitMinuitMinos:
    """Tests for fitting.fit_minuit_minos."""

    def test_simple_parabola(self):
        """Minimize (x-3)^2 => should find x=3 with symmetric MINOS errors."""
        def cost(x):
            return (x - 3.0) ** 2

        m = fitting.fit_minuit_minos(cost, [0.0], param_names=["x"])
        assert m.valid
        assert m.values["x"] == pytest.approx(3.0, abs=0.01)
        # MINOS errors should exist
        assert "x" in m.merrors
        # For a parabola, MINOS errors are symmetric
        me = m.merrors["x"]
        assert me.lower == pytest.approx(-me.upper, rel=0.1)

    def test_two_parameter_fit(self):
        """Minimize (x-1)^2 + 4*(y-2)^2."""
        def cost(x, y):
            return (x - 1.0) ** 2 + 4.0 * (y - 2.0) ** 2

        m = fitting.fit_minuit_minos(cost, [0.0, 0.0], param_names=["x", "y"])
        assert m.valid
        assert m.values["x"] == pytest.approx(1.0, abs=0.01)
        assert m.values["y"] == pytest.approx(2.0, abs=0.01)
        assert "x" in m.merrors
        assert "y" in m.merrors
        # y has steeper curvature, so smaller error
        assert abs(m.merrors["y"].upper) < abs(m.merrors["x"].upper)

    def test_errordef_nll(self):
        """Test with errordef=0.5 (NLL convention)."""
        def cost(mu):
            return (mu - 5.0) ** 2

        m = fitting.fit_minuit_minos(cost, [0.0], param_names=["mu"],
                                     errordef=0.5)
        assert m.valid
        assert m.values["mu"] == pytest.approx(5.0, abs=0.01)
        assert m.errordef == 0.5

    def test_with_limits(self):
        """Parameter with limits: true minimum at 10, but limited to [0, 5]."""
        def cost(x):
            return (x - 10.0) ** 2

        m = fitting.fit_minuit_minos(cost, [1.0], param_names=["x"],
                                     limits={"x": (0, 5)})
        assert m.values["x"] == pytest.approx(5.0, abs=0.1)

    def test_minos_errors_finite(self):
        """MINOS errors should be finite for well-behaved problems."""
        def cost(a, b):
            return (a - 2.0) ** 2 + (b + 1.0) ** 2

        m = fitting.fit_minuit_minos(cost, [0.0, 0.0], param_names=["a", "b"])
        for name in ["a", "b"]:
            me = m.merrors[name]
            assert np.isfinite(me.lower)
            assert np.isfinite(me.upper)
            assert me.lower < 0  # lower error is negative by convention
            assert me.upper > 0  # upper error is positive


# ============================================================================
# fit_lmfit
# ============================================================================

@pytest.mark.skipif(not HAS_LMFIT,
                    reason="lmfit not available")
class TestFitLmfit:
    """Tests for fitting.fit_lmfit."""

    def test_linear_fit(self):
        """Fit y = a*x + b to linear data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = 2.5 * x + 1.0
        noise = 0.2 * np.random.standard_normal(50)
        y = y_true + noise
        y_err = 0.2 * np.ones(50)

        def model(x, a, b):
            return a * x + b

        result = fitting.fit_lmfit(model, x, y, y_err,
                                   params_dict={'a': 1.0, 'b': 0.0})
        assert result.success
        params = result.params
        assert params['a'].value == pytest.approx(2.5, abs=0.2)
        assert params['b'].value == pytest.approx(1.0, abs=0.5)

    def test_params_with_bounds(self):
        """Test that parameter bounds are respected."""
        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = 3.0 * x + np.random.standard_normal(30) * 0.1
        y_err = 0.1 * np.ones(30)

        def model(x, a, b):
            return a * x + b

        # Constrain a to [0, 10] and b to [-1, 1]
        result = fitting.fit_lmfit(model, x, y, y_err,
                                   params_dict={'a': (1.0, 0.0, 10.0),
                                                'b': (0.0, -1.0, 1.0)})
        assert result.success
        assert 0.0 <= result.params['a'].value <= 10.0
        assert -1.0 <= result.params['b'].value <= 1.0

    def test_exponential_fit(self):
        """Fit y = A * exp(-k * x) to exponential data."""
        np.random.seed(42)
        x = np.linspace(0, 5, 40)
        A_true, k_true = 10.0, 0.5
        y = A_true * np.exp(-k_true * x) + 0.1 * np.random.standard_normal(40)
        y_err = 0.2 * np.ones(40)

        def model(x, A, k):
            return A * np.exp(-k * x)

        result = fitting.fit_lmfit(model, x, y, y_err,
                                   params_dict={'A': (5.0, 0.0, 50.0),
                                                'k': (1.0, 0.0, 10.0)})
        assert result.success
        assert result.params['A'].value == pytest.approx(A_true, abs=1.0)
        assert result.params['k'].value == pytest.approx(k_true, abs=0.2)

    def test_has_uncertainties(self):
        """Fitted parameters should have .stderr set."""
        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = 2.0 * x + 1.0 + 0.1 * np.random.standard_normal(30)
        y_err = 0.1 * np.ones(30)

        def model(x, a, b):
            return a * x + b

        result = fitting.fit_lmfit(model, x, y, y_err,
                                   params_dict={'a': 1.0, 'b': 0.0})
        assert result.params['a'].stderr is not None
        assert result.params['a'].stderr > 0
        assert result.params['b'].stderr is not None
        assert result.params['b'].stderr > 0

    def test_residuals_shape(self):
        """Residuals array should have same length as data."""
        np.random.seed(42)
        n = 25
        x = np.linspace(0, 5, n)
        y = x + 0.1 * np.random.standard_normal(n)
        y_err = 0.1 * np.ones(n)

        def model(x, a, b):
            return a * x + b

        result = fitting.fit_lmfit(model, x, y, y_err,
                                   params_dict={'a': 1.0, 'b': 0.0})
        assert len(result.residual) == n
