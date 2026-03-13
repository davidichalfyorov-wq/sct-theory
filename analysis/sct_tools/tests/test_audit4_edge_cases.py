"""
Edge-case tests for Audit Round 4 fixes.

Covers:
    C1: x < 0 ValueError in all _fast and _mp form factor functions
    C2: chi2_cov Tikhonov regularization warning on singular cov matrix
    M2: dhR_scalar_dx accuracy near x = 0 (Taylor series derivative)
    M4: likelihood_ratio_test negative statistic clamping
    Minor: chi2 rejects err <= 0 (not just err == 0)
    M3: verify_vacuum numerical fallback
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting
from sct_tools import form_factors as ff

# ============================================================================
# C1: x < 0 raises ValueError in all _fast form factor functions
# ============================================================================

class TestNegativeXFast:
    """All _fast form factor functions must raise ValueError for x < 0."""

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_phi_fast_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.phi_fast(x)

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_phi_vec_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.phi_vec(np.array([x]))

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_hC_scalar_fast_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_scalar_fast(x)

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_hR_scalar_fast_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hR_scalar_fast(x)

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_hC_dirac_fast_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_dirac_fast(x)

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_hR_dirac_fast_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hR_dirac_fast(x)

    @pytest.mark.parametrize("x", [-0.001, -1.0, -100.0])
    def test_dhR_scalar_dx_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.dhR_scalar_dx(x)


# ============================================================================
# C1: x < 0 raises ValueError in all _mp form factor functions
# ============================================================================

class TestNegativeXMp:
    """All _mp form factor functions must raise ValueError for x < 0."""

    @pytest.mark.parametrize("x", [-0.01, -5.0])
    def test_hC_scalar_mp_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_scalar_mp(x, dps=30)

    @pytest.mark.parametrize("x", [-0.01, -5.0])
    def test_hR_scalar_mp_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hR_scalar_mp(x, dps=30)

    @pytest.mark.parametrize("x", [-0.01, -5.0])
    def test_hC_dirac_mp_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_dirac_mp(x, dps=30)

    @pytest.mark.parametrize("x", [-0.01, -5.0])
    def test_hR_dirac_mp_negative(self, x):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hR_dirac_mp(x, dps=30)


# ============================================================================
# C1: x = 0 still works (boundary)
# ============================================================================

class TestZeroBoundary:
    """x = 0 must still produce correct values (not raise)."""

    def test_phi_fast_zero(self):
        assert ff.phi_fast(0) == 1.0

    def test_hC_scalar_fast_zero(self):
        assert ff.hC_scalar_fast(0) == pytest.approx(1.0 / 120, rel=1e-12)

    def test_hR_scalar_fast_zero_minimal(self):
        assert ff.hR_scalar_fast(0, xi=0) == pytest.approx(1.0 / 72, rel=1e-12)

    def test_hR_scalar_fast_zero_conformal(self):
        assert ff.hR_scalar_fast(0, xi=1.0/6) == pytest.approx(0.0, abs=1e-14)

    def test_hC_dirac_fast_zero(self):
        assert ff.hC_dirac_fast(0) == pytest.approx(-1.0 / 20, rel=1e-12)

    def test_hR_dirac_fast_zero(self):
        assert ff.hR_dirac_fast(0) == pytest.approx(0.0, abs=1e-14)


# ============================================================================
# C2: chi2_cov warns on Tikhonov regularization
# ============================================================================

class TestChi2CovTikhonovWarning:
    """chi2_cov must emit a warning when using Tikhonov regularization
    for a singular (non-positive-definite) covariance matrix."""

    def test_singular_cov_warns(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.1, 3.1])
        # Singular cov: all rows/cols identical
        cov = np.ones((3, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fitting.chi2_cov(obs, exp, cov)
            tikhonov_warnings = [
                x for x in w
                if "Tikhonov" in str(x.message) or "positive definite" in str(x.message)
            ]
            assert len(tikhonov_warnings) >= 1, (
                f"Expected Tikhonov warning, got {[str(x.message) for x in w]}"
            )
        # Result should still be a valid tuple
        assert len(result) == 4
        chi2_val, ndof, chi2_red, p_val = result
        assert np.isfinite(chi2_val)
        assert chi2_val >= 0

    def test_good_cov_no_warning(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.1, 3.1])
        cov = np.diag([0.1, 0.2, 0.3])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitting.chi2_cov(obs, exp, cov)
            tikhonov_warnings = [
                x for x in w
                if "Tikhonov" in str(x.message)
            ]
            assert len(tikhonov_warnings) == 0, (
                f"Unexpected Tikhonov warning for valid cov: {[str(x.message) for x in w]}"
            )


# ============================================================================
# M4: likelihood_ratio_test clamps negative statistic
# ============================================================================

class TestLikelihoodRatioNegativeStat:
    """When null model fits better than alt, test statistic < 0.
    Should be clamped to 0 with a warning."""

    def test_negative_stat_clamped(self):
        """null=-90 (better) vs alt=-100 => raw stat = -2*(-90 - (-100)) = -20.
        Should clamp to 0, p_value = 1."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fitting.likelihood_ratio_test(-90.0, -100.0, df_diff=2)
            neg_warnings = [
                x for x in w
                if "negative" in str(x.message).lower()
            ]
            assert len(neg_warnings) >= 1, (
                f"Expected negative-stat warning, got {[str(x.message) for x in w]}"
            )
        assert result['statistic'] == pytest.approx(0.0, abs=1e-14)
        assert result['p_value'] == pytest.approx(1.0, abs=1e-14)
        assert bool(result['reject_null_5pct']) is False

    def test_equal_loglik_no_warning(self):
        """Equal log-likelihoods => stat=0, no negative warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fitting.likelihood_ratio_test(-100.0, -100.0, df_diff=1)
            neg_warnings = [
                x for x in w
                if "negative" in str(x.message).lower()
            ]
            assert len(neg_warnings) == 0
        assert result['statistic'] == pytest.approx(0.0, abs=1e-14)


# ============================================================================
# Minor: chi2 rejects negative uncertainties
# ============================================================================

class TestChi2NegativeErrors:
    """chi2() should reject zero AND negative error values."""

    def test_zero_error_raises(self):
        obs = np.array([1.0, 2.0])
        exp = np.array([1.0, 2.0])
        err = np.array([0.1, 0.0])
        with pytest.raises(ValueError, match="zero or negative"):
            fitting.chi2(obs, exp, err)

    def test_negative_error_raises(self):
        obs = np.array([1.0, 2.0])
        exp = np.array([1.0, 2.0])
        err = np.array([0.1, -0.5])
        with pytest.raises(ValueError, match="zero or negative"):
            fitting.chi2(obs, exp, err)

    def test_positive_errors_ok(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 1.9, 3.05])
        err = np.array([0.1, 0.2, 0.1])
        result = fitting.chi2(obs, exp, err)
        assert len(result) == 4
        assert result[0] > 0


# ============================================================================
# M2: dhR_scalar_dx accuracy near x = 0
# ============================================================================

class TestDhRScalarDxAccuracy:
    """dhR_scalar_dx must be accurate near x = 0 where old numerical
    differentiation had problems."""

    def test_zero_value_minimal(self):
        """d/dx h_R at x=0, xi=0: leading Taylor term k=1 gives the slope."""
        val = ff.dhR_scalar_dx(0.0, xi=0)
        # Should be finite and well-defined (not NaN or huge)
        assert np.isfinite(val)

    def test_zero_value_conformal(self):
        """At xi=1/6 (conformal), h_R = 0 at x=0 and derivative should be finite."""
        val = ff.dhR_scalar_dx(0.0, xi=1.0/6)
        assert np.isfinite(val)

    def test_small_x_positive(self):
        """For small x > 0 with xi=0, derivative should be smooth and finite."""
        for x in [1e-8, 1e-6, 1e-4, 0.01, 0.1, 0.5, 1.0, 1.9]:
            val = ff.dhR_scalar_dx(x, xi=0)
            assert np.isfinite(val), f"dhR_scalar_dx({x}) is not finite: {val}"

    def test_continuity_at_threshold(self):
        """Verify continuity of dhR_scalar_dx at the Taylor/numerical boundary (x=2)."""
        # Evaluate just below and above threshold
        x_below = 1.99
        x_above = 2.01
        val_below = ff.dhR_scalar_dx(x_below, xi=0)
        val_above = ff.dhR_scalar_dx(x_above, xi=0)
        # Should be close (continuity)
        # Taylor (x<2) and numerical (x>=2) methods agree to ~1% at crossover
        assert val_below == pytest.approx(val_above, rel=1e-2), (
            f"Discontinuity at threshold: f(1.99)={val_below}, f(2.01)={val_above}"
        )

    def test_sign_consistency(self):
        """h_R^(0)(x; xi=0) is monotonically decreasing for small x > 0.
        Therefore the derivative should be negative near x = 0."""
        # h_R(0) = 1/72, h_R(1) < h_R(0), so derivative should be < 0
        val_small = ff.dhR_scalar_dx(0.01, xi=0)
        # h_R^(0)(x; xi=0) decreases from h_R(0) = 1/72: derivative must be negative
        assert np.isfinite(val_small)
        assert val_small < 0, f"expected negative derivative near x=0, got {val_small}"

    @pytest.mark.parametrize("xi", [0, 1/6, 0.5, 1.0])
    def test_vs_mpmath_finite_difference(self, xi):
        """Cross-check dhR_scalar_dx against mpmath finite difference at x=0.5."""
        x = 0.5
        h = 1e-8
        # Ground truth via mpmath finite difference
        fp = float(ff.hR_scalar_mp(x + h, xi=xi, dps=50))
        fm = float(ff.hR_scalar_mp(x - h, xi=xi, dps=50))
        mp_deriv = (fp - fm) / (2 * h)
        fast_deriv = ff.dhR_scalar_dx(x, xi)
        assert fast_deriv == pytest.approx(mp_deriv, rel=1e-5), (
            f"dhR_scalar_dx({x}, xi={xi})={fast_deriv} vs mpmath={mp_deriv}"
        )


# ============================================================================
# Seed diversification for fitting tests
# ============================================================================

class TestFittingSeedDiversity:
    """Verify fitting functions produce correct results across different seeds.

    Previous tests all used seed(42). These use varied seeds to ensure
    no seed-dependent false passes.
    """

    def test_ks_test_seed_137(self):
        """Normal data vs 'norm' with seed 137."""
        np.random.seed(137)
        data = np.random.standard_normal(500)
        result = fitting.ks_test(data, 'norm')
        assert result['p_value'] > 0.05

    def test_ks_test_seed_2025(self):
        """Normal data vs 'norm' with seed 2025."""
        np.random.seed(2025)
        data = np.random.standard_normal(500)
        result = fitting.ks_test(data, 'norm')
        assert result['p_value'] > 0.05

    def test_residual_diagnostics_seed_99(self):
        """Normal residuals with seed 99: DW close to 2."""
        np.random.seed(99)
        residuals = np.random.standard_normal(200)
        result = fitting.residual_diagnostics(residuals)
        dw = result['durbin_watson']['statistic']
        assert 1.5 < dw < 2.5

    def test_residual_diagnostics_seed_777(self):
        """Normal residuals with seed 777: Shapiro-Wilk passes."""
        np.random.seed(777)
        residuals = np.random.standard_normal(100)
        result = fitting.residual_diagnostics(residuals)
        assert result['shapiro_wilk']['p_value'] > 0.05

    def test_chi2_seed_314(self):
        """chi2 with known linear model, seed 314."""
        np.random.seed(314)
        x = np.linspace(0, 10, 50)
        y_true = 2.0 * x + 1.0
        noise = 0.2 * np.random.standard_normal(50)
        y = y_true + noise
        err = 0.2 * np.ones(50)
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(y, y_true, err)
        # chi2_red should be close to 1 for correct model
        assert 0.3 < chi2_red < 3.0, f"chi2_red={chi2_red} out of expected range"

    def test_chi2_seed_1618(self):
        """chi2 with exact match data, seed 1618."""
        np.random.seed(1618)
        n = 30
        obs = np.random.standard_normal(n)
        err = np.ones(n)
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(obs, obs, err)
        assert chi2_val == pytest.approx(0.0, abs=1e-14)
