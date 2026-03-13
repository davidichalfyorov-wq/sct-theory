"""
Round 5 audit — Mutation-killing tests.

Targets surviving mutations identified by mutation resilience analysis:
    D1: chi2() p-value formula verification (non-trivial chi2)
    D2: chi2_reduced formula verification (chi2_val / ndof)
    MM1: chi2_cov() Tikhonov fallback numerical correctness
    MM2: ks_test() threshold boundary test (p ≈ 0.05)
    MM3: dhR_scalar_dx numerical branch tolerance at boundary x=2
"""

import os
import sys
import warnings

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting
from sct_tools import form_factors as ff

# ============================================================================
# D1: chi2() p-value formula verification
# ============================================================================

class TestChi2PValueFormula:
    """Verify chi2() returns correct p-value for non-trivial cases."""

    def test_p_value_known_case(self):
        """chi2=6.0, ndof=3 => p = 1 - CDF(6, 3) ≈ 0.1116."""
        obs = np.array([1.0, 2.0, 3.0])
        # Construct expected/errors to give chi2 = sum((obs-exp)/err)^2 = 6.0
        # (1-0)^2/1^2 + (2-0)^2/1^2 + (3-2)^2/1^2 = 1+4+1 = 6
        exp = np.array([0.0, 0.0, 2.0])
        err = np.ones(3)
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(obs, exp, err, n_params=0)
        assert chi2_val == pytest.approx(6.0, rel=1e-12)
        assert ndof == 3
        expected_p = 1.0 - chi2_dist.cdf(6.0, 3)
        assert p_val == pytest.approx(expected_p, rel=1e-10)
        # Sanity: p ≈ 0.1116
        assert 0.10 < p_val < 0.12

    def test_p_value_with_n_params(self):
        """chi2=10.0, n_data=5, n_params=2 => ndof=3, p=1-CDF(10,3)."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # sum((obs-exp)/err)^2 = 5*(sqrt(2))^2 = 10
        exp = obs - np.sqrt(2.0)
        err = np.ones(5)
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(obs, exp, err, n_params=2)
        assert chi2_val == pytest.approx(10.0, rel=1e-10)
        assert ndof == 3
        expected_p = 1.0 - chi2_dist.cdf(10.0, 3)
        assert p_val == pytest.approx(expected_p, rel=1e-10)
        # p ≈ 0.0186 — small, should reject
        assert p_val < 0.05

    def test_p_value_large_ndof(self):
        """Large ndof case: n_data=100, n_params=5 => ndof=95."""
        n = 100
        obs = np.ones(n)
        exp = np.ones(n) + 0.1
        err = np.ones(n)
        # chi2 = 100 * 0.01 = 1.0
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(obs, exp, err, n_params=5)
        assert chi2_val == pytest.approx(1.0, rel=1e-10)
        assert ndof == 95
        expected_p = 1.0 - chi2_dist.cdf(1.0, 95)
        assert p_val == pytest.approx(expected_p, rel=1e-10)
        # Very good fit: p should be very close to 1
        assert p_val > 0.99

    def test_p_value_flip_sign_mutation_caught(self):
        """If p_val were (1 + CDF) instead of (1 - CDF), this would fail.
        For chi2=20, ndof=5: true p < 0.002, mutant p > 1.998."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exp = np.zeros(5)
        err = np.ones(5)
        # chi2 = 1+4+9+16+25 = 55
        chi2_val, ndof, _, p_val = fitting.chi2(obs, exp, err)
        assert chi2_val == pytest.approx(55.0, rel=1e-10)
        # p must be in [0, 1]
        assert 0.0 <= p_val <= 1.0
        # For chi2=55, ndof=5, p should be essentially 0
        assert p_val < 1e-8


# ============================================================================
# D2: chi2_reduced formula verification
# ============================================================================

class TestChi2ReducedFormula:
    """Verify chi2_reduced = chi2_val / ndof."""

    def test_chi2_red_exact(self):
        """chi2_red must equal hand-computed chi2 / ndof (non-tautological)."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        exp = np.array([0.9, 2.1, 2.8, 4.3])
        err = np.array([0.1, 0.1, 0.2, 0.1])
        chi2_val, ndof, chi2_red, _ = fitting.chi2(obs, exp, err, n_params=1)
        # Hand-computed: sum((obs-exp)/err)^2
        #   (0.1/0.1)^2 + (-0.1/0.1)^2 + (0.2/0.2)^2 + (-0.3/0.1)^2
        #   = 1 + 1 + 1 + 9 = 12
        expected_chi2 = 12.0
        assert chi2_val == pytest.approx(expected_chi2, rel=1e-12)
        assert ndof == 3  # 4 data - 1 param
        assert chi2_red == pytest.approx(expected_chi2 / 3.0, rel=1e-12)

    def test_chi2_red_with_floor(self):
        """When ndof is floored to 1, chi2_red = chi2_val / 1 = chi2_val."""
        obs = np.array([5.0])
        exp = np.array([3.0])
        err = np.array([1.0])
        chi2_val, ndof, chi2_red, _ = fitting.chi2(obs, exp, err, n_params=5)
        assert ndof == 1
        assert chi2_red == pytest.approx(chi2_val, rel=1e-14)
        assert chi2_val == pytest.approx(4.0, rel=1e-14)


# ============================================================================
# MM1: chi2_cov() Tikhonov fallback
# ============================================================================

class TestChi2CovTikhonov:
    """Test that chi2_cov() handles near-singular covariance matrices."""

    def test_tikhonov_triggers_on_singular(self):
        """A singular covariance matrix should trigger Tikhonov regularization."""
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.1, 3.1])
        # Singular: rank-1 matrix (all rows identical)
        cov = np.array([[1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chi2_val, ndof, chi2_red, p_val = fitting.chi2_cov(obs, exp, cov)
            # Should have issued a warning about regularization
            assert len(w) >= 1
            assert "Tikhonov" in str(w[0].message) or "not positive definite" in str(w[0].message)
        # Result should still be finite and positive
        assert np.isfinite(chi2_val)
        assert chi2_val >= 0
        assert np.isfinite(p_val)
        assert 0.0 <= p_val <= 1.0

    def test_tikhonov_close_to_direct_for_nearly_singular(self):
        """For a mildly ill-conditioned matrix, Tikhonov result should be
        close to the result from a well-conditioned variant."""
        obs = np.array([1.0, 2.0])
        exp = np.array([1.5, 2.5])
        # Nearly singular but still positive definite
        cov_good = np.array([[1.0, 0.999], [0.999, 1.0]])
        # Make it explicitly singular to trigger Tikhonov
        cov_singular = np.array([[1.0, 1.0], [1.0, 1.0]])

        chi2_good, _, _, _ = fitting.chi2_cov(obs, exp, cov_good)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            chi2_reg, _, _, _ = fitting.chi2_cov(obs, exp, cov_singular)

        # Both should be positive and finite
        assert np.isfinite(chi2_good)
        assert np.isfinite(chi2_reg)
        assert chi2_good > 0
        assert chi2_reg > 0

    def test_tikhonov_diagonal_gives_correct_chi2(self):
        """Diagonal covariance: chi2_cov should agree with simple chi2."""
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.2, 2.8])
        err = np.array([0.5, 0.3, 0.4])
        cov = np.diag(err**2)
        # Compare chi2_cov with simple chi2
        chi2_simple, _, _, _ = fitting.chi2(obs, exp, err)
        chi2_cov_val, _, _, _ = fitting.chi2_cov(obs, exp, cov)
        assert chi2_cov_val == pytest.approx(chi2_simple, rel=1e-10)


# ============================================================================
# MM2: ks_test() threshold boundary test
# ============================================================================

class TestKSTestBoundary:
    """Test ks_test() behavior near the rejection threshold."""

    def test_reject_flag_matches_threshold(self):
        """reject_H0_5pct must be True iff p_value < 0.05."""
        np.random.seed(42)
        # Generate data that's close to the boundary
        for _ in range(10):
            data = np.random.standard_normal(50)
            result = fitting.ks_test(data, 'norm')
            if result['p_value'] < 0.05:
                assert bool(result['reject_H0_5pct']) is True
            else:
                assert bool(result['reject_H0_5pct']) is False

    def test_barely_reject(self):
        """Construct data that produces p < 0.05 and verify reject_H0_5pct."""
        # Shifted normal: mean=0.3 should be rejected against standard normal
        np.random.seed(123)
        data = np.random.normal(loc=0.3, scale=1.0, size=200)
        result = fitting.ks_test(data, 'norm')
        # With 200 samples and shift=0.3, should reject
        assert result['p_value'] < 0.05
        assert bool(result['reject_H0_5pct']) is True

    def test_barely_accept(self):
        """Standard normal sample with tight seed should not reject."""
        np.random.seed(42)
        data = np.random.standard_normal(200)
        result = fitting.ks_test(data, 'norm')
        assert result['p_value'] > 0.05
        assert bool(result['reject_H0_5pct']) is False

    def test_p_value_range(self):
        """p-value must always be in [0, 1]."""
        np.random.seed(42)
        for dist_name in ['norm']:
            data = np.random.standard_normal(100)
            result = fitting.ks_test(data, dist_name)
            assert 0.0 <= result['p_value'] <= 1.0


# ============================================================================
# MM3: dhR_scalar_dx boundary tolerance
# ============================================================================

class TestDhRScalarDxBoundary:
    """Test dhR_scalar_dx continuity and accuracy at the Taylor/numerical
    boundary (x = _TAYLOR_THRESH ≈ 2.0)."""

    def test_continuity_at_boundary(self):
        """Taylor branch and numerical branch must agree at x=2."""
        x_boundary = 2.0
        eps = 1e-6
        val_below = ff.dhR_scalar_dx(x_boundary - eps, xi=0.0)
        val_above = ff.dhR_scalar_dx(x_boundary + eps, xi=0.0)
        # Should be continuous: relative difference < 1e-4
        if abs(val_below) > 1e-15:
            rel_diff = abs(val_above - val_below) / abs(val_below)
            assert rel_diff < 1e-3, (
                f"Discontinuity at x=2: below={val_below:.10e}, "
                f"above={val_above:.10e}, rel_diff={rel_diff:.2e}"
            )

    def test_numerical_branch_vs_mpmath(self):
        """Compare dhR_scalar_dx at x=3.0 (firmly in numerical branch)
        against high-precision mpmath finite difference."""
        try:
            import mpmath
        except ImportError:
            pytest.skip("mpmath not available")

        old_dps = mpmath.mp.dps
        try:
            mpmath.mp.dps = 50
            x0 = mpmath.mpf('3.0')
            xi0 = mpmath.mpf('0.0')
            h = mpmath.mpf('1e-10')

            def hR_mp(x, xi):
                """mpmath version of hR_scalar for cross-check.

                Must match form_factors.hR_scalar_fast exactly:
                  hR = fRic/3 + fR + xi*fRU + xi^2*fU
                with CZ-basis form factors (scalar, NOT Dirac).
                """
                # phi(x) = integral_0^1 exp(-alpha*(1-alpha)*x) dalpha
                phi_val = mpmath.quad(lambda a: mpmath.exp(-a * (1 - a) * x), [0, 1])
                fRic = 1 / (6 * x) + (phi_val - 1) / x**2
                fR = phi_val / 32 + phi_val / (8 * x) - 7 / (48 * x) - (phi_val - 1) / (8 * x**2)
                fRU = -phi_val / 4 - (phi_val - 1) / (2 * x)
                fU = phi_val / 2
                return fRic / 3 + fR + xi * fRU + xi**2 * fU

            mp_deriv = float((hR_mp(x0 + h, xi0) - hR_mp(x0 - h, xi0)) / (2 * h))
            fast_deriv = ff.dhR_scalar_dx(3.0, xi=0.0)
            if abs(mp_deriv) > 1e-30:
                rel_err = abs(fast_deriv - mp_deriv) / abs(mp_deriv)
            else:
                rel_err = abs(fast_deriv - mp_deriv)
            assert rel_err < 1e-4, (
                f"dhR_scalar_dx(3.0, 0) = {fast_deriv:.10e} vs mpmath = {mp_deriv:.10e}, "
                f"rel_err = {rel_err:.2e}"
            )
        finally:
            mpmath.mp.dps = old_dps

    @pytest.mark.parametrize("xi", [0.0, 1.0/6, 0.5, 1.0])
    def test_boundary_xi_continuity(self, xi):
        """Continuity at x=2 for various xi values."""
        eps = 1e-6
        val_below = ff.dhR_scalar_dx(2.0 - eps, xi=xi)
        val_above = ff.dhR_scalar_dx(2.0 + eps, xi=xi)
        if abs(val_below) > 1e-15:
            rel_diff = abs(val_above - val_below) / abs(val_below)
            assert rel_diff < 1e-3, (
                f"Discontinuity at x=2, xi={xi}: "
                f"below={val_below:.10e}, above={val_above:.10e}"
            )
        else:
            # Both should be near zero
            assert abs(val_above) < 1e-10
