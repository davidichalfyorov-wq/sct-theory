"""
Tests for Iteration 30: asymptotic_expansion corrected 1/x^3 coefficients.

Covers:
    A1: hC_dirac 1/x^3 coefficient corrected from 4 to 6
    A2: hR_dirac 1/x^3 coefficient corrected from 5/3 to 2
    A3: hC_vector 1/x^3 term added (was missing, value 12)
    A4: hC_scalar 1/x^3 coefficient verified unchanged at 1
    A5: hR_vector 1/x^3 coefficient verified at ~0 (no term needed)
    A6: All asymptotic expansions converge at large x with correct rate
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors

# ============================================================================
# A1: hC_dirac corrected 1/x^3 coefficient = 6
# ============================================================================


class TestHCDiracAsymptotic:
    """hC_dirac asymptotic expansion with corrected C3=6."""

    def test_c3_coefficient_numerical_extraction(self):
        """Extract C3 from exact function: should converge to 6."""
        x = 1e4
        exact = form_factors.hC_dirac(x)
        leading = -1.0 / (6.0 * x) - 1.0 / x**2
        C3 = (exact - leading) * x**3
        assert C3 == pytest.approx(6.0, rel=1e-3)

    def test_rel_error_x100_below_0_02pct(self):
        exact = form_factors.hC_dirac(100)
        asym = form_factors.asymptotic_expansion('hC_dirac', 100)
        assert abs(asym / exact - 1) < 2e-4

    def test_rel_error_x1000_below_2e_7(self):
        exact = form_factors.hC_dirac(1000)
        asym = form_factors.asymptotic_expansion('hC_dirac', 1000)
        assert abs(asym / exact - 1) < 2e-7

    def test_sign_negative(self):
        """hC_dirac asymptotic is negative for all x > 0."""
        for x in [50, 100, 1000]:
            assert form_factors.asymptotic_expansion('hC_dirac', x) < 0


# ============================================================================
# A2: hR_dirac corrected 1/x^3 coefficient = 2
# ============================================================================


class TestHRDiracAsymptotic:
    """hR_dirac asymptotic expansion with corrected C3=2."""

    def test_c3_coefficient_numerical_extraction(self):
        """Extract C3 from exact function: should converge to 2."""
        x = 1e4
        exact = form_factors.hR_dirac(x)
        leading = 1.0 / (18.0 * x) - 2.0 / (3.0 * x**2)
        C3 = (exact - leading) * x**3
        assert C3 == pytest.approx(2.0, rel=1e-3)

    def test_rel_error_x100_below_0_02pct(self):
        exact = form_factors.hR_dirac(100)
        asym = form_factors.asymptotic_expansion('hR_dirac', 100)
        assert abs(asym / exact - 1) < 2e-4

    def test_rel_error_x1000_below_1e_7(self):
        exact = form_factors.hR_dirac(1000)
        asym = form_factors.asymptotic_expansion('hR_dirac', 1000)
        assert abs(asym / exact - 1) < 1e-7

    def test_sign_positive(self):
        """hR_dirac asymptotic is positive for all x > 0."""
        for x in [50, 100, 1000]:
            assert form_factors.asymptotic_expansion('hR_dirac', x) > 0


# ============================================================================
# A3: hC_vector 1/x^3 term added (C3 = 12)
# ============================================================================


class TestHCVectorAsymptotic:
    """hC_vector asymptotic expansion with added C3=12."""

    def test_c3_coefficient_numerical_extraction(self):
        """Extract C3 from exact function: should converge to 12."""
        x = 1e4
        exact = form_factors.hC_vector(x)
        leading = -1.0 / (3.0 * x) + 2.0 / x**2
        C3 = (exact - leading) * x**3
        assert C3 == pytest.approx(12.0, rel=1e-3)

    def test_rel_error_x100_below_0_04pct(self):
        exact = form_factors.hC_vector(100)
        asym = form_factors.asymptotic_expansion('hC_vector', 100)
        assert abs(asym / exact - 1) < 4e-4

    def test_rel_error_x1000_below_3e_7(self):
        exact = form_factors.hC_vector(1000)
        asym = form_factors.asymptotic_expansion('hC_vector', 1000)
        assert abs(asym / exact - 1) < 3e-7


# ============================================================================
# A4: hC_scalar 1/x^3 coefficient verified = 1 (unchanged)
# ============================================================================


class TestHCScalarAsymptotic:
    """hC_scalar asymptotic expansion — C3=1 verified correct."""

    def test_c3_coefficient_numerical_extraction(self):
        x = 1e4
        exact = form_factors.hC_scalar(x)
        leading = 1.0 / (12.0 * x) - 1.0 / (2.0 * x**2)
        C3 = (exact - leading) * x**3
        assert C3 == pytest.approx(1.0, rel=1e-3)

    def test_rel_error_x1000_below_3e_8(self):
        exact = form_factors.hC_scalar(1000)
        asym = form_factors.asymptotic_expansion('hC_scalar', 1000)
        assert abs(asym / exact - 1) < 3e-8


# ============================================================================
# A5: hR_vector 1/x^3 coefficient ~ 0 (no term needed)
# ============================================================================


class TestHRVectorAsymptotic:
    """hR_vector asymptotic — C3 vanishes (interesting UV property)."""

    def test_c3_coefficient_near_zero(self):
        """C3 contribution is negligible compared to 1/x^2 term."""
        x = 1e4
        exact = form_factors.hR_vector(x)
        leading = 1.0 / (9.0 * x) - 2.0 / (3.0 * x**2)
        residual = abs(exact - leading) * x**3
        # Residual should be << 1 (near zero), much smaller than other C3 values
        assert residual < 0.01

    def test_rel_error_x1000_below_5e_8(self):
        exact = form_factors.hR_vector(1000)
        asym = form_factors.asymptotic_expansion('hR_vector', 1000)
        assert abs(asym / exact - 1) < 5e-8


# ============================================================================
# A6: Convergence rate — error ~ O(1/x^4) for all expansions
# ============================================================================


class TestAsymptoticConvergenceRate:
    """With 1/x^3 terms correct, error should scale as O(1/x^4)."""

    @pytest.mark.parametrize("ff", [
        'hC_scalar', 'hC_dirac', 'hR_dirac', 'hC_vector', 'hR_vector',
    ])
    def test_error_decreases_as_x4(self, ff):
        """Doubling x should reduce relative error by ~16x."""
        x1, x2 = 200, 400
        exact1 = getattr(form_factors, ff)(x1)
        asym1 = form_factors.asymptotic_expansion(ff, x1)
        err1 = abs(asym1 / exact1 - 1)

        exact2 = getattr(form_factors, ff)(x2)
        asym2 = form_factors.asymptotic_expansion(ff, x2)
        err2 = abs(asym2 / exact2 - 1)

        # Error ratio should be ~(x1/x2)^4 = (1/2)^4 = 1/16
        # Allow generous margin: ratio < 1/8 (factor of 8 improvement minimum)
        if err2 > 0:
            ratio = err2 / err1
            assert ratio < 1.0 / 8.0, (
                f"{ff}: error ratio {ratio:.4f} > 1/8, not O(1/x^4) convergence"
            )
