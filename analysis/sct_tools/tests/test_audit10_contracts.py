"""Round 10 audit: API contract consistency tests.

Tests that all form factor functions follow uniform guard patterns:
- x < 0 rejection (physics: x = Box/Lambda^2 >= 0)
- hR_scalar direct x validation (not delegated to sub-functions)
- f_U uses converted x (not raw input)
- dphi_dx_fast x < 0 guard before Taylor early-return
"""

import numpy as np
import pytest

from sct_tools import form_factors as ff

# ============================================================================
# Base form factors: x < 0 rejection
# ============================================================================


class TestBaseFormFactorsRejectNegativeX:
    """hC_scalar, hC_dirac, hR_dirac must reject x < 0."""

    def test_hC_scalar_negative(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_scalar(-1.0)

    def test_hC_scalar_negative_small(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_scalar(-1e-15)

    def test_hC_dirac_negative(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_dirac(-1.0)

    def test_hC_dirac_negative_small(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hC_dirac(-1e-15)

    def test_hR_dirac_negative(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hR_dirac(-1.0)

    def test_hR_dirac_negative_small(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.hR_dirac(-1e-15)

    def test_hC_scalar_zero_ok(self):
        """x=0 should still work (Taylor branch)."""
        result = ff.hC_scalar(0.0)
        assert result == pytest.approx(1.0 / 120, rel=1e-10)

    def test_hC_dirac_zero_ok(self):
        result = ff.hC_dirac(0.0)
        assert result == pytest.approx(-1.0 / 20, rel=1e-10)

    def test_hR_dirac_zero_ok(self):
        result = ff.hR_dirac(0.0)
        assert result == pytest.approx(0.0, abs=1e-14)


# ============================================================================
# hR_scalar: direct x validation
# ============================================================================


class TestHRScalarDirectValidation:
    """hR_scalar must validate x directly, not via sub-functions."""

    def test_negative_x_rejected(self):
        with pytest.raises(ValueError, match="hR_scalar: requires x >= 0"):
            ff.hR_scalar(-1.0)

    def test_negative_x_small_rejected(self):
        with pytest.raises(ValueError, match="hR_scalar: requires x >= 0"):
            ff.hR_scalar(-1e-15)

    def test_nan_x_rejected(self):
        with pytest.raises(ValueError, match="hR_scalar: requires finite x"):
            ff.hR_scalar(float("nan"))

    def test_inf_x_rejected(self):
        with pytest.raises(ValueError, match="hR_scalar: requires finite x"):
            ff.hR_scalar(float("inf"))

    def test_error_message_names_hR_scalar(self):
        """Error message must name hR_scalar, not a sub-function."""
        try:
            ff.hR_scalar(-1.0)
        except ValueError as e:
            assert "hR_scalar" in str(e)
        else:
            pytest.fail("hR_scalar(-1.0) did not raise ValueError")

    def test_valid_x_still_works(self):
        """Positive x, default xi=0."""
        result = ff.hR_scalar(1.0)
        assert np.isfinite(result)

    def test_valid_x_with_xi(self):
        """Positive x with non-zero xi."""
        result = ff.hR_scalar(1.0, xi=1.0 / 6)
        assert np.isfinite(result)


# ============================================================================
# dphi_dx_fast: x < 0 guard before Taylor early-return
# ============================================================================


class TestDphiDxFastNegativeGuard:
    """dphi_dx_fast must reject x < 0 BEFORE the abs(x) <= 1e-12 check."""

    def test_negative_large(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.dphi_dx_fast(-1.0)

    def test_negative_tiny(self):
        """This was the Round 10 CRITICAL bug: -1e-13 hit Taylor branch."""
        with pytest.raises(ValueError, match="x >= 0"):
            ff.dphi_dx_fast(-1e-13)

    def test_negative_at_threshold(self):
        """Exactly -1e-12 must be caught by x < 0 guard, not Taylor."""
        with pytest.raises(ValueError, match="x >= 0"):
            ff.dphi_dx_fast(-1e-12)

    def test_zero_still_works(self):
        assert ff.dphi_dx_fast(0.0) == pytest.approx(-1.0 / 6, rel=1e-14)

    def test_positive_tiny_still_works(self):
        result = ff.dphi_dx_fast(1e-13)
        assert result == pytest.approx(-1.0 / 6, rel=1e-6)


# ============================================================================
# f_U: converted x passed to phi()
# ============================================================================


class TestFUConvertedX:
    """f_U must use float-converted x consistently."""

    def test_integer_input(self):
        """Integer x should be handled cleanly via float conversion."""
        result = ff.f_U(1)
        assert np.isfinite(result)

    def test_numpy_scalar(self):
        """np.float64 input works."""
        result = ff.f_U(np.float64(2.0))
        assert np.isfinite(result)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="f_U: requires finite x"):
            ff.f_U(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="f_U: requires finite x"):
            ff.f_U(float("inf"))

    def test_value_correctness(self):
        """f_U(x) = phi(x)/2."""
        x = 5.0
        expected = ff.phi(x) / 2
        assert ff.f_U(x) == pytest.approx(expected, rel=1e-12)


# ============================================================================
# Guard pattern uniformity: all form factor families reject x < 0
# ============================================================================


class TestUniformNegativeXRejection:
    """ALL form factor functions must reject x < 0.

    This is a comprehensive sweep: base (quad), _fast, _mp, _taylor.
    The _fast/_mp/_taylor variants were already tested in Rounds 4+7;
    this tests that base functions now match.
    """

    @pytest.mark.parametrize("func_name", [
        "hC_scalar", "hR_dirac", "hC_dirac",
        "hC_scalar_fast", "hR_scalar_fast", "hC_dirac_fast", "hR_dirac_fast",
    ])
    def test_negative_x_raises(self, func_name):
        func = getattr(ff, func_name)
        with pytest.raises(ValueError):
            func(-1.0)

    def test_hR_scalar_negative_x_raises(self):
        """hR_scalar has xi parameter, test separately."""
        with pytest.raises(ValueError):
            ff.hR_scalar(-1.0, xi=0.0)

    def test_hR_scalar_fast_negative_x_raises(self):
        with pytest.raises(ValueError):
            ff.hR_scalar_fast(-1.0, xi=0.0)

    @pytest.mark.parametrize("func_name", [
        "dphi_dx_fast", "dhC_scalar_dx", "dhR_scalar_dx",
    ])
    def test_derivative_negative_x_raises(self, func_name):
        func = getattr(ff, func_name)
        with pytest.raises(ValueError):
            func(-1.0)
