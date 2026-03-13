"""
Tests for Iteration 44: F1_total/F2_total input validation,
scan_* array-level guards, asymptotic_expansion docstring coverage.
"""

import numpy as np
import pytest

from sct_tools import form_factors as ff

# ============================================================================
# F1_total input validation
# ============================================================================


class TestF1TotalValidation:
    """F1_total must reject invalid x, N_s, N_f, N_v, xi."""

    def test_nan_x(self):
        with pytest.raises(ValueError, match="^F1_total: requires finite x"):
            ff.F1_total(float("nan"))

    def test_inf_x(self):
        with pytest.raises(ValueError, match="^F1_total: requires finite x"):
            ff.F1_total(float("inf"))

    def test_negative_x(self):
        with pytest.raises(ValueError, match="^F1_total: requires x >= 0"):
            ff.F1_total(-1.0)

    def test_nan_N_s(self):
        with pytest.raises(ValueError, match="^F1_total: requires finite N_s"):
            ff.F1_total(1.0, N_s=float("nan"))

    def test_nan_N_f(self):
        with pytest.raises(ValueError, match="^F1_total: requires finite N_f"):
            ff.F1_total(1.0, N_f=float("nan"))

    def test_nan_N_v(self):
        with pytest.raises(ValueError, match="^F1_total: requires finite N_v"):
            ff.F1_total(1.0, N_v=float("nan"))

    def test_nan_xi(self):
        with pytest.raises(ValueError, match="^F1_total: requires finite xi"):
            ff.F1_total(1.0, xi=float("nan"))

    def test_valid_returns_finite(self):
        result = ff.F1_total(1.0)
        assert np.isfinite(result)

    def test_zero_x_works(self):
        result = ff.F1_total(0.0)
        assert np.isfinite(result)


# ============================================================================
# F2_total input validation
# ============================================================================


class TestF2TotalValidation:
    """F2_total must reject invalid x, N_s, N_f, N_v, xi."""

    def test_nan_x(self):
        with pytest.raises(ValueError, match="^F2_total: requires finite x"):
            ff.F2_total(float("nan"))

    def test_inf_x(self):
        with pytest.raises(ValueError, match="^F2_total: requires finite x"):
            ff.F2_total(float("inf"))

    def test_negative_x(self):
        with pytest.raises(ValueError, match="^F2_total: requires x >= 0"):
            ff.F2_total(-1.0)

    def test_nan_N_s(self):
        with pytest.raises(ValueError, match="^F2_total: requires finite N_s"):
            ff.F2_total(1.0, N_s=float("nan"))

    def test_inf_N_v(self):
        with pytest.raises(ValueError, match="^F2_total: requires finite N_v"):
            ff.F2_total(1.0, N_v=float("inf"))

    def test_nan_xi(self):
        with pytest.raises(ValueError, match="^F2_total: requires finite xi"):
            ff.F2_total(1.0, xi=float("nan"))

    def test_valid_returns_finite(self):
        result = ff.F2_total(1.0)
        assert np.isfinite(result)


# ============================================================================
# scan_* array-level NaN/inf/negative guards
# ============================================================================


class TestScanArrayGuards:
    """All scan_* functions must reject NaN/inf/negative at array level."""

    _nan_arr = np.array([1.0, float("nan"), 3.0])
    _inf_arr = np.array([1.0, float("inf"), 3.0])
    _neg_arr = np.array([1.0, -0.5, 3.0])

    # --- scan_hC_scalar ---
    def test_scan_hC_scalar_nan(self):
        with pytest.raises(ValueError, match="^scan_hC_scalar: received NaN"):
            ff.scan_hC_scalar(self._nan_arr)

    def test_scan_hC_scalar_inf(self):
        with pytest.raises(ValueError, match="^scan_hC_scalar: received NaN"):
            ff.scan_hC_scalar(self._inf_arr)

    def test_scan_hC_scalar_neg(self):
        with pytest.raises(ValueError, match="^scan_hC_scalar: requires all x >= 0"):
            ff.scan_hC_scalar(self._neg_arr)

    # --- scan_hR_scalar ---
    def test_scan_hR_scalar_nan(self):
        with pytest.raises(ValueError, match="^scan_hR_scalar: received NaN"):
            ff.scan_hR_scalar(self._nan_arr)

    def test_scan_hR_scalar_neg(self):
        with pytest.raises(ValueError, match="^scan_hR_scalar: requires all x >= 0"):
            ff.scan_hR_scalar(self._neg_arr)

    # --- scan_hC_dirac ---
    def test_scan_hC_dirac_nan(self):
        with pytest.raises(ValueError, match="^scan_hC_dirac: received NaN"):
            ff.scan_hC_dirac(self._nan_arr)

    def test_scan_hC_dirac_neg(self):
        with pytest.raises(ValueError, match="^scan_hC_dirac: requires all x >= 0"):
            ff.scan_hC_dirac(self._neg_arr)

    # --- scan_hR_dirac ---
    def test_scan_hR_dirac_nan(self):
        with pytest.raises(ValueError, match="^scan_hR_dirac: received NaN"):
            ff.scan_hR_dirac(self._nan_arr)

    def test_scan_hR_dirac_neg(self):
        with pytest.raises(ValueError, match="^scan_hR_dirac: requires all x >= 0"):
            ff.scan_hR_dirac(self._neg_arr)

    # --- scan_hC_vector ---
    def test_scan_hC_vector_nan(self):
        with pytest.raises(ValueError, match="^scan_hC_vector: received NaN"):
            ff.scan_hC_vector(self._nan_arr)

    def test_scan_hC_vector_neg(self):
        with pytest.raises(ValueError, match="^scan_hC_vector: requires all x >= 0"):
            ff.scan_hC_vector(self._neg_arr)

    # --- scan_hR_vector ---
    def test_scan_hR_vector_nan(self):
        with pytest.raises(ValueError, match="^scan_hR_vector: received NaN"):
            ff.scan_hR_vector(self._nan_arr)

    def test_scan_hR_vector_neg(self):
        with pytest.raises(ValueError, match="^scan_hR_vector: requires all x >= 0"):
            ff.scan_hR_vector(self._neg_arr)

    # --- empty arrays still pass ---
    def test_scan_empty_arrays_pass(self):
        empty = np.array([])
        assert len(ff.scan_hC_scalar(empty)) == 0
        assert len(ff.scan_hR_scalar(empty)) == 0
        assert len(ff.scan_hC_dirac(empty)) == 0
        assert len(ff.scan_hR_dirac(empty)) == 0
        assert len(ff.scan_hC_vector(empty)) == 0
        assert len(ff.scan_hR_vector(empty)) == 0


# ============================================================================
# asymptotic_expansion: vector form factors are supported
# ============================================================================


class TestAsymptoticExpansionVector:
    """asymptotic_expansion must support hC_vector and hR_vector."""

    def test_hC_vector_large_x(self):
        x = 500.0
        approx_val = ff.asymptotic_expansion("hC_vector", x)
        exact_val = ff.hC_vector_fast(x)
        assert approx_val == pytest.approx(exact_val, rel=0.05)

    def test_hR_vector_large_x(self):
        x = 500.0
        approx_val = ff.asymptotic_expansion("hR_vector", x)
        exact_val = ff.hR_vector_fast(x)
        assert approx_val == pytest.approx(exact_val, rel=0.05)

    def test_invalid_form_factor(self):
        with pytest.raises(ValueError, match="^asymptotic_expansion: form factor"):
            ff.asymptotic_expansion("bogus", 100.0)
