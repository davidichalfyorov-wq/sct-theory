"""
Tests for Iteration 36: constants.py + tensors.py hardening.

Covers:
    C-06/C-07: check_dimensions NaN/float robustness
    C-04: beta_R_scalar NaN/Inf validation
    T-05/T-06: metric functions symbols length validation
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import constants, tensors

# ============================================================================
# C-06/C-07: check_dimensions NaN/float robustness
# ============================================================================


class TestCheckDimensionsRobustness:
    """check_dimensions must handle NaN and float near-miss correctly."""

    def test_nan_expr_dim_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            constants.check_dimensions(float('nan'), 2)

    def test_nan_expected_dim_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            constants.check_dimensions(2, float('nan'))

    def test_nan_np_float_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            constants.check_dimensions(np.float64('nan'), 2)

    def test_float_near_miss_passes(self):
        """2.0 + 1e-15 should match 2 (float arithmetic artifact)."""
        assert constants.check_dimensions(2.0 + 1e-15, 2) is True

    def test_float_exact_match(self):
        assert constants.check_dimensions(2.0, 2) is True

    def test_integer_mismatch_still_raises(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            constants.check_dimensions(3, 2)

    def test_label_in_nan_error(self):
        with pytest.raises(ValueError, match="test_field"):
            constants.check_dimensions(float('nan'), 2, label="test_field")


# ============================================================================
# C-04: beta_R_scalar NaN/Inf validation
# ============================================================================


class TestBetaRScalarValidation:
    """beta_R_scalar must reject NaN, Inf, non-numeric xi."""

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            constants.beta_R_scalar(float('nan'))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            constants.beta_R_scalar(float('inf'))

    def test_string_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            constants.beta_R_scalar("0.5")

    def test_conformal_coupling(self):
        """xi = 1/6 gives beta_R = 0 (conformal coupling)."""
        assert constants.beta_R_scalar(1 / 6) == pytest.approx(0.0, abs=1e-15)

    def test_minimal_coupling(self):
        """xi = 0 gives beta_R = (1/2)(1/6)^2 = 1/72."""
        assert constants.beta_R_scalar(0) == pytest.approx(1 / 72, rel=1e-10)


# ============================================================================
# T-05/T-06: metric functions symbols length validation
# ============================================================================


class TestMetricSymbolsValidation:
    """Metric functions must reject symbols tuple of wrong length."""

    def test_schwarzschild_wrong_symbols_length(self):
        import sympy as sp
        with pytest.raises(ValueError, match="4 elements"):
            tensors.schwarzschild(symbols=(sp.Symbol('t'), sp.Symbol('r'), sp.Symbol('x')))

    def test_flrw_wrong_symbols_length(self):
        import sympy as sp
        with pytest.raises(ValueError, match="4 elements"):
            tensors.flrw(symbols=(sp.Symbol('t'), sp.Symbol('r')))

    def test_kerr_wrong_symbols_length(self):
        import sympy as sp
        with pytest.raises(ValueError, match="4 elements"):
            tensors.kerr(symbols=(sp.Symbol('t'),))

    def test_schwarzschild_default_symbols_works(self):
        metric, params = tensors.schwarzschild()
        assert params['M'] is not None
        assert len(params['symbols']) == 4

    def test_flrw_default_symbols_works(self):
        metric, params = tensors.flrw()
        assert params['a'] is not None
        assert len(params['symbols']) == 4

    def test_kerr_default_symbols_works(self):
        metric, params = tensors.kerr()
        assert params['M'] is not None
        assert len(params['symbols']) == 4
