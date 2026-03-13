"""
Tests for Iteration 37: form_factors.py + verification.py hardening.

Covers:
    FF-BUG2: hR_scalar_mp x==0 guard ordering (before phi_mp call)
    FF-BUG3: phi() rejects x < 0 (consistent with phi_fast/phi_vec)
    V-H1: check_value_mp NaN/Inf guard
    V-H2: check_value_mp tol_digits validation
    V-M1: check_with_uncertainties n_sigma validation
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors as ff
from sct_tools import verification

# ============================================================================
# FF-BUG3: phi() rejects x < 0
# ============================================================================


class TestPhiNegativeXGuard:
    """phi() must reject x < 0, consistent with phi_fast/phi_vec."""

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.phi(-1.0)

    def test_negative_small_raises(self):
        with pytest.raises(ValueError, match="x >= 0"):
            ff.phi(-1e-6)

    def test_zero_works(self):
        assert ff.phi(0) == pytest.approx(1.0, abs=1e-12)

    def test_positive_works(self):
        val = ff.phi(1.0)
        assert 0 < val < 1  # phi(x>0) < 1

    def test_consistent_with_phi_fast(self):
        """phi and phi_fast agree for x > 0."""
        for x in [0.1, 1.0, 5.0, 10.0]:
            assert ff.phi(x) == pytest.approx(ff.phi_fast(x), rel=1e-8)


# ============================================================================
# FF-BUG2: hR_scalar_mp x==0 guard ordering
# ============================================================================


class TestHRScalarMpZeroGuard:
    """hR_scalar_mp must return correct value at x=0 without unnecessary phi_mp call."""

    def test_x_zero_conformal(self):
        """xi=1/6 -> beta_R = 0."""
        from mpmath import mpf
        result = ff.hR_scalar_mp(0, xi=mpf(1) / 6, dps=50)
        assert float(abs(result)) < 1e-30

    def test_x_zero_minimal(self):
        """xi=0 -> (1/2)(1/6)^2 = 1/72."""
        from mpmath import mp, mpf
        with mp.workdps(50):
            result = ff.hR_scalar_mp(0, xi=0, dps=50)
            expected = mpf(1) / 72
            assert float(abs(result - expected)) < 1e-30

    def test_x_positive_still_works(self):
        """Normal case x > 0 unaffected."""
        result = ff.hR_scalar_mp(1.0, xi=0, dps=50)
        ref = ff.hR_scalar(1.0, xi=0)
        assert float(result) == pytest.approx(ref, rel=1e-10)


# ============================================================================
# V-H1: check_value_mp NaN/Inf guard
# ============================================================================


class TestCheckValueMpNanInf:
    """check_value_mp must handle NaN/Inf inputs gracefully."""

    def test_nan_computed_fails(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        result = v.check_value_mp("nan input", mpf('nan'), mpf(1))
        assert result is False

    def test_inf_computed_fails(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        result = v.check_value_mp("inf input", mpf('inf'), mpf(1))
        assert result is False

    def test_inf_expected_fails(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        result = v.check_value_mp("inf expected", mpf(1), mpf('inf'))
        assert result is False

    def test_normal_values_pass(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        result = v.check_value_mp("exact", mpf(1), mpf(1), tol_digits=30)
        assert result is True


# ============================================================================
# V-H2: check_value_mp tol_digits validation
# ============================================================================


class TestCheckValueMpTolDigits:
    """check_value_mp must reject non-positive tol_digits."""

    def test_zero_raises(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="positive integer"):
            v.check_value_mp("bad tol", mpf(1), mpf(1), tol_digits=0)

    def test_negative_raises(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="positive integer"):
            v.check_value_mp("bad tol", mpf(1), mpf(1), tol_digits=-5)

    def test_float_raises(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="positive integer"):
            v.check_value_mp("bad tol", mpf(1), mpf(1), tol_digits=3.5)

    def test_valid_tol_digits(self):
        from mpmath import mpf
        v = verification.Verifier("test", quiet=True)
        result = v.check_value_mp("ok", mpf("3.14159"), mpf("3.14159"), tol_digits=10)
        assert result is True


# ============================================================================
# V-M1: check_with_uncertainties n_sigma validation
# ============================================================================


class TestCheckWithUncertaintiesNSigma:
    """check_with_uncertainties must reject non-positive n_sigma."""

    def test_zero_raises(self):
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="n_sigma must be positive"):
            v.check_with_uncertainties("bad", (1.0, 0.1), 1.0, n_sigma=0)

    def test_negative_raises(self):
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="n_sigma must be positive"):
            v.check_with_uncertainties("bad", (1.0, 0.1), 1.0, n_sigma=-1)

    def test_normal_works(self):
        v = verification.Verifier("test", quiet=True)
        result = v.check_with_uncertainties("ok", (1.0, 0.1), 1.0, n_sigma=3)
        assert result is True
