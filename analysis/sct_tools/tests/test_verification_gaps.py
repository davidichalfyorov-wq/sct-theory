"""
Tests for all previously untested methods in sct_tools.verification.

Covers:
    1.  Verifier.check_value_mp     (mpmath high-precision checks)
    2.  Verifier.check_dimensions   (dimensional analysis)
    3.  Verifier.check_limit        (asymptotic convergence)
    4.  Verifier.check_symmetry     (symmetry properties)
    5.  Verifier.check_pole_cancellation (residue checks)
    6.  Verifier.check_literature   (literature comparison)
    7.  Verifier.check_with_uncertainties (error-bar checks)
    8.  Verifier.summary()          (report + return value)
    9.  Verifier.all_passed         (property)
    10. verify_form_factor_limits() (regression test)
    11. verify_uv_asymptotics()     (regression test)
    12. run_all_checks()            (full regression suite)
    13. check_numerical_stability() (stability scan)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools.verification import (
    Verifier,
    check_numerical_stability,
    run_all_checks,
    verify_form_factor_limits,
    verify_uv_asymptotics,
)

# =========================================================================
# 1. check_value_mp
# =========================================================================

class TestCheckValueMp:
    def test_matching_values_pass(self):
        from mpmath import mpf
        v = Verifier("test_mp_pass", quiet=True)
        result = v.check_value_mp(
            "pi match", mpf("3.14159265358979"), mpf("3.14159265358979"),
            tol_digits=10,
        )
        assert result is True
        assert v.n_pass == 1
        assert v.n_fail == 0

    def test_different_values_fail(self):
        from mpmath import mpf
        v = Verifier("test_mp_fail", quiet=True)
        # Relative error |3.14 - 3.15|/3.15 ~ 3.2e-3, so tol_digits=3 (tol=1e-3)
        # requires better than 0.1% agreement, which 3.14 vs 3.15 does not satisfy.
        result = v.check_value_mp(
            "pi mismatch", mpf("3.14"), mpf("3.15"), tol_digits=3,
        )
        assert result is False
        assert v.n_fail == 1

    def test_zero_expected(self):
        from mpmath import mpf
        v = Verifier("test_mp_zero", quiet=True)
        # Exact zero should pass
        assert v.check_value_mp("zero", mpf("0"), mpf("0"), tol_digits=50)
        # Tiny nonzero vs zero should fail for high precision
        assert not v.check_value_mp(
            "near-zero", mpf("1e-10"), mpf("0"), tol_digits=20,
        )

    def test_high_precision(self):
        from mpmath import mp, pi
        old_dps = mp.dps
        mp.dps = 120
        try:
            v = Verifier("test_mp_highprec", quiet=True)
            # Two values agreeing to 100+ digits
            val = pi
            assert v.check_value_mp("pi 100-digit", val, val, tol_digits=100)
        finally:
            mp.dps = old_dps


# =========================================================================
# 2. check_dimensions
# =========================================================================

class TestCheckDimensions:
    def test_matching_dimensions_pass(self):
        v = Verifier("test_dim", quiet=True)
        # Action is dimensionless (dim=0)
        assert v.check_dimensions("action", 0, 0)
        assert v.n_pass == 1

    def test_mismatched_dimensions_fail(self):
        v = Verifier("test_dim_fail", quiet=True)
        # Length has dim 1, but expected 0
        assert not v.check_dimensions("length", 1, 0)
        assert v.n_fail == 1

    def test_nonzero_expected(self):
        v = Verifier("test_dim_nonzero", quiet=True)
        # Energy has mass dimension 1
        assert v.check_dimensions("energy", 1, 1)
        # Curvature tensor has mass dimension 2
        assert v.check_dimensions("curvature", 2, 2)
        assert v.n_pass == 2
        assert v.n_fail == 0

    def test_negative_dimension(self):
        v = Verifier("test_dim_neg", quiet=True)
        # Propagator has mass dimension -2
        assert v.check_dimensions("propagator", -2, -2)
        assert not v.check_dimensions("propagator_wrong", -2, -1)


# =========================================================================
# 3. check_limit
# =========================================================================

class TestCheckLimit:
    def test_converging_to_zero(self):
        v = Verifier("test_limit_zero", quiet=True)
        x = [10, 50, 100, 500, 1000]
        y = [1.0 / xi for xi in x]  # -> 0
        assert v.check_limit("1/x -> 0", x, y, 0)

    def test_converging_to_nonzero(self):
        v = Verifier("test_limit_nonzero", quiet=True)
        x = [10, 50, 100, 500, 1000]
        y = [1.0 + 1.0 / xi for xi in x]  # -> 1
        assert v.check_limit("1+1/x -> 1", x, y, 1)

    def test_diverging_fails(self):
        v = Verifier("test_limit_diverge", quiet=True)
        x = [10, 50, 100, 500, 1000]
        y = [float(xi) for xi in x]  # diverges
        assert not v.check_limit("x -> 0?", x, y, 0)

    def test_wrong_target_fails(self):
        v = Verifier("test_limit_wrong", quiet=True)
        x = [10, 50, 100, 500, 1000]
        y = [1.0 + 1.0 / xi for xi in x]  # -> 1, not 2
        assert not v.check_limit("1+1/x -> 2?", x, y, 2)

    def test_negative_target(self):
        v = Verifier("test_limit_neg", quiet=True)
        x = [10, 50, 100, 500, 1000]
        y = [-1.0 / 6.0 + 1.0 / xi for xi in x]  # -> -1/6
        assert v.check_limit("approach -1/6", x, y, -1.0 / 6.0)


# =========================================================================
# 4. check_symmetry
# =========================================================================

class TestCheckSymmetry:
    def test_even_function_pass(self):
        v = Verifier("test_sym_even", quiet=True)

        def f_even(x):
            return x**2

        assert v.check_symmetry(
            "x^2 even", f_even, 3.0, lambda x: -x, "equal",
        )

    def test_odd_function_pass(self):
        v = Verifier("test_sym_odd", quiet=True)

        def f_odd(x):
            return x**3

        assert v.check_symmetry(
            "x^3 odd", f_odd, 2.0, lambda x: -x, "negative",
        )

    def test_non_symmetric_fail(self):
        v = Verifier("test_sym_fail", quiet=True)

        def f_asym(x):
            return x**2 + x  # not even

        assert not v.check_symmetry(
            "x^2+x not even", f_asym, 2.0, lambda x: -x, "equal",
        )

    def test_cosine_even(self):
        v = Verifier("test_sym_cos", quiet=True)
        assert v.check_symmetry(
            "cos even", np.cos, 1.5, lambda x: -x, "equal",
        )

    def test_sine_odd(self):
        v = Verifier("test_sym_sin", quiet=True)
        assert v.check_symmetry(
            "sin odd", np.sin, 1.5, lambda x: -x, "negative",
        )


# =========================================================================
# 5. check_pole_cancellation
# =========================================================================

class TestCheckPoleCancellation:
    def test_zero_residue_pass(self):
        v = Verifier("test_pole_zero", quiet=True)
        assert v.check_pole_cancellation("exact zero", 0.0)
        assert v.n_pass == 1

    def test_tiny_residue_pass(self):
        v = Verifier("test_pole_tiny", quiet=True)
        assert v.check_pole_cancellation("near zero", 1e-16)

    def test_nonzero_residue_fail(self):
        v = Verifier("test_pole_nonzero", quiet=True)
        assert not v.check_pole_cancellation("nonzero residue", 0.1)
        assert v.n_fail == 1

    def test_custom_atol(self):
        v = Verifier("test_pole_atol", quiet=True)
        # Value that passes with loose tolerance
        assert v.check_pole_cancellation("loose", 1e-8, atol=1e-6)
        # Same value fails with tight tolerance
        assert not v.check_pole_cancellation("tight", 1e-8, atol=1e-10)


# =========================================================================
# 6. check_literature
# =========================================================================

class TestCheckLiterature:
    def test_matching_literature_pass(self):
        v = Verifier("test_lit", quiet=True)
        result = v.check_literature(
            "beta_W^(0)",
            1.0 / 120,
            "hep-th/0306138",
            "4.3",
            1.0 / 120,
        )
        assert result is True
        assert v.n_pass == 1

    def test_mismatched_literature_fail(self):
        v = Verifier("test_lit_fail", quiet=True)
        result = v.check_literature(
            "wrong value",
            0.5,
            "hep-th/0306138",
            "4.3",
            1.0 / 120,
        )
        assert result is False
        assert v.n_fail == 1

    def test_zero_literature_value(self):
        v = Verifier("test_lit_zero", quiet=True)
        # h_R^(1/2)(0) = 0 is a known result
        assert v.check_literature(
            "h_R^(1/2)(0) = 0", 0.0, "hep-th/0306138", "4.5", 0.0,
        )
        # Nonzero vs zero should fail
        assert not v.check_literature(
            "nonzero vs 0", 0.1, "hep-th/0306138", "4.5", 0.0,
        )

    def test_custom_rtol(self):
        v = Verifier("test_lit_rtol", quiet=True)
        # Loose tolerance passes
        assert v.check_literature(
            "loose", 1.01, "test/1234", "1.1", 1.0, rtol=0.1,
        )
        # Tight tolerance fails
        assert not v.check_literature(
            "tight", 1.01, "test/1234", "1.1", 1.0, rtol=1e-6,
        )


# =========================================================================
# 7. check_with_uncertainties
# =========================================================================

class TestCheckWithUncertainties:
    def test_within_error_bars_pass(self):
        from uncertainties import ufloat
        v = Verifier("test_unc_pass", quiet=True)
        result = v.check_with_uncertainties(
            "within 3 sigma", ufloat(1.0, 0.1), 1.05, n_sigma=3,
        )
        assert result is True

    def test_outside_error_bars_fail(self):
        from uncertainties import ufloat
        v = Verifier("test_unc_fail", quiet=True)
        result = v.check_with_uncertainties(
            "outside 3 sigma", ufloat(1.0, 0.01), 2.0, n_sigma=3,
        )
        assert result is False

    def test_tuple_input(self):
        v = Verifier("test_unc_tuple", quiet=True)
        # (value, error) tuple form
        assert v.check_with_uncertainties(
            "tuple form", (1.0, 0.1), 1.05, n_sigma=3,
        )

    def test_exact_match(self):
        from uncertainties import ufloat
        v = Verifier("test_unc_exact", quiet=True)
        assert v.check_with_uncertainties(
            "exact", ufloat(1.0, 0.1), 1.0, n_sigma=1,
        )

    def test_zero_error(self):
        from uncertainties import ufloat
        v = Verifier("test_unc_zero_err", quiet=True)
        # Zero error, exact match
        assert v.check_with_uncertainties(
            "zero err match", ufloat(1.0, 0.0), 1.0,
        )
        # Zero error, mismatch
        assert not v.check_with_uncertainties(
            "zero err mismatch", ufloat(1.0, 0.0), 2.0,
        )

    def test_n_sigma_boundary(self):
        from uncertainties import ufloat
        v = Verifier("test_unc_boundary", quiet=True)
        # Exactly at n_sigma boundary: diff/sigma = 2.0 < 3 => PASS
        assert v.check_with_uncertainties(
            "at boundary", ufloat(1.0, 0.5), 2.0, n_sigma=3,
        )
        # diff/sigma = 2.0, n_sigma=1 => FAIL
        assert not v.check_with_uncertainties(
            "above boundary", ufloat(1.0, 0.5), 2.0, n_sigma=1,
        )


# =========================================================================
# 8. summary()
# =========================================================================

class TestSummary:
    def test_all_pass_returns_true(self, capsys):
        v = Verifier("test_summary_ok", quiet=True)
        v.check_value("a", 1.0, 1.0)
        v.check_value("b", 2.0, 2.0)
        result = v.summary()
        assert result is True
        captured = capsys.readouterr()
        assert "ALL CHECKS PASSED" in captured.out

    def test_some_fail_returns_false(self, capsys):
        v = Verifier("test_summary_fail", quiet=True)
        v.check_value("a", 1.0, 1.0)
        v.check_value("b", 1.0, 2.0)  # FAIL
        result = v.summary()
        assert result is False
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_empty_verifier(self, capsys):
        v = Verifier("test_summary_empty", quiet=True)
        result = v.summary()
        # No checks => 0 failures => returns True
        assert result is True

    def test_summary_counts(self, capsys):
        v = Verifier("test_counts", quiet=True)
        v.check_value("pass1", 1.0, 1.0)
        v.check_value("pass2", 2.0, 2.0)
        v.check_value("fail1", 1.0, 999.0)
        v.summary()
        captured = capsys.readouterr()
        assert "PASS:          2" in captured.out
        assert "FAIL:          1" in captured.out


# =========================================================================
# 9. all_passed property
# =========================================================================

class TestAllPassed:
    def test_true_when_all_pass(self):
        v = Verifier("test_ap_true", quiet=True)
        v.check_value("a", 1.0, 1.0)
        v.check_value("b", 2.0, 2.0)
        assert v.all_passed is True

    def test_false_when_some_fail(self):
        v = Verifier("test_ap_false", quiet=True)
        v.check_value("a", 1.0, 1.0)
        v.check_value("b", 1.0, 2.0)
        assert v.all_passed is False

    def test_true_when_empty(self):
        v = Verifier("test_ap_empty", quiet=True)
        assert v.all_passed is True

    def test_false_when_all_fail(self):
        v = Verifier("test_ap_all_fail", quiet=True)
        v.check_value("a", 1.0, 999.0)
        v.check_value("b", 2.0, 999.0)
        assert v.all_passed is False


# =========================================================================
# 10. verify_form_factor_limits()
# =========================================================================

class TestVerifyFormFactorLimits:
    def test_returns_passing_verifier(self):
        v = verify_form_factor_limits()
        assert isinstance(v, Verifier)
        assert v.all_passed is True
        assert v.n_pass > 0


# =========================================================================
# 11. verify_uv_asymptotics()
# =========================================================================

class TestVerifyUvAsymptotics:
    def test_returns_passing_verifier(self):
        v = verify_uv_asymptotics()
        assert isinstance(v, Verifier)
        assert v.all_passed is True
        assert v.n_pass > 0


# =========================================================================
# 12. run_all_checks()
# =========================================================================

class TestRunAllChecks:
    def test_returns_true(self):
        result = run_all_checks()
        assert result is True


# =========================================================================
# 13. check_numerical_stability()
# =========================================================================

class TestCheckNumericalStability:
    def test_stable_function(self, capsys):
        # Use a smooth, slowly-varying function with dense sampling
        # to avoid false positives from the jump detector (rtol_consecutive=0.1)
        result = check_numerical_stability(
            lambda x: np.exp(-x / 100.0), np.linspace(0, 10, 50),
            label="exp(-x/100)",
        )
        assert result['stable'] is True
        assert len(result['issues']) == 0
        assert len(result['values']) == 50

    def test_function_with_nan(self, capsys):
        def bad_func(x):
            if x > 5:
                return float('nan')
            return x

        result = check_numerical_stability(
            bad_func, np.linspace(0, 10, 20), label="bad",
        )
        assert result['stable'] is False
        assert any("NaN" in desc for _, desc in result['issues'])

    def test_function_with_overflow(self, capsys):
        def overflow_func(x):
            return np.exp(x)

        x_vals = np.linspace(0, 800, 20)
        result = check_numerical_stability(
            overflow_func, x_vals, label="exp overflow",
        )
        assert result['stable'] is False

    def test_function_with_exception(self, capsys):
        def raising_func(x):
            if x > 3:
                raise ZeroDivisionError("boom")
            return x

        result = check_numerical_stability(
            raising_func, np.linspace(0, 10, 20), label="raises",
        )
        assert result['stable'] is False
        assert any("Exception" in desc for _, desc in result['issues'])

    def test_jump_detection(self, capsys):
        def jumpy(x):
            return 1.0 if x < 5 else 1000.0

        x_vals = np.linspace(1, 10, 20)
        result = check_numerical_stability(
            jumpy, x_vals, label="jump", rtol_consecutive=0.1,
        )
        assert result['stable'] is False
        assert any("Jump" in desc for _, desc in result['issues'])

    def test_form_factor_stability(self, capsys):
        """Verify that hC_scalar is numerically stable across wide range.

        Uses dense linear sampling within each decade to avoid false positives
        from the natural 1/x decay of the form factor on logspace grids.
        """
        from sct_tools import form_factors as ff
        x_vals = np.linspace(0.01, 100, 500)
        result = check_numerical_stability(
            ff.hC_scalar, x_vals, label="hC_scalar",
        )
        assert result['stable'] is True
