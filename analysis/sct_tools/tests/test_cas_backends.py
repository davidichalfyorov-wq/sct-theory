"""Tests for sct_tools.cas_backends — Triple CAS cross-verification engine."""

import numpy as np
import pytest

from sct_tools import cas_backends


class TestBackendDetection:
    def test_check_backends_returns_dict(self):
        result = cas_backends.check_backends()
        assert isinstance(result, dict)
        assert 'sympy' in result
        assert 'ginac' in result
        assert 'mpmath' in result
        assert 'all_available' in result

    def test_sympy_available(self):
        """SymPy is a core dependency — must always be available."""
        assert cas_backends._SYMPY_OK is True

    def test_mpmath_available(self):
        """mpmath is a core dependency — must always be available."""
        assert cas_backends._MPMATH_OK is True


class TestCASResult:
    def test_dataclass_creation(self):
        r = cas_backends.CASResult(label="test", agree=True, max_reldiff=1e-15)
        assert r.label == "test"
        assert r.agree is True
        assert r.max_reldiff == pytest.approx(1e-15)

    def test_repr_agree(self):
        r = cas_backends.CASResult(label="phi(1)", agree=True, max_reldiff=1e-14)
        assert "AGREE" in repr(r)

    def test_repr_disagree(self):
        r = cas_backends.CASResult(label="phi(1)", agree=False, max_reldiff=0.1)
        assert "DISAGREE" in repr(r)

    def test_default_errors_list(self):
        r = cas_backends.CASResult(label="test")
        assert r.errors == []


class TestSympyHelpers:
    def test_sympy_phi_at_zero(self):
        val = cas_backends._sympy_phi(0)
        assert val == pytest.approx(1.0, abs=1e-12)

    def test_sympy_phi_positive(self):
        val = cas_backends._sympy_phi(1.0)
        assert val > 0

    def test_sympy_phi_decreasing(self):
        v1 = cas_backends._sympy_phi(0.5)
        v2 = cas_backends._sympy_phi(5.0)
        assert v1 > v2

    def test_sympy_polylog_li2_known(self):
        """Li_2(1) = pi^2/6."""
        val = cas_backends._sympy_polylog(2, 1.0)
        assert val == pytest.approx(np.pi**2 / 6, rel=1e-10)


class TestMpmathHelpers:
    def test_mpmath_phi_at_zero(self):
        val = cas_backends._mpmath_phi(0)
        assert val == pytest.approx(1.0, abs=1e-12)

    def test_mpmath_phi_positive(self):
        val = cas_backends._mpmath_phi(1.0)
        assert val > 0

    def test_mpmath_polylog_li2_known(self):
        """Li_2(1) = pi^2/6."""
        val = cas_backends._mpmath_polylog(2, 1.0)
        assert val == pytest.approx(np.pi**2 / 6, rel=1e-10)


class TestGiNaCHelpers:
    @pytest.mark.skipif(not cas_backends._GINAC_OK,
                        reason="ginacsympy not installed")
    def test_ginac_phi_at_zero(self):
        val = cas_backends._ginac_phi(0)
        assert val == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.skipif(not cas_backends._GINAC_OK,
                        reason="ginacsympy not installed")
    def test_ginac_phi_positive(self):
        val = cas_backends._ginac_phi(1.0)
        assert val > 0

    @pytest.mark.skipif(not cas_backends._GINAC_OK,
                        reason="ginacsympy not installed")
    def test_ginac_eval_rational(self):
        val = cas_backends._ginac_eval_rational("1/3")
        assert val == pytest.approx(1/3, rel=1e-12)

    @pytest.mark.skipif(not cas_backends._GINAC_OK,
                        reason="ginacsympy not installed")
    def test_ginac_polylog_li2(self):
        val = cas_backends._ginac_polylog(2, 0.5)
        # Li_2(1/2) = pi^2/12 - ln(2)^2/2
        expected = np.pi**2 / 12 - np.log(2)**2 / 2
        assert val == pytest.approx(expected, rel=1e-8)


class TestSymPyMpmathCrossCheck:
    """Cross-check SymPy vs mpmath (both always available)."""

    def test_phi_agreement(self):
        for x in [0.0, 0.1, 1.0, 5.0, 20.0]:
            s = cas_backends._sympy_phi(x)
            m = cas_backends._mpmath_phi(x)
            assert s == pytest.approx(m, rel=1e-10), (
                f"phi({x}): sympy={s}, mpmath={m}"
            )

    def test_polylog_agreement(self):
        for x in [0.1, 0.5, 0.9]:
            s = cas_backends._sympy_polylog(2, x)
            m = cas_backends._mpmath_polylog(2, x)
            assert s == pytest.approx(m, rel=1e-10), (
                f"Li_2({x}): sympy={s}, mpmath={m}"
            )


class TestTripleCASComparison:
    """Test the _compare method of TripleCAS."""

    def test_compare_all_agree(self):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        result = tc._compare("test", 1.0, 1.0, 1.0)
        assert result.agree is True
        assert result.max_reldiff < 1e-10

    def test_compare_disagreement(self):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        result = tc._compare("test", 1.0, 1.0, 2.0)
        assert result.agree is False

    def test_compare_missing_backend(self):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        result = tc._compare("test", 1.0, None, 1.0)
        assert "ginac: evaluation failed" in result.errors

    def test_compare_two_agree(self):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        result = tc._compare("test", 1.0, None, 1.0 + 1e-15)
        assert result.agree is True


class TestTripleCASEvalPhi:
    """Test TripleCAS.eval_phi() using at minimum SymPy + mpmath."""

    def test_eval_phi_at_zero(self):
        tc = cas_backends.TripleCAS(tol_digits=8, require_all=False)
        result = tc.eval_phi(0)
        assert result.sympy_val == pytest.approx(1.0, abs=1e-12)
        assert result.mpmath_val == pytest.approx(1.0, abs=1e-12)

    def test_eval_phi_agree_at_1(self):
        tc = cas_backends.TripleCAS(tol_digits=8, require_all=False)
        result = tc.eval_phi(1.0)
        # At least SymPy and mpmath must agree
        assert result.sympy_val == pytest.approx(result.mpmath_val, rel=1e-8)

    def test_eval_phi_at_large_x(self):
        tc = cas_backends.TripleCAS(tol_digits=6, require_all=False)
        result = tc.eval_phi(50.0)
        assert result.sympy_val > 0
        assert result.mpmath_val > 0


class TestTripleCASEvalRational:
    def test_rational_one_third(self):
        tc = cas_backends.TripleCAS(tol_digits=12, require_all=False)
        result = tc.eval_rational("1/3", 1, 3)
        assert result.agree is True

    def test_rational_beta_W_scalar(self):
        tc = cas_backends.TripleCAS(tol_digits=12, require_all=False)
        result = tc.eval_rational("beta_W^(0)", 1, 120)
        assert result.agree is True
        assert result.sympy_val == pytest.approx(1/120, rel=1e-14)


class TestTripleCASFormFactors:
    """Test form factor verification at specific points."""

    def test_scalar_hC_at_zero(self):
        tc = cas_backends.TripleCAS(tol_digits=6, require_all=False)
        result = tc.verify_form_factor_at(0, spin=0, component='C')
        assert result.sympy_val == pytest.approx(1/120, rel=1e-8)

    def test_dirac_hC_at_zero(self):
        tc = cas_backends.TripleCAS(tol_digits=6, require_all=False)
        result = tc.verify_form_factor_at(0, spin=0.5, component='C')
        assert result.sympy_val == pytest.approx(-1/20, rel=1e-8)

    def test_vector_hC_at_zero(self):
        tc = cas_backends.TripleCAS(tol_digits=6, require_all=False)
        result = tc.verify_form_factor_at(0, spin=1, component='C')
        assert result.sympy_val == pytest.approx(1/10, rel=1e-8)

    def test_verify_all_form_factors_returns_6(self):
        tc = cas_backends.TripleCAS(tol_digits=6, require_all=False)
        results = tc.verify_all_form_factors(x=1.0)
        assert len(results) == 6  # 3 spins × 2 components

    def test_verify_beta_coefficients_returns_6(self):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        results = tc.verify_beta_coefficients()
        assert len(results) == 6


class TestTripleCASEvalExpression:
    def test_custom_expression(self):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        result = tc.eval_expression(
            "sqrt(2)",
            sympy_func=lambda: float(__import__('sympy').sqrt(2).evalf(50)),
            ginac_func=lambda: np.sqrt(2),
            mpmath_func=lambda: float(__import__('mpmath').sqrt(2)),
        )
        assert result.agree is True
        assert result.max_reldiff < 1e-10


class TestTripleCASummary:
    def test_summary_all_pass(self, capsys):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        results = [
            cas_backends.CASResult(label="test1", agree=True, max_reldiff=1e-15),
            cas_backends.CASResult(label="test2", agree=True, max_reldiff=1e-14),
        ]
        ok = tc.summary(results)
        assert ok is True
        captured = capsys.readouterr()
        assert "ALL TRIPLE-CAS CHECKS PASSED" in captured.out

    def test_summary_with_failure(self, capsys):
        tc = cas_backends.TripleCAS(tol_digits=10, require_all=False)
        results = [
            cas_backends.CASResult(label="test1", agree=True, max_reldiff=1e-15),
            cas_backends.CASResult(label="test2", agree=False, max_reldiff=0.5,
                                   sympy_val=1.0, ginac_val=1.5, mpmath_val=1.0),
        ]
        ok = tc.summary(results)
        assert ok is False
        captured = capsys.readouterr()
        assert "DISAGREE" in captured.out
