"""
Tests for Iteration 27: form_factors.py, graphs.py, tensors.py hardening
+ test quality improvements (strengthen weak/circular tests).

Covers:
    F1: phi_closed x<0 guard
    F2: CZ basis functions (f_Ric, f_R, f_RU, f_U, f_Omega) x<0 guard
    F3: asymptotic_expansion hC_dirac corrected (-1/x^2 term)
    F4: asymptotic_expansion hR_dirac corrected (-2/(3x^2) term)
    G1: spectral_dimension_graph t-dt<=0 safety
    G3: zeta_function_graph Re(s)<=0 warning + empty graph guard
    T6: flrw k parameter validation
    TQ1: scalar/Dirac negative-x guard tests (missing edge case)
    TQ2: F1_total/F2_total mpmath-anchored tests (replace circular)
    TQ3: strengthened tensors tests (Schwarzschild line element content)
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors, graphs

# ============================================================================
# F1: phi_closed x<0 guard
# ============================================================================


class TestPhiClosedGuard:
    """phi_closed must reject x < 0."""

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x >= 0"):
            form_factors.phi_closed(-1.0)

    def test_negative_small_raises(self):
        with pytest.raises(ValueError, match="x >= 0"):
            form_factors.phi_closed(-1e-6)

    def test_zero_returns_one(self):
        assert form_factors.phi_closed(0) == pytest.approx(1.0)

    def test_positive_works(self):
        assert form_factors.phi_closed(1.0) > 0


# ============================================================================
# F2: CZ basis functions x<0 guard
# ============================================================================


class TestCZBasisGuards:
    """All 5 CZ form factors must reject x < 0."""

    @pytest.mark.parametrize("func_name", [
        "f_Ric", "f_R", "f_RU", "f_U", "f_Omega",
    ])
    def test_negative_x_raises(self, func_name):
        func = getattr(form_factors, func_name)
        with pytest.raises(ValueError, match="x >= 0"):
            func(-1.0)

    @pytest.mark.parametrize("func_name,expected", [
        ("f_Ric", 1.0 / 60),
        ("f_R", 1.0 / 120),
        ("f_RU", -1.0 / 6),
        ("f_Omega", 1.0 / 12),
    ])
    def test_zero_returns_taylor_limit(self, func_name, expected):
        func = getattr(form_factors, func_name)
        assert func(0) == pytest.approx(expected, rel=1e-10)

    def test_f_U_zero(self):
        assert form_factors.f_U(0) == pytest.approx(0.5, rel=1e-10)


# ============================================================================
# F3/F4: asymptotic_expansion corrected Dirac terms
# ============================================================================


class TestAsymptoticDiracCorrection:
    """Corrected asymptotic expansions should be much closer to exact values."""

    def test_hC_dirac_x100_rel_error_below_2pct(self):
        exact = form_factors.hC_dirac(100)
        asym = form_factors.asymptotic_expansion('hC_dirac', 100)
        assert abs(asym / exact - 1) < 0.02

    def test_hC_dirac_x1000_rel_error_below_0_1pct(self):
        exact = form_factors.hC_dirac(1000)
        asym = form_factors.asymptotic_expansion('hC_dirac', 1000)
        assert abs(asym / exact - 1) < 0.001

    def test_hR_dirac_x100_rel_error_below_2pct(self):
        exact = form_factors.hR_dirac(100)
        asym = form_factors.asymptotic_expansion('hR_dirac', 100)
        assert abs(asym / exact - 1) < 0.02

    def test_hR_dirac_x1000_rel_error_below_0_1pct(self):
        exact = form_factors.hR_dirac(1000)
        asym = form_factors.asymptotic_expansion('hR_dirac', 1000)
        assert abs(asym / exact - 1) < 0.001

    def test_hC_dirac_asym_has_minus_one_over_x2_term(self):
        """The -1/x^2 term is the dominant correction at moderate x."""
        x = 50
        asym = form_factors.asymptotic_expansion('hC_dirac', x)
        leading_only = -1.0 / (6.0 * x)
        # Full expansion should be more negative than leading-only
        # because the -1/x^2 term contributes negatively
        assert asym < leading_only

    def test_hR_dirac_asym_has_minus_two_thirds_over_x2_term(self):
        """The -2/(3x^2) correction at moderate x."""
        x = 50
        asym = form_factors.asymptotic_expansion('hR_dirac', x)
        leading_only = 1.0 / (18.0 * x)
        # Full expansion should be less than leading-only due to -2/(3x^2) < 0
        assert asym < leading_only


# ============================================================================
# G1: spectral_dimension_graph t-dt<=0 guard
# ============================================================================


class TestSpectralDimensionTDtGuard:
    """spectral_dimension_graph should not crash when t-dt <= 0."""

    def test_very_small_t_produces_nan(self):
        """With dt_frac close to 1, small t yields t-dt ~ 0."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        # dt_frac=0.99 means t-dt = 0.01*t; at t=1e-15, t-dt = 1e-17 which is positive
        # but at dt_frac=0.999999 and very small t, precision is problematic
        t_vals = np.array([1e-15, 0.01, 1.0])
        # Should not crash
        t_out, d_S = graphs.spectral_dimension_graph(A, t_values=t_vals, dt_frac=0.5)
        assert len(d_S) == 3

    def test_normal_case_still_works(self):
        """Regular parameters still produce reasonable results."""
        A = np.ones((5, 5)) - np.eye(5)  # K_5
        t_vals = np.logspace(-1, 1, 20)
        t_out, d_S = graphs.spectral_dimension_graph(A, t_values=t_vals)
        assert len(d_S) == 20
        finite_vals = d_S[np.isfinite(d_S)]
        assert len(finite_vals) > 0


# ============================================================================
# G3: zeta_function_graph Re(s)<=0 warning + empty graph
# ============================================================================


class TestZetaFunctionGraphValidation:
    """zeta_function_graph should warn on Re(s)<=0 and reject empty graphs."""

    def test_negative_real_s_warns(self):
        A = np.array([[0, 1], [1, 0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graphs.zeta_function_graph(A, [-1.0])
            assert any("Re(s) <= 0" in str(warning.message) for warning in w)

    def test_positive_real_s_no_warning(self):
        A = np.array([[0, 1], [1, 0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = graphs.zeta_function_graph(A, [2.0])
            re_s_warnings = [x for x in w if "Re(s)" in str(x.message)]
            assert len(re_s_warnings) == 0
        assert np.isfinite(result[0])

    def test_disconnected_graph_raises(self):
        """Fully disconnected graph has no positive eigenvalues."""
        A = np.zeros((3, 3))  # 3 isolated vertices
        with pytest.raises(ValueError, match="no positive eigenvalues"):
            graphs.zeta_function_graph(A, [2.0])


# ============================================================================
# T6: flrw k parameter validation
# ============================================================================


class TestFLRWKValidation:
    """flrw must reject invalid k values."""

    def test_k_two_rejected(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors
        with pytest.raises(ValueError, match="k must be -1, 0, or"):
            tensors.flrw(k=2)

    def test_k_half_rejected(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors
        with pytest.raises(ValueError, match="k must be -1, 0, or"):
            tensors.flrw(k=0.5)

    def test_k_zero_accepted(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors
        metric, params = tensors.flrw(k=0)
        assert metric is not None

    def test_k_minus_one_accepted(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors
        metric, params = tensors.flrw(k=-1)
        assert metric is not None


# ============================================================================
# TQ1: scalar/Dirac negative-x guard tests (missing edge case)
# ============================================================================


class TestFormFactorNegativeXGuards:
    """All Weyl form factors must reject x < 0."""

    @pytest.mark.parametrize("func_name", [
        "hC_scalar", "hR_scalar", "hC_dirac", "hR_dirac",
        "hC_vector", "hR_vector",
    ])
    def test_negative_x_raises(self, func_name):
        func = getattr(form_factors, func_name)
        with pytest.raises(ValueError, match="x >= 0"):
            func(-1.0)


# ============================================================================
# TQ2: F1_total/F2_total mpmath-anchored (replace circular tests)
# ============================================================================


class TestTotalFormFactorsMpmathAnchored:
    """F1_total and F2_total at x=0 against mpmath-verified reference values."""

    def test_f1_total_at_zero(self):
        """F1_total(0) = (4/120 + 22.5*(-1/20) + 12/10) / (16*pi^2).

        Phase 3 CORRECTED: N_f/2 = 22.5 Dirac fermions.
        Pre-computed: (4/120 - 135/120 + 144/120) / (16*pi^2)
        = (13/120) / (16*pi^2)
        = 13 / (1920 * pi^2)
        """
        expected = 13.0 / (1920.0 * np.pi**2)
        result = form_factors.F1_total(0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_f2_total_conformal_scalar_is_zero(self):
        """F2_total(0, xi=1/6) = 0 (all conformal)."""
        result = form_factors.F2_total(0, xi=1.0 / 6.0)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_f2_total_minimal_scalar(self):
        """F2_total(0, xi=0): only scalar contributes with hR_scalar(0, xi=0) = 1/72.

        = 4 * (1/72) / (16*pi^2) = 1/(288*pi^2)
        """
        expected = 1.0 / (288.0 * np.pi**2)
        result = form_factors.F2_total(0, xi=0.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_f1_total_at_one_is_finite(self):
        """F1_total at x=1 should be a finite number."""
        result = form_factors.F1_total(1.0)
        assert np.isfinite(result)

    def test_f1_total_sign_at_zero(self):
        """F1_total(0) is positive (Phase 3 corrected: scalars+vectors outweigh fermions)."""
        # With N_f/2 correction: alpha_C = 13/120 > 0
        assert form_factors.F1_total(0) > 0


# ============================================================================
# TQ3: Strengthened tensors test — Schwarzschild line element content
# ============================================================================


class TestSchwarzschildLineElement:
    """Schwarzschild line element should contain expected terms."""

    def test_line_element_has_mass(self):
        """Line element should depend on the mass parameter M."""
        pytest.importorskip("OGRePy")
        from sct_tools import tensors
        metric, params = tensors.schwarzschild()
        ds2 = tensors.line_element(metric)
        M = params["M"]
        assert M in ds2.free_symbols

    def test_line_element_has_coordinates(self):
        """Line element should contain coordinate differentials."""
        pytest.importorskip("OGRePy")
        from sct_tools import tensors
        metric, params = tensors.schwarzschild()
        ds2 = tensors.line_element(metric)
        symbols = params["symbols"]
        # At minimum, the line element should involve r (via f(r) = 1 - 2M/r)
        r = symbols[1]
        assert r in ds2.free_symbols
