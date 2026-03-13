"""
Pytest regression tests for sct_tools.form_factors.

WARNING — ANTI-CIRCULARITY RULE:
    These tests check the LIBRARY against its own documented values.
    They are REGRESSION TESTS, not derivation verification.
    For derivation verification, use custom crosscheck scripts.
"""

import os
import sys

import numpy as np
import pytest

# Add analysis/ to path so sct_tools can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors as ff

# =============================================================================
# MASTER FUNCTION phi(x)
# =============================================================================

class TestPhi:
    def test_phi_zero(self):
        assert ff.phi(0) == pytest.approx(1.0, abs=1e-14)

    def test_phi_positive(self):
        # phi(x) > 0 for all real x
        for x in [0.01, 0.1, 1, 10, 100]:
            assert ff.phi(x) > 0

    def test_phi_decreasing(self):
        # phi(x) is decreasing for x > 0
        vals = [ff.phi(x) for x in [0, 1, 5, 10, 50]]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]

    def test_phi_closed_form(self):
        # phi_closed should agree with phi (quad) to ~14 digits
        for x in [0.1, 1, 5, 20]:
            assert ff.phi(x) == pytest.approx(ff.phi_closed(x), rel=1e-12)


# =============================================================================
# SCALAR FORM FACTORS
# =============================================================================

class TestScalar:
    def test_hC_scalar_local_limit(self):
        assert ff.hC_scalar(0) == pytest.approx(1 / 120, rel=1e-14)

    def test_hR_scalar_minimal(self):
        assert ff.hR_scalar(0, xi=0) == pytest.approx(1 / 72, rel=1e-14)

    def test_hR_scalar_conformal(self):
        assert ff.hR_scalar(0, xi=1 / 6) == pytest.approx(0, abs=1e-14)

    def test_hR_scalar_xi1(self):
        assert ff.hR_scalar(0, xi=1) == pytest.approx(25 / 72, rel=1e-14)

    def test_hC_scalar_positive_x(self):
        # h_C^(0)(x) > 0 for x > 0
        for x in [0.1, 1, 10]:
            assert ff.hC_scalar(x) > 0

    def test_hC_scalar_uv_decay(self):
        # x * h_C^(0)(x) -> 1/12 as x -> inf (slow convergence)
        assert 5000 * ff.hC_scalar(5000) == pytest.approx(1 / 12, rel=0.01)


# =============================================================================
# DIRAC FORM FACTORS
# =============================================================================

class TestDirac:
    def test_hC_dirac_local_limit(self):
        assert ff.hC_dirac(0) == pytest.approx(-1 / 20, rel=1e-14)

    def test_hR_dirac_local_limit(self):
        assert ff.hR_dirac(0) == pytest.approx(0, abs=1e-14)

    def test_hC_dirac_uv_decay(self):
        # x * h_C^(1/2)(x) -> -1/6 as x -> inf
        assert 1000 * ff.hC_dirac(1000) == pytest.approx(-1 / 6, rel=0.01)

    def test_hR_dirac_uv_decay(self):
        # x * h_R^(1/2)(x) -> 1/18 as x -> inf (slow convergence)
        assert 5000 * ff.hR_dirac(5000) == pytest.approx(1 / 18, rel=0.01)


# =============================================================================
# CZ FORM FACTORS
# =============================================================================

class TestCZ:
    def test_f_Ric_local(self):
        assert ff.f_Ric(0) == pytest.approx(1 / 60, rel=1e-14)

    def test_f_R_local(self):
        assert ff.f_R(0) == pytest.approx(1 / 120, rel=1e-14)

    def test_f_RU_local(self):
        assert ff.f_RU(0) == pytest.approx(-1 / 6, rel=1e-14)

    def test_f_U_local(self):
        assert ff.f_U(0) == pytest.approx(1 / 2, rel=1e-14)

    def test_f_Omega_local(self):
        assert ff.f_Omega(0) == pytest.approx(1 / 12, rel=1e-14)


# =============================================================================
# VECTOR FORM FACTORS (placeholder)
# =============================================================================

class TestVector:
    def test_hC_vector_local_limit(self):
        """B1: h_C^(1)(0) = 1/10"""
        assert ff.hC_vector(0) == pytest.approx(1 / 10, abs=1e-12)

    def test_hR_vector_local_limit(self):
        """B2: h_R^(1)(0) = 0"""
        assert ff.hR_vector(0) == pytest.approx(0, abs=1e-12)

    def test_hC_vector_asymptotic(self):
        """B5: h_C^(1)(x) ~ -1/(3x) as x -> inf"""
        x = 1000.0
        assert ff.hC_vector(x) == pytest.approx(-1 / (3 * x), rel=0.01)

    def test_hR_vector_asymptotic(self):
        """B6: h_R^(1)(x) ~ 1/(9x) as x -> inf"""
        x = 1000.0
        assert ff.hR_vector(x) == pytest.approx(1 / (9 * x), rel=0.01)

    def test_hC_vector_negative_x_raises(self):
        with pytest.raises(ValueError):
            ff.hC_vector(-1)

    def test_hR_vector_negative_x_raises(self):
        with pytest.raises(ValueError):
            ff.hR_vector(-1)

    def test_hC_vector_fast_matches(self):
        """_fast variant matches standard at moderate x"""
        x = 5.0
        assert ff.hC_vector_fast(x) == pytest.approx(ff.hC_vector(x), rel=1e-10)

    def test_hR_vector_fast_matches(self):
        """_fast variant matches standard at moderate x"""
        x = 5.0
        assert ff.hR_vector_fast(x) == pytest.approx(ff.hR_vector(x), rel=1e-10)


# =============================================================================
# HIGH-PRECISION (mpmath)
# =============================================================================

class TestMpmath:
    def test_phi_mp_zero(self):
        from mpmath import mpf
        assert ff.phi_mp(0) == mpf(1)

    def test_hC_scalar_mp_limit(self):
        from mpmath import mp, mpf
        old_dps = mp.dps
        mp.dps = 100
        try:
            assert abs(ff.hC_scalar_mp(0) - mpf(1) / 120) < mpf(10)**(-90)
        finally:
            mp.dps = old_dps

    def test_hC_dirac_mp_limit(self):
        from mpmath import mp, mpf
        old_dps = mp.dps
        mp.dps = 100
        try:
            assert abs(ff.hC_dirac_mp(0) - mpf(-1) / 20) < mpf(10)**(-90)
        finally:
            mp.dps = old_dps

    def test_hR_dirac_mp_limit(self):
        from mpmath import mpf
        assert abs(ff.hR_dirac_mp(0)) < mpf(10)**(-90)

    def test_mp_agrees_with_numpy(self):
        # mpmath and numpy versions should agree to ~14 digits
        for x in [0.5, 2.0, 10.0]:
            np_val = ff.hC_scalar(x)
            mp_val = float(ff.hC_scalar_mp(x, dps=50))
            assert np_val == pytest.approx(mp_val, rel=1e-12)


# =============================================================================
# TAYLOR SERIES
# =============================================================================

class TestTaylor:
    def test_taylor_hC_agrees_small_x(self):
        # Taylor should agree with direct for small x
        for x in [0.01, 0.1, 0.5]:
            taylor = float(ff.hC_scalar_taylor(x))
            direct = float(ff.hC_scalar_mp(x, dps=50))
            assert taylor == pytest.approx(direct, rel=1e-10)

    def test_taylor_hR_agrees_small_x(self):
        for x in [0.01, 0.1]:
            for xi in [0, 1 / 6, 1]:
                taylor = float(ff.hR_scalar_taylor(x, xi=xi))
                direct = float(ff.hR_scalar_mp(x, xi=xi, dps=50))
                assert taylor == pytest.approx(direct, rel=1e-8)


# =============================================================================
# COMBINED (incomplete until vector done)
# =============================================================================

class TestAsymptoticExpansion:
    def test_hC_scalar_sign(self):
        # For large x, asymptotic should be close to exact
        x = 1000.0
        asymp = ff.asymptotic_expansion('hC_scalar', x)
        exact = ff.hC_scalar(x)
        assert asymp == pytest.approx(exact, rel=0.05)

    def test_hC_scalar_asymptotic_positive(self):
        # Leading term 1/(12x) dominates and is positive for large x
        x = 100.0
        assert ff.asymptotic_expansion('hC_scalar', x) > 0

    def test_hC_dirac_asymptotic(self):
        x = 1000.0
        asymp = ff.asymptotic_expansion('hC_dirac', x)
        exact = ff.hC_dirac(x)
        assert asymp == pytest.approx(exact, rel=0.05)

    def test_hR_dirac_asymptotic(self):
        x = 1000.0
        asymp = ff.asymptotic_expansion('hR_dirac', x)
        exact = ff.hR_dirac(x)
        assert asymp == pytest.approx(exact, rel=0.05)

    def test_invalid_form_factor(self):
        with pytest.raises(ValueError, match="not available"):
            ff.asymptotic_expansion('nonexistent', 100.0)


class TestCombined:
    def test_F1_total_at_zero(self):
        """F1_total(0) = alpha_C / (16*pi^2) = 13/(1920*pi^2)."""
        val = ff.F1_total(0)
        # Phase 3 CORRECTED: N_f/2 Dirac fermions (not N_f)
        # F1 = [N_s*h_C^(0)(0) + (N_f/2)*h_C^(1/2)(0) + N_v*h_C^(1)(0)] / (16*pi^2)
        # = [4/120 + 22.5*(-1/20) + 12/10] / (16*pi^2)
        # = [4/120 - 135/120 + 144/120] / (16*pi^2)
        # = [13/120] / (16*pi^2)  (positive: scalars+vectors outweigh fermions)
        expected = (4 / 120 + 22.5 * (-1 / 20) + 12 / 10) / (16 * np.pi**2)
        assert val == pytest.approx(expected, rel=1e-10)

    def test_F2_total_at_zero(self):
        """F2_total(0) should return a finite number."""
        val = ff.F2_total(0)
        # h_R^(0)(0,xi=0) = (1/2)(0-1/6)^2 = 1/72 (not zero!)
        # h_R^(1/2)(0) = 0, h_R^(1)(0) = 0
        # So F2(0, xi=0) = 4*(1/72) / (16*pi^2)
        expected = (4 / 72) / (16 * np.pi**2)
        assert val == pytest.approx(expected, rel=1e-10)

    def test_F1_total_x1_crosscheck(self):
        """F1_total(1) against mpmath 50-digit reference value."""
        val = ff.F1_total(1.0)
        # mpmath reference (Phase 3 CORRECTED with N_f/2):
        # (4*hC_s_mp + 22.5*hC_d_mp + 12*hC_v_mp)/(16*pi^2)
        expected = -3.182959362479449630e-04
        assert val == pytest.approx(expected, rel=1e-10)

    def test_F2_total_x1_crosscheck(self):
        """F2_total(1) against mpmath 50-digit reference value."""
        val = ff.F2_total(1.0)
        # mpmath reference (Phase 3 CORRECTED with N_f/2):
        # (4*hR_s_mp(xi=0) + 22.5*hR_d_mp + 12*hR_v_mp)/(16*pi^2)
        expected = 4.312611841226462e-04
        assert val == pytest.approx(expected, rel=1e-10)
