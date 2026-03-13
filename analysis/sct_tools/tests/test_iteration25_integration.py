"""
Tests for Iteration 25: cross-module integration and constants hardening.

Covers:
    I1: BETA_W[s] == |hC_spin(0)| for all 3 spins (constants ↔ form_factors)
    I2: BETA_R matches hR(0) for Dirac and vector (constants ↔ form_factors)
    I3: beta_R_scalar(xi) == hR_scalar(0, xi) (constants ↔ form_factors)
    I4: F1_total/F2_total at x=0 match hand-computed values
    I5: check_dimensions input validation (constants)
    I6: NaturalUnits round-trip consistency (constants)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import constants, form_factors

# ============================================================================
# I1: BETA_W[s] == |hC_spin(0)| for all 3 spins
# ============================================================================


class TestBetaWConsistency:
    """BETA_W dict in constants must match form_factors h_C(0) values."""

    def test_scalar_beta_w(self):
        """BETA_W[0] = hC_scalar(0) = 1/120."""
        assert abs(form_factors.hC_scalar(0)) == pytest.approx(
            float(constants.BETA_W[0]), rel=1e-12
        )

    def test_dirac_beta_w(self):
        """BETA_W[0.5] = |hC_dirac(0)| = 1/20."""
        assert abs(form_factors.hC_dirac(0)) == pytest.approx(
            float(constants.BETA_W[0.5]), rel=1e-12
        )

    def test_vector_beta_w(self):
        """BETA_W[1] = |hC_vector(0)| = 1/10."""
        assert abs(form_factors.hC_vector(0)) == pytest.approx(
            float(constants.BETA_W[1]), rel=1e-12
        )

    def test_scalar_sign_positive(self):
        """hC_scalar(0) > 0 in our convention."""
        assert form_factors.hC_scalar(0) > 0

    def test_dirac_sign_negative(self):
        """hC_dirac(0) < 0 in our convention."""
        assert form_factors.hC_dirac(0) < 0

    def test_vector_sign_positive(self):
        """hC_vector(0) > 0 (physical = unconstrained - 2*ghost)."""
        assert form_factors.hC_vector(0) > 0


# ============================================================================
# I2: BETA_R matches hR(0) for Dirac and vector
# ============================================================================


class TestBetaRConsistency:
    """BETA_R dict values match form_factors h_R(0) at appropriate couplings."""

    def test_dirac_beta_r(self):
        """BETA_R[0.5] = hR_dirac(0) = 0 (conformal)."""
        assert form_factors.hR_dirac(0) == pytest.approx(
            constants.BETA_R[0.5], abs=1e-14
        )

    def test_vector_beta_r(self):
        """BETA_R[1] = hR_vector(0) = 0 (conformal)."""
        assert form_factors.hR_vector(0) == pytest.approx(
            constants.BETA_R[1], abs=1e-14
        )


# ============================================================================
# I3: beta_R_scalar(xi) == hR_scalar(0, xi)
# ============================================================================


class TestBetaRScalarConsistency:
    """beta_R_scalar(xi) in constants must match hR_scalar(0, xi) from form_factors."""

    @pytest.mark.parametrize("xi", [0.0, 1/6, 0.5, 1.0, -1.0])
    def test_beta_r_scalar_matches_form_factor(self, xi):
        """beta_R_scalar(xi) = hR_scalar(0, xi) for various xi."""
        from_constants = constants.beta_R_scalar(xi)
        from_form_factors = form_factors.hR_scalar(0, xi=xi)
        assert from_form_factors == pytest.approx(from_constants, abs=1e-12)


# ============================================================================
# I4: F1_total/F2_total at x=0 match hand-computed values
# ============================================================================


class TestTotalFormFactorsAtZero:
    """F1_total(0) and F2_total(0) match analytic SM values."""

    def test_f1_total_at_zero(self):
        """F1_total(0) = (N_s*hC_s(0) + (N_f/2)*hC_d(0) + N_v*hC_v(0)) / (16*pi^2)."""
        # Phase 3 CORRECTED: N_f/2 Dirac fermions
        expected = (
            4 * form_factors.hC_scalar(0)
            + 22.5 * form_factors.hC_dirac(0)
            + 12 * form_factors.hC_vector(0)
        ) / (16 * np.pi**2)
        result = form_factors.F1_total(0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_f2_total_at_zero_conformal_scalar(self):
        """F2_total(0, xi=1/6) = 0 (conformal scalar + conformal Dirac + conformal vector)."""
        # At xi=1/6: hR_scalar(0) = 0, hR_dirac(0) = 0, hR_vector(0) = 0
        # So F2_total(0) = 0
        result = form_factors.F2_total(0, xi=1.0/6.0)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_f2_total_at_zero_minimal_scalar(self):
        """F2_total(0, xi=0) has non-zero contribution from scalar."""
        # Only scalar contributes: hR_scalar(0, xi=0) = (1/2)(0 - 1/6)^2 = 1/72
        expected = 4 * (1/72) / (16 * np.pi**2)  # N_s=4, only scalar
        result = form_factors.F2_total(0, xi=0.0)
        assert result == pytest.approx(expected, rel=1e-10)


# ============================================================================
# I5: check_dimensions input validation
# ============================================================================


class TestCheckDimensionsValidation:
    """check_dimensions rejects non-numeric inputs."""

    def test_string_expr_dim_rejected(self):
        """String expr_dim raises TypeError."""
        with pytest.raises(TypeError, match="expr_dim must be numeric"):
            constants.check_dimensions("hello", 1)

    def test_string_expected_dim_rejected(self):
        """String expected_dim raises TypeError."""
        with pytest.raises(TypeError, match="expected_dim must be numeric"):
            constants.check_dimensions(1, "world")

    def test_none_rejected(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            constants.check_dimensions(None, 1)

    def test_matching_dims_pass(self):
        """Matching integer dims return True."""
        assert constants.check_dimensions(2, 2) is True

    def test_mismatching_dims_raise(self):
        """Mismatching dims raise ValueError."""
        with pytest.raises(ValueError, match="Dimension mismatch"):
            constants.check_dimensions(2, 3)

    def test_float_dims_accepted(self):
        """Float dims are accepted (for fractional mass dimensions)."""
        assert constants.check_dimensions(1.5, 1.5) is True

    def test_numpy_int_accepted(self):
        """numpy integer accepted."""
        assert constants.check_dimensions(np.int64(2), 2) is True


# ============================================================================
# I6: NaturalUnits round-trip consistency
# ============================================================================


class TestNaturalUnitsConsistency:
    """NaturalUnits conversions should be self-consistent."""

    def test_planck_mass_to_kg(self):
        """M_Pl in GeV → kg should give ~2.18e-8 kg."""
        with constants.NaturalUnits() as nu:
            m_kg = nu.to_kg(nu.M_Pl)
        # Planck mass ≈ 2.176e-8 kg
        assert m_kg == pytest.approx(2.176e-8, rel=0.01)

    def test_planck_length_to_meters(self):
        """1/M_Pl in GeV^-1 → meters should give ~1.6e-35 m."""
        with constants.NaturalUnits() as nu:
            l_m = nu.to_meters(1.0 / nu.M_Pl)
        # Planck length ≈ 1.616e-35 m
        assert l_m == pytest.approx(1.616e-35, rel=0.01)

    def test_cross_section_barn(self):
        """Known conversion: 1 GeV^{-2} ≈ 0.389 mb = 3.89e8 pb."""
        with constants.NaturalUnits() as nu:
            sigma_pb = nu.to_pb(1.0)
        # 1 GeV^{-2} ≈ 0.389 mb = 3.894e8 pb
        assert sigma_pb == pytest.approx(3.894e8, rel=0.01)

    def test_temperature_conversion(self):
        """1 MeV ≈ 1.16e10 K."""
        with constants.NaturalUnits() as nu:
            T_K = nu.to_kelvin(1e-3)  # 1 MeV = 1e-3 GeV
        assert T_K == pytest.approx(1.16e10, rel=0.01)


# ============================================================================
# I7: BETA_R dict completeness
# ============================================================================


class TestBetaRCompleteness:
    """BETA_R dict should have entries for spin 0.5 and spin 1."""

    def test_dirac_entry_exists(self):
        assert 0.5 in constants.BETA_R

    def test_vector_entry_exists(self):
        assert 1 in constants.BETA_R

    def test_values_are_zero(self):
        """Both Dirac and vector have conformal beta_R = 0."""
        assert constants.BETA_R[0.5] == 0
        assert constants.BETA_R[1] == 0
