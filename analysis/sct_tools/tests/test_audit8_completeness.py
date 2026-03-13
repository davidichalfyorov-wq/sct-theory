"""
Audit Round 8 — Completeness: Taylor, spectral, asymptotic, phi_vec Inf,
fitting NaN gaps, dhR_scalar_dx xi validation.

Tests all Round-8 fixes:
  - phi_vec rejects Inf (not just NaN)
  - hC_scalar_taylor / hR_scalar_taylor guard NaN/Inf/negative
  - F1_spectral / F2_spectral guard NaN (not just z<=0)
  - asymptotic_expansion guards NaN/Inf/non-positive
  - dhR_scalar_dx validates xi for NaN/Inf
  - model_comparison n_data NaN
  - likelihood_ratio_test df_diff NaN

Created: 2026-03-10
"""

import numpy as np
import pytest

from sct_tools import fitting
from sct_tools.form_factors import (
    F1_spectral,
    F2_spectral,
    asymptotic_expansion,
    dhR_scalar_dx,
    hC_scalar_taylor,
    hR_scalar_taylor,
    phi_vec,
)

# ============================================================================
# phi_vec: Inf rejection (was only catching NaN in Round 6)
# ============================================================================


class TestPhiVecInfRejection:
    """phi_vec must reject Inf, not just NaN."""

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            phi_vec([1.0, float("inf"), 2.0])

    def test_neg_inf_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            phi_vec([1.0, float("-inf"), 2.0])

    def test_nan_still_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            phi_vec([1.0, float("nan"), 2.0])

    def test_mixed_nan_inf_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            phi_vec([float("nan"), float("inf")])

    def test_valid_still_works(self):
        result = phi_vec([0.0, 0.5, 1.0, 10.0])
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        # phi(0) = 1 exactly
        assert result[0] == pytest.approx(1.0, abs=1e-12)


# ============================================================================
# hC_scalar_taylor: NaN/Inf/x<0 guards
# ============================================================================


class TestHCScalarTaylorGuards:
    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            hC_scalar_taylor(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            hC_scalar_taylor(float("inf"))

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x >= 0"):
            hC_scalar_taylor(-1.0)

    def test_valid_x0(self):
        """h_C^(0)(0) = 1/120 (beta_W^(0))."""
        result = float(hC_scalar_taylor(0.0, N=60))
        assert result == pytest.approx(1.0 / 120.0, rel=1e-10)

    def test_valid_x1(self):
        """Taylor at x=1 should match the known value."""
        result = float(hC_scalar_taylor(1.0, N=60))
        assert np.isfinite(result)
        # Cross-check: hC_scalar(1.0) via quad should be close
        from sct_tools.form_factors import hC_scalar
        ref = hC_scalar(1.0)
        assert result == pytest.approx(ref, rel=1e-6)


# ============================================================================
# hR_scalar_taylor: NaN/Inf/x<0/xi-NaN guards
# ============================================================================


class TestHRScalarTaylorGuards:
    def test_nan_x_raises(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar_taylor(float("nan"))

    def test_inf_x_raises(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar_taylor(float("inf"))

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x >= 0"):
            hR_scalar_taylor(-0.5)

    def test_nan_xi_raises(self):
        with pytest.raises(ValueError, match="finite xi"):
            hR_scalar_taylor(1.0, xi=float("nan"))

    def test_inf_xi_raises(self):
        with pytest.raises(ValueError, match="finite xi"):
            hR_scalar_taylor(1.0, xi=float("inf"))

    def test_valid_x0_xi0(self):
        """h_R^(0)(0; xi=0) = (1/2)(0 - 1/6)^2 = 1/72."""
        result = float(hR_scalar_taylor(0.0, xi=0, N=60))
        assert result == pytest.approx(1.0 / 72.0, rel=1e-10)

    def test_valid_conformal(self):
        """h_R^(0)(0; xi=1/6) = 0 (conformal invariance)."""
        result = float(hR_scalar_taylor(0.0, xi=1.0 / 6.0, N=60))
        assert abs(result) < 1e-14

    def test_valid_x1_crosscheck(self):
        """Taylor at x=1 should match quad-based hR_scalar."""
        from sct_tools.form_factors import hR_scalar
        result_taylor = float(hR_scalar_taylor(1.0, xi=0, N=60))
        result_quad = hR_scalar(1.0, xi=0)
        assert result_taylor == pytest.approx(result_quad, rel=1e-6)


# ============================================================================
# F1_spectral / F2_spectral: NaN guard (z<=0 doesn't catch NaN)
# ============================================================================

# Helper: exponential spectral function psi = e^{-u}
def _psi_exp(u):
    return np.exp(-u)


def _psi1_exp(u):
    return np.exp(-u)  # Psi_1 = int_u^inf e^{-v} dv = e^{-u}


def _psi2_exp(u):
    return np.exp(-u)  # Psi_2 = int_u^inf (v-u) e^{-v} dv = e^{-u}


class TestF1SpectralGuards:
    def test_nan_z_raises(self):
        with pytest.raises(ValueError, match="finite"):
            F1_spectral(float("nan"), _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_inf_z_raises(self):
        with pytest.raises(ValueError, match="finite"):
            F1_spectral(float("inf"), _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_neg_inf_z_raises(self):
        with pytest.raises(ValueError, match="finite|z > 0"):
            F1_spectral(float("-inf"), _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_neg_z_still_caught(self):
        with pytest.raises(ValueError, match="z > 0"):
            F1_spectral(-1.0, _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_valid(self):
        result = F1_spectral(1.0, _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)
        assert np.isfinite(result)


class TestF2SpectralGuards:
    def test_nan_z_raises(self):
        with pytest.raises(ValueError, match="finite"):
            F2_spectral(float("nan"), _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_inf_z_raises(self):
        with pytest.raises(ValueError, match="finite"):
            F2_spectral(float("inf"), _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_neg_z_still_caught(self):
        with pytest.raises(ValueError, match="z > 0"):
            F2_spectral(-1.0, _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)

    def test_valid(self):
        result = F2_spectral(1.0, _psi_exp, _psi1_exp, _psi2_exp, 1.0, 1.0)
        assert np.isfinite(result)


# ============================================================================
# asymptotic_expansion: NaN/Inf/x<=0 guards
# ============================================================================


class TestAsymptoticExpansionGuards:
    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            asymptotic_expansion("hC_scalar", float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            asymptotic_expansion("hC_scalar", float("inf"))

    def test_neg_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            asymptotic_expansion("hC_scalar", float("-inf"))

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="x > 0"):
            asymptotic_expansion("hC_scalar", 0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="x > 0"):
            asymptotic_expansion("hC_scalar", -1.0)

    def test_valid_hC_scalar(self):
        """Large x: hC_scalar asymptotic ~ 1/(12x)."""
        result = asymptotic_expansion("hC_scalar", 1e6)
        assert np.isfinite(result)
        # Leading term: 1/(12 * 1e6) ≈ 8.33e-8
        assert abs(result) < 1e-5

    def test_valid_hC_dirac(self):
        result = asymptotic_expansion("hC_dirac", 100.0)
        assert np.isfinite(result)

    def test_valid_hR_dirac(self):
        result = asymptotic_expansion("hR_dirac", 100.0)
        assert np.isfinite(result)

    def test_unknown_form_factor(self):
        with pytest.raises(ValueError, match="not available"):
            asymptotic_expansion("nonexistent", 10.0)


# ============================================================================
# dhR_scalar_dx: xi NaN/Inf validation
# ============================================================================


class TestDhRScalarDxXiGuards:
    def test_nan_xi_raises(self):
        with pytest.raises(ValueError, match="finite xi"):
            dhR_scalar_dx(1.0, xi=float("nan"))

    def test_inf_xi_raises(self):
        with pytest.raises(ValueError, match="finite xi"):
            dhR_scalar_dx(1.0, xi=float("inf"))

    def test_nan_xi_taylor_branch(self):
        """x < 2 uses Taylor branch — xi NaN must still be caught."""
        with pytest.raises(ValueError, match="finite xi"):
            dhR_scalar_dx(0.5, xi=float("nan"))

    def test_nan_xi_fd_branch(self):
        """x >= 2 uses finite-difference branch — xi NaN must still be caught."""
        with pytest.raises(ValueError, match="finite xi"):
            dhR_scalar_dx(5.0, xi=float("nan"))

    def test_valid_xi_0(self):
        result = dhR_scalar_dx(1.0, xi=0.0)
        assert np.isfinite(result)

    def test_valid_xi_conformal(self):
        result = dhR_scalar_dx(1.0, xi=1.0 / 6.0)
        assert np.isfinite(result)


# ============================================================================
# fitting.py: model_comparison n_data NaN + likelihood_ratio_test df_diff NaN
# ============================================================================


class TestModelComparisonNDataNaN:
    """Round 8: n_data NaN must be caught (was slipping through n_data <= 0)."""

    def test_nan_n_data(self):
        with pytest.raises(ValueError, match="finite"):
            fitting.model_comparison(10.0, 2, 8.0, 3, n_data=float("nan"))

    def test_inf_n_data(self):
        with pytest.raises(ValueError, match="finite"):
            fitting.model_comparison(10.0, 2, 8.0, 3, n_data=float("inf"))

    def test_negative_n_data_still_caught(self):
        with pytest.raises(ValueError, match="n_data >= 2"):
            fitting.model_comparison(10.0, 2, 8.0, 3, n_data=-5)

    def test_valid_still_works(self):
        result = fitting.model_comparison(10.0, 2, 8.0, 3, n_data=50)
        assert np.isfinite(result["dAIC"])


class TestLikelihoodRatioTestDfDiffNaN:
    """Round 8: df_diff NaN must be caught (was slipping through df_diff <= 0)."""

    def test_nan_df_diff(self):
        with pytest.raises(ValueError, match="finite"):
            fitting.likelihood_ratio_test(-100.0, -95.0, df_diff=float("nan"))

    def test_inf_df_diff(self):
        with pytest.raises(ValueError, match="finite"):
            fitting.likelihood_ratio_test(-100.0, -95.0, df_diff=float("inf"))

    def test_negative_df_diff_still_caught(self):
        with pytest.raises(ValueError, match="positive"):
            fitting.likelihood_ratio_test(-100.0, -95.0, df_diff=-1)

    def test_valid_still_works(self):
        result = fitting.likelihood_ratio_test(-100.0, -95.0, df_diff=1)
        assert np.isfinite(result["statistic"])
        assert np.isfinite(result["p_value"])
        # stat = -2(-100 - (-95)) = -2(-5) = 10
        assert result["statistic"] == pytest.approx(10.0, rel=1e-12)
