"""
Tests for Iteration 45: Error message colon standardization in form_factors.py.

Validates that all form_factors error messages follow the 'func: message' pattern.
"""

import numpy as np
import pytest

from sct_tools import form_factors as ff

# ============================================================================
# Representative colon-format checks across all function categories
# ============================================================================


class TestPhiColonFormat:
    """phi/phi_fast/phi_closed error messages must use 'func: requires' pattern."""

    def test_phi_finite(self):
        with pytest.raises(ValueError, match="^phi: requires finite x"):
            ff.phi(float("nan"))

    def test_phi_nonneg(self):
        with pytest.raises(ValueError, match="^phi: requires x >= 0"):
            ff.phi(-1.0)

    def test_phi_fast_finite(self):
        with pytest.raises(ValueError, match="^phi_fast: requires finite x"):
            ff.phi_fast(float("inf"))

    def test_phi_fast_nonneg(self):
        with pytest.raises(ValueError, match="^phi_fast: requires x >= 0"):
            ff.phi_fast(-0.1)

    def test_phi_closed_finite(self):
        with pytest.raises(ValueError, match="^phi_closed: requires finite x"):
            ff.phi_closed(float("nan"))

    def test_phi_closed_nonneg(self):
        with pytest.raises(ValueError, match="^phi_closed: requires x >= 0"):
            ff.phi_closed(-5.0)

    def test_phi_vec_received(self):
        with pytest.raises(ValueError, match="^phi_vec: received NaN"):
            ff.phi_vec(np.array([1.0, float("nan")]))

    def test_phi_vec_nonneg(self):
        with pytest.raises(ValueError, match="^phi_vec: requires all x >= 0"):
            ff.phi_vec(np.array([1.0, -1.0]))


class TestScalarFastColonFormat:
    """hC_scalar_fast / hR_scalar_fast must use 'func: requires' pattern."""

    def test_hC_scalar_fast_finite(self):
        with pytest.raises(ValueError, match="^hC_scalar_fast: requires finite x"):
            ff.hC_scalar_fast(float("nan"))

    def test_hC_scalar_fast_nonneg(self):
        with pytest.raises(ValueError, match="^hC_scalar_fast: requires x >= 0"):
            ff.hC_scalar_fast(-1.0)

    def test_hR_scalar_fast_finite(self):
        with pytest.raises(ValueError, match="^hR_scalar_fast: requires finite x"):
            ff.hR_scalar_fast(float("nan"))

    def test_hR_scalar_fast_xi(self):
        with pytest.raises(ValueError, match="^hR_scalar_fast: requires finite xi"):
            ff.hR_scalar_fast(1.0, xi=float("inf"))

    def test_hR_scalar_fast_nonneg(self):
        with pytest.raises(ValueError, match="^hR_scalar_fast: requires x >= 0"):
            ff.hR_scalar_fast(-1.0)


class TestDiracFastColonFormat:
    """hC_dirac_fast / hR_dirac_fast must use 'func: requires' pattern."""

    def test_hC_dirac_fast_finite(self):
        with pytest.raises(ValueError, match="^hC_dirac_fast: requires finite x"):
            ff.hC_dirac_fast(float("nan"))

    def test_hR_dirac_fast_nonneg(self):
        with pytest.raises(ValueError, match="^hR_dirac_fast: requires x >= 0"):
            ff.hR_dirac_fast(-1.0)


class TestVectorFastColonFormat:
    """hC_vector_fast / hR_vector_fast must use 'func: requires' pattern."""

    def test_hC_vector_fast_finite(self):
        with pytest.raises(ValueError, match="^hC_vector_fast: requires finite x"):
            ff.hC_vector_fast(float("inf"))

    def test_hR_vector_fast_nonneg(self):
        with pytest.raises(ValueError, match="^hR_vector_fast: requires x >= 0"):
            ff.hR_vector_fast(-1.0)


class TestCZBasisColonFormat:
    """CZ basis functions (f_Ric, f_R, f_RU, f_U, f_Omega) must use colon pattern."""

    def test_f_Ric_finite(self):
        with pytest.raises(ValueError, match="^f_Ric: requires finite x"):
            ff.f_Ric(float("nan"))

    def test_f_R_nonneg(self):
        with pytest.raises(ValueError, match="^f_R: requires x >= 0"):
            ff.f_R(-1.0)

    def test_f_RU_finite(self):
        with pytest.raises(ValueError, match="^f_RU: requires finite x"):
            ff.f_RU(float("inf"))

    def test_f_U_nonneg(self):
        with pytest.raises(ValueError, match="^f_U: requires x >= 0"):
            ff.f_U(-1.0)

    def test_f_Omega_finite(self):
        with pytest.raises(ValueError, match="^f_Omega: requires finite x"):
            ff.f_Omega(float("nan"))


class TestQuadIntegratorColonFormat:
    """Quad-based hC_scalar, hR_scalar, hC_dirac, hR_dirac, hC_vector, hR_vector."""

    def test_hC_scalar_finite(self):
        with pytest.raises(ValueError, match="^hC_scalar: requires finite x"):
            ff.hC_scalar(float("nan"))

    def test_hR_scalar_nonneg(self):
        with pytest.raises(ValueError, match="^hR_scalar: requires x >= 0"):
            ff.hR_scalar(-1.0)

    def test_hR_scalar_xi_finite(self):
        with pytest.raises(ValueError, match="^hR_scalar: requires finite xi"):
            ff.hR_scalar(1.0, xi=float("nan"))

    def test_hC_dirac_nonneg(self):
        with pytest.raises(ValueError, match="^hC_dirac: requires x >= 0"):
            ff.hC_dirac(-1.0)

    def test_hR_dirac_finite(self):
        with pytest.raises(ValueError, match="^hR_dirac: requires finite x"):
            ff.hR_dirac(float("inf"))

    def test_hC_vector_finite(self):
        with pytest.raises(ValueError, match="^hC_vector: requires finite x"):
            ff.hC_vector(float("nan"))

    def test_hR_vector_nonneg(self):
        with pytest.raises(ValueError, match="^hR_vector: requires x >= 0"):
            ff.hR_vector(-1.0)


class TestMpmathColonFormat:
    """mpmath variants must use 'func: requires' pattern."""

    def test_phi_mp_finite(self):
        with pytest.raises(ValueError, match="^phi_mp: requires finite x"):
            ff.phi_mp(float("nan"))

    def test_hC_scalar_mp_nonneg(self):
        with pytest.raises(ValueError, match="^hC_scalar_mp: requires x >= 0"):
            ff.hC_scalar_mp(-1.0)

    def test_hR_scalar_mp_xi(self):
        with pytest.raises(ValueError, match="^hR_scalar_mp: requires finite xi"):
            ff.hR_scalar_mp(1.0, xi=float("inf"))

    def test_hC_dirac_mp_finite(self):
        with pytest.raises(ValueError, match="^hC_dirac_mp: requires finite x"):
            ff.hC_dirac_mp(float("nan"))

    def test_hR_dirac_mp_nonneg(self):
        with pytest.raises(ValueError, match="^hR_dirac_mp: requires x >= 0"):
            ff.hR_dirac_mp(-1.0)

    def test_hC_vector_mp_finite(self):
        with pytest.raises(ValueError, match="^hC_vector_mp: requires finite x"):
            ff.hC_vector_mp(float("inf"))

    def test_hR_vector_mp_nonneg(self):
        with pytest.raises(ValueError, match="^hR_vector_mp: requires x >= 0"):
            ff.hR_vector_mp(-1.0)


class TestTaylorColonFormat:
    """Taylor expansion functions must use colon pattern."""

    def test_hC_scalar_taylor_finite(self):
        with pytest.raises(ValueError, match="^hC_scalar_taylor: requires finite x"):
            ff.hC_scalar_taylor(float("nan"))

    def test_hR_scalar_taylor_nonneg(self):
        with pytest.raises(ValueError, match="^hR_scalar_taylor: requires x >= 0"):
            ff.hR_scalar_taylor(-1.0)

    def test_hR_scalar_taylor_xi(self):
        with pytest.raises(ValueError, match="^hR_scalar_taylor: requires finite xi"):
            ff.hR_scalar_taylor(1.0, xi=float("inf"))


class TestDerivativeColonFormat:
    """Derivative functions must use colon pattern."""

    def test_dphi_dx_finite(self):
        with pytest.raises(ValueError, match="^dphi_dx: requires finite x"):
            ff.dphi_dx(float("nan"))

    def test_dphi_dx_fast_nonneg(self):
        with pytest.raises(ValueError, match="^dphi_dx_fast: requires x >= 0"):
            ff.dphi_dx_fast(-1.0)

    def test_dhC_scalar_dx_finite(self):
        with pytest.raises(ValueError, match="^dhC_scalar_dx: requires finite x"):
            ff.dhC_scalar_dx(float("nan"))

    def test_dhC_dirac_dx_nonneg(self):
        with pytest.raises(ValueError, match="^dhC_dirac_dx: requires x >= 0"):
            ff.dhC_dirac_dx(-1.0)

    def test_dhR_dirac_dx_finite(self):
        with pytest.raises(ValueError, match="^dhR_dirac_dx: requires finite x"):
            ff.dhR_dirac_dx(float("inf"))

    def test_dhR_scalar_dx_nonneg(self):
        with pytest.raises(ValueError, match="^dhR_scalar_dx: requires x >= 0"):
            ff.dhR_scalar_dx(-1.0)

    def test_dhR_scalar_dx_xi(self):
        with pytest.raises(ValueError, match="^dhR_scalar_dx: requires finite xi"):
            ff.dhR_scalar_dx(1.0, xi=float("nan"))

    def test_dhC_vector_dx_finite(self):
        with pytest.raises(ValueError, match="^dhC_vector_dx: requires finite x"):
            ff.dhC_vector_dx(float("inf"))

    def test_dhR_vector_dx_nonneg(self):
        with pytest.raises(ValueError, match="^dhR_vector_dx: requires x >= 0"):
            ff.dhR_vector_dx(-1.0)


class TestAsymptoticColonFormat:
    """asymptotic_expansion must use colon pattern."""

    def test_requires_finite(self):
        with pytest.raises(ValueError, match="^asymptotic_expansion: requires finite x"):
            ff.asymptotic_expansion("hC_scalar", float("nan"))

    def test_requires_positive(self):
        with pytest.raises(ValueError, match="^asymptotic_expansion: requires x > 0"):
            ff.asymptotic_expansion("hC_scalar", -1.0)
