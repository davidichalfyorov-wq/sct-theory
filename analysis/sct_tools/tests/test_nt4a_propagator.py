# ruff: noqa: I001
"""
NT-4a tests: propagator structure, projectors, gauge invariance, Bianchi identity.

ANTI-CIRCULARITY: These tests verify the Phase 4 scripts against
independently-computed reference values and algebraic identities.

Coverage:
  - Pi_TT, Pi_scalar: zero-momentum limits, finiteness, xi-dependence
  - Scalar mode decoupling at xi = 1/6
  - Barnes-Rivers projectors: orthogonality, completeness, traces, transversality
  - Gauge invariance: G^(1)_{mn}(k; delta_xi h) = 0
  - Bianchi identity: k^mu G^(1)_{mn} = 0
  - Kinetic operator: shape, symmetry
  - Propagator G_TT, G_scalar: massless pole, large-k^2 asymptotic
  - Effective Newton kernel: Newton limit
  - Wrapper exports (sct_tools.propagator)
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


from scripts import nt4a_propagator as nt4a  # noqa: E402
from scripts.nt4a_linearize import (  # noqa: E402
    check_off_shell_bianchi_identity,
    check_off_shell_gauge_invariance,
    contract_first_index_with_k,
    gauge_mode_tensor,
    linearized_curvature_identities,
    linearized_einstein_tensor,
    random_k_vectors,
    random_symmetric_tensors,
    scalar_projector,
    theta_projector,
    tt_projector,
)
from sct_tools import propagator  # noqa: E402


ALPHA_C = 13 / 120
LOCAL_C2 = 13 / 60


# ============================================================================
# PROPAGATOR DENOMINATORS
# ============================================================================

class TestPropagatorDenominators:
    """Pi_TT(z) and Pi_scalar(z) structure."""

    def test_Pi_TT_at_zero(self):
        """Pi_TT(0) = 1 exactly."""
        assert float(nt4a.Pi_TT(0).real) == pytest.approx(1.0, abs=1e-14)

    def test_Pi_scalar_at_zero(self):
        """Pi_scalar(0) = 1 for any xi."""
        for xi in [0.0, 1 / 6, 1.0]:
            assert float(nt4a.Pi_scalar(0, xi=xi).real) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_Pi_TT_is_finite(self, z):
        val = nt4a.Pi_TT(z)
        assert math.isfinite(float(val.real))
        assert math.isfinite(float(val.imag))

    @pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_Pi_scalar_is_finite(self, z):
        val = nt4a.Pi_scalar(z, xi=0.0)
        assert math.isfinite(float(val.real))
        assert math.isfinite(float(val.imag))

    def test_Pi_TT_is_real_on_real_axis(self):
        for z in [0.1, 1.0, 5.0, 10.0]:
            val = nt4a.Pi_TT(z)
            assert abs(float(val.imag)) < 1e-18

    def test_Pi_TT_xi_independence(self):
        """Pi_TT does not depend on xi (it involves only F1/Weyl^2)."""
        val_0 = float(nt4a.Pi_TT(1.0, xi=0.0).real)
        val_conf = float(nt4a.Pi_TT(1.0, xi=1 / 6).real)
        val_1 = float(nt4a.Pi_TT(1.0, xi=1.0).real)
        assert val_0 == pytest.approx(val_conf, rel=1e-12)
        assert val_0 == pytest.approx(val_1, rel=1e-12)

    def test_Pi_TT_nontrivial_for_positive_z(self):
        """Pi_TT(z) != 1 for z > 0 (nonlocal correction is nonzero)."""
        val = float(nt4a.Pi_TT(1.0).real)
        assert abs(val - 1.0) > 0.01, f"Pi_TT(1) = {val}, expected nontrivial correction"

    def test_Pi_TT_reference_values(self):
        refs = {
            0.1: 1.0180639123257555,
            0.5: 1.0232520423925255,
            1.0: 0.8994734408562696,
        }
        for z, expected in refs.items():
            val = float(nt4a.Pi_TT(z).real)
            assert val == pytest.approx(expected, rel=1e-12)

    def test_Pi_scalar_reference_values(self):
        refs = {
            0.1: 1.0170949568754593,
            0.5: 1.0934412650431455,
            1.0: 1.2043061094801226,
        }
        for z, expected in refs.items():
            val = float(nt4a.Pi_scalar(z, xi=0.0).real)
            assert val == pytest.approx(expected, rel=1e-12)

    def test_Pi_TT_changes_sign_on_positive_real_axis(self):
        assert float(nt4a.Pi_TT(2.0).real) > 0
        assert float(nt4a.Pi_TT(5.0).real) < 0

    def test_positive_real_tt_zero_is_detected(self):
        root = float(nt4a.find_first_positive_real_tt_zero())
        assert root == pytest.approx(2.414838889865369, rel=1e-10)

    def test_spin2_local_coefficient(self):
        """The local spin-2 coefficient is exposed explicitly for downstream use."""
        assert float(nt4a.spin2_local_coefficient()) == pytest.approx(13 / 60, rel=1e-14)

    def test_spin2_local_mass(self):
        """m_2 = Lambda / sqrt(c_2) is available from the propagator layer."""
        val = float(nt4a.spin2_local_mass(1.0))
        assert val == pytest.approx(math.sqrt(60 / 13), rel=1e-12)


# ============================================================================
# SCALAR MODE DECOUPLING
# ============================================================================

class TestScalarDecoupling:
    """At xi = 1/6, the scalar mode decouples."""

    def test_scalar_mode_coefficient_conformal(self):
        assert nt4a.scalar_mode_coefficient(1 / 6) == pytest.approx(0.0, abs=1e-14)

    def test_scalar_mode_coefficient_minimal(self):
        """6 * (0 - 1/6)^2 = 1/6."""
        assert float(nt4a.scalar_mode_coefficient(0)) == pytest.approx(1 / 6, rel=1e-14)

    def test_scalar_mode_coefficient_xi1(self):
        """6 * (1 - 1/6)^2 = 25/6."""
        assert float(nt4a.scalar_mode_coefficient(1)) == pytest.approx(25 / 6, rel=1e-14)

    def test_Pi_scalar_identically_one_at_conformal(self):
        """When xi=1/6, Pi_scalar(z) = 1 for all z."""
        for z in [0.0, 0.5, 1.0, 5.0, 10.0]:
            assert float(nt4a.Pi_scalar(z, xi=1 / 6).real) == pytest.approx(1.0, abs=1e-12)

    def test_alpha_R_conformal(self):
        assert float(nt4a.alpha_R(1 / 6)) == pytest.approx(0.0, abs=1e-14)

    def test_alpha_R_formula(self):
        """alpha_R(xi) = 2*(xi - 1/6)^2."""
        for xi in [0.0, 0.1, 0.5, 1.0, 2.0]:
            expected = 2 * (xi - 1 / 6) ** 2
            assert float(nt4a.alpha_R(xi)) == pytest.approx(expected, rel=1e-12)


# ============================================================================
# BARNES-RIVERS PROJECTORS
# ============================================================================

class TestProjectors:
    """Barnes-Rivers spin projectors P^(2) and P^(0-s)."""

    @pytest.fixture
    def k_vec(self):
        return np.array([1.0, 0.5, -0.25, 0.75])

    def test_theta_projector_trace(self, k_vec):
        """tr(theta) = 3 (in 4D)."""
        theta = theta_projector(k_vec)
        assert np.trace(theta) == pytest.approx(3.0, abs=1e-12)

    def test_theta_projector_is_idempotent(self, k_vec):
        """theta^2 = theta."""
        theta = theta_projector(k_vec)
        assert np.allclose(theta @ theta, theta, atol=1e-12)

    def test_theta_projector_is_symmetric(self, k_vec):
        theta = theta_projector(k_vec)
        assert np.allclose(theta, theta.T, atol=1e-14)

    def test_theta_annihilates_k(self, k_vec):
        """theta_{mu nu} k^nu = 0."""
        theta = theta_projector(k_vec)
        result = theta @ k_vec
        assert np.allclose(result, 0, atol=1e-12)

    def test_tt_projector_trace(self, k_vec):
        """P^(2)_{mn mn} = 5 (five spin-2 modes)."""
        P2 = tt_projector(k_vec)
        trace = sum(P2[m, n, m, n] for m in range(4) for n in range(4))
        assert trace == pytest.approx(5.0, abs=1e-10)

    def test_scalar_projector_trace(self, k_vec):
        """P^(0-s)_{mn mn} = 1."""
        P0s = scalar_projector(k_vec)
        trace = sum(P0s[m, n, m, n] for m in range(4) for n in range(4))
        assert trace == pytest.approx(1.0, abs=1e-10)

    def test_projector_orthogonality(self, k_vec):
        """P^(2) * P^(0-s) = 0 (contraction on last two indices)."""
        P2 = tt_projector(k_vec)
        P0s = scalar_projector(k_vec)
        product = np.einsum("mnrs,rsab->mnab", P2, P0s)
        assert np.allclose(product, 0, atol=1e-10)

    def test_tt_projector_transverse(self, k_vec):
        """k^mu P^(2)_{mu nu rho sigma} = 0."""
        contracted = contract_first_index_with_k(tt_projector(k_vec), k_vec)
        assert np.allclose(contracted, 0, atol=1e-10)

    def test_scalar_projector_transverse(self, k_vec):
        """k^mu P^(0-s)_{mu nu rho sigma} = 0."""
        contracted = contract_first_index_with_k(scalar_projector(k_vec), k_vec)
        assert np.allclose(contracted, 0, atol=1e-10)

    def test_tt_projector_idempotent(self, k_vec):
        """P^(2) * P^(2) = P^(2)."""
        P2 = tt_projector(k_vec)
        P2_sq = np.einsum("mnrs,rsab->mnab", P2, P2)
        assert np.allclose(P2_sq, P2, atol=1e-10)

    def test_scalar_projector_idempotent(self, k_vec):
        """P^(0-s) * P^(0-s) = P^(0-s)."""
        P0s = scalar_projector(k_vec)
        P0s_sq = np.einsum("mnrs,rsab->mnab", P0s, P0s)
        assert np.allclose(P0s_sq, P0s, atol=1e-10)


# ============================================================================
# LINEARIZED CURVATURE IDENTITIES (TT GAUGE)
# ============================================================================

class TestLinearizedCurvatures:
    """Curvature identities in TT gauge on flat background."""

    def test_ricci_scalar_vanishes(self):
        ids = linearized_curvature_identities()
        assert ids["Ricci_scalar_TT"] == 0

    def test_r_squared_vanishes(self):
        ids = linearized_curvature_identities()
        assert ids["R_squared_TT"] == 0


# ============================================================================
# GAUGE INVARIANCE (OFF-SHELL)
# ============================================================================

class TestGaugeInvariance:
    """G^(1)_{mn}(k; delta_xi h) = 0 for gauge modes."""

    @pytest.mark.parametrize("seed_idx", range(5))
    def test_gauge_invariance_random_k(self, seed_idx):
        k_vec = random_k_vectors(seed=42, n_vectors=5)[seed_idx]
        assert check_off_shell_gauge_invariance(k_vec)

    def test_gauge_invariance_with_custom_xi(self):
        k_vec = np.array([2.0, 1.0, 0.5, -0.3])
        xi_vec = np.array([0.1, -0.2, 0.3, 0.4])
        assert check_off_shell_gauge_invariance(k_vec, xi_vec=xi_vec)


# ============================================================================
# BIANCHI IDENTITY (OFF-SHELL)
# ============================================================================

class TestBianchiIdentity:
    """k^mu G^(1)_{mn}(k; h) = 0 for any symmetric h."""

    @pytest.mark.parametrize("seed_idx", range(5))
    def test_bianchi_random_samples(self, seed_idx):
        k_vec = random_k_vectors(seed=42, n_vectors=5)[seed_idx]
        h_tensor = random_symmetric_tensors(seed=43, n_tensors=5)[seed_idx]
        assert check_off_shell_bianchi_identity(k_vec, h_tensor=h_tensor)

    def test_bianchi_with_default_tensor(self):
        k_vec = np.array([1.0, 0.5, -0.25, 0.75])
        assert check_off_shell_bianchi_identity(k_vec)


# ============================================================================
# KINETIC OPERATOR
# ============================================================================

class TestKineticOperator:
    """Kinetic operator K_{mn rs}(k) = k^2 [Pi_TT P^(2) + Pi_s P^(0-s)]."""

    def test_shape(self):
        k = np.array([1.0, 0.5, -0.25, 0.75])
        K = nt4a.kinetic_operator(k)
        assert K.shape == (4, 4, 4, 4)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError, match="positive"):
            nt4a.kinetic_operator(np.zeros(4))

    def test_kinetic_operator_transverse(self):
        """k^mu K_{mu nu rho sigma} should be zero (both P^(2) and P^(0-s) are TT)."""
        k = np.array([1.0, 0.5, -0.25, 0.75])
        K = nt4a.kinetic_operator(k)
        contracted = np.tensordot(k, K, axes=(0, 0))
        assert np.allclose(contracted, 0, atol=1e-8)

    def test_kinetic_operator_minor_symmetry(self):
        """K_{mnrs} = K_{nmrs} = K_{mnsr}."""
        k = np.array([1.0, 0.5, -0.25, 0.75])
        K = nt4a.kinetic_operator(k)
        assert np.allclose(K, K.transpose(1, 0, 2, 3), atol=1e-10)
        assert np.allclose(K, K.transpose(0, 1, 3, 2), atol=1e-10)


# ============================================================================
# PROPAGATOR GREEN'S FUNCTIONS
# ============================================================================

class TestGreensFunction:
    """G_TT and G_scalar propagators."""

    def test_G_TT_pole_at_zero(self):
        with pytest.raises(ZeroDivisionError):
            nt4a.G_TT(0)

    def test_G_scalar_pole_at_zero(self):
        with pytest.raises(ZeroDivisionError):
            nt4a.G_scalar(0)

    def test_G_TT_uv_suppressed(self):
        """G_TT(k^2) is UV-suppressed: |G_TT| << 1/k^2 for k^2 >> Lambda^2."""
        k2 = 1e6
        val = float(nt4a.G_TT(k2, Lambda2=1.0).real)
        assert abs(val) < 1 / k2, f"|G_TT({k2})| = {abs(val)} should be less than {1/k2}"

    def test_effective_newton_kernel_at_zero(self):
        """At z=0: kernel = 4/3 - 1/3 = 1 (standard Newton)."""
        val = float(nt4a.effective_newton_kernel(0, xi=0.0).real)
        assert val == pytest.approx(1.0, abs=1e-12)


# ============================================================================
# WRAPPER EXPORTS
# ============================================================================

class TestWrapperExports:
    """sct_tools.propagator re-exports match nt4a_propagator directly."""

    def test_Pi_TT_wrapper(self):
        assert propagator.Pi_TT(0.5) == nt4a.Pi_TT(0.5)

    def test_Pi_scalar_wrapper(self):
        assert propagator.Pi_scalar(0.5, xi=0.0) == nt4a.Pi_scalar(0.5, xi=0.0)

    def test_G_TT_wrapper(self):
        assert propagator.G_TT(1.0) == nt4a.G_TT(1.0)

    def test_G_scalar_wrapper(self):
        assert propagator.G_scalar(1.0, xi=0.0) == nt4a.G_scalar(1.0, xi=0.0)

    def test_spin2_local_mass_wrapper(self):
        assert propagator.spin2_local_mass(1.0) == nt4a.spin2_local_mass(1.0)

    def test_scalar_mode_coefficient_wrapper(self):
        assert propagator.scalar_mode_coefficient(0.0) == nt4a.scalar_mode_coefficient(0.0)

    def test_positive_real_tt_zero_wrapper(self):
        assert (
            propagator.find_first_positive_real_tt_zero()
            == nt4a.find_first_positive_real_tt_zero()
        )


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class TestInputValidation:
    """Error handling in linearize module."""

    def test_invalid_k_shape(self):
        with pytest.raises(ValueError, match="shape"):
            theta_projector(np.array([1.0, 2.0]))

    def test_negative_norm_k(self):
        with pytest.raises(ValueError, match="positive"):
            theta_projector(np.zeros(4))

    def test_nonsymmetric_h_rejected(self):
        k = np.array([1.0, 0.5, -0.25, 0.75])
        h_bad = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=float,
        )
        with pytest.raises(ValueError, match="symmetric"):
            linearized_einstein_tensor(k, h_bad)

    def test_gauge_mode_tensor_shape(self):
        k = np.array([1.0, 0.5, -0.25, 0.75])
        xi = np.array([0.1, -0.2, 0.3, 0.4])
        gm = gauge_mode_tensor(k, xi)
        assert gm.shape == (4, 4)
        assert np.allclose(gm, gm.T, atol=1e-14)
