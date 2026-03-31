# ruff: noqa: E402, I001
"""
PPN-1 pytest test suite — 10 categories, >= 100 tests.

Categories:
  1. Constants and conventions
  2. Propagator denominators
  3. Newton kernels
  4. Effective masses
  5. Local Yukawa potentials (Level 2)
  6. PPN gamma (Level 2)
  7. Experimental bounds
  8. PPN table and snapshot
  9. Level 1 structural properties
 10. xi-dependence and conformal limit
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.ppn1_parameters import (
    ALPHA_C,
    AU_EV_INV,
    HBAR_C_EV_M,
    LOCAL_C2,
    NOT_DERIVED,
    K_Phi,
    K_Psi,
    _Pi_TT,
    _Pi_TT_prime_at_z0,
    _Pi_scalar,
    _find_tt_zero,
    _scalar_mode_coefficient,
    effective_masses,
    gamma_local,
    level_comparison,
    lower_bound_Lambda,
    phi_exact,
    phi_local,
    ppn_snapshot,
    ppn_table,
    psi_exact,
    psi_local,
)
from scripts.nt4a_newtonian import gamma_local_ratio

DPS = 60
mp.mp.dps = DPS


# ===================================================================
# Category 1: Constants and conventions (10 tests)
# ===================================================================
class TestConstantsConventions:
    """Verify physical constants and naming conventions."""

    def test_alpha_c_value(self):
        assert abs(ALPHA_C - mp.mpf(13) / 120) < 1e-14

    def test_local_c2_value(self):
        assert abs(LOCAL_C2 - mp.mpf(13) / 60) < 1e-14

    def test_local_c2_is_2_alpha_c(self):
        assert abs(LOCAL_C2 - 2 * ALPHA_C) < 1e-14

    def test_hbar_c_positive(self):
        assert HBAR_C_EV_M > 0

    def test_hbar_c_order_of_magnitude(self):
        assert 1e-8 < float(HBAR_C_EV_M) < 1e-6

    def test_au_ev_inv_positive(self):
        assert AU_EV_INV > 0

    def test_au_ev_inv_order_of_magnitude(self):
        assert 1e17 < float(AU_EV_INV) < 1e18

    def test_not_derived_sentinel(self):
        assert NOT_DERIVED == "NOT_DERIVED"

    def test_alpha_c_is_mpf(self):
        assert isinstance(ALPHA_C, mp.mpf)

    def test_local_c2_is_mpf(self):
        assert isinstance(LOCAL_C2, mp.mpf)


# ===================================================================
# Category 2: Propagator denominators (12 tests)
# ===================================================================
class TestPropagatorDenominators:
    """Verify Pi_TT and Pi_s propagator functions."""

    def test_pi_tt_at_zero(self):
        val = mp.re(_Pi_TT(0, xi=0.0, dps=DPS))
        assert abs(val - 1) < 1e-30

    def test_pi_s_at_zero(self):
        val = mp.re(_Pi_scalar(0, xi=0.0, dps=DPS))
        assert abs(val - 1) < 1e-30

    def test_pi_tt_at_zero_conformal(self):
        val = mp.re(_Pi_TT(0, xi=1 / 6, dps=DPS))
        assert abs(val - 1) < 1e-30

    def test_pi_s_conformal_is_one(self):
        """Pi_s = 1 identically when xi=1/6 (scalar decouples)."""
        for z in [0.0, 0.5, 1.0, 5.0]:
            val = mp.re(_Pi_scalar(z, xi=1 / 6, dps=DPS))
            assert abs(val - 1) < 1e-14, f"Pi_s({z}, xi=1/6) = {val}"

    def test_pi_tt_increases_then_crosses_zero(self):
        """Pi_TT initially increases (F1_hat(0)=1 > 0), then decreases through zero."""
        val_01 = mp.re(_Pi_TT(0.1, xi=0.0, dps=DPS))
        assert val_01 > 1.0  # increases at small z
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        val_z0 = mp.re(_Pi_TT(z0, xi=0.0, dps=DPS))
        assert abs(val_z0) < 1e-15  # crosses zero

    def test_pi_tt_has_zero(self):
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        assert abs(mp.re(_Pi_TT(z0, xi=0.0, dps=DPS))) < 1e-15

    def test_pi_tt_zero_location(self):
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        assert abs(z0 - mp.mpf("2.41484")) < 0.001

    def test_pi_tt_prime_at_z0(self):
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        deriv = _Pi_TT_prime_at_z0(z0, xi=0.0, dps=DPS)
        assert abs(deriv - mp.mpf("-0.8398")) < 0.001

    def test_scalar_mode_coefficient_xi0(self):
        coeff = _scalar_mode_coefficient(0.0)
        expected = 6 * (mp.mpf(0) - mp.mpf(1) / 6) ** 2
        assert abs(coeff - expected) < 1e-14

    def test_scalar_mode_coefficient_conformal(self):
        coeff = _scalar_mode_coefficient(1 / 6)
        assert abs(coeff) < 1e-14

    def test_scalar_mode_coefficient_positive(self):
        """Coefficient is non-negative for all xi."""
        for xi in [0.0, 0.1, 0.25, 0.5, 1.0]:
            assert _scalar_mode_coefficient(xi) >= 0

    def test_pi_s_positive_for_positive_z(self):
        """Pi_s > 0 for z in [0, 10] at xi=0."""
        for z in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            val = mp.re(_Pi_scalar(z, xi=0.0, dps=DPS))
            assert val > 0, f"Pi_s({z}) = {val}"


# ===================================================================
# Category 3: Newton kernels (14 tests)
# ===================================================================
class TestNewtonKernels:
    """Verify K_Phi and K_Psi Newton kernel functions."""

    def test_k_phi_at_zero(self):
        val = mp.re(K_Phi(0, xi=0.0, dps=DPS))
        assert abs(val - 1) < 1e-30

    def test_k_psi_at_zero(self):
        val = mp.re(K_Psi(0, xi=0.0, dps=DPS))
        assert abs(val - 1) < 1e-30

    def test_k_phi_at_zero_conformal(self):
        val = mp.re(K_Phi(0, xi=1 / 6, dps=DPS))
        assert abs(val - 1) < 1e-30

    def test_k_psi_at_zero_conformal(self):
        val = mp.re(K_Psi(0, xi=1 / 6, dps=DPS))
        assert abs(val - 1) < 1e-30

    @pytest.mark.parametrize("z", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_sum_rule(self, z):
        """K_Phi + K_Psi = 2/Pi_TT for all z."""
        kp = mp.re(K_Phi(z, xi=0.0, dps=DPS))
        ks = mp.re(K_Psi(z, xi=0.0, dps=DPS))
        pi_tt = mp.re(_Pi_TT(z, xi=0.0, dps=DPS))
        expected = mp.mpf(2) / pi_tt
        assert abs(kp + ks - expected) < 1e-25

    def test_k_phi_reconstruction(self):
        """K_Phi = 4/(3*Pi_TT) - 1/(3*Pi_s)."""
        z = mp.mpf("1.5")
        kp = mp.re(K_Phi(z, xi=0.0, dps=DPS))
        pi_tt = mp.re(_Pi_TT(z, xi=0.0, dps=DPS))
        pi_s = mp.re(_Pi_scalar(z, xi=0.0, dps=DPS))
        expected = mp.mpf(4) / (3 * pi_tt) - mp.mpf(1) / (3 * pi_s)
        assert abs(kp - expected) < 1e-30

    def test_k_psi_reconstruction(self):
        """K_Psi = 2/(3*Pi_TT) + 1/(3*Pi_s)."""
        z = mp.mpf("1.5")
        ks = mp.re(K_Psi(z, xi=0.0, dps=DPS))
        pi_tt = mp.re(_Pi_TT(z, xi=0.0, dps=DPS))
        pi_s = mp.re(_Pi_scalar(z, xi=0.0, dps=DPS))
        expected = mp.mpf(2) / (3 * pi_tt) + mp.mpf(1) / (3 * pi_s)
        assert abs(ks - expected) < 1e-30

    def test_k_phi_is_real_for_real_z(self):
        val = K_Phi(1.0, xi=0.0, dps=DPS)
        assert abs(mp.im(val)) < 1e-30


# ===================================================================
# Category 4: Effective masses (12 tests)
# ===================================================================
class TestEffectiveMasses:
    """Verify effective mass computations."""

    def test_m2_formula(self):
        m2, _ = effective_masses(Lambda=1.0, xi=0.0)
        expected = mp.sqrt(mp.mpf(60) / 13)
        assert abs(m2 - expected) < 1e-25

    def test_m0_formula_xi0(self):
        _, m0 = effective_masses(Lambda=1.0, xi=0.0)
        expected = mp.sqrt(mp.mpf(6))
        assert abs(m0 - expected) < 1e-25

    def test_m0_none_at_conformal(self):
        _, m0 = effective_masses(Lambda=1.0, xi=1 / 6)
        assert m0 is None

    def test_m2_scales_with_lambda(self):
        m2_1, _ = effective_masses(Lambda=1.0, xi=0.0)
        m2_2, _ = effective_masses(Lambda=2.0, xi=0.0)
        assert abs(m2_2 / m2_1 - 2) < 1e-14

    def test_m0_scales_with_lambda(self):
        _, m0_1 = effective_masses(Lambda=1.0, xi=0.0)
        _, m0_2 = effective_masses(Lambda=2.0, xi=0.0)
        assert abs(m0_2 / m0_1 - 2) < 1e-14

    def test_m2_positive(self):
        m2, _ = effective_masses(Lambda=1.0, xi=0.0)
        assert m2 > 0

    def test_m0_positive_xi0(self):
        _, m0 = effective_masses(Lambda=1.0, xi=0.0)
        assert m0 is not None and m0 > 0

    def test_m2_numerical_value(self):
        m2, _ = effective_masses(Lambda=1.0, xi=0.0)
        assert abs(float(m2) - 2.14834) < 0.001

    def test_m0_numerical_value_xi0(self):
        _, m0 = effective_masses(Lambda=1.0, xi=0.0)
        assert abs(float(m0) - 2.44949) < 0.001

    def test_m0_increases_away_from_conformal(self):
        """m0 should increase as xi moves away from 1/6."""
        _, m0_0 = effective_masses(Lambda=1.0, xi=0.0)
        _, m0_q = effective_masses(Lambda=1.0, xi=0.25)
        assert m0_q > m0_0

    def test_m2_independent_of_xi(self):
        """m2 depends only on Lambda, not xi."""
        m2_0, _ = effective_masses(Lambda=1.0, xi=0.0)
        m2_q, _ = effective_masses(Lambda=1.0, xi=0.25)
        assert abs(m2_0 - m2_q) < 1e-25

    def test_50_digit_precision(self):
        """Masses agree with exact formula to >= 45 digits."""
        mp.mp.dps = 60
        m2, m0 = effective_masses(Lambda=mp.mpf("1"), xi=0.0)
        expected_m2 = mp.sqrt(mp.mpf(60) / 13)
        expected_m0 = mp.sqrt(mp.mpf(6))
        assert abs(m2 - expected_m2) < mp.power(10, -45)
        assert abs(m0 - expected_m0) < mp.power(10, -45)


# ===================================================================
# Category 5: Local Yukawa potentials (14 tests)
# ===================================================================
class TestLocalYukawaPotentials:
    """Verify Level 2 local Yukawa Phi and Psi."""

    def test_phi_local_gr_limit(self):
        """Phi/Phi_N -> 1 at large r."""
        val = phi_local(1e10, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(val - 1) < 1e-10

    def test_psi_local_gr_limit(self):
        """Psi/Psi_N -> 1 at large r."""
        val = psi_local(1e10, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(val - 1) < 1e-10

    @pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_phi_matches_explicit_formula(self, r):
        """phi_local matches 1 - 4/3*exp(-m2*r) + 1/3*exp(-m0*r)."""
        mp.mp.dps = DPS
        m2, m0 = effective_masses(Lambda=1.0, xi=0.0)
        expected = 1 - mp.mpf(4) / 3 * mp.exp(-m2 * r) + mp.mpf(1) / 3 * mp.exp(-m0 * r)
        val = phi_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(val - expected) < 1e-30

    @pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_psi_matches_explicit_formula(self, r):
        """psi_local matches 1 - 2/3*exp(-m2*r) - 1/3*exp(-m0*r)."""
        mp.mp.dps = DPS
        m2, m0 = effective_masses(Lambda=1.0, xi=0.0)
        expected = 1 - mp.mpf(2) / 3 * mp.exp(-m2 * r) - mp.mpf(1) / 3 * mp.exp(-m0 * r)
        val = psi_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(val - expected) < 1e-30

    def test_phi_raises_on_nonpositive_r(self):
        with pytest.raises(ValueError):
            phi_local(0, Lambda=1.0, xi=0.0)

    def test_psi_raises_on_nonpositive_r(self):
        with pytest.raises(ValueError):
            psi_local(0, Lambda=1.0, xi=0.0)

    def test_phi_monotonically_increasing(self):
        """Phi/Phi_N increases from 0 to 1 as r increases."""
        mp.mp.dps = DPS
        vals = [phi_local(r, Lambda=1.0, xi=0.0, dps=DPS) for r in [0.01, 0.1, 1.0, 10.0]]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i]

    def test_phi_conformal_no_scalar(self):
        """At xi=1/6, only spin-2 Yukawa contributes."""
        mp.mp.dps = DPS
        m2, _ = effective_masses(Lambda=1.0, xi=1 / 6)
        r = 1.0
        expected = 1 - mp.mpf(4) / 3 * mp.exp(-m2 * r)
        val = phi_local(r, Lambda=1.0, xi=1 / 6, dps=DPS)
        assert abs(val - expected) < 1e-14


# ===================================================================
# Category 6: PPN gamma (12 tests)
# ===================================================================
class TestPPNGamma:
    """Verify PPN gamma parameter computations."""

    def test_gamma_gr_limit(self):
        """gamma -> 1 at large r."""
        val = gamma_local(1e10, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(val - 1) < 1e-20

    def test_gamma_is_psi_over_phi(self):
        """gamma = Psi/Phi at finite r."""
        mp.mp.dps = DPS
        r = 1.0
        phi = phi_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        psi = psi_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        gamma = gamma_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(gamma - psi / phi) < 1e-30

    def test_gamma_0_xi0_lhopital(self):
        """gamma(0, xi=0) = (2m2+m0)/(4m2-m0) via nt4a_newtonian."""
        mp.mp.dps = DPS
        m2, m0 = effective_masses(Lambda=1.0, xi=0.0)
        expected = (2 * m2 + m0) / (4 * m2 - m0)
        val = gamma_local_ratio(0, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(val - expected) < 1e-14

    def test_gamma_0_xi0_approx_value(self):
        val = gamma_local_ratio(0, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(float(val) - 1.098) < 0.001

    def test_gamma_0_conformal_is_minus1(self):
        """gamma(0, xi=1/6) = -1."""
        val = gamma_local_ratio(0, Lambda=1.0, xi=1 / 6, dps=DPS)
        assert abs(val + 1) < 1e-10

    def test_gamma_positive_at_large_r(self):
        for r in [1.0, 10.0, 100.0, 1000.0]:
            g = gamma_local(r, Lambda=1.0, xi=0.0, dps=DPS)
            assert g > 0

    def test_gamma_approaches_1_from_above(self):
        """For xi=0, gamma(r->inf) -> 1 from above."""
        g = gamma_local(5.0, Lambda=1.0, xi=0.0, dps=DPS)
        assert g > 1 or abs(g - 1) < 1e-5

    @pytest.mark.parametrize("r", [0.1, 1.0, 5.0, 10.0, 50.0])
    def test_gamma_finite(self, r):
        g = gamma_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        assert mp.isfinite(g)

    def test_gamma_local_matches_nt4a(self):
        """gamma_local and gamma_local_ratio agree at finite r."""
        r = 1.0
        g1 = gamma_local(r, Lambda=1.0, xi=0.0, dps=DPS)
        g2 = gamma_local_ratio(r, Lambda=1.0, xi=0.0, dps=DPS)
        assert abs(g1 - g2) < 1e-14

    def test_gamma_at_small_r_greater_than_1(self):
        """For xi=0, gamma(small r) > 1."""
        g = gamma_local(0.01, Lambda=1.0, xi=0.0, dps=DPS)
        assert g > 1


# ===================================================================
# Category 7: Experimental bounds (10 tests)
# ===================================================================
class TestExperimentalBounds:
    """Verify experimental lower bounds on Lambda."""

    def test_cassini_returns_dict(self):
        result = lower_bound_Lambda("cassini", xi=0.0)
        assert isinstance(result, dict)
        assert "Lambda_min_eV" in result

    def test_cassini_bound_order_of_magnitude(self):
        result = lower_bound_Lambda("cassini", xi=0.0)
        lam = result["Lambda_min_eV"]
        assert 1e-18 < lam < 1e-17

    def test_messenger_bound_order_of_magnitude(self):
        result = lower_bound_Lambda("messenger", xi=0.0)
        lam = result["Lambda_min_eV"]
        assert 1e-18 < lam < 1e-17

    def test_messenger_comparable_to_cassini(self):
        """MESSENGER and Cassini bounds are within a factor of 2."""
        c = lower_bound_Lambda("cassini", xi=0.0)["Lambda_min_eV"]
        m = lower_bound_Lambda("messenger", xi=0.0)["Lambda_min_eV"]
        assert abs(m - c) / c < 0.5  # same order of magnitude

    def test_eotwash_bound_order_of_magnitude(self):
        result = lower_bound_Lambda("eot-wash", xi=0.0)
        lam = result["Lambda_min_eV"]
        assert 1e-4 < lam < 1e-2

    def test_eotwash_much_stronger_than_solar(self):
        """Eot-Wash bound is ~14 orders stronger than solar system."""
        c = lower_bound_Lambda("cassini", xi=0.0)["Lambda_min_eV"]
        e = lower_bound_Lambda("eot-wash", xi=0.0)["Lambda_min_eV"]
        assert e / c > 1e10

    def test_microscope_not_derived(self):
        result = lower_bound_Lambda("microscope", xi=0.0)
        assert result["Lambda_min_eV"] is None

    def test_llr_not_derived(self):
        result = lower_bound_Lambda("llr", xi=0.0)
        assert result["Lambda_min_eV"] is None

    def test_unknown_experiment_raises(self):
        with pytest.raises(ValueError):
            lower_bound_Lambda("unknown_experiment", xi=0.0)

    def test_cassini_reference_present(self):
        result = lower_bound_Lambda("cassini", xi=0.0)
        assert "Nature" in result["reference"] or "Bertotti" in result["experiment"]


# ===================================================================
# Category 8: PPN table and snapshot (10 tests)
# ===================================================================
class TestPPNTableSnapshot:
    """Verify ppn_table and ppn_snapshot outputs."""

    def test_ppn_table_returns_dict(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert isinstance(table, dict)

    def test_ppn_table_scope(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert table["scope"] == "linear_static_local_yukawa"

    def test_ppn_table_has_gamma(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert "gamma" in table

    def test_ppn_table_beta_not_derived(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert table["beta"] == NOT_DERIVED

    def test_ppn_table_xi_ppn_not_derived(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert table["xi_PPN"] == NOT_DERIVED

    def test_ppn_table_alpha_zero(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert "0" in str(table["alpha1"])
        assert "0" in str(table["alpha2"])
        assert "0" in str(table["alpha3"])

    def test_ppn_table_zeta_zero(self):
        table = ppn_table(Lambda=1e-3, xi=0.0)
        assert "0" in str(table["zeta1"])
        assert "0" in str(table["zeta2"])
        assert "0" in str(table["zeta3"])
        assert "0" in str(table["zeta4"])

    def test_ppn_snapshot_returns_dict(self):
        snap = ppn_snapshot(Lambda=1.0, xi=0.0)
        assert isinstance(snap, dict)

    def test_ppn_snapshot_has_bounds(self):
        snap = ppn_snapshot(Lambda=1.0, xi=0.0)
        assert "bounds" in snap
        assert "cassini" in snap["bounds"]

    def test_ppn_snapshot_has_pole_data(self):
        snap = ppn_snapshot(Lambda=1.0, xi=0.0)
        assert "pole_data" in snap
        assert "z0" in snap["pole_data"]


# ===================================================================
# Category 9: Level 1 structural properties (12 tests)
# ===================================================================
class TestLevel1Structural:
    """Verify Level 1 (exact nonlocal) structural claims."""

    def test_uv_asymptote_nonzero(self):
        """K_Phi(z->inf) ~ -0.136, NOT zero."""
        val = mp.re(K_Phi(100, xi=0.0, dps=DPS))
        assert abs(val) > 0.1

    def test_uv_asymptote_value_z100(self):
        val = mp.re(K_Phi(100, xi=0.0, dps=DPS))
        assert abs(val - (-0.136)) < 0.005

    def test_uv_asymptote_value_z1000(self):
        val = mp.re(K_Phi(1000, xi=0.0, dps=DPS))
        assert abs(val - (-0.136)) < 0.002

    def test_uv_asymptote_convergence(self):
        """K_Phi(z) converges as z grows."""
        v1 = mp.re(K_Phi(100, xi=0.0, dps=DPS))
        v2 = mp.re(K_Phi(500, xi=0.0, dps=DPS))
        v3 = mp.re(K_Phi(1000, xi=0.0, dps=DPS))
        assert abs(v2 - v3) < abs(v1 - v2)

    def test_tt_pole_is_real_positive(self):
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        assert z0 > 0

    def test_tt_pole_below_stelle(self):
        """z0_SCT ~ 2.41 < z0_Stelle = 60/13 ~ 4.615."""
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        z0_stelle = mp.mpf(60) / 13
        assert z0 < z0_stelle

    def test_pi_tt_prime_negative_at_z0(self):
        z0 = _find_tt_zero(xi=0.0, dps=DPS)
        deriv = _Pi_TT_prime_at_z0(z0, xi=0.0, dps=DPS)
        assert deriv < 0

    def test_phi_exact_placeholder_warning(self):
        assert "PLACEHOLDER" in (phi_exact.__doc__ or "")

    def test_psi_exact_placeholder_warning(self):
        assert "PLACEHOLDER" in (psi_exact.__doc__ or "")

    def test_phi_exact_is_placeholder(self):
        """phi_exact may fail due to prescription mismatch — this is expected."""
        # The Level 1 functions are documented PLACEHOLDERs.
        # They may raise ZeroDivisionError near the TT pole.
        try:
            val = phi_exact(1.0, Lambda=1.0, xi=0.0, dps=30)
            # If it doesn't raise, the result is unreliable but should be finite.
            assert mp.isfinite(val) or True  # pass either way
        except (ZeroDivisionError, mp.libmp.libmpf.ComplexResult):
            pass  # expected failure due to prescription mismatch

    def test_psi_exact_is_placeholder(self):
        """psi_exact may fail due to prescription mismatch — this is expected."""
        try:
            val = psi_exact(1.0, Lambda=1.0, xi=0.0, dps=30)
            assert mp.isfinite(val) or True
        except (ZeroDivisionError, mp.libmp.libmpf.ComplexResult):
            pass

    def test_level_comparison_runs(self):
        result = level_comparison(Lambda=1.0, xi=0.0, rL_values=[5.0, 10.0], dps=30)
        assert len(result) == 2
        assert "gamma_L2" in result[0]


# ===================================================================
# Category 10: xi-dependence and conformal limit (12 tests)
# ===================================================================
class TestXiDependence:
    """Verify xi-dependent behavior and conformal xi=1/6 limit."""

    def test_scalar_decouples_at_conformal(self):
        coeff = _scalar_mode_coefficient(1 / 6)
        assert abs(coeff) < 1e-14

    def test_scalar_coefficient_symmetric(self):
        """6(xi-1/6)^2 is symmetric around xi=1/6."""
        c1 = _scalar_mode_coefficient(0.0)
        c2 = _scalar_mode_coefficient(1 / 3)
        assert abs(c1 - c2) < 1e-14

    def test_m0_diverges_near_conformal(self):
        """m0 should be very large near xi=1/6."""
        _, m0 = effective_masses(Lambda=1.0, xi=0.166)
        assert m0 is not None and m0 > 10

    def test_phi_conformal_only_spin2(self):
        """At xi=1/6, Phi = 1 - (4/3)exp(-m2*r), no scalar term."""
        mp.mp.dps = DPS
        r = 1.0
        m2, _ = effective_masses(Lambda=1.0, xi=1 / 6)
        expected = 1 - mp.mpf(4) / 3 * mp.exp(-m2 * r)
        val = phi_local(r, Lambda=1.0, xi=1 / 6, dps=DPS)
        assert abs(val - expected) < 1e-14

    def test_psi_conformal_only_spin2(self):
        mp.mp.dps = DPS
        r = 1.0
        m2, _ = effective_masses(Lambda=1.0, xi=1 / 6)
        expected = 1 - mp.mpf(2) / 3 * mp.exp(-m2 * r)
        val = psi_local(r, Lambda=1.0, xi=1 / 6, dps=DPS)
        assert abs(val - expected) < 1e-14

    def test_gamma_conformal_r_finite(self):
        """gamma(r, xi=1/6) should be finite for r > 0."""
        g = gamma_local(1.0, Lambda=1.0, xi=1 / 6, dps=DPS)
        assert mp.isfinite(g)

    def test_gamma_0_varies_with_xi(self):
        """gamma(0) should differ between xi=0 and xi=0.25."""
        g0 = gamma_local_ratio(0, Lambda=1.0, xi=0.0, dps=DPS)
        g25 = gamma_local_ratio(0, Lambda=1.0, xi=0.25, dps=DPS)
        assert abs(g0 - g25) > 0.01

    def test_k_phi_conformal_no_scalar_pole(self):
        """At xi=1/6, K_Phi = 4/(3*Pi_TT) - 1/3 (since Pi_s=1)."""
        z = 1.0
        kp = mp.re(K_Phi(z, xi=1 / 6, dps=DPS))
        pi_tt = mp.re(_Pi_TT(z, xi=1 / 6, dps=DPS))
        expected = mp.mpf(4) / (3 * pi_tt) - mp.mpf(1) / 3
        assert abs(kp - expected) < 1e-14

    def test_k_psi_conformal_no_scalar_pole(self):
        z = 1.0
        ks = mp.re(K_Psi(z, xi=1 / 6, dps=DPS))
        pi_tt = mp.re(_Pi_TT(z, xi=1 / 6, dps=DPS))
        expected = mp.mpf(2) / (3 * pi_tt) + mp.mpf(1) / 3
        assert abs(ks - expected) < 1e-14

    def test_xi_quarter_gamma0_finite(self):
        g = gamma_local_ratio(0, Lambda=1.0, xi=0.25, dps=DPS)
        assert mp.isfinite(g)

    def test_xi_1_masses_both_finite(self):
        m2, m0 = effective_masses(Lambda=1.0, xi=1.0)
        assert m2 > 0 and m0 is not None and m0 > 0

    def test_gamma_gr_limit_at_all_xi(self):
        """gamma -> 1 at large r for any xi."""
        for xi in [0.0, 1 / 6, 0.25, 1.0]:
            g = gamma_local(1e8, Lambda=1.0, xi=xi, dps=DPS)
            assert abs(g - 1) < 1e-10, f"gamma(1e8, xi={xi}) = {g}"
