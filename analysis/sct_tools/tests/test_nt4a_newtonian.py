# ruff: noqa: I001
"""
NT-4a tests: modified Newtonian potential.

ANTI-CIRCULARITY: These tests verify the Phase 4 Newtonian-potential scripts
against independently-computed reference values and physical limits.

Coverage:
  - Effective masses: m_2 and m_0 from spectral action coefficients
  - Potential ratio: V(r)/V_Newton(r) limits at large and small r
  - V_modified: finiteness at r=0, Newtonian recovery at large r
  - Scalar decoupling: conformal coupling eliminates scalar Yukawa
  - Sample curves and report generation
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend for CI/test environments

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mpmath as mp  # noqa: E402

from scripts import nt4a_newtonian as newt  # noqa: E402
from scripts.nt4a_propagator import scalar_mode_coefficient  # noqa: E402


ALPHA_C = 13 / 120
C2 = 13 / 60


# ============================================================================
# EFFECTIVE MASSES
# ============================================================================

class TestEffectiveMasses:
    """Yukawa masses from propagator denominators."""

    def test_m2_formula(self):
        """m_2 = Lambda / sqrt(c_2) = Lambda * sqrt(60/13)."""
        m2, _ = newt.effective_masses(Lambda=1.0, xi=0.0)
        expected = 1.0 / math.sqrt(C2)
        assert float(m2) == pytest.approx(expected, rel=1e-12)

    def test_m2_numerical_value(self):
        """m_2 = sqrt(60/13) ≈ 2.14834."""
        m2, _ = newt.effective_masses(Lambda=1.0)
        assert float(m2) == pytest.approx(math.sqrt(60 / 13), rel=1e-10)

    def test_m0_at_xi0(self):
        """m_0(xi=0) = Lambda / sqrt(1/6) = Lambda * sqrt(6) ≈ 2.449."""
        _, m0 = newt.effective_masses(Lambda=1.0, xi=0.0)
        assert m0 is not None
        coeff = scalar_mode_coefficient(0.0)
        expected = 1.0 / math.sqrt(float(coeff))
        assert float(m0) == pytest.approx(expected, rel=1e-12)

    def test_m0_at_conformal_is_none(self):
        """At xi=1/6, scalar mode decouples -> m_0 is None."""
        _, m0 = newt.effective_masses(Lambda=1.0, xi=1 / 6)
        assert m0 is None

    def test_m2_scales_with_lambda(self):
        """m_2 is proportional to Lambda."""
        m2_1, _ = newt.effective_masses(Lambda=1.0)
        m2_2, _ = newt.effective_masses(Lambda=2.0)
        assert float(m2_2) == pytest.approx(2.0 * float(m2_1), rel=1e-12)


# ============================================================================
# POTENTIAL RATIO V(r) / V_Newton(r)
# ============================================================================

class TestPotentialRatio:
    """V(r) / V_Newton(r) = 1 - (4/3) e^{-m_2 r} + (1/3) e^{-m_0 r}."""

    def test_large_r_approaches_one(self):
        """At r >> 1/m_2, ratio -> 1."""
        ratio = newt.potential_ratio(1e4, Lambda=1.0, xi=0.0, dps=50)
        assert float(ratio) == pytest.approx(1.0, abs=1e-3)

    def test_small_r_deviation_from_newton(self):
        """At r << 1/m_2, ratio deviates significantly from 1."""
        ratio = newt.potential_ratio(0.01, Lambda=1.0, xi=0.0, dps=50)
        assert abs(float(ratio) - 1.0) > 0.1

    def test_ratio_at_r_zero_limit(self):
        """At r->0, Yukawa approximation gives: 1 - 4/3 + 1/3 = 0."""
        ratio = newt.potential_ratio(1e-10, Lambda=1.0, xi=0.0, dps=50)
        assert float(ratio) == pytest.approx(0.0, abs=0.01)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError, match="positive"):
            newt.potential_ratio(-1.0)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError, match="positive"):
            newt.potential_ratio(0.0)

    def test_ratio_conformal_only_spin2(self):
        """At xi=1/6, only spin-2 Yukawa: ratio = 1 - (4/3) exp(-m_2 r)."""
        r = 1.0
        ratio = float(newt.potential_ratio(r, Lambda=1.0, xi=1 / 6, dps=50))
        m2, _ = newt.effective_masses(Lambda=1.0, xi=1 / 6)
        expected = 1 - 4 / 3 * math.exp(-float(m2) * r)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_ratio_conformal_short_distance_tends_to_minus_one_third(self):
        """At exact conformal coupling, the scalar cancellation is absent."""
        ratio = float(newt.potential_ratio(1e-10, Lambda=1.0, xi=1 / 6, dps=50))
        assert ratio == pytest.approx(-1 / 3, abs=1e-3)

    def test_phi_local_ratio_matches_closed_form(self):
        r = 1.0
        phi = float(newt.phi_local_ratio(r, Lambda=1.0, xi=0.0, dps=50))
        m2, m0 = newt.effective_masses(Lambda=1.0, xi=0.0)
        expected = 1 - 4 / 3 * math.exp(-float(m2) * r) + 1 / 3 * math.exp(-float(m0) * r)
        assert phi == pytest.approx(expected, rel=1e-10)

    def test_psi_local_ratio_matches_closed_form(self):
        r = 1.0
        psi = float(newt.psi_local_ratio(r, Lambda=1.0, xi=0.0, dps=50))
        m2, m0 = newt.effective_masses(Lambda=1.0, xi=0.0)
        expected = 1 - 2 / 3 * math.exp(-float(m2) * r) - 1 / 3 * math.exp(-float(m0) * r)
        assert psi == pytest.approx(expected, rel=1e-10)

    def test_gamma_local_ratio_matches_phi_psi_ratio(self):
        r = 1.0
        gamma = float(newt.gamma_local_ratio(r, Lambda=1.0, xi=0.0, dps=50))
        phi = float(newt.phi_local_ratio(r, Lambda=1.0, xi=0.0, dps=50))
        psi = float(newt.psi_local_ratio(r, Lambda=1.0, xi=0.0, dps=50))
        assert gamma == pytest.approx(psi / phi, rel=1e-12)


# ============================================================================
# MODIFIED POTENTIAL V(r)
# ============================================================================

class TestVModified:
    """Full modified potential V(r)."""

    def test_finite_at_origin(self):
        """V(r=0) is finite (resolves the 1/r singularity)."""
        val = newt.V_modified(1e-6, Lambda=1.0, xi=0.0, G=1.0, M=1.0, dps=50)
        assert math.isfinite(float(val))

    def test_finite_at_origin_value(self):
        """V(0) = -G*M*(4/3 m_2 - 1/3 m_0) for xi != 1/6."""
        val = float(newt.small_r_limit_potential(Lambda=1.0, xi=0.0, G=1.0, M=1.0, dps=50))
        m2, m0 = newt.effective_masses(Lambda=1.0, xi=0.0)
        expected = -(4 / 3 * float(m2) - 1 / 3 * float(m0))
        assert val == pytest.approx(expected, rel=1e-10)

    def test_newtonian_recovery_large_r(self):
        """V(r) -> -G*M/r for r >> 1/Lambda."""
        r = 1e4
        V = float(newt.V_modified(r, Lambda=1.0, G=1.0, M=1.0, dps=50))
        V_newton = -1.0 / r
        assert V == pytest.approx(V_newton, rel=1e-2)

    def test_potential_negative(self):
        """Gravitational potential is attractive (V < 0) for r > 0."""
        for r in [0.1, 1.0, 10.0, 100.0]:
            val = float(newt.V_modified(r, Lambda=1.0, G=1.0, M=1.0, dps=50))
            assert val < 0, f"V({r}) = {val} should be negative"

    def test_potential_scales_with_G_M(self):
        """V is proportional to G*M."""
        V1 = float(newt.V_modified(1.0, G=1.0, M=1.0, dps=50))
        V2 = float(newt.V_modified(1.0, G=2.0, M=3.0, dps=50))
        assert V2 == pytest.approx(6.0 * V1, rel=1e-10)

    def test_conformal_coupling_reopens_short_distance_divergence(self):
        """At xi=1/6, the local Yukawa approximation diverges as +GM/(3r)."""
        limit = newt.small_r_limit_potential(Lambda=1.0, xi=1 / 6, G=1.0, M=1.0, dps=50)
        assert limit == mp.inf
        val = float(newt.V_modified(1e-8, Lambda=1.0, xi=1 / 6, G=1.0, M=1.0, dps=80))
        assert val > 0
        assert val == pytest.approx(1.0 / (3e-8), rel=1e-3)


# ============================================================================
# SAMPLE CURVES
# ============================================================================

class TestSampleCurves:
    """Potential sampling and curve generation."""

    def test_sample_curve_length(self):
        radii = [0.1, 1.0, 10.0, 100.0]
        samples = newt.sample_potential_curve(radii, Lambda=1.0, xi=0.0)
        assert len(samples) == 4

    def test_sample_curve_has_required_keys(self):
        radii = [1.0]
        samples = newt.sample_potential_curve(radii)
        assert "r" in samples[0]
        assert "V" in samples[0]
        assert "ratio" in samples[0]

    def test_sample_ratio_converges(self):
        """Ratio approaches 1 at large r."""
        radii = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        samples = newt.sample_potential_curve(radii, Lambda=1.0, xi=0.0)
        ratios = [s["ratio"] for s in samples]
        assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)


# ============================================================================
# REPORT GENERATION
# ============================================================================

class TestReportGeneration:
    """Newtonian potential report and figures."""

    def test_report_has_correct_structure(self, tmp_path):
        output = tmp_path / "test_newtonian.json"
        report = newt.generate_newtonian_report(output_path=output)
        assert output.exists()
        assert "phase" in report
        assert report["phase"] == "NT-4a"
        assert "curves" in report
        assert "Lambda" in report

    def test_report_contains_three_xi_values(self, tmp_path):
        output = tmp_path / "test_newtonian2.json"
        report = newt.generate_newtonian_report(output_path=output)
        assert "0.0" in report["curves"]
        assert len(report["curves"]) == 3

    def test_report_custom_xi_values(self, tmp_path):
        output = tmp_path / "test_newtonian3.json"
        report = newt.generate_newtonian_report(
            xi_values=[0.0, 0.5], output_path=output
        )
        assert len(report["curves"]) == 2


# ============================================================================
# CONSISTENCY CHECKS
# ============================================================================

class TestConsistency:
    """Cross-checks between effective-mass and potential modules."""

    def test_small_r_equals_V_modified_at_tiny_r(self):
        """small_r_limit_potential should match V_modified(tiny r)."""
        limit = float(newt.small_r_limit_potential(Lambda=1.0, xi=0.0, dps=80))
        direct = float(newt.V_modified(1e-7, Lambda=1.0, xi=0.0, dps=80))
        assert limit == pytest.approx(direct, rel=1e-3)

    def test_m2_positive(self):
        """m_2 > 0 for all Lambda > 0."""
        m2, _ = newt.effective_masses(Lambda=1.0)
        assert float(m2) > 0

    def test_m0_positive_when_exists(self):
        """m_0 > 0 when it exists (xi != 1/6)."""
        _, m0 = newt.effective_masses(Lambda=1.0, xi=0.0)
        assert m0 is not None
        assert float(m0) > 0
