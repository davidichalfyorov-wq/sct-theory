# ruff: noqa: I001
"""
NT-2 tests: entire-function diagnostics and complex-plane evaluators.

ANTI-CIRCULARITY: These tests verify the Phase 4 scripts against
independently-computed reference values.  They are regression + cross-check
tests, not derivation evidence.

Coverage:
  - phi(z) complex domain: known values, Taylor series agreement, symmetry
  - Per-spin form factors: local limits, pole cancellation, series vs closed-form
  - Total SM form factors F1, F2: known values, xi-dependence
  - Growth rate: order rho=1, type sigma=1/4 proxies
  - Zero search: no positive-real-axis zeros of the legacy NT-2 proxy
  - Hadamard: Pi_entire is entire (no poles) as a proxy object
  - alpha_C, alpha_R canonical values
  - Serialization: snapshot generation
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mpmath as mp  # noqa: E402

from scripts import nt2_entire_function as nt2  # noqa: E402
from scripts import nt2_hadamard  # noqa: E402
from sct_tools import entire_function as ef  # noqa: E402
from sct_tools import form_factors  # noqa: E402


# ============================================================================
# CANONICAL CONSTANTS
# ============================================================================

ALPHA_C = 13 / 120  # = 0.10833...
N_S = 4
N_F = 45
N_V = 12
N_D = N_F / 2  # = 22.5


# ============================================================================
# PHI COMPLEX DOMAIN
# ============================================================================

class TestPhiComplex:
    """phi(z) evaluated in the complex plane."""

    def test_phi_zero(self):
        """phi(0) = 1 exactly."""
        val = ef.phi_complex_mp(0, dps=80)
        assert float(val.real) == pytest.approx(1.0, abs=1e-30)
        assert float(val.imag) == pytest.approx(0.0, abs=1e-30)

    def test_matches_real_axis_reference(self):
        """phi_complex_mp agrees with form_factors.phi_mp on the real axis."""
        for x in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
            expected = form_factors.phi_mp(x, dps=80)
            actual = ef.phi_complex_mp(x, dps=80)
            assert float(actual.real) == pytest.approx(float(expected), rel=1e-12)
            assert float(actual.imag) == pytest.approx(0.0, abs=1e-12)

    def test_phi_purely_imaginary(self):
        """phi(iy) has well-defined real and imaginary parts."""
        val = ef.phi_complex_mp(1j, dps=80)
        assert math.isfinite(float(val.real))
        assert math.isfinite(float(val.imag))

    def test_phi_conjugation_symmetry(self):
        """phi(conj(z)) = conj(phi(z)) because Taylor coefficients are real."""
        z = 3.0 + 2.0j
        val = ef.phi_complex_mp(z, dps=80)
        val_conj = ef.phi_complex_mp(z.conjugate(), dps=80)
        assert float(abs(val.conjugate() - val_conj)) < 1e-20

    def test_phi_series_agrees_with_closed_form(self):
        """Taylor series of phi(z) matches closed-form for small |z|."""
        for z in [0.1, 0.3, 0.1 + 0.1j]:
            series = nt2.phi_series(z, n_terms=40, dps=80)
            closed = ef.phi_complex_mp(z, dps=80)
            assert float(abs(series - closed)) < 1e-20

    def test_phi_large_z_decays(self):
        """phi(x) -> 0 as x -> +inf along the real axis."""
        val_100 = float(ef.phi_complex_mp(100.0, dps=80).real)
        val_1000 = float(ef.phi_complex_mp(1000.0, dps=80).real)
        assert abs(val_1000) < abs(val_100)


class TestPhiTaylorCoefficients:
    """phi(z) Taylor coefficients a_n = (-1)^n n! / (2n+1)!."""

    def test_a0(self):
        assert float(nt2.phi_series_coefficient(0)) == pytest.approx(1.0, abs=1e-30)

    def test_a1(self):
        """a_1 = -1/6 => phi'(0) = -1/6."""
        assert float(nt2.phi_series_coefficient(1)) == pytest.approx(-1 / 6, rel=1e-14)

    def test_a2(self):
        """a_2 = (-1)^2 * 2! / 5! = 2/120 = 1/60."""
        assert float(nt2.phi_series_coefficient(2)) == pytest.approx(1 / 60, rel=1e-14)

    def test_alternating_sign(self):
        """Coefficients alternate in sign: a_n = (-1)^n |a_n|."""
        for n in range(10):
            coeff = float(nt2.phi_series_coefficient(n))
            assert (coeff > 0) == (n % 2 == 0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            nt2.phi_series_coefficient(-1)


# ============================================================================
# PER-SPIN FORM FACTORS: LOCAL LIMITS (POLE CANCELLATION)
# ============================================================================

class TestPoleCancellation:
    """z=0 limits of per-spin form factors (apparent poles cancel)."""

    def test_all_local_limits_are_finite(self):
        """All 8 form factors have finite z->0 limits to < 1e-10 error."""
        report = {e.name: e for e in nt2.pole_cancellation_report(dps=80)}
        for name, entry in report.items():
            assert entry.absolute_error < 1e-10, f"{name}: error = {entry.absolute_error}"

    def test_hC_scalar_local_limit(self):
        """h_C^(0)(0) = 1/120."""
        val = float(nt2.hC_scalar_complex(0, dps=80).real)
        assert val == pytest.approx(1 / 120, rel=1e-12)

    def test_hR_scalar_conformal_limit(self):
        """h_R^(0)(0, xi=1/6) = 0."""
        val = float(nt2.hR_scalar_complex(0, xi=1 / 6, dps=80).real)
        assert val == pytest.approx(0.0, abs=1e-14)

    def test_hR_scalar_minimal_limit(self):
        """h_R^(0)(0, xi=0) = 1/72."""
        val = float(nt2.hR_scalar_complex(0, xi=0, dps=80).real)
        assert val == pytest.approx(1 / 72, rel=1e-12)

    def test_hC_dirac_local_limit(self):
        """h_C^(1/2)(0) = -1/20 (sign from CZ definition)."""
        val = float(nt2.hC_dirac_complex(0, dps=80).real)
        assert val == pytest.approx(-1 / 20, rel=1e-12)

    def test_hR_dirac_local_limit(self):
        """h_R^(1/2)(0) = 0 (conformal invariance of massless Dirac)."""
        val = float(nt2.hR_dirac_complex(0, dps=80).real)
        assert val == pytest.approx(0.0, abs=1e-14)

    def test_hC_vector_local_limit(self):
        """h_C^(1)(0) = 1/10 (corrected ghost counting)."""
        val = float(nt2.hC_vector_complex(0, dps=80).real)
        assert val == pytest.approx(1 / 10, rel=1e-12)

    def test_hR_vector_local_limit(self):
        """h_R^(1)(0) = 0."""
        val = float(nt2.hR_vector_complex(0, dps=80).real)
        assert val == pytest.approx(0.0, abs=1e-14)

    def test_pole_cancellation_xi1(self):
        """Pole cancellation also works at xi=1."""
        report = {e.name: e for e in nt2.pole_cancellation_report(xi=1.0, dps=80)}
        assert report["hR_scalar"].absolute_error < 1e-10


# ============================================================================
# PER-SPIN FORM FACTORS: COMPLEX-DOMAIN CROSS-CHECKS
# ============================================================================

class TestPerSpinComplex:
    """Per-spin form factors at nonzero z."""

    @pytest.mark.parametrize("z", [0.1, 1.0, 5.0, 10.0])
    def test_hC_scalar_is_real_on_real_axis(self, z):
        val = nt2.hC_scalar_complex(z, dps=80)
        assert abs(float(val.imag)) < 1e-20

    @pytest.mark.parametrize("z", [0.1, 1.0, 5.0, 10.0])
    def test_hC_dirac_is_real_on_real_axis(self, z):
        val = nt2.hC_dirac_complex(z, dps=80)
        assert abs(float(val.imag)) < 1e-20

    @pytest.mark.parametrize("z", [0.1, 1.0, 5.0, 10.0])
    def test_hC_vector_is_real_on_real_axis(self, z):
        val = nt2.hC_vector_complex(z, dps=80)
        assert abs(float(val.imag)) < 1e-20

    def test_hC_scalar_positive_for_positive_x(self):
        """h_C^(0)(x) > 0 for x > 0."""
        for x in [0.1, 1.0, 10.0, 100.0]:
            assert float(nt2.hC_scalar_complex(x, dps=80).real) > 0

    def test_hC_dirac_negative_near_zero(self):
        """h_C^(1/2)(x) < 0 near x=0 (fermionic sign)."""
        assert float(nt2.hC_dirac_complex(0.01, dps=80).real) < 0

    def test_conjugation_symmetry_hC_vector(self):
        """h_C^(1)(conj(z)) = conj(h_C^(1)(z))."""
        z = 2.0 + 1.0j
        val = nt2.hC_vector_complex(z, dps=80)
        val_conj = nt2.hC_vector_complex(z.conjugate(), dps=80)
        assert float(abs(val.conjugate() - val_conj)) < 1e-18


# ============================================================================
# TOTAL SM FORM FACTORS
# ============================================================================

class TestTotalFormFactors:
    """SM-summed F1(z), F2(z)."""

    def test_F1_zero_canonical(self):
        """F1(0) = alpha_C / (16 pi^2) = 13 / (1920 pi^2)."""
        val = float(ef.F1_total_complex(0, dps=80).real)
        expected = ALPHA_C / (16 * math.pi**2)
        assert val == pytest.approx(expected, rel=1e-12)

    def test_F1_zero_numerical_value(self):
        """F1(0) ≈ 6.860288e-04."""
        val = float(ef.F1_total_complex(0, dps=80).real)
        assert val == pytest.approx(6.860288475783305e-04, rel=1e-8)

    def test_F2_zero_xi0(self):
        """F2(0, xi=0) ≈ 3.518097e-04."""
        val = float(ef.F2_total_complex(0, xi=0, dps=80).real)
        assert val == pytest.approx(3.518096654247839e-04, rel=1e-8)

    def test_F2_zero_conformal(self):
        """F2(0, xi=1/6) = 0 (scalar mode decouples)."""
        val = float(ef.F2_total_complex(0, xi=1 / 6, dps=80).real)
        assert val == pytest.approx(0.0, abs=1e-14)

    def test_F1_is_real_on_real_axis(self):
        for z in [0.1, 1.0, 5.0, 10.0]:
            val = ef.F1_total_complex(z, dps=80)
            assert abs(float(val.imag)) < 1e-18

    def test_F2_is_real_on_real_axis(self):
        for z in [0.1, 1.0, 5.0, 10.0]:
            val = ef.F2_total_complex(z, xi=0.0, dps=80)
            assert abs(float(val.imag)) < 1e-18

    def test_F1_xi_independence(self):
        """F1 is independent of xi (Weyl^2 does not depend on scalar coupling)."""
        val_0 = float(ef.F1_total_complex(1.0, xi=0.0, dps=80).real)
        val_conf = float(ef.F1_total_complex(1.0, xi=1 / 6, dps=80).real)
        val_1 = float(ef.F1_total_complex(1.0, xi=1.0, dps=80).real)
        assert val_0 == pytest.approx(val_conf, rel=1e-12)
        assert val_0 == pytest.approx(val_1, rel=1e-12)

    def test_F1_at_z1(self):
        """F1(1) reference value from Phase 3 handoff."""
        val = float(ef.F1_total_complex(1.0, dps=80).real)
        assert val == pytest.approx(-3.18296e-04, rel=1e-3)

    def test_F2_at_z1_xi0(self):
        """F2(1, xi=0) reference value."""
        val = float(ef.F2_total_complex(1.0, xi=0.0, dps=80).real)
        assert val == pytest.approx(4.31261e-04, rel=1e-3)

    def test_SM_counting(self):
        """Verify SM field counting: N_s=4, N_D=22.5, N_v=12."""
        assert nt2.N_S == 4
        assert float(nt2.N_D) == pytest.approx(22.5)
        assert nt2.N_V == 12


# ============================================================================
# ALPHA_C AND ALPHA_R
# ============================================================================

class TestAlphaCoefficients:
    """Total spectral action coefficients."""

    def test_alpha_C(self):
        """alpha_C = 13/120 (parameter-free, xi-independent)."""
        assert float(nt2.ALPHA_C) == pytest.approx(ALPHA_C, rel=1e-14)

    def test_alpha_R_conformal(self):
        """alpha_R(1/6) = 0."""
        assert float(nt2.alpha_R(1 / 6)) == pytest.approx(0.0, abs=1e-14)

    def test_alpha_R_minimal(self):
        """alpha_R(0) = 2*(0 - 1/6)^2 = 1/18."""
        assert float(nt2.alpha_R(0)) == pytest.approx(1 / 18, rel=1e-14)

    def test_alpha_R_xi1(self):
        """alpha_R(1) = 2*(1 - 1/6)^2 = 25/18."""
        assert float(nt2.alpha_R(1)) == pytest.approx(25 / 18, rel=1e-14)

    def test_alpha_C_from_individual_spins(self):
        """alpha_C = N_s * 1/120 + N_D * (-1/20) + N_v * 1/10
        = 4/120 - 22.5/20 + 12/10 = 4/120 - 135/120 + 144/120 = 13/120."""
        val = N_S / 120 + N_D * (-1 / 20) + N_V * (1 / 10)
        assert val == pytest.approx(ALPHA_C, rel=1e-14)


# ============================================================================
# GROWTH RATE (ORDER AND TYPE)
# ============================================================================

class TestGrowthRate:
    """Growth-rate diagnostics for entire-function property."""

    def test_F1_order_near_one(self):
        report = ef.estimate_growth_rate(
            lambda z: ef.F1_total_complex(z, dps=60), dps=60
        )
        assert 0.5 <= report["order"] <= 1.5

    def test_F2_order_near_one(self):
        report = ef.estimate_growth_rate(
            lambda z: ef.F2_total_complex(z, dps=60), dps=60
        )
        assert 0.5 <= report["order"] <= 1.5

    def test_F1_type_positive(self):
        report = ef.estimate_growth_rate(
            lambda z: ef.F1_total_complex(z, dps=60), dps=60
        )
        assert report["type"] > 0.0

    def test_growth_returns_samples(self):
        report = ef.estimate_growth_rate(
            lambda z: ef.F1_total_complex(z, dps=60), dps=60
        )
        assert "samples" in report
        assert len(report["samples"]) == 5  # default 5 radii


# ============================================================================
# ZERO SEARCH
# ============================================================================

class TestZeroSearch:
    """Search for zeros of the legacy NT-2 proxy on the real axis."""

    def test_no_positive_real_axis_zeros_of_proxy(self):
        """The legacy NT-2 proxy has no real positive zeros on the scan window."""
        roots = ef.find_real_axis_zeros(
            lambda z: nt2_hadamard.Pi_entire(z, dps=80),
            interval=(0.0, 200.0),
            dps=80,
        )
        assert roots == []

    def test_Pi_entire_at_zero_near_one(self):
        """The legacy proxy is normalized near 1 at the origin."""
        val = float(nt2_hadamard.Pi_entire(0, dps=80).real)
        assert val == pytest.approx(1.0, abs=1e-3)

    def test_Pi_entire_positive_at_moderate_z(self):
        """The legacy proxy stays positive for sampled z in [0, 50]."""
        for z in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
            val = float(nt2_hadamard.Pi_entire(z, dps=80).real)
            assert val > 0, f"Pi({z}) = {val} is not positive"


# ============================================================================
# SERIALIZATION
# ============================================================================

class TestSerialization:
    """Snapshot JSON generation."""

    def test_snapshot_generates_file(self, tmp_path):
        output = tmp_path / "test_snapshot.json"
        path = nt2.serialize_nt2_snapshot(output, xi=0.0, dps=50)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["phase"] == "NT-2"
        assert "pole_cancellation" in data
        assert "growth_F1" in data
        assert "growth_F2" in data

    def test_snapshot_has_correct_number_of_pole_entries(self, tmp_path):
        output = tmp_path / "test_snapshot2.json"
        nt2.serialize_nt2_snapshot(output, xi=0.0, dps=50)
        import json
        data = json.loads(output.read_text())
        assert len(data["pole_cancellation"]) == 8


# ============================================================================
# UV ASYMPTOTIC
# ============================================================================

class TestUVAsymptotic:
    """Large-z (UV) behaviour of the total form factor."""

    def test_x_alpha_C_asymptotic(self):
        """x * alpha_C(x) -> -89/12 as x -> infinity (from NT-1b Phase 3)."""
        x = 5000.0
        hC_s = float(nt2.hC_scalar_complex(x, dps=60).real)
        hC_d = float(nt2.hC_dirac_complex(x, dps=60).real)
        hC_v = float(nt2.hC_vector_complex(x, dps=60).real)
        total_hC = N_S * hC_s + N_D * hC_d + N_V * hC_v
        product = x * total_hC
        assert product == pytest.approx(-89 / 12, rel=0.02)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Input validation and edge cases."""

    def test_dps_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            nt2._set_dps(0)

    def test_find_real_axis_zeros_invalid_interval(self):
        with pytest.raises(ValueError, match="invalid interval"):
            nt2.find_real_axis_zeros(lambda z: z, interval=(10.0, 5.0))

    def test_growth_rate_insufficient_data(self):
        """Constant function doesn't grow -> raises ValueError."""
        with pytest.raises(ValueError, match="not enough"):
            nt2.estimate_growth_rate(lambda z: mp.mpc(1), dps=30)
