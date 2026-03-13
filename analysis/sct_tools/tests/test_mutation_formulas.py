"""
Mutation tests for critical form factor formulas.

Strategy: For each _fast function, compute the CORRECT value at several test
points in the Dawson-evaluation regime (x >= 2), then verify that specific
mutations (sign flips, coefficient changes) produce DETECTABLY DIFFERENT results.

This ensures the test suite would catch typos/regressions in the exact
numerical coefficients of the physics formulas.

Test points are in [3, 10, 100] to stay firmly in the Dawson branch (x >= 2).
"""

import os
import sys

import numpy as np
import pytest
from scipy.special import dawsn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools.form_factors import (
    hC_dirac_fast,
    hC_scalar_fast,
    hC_vector_fast,
    hR_dirac_fast,
    hR_scalar_fast,
    hR_vector_fast,
    phi_fast,
)

# Test points: firmly in Dawson branch
X_POINTS = [3.0, 10.0, 50.0, 100.0]


def _phi(x):
    """Independent phi(x) via Dawson function (not importing phi_fast)."""
    sx = np.sqrt(x)
    return 2.0 * float(dawsn(sx / 2.0)) / sx


# =============================================================================
# Helper: assert mutation is detectable
# =============================================================================

def assert_mutation_detectable(correct_fn, mutant_fn, x_vals, label,
                               min_rel_diff=1e-3):
    """Assert that mutant_fn differs from correct_fn by at least min_rel_diff
    at one of the test points."""
    max_diff = 0.0
    for x in x_vals:
        c = correct_fn(x)
        m = mutant_fn(x)
        if abs(c) > 1e-30:
            rel = abs(c - m) / abs(c)
        else:
            rel = abs(c - m)
        max_diff = max(max_diff, rel)
    assert max_diff > min_rel_diff, (
        f"Mutation '{label}' NOT detectable: max relative diff = {max_diff:.2e}. "
        f"This means tests cannot catch this coefficient error."
    )


# =============================================================================
# 1. phi_fast mutations
# =============================================================================

class TestPhiFastMutations:
    """phi_fast(x) = 2 * dawsn(sqrt(x)/2) / sqrt(x)"""

    def test_factor_2_to_3(self):
        """Mutation: 2.0 -> 3.0 in front coefficient."""
        def mutant(x):
            sx = np.sqrt(x)
            return 3.0 * float(dawsn(sx / 2.0)) / sx
        assert_mutation_detectable(phi_fast, mutant, X_POINTS,
                                   "phi: 2->3 coefficient")

    def test_dawsn_arg_half_to_one(self):
        """Mutation: dawsn(sx/2) -> dawsn(sx)."""
        def mutant(x):
            sx = np.sqrt(x)
            return 2.0 * float(dawsn(sx)) / sx
        assert_mutation_detectable(phi_fast, mutant, X_POINTS,
                                   "phi: dawsn(sx/2)->dawsn(sx)")


# =============================================================================
# 2. hC_scalar_fast mutations
# =============================================================================

class TestHCScalarMutations:
    """hC_scalar(x) = 1/(12x) + (phi-1)/(2x^2)"""

    def test_sign_first_term(self):
        """Mutation: +1/(12x) -> -1/(12x)."""
        def mutant(x):
            p = _phi(x)
            return -1.0 / (12.0 * x) + (p - 1.0) / (2.0 * x * x)
        assert_mutation_detectable(hC_scalar_fast, mutant, X_POINTS,
                                   "hC_scalar: sign flip 1/(12x)")

    def test_coeff_12_to_6(self):
        """Mutation: 1/12 -> 1/6."""
        def mutant(x):
            p = _phi(x)
            return 1.0 / (6.0 * x) + (p - 1.0) / (2.0 * x * x)
        assert_mutation_detectable(hC_scalar_fast, mutant, X_POINTS,
                                   "hC_scalar: 1/12->1/6")

    def test_coeff_2_to_3(self):
        """Mutation: (phi-1)/(2x^2) -> (phi-1)/(3x^2)."""
        def mutant(x):
            p = _phi(x)
            return 1.0 / (12.0 * x) + (p - 1.0) / (3.0 * x * x)
        assert_mutation_detectable(hC_scalar_fast, mutant, X_POINTS,
                                   "hC_scalar: 1/2->1/3 in second term")

    def test_sign_second_term(self):
        """Mutation: +(phi-1) -> -(phi-1)."""
        def mutant(x):
            p = _phi(x)
            return 1.0 / (12.0 * x) - (p - 1.0) / (2.0 * x * x)
        assert_mutation_detectable(hC_scalar_fast, mutant, X_POINTS,
                                   "hC_scalar: sign flip (phi-1)")

    def test_local_limit(self):
        """Verify hC_scalar(0) = 1/120 = beta_W^(0)."""
        val = hC_scalar_fast(0.0)
        assert val == pytest.approx(1.0 / 120.0, rel=1e-12)


# =============================================================================
# 3. hR_scalar_fast mutations (xi=0 and xi=1/6)
# =============================================================================

class TestHRScalarMutations:
    """hR_scalar(x; xi) = fRic/3 + fR + xi*fRU + xi^2*fU"""

    def test_fRic_coeff_6_to_12(self):
        """Mutation in fRic: 1/(6x) -> 1/(12x)."""
        def mutant(x):
            p = _phi(x)
            fRic = 1.0 / (12.0 * x) + (p - 1.0) / (x * x)  # mutated
            fR = p / 32.0 + p / (8.0 * x) - 7.0 / (48.0 * x) - (p - 1.0) / (8.0 * x * x)
            return fRic / 3.0 + fR
        assert_mutation_detectable(
            lambda x: hR_scalar_fast(x, xi=0.0), mutant, X_POINTS,
            "hR_scalar: fRic 1/6->1/12")

    def test_fR_phi_coeff_32_to_16(self):
        """Mutation in fR: phi/32 -> phi/16."""
        def mutant(x):
            p = _phi(x)
            fRic = 1.0 / (6.0 * x) + (p - 1.0) / (x * x)
            fR = p / 16.0 + p / (8.0 * x) - 7.0 / (48.0 * x) - (p - 1.0) / (8.0 * x * x)
            return fRic / 3.0 + fR
        assert_mutation_detectable(
            lambda x: hR_scalar_fast(x, xi=0.0), mutant, X_POINTS,
            "hR_scalar: fR phi/32->phi/16")

    def test_fR_7_48_to_5_48(self):
        """Mutation in fR: 7/(48x) -> 5/(48x)."""
        def mutant(x):
            p = _phi(x)
            fRic = 1.0 / (6.0 * x) + (p - 1.0) / (x * x)
            fR = p / 32.0 + p / (8.0 * x) - 5.0 / (48.0 * x) - (p - 1.0) / (8.0 * x * x)
            return fRic / 3.0 + fR
        assert_mutation_detectable(
            lambda x: hR_scalar_fast(x, xi=0.0), mutant, X_POINTS,
            "hR_scalar: fR 7/48->5/48")

    def test_fRU_sign_flip(self):
        """Mutation: fRU = +phi/4 instead of -phi/4 (at xi=0.5)."""
        def mutant(x):
            p = _phi(x)
            fRic = 1.0 / (6.0 * x) + (p - 1.0) / (x * x)
            fR = p / 32.0 + p / (8.0 * x) - 7.0 / (48.0 * x) - (p - 1.0) / (8.0 * x * x)
            fRU = +p / 4.0 - (p - 1.0) / (2.0 * x)  # sign flipped
            fU = p / 2.0
            xi = 0.5
            return fRic / 3.0 + fR + xi * fRU + xi * xi * fU
        assert_mutation_detectable(
            lambda x: hR_scalar_fast(x, xi=0.5), mutant, X_POINTS,
            "hR_scalar: fRU sign flip at xi=0.5")

    def test_assembly_fRic_3_to_4(self):
        """Mutation in assembly: fRic/3 -> fRic/4."""
        def mutant(x):
            p = _phi(x)
            fRic = 1.0 / (6.0 * x) + (p - 1.0) / (x * x)
            fR = p / 32.0 + p / (8.0 * x) - 7.0 / (48.0 * x) - (p - 1.0) / (8.0 * x * x)
            return fRic / 4.0 + fR  # mutated: /3 -> /4
        assert_mutation_detectable(
            lambda x: hR_scalar_fast(x, xi=0.0), mutant, X_POINTS,
            "hR_scalar: assembly fRic/3->fRic/4")

    def test_local_limit_xi0(self):
        """hR_scalar(0, xi=0) = 1/72."""
        val = hR_scalar_fast(0.0, xi=0.0)
        assert val == pytest.approx(1.0 / 72.0, rel=1e-12)

    def test_local_limit_conformal(self):
        """hR_scalar(0, xi=1/6) = 0 (conformal coupling)."""
        val = hR_scalar_fast(0.0, xi=1.0 / 6.0)
        assert abs(val) < 1e-14

    def test_beta_R_formula(self):
        """beta_R^(0)(xi) = (1/2)(xi - 1/6)^2 at x=0."""
        for xi in [0.0, 0.1, 1.0 / 6.0, 0.25, 0.5, 1.0]:
            val = hR_scalar_fast(0.0, xi=xi)
            expected = 0.5 * (xi - 1.0 / 6.0) ** 2
            assert val == pytest.approx(expected, abs=1e-14)


# =============================================================================
# 4. hC_dirac_fast mutations
# =============================================================================

class TestHCDiracMutations:
    """hC_dirac(x) = (3*phi - 1)/(6x) + 2*(phi - 1)/x^2"""

    def test_sign_first_term(self):
        """Mutation: (3p - 1) -> (3p + 1)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p + 1.0) / (6.0 * x) + 2.0 * (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_dirac_fast, mutant, X_POINTS,
                                   "hC_dirac: (3p-1)->(3p+1)")

    def test_coeff_3_to_2(self):
        """Mutation: 3*phi -> 2*phi."""
        def mutant(x):
            p = _phi(x)
            return (2.0 * p - 1.0) / (6.0 * x) + 2.0 * (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_dirac_fast, mutant, X_POINTS,
                                   "hC_dirac: 3p->2p")

    def test_coeff_6_to_12(self):
        """Mutation: /(6x) -> /(12x)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p - 1.0) / (12.0 * x) + 2.0 * (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_dirac_fast, mutant, X_POINTS,
                                   "hC_dirac: /(6x)->/(12x)")

    def test_coeff_2_to_1(self):
        """Mutation: 2*(phi-1)/x^2 -> 1*(phi-1)/x^2."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p - 1.0) / (6.0 * x) + 1.0 * (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_dirac_fast, mutant, X_POINTS,
                                   "hC_dirac: 2->1 in second term")

    def test_sign_second_term(self):
        """Mutation: +2*(phi-1) -> -2*(phi-1)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p - 1.0) / (6.0 * x) - 2.0 * (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_dirac_fast, mutant, X_POINTS,
                                   "hC_dirac: sign flip second term")

    def test_local_limit(self):
        """hC_dirac(0) = -1/20 (note: negative in our sign convention)."""
        val = hC_dirac_fast(0.0)
        assert val == pytest.approx(-1.0 / 20.0, rel=1e-12)


# =============================================================================
# 5. hR_dirac_fast mutations
# =============================================================================

class TestHRDiracMutations:
    """hR_dirac(x) = (3*phi + 2)/(36x) + 5*(phi - 1)/(6x^2)"""

    def test_sign_numerator(self):
        """Mutation: (3p + 2) -> (3p - 2)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p - 2.0) / (36.0 * x) + 5.0 * (p - 1.0) / (6.0 * x * x)
        assert_mutation_detectable(hR_dirac_fast, mutant, X_POINTS,
                                   "hR_dirac: (3p+2)->(3p-2)")

    def test_coeff_36_to_18(self):
        """Mutation: /(36x) -> /(18x)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p + 2.0) / (18.0 * x) + 5.0 * (p - 1.0) / (6.0 * x * x)
        assert_mutation_detectable(hR_dirac_fast, mutant, X_POINTS,
                                   "hR_dirac: /(36x)->/(18x)")

    def test_coeff_5_to_3(self):
        """Mutation: 5*(phi-1) -> 3*(phi-1)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p + 2.0) / (36.0 * x) + 3.0 * (p - 1.0) / (6.0 * x * x)
        assert_mutation_detectable(hR_dirac_fast, mutant, X_POINTS,
                                   "hR_dirac: 5->3 in second term")

    def test_coeff_6_to_12(self):
        """Mutation: /(6x^2) -> /(12x^2)."""
        def mutant(x):
            p = _phi(x)
            return (3.0 * p + 2.0) / (36.0 * x) + 5.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_dirac_fast, mutant, X_POINTS,
                                   "hR_dirac: /(6x^2)->/(12x^2)")

    def test_local_limit(self):
        """hR_dirac(0) = 0 (conformal invariance of Dirac)."""
        val = hR_dirac_fast(0.0)
        assert abs(val) < 1e-14


# =============================================================================
# 6. hC_vector_fast mutations
# =============================================================================

class TestHCVectorMutations:
    """hC_vector(x) = phi/4 + (6*phi - 5)/(6x) + (phi - 1)/x^2"""

    def test_coeff_4_to_2(self):
        """Mutation: phi/4 -> phi/2."""
        def mutant(x):
            p = _phi(x)
            return p / 2.0 + (6.0 * p - 5.0) / (6.0 * x) + (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_vector_fast, mutant, X_POINTS,
                                   "hC_vector: phi/4->phi/2")

    def test_sign_first_term(self):
        """Mutation: +phi/4 -> -phi/4."""
        def mutant(x):
            p = _phi(x)
            return -p / 4.0 + (6.0 * p - 5.0) / (6.0 * x) + (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_vector_fast, mutant, X_POINTS,
                                   "hC_vector: sign flip phi/4")

    def test_numerator_5_to_3(self):
        """Mutation: (6p - 5) -> (6p - 3)."""
        def mutant(x):
            p = _phi(x)
            return p / 4.0 + (6.0 * p - 3.0) / (6.0 * x) + (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_vector_fast, mutant, X_POINTS,
                                   "hC_vector: (6p-5)->(6p-3)")

    def test_sign_second_term(self):
        """Mutation: +(6p-5)/(6x) -> -(6p-5)/(6x)."""
        def mutant(x):
            p = _phi(x)
            return p / 4.0 - (6.0 * p - 5.0) / (6.0 * x) + (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_vector_fast, mutant, X_POINTS,
                                   "hC_vector: sign flip second term")

    def test_sign_third_term(self):
        """Mutation: +(phi-1)/x^2 -> -(phi-1)/x^2."""
        def mutant(x):
            p = _phi(x)
            return p / 4.0 + (6.0 * p - 5.0) / (6.0 * x) - (p - 1.0) / (x * x)
        assert_mutation_detectable(hC_vector_fast, mutant, X_POINTS,
                                   "hC_vector: sign flip (phi-1)/x^2")

    def test_local_limit(self):
        """hC_vector(0) = 1/10 = beta_W^(1)."""
        val = hC_vector_fast(0.0)
        assert val == pytest.approx(1.0 / 10.0, rel=1e-12)


# =============================================================================
# 7. hR_vector_fast mutations
# =============================================================================

class TestHRVectorMutations:
    """hR_vector(x) = -phi/48 + (11 - 6*phi)/(72x) + 5*(phi - 1)/(12x^2)"""

    def test_sign_first_term(self):
        """Mutation: -phi/48 -> +phi/48."""
        def mutant(x):
            p = _phi(x)
            return +p / 48.0 + (11.0 - 6.0 * p) / (72.0 * x) + 5.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_vector_fast, mutant, X_POINTS,
                                   "hR_vector: sign flip phi/48")

    def test_coeff_48_to_24(self):
        """Mutation: phi/48 -> phi/24."""
        def mutant(x):
            p = _phi(x)
            return -p / 24.0 + (11.0 - 6.0 * p) / (72.0 * x) + 5.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_vector_fast, mutant, X_POINTS,
                                   "hR_vector: phi/48->phi/24")

    def test_numerator_11_to_7(self):
        """Mutation: (11 - 6p) -> (7 - 6p)."""
        def mutant(x):
            p = _phi(x)
            return -p / 48.0 + (7.0 - 6.0 * p) / (72.0 * x) + 5.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_vector_fast, mutant, X_POINTS,
                                   "hR_vector: (11-6p)->(7-6p)")

    def test_coeff_72_to_36(self):
        """Mutation: /(72x) -> /(36x)."""
        def mutant(x):
            p = _phi(x)
            return -p / 48.0 + (11.0 - 6.0 * p) / (36.0 * x) + 5.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_vector_fast, mutant, X_POINTS,
                                   "hR_vector: /(72x)->/(36x)")

    def test_coeff_5_to_3(self):
        """Mutation: 5*(phi-1) -> 3*(phi-1)."""
        def mutant(x):
            p = _phi(x)
            return -p / 48.0 + (11.0 - 6.0 * p) / (72.0 * x) + 3.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_vector_fast, mutant, X_POINTS,
                                   "hR_vector: 5->3 in third term")

    def test_sign_third_term(self):
        """Mutation: +5*(phi-1)/(12x^2) -> -5*(phi-1)/(12x^2)."""
        def mutant(x):
            p = _phi(x)
            return -p / 48.0 + (11.0 - 6.0 * p) / (72.0 * x) - 5.0 * (p - 1.0) / (12.0 * x * x)
        assert_mutation_detectable(hR_vector_fast, mutant, X_POINTS,
                                   "hR_vector: sign flip third term")

    def test_local_limit(self):
        """hR_vector(0) = 0 (conformal invariance of Maxwell)."""
        val = hR_vector_fast(0.0)
        assert abs(val) < 1e-14


# =============================================================================
# 8. Cross-spin consistency mutations
# =============================================================================

class TestCrossSpinMutations:
    """Verify that swapping spin form factors is detectable."""

    @pytest.mark.parametrize("x", X_POINTS)
    def test_hC_scalar_ne_dirac(self, x):
        """h_C^(0) and h_C^(1/2) must differ."""
        assert hC_scalar_fast(x) != pytest.approx(hC_dirac_fast(x), rel=1e-3)

    @pytest.mark.parametrize("x", X_POINTS)
    def test_hC_scalar_ne_vector(self, x):
        """h_C^(0) and h_C^(1) must differ."""
        assert hC_scalar_fast(x) != pytest.approx(hC_vector_fast(x), rel=1e-3)

    @pytest.mark.parametrize("x", X_POINTS)
    def test_hC_dirac_ne_vector(self, x):
        """h_C^(1/2) and h_C^(1) must differ."""
        assert hC_dirac_fast(x) != pytest.approx(hC_vector_fast(x), rel=1e-3)

    @pytest.mark.parametrize("x", X_POINTS)
    def test_hR_scalar_ne_dirac(self, x):
        """h_R^(0)(xi=0) and h_R^(1/2) must differ."""
        assert hR_scalar_fast(x, xi=0.0) != pytest.approx(hR_dirac_fast(x), rel=1e-3)

    @pytest.mark.parametrize("x", X_POINTS)
    def test_hR_scalar_ne_vector(self, x):
        """h_R^(0)(xi=0) and h_R^(1) must differ."""
        assert hR_scalar_fast(x, xi=0.0) != pytest.approx(hR_vector_fast(x), rel=1e-3)


# =============================================================================
# 9. UV asymptotic mutations
# =============================================================================

class TestUVAsymptoticMutations:
    """Verify UV asymptotics at large x match known leading behavior."""

    def test_hC_scalar_uv(self):
        """x * hC_scalar(x) -> 1/12 as x -> inf.
        phi -> 0, so hC ~ 1/(12x) - 1/(2x^2)."""
        x = 1e4
        val = x * hC_scalar_fast(x)
        assert val == pytest.approx(1.0 / 12.0, rel=0.01)
        # Mutant 1/6 would be detectable
        assert abs(val - 1.0 / 6.0) > 0.01

    def test_hR_scalar_uv_xi0(self):
        """x * hR_scalar(xi=0) -> -1/36 as x -> inf.
        Multiple cancellations: 1/18 + 1/16 - 7/48 = -1/36."""
        x = 1e4
        val = x * hR_scalar_fast(x, xi=0.0)
        assert val == pytest.approx(-1.0 / 36.0, rel=0.02)

    def test_hC_dirac_uv(self):
        """x * hC_dirac(x) -> -1/6 as x -> inf.
        phi -> 0, so hC ~ -1/(6x) - 2/x^2."""
        x = 1e4
        val = x * hC_dirac_fast(x)
        assert val == pytest.approx(-1.0 / 6.0, rel=0.01)

    def test_hR_dirac_uv(self):
        """x * hR_dirac(x) -> 1/18 as x -> inf."""
        x = 1e4
        val = x * hR_dirac_fast(x)
        assert val == pytest.approx(1.0 / 18.0, rel=0.02)

    def test_hC_vector_uv(self):
        """hC_vector ~ -1/(3x) at large x (leading term after phi/4 -> 0)."""
        x = 1e4
        # phi(1e4) ~ 2/x -> 0, so hC ~ 0 + (-5)/(6x) + (-1)/x^2 ~ -5/(6x)
        val = x * hC_vector_fast(x)
        # Leading: phi/4 * x -> 0 linearly isn't right; phi ~ 2/x so phi/4*x ~ 1/2
        # Actually x * hC = x * phi/4 + (6phi-5)/6 + (phi-1)/x
        # = x * (2/x)/4 + (12/x - 5)/6 + (2/x - 1)/x
        # = 1/2 + (-5/6 + 2/(x)) + ...
        # So x * hC_vector -> 1/2 - 5/6 = -1/3 as x -> inf
        assert val == pytest.approx(-1.0 / 3.0, rel=0.02)

    def test_hR_vector_uv(self):
        """x * hR_vector -> 1/9 as x -> inf."""
        x = 1e4
        val = x * hR_vector_fast(x)
        # Leading: x * (-phi/48) -> -1/24; (11-6phi)/72 -> 11/72;
        # 5(phi-1)/(12x) -> 0
        # x * hR = -x*phi/48 + (11-6phi)/72 + 5(phi-1)/(12x)
        # -> -(2)/(48) + 11/72 = -1/24 + 11/72 = -3/72 + 11/72 = 8/72 = 1/9
        assert val == pytest.approx(1.0 / 9.0, rel=0.02)


# =============================================================================
# 10. Taylor coefficient identity tests
# =============================================================================

class TestTaylorCoefficientIdentities:
    """Verify that Taylor coefficients reproduce known local limits."""

    def test_hC_scalar_taylor_c0(self):
        """c_0 of hC_scalar Taylor = 1/120."""
        from sct_tools.form_factors import _HC0_TAYLOR
        assert _HC0_TAYLOR[0] == pytest.approx(1.0 / 120.0, rel=1e-14)

    def test_hC_dirac_taylor_c0(self):
        """c_0 of hC_dirac Taylor = -1/20."""
        from sct_tools.form_factors import _HCD_TAYLOR
        assert _HCD_TAYLOR[0] == pytest.approx(-1.0 / 20.0, rel=1e-14)

    def test_hR_dirac_taylor_c0(self):
        """c_0 of hR_dirac Taylor = 0."""
        from sct_tools.form_factors import _HRD_TAYLOR
        assert abs(_HRD_TAYLOR[0]) < 1e-15

    def test_hC_vector_taylor_c0(self):
        """c_0 of hC_vector Taylor = 1/10."""
        from sct_tools.form_factors import _HCV_TAYLOR
        assert _HCV_TAYLOR[0] == pytest.approx(1.0 / 10.0, rel=1e-14)

    def test_hR_vector_taylor_c0(self):
        """c_0 of hR_vector Taylor = 0."""
        from sct_tools.form_factors import _HRV_TAYLOR
        assert abs(_HRV_TAYLOR[0]) < 1e-15

    def test_hC_vector_taylor_c1(self):
        """c_1 of hC_vector Taylor = -11/420 (benchmark B7)."""
        from sct_tools.form_factors import _HCV_TAYLOR
        assert _HCV_TAYLOR[1] == pytest.approx(-11.0 / 420.0, rel=1e-13)

    def test_hR_vector_taylor_c1(self):
        """c_1 of hR_vector Taylor = 1/630 (benchmark B8)."""
        from sct_tools.form_factors import _HRV_TAYLOR
        assert _HRV_TAYLOR[1] == pytest.approx(1.0 / 630.0, rel=1e-13)

    def test_hR_scalar_A_c0(self):
        """A(0) for hR_scalar = 1/72 (xi=0 limit)."""
        from sct_tools.form_factors import _HR0_A
        assert _HR0_A[0] == pytest.approx(1.0 / 72.0, rel=1e-14)

    def test_hR_scalar_B_c0(self):
        """B(0) for hR_scalar = -1/6 (fRU(0))."""
        from sct_tools.form_factors import _HR0_B
        assert _HR0_B[0] == pytest.approx(-1.0 / 6.0, rel=1e-14)

    def test_hR_scalar_C_c0(self):
        """C(0) for hR_scalar = 1/2 (fU(0))."""
        from sct_tools.form_factors import _HR0_C
        assert _HR0_C[0] == pytest.approx(1.0 / 2.0, rel=1e-14)
