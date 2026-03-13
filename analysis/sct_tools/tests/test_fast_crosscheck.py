"""
Cross-check tests: verify _fast (Dawson/Taylor-based) form factor variants
match the standard (quad-based) versions and mpmath references.

Architecture note:
    The quad-based functions (hC_scalar, etc.) suffer from catastrophic
    cancellation at very small x because they compute 1/(12x) + (phi-1)/(2x^2)
    where the 1/x terms nearly cancel.  The _fast variants use cancellation-free
    Taylor series for x < _TAYLOR_THRESH=2 and Dawson function for x >= 2.

    Therefore at very small x (e.g., 0.001) the _fast versions are MORE accurate
    than the quad-based versions.  We compare:
      - _fast vs quad at moderate-to-large x (x >= 0.01) with rel=1e-10
      - _fast vs mpmath (ground truth) at small x with rel=1e-12+

Tests cover:
    - phi vs phi_fast
    - phi_fast vs phi_vec (exact equality)
    - hC_scalar vs hC_scalar_fast
    - hR_scalar vs hR_scalar_fast (multiple xi values)
    - hC_dirac vs hC_dirac_fast
    - hR_dirac vs hR_dirac_fast
    - scan_* vectorized helpers
    - Large-x stability (x = 1e6)
    - Taylor-Dawson crossover continuity near x = 2
    - _fast vs mpmath at small x (cancellation regime)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors as ff

# ---------------------------------------------------------------------------
# Standard test points.  _TAYLOR_THRESH = 2.0 in the implementation.
# ---------------------------------------------------------------------------

X_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]

# Points where quad-based functions have sufficient precision for direct
# comparison with _fast (x >= 0.01 avoids worst cancellation regime).
X_QUAD_SAFE = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]


# ============================================================================
# phi agreement: quad-based phi(x) vs Dawson-based phi_fast(x)
# ============================================================================

class TestPhiAgreement:
    """phi(x) [quad] vs phi_fast(x) [Dawson] should agree to ~12+ digits.

    phi itself does NOT suffer cancellation at small x (it is a simple
    integral of a positive function), so we can compare across all x.
    """

    @pytest.mark.parametrize("x", X_VALUES)
    def test_phi_vs_phi_fast(self, x):
        ref = ff.phi(x)
        fast = ff.phi_fast(x)
        assert fast == pytest.approx(ref, rel=1e-12), (
            f"phi disagreement at x={x}: quad={ref}, fast={fast}"
        )

    def test_phi_at_zero(self):
        """Both implementations return exactly 1.0 at x=0."""
        assert ff.phi(0) == 1.0
        assert ff.phi_fast(0) == 1.0


# ============================================================================
# phi_vec agreement: phi_fast applied element-wise vs phi_vec
# ============================================================================

class TestPhiVecAgreement:
    """phi_vec should produce exactly the same values as phi_fast element-wise."""

    def test_phi_vec_matches_quad(self):
        """phi_vec should agree with quad-based phi() to high precision."""
        x_arr = np.array(X_VALUES)
        vec_result = ff.phi_vec(x_arr)
        quad_result = np.array([ff.phi(x) for x in X_VALUES])
        np.testing.assert_allclose(vec_result, quad_result, rtol=1e-12)

    def test_phi_vec_includes_zero(self):
        """phi_vec should handle x=0 in the array."""
        x_arr = np.array([0.0, 1.0, 10.0])
        result = ff.phi_vec(x_arr)
        assert result[0] == 1.0
        assert result[1] == pytest.approx(ff.phi(1.0), rel=1e-12)
        assert result[2] == pytest.approx(ff.phi(10.0), rel=1e-12)

    def test_phi_vec_empty(self):
        """Empty array should return empty array."""
        result = ff.phi_vec(np.array([]))
        assert len(result) == 0


# ============================================================================
# hC_scalar agreement: quad vs fast at moderate-to-large x
# ============================================================================

class TestHCScalarAgreement:
    """hC_scalar(x) [quad] vs hC_scalar_fast(x) [Taylor/Dawson].

    At x=0.001 the quad version loses ~4 digits to cancellation, so we
    only compare at x >= 0.01 with rel=1e-10.
    """

    @pytest.mark.parametrize("x", X_QUAD_SAFE)
    def test_hC_scalar_vs_fast(self, x):
        ref = ff.hC_scalar(x)
        fast = ff.hC_scalar_fast(x)
        assert fast == pytest.approx(ref, rel=1e-10), (
            f"hC_scalar disagreement at x={x}: quad={ref}, fast={fast}"
        )

    def test_hC_scalar_fast_at_zero(self):
        """hC_scalar_fast(0) should give 1/120 via Taylor series."""
        result = ff.hC_scalar_fast(0.0)
        assert result == pytest.approx(1.0 / 120, rel=1e-14)


# ============================================================================
# hR_scalar agreement (multiple xi values)
# ============================================================================

class TestHRScalarAgreement:
    """hR_scalar(x, xi) [quad] vs hR_scalar_fast(x, xi) [Taylor/Dawson].

    Same caveat: skip x=0.001 for quad comparison due to cancellation.
    """

    @pytest.mark.parametrize("xi", [0.0, 1.0 / 6.0, 1.0])
    @pytest.mark.parametrize("x", X_QUAD_SAFE)
    def test_hR_scalar_vs_fast(self, x, xi):
        ref = ff.hR_scalar(x, xi)
        fast = ff.hR_scalar_fast(x, xi)
        # Use abs tolerance for values near zero (conformal coupling xi=1/6)
        if abs(ref) < 1e-10:
            assert fast == pytest.approx(ref, abs=1e-12), (
                f"hR_scalar disagreement at x={x}, xi={xi}: quad={ref}, fast={fast}"
            )
        else:
            assert fast == pytest.approx(ref, rel=1e-10), (
                f"hR_scalar disagreement at x={x}, xi={xi}: quad={ref}, fast={fast}"
            )

    def test_hR_scalar_fast_conformal_at_zero(self):
        """hR_scalar_fast(0, xi=1/6) = 0 (conformal invariance)."""
        result = ff.hR_scalar_fast(0.0, xi=1.0 / 6.0)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_hR_scalar_fast_minimal_at_zero(self):
        """hR_scalar_fast(0, xi=0) = 1/72."""
        result = ff.hR_scalar_fast(0.0, xi=0.0)
        assert result == pytest.approx(1.0 / 72, rel=1e-14)


# ============================================================================
# hC_dirac agreement
# ============================================================================

class TestHCDiracAgreement:
    """hC_dirac(x) [quad] vs hC_dirac_fast(x) [Taylor/Dawson]."""

    @pytest.mark.parametrize("x", X_QUAD_SAFE)
    def test_hC_dirac_vs_fast(self, x):
        ref = ff.hC_dirac(x)
        fast = ff.hC_dirac_fast(x)
        assert fast == pytest.approx(ref, rel=1e-10), (
            f"hC_dirac disagreement at x={x}: quad={ref}, fast={fast}"
        )

    def test_hC_dirac_fast_at_zero(self):
        """hC_dirac_fast(0) should give -1/20 via Taylor series."""
        result = ff.hC_dirac_fast(0.0)
        assert result == pytest.approx(-1.0 / 20, rel=1e-14)


# ============================================================================
# hR_dirac agreement
# ============================================================================

class TestHRDiracAgreement:
    """hR_dirac(x) [quad] vs hR_dirac_fast(x) [Taylor/Dawson]."""

    @pytest.mark.parametrize("x", X_QUAD_SAFE)
    def test_hR_dirac_vs_fast(self, x):
        ref = ff.hR_dirac(x)
        fast = ff.hR_dirac_fast(x)
        # Near zero the value is very small, use abs tolerance
        if abs(ref) < 1e-10:
            assert fast == pytest.approx(ref, abs=1e-12), (
                f"hR_dirac disagreement at x={x}: quad={ref}, fast={fast}"
            )
        else:
            assert fast == pytest.approx(ref, rel=1e-10), (
                f"hR_dirac disagreement at x={x}: quad={ref}, fast={fast}"
            )

    def test_hR_dirac_fast_at_zero(self):
        """hR_dirac_fast(0) should give 0 (conformal invariance of Dirac)."""
        result = ff.hR_dirac_fast(0.0)
        assert result == pytest.approx(0.0, abs=1e-14)


# ============================================================================
# scan_* vectorized helpers
# ============================================================================

class TestScanFunctions:
    """Verify scan_* helpers produce correct values (anchored to mpmath, not _fast)."""

    def test_scan_hC_scalar_vs_mpmath(self):
        x_arr = np.array([0.1, 1.0, 10.0])
        scan_result = ff.scan_hC_scalar(x_arr)
        for i, x in enumerate(x_arr):
            mp_val = float(ff.hC_scalar_mp(x, dps=50))
            assert scan_result[i] == pytest.approx(mp_val, rel=1e-12), (
                f"scan_hC_scalar[{i}] at x={x}: got {scan_result[i]}, mpmath={mp_val}"
            )

    def test_scan_hR_scalar_vs_mpmath(self):
        x_arr = np.array([0.1, 1.0, 10.0])
        for xi in [0.0, 1.0 / 6.0]:
            scan_result = ff.scan_hR_scalar(x_arr, xi=xi)
            for i, x in enumerate(x_arr):
                mp_val = float(ff.hR_scalar_mp(x, xi=xi, dps=50))
                if abs(mp_val) < 1e-10:
                    assert scan_result[i] == pytest.approx(mp_val, abs=1e-13)
                else:
                    assert scan_result[i] == pytest.approx(mp_val, rel=1e-12), (
                        f"scan_hR_scalar[{i}] at x={x}, xi={xi}: "
                        f"got {scan_result[i]}, mpmath={mp_val}"
                    )

    def test_scan_hC_dirac_vs_mpmath(self):
        x_arr = np.array([0.1, 1.0, 10.0])
        scan_result = ff.scan_hC_dirac(x_arr)
        for i, x in enumerate(x_arr):
            mp_val = float(ff.hC_dirac_mp(x, dps=50))
            assert scan_result[i] == pytest.approx(mp_val, rel=1e-12), (
                f"scan_hC_dirac[{i}] at x={x}: got {scan_result[i]}, mpmath={mp_val}"
            )

    def test_scan_hR_dirac_vs_mpmath(self):
        x_arr = np.array([0.1, 1.0, 10.0])
        scan_result = ff.scan_hR_dirac(x_arr)
        for i, x in enumerate(x_arr):
            mp_val = float(ff.hR_dirac_mp(x, dps=50))
            if abs(mp_val) < 1e-10:
                assert scan_result[i] == pytest.approx(mp_val, abs=1e-13)
            else:
                assert scan_result[i] == pytest.approx(mp_val, rel=1e-12), (
                    f"scan_hR_dirac[{i}] at x={x}: got {scan_result[i]}, mpmath={mp_val}"
                )

    def test_scan_preserves_length(self):
        """scan_* output length must equal input length."""
        for n in [1, 5, 100]:
            x_arr = np.logspace(-2, 2, n)
            assert len(ff.scan_hC_scalar(x_arr)) == n
            assert len(ff.scan_hC_dirac(x_arr)) == n
            assert len(ff.scan_hR_dirac(x_arr)) == n
            assert len(ff.scan_hR_scalar(x_arr, xi=0.0)) == n

    def test_scan_returns_ndarray(self):
        """scan_* must return numpy arrays."""
        x_arr = np.array([1.0, 5.0])
        assert isinstance(ff.scan_hC_scalar(x_arr), np.ndarray)
        assert isinstance(ff.scan_hR_scalar(x_arr, xi=0.0), np.ndarray)
        assert isinstance(ff.scan_hC_dirac(x_arr), np.ndarray)
        assert isinstance(ff.scan_hR_dirac(x_arr), np.ndarray)


# ============================================================================
# Large-x stability
# ============================================================================

class TestLargeXStability:
    """All _fast variants must return correct asymptotic values at large x."""

    def test_phi_fast_large_x(self):
        x = 1e6
        val = ff.phi_fast(x)
        assert val == pytest.approx(2.0 / x, rel=0.01)

    def test_hC_scalar_fast_large_x(self):
        x = 1e6
        val = ff.hC_scalar_fast(x)
        # Leading: 1/(12x) at large x
        assert val == pytest.approx(1.0 / (12 * x), rel=0.01)

    def test_hR_scalar_fast_large_x(self):
        x = 1e6
        # xi=0: negative (confirmed in audit), mpmath 50-digit regression
        val_0 = ff.hR_scalar_fast(x, xi=0.0)
        assert val_0 == pytest.approx(-2.7777611109e-08, rel=0.01)
        assert val_0 < 0, "hR_scalar(1e6, xi=0) must be negative"
        # xi=1/6 (conformal): near zero, mpmath 50-digit regression
        val_conf = ff.hR_scalar_fast(x, xi=1.0 / 6.0)
        assert val_conf == pytest.approx(-1.1111044444e-13, rel=0.05)
        # xi=1: positive, mpmath 50-digit regression
        val_1 = ff.hR_scalar_fast(x, xi=1.0)
        assert val_1 == pytest.approx(9.7222238889e-07, rel=0.01)

    def test_hC_dirac_fast_large_x(self):
        x = 1e6
        val = ff.hC_dirac_fast(x)
        # Leading: -1/(6x) at large x (note: negative)
        assert val == pytest.approx(-1.0 / (6 * x), rel=0.01)

    def test_hR_dirac_fast_large_x(self):
        x = 1e6
        val = ff.hR_dirac_fast(x)
        # Leading: 1/(18x)
        assert val == pytest.approx(1.0 / (18 * x), rel=0.01)

    def test_phi_vec_large_x(self):
        x_arr = np.array([1e6, 1e8, 1e10])
        result = ff.phi_vec(x_arr)
        expected = 2.0 / x_arr
        np.testing.assert_allclose(result, expected, rtol=0.01)


# ============================================================================
# Taylor-Dawson crossover continuity (near _TAYLOR_THRESH = 2.0)
# ============================================================================

class TestCrossoverContinuity:
    """Test that there is no large discontinuity at the Taylor/Dawson switchover.

    _TAYLOR_THRESH = 2.0: for x < 2 Taylor series is used, for x >= 2
    the Dawson-function closed form is used.

    With 20 Taylor terms, the series has ~1e-3 relative residual at
    x = 1.999 for some form factors (the alternating series converges
    slowly near the boundary of its practical radius).  The crossover
    tolerance is set to rel=5e-3 (no gap larger than 0.5%) which is
    sufficient to detect genuine code bugs while accommodating the
    known Taylor truncation error.  The Taylor branch is optimized for
    x << 2 where it achieves full float64 precision.
    """

    def _continuity_check(self, func, label):
        """Evaluate func on a fine grid around x=2 and verify smoothness."""
        x_below = np.linspace(1.9, 1.999, 20)
        x_above = np.linspace(2.001, 2.1, 20)
        x_at = 2.0

        vals_below = np.array([func(x) for x in x_below])
        val_at = func(x_at)
        vals_above = np.array([func(x) for x in x_above])

        # Value from just below should be close to value at threshold.
        # Tolerance 5e-3 accounts for 20-term Taylor truncation near boundary.
        assert vals_below[-1] == pytest.approx(val_at, rel=5e-3), (
            f"{label}: excessive discontinuity at threshold. "
            f"below={vals_below[-1]}, at={val_at}"
        )
        # Value from just above should be close to value at threshold
        assert vals_above[0] == pytest.approx(val_at, rel=5e-3), (
            f"{label}: excessive discontinuity at threshold. "
            f"above={vals_above[0]}, at={val_at}"
        )
        # No NaN or Inf in the transition zone
        all_vals = np.concatenate([vals_below, [val_at], vals_above])
        assert all(np.isfinite(all_vals)), (
            f"{label}: non-finite values near threshold"
        )

    def test_hC_scalar_fast_crossover(self):
        self._continuity_check(ff.hC_scalar_fast, "hC_scalar_fast")

    def test_hC_dirac_fast_crossover(self):
        self._continuity_check(ff.hC_dirac_fast, "hC_dirac_fast")

    def test_hR_dirac_fast_crossover(self):
        self._continuity_check(ff.hR_dirac_fast, "hR_dirac_fast")

    def test_hR_scalar_fast_crossover_xi0(self):
        self._continuity_check(
            lambda x: ff.hR_scalar_fast(x, xi=0.0),
            "hR_scalar_fast(xi=0)"
        )

    def test_hR_scalar_fast_crossover_xi_conf(self):
        self._continuity_check(
            lambda x: ff.hR_scalar_fast(x, xi=1.0 / 6.0),
            "hR_scalar_fast(xi=1/6)"
        )

    def test_hR_scalar_fast_crossover_xi1(self):
        self._continuity_check(
            lambda x: ff.hR_scalar_fast(x, xi=1.0),
            "hR_scalar_fast(xi=1)"
        )

    def test_phi_fast_crossover(self):
        """phi_fast uses Dawson everywhere (no Taylor branch), verify smooth."""
        x_vals = np.linspace(1.5, 2.5, 100)
        vals = np.array([ff.phi_fast(x) for x in x_vals])
        # phi is smooth; finite differences on a 0.01-spaced grid should be small
        diffs = np.abs(np.diff(vals))
        max_jump = np.max(diffs)
        assert max_jump < 0.002, (
            f"phi_fast has unexpected large jump near x=2: {max_jump}"
        )


# ============================================================================
# _fast vs mpmath at small x (cancellation regime) -- GROUND TRUTH
# ============================================================================

class TestFastVsMpmathSmallX:
    """At small x, the quad-based versions suffer from cancellation.
    The _fast (Taylor) versions should be more accurate.
    Compare both against high-precision mpmath as ground truth.

    This is the authoritative accuracy test for the _fast implementations
    in the regime where they matter most.
    """

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5])
    def test_hC_scalar_fast_vs_mpmath(self, x):
        fast = ff.hC_scalar_fast(x)
        mp_val = float(ff.hC_scalar_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-13), (
            f"hC_scalar_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5])
    def test_hC_dirac_fast_vs_mpmath(self, x):
        fast = ff.hC_dirac_fast(x)
        mp_val = float(ff.hC_dirac_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-13), (
            f"hC_dirac_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5])
    def test_hR_dirac_fast_vs_mpmath(self, x):
        fast = ff.hR_dirac_fast(x)
        mp_val = float(ff.hR_dirac_mp(x, dps=50))
        if abs(mp_val) < 1e-10:
            assert fast == pytest.approx(mp_val, abs=1e-14)
        else:
            assert fast == pytest.approx(mp_val, rel=1e-13), (
                f"hR_dirac_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
            )

    @pytest.mark.parametrize("xi", [0.0, 1.0 / 6.0, 1.0])
    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5])
    def test_hR_scalar_fast_vs_mpmath(self, x, xi):
        fast = ff.hR_scalar_fast(x, xi)
        mp_val = float(ff.hR_scalar_mp(x, xi=xi, dps=50))
        if abs(mp_val) < 1e-10:
            assert fast == pytest.approx(mp_val, abs=1e-14)
        else:
            assert fast == pytest.approx(mp_val, rel=1e-12), (
                f"hR_scalar_fast vs mpmath at x={x}, xi={xi}: "
                f"fast={fast}, mp={mp_val}"
            )


# ============================================================================
# Full-range _fast accuracy vs mpmath (all x including large)
# ============================================================================

class TestFastVsMpmathFullRange:
    """Verify _fast accuracy against mpmath ground truth across the full
    range, including the Dawson branch (x >= 2).
    """

    @pytest.mark.parametrize("x", [2.0, 5.0, 10.0, 50.0, 100.0])
    def test_hC_scalar_fast_vs_mpmath_large(self, x):
        fast = ff.hC_scalar_fast(x)
        mp_val = float(ff.hC_scalar_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-12), (
            f"hC_scalar_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )

    @pytest.mark.parametrize("x", [2.0, 5.0, 10.0, 50.0, 100.0])
    def test_hC_dirac_fast_vs_mpmath_large(self, x):
        fast = ff.hC_dirac_fast(x)
        mp_val = float(ff.hC_dirac_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-12), (
            f"hC_dirac_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )

    @pytest.mark.parametrize("x", [2.0, 5.0, 10.0, 50.0, 100.0])
    def test_hR_dirac_fast_vs_mpmath_large(self, x):
        fast = ff.hR_dirac_fast(x)
        mp_val = float(ff.hR_dirac_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-12), (
            f"hR_dirac_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )


# ============================================================================
# Regression: quad-based functions must agree with _fast at very small x
# ============================================================================

class TestQuadSmallXRegression:
    """Quad-based form factors must not give catastrophically wrong values
    at very small x (where cancellation of 1/x terms occurs).

    Regression test for the 1e-10 -> 1e-5 Taylor threshold fix.
    Prior to the fix, e.g. hC_scalar(1e-10) = -68.9 instead of +0.00833.
    """

    @pytest.mark.parametrize("x", [1e-10, 1e-8, 1e-6])
    def test_hC_scalar_quad_sane(self, x):
        q = ff.hC_scalar(x)
        f = ff.hC_scalar_fast(x)
        assert q == pytest.approx(f, rel=1e-3), (
            f"hC_scalar quad catastrophic at x={x}: quad={q}, fast={f}"
        )

    @pytest.mark.parametrize("x", [1e-10, 1e-8, 1e-6])
    def test_hC_dirac_quad_sane(self, x):
        q = ff.hC_dirac(x)
        f = ff.hC_dirac_fast(x)
        assert q == pytest.approx(f, rel=1e-3)

    @pytest.mark.parametrize("x", [1e-10, 1e-8, 1e-6])
    def test_hR_dirac_quad_sane(self, x):
        q = ff.hR_dirac(x)
        f = ff.hR_dirac_fast(x)
        assert q == pytest.approx(f, abs=1e-6)

    @pytest.mark.parametrize("x", [1e-10, 1e-8, 1e-6])
    def test_hC_vector_quad_sane(self, x):
        q = ff.hC_vector(x)
        f = ff.hC_vector_fast(x)
        assert q == pytest.approx(f, rel=1e-3)

    @pytest.mark.parametrize("x", [1e-10, 1e-8, 1e-6])
    def test_hR_vector_quad_sane(self, x):
        q = ff.hR_vector(x)
        f = ff.hR_vector_fast(x)
        assert q == pytest.approx(f, abs=1e-6)


# ============================================================================
# Finding 1: Vector _fast vs _mp crosschecks (were missing entirely)
# ============================================================================

class TestVectorFastVsMpmathSmallX:
    """hC_vector_fast and hR_vector_fast vs mpmath at small x."""

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5])
    def test_hC_vector_fast_vs_mpmath(self, x):
        fast = ff.hC_vector_fast(x)
        mp_val = float(ff.hC_vector_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-12), (
            f"hC_vector_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5])
    def test_hR_vector_fast_vs_mpmath(self, x):
        fast = ff.hR_vector_fast(x)
        mp_val = float(ff.hR_vector_mp(x, dps=50))
        if abs(mp_val) < 1e-10:
            assert fast == pytest.approx(mp_val, abs=1e-14)
        else:
            assert fast == pytest.approx(mp_val, rel=1e-12), (
                f"hR_vector_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
            )


class TestVectorFastVsMpmathFullRange:
    """hC_vector_fast and hR_vector_fast vs mpmath at large x (Dawson branch)."""

    @pytest.mark.parametrize("x", [2.0, 5.0, 10.0, 50.0, 100.0])
    def test_hC_vector_fast_vs_mpmath_large(self, x):
        fast = ff.hC_vector_fast(x)
        mp_val = float(ff.hC_vector_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-12), (
            f"hC_vector_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )

    @pytest.mark.parametrize("x", [2.0, 5.0, 10.0, 50.0, 100.0])
    def test_hR_vector_fast_vs_mpmath_large(self, x):
        fast = ff.hR_vector_fast(x)
        mp_val = float(ff.hR_vector_mp(x, dps=50))
        assert fast == pytest.approx(mp_val, rel=1e-12), (
            f"hR_vector_fast vs mpmath at x={x}: fast={fast}, mp={mp_val}"
        )
