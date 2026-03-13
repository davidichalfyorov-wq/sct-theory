"""
Audit Round 9 — Numerical precision, boundary continuity, and overflow safety.

Tests all Round-9 fixes:
  - dphi_dx_fast threshold boundary (< → <=) at x = 1e-12
  - phi_closed overflow guard at x > 2800
  - scan function variable naming (xi → x_val) — functional correctness
  - hR_scalar_fast precision warning at extreme x
  - CZ form factor threshold behavior documentation

Created: 2026-03-10
"""

import numpy as np
import pytest

from sct_tools import form_factors as ff

# ============================================================================
# dphi_dx_fast: threshold boundary fix (abs(x) < 1e-12 → abs(x) <= 1e-12)
# ============================================================================


class TestDphiDxFastThreshold:
    """dphi_dx_fast must use Taylor branch at x = 1e-12 exactly."""

    def test_at_exact_threshold(self):
        """x = 1e-12 should return -1/6 (Taylor branch), not Dawson formula."""
        result = ff.dphi_dx_fast(1e-12)
        assert result == pytest.approx(-1.0 / 6, rel=1e-10)

    def test_just_below_threshold(self):
        """x = 5e-13 should also hit Taylor branch."""
        result = ff.dphi_dx_fast(5e-13)
        assert result == pytest.approx(-1.0 / 6, rel=1e-10)

    def test_just_above_threshold(self):
        """x = 2e-12 should use Dawson branch, still accurate."""
        result = ff.dphi_dx_fast(2e-12)
        # phi'(0) = -1/6, so near-zero x should be close to -1/6
        assert result == pytest.approx(-1.0 / 6, rel=1e-3)

    def test_continuity_at_threshold(self):
        """No large discontinuity at the Taylor/Dawson boundary for dphi_dx."""
        x_below = 9e-13  # inside Taylor branch
        x_above = 2e-12  # outside Taylor branch
        val_below = ff.dphi_dx_fast(x_below)
        val_above = ff.dphi_dx_fast(x_above)
        # Both should be ~ -1/6; Dawson formula at tiny x loses some
        # precision due to (1 - phi*(1+x/2))/(2x) cancellation
        assert abs(val_below - val_above) < 1e-3

    def test_zero_exact(self):
        """phi'(0) = -1/6 exactly."""
        assert ff.dphi_dx_fast(0.0) == pytest.approx(-1.0 / 6, rel=1e-14)

    def test_negative_threshold(self):
        """Negative x must be rejected — x = Box/Lambda^2 >= 0 always."""
        with pytest.raises(ValueError, match="x >= 0"):
            ff.dphi_dx_fast(-1e-13)


# ============================================================================
# phi_closed: overflow guard at x > 2800
# ============================================================================


class TestPhiClosedOverflow:
    """phi_closed must not overflow for large x."""

    def test_x_3000(self):
        """x = 3000: erfi would overflow, but guard falls back to phi_fast."""
        result = ff.phi_closed(3000.0)
        assert np.isfinite(result)
        # phi(3000) ~ 2/3000 ≈ 6.67e-4
        assert result == pytest.approx(2.0 / 3000.0, rel=0.01)

    def test_x_5000(self):
        """x = 5000: well beyond overflow point."""
        result = ff.phi_closed(5000.0)
        assert np.isfinite(result)
        assert result == pytest.approx(2.0 / 5000.0, rel=0.01)

    def test_x_1e6(self):
        """x = 1e6: extreme value."""
        result = ff.phi_closed(1e6)
        assert np.isfinite(result)
        assert result == pytest.approx(2.0 / 1e6, rel=0.01)

    def test_x_2800_no_guard(self):
        """x = 2800: just below guard, uses original erfi formula."""
        result = ff.phi_closed(2800.0)
        assert np.isfinite(result)
        # Should match phi_fast
        expected = ff.phi_fast(2800.0)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_x_2801_guard_active(self):
        """x = 2801: just above guard, uses phi_fast fallback."""
        result = ff.phi_closed(2801.0)
        assert np.isfinite(result)
        expected = ff.phi_fast(2801.0)
        assert result == pytest.approx(expected, rel=1e-14)

    def test_guard_matches_fast(self):
        """Above threshold, phi_closed == phi_fast exactly."""
        for x in [3000, 5000, 10000, 1e8]:
            closed = ff.phi_closed(x)
            fast = ff.phi_fast(x)
            assert closed == pytest.approx(fast, rel=1e-14), (
                f"phi_closed({x}) = {closed} != phi_fast({x}) = {fast}"
            )

    def test_small_x_unchanged(self):
        """Small x still uses the original erfi formula, not the fallback."""
        # phi(1) is well-known
        result = ff.phi_closed(1.0)
        expected = ff.phi_fast(1.0)
        assert result == pytest.approx(expected, rel=1e-12)


# ============================================================================
# scan functions: correctness after xi → x_val rename
# ============================================================================


class TestScanFunctionsCorrectness:
    """Verify scan functions still produce correct results after variable rename."""

    def test_scan_hC_scalar_matches_fast(self):
        x_arr = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        scan_result = ff.scan_hC_scalar(x_arr)
        for i, x in enumerate(x_arr):
            assert scan_result[i] == pytest.approx(
                ff.hC_scalar_fast(x), rel=1e-14
            )

    def test_scan_hC_dirac_matches_fast(self):
        x_arr = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        scan_result = ff.scan_hC_dirac(x_arr)
        for i, x in enumerate(x_arr):
            assert scan_result[i] == pytest.approx(
                ff.hC_dirac_fast(x), rel=1e-14
            )

    def test_scan_hR_dirac_matches_fast(self):
        x_arr = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        scan_result = ff.scan_hR_dirac(x_arr)
        for i, x in enumerate(x_arr):
            assert scan_result[i] == pytest.approx(
                ff.hR_dirac_fast(x), rel=1e-14
            )

    def test_scan_hR_scalar_with_xi(self):
        """scan_hR_scalar must correctly pass xi parameter."""
        x_arr = np.array([0.1, 1.0, 10.0])
        for xi in [0.0, 1.0 / 6, 1.0]:
            scan_result = ff.scan_hR_scalar(x_arr, xi=xi)
            for i, x in enumerate(x_arr):
                assert scan_result[i] == pytest.approx(
                    ff.hR_scalar_fast(x, xi=xi), rel=1e-14
                ), f"Mismatch at x={x}, xi={xi}"

    def test_scan_empty_array(self):
        """Empty array should return empty result."""
        result = ff.scan_hC_scalar(np.array([]))
        assert len(result) == 0

    def test_scan_single_element(self):
        """Single-element scan should work."""
        result = ff.scan_hC_scalar(np.array([1.0]))
        assert len(result) == 1
        assert result[0] == pytest.approx(ff.hC_scalar_fast(1.0), rel=1e-14)


# ============================================================================
# hR_scalar_fast precision at extreme x (documentation, not code fix)
# ============================================================================


class TestHRScalarFastPrecisionLimits:
    """Document (not fix) precision limits of hR_scalar_fast at extreme x.

    At x > ~1e35, the multi-term Dawson formula loses precision due to
    catastrophic cancellation. In physical applications x < 1000 always.
    """

    def test_physical_range_accurate(self):
        """x in [0.01, 1000]: hR_scalar_fast matches _mp to < 1e-8 rel."""
        for x in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            fast = ff.hR_scalar_fast(x, xi=0.0)
            mp_val = float(ff.hR_scalar_mp(x, xi=0.0, dps=30))
            if abs(mp_val) > 1e-20:
                rel = abs(fast - mp_val) / abs(mp_val)
                assert rel < 1e-8, (
                    f"x={x}: fast={fast}, mp={mp_val}, rel_err={rel}"
                )
            else:
                assert abs(fast - mp_val) < 1e-20

    def test_moderate_large_x_still_ok(self):
        """x = 1e6: still reasonable accuracy (<1% rel error)."""
        fast = ff.hR_scalar_fast(1e6, xi=0.0)
        mp_val = float(ff.hR_scalar_mp(1e6, xi=0.0, dps=30))
        rel = abs(fast - mp_val) / abs(mp_val)
        assert rel < 0.01

    def test_conformal_xi_physical_range(self):
        """xi = 1/6 (conformal) in physical range: near-zero values accurate."""
        for x in [0.1, 1.0, 10.0, 100.0]:
            fast = ff.hR_scalar_fast(x, xi=1.0 / 6.0)
            mp_val = float(ff.hR_scalar_mp(x, xi=1.0 / 6.0, dps=30))
            assert abs(fast - mp_val) < 1e-10, (
                f"x={x}, xi=1/6: fast={fast}, mp={mp_val}"
            )


# ============================================================================
# CZ form factors: threshold behavior
# ============================================================================


class TestCZFormFactorThresholds:
    """CZ functions use |x| < 1e-10 hard-coded Taylor limits.

    The discontinuity at the threshold is an inherent limitation of the
    quad-based reference implementation. Verify limits are correct.
    """

    def test_f_Ric_limit(self):
        """f_Ric(0) = 1/60."""
        assert ff.f_Ric(1e-15) == pytest.approx(1.0 / 60, rel=1e-14)

    def test_f_R_limit(self):
        """f_R(0) = 1/120."""
        assert ff.f_R(1e-15) == pytest.approx(1.0 / 120, rel=1e-14)

    def test_f_RU_limit(self):
        """f_RU(0) = -1/6."""
        assert ff.f_RU(1e-15) == pytest.approx(-1.0 / 6, rel=1e-14)

    def test_f_Omega_limit(self):
        """f_Omega(0) = 1/12."""
        assert ff.f_Omega(1e-15) == pytest.approx(1.0 / 12, rel=1e-14)

    def test_f_U_at_zero(self):
        """f_U(0) = phi(0)/2 = 1/2."""
        result = ff.f_U(1e-15)
        assert result == pytest.approx(0.5, rel=1e-10)

    def test_fast_matches_cz_at_x1(self):
        """At x=1, _fast and CZ-based should agree."""
        x = 1.0
        # hC_scalar = f_Ric/2  (CZ-based)
        hC_cz = ff.hC_scalar(x)
        hC_fast = ff.hC_scalar_fast(x)
        assert hC_fast == pytest.approx(hC_cz, rel=1e-6)

    def test_hR_scalar_cz_vs_fast_at_x1(self):
        """hR_scalar (CZ quad-based) vs hR_scalar_fast at x=1."""
        x = 1.0
        for xi in [0.0, 1.0 / 6, 0.5]:
            cz = ff.hR_scalar(x, xi=xi)
            fast = ff.hR_scalar_fast(x, xi=xi)
            assert fast == pytest.approx(cz, rel=1e-6), (
                f"xi={xi}: cz={cz}, fast={fast}"
            )


# ============================================================================
# Taylor/Dawson crossover continuity for derivative functions
# ============================================================================


class TestDerivativeCrossoverContinuity:
    """Test continuity at Taylor/Dawson threshold for derivative functions.

    dhC_scalar_dx and dhR_scalar_dx use _TAYLOR_THRESH = 2.0 boundary.
    """

    def test_dhC_scalar_dx_continuity(self):
        """dhC_scalar_dx continuous at x = 2.0."""
        eps = 1e-6
        val_below = ff.dhC_scalar_dx(2.0 - eps)
        val_above = ff.dhC_scalar_dx(2.0 + eps)
        # Relative gap should be tiny
        if abs(val_below) > 1e-20:
            rel_gap = abs(val_below - val_above) / abs(val_below)
            assert rel_gap < 1e-3, (
                f"dhC discontinuity: below={val_below}, above={val_above}, "
                f"rel_gap={rel_gap}"
            )
        else:
            assert abs(val_below - val_above) < 1e-15

    def test_dhR_scalar_dx_continuity_xi0(self):
        """dhR_scalar_dx continuous at x = 2.0, xi=0."""
        eps = 1e-6
        val_below = ff.dhR_scalar_dx(2.0 - eps, xi=0.0)
        val_above = ff.dhR_scalar_dx(2.0 + eps, xi=0.0)
        if abs(val_below) > 1e-20:
            rel_gap = abs(val_below - val_above) / abs(val_below)
            assert rel_gap < 1e-3
        else:
            assert abs(val_below - val_above) < 1e-15

    def test_dhR_scalar_dx_continuity_conformal(self):
        """dhR_scalar_dx continuous at x = 2.0, xi=1/6."""
        eps = 1e-6
        xi = 1.0 / 6.0
        val_below = ff.dhR_scalar_dx(2.0 - eps, xi=xi)
        val_above = ff.dhR_scalar_dx(2.0 + eps, xi=xi)
        if abs(val_below) > 1e-20:
            rel_gap = abs(val_below - val_above) / abs(val_below)
            assert rel_gap < 1e-3
        else:
            assert abs(val_below - val_above) < 1e-15

    def test_dhC_dirac_dx_continuity(self):
        """dhC_dirac_dx continuous at x = 2.0."""
        eps = 1e-6
        val_below = ff.dhC_dirac_dx(2.0 - eps)
        val_above = ff.dhC_dirac_dx(2.0 + eps)
        rel_gap = abs(val_below - val_above) / max(abs(val_below), 1e-20)
        assert rel_gap < 1e-3, (
            f"dhC_dirac discontinuity: below={val_below}, above={val_above}"
        )

    def test_dhR_dirac_dx_continuity(self):
        """dhR_dirac_dx continuous at x = 2.0."""
        eps = 1e-6
        val_below = ff.dhR_dirac_dx(2.0 - eps)
        val_above = ff.dhR_dirac_dx(2.0 + eps)
        rel_gap = abs(val_below - val_above) / max(abs(val_below), 1e-20)
        assert rel_gap < 1e-3, (
            f"dhR_dirac discontinuity: below={val_below}, above={val_above}"
        )

    def test_dhC_vector_dx_continuity(self):
        """dhC_vector_dx continuous at x = 2.0."""
        eps = 1e-6
        val_below = ff.dhC_vector_dx(2.0 - eps)
        val_above = ff.dhC_vector_dx(2.0 + eps)
        rel_gap = abs(val_below - val_above) / max(abs(val_below), 1e-20)
        assert rel_gap < 1e-3, (
            f"dhC_vector discontinuity: below={val_below}, above={val_above}"
        )

    def test_dhR_vector_dx_continuity(self):
        """dhR_vector_dx continuous at x = 2.0."""
        eps = 1e-6
        val_below = ff.dhR_vector_dx(2.0 - eps)
        val_above = ff.dhR_vector_dx(2.0 + eps)
        rel_gap = abs(val_below - val_above) / max(abs(val_below), 1e-20)
        assert rel_gap < 1e-3, (
            f"dhR_vector discontinuity: below={val_below}, above={val_above}"
        )


# ============================================================================
# Additional numerical edge cases
# ============================================================================


class TestNumericalEdgeCases:
    """Miscellaneous numerical safety tests."""

    def test_phi_fast_subnormal(self):
        """phi_fast at subnormal float input."""
        x = 5e-324  # smallest positive subnormal float64
        result = ff.phi_fast(x)
        # phi(~0) = 1
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_phi_vec_all_zeros(self):
        """phi_vec on all-zeros array."""
        result = ff.phi_vec(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], atol=1e-12)

    def test_hC_scalar_fast_at_taylor_threshold(self):
        """hC_scalar_fast at exact Taylor threshold x=2.0."""
        # Should use Dawson branch (x < 2.0 is Taylor, x >= 2.0 is Dawson)
        result = ff.hC_scalar_fast(2.0)
        mp_ref = float(ff.hC_scalar_mp(2.0, dps=30))
        assert result == pytest.approx(mp_ref, rel=1e-6)

    def test_hR_scalar_fast_at_taylor_threshold(self):
        """hR_scalar_fast at exact Taylor threshold x=2.0."""
        result = ff.hR_scalar_fast(2.0, xi=0.0)
        mp_ref = float(ff.hR_scalar_mp(2.0, xi=0.0, dps=30))
        assert result == pytest.approx(mp_ref, rel=1e-6)

    def test_dphi_dx_fast_vs_mpmath_at_1(self):
        """dphi_dx_fast(1.0) vs mpmath finite difference."""
        from mpmath import mp, mpf

        old_dps = mp.dps
        try:
            mp.dps = 50
            h = mpf('1e-8')
            x_mp = mpf('1.0')

            def phi_mp(xv):
                return mpf(ff.phi_mp(float(xv), dps=50))

            deriv_mp = float(
                (-phi_mp(x_mp + 2 * h) + 8 * phi_mp(x_mp + h)
                 - 8 * phi_mp(x_mp - h) + phi_mp(x_mp - 2 * h)) / (12 * h)
            )
            result = ff.dphi_dx_fast(1.0)
            assert result == pytest.approx(deriv_mp, rel=1e-6)
        finally:
            mp.dps = old_dps
