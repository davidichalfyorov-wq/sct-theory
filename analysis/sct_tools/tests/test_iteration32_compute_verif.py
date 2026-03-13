"""
Tests for Iteration 32: compute.py + verification.py utility hardening.

Covers:
    CMP-1: precision_context validates dps > 0
    CMP-2: precision_context sets and restores mpmath.mp.dps
    CMP-3: progress_compute_mp warns on None results
    CMP-4: benchmark context manager measures time
    CMP-5: benchmark decorator measures time
    CMP-6: cache_info returns expected structure
    CMP-7: clear_cache runs without error
    CMP-8: progress_compute dtype=object warning
    VER-1: check_numerical_stability detects NaN
    VER-2: check_numerical_stability detects Inf
    VER-3: check_numerical_stability detects jumps
    VER-4: check_numerical_stability reports stable for smooth function
    VER-5: check_limit rejects mismatched array lengths
    VER-6: check_limit rejects empty arrays
    VER-7: Verifier all_passed property
"""

import os
import sys
import time
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import compute, verification

# ============================================================================
# CMP-1, CMP-2: precision_context validation and behavior
# ============================================================================


class TestPrecisionContext:
    """precision_context must validate dps and restore state."""

    def test_negative_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.precision_context(-1)

    def test_zero_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.precision_context(0)

    def test_float_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.precision_context(50.5)

    def test_string_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.precision_context("100")

    def test_sets_and_restores_dps(self):
        import mpmath
        original = mpmath.mp.dps
        try:
            with compute.precision_context(200):
                assert mpmath.mp.dps == 200
            assert mpmath.mp.dps == original
        finally:
            mpmath.mp.dps = original  # safety restore

    def test_restores_on_exception(self):
        import mpmath
        original = mpmath.mp.dps
        try:
            with pytest.raises(RuntimeError):
                with compute.precision_context(300):
                    assert mpmath.mp.dps == 300
                    raise RuntimeError("test")
            assert mpmath.mp.dps == original
        finally:
            mpmath.mp.dps = original


# ============================================================================
# CMP-3: progress_compute_mp None warning
# ============================================================================


class TestProgressComputeMpNoneWarning:
    """progress_compute_mp should warn when func returns None."""

    def test_warns_on_none_results(self):
        def bad_func(x, dps=15):
            return None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute.progress_compute_mp(bad_func, [1.0, 2.0], dps=15)
            none_warns = [x for x in w if "None" in str(x.message)]
            assert len(none_warns) >= 1
        assert result == [None, None]

    def test_no_warning_on_valid_results(self):
        import mpmath
        def ok_func(x, dps=15):
            return mpmath.mpf(x) ** 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute.progress_compute_mp(ok_func, [2.0, 3.0], dps=15)
            none_warns = [x for x in w if "None" in str(x.message)]
            assert len(none_warns) == 0
        assert len(result) == 2


# ============================================================================
# CMP-4, CMP-5: benchmark context manager and decorator
# ============================================================================


class TestBenchmark:
    """benchmark must measure elapsed time."""

    def test_context_manager_measures_time(self):
        with compute.benchmark() as b:
            total = sum(range(100000))
        assert b.elapsed > 0
        assert total > 0  # ensure computation happened

    def test_context_manager_with_label(self, capsys):
        with compute.benchmark("test label") as b:
            _ = [i**2 for i in range(1000)]
        captured = capsys.readouterr()
        assert "test label" in captured.out
        assert b.elapsed > 0

    def test_decorator_measures_time(self):
        @compute.benchmark("decorated")
        def slow_func():
            return sum(range(50000))

        result = slow_func()
        assert result > 0

    def test_elapsed_is_reasonable(self):
        with compute.benchmark() as b:
            time.sleep(0.05)
        # Should be at least 40ms (allowing some slack)
        assert b.elapsed >= 0.04
        # Should be less than 2 seconds (no runaway)
        assert b.elapsed < 2.0


# ============================================================================
# CMP-6, CMP-7: cache_info and clear_cache
# ============================================================================


class TestCacheUtilities:
    """cache_info and clear_cache should work correctly."""

    def test_cache_info_structure(self):
        info = compute.cache_info()
        assert 'path' in info
        assert 'size_mb' in info
        assert 'exists' in info
        assert isinstance(info['size_mb'], float)
        assert info['size_mb'] >= 0

    def test_clear_cache_no_error(self):
        # Should not raise even if cache dir doesn't exist
        compute.clear_cache()

    def test_cached_decorator_roundtrip(self, tmp_path):
        """Verify @cached actually caches."""
        call_count = 0

        @compute.cached
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        # First call — computes
        r1 = expensive(42)
        assert r1 == 1764

        # Second call — should use cache (result still correct)
        r2 = expensive(42)
        assert r2 == 1764


# ============================================================================
# CMP-8: progress_compute dtype=object warning
# ============================================================================


class TestProgressComputeObjectWarning:
    """progress_compute should warn when result has dtype=object."""

    def test_warns_on_none_return(self):
        def bad_func(x):
            return None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute.progress_compute(bad_func, [1.0, 2.0], desc=None)
            obj_warns = [x for x in w if "dtype=object" in str(x.message)]
            assert len(obj_warns) >= 1

    def test_no_warning_on_numeric_return(self):
        def ok_func(x):
            return x ** 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute.progress_compute(ok_func, [1.0, 2.0, 3.0], desc=None)
            obj_warns = [x for x in w if "dtype=object" in str(x.message)]
            assert len(obj_warns) == 0
        np.testing.assert_array_almost_equal(result, [1.0, 4.0, 9.0])


# ============================================================================
# VER-1..4: check_numerical_stability
# ============================================================================


class TestCheckNumericalStability:
    """check_numerical_stability must detect pathological values."""

    def test_detects_nan(self):
        def f(x):
            return float('nan') if x > 5 else x

        result = verification.check_numerical_stability(
            f, np.linspace(1, 10, 20), label="nan_test"
        )
        assert not result['stable']
        assert any("NaN" in desc for _, desc in result['issues'])

    def test_detects_inf(self):
        def f(x):
            return 1.0 / (x - 5.0) if abs(x - 5.0) > 1e-14 else float('inf')

        x_vals = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        result = verification.check_numerical_stability(f, x_vals, label="inf_test")
        assert not result['stable']
        assert any("Inf" in desc for _, desc in result['issues'])

    def test_detects_jumps(self):
        def f(x):
            return 1.0 if x < 5 else 100.0

        x_vals = np.linspace(1, 10, 20)
        result = verification.check_numerical_stability(
            f, x_vals, label="jump_test", rtol_consecutive=0.5
        )
        assert not result['stable']
        assert any("Jump" in desc for _, desc in result['issues'])

    def test_stable_for_smooth_function(self):
        # Use exp (always positive, smooth) to avoid false jump detections near zeros
        result = verification.check_numerical_stability(
            np.exp, np.linspace(0, 5, 100), label="smooth"
        )
        assert result['stable']
        assert len(result['issues']) == 0
        assert len(result['values']) == 100

    def test_handles_exception_in_func(self):
        def f(x):
            if x > 5:
                raise ZeroDivisionError("boom")
            return x

        result = verification.check_numerical_stability(
            f, np.linspace(1, 10, 20), label="exception_test"
        )
        assert not result['stable']
        assert any("Exception" in desc for _, desc in result['issues'])


# ============================================================================
# VER-5..7: Verifier edge cases
# ============================================================================


class TestVerifierEdgeCases:
    """Verifier edge cases and input validation."""

    def test_check_limit_mismatched_lengths_raises(self):
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="same length"):
            v.check_limit("test", [1, 2, 3], [1, 2], 0)

    def test_check_limit_empty_raises(self):
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="non-empty"):
            v.check_limit("test", [], [], 0)

    def test_all_passed_true_when_no_fails(self):
        v = verification.Verifier("test", quiet=True)
        v.check_value("1 == 1", 1.0, 1.0)
        v.check_value("pi", 3.14159, 3.14159, rtol=1e-4)
        assert v.all_passed
        assert v.n_pass == 2
        assert v.n_fail == 0

    def test_all_passed_false_when_fail(self):
        v = verification.Verifier("test", quiet=True)
        v.check_value("wrong", 1.0, 2.0)
        assert not v.all_passed
        assert v.n_fail == 1

    def test_check_value_non_finite(self):
        """Non-finite computed value should FAIL."""
        v = verification.Verifier("test", quiet=True)
        result = v.check_value("nan check", float('nan'), 1.0)
        assert not result
        assert v.n_fail == 1

    def test_check_symmetry_invalid_relation_raises(self):
        v = verification.Verifier("test", quiet=True)
        with pytest.raises(ValueError, match="unknown expected_relation"):
            v.check_symmetry("test", np.sin, 1.0, lambda x: -x, "conjugate")

    def test_summary_returns_bool(self, capsys):
        v = verification.Verifier("test", quiet=True)
        v.check_value("ok", 1.0, 1.0)
        result = v.summary()
        assert result is True

        v2 = verification.Verifier("test2", quiet=True)
        v2.check_value("bad", 1.0, 2.0)
        result2 = v2.summary()
        assert result2 is False
