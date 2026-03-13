"""
Tests for Iteration 26: verification.py + lean.py hardening.

Covers:
    V1: check_value NaN/Inf guard
    V2: check_limit empty/mismatched array validation
    V3: check_limit denormalized target fallback
    V4: check_symmetry invalid expected_relation raises ValueError
    L1: _validate_lean_name accepts/rejects identifiers
    L2: set_api_key rejects non-string/empty
    L3: _has_sorry word-boundary detection
    L4: verify_identity input validation
    L5: verify_deep both-backends-disabled guard
    L6: physlean_anomaly_proof empty charges guard
    L7: prove() empty code validation
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import lean, verification

# ============================================================================
# V1: check_value NaN/Inf guard
# ============================================================================


class TestCheckValueNanInf:
    """check_value must return False for NaN/Inf inputs without crashing."""

    def setup_method(self):
        self.v = verification.Verifier("nan_test", quiet=True)

    def test_nan_computed_returns_false(self):
        assert not self.v.check_value("nan", float('nan'), 1.0)

    def test_nan_expected_returns_false(self):
        assert not self.v.check_value("nan", 1.0, float('nan'))

    def test_both_nan_returns_false(self):
        assert not self.v.check_value("nan", float('nan'), float('nan'))

    def test_inf_computed_returns_false(self):
        assert not self.v.check_value("inf", float('inf'), 1.0)

    def test_neg_inf_expected_returns_false(self):
        assert not self.v.check_value("inf", 1.0, float('-inf'))

    def test_finite_values_still_pass(self):
        assert self.v.check_value("ok", 1.0, 1.0)

    def test_near_zero_with_combined_tolerance(self):
        """Combined tolerance: |c - e| <= atol + rtol * |e|."""
        # e=1e-16, c=2e-16: diff=1e-16, atol=1e-15 >> diff -> passes
        assert self.v.check_value("small", 2e-16, 1e-16, atol=1e-15)


# ============================================================================
# V2: check_limit empty/mismatched array validation
# ============================================================================


class TestCheckLimitValidation:
    """check_limit must reject empty and mismatched arrays."""

    def setup_method(self):
        self.v = verification.Verifier("limit_test", quiet=True)

    def test_empty_x_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self.v.check_limit("empty", [], [1.0], 0.0)

    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self.v.check_limit("empty", [1.0], [], 0.0)

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self.v.check_limit("empty", [], [], 0.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            self.v.check_limit("mismatch", [1, 2, 3], [1, 2], 0.0)

    def test_valid_arrays_pass(self):
        """Normal convergence case should still work."""
        x = [10, 100, 1000]
        y = [0.1, 0.01, 0.001]
        assert self.v.check_limit("ok", x, y, 0.0, rtol=0.01)


# ============================================================================
# V3: check_limit denormalized target
# ============================================================================


class TestCheckLimitDenormTarget:
    """check_limit uses absolute error for denormalized targets."""

    def setup_method(self):
        self.v = verification.Verifier("denorm_test", quiet=True)

    def test_very_small_target_no_overflow(self):
        """Target ~1e-310 should not overflow relative error."""
        target = 1e-310
        y = [target * 1.5, target * 1.1, target * 1.001]
        # With absolute error this should work (errors are ~1e-310 scale)
        result = self.v.check_limit("denorm", [1, 2, 3], y, target, rtol=0.01)
        # Just verify no crash (overflow would produce inf)
        assert result is not None


# ============================================================================
# V4: check_symmetry invalid expected_relation
# ============================================================================


class TestCheckSymmetryValidation:
    """check_symmetry raises ValueError for invalid expected_relation."""

    def setup_method(self):
        self.v = verification.Verifier("sym_test", quiet=True)

    def test_invalid_relation_raises(self):
        with pytest.raises(ValueError, match="unknown expected_relation"):
            self.v.check_symmetry("bad", lambda x: x, 1.0, lambda x: x,
                                  expected_relation="negate")

    def test_typo_relation_raises(self):
        with pytest.raises(ValueError, match="unknown expected_relation"):
            self.v.check_symmetry("bad", lambda x: x, 1.0, lambda x: x,
                                  expected_relation="equals")

    def test_equal_relation_works(self):
        assert self.v.check_symmetry("ok", lambda x: x**2, 3.0,
                                     lambda x: -x, expected_relation="equal")

    def test_negative_relation_works(self):
        assert self.v.check_symmetry("ok", lambda x: x, 3.0,
                                     lambda x: -x, expected_relation="negative")


# ============================================================================
# L1: _validate_lean_name
# ============================================================================


class TestValidateLeanName:
    """_validate_lean_name accepts valid and rejects invalid Lean identifiers."""

    def test_simple_name(self):
        lean._validate_lean_name("foo")  # should not raise

    def test_name_with_underscore(self):
        lean._validate_lean_name("sct_beta_W")

    def test_name_with_prime(self):
        lean._validate_lean_name("x'")

    def test_name_starts_with_underscore(self):
        lean._validate_lean_name("_internal")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean._validate_lean_name("")

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean._validate_lean_name(None)

    def test_spaces_rejected(self):
        with pytest.raises(ValueError, match="valid Lean identifier"):
            lean._validate_lean_name("foo bar")

    def test_newline_injection_rejected(self):
        with pytest.raises(ValueError, match="valid Lean identifier"):
            lean._validate_lean_name("foo\n#eval 42")

    def test_special_chars_rejected(self):
        with pytest.raises(ValueError, match="valid Lean identifier"):
            lean._validate_lean_name("foo;bar")

    def test_starts_with_digit_rejected(self):
        with pytest.raises(ValueError, match="valid Lean identifier"):
            lean._validate_lean_name("42foo")


# ============================================================================
# L2: set_api_key validation
# ============================================================================


class TestSetApiKeyValidation:
    """set_api_key rejects non-string and empty keys."""

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.set_api_key("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.set_api_key("   ")

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.set_api_key(None)

    def test_int_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.set_api_key(12345)

    def test_valid_key_sets_env(self):
        old = os.environ.get("ARISTOTLE_API_KEY")
        try:
            lean.set_api_key("test_key_abc123")
            assert os.environ["ARISTOTLE_API_KEY"] == "test_key_abc123"
        finally:
            if old is not None:
                os.environ["ARISTOTLE_API_KEY"] = old
            else:
                os.environ.pop("ARISTOTLE_API_KEY", None)


# ============================================================================
# L3: _has_sorry word-boundary detection
# ============================================================================


class TestHasSorry:
    """_has_sorry uses word boundaries and strips comments."""

    def test_sorry_detected(self):
        assert lean._has_sorry("theorem t : True := by sorry")

    def test_sorry_in_word_not_detected(self):
        """'sorry' as part of another word should not trigger."""
        assert not lean._has_sorry("def nosorry := 42")

    def test_sorry_in_line_comment_not_detected(self):
        assert not lean._has_sorry("-- sorry\ntheorem t : True := by rfl")

    def test_sorry_in_block_comment_not_detected(self):
        assert not lean._has_sorry("/- sorry -/\ntheorem t : True := by rfl")

    def test_no_sorry(self):
        assert not lean._has_sorry("theorem t : True := by rfl")

    def test_sorry_with_extra_spaces(self):
        assert lean._has_sorry("  sorry  ")


# ============================================================================
# L4: verify_identity input validation
# ============================================================================


class TestVerifyIdentityValidation:
    """verify_identity rejects invalid name/lhs/rhs."""

    def test_invalid_name_rejected(self):
        with pytest.raises(ValueError, match="valid Lean identifier"):
            lean.verify_identity("foo bar", "1", "1")

    def test_empty_lhs_rejected(self):
        with pytest.raises(ValueError, match="lhs must be"):
            lean.verify_identity("t", "", "1")

    def test_empty_rhs_rejected(self):
        with pytest.raises(ValueError, match="rhs must be"):
            lean.verify_identity("t", "1", "")

    def test_none_lhs_rejected(self):
        with pytest.raises(ValueError, match="lhs must be"):
            lean.verify_identity("t", None, "1")


# ============================================================================
# L5: verify_deep both-backends-disabled guard
# ============================================================================


class TestVerifyDeepValidation:
    """verify_deep raises when both backends disabled."""

    def test_both_disabled_raises(self):
        with pytest.raises(ValueError, match="at least one backend"):
            lean.verify_deep("t", "1", "1",
                             use_aristotle=False, use_local=False)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="valid Lean identifier"):
            lean.verify_deep("bad name!", "1", "1",
                             use_aristotle=False, use_local=True)


# ============================================================================
# L6: physlean_anomaly_proof empty charges
# ============================================================================


class TestAnomalyProofValidation:
    """physlean_anomaly_proof rejects empty charges list."""

    def test_empty_charges_raises(self):
        with pytest.raises(ValueError, match="charges list must not be empty"):
            lean.physlean_anomaly_proof([])


# ============================================================================
# L7: prove() empty code validation
# ============================================================================


class TestProveValidation:
    """prove() rejects empty/non-string code."""

    def test_empty_code_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove("")

    def test_none_code_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove(None)

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            lean.prove("   ")
