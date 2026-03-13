"""
Tests for Iteration 40: entanglement.py + lean.py hardening.

Covers:
    E-F4: area_law_scan rejects empty/invalid L_values
    E-F7: fit_cft_entropy rejects NaN/Inf entropies
    E-F8: random_mps validates L, bond_dim, phys_dim
    L-F1: prove_local/prove_scilean/build_sctlean timeout validation
    L-F5: physlean_sm_dof_proof rejects non-int dof_value
"""

import numpy as np
import pytest

from sct_tools import entanglement, lean

# ============================================================================
# E-F4: area_law_scan rejects empty / invalid L_values
# ============================================================================


class TestAreaLawScanLValues:
    """area_law_scan must reject empty or invalid L_values."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            entanglement.area_law_scan([])

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="integers >= 2"):
            entanglement.area_law_scan([0, 4, 6])

    def test_one_raises(self):
        with pytest.raises(ValueError, match="integers >= 2"):
            entanglement.area_law_scan([1])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="integers >= 2"):
            entanglement.area_law_scan([-3, 4])

    def test_float_raises(self):
        with pytest.raises(ValueError, match="integers >= 2"):
            entanglement.area_law_scan([2.5, 4])


# ============================================================================
# E-F7: fit_cft_entropy rejects NaN/Inf entropies
# ============================================================================


class TestFitCftEntropyNaN:
    """fit_cft_entropy must reject NaN/Inf in entropies."""

    def test_nan_entropy_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            entanglement.fit_cft_entropy([4, 8, 16], [0.5, float("nan"), 1.2])

    def test_inf_entropy_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            entanglement.fit_cft_entropy([4, 8, 16], [0.5, float("inf"), 1.2])

    def test_neg_inf_entropy_raises(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            entanglement.fit_cft_entropy([4, 8], [float("-inf"), 0.5])

    def test_valid_still_works(self):
        # S ~ (c/3) log(L) + const with c=1
        L = [4, 8, 16, 32]
        S = [np.log2(lv) / 3.0 + 0.1 for lv in L]
        result = entanglement.fit_cft_entropy(L, S)
        assert "central_charge" in result
        assert np.isfinite(result["central_charge"])


# ============================================================================
# E-F8: random_mps validates L, bond_dim, phys_dim
# ============================================================================


class TestRandomMpsValidation:
    """random_mps must reject invalid L, bond_dim, phys_dim."""

    def test_zero_L_raises(self):
        with pytest.raises(ValueError, match="L must be a positive integer"):
            entanglement.random_mps(0, bond_dim=4)

    def test_negative_L_raises(self):
        with pytest.raises(ValueError, match="L must be a positive integer"):
            entanglement.random_mps(-2, bond_dim=4)

    def test_float_L_raises(self):
        with pytest.raises(ValueError, match="L must be a positive integer"):
            entanglement.random_mps(3.5, bond_dim=4)

    def test_zero_bond_dim_raises(self):
        with pytest.raises(ValueError, match="bond_dim must be a positive integer"):
            entanglement.random_mps(4, bond_dim=0)

    def test_zero_phys_dim_raises(self):
        with pytest.raises(ValueError, match="phys_dim must be a positive integer"):
            entanglement.random_mps(4, bond_dim=4, phys_dim=0)


# ============================================================================
# L-F1: prove_local / prove_scilean / build_sctlean timeout validation
# ============================================================================


class TestProveLocalTimeout:
    """prove_local must reject non-positive timeout."""

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            lean.prove_local("theorem t : True := trivial", timeout=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            lean.prove_local("theorem t : True := trivial", timeout=-10)


class TestProveScileanTimeout:
    """prove_scilean must reject non-positive timeout."""

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            lean.prove_scilean("theorem t : True := trivial", timeout=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            lean.prove_scilean("theorem t : True := trivial", timeout=-10)


class TestBuildSctleanTimeout:
    """build_sctlean must reject non-positive timeout."""

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            lean.build_sctlean(timeout=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            lean.build_sctlean(timeout=-5)


# ============================================================================
# L-F5: physlean_sm_dof_proof rejects non-int dof_value
# ============================================================================


class TestPhysleanSmDofProofType:
    """physlean_sm_dof_proof must reject non-int dof_value."""

    def test_float_raises(self):
        with pytest.raises(TypeError, match="integer"):
            lean.physlean_sm_dof_proof("photon", 4.0)

    def test_string_raises(self):
        with pytest.raises(TypeError, match="integer"):
            lean.physlean_sm_dof_proof("photon", "4")

    def test_none_raises(self):
        with pytest.raises(TypeError, match="integer"):
            lean.physlean_sm_dof_proof("photon", None)
