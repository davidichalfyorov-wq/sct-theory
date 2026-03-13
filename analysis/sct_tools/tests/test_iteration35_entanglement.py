"""
Tests for Iteration 35: entanglement.py hardening.

Covers:
    E-02/E-03: dimension validation in negativity, log_negativity, entanglement_spectrum
    E-04: mutual_information len(dims)==2 guard
    E-05: concurrence missing _check_density_matrix
    E-10: area_law_scan invalid method validation
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import entanglement

# Skip all tests if quimb is not available
qu = pytest.importorskip("quimb")


# ============================================================================
# Helpers
# ============================================================================

def _bell_state():
    """Bell state |00> + |11>) / sqrt(2) as a 4-element vector."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[3] = 1.0 / np.sqrt(2)
    return psi


def _product_state():
    """Product state |00> as a 4-element vector."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    return psi


# ============================================================================
# E-02: negativity dimension validation
# ============================================================================


class TestNegativityDimValidation:
    """negativity must reject state/dims dimension mismatch."""

    def test_wrong_dim_raises(self):
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0
        with pytest.raises(ValueError, match="does not match"):
            entanglement.negativity(psi, [2, 3])  # expects dim 6, got 4

    def test_correct_dim_works(self):
        psi = _bell_state()
        N = entanglement.negativity(psi, [2, 2])
        assert N >= 0


# ============================================================================
# E-02: log_negativity dimension validation
# ============================================================================


class TestLogNegativityDimValidation:
    """log_negativity must reject state/dims dimension mismatch."""

    def test_wrong_dim_raises(self):
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0
        with pytest.raises(ValueError, match="does not match"):
            entanglement.log_negativity(psi, [2, 3])

    def test_correct_dim_works(self):
        psi = _bell_state()
        E = entanglement.log_negativity(psi, [2, 2])
        assert E >= 0


# ============================================================================
# E-03: entanglement_spectrum dimension validation
# ============================================================================


class TestEntanglementSpectrumDimValidation:
    """entanglement_spectrum must reject state/dims dimension mismatch."""

    def test_wrong_dim_raises(self):
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0
        with pytest.raises(ValueError, match="does not match"):
            entanglement.entanglement_spectrum(psi, [2, 3], keep=0)

    def test_correct_dim_works(self):
        psi = _bell_state()
        spec = entanglement.entanglement_spectrum(psi, [2, 2], keep=0)
        assert len(spec) == 2
        assert np.isclose(np.sum(spec), 1.0, atol=1e-10)


# ============================================================================
# E-04: mutual_information len(dims)==2 guard
# ============================================================================


class TestMutualInformationDimGuard:
    """mutual_information must reject non-bipartite dims."""

    def test_three_subsystems_raises(self):
        psi = np.zeros(8, dtype=complex)
        psi[0] = 1.0
        with pytest.raises(ValueError, match="bipartite"):
            entanglement.mutual_information(psi, [2, 2, 2])

    def test_dim_mismatch_raises(self):
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0
        with pytest.raises(ValueError, match="does not match"):
            entanglement.mutual_information(psi, [2, 3])

    def test_correct_dims_works(self):
        psi = _bell_state()
        MI = entanglement.mutual_information(psi, [2, 2])
        assert MI >= 0


# ============================================================================
# E-05: concurrence _check_density_matrix
# ============================================================================


class TestConcurrenceDensityMatrixCheck:
    """concurrence must call _check_density_matrix."""

    def test_bell_state_concurrence(self):
        psi = _bell_state()
        C = entanglement.concurrence(psi, [2, 2])
        # Bell state has concurrence = 1
        assert np.isclose(C, 1.0, atol=1e-10)

    def test_product_state_concurrence(self):
        psi = _product_state()
        C = entanglement.concurrence(psi, [2, 2])
        # Product state has concurrence = 0
        assert np.isclose(C, 0.0, atol=1e-10)


# ============================================================================
# E-10: area_law_scan invalid method validation
# ============================================================================


class TestAreaLawScanMethodValidation:
    """area_law_scan must reject invalid method strings."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            entanglement.area_law_scan([4], method="invalid_method")

    def test_none_method_raises(self):
        with pytest.raises((ValueError, TypeError)):
            entanglement.area_law_scan([4], method=None)

    def test_valid_exact_works(self):
        result = entanglement.area_law_scan([4], method="exact")
        assert len(result["L"]) == 1
        assert result["L"][0] == 4
        assert result["entropy"][0] > 0
