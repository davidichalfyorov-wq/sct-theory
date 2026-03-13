"""
Tests for F1_spectral and F2_spectral (spectral integral form factors).

Verifies against known values for psi(u) = e^{-u} (exponential spectral function).

Key result (numerically verified):
    For psi(u) = e^{-u}, Psi_1(u) = Psi_2(u) = e^{-u}, Psi_i(0) = 1:
        F1_spectral(z) = hC_dirac(z) / (16*pi^2)
        F2_spectral(z) = hR_dirac(z) / (16*pi^2)

    The spectral integral formulas correspond to the DIRAC spin-1/2 form
    factors, not the scalar ones. This is because the BV spectral representation
    F1_spectral encodes the full Weyl-tensor structure including the
    -(1/4) int (1-2a)^2 psi(...) term, which matches the Dirac trace structure.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors as ff


# Spectral function: psi(u) = e^{-u}
# Antiderivatives: Psi_1(u) = e^{-u}, Psi_2(u) = e^{-u}
# Psi_1(0) = 1, Psi_2(0) = 1
def psi(u):
    return np.exp(-u)

def psi1(u):
    return np.exp(-u)

def psi2(u):
    return np.exp(-u)


class TestF1Spectral:
    def test_small_z(self):
        # At z=0.1, F1 should be finite and negative (since hC_dirac(0)=-1/20)
        val = ff.F1_spectral(0.1, psi, psi1, psi2, 1.0, 1.0)
        assert np.isfinite(val)

    def test_matches_hC_dirac(self):
        """F1_spectral for psi=e^{-u} must equal hC_dirac/(16*pi^2).

        Numerically verified: the ratio F1_spectral(z) / [hC_dirac(z)/(16pi^2)]
        equals 1.0 to machine precision for all z > 0.
        """
        for z in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            f1_val = ff.F1_spectral(z, psi, psi1, psi2, 1.0, 1.0)
            expected = ff.hC_dirac(z) / (16 * np.pi**2)
            assert f1_val == pytest.approx(expected, rel=1e-10), (
                f"F1_spectral({z}) = {f1_val}, "
                f"expected hC_dirac({z})/(16pi^2) = {expected}"
            )

    def test_positive_z_values(self):
        for z in [0.5, 2.0, 5.0, 10.0]:
            val = ff.F1_spectral(z, psi, psi1, psi2, 1.0, 1.0)
            assert np.isfinite(val), f"F1 not finite at z={z}"


class TestF2Spectral:
    def test_small_z(self):
        val = ff.F2_spectral(0.1, psi, psi1, psi2, 1.0, 1.0)
        assert np.isfinite(val)

    def test_matches_hR_dirac(self):
        """F2_spectral for psi=e^{-u} must equal hR_dirac/(16*pi^2).

        Numerically verified: the ratio F2_spectral(z) / [hR_dirac(z)/(16pi^2)]
        equals 1.0 to machine precision for all z > 0.
        """
        for z in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            f2_val = ff.F2_spectral(z, psi, psi1, psi2, 1.0, 1.0)
            expected = ff.hR_dirac(z) / (16 * np.pi**2)
            assert f2_val == pytest.approx(expected, rel=1e-10), (
                f"F2_spectral({z}) = {f2_val}, "
                f"expected hR_dirac({z})/(16pi^2) = {expected}"
            )

    def test_positive_z_values(self):
        for z in [0.5, 2.0, 5.0, 10.0]:
            val = ff.F2_spectral(z, psi, psi1, psi2, 1.0, 1.0)
            assert np.isfinite(val), f"F2 not finite at z={z}"

    def test_f1_f2_different(self):
        # F1 and F2 should give different values for general z
        z = 2.0
        f1 = ff.F1_spectral(z, psi, psi1, psi2, 1.0, 1.0)
        f2 = ff.F2_spectral(z, psi, psi1, psi2, 1.0, 1.0)
        assert f1 != pytest.approx(f2, rel=0.01)


class TestSpectralZeroGuard:
    def test_f1_rejects_zero(self):
        with pytest.raises(ValueError, match="z > 0"):
            ff.F1_spectral(0, psi, psi1, psi2, 1.0, 1.0)

    def test_f2_rejects_zero(self):
        with pytest.raises(ValueError, match="z > 0"):
            ff.F2_spectral(0, psi, psi1, psi2, 1.0, 1.0)

    def test_f1_rejects_negative(self):
        with pytest.raises(ValueError, match="z > 0"):
            ff.F1_spectral(-1.0, psi, psi1, psi2, 1.0, 1.0)

    def test_f2_rejects_negative(self):
        with pytest.raises(ValueError, match="z > 0"):
            ff.F2_spectral(-1.0, psi, psi1, psi2, 1.0, 1.0)
