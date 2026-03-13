"""
Pytest regression tests for sct_tools.constants.

Verifies that physical constants and SCT parameters are correctly defined.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import constants as const


class TestFundamentalConstants:
    def test_speed_of_light(self):
        assert const.c == pytest.approx(2.998e8, rel=1e-3)

    def test_planck_constant(self):
        assert const.hbar == pytest.approx(1.055e-34, rel=1e-3)

    def test_newton_G(self):
        assert const.G_N == pytest.approx(6.674e-11, rel=1e-3)

    def test_planck_mass(self):
        # M_Pl ~ 2.176e-8 kg
        assert const.M_Pl == pytest.approx(2.176e-8, rel=1e-2)

    def test_planck_mass_GeV(self):
        # M_Pl ~ 1.22e19 GeV
        assert const.M_Pl_GeV == pytest.approx(1.22e19, rel=0.01)

    def test_fine_structure(self):
        assert const.alpha_em == pytest.approx(1 / 137.036, rel=1e-4)


class TestSMMultiplicities:
    def test_N_scalar(self):
        assert const.N_scalar == 4  # Higgs doublet

    def test_N_dirac(self):
        assert const.N_dirac == 45

    def test_N_vector(self):
        assert const.N_vector == 12  # 8 + 3 + 1


class TestSCTParameters:
    def test_beta_donoghue(self):
        assert const.beta_Donoghue == pytest.approx(41 / (10 * np.pi), rel=1e-14)

    def test_c1c2_ratio(self):
        assert const.c1_c2_ratio_original == pytest.approx(-1 / 3, rel=1e-14)

    def test_beta_R_scalar_conformal(self):
        # At conformal coupling xi=1/6, beta_R = 0
        assert const.beta_R_scalar(1 / 6) == pytest.approx(0, abs=1e-14)

    def test_beta_R_scalar_minimal(self):
        # At minimal coupling xi=0, beta_R = (1/2)(1/6)^2 = 1/72
        assert const.beta_R_scalar(0) == pytest.approx(1 / 72, rel=1e-14)
