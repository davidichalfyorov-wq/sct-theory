from __future__ import annotations

import mpmath
import pytest
from analysis.scripts.nt4a_newtonian import gamma_local_ratio
from analysis.scripts.ppn1_parameters import (
    beta_ppn,
    gamma_ppn,
    lower_bound_Lambda,
    m0_mass,
    m2_mass,
    ppn_table,
)

mpmath.mp.dps = 100

def test_dimensions():
    # Test limiting cases for gamma_ppn
    r = mpmath.mpf('1e10')
    L = mpmath.mpf('1')
    xi = mpmath.mpf('0')
    gamma = gamma_ppn(r, L, xi)
    # For large Lambda*r, gamma should be exactly 1
    assert abs(gamma - mpmath.mpf('1')) < 1e-10

def test_gamma_r_zero():
    r = mpmath.mpf('0')
    L = mpmath.mpf('1')
    xi = mpmath.mpf('0')
    # At r=0, gamma = (2m2 + m0)/(4m2 - m0)
    m2 = m2_mass(L)
    m0 = m0_mass(L, xi)
    expected = (mpmath.mpf('2') * m2 + m0) / (mpmath.mpf('4') * m2 - m0)
    gamma = gamma_ppn(r, L, xi)
    assert abs(gamma - expected) < 1e-50

def test_gamma_conformal():
    r = mpmath.mpf('1e-5')
    L = mpmath.mpf('1')
    xi = mpmath.mpf('1') / 6
    # For conformal coupling, m0 -> inf, so e^{-m0 r} -> 0
    m2 = m2_mass(L)
    term2 = mpmath.exp(-m2 * r)
    expected = (
        mpmath.mpf('1') - (mpmath.mpf('2') / 3) * term2
    ) / (
        mpmath.mpf('1') - (mpmath.mpf('4') / 3) * term2
    )
    gamma = gamma_ppn(r, L, xi)
    assert abs(gamma - expected) < 1e-50

def test_gamma_ppn_delegates_to_nt4a_local_layer():
    r = mpmath.mpf('1.0')
    L = mpmath.mpf('1.0')
    xi = mpmath.mpf('0')
    expected = gamma_local_ratio(r, Lambda=L, xi=xi, dps=100)
    gamma = gamma_ppn(r, L, xi)
    assert abs(gamma - expected) < 1e-40

def test_beta_not_derived_yet():
    with pytest.raises(NotImplementedError, match="not derived"):
        beta_ppn(mpmath.mpf('1.0'), mpmath.mpf('1.0'), mpmath.mpf('0'))

def test_lower_bounds():
    cassini_bound = lower_bound_Lambda("cassini")
    eot_wash_bound = lower_bound_Lambda("eot-wash")
    assert cassini_bound > 0
    assert eot_wash_bound > cassini_bound

def test_ppn_table():
    table = ppn_table(mpmath.mpf('1e-3'), mpmath.mpf('0'))
    assert table["scope"] == "linear_static_local_yukawa"
    assert "gamma" in table
    assert table["beta"] == "not_derived"
    assert table["alpha1"] == "not_derived"
    assert table["zeta1"] == "not_derived"

@pytest.mark.parametrize("r", ["0.1", "1.0", "10.0", "100.0"])
def test_gamma_nontrivial_regime_is_finite(r):
    gamma = gamma_ppn(mpmath.mpf(r), mpmath.mpf('1.0'), mpmath.mpf('0'))
    assert mpmath.isfinite(gamma)
