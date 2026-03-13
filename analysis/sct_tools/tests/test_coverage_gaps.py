"""
Tests for previously untested public functions across sct_tools.

Covers:
    - constants: log_dimensions, natural_to_si, NaturalUnits.to_cm2/to_kg/to_seconds
    - form_factors: dhR_scalar_dx
    - tensors: anti_de_sitter, reissner_nordstrom
    - entanglement: mps_bond_dimensions
    - compute: cached, clear_cache, vegas_integrate, jax_grad
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import constants as const
from sct_tools import form_factors as ff

# Optional packages
try:
    import OGRePy  # noqa: F401
    HAS_OGREPY = True
except ImportError:
    HAS_OGREPY = False

try:
    import quimb  # noqa: F401
    HAS_QUIMB = True
except ImportError:
    HAS_QUIMB = False

try:
    import jax  # noqa: F401
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import vegas  # noqa: F401
    HAS_VEGAS = True
except ImportError:
    HAS_VEGAS = False


# ============================================================================
# constants.log_dimensions (backward-compat no-op)
# ============================================================================

class TestLogDimensions:
    """log_dimensions is a backward-compat no-op that just prints."""

    def test_runs_without_error(self):
        """Should complete without exception."""
        const.log_dimensions(1.23e-5, 2, label="test quantity")

    def test_no_return_value(self):
        """Returns None (it only prints)."""
        result = const.log_dimensions(42.0, -1)
        assert result is None

    def test_various_inputs(self):
        """Should handle edge values gracefully."""
        const.log_dimensions(0.0, 0, label="zero")
        const.log_dimensions(-1e30, 4, label="large negative")
        const.log_dimensions(float('inf'), 2, label="infinity")


# ============================================================================
# constants.natural_to_si
# ============================================================================

class TestNaturalToSI:
    """natural_to_si converts from natural units (c=hbar=1) to SI."""

    def test_energy_identity(self):
        """dim_mass only: factor = hbar^0 * c^0 = 1 (mass is the base unit).
        natural_to_si(1.0, dim_mass=1) = 1.0 exactly."""
        result = const.natural_to_si(1.0, dim_mass=1, dim_length=0, dim_time=0)
        assert result == pytest.approx(1.0, abs=1e-15)

    def test_length_conversion(self):
        """dim_mass=0, dim_length=1 => converts from GeV^{-1} to meters.
        1 GeV^{-1} ~ 1.97e-16 m (hbar*c/GeV)."""
        result = const.natural_to_si(1.0, dim_mass=0, dim_length=1, dim_time=0)
        assert result == pytest.approx(1.97e-16, rel=0.01)

    def test_time_conversion(self):
        """dim_mass=0, dim_length=0, dim_time=1 => GeV^{-1} to seconds.
        1 GeV^{-1} ~ 6.58e-25 s (hbar/GeV)."""
        result = const.natural_to_si(1.0, dim_mass=0, dim_length=0, dim_time=1)
        assert result == pytest.approx(6.58e-25, rel=0.01)

    def test_zero_input(self):
        """Zero input should give zero output."""
        result = const.natural_to_si(0.0, dim_mass=2, dim_length=-1, dim_time=0)
        assert result == 0.0

    def test_scaling(self):
        """Doubling the input should double the output (linearity)."""
        r1 = const.natural_to_si(1.0, dim_mass=1, dim_length=0, dim_time=0)
        r2 = const.natural_to_si(2.0, dim_mass=1, dim_length=0, dim_time=0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-14)


# ============================================================================
# constants.NaturalUnits conversions
# ============================================================================

class TestNaturalUnitsConversions:
    """Test NaturalUnits static methods for unit conversions."""

    def test_to_seconds(self):
        """1 GeV^{-1} ~ 6.58e-25 s."""
        from scipy import constants as sc
        result = const.NaturalUnits.to_seconds(1.0)
        expected = sc.hbar / (sc.eV * 1e9)  # hbar / GeV
        assert result == pytest.approx(expected, rel=1e-10)

    def test_to_kg(self):
        """1 GeV ~ 1.783e-27 kg."""
        result = const.NaturalUnits.to_kg(1.0)
        expected = 1.783e-27
        assert result == pytest.approx(expected, rel=0.01)

    def test_to_cm2(self):
        """1 GeV^{-2} ~ 3.89e-28 cm^2 (= 0.389 mb)."""
        result = const.NaturalUnits.to_cm2(1.0)
        # hbar*c = 0.197 GeV·fm => (hbar*c)^2 = 0.0389 GeV^2·fm^2
        # 1 fm^2 = 1e-26 cm^2 => 3.89e-28 cm^2
        assert result == pytest.approx(3.89e-28, rel=0.01)

    def test_to_seconds_zero(self):
        """Zero input gives zero."""
        assert const.NaturalUnits.to_seconds(0.0) == 0.0

    def test_to_kg_scaling(self):
        """Linear scaling: 10 GeV -> 10x the mass in kg."""
        r1 = const.NaturalUnits.to_kg(1.0)
        r10 = const.NaturalUnits.to_kg(10.0)
        assert r10 == pytest.approx(10.0 * r1, rel=1e-14)

    def test_to_cm2_negative(self):
        """Negative input should give negative output (linear map)."""
        result = const.NaturalUnits.to_cm2(-1.0)
        assert result < 0


# ============================================================================
# form_factors.dhR_scalar_dx (numerical derivative)
# ============================================================================

class TestDhRScalarDx:
    """Test the numerical derivative d/dx h_R^(0)(x; xi)."""

    def test_vs_mpmath_central_diff(self):
        """Compare dhR_scalar_dx against independent mpmath finite difference."""
        from mpmath import mp, mpf
        old_dps = mp.dps
        try:
            mp.dps = 50

            x0 = 1.0
            xi0 = 0.0
            h = mpf('1e-8')
            # Independent 4th-order central difference using mpmath
            x_mp = mpf(x0)

            def hR_mp(x_val):
                return mpf(ff.hR_scalar_mp(float(x_val), xi=xi0, dps=50))

            deriv_mp = float(
                (-hR_mp(x_mp + 2 * h) + 8 * hR_mp(x_mp + h)
                 - 8 * hR_mp(x_mp - h) + hR_mp(x_mp - 2 * h)) / (12 * h)
            )

            result = ff.dhR_scalar_dx(x0, xi=xi0)
            assert result == pytest.approx(deriv_mp, rel=1e-4)
        finally:
            mp.dps = old_dps

    def test_multiple_x_values(self):
        """Check dhR_scalar_dx against mpmath 50-digit central differences."""
        # mpmath reference: central differences with h=1e-8, dps=50
        refs = {
            0.01: -3.3653086852e-03,
            0.1:  -3.2967354034e-03,
            1.0:  -2.6836182437e-03,
            10.0: -3.4005938739e-04,
            100.0: 2.3890189399e-06,
        }
        for x, expected in refs.items():
            val = ff.dhR_scalar_dx(x, xi=0.0)
            assert val == pytest.approx(expected, rel=0.01), f"Failed at x={x}"

    def test_xi_dependence(self):
        """Different xi values should give different derivatives (in general)."""
        x = 1.0
        d0 = ff.dhR_scalar_dx(x, xi=0.0)
        d_conf = ff.dhR_scalar_dx(x, xi=1 / 6)
        d1 = ff.dhR_scalar_dx(x, xi=1.0)
        # These should generally be different
        assert d0 != pytest.approx(d1, rel=0.01)
        # All should be finite
        assert np.isfinite(d0) and np.isfinite(d_conf) and np.isfinite(d1)

    def test_large_x_derivative(self):
        """For large x, dh_R/dx ~ O(1/x^2). mpmath ref at x=1000."""
        x = 1000.0
        val = ff.dhR_scalar_dx(x, xi=0.0)
        # mpmath 50-digit reference
        assert val == pytest.approx(2.7439398397e-08, rel=0.05)

    def test_conformal_xi_derivative(self):
        """At xi=1/6 (conformal), h_R has specific structure. Derivative
        should match mpmath reference."""
        x0 = 2.0
        from mpmath import mp, mpf
        old_dps = mp.dps
        try:
            mp.dps = 50
            h = mpf('1e-8')
            x_mp = mpf(x0)
            xi_val = 1 / 6

            def hR_mp(xv):
                return mpf(ff.hR_scalar_mp(float(xv), xi=xi_val, dps=50))

            deriv_mp = float(
                (-hR_mp(x_mp + 2 * h) + 8 * hR_mp(x_mp + h)
                 - 8 * hR_mp(x_mp - h) + hR_mp(x_mp - 2 * h)) / (12 * h)
            )
            result = ff.dhR_scalar_dx(x0, xi=xi_val)
            assert result == pytest.approx(deriv_mp, rel=1e-4)
        finally:
            mp.dps = old_dps


# ============================================================================
# tensors.anti_de_sitter
# ============================================================================

@pytest.mark.skipif(not HAS_OGREPY, reason="OGRePy not installed")
class TestAntiDeSitter:
    """Test Anti-de Sitter metric construction."""

    def test_creates_metric(self):
        """anti_de_sitter() should return (metric, params) tuple."""
        from sct_tools import tensors
        result = tensors.anti_de_sitter()
        assert isinstance(result, tuple) and len(result) == 2
        metric, params = result
        assert 'Lambda' in params

    def test_curvature_invariants(self):
        """AdS Ricci scalar: R = -4*Lambda in 4D (OGRePy convention)."""
        import sympy as sp

        from sct_tools import tensors
        metric, params = tensors.anti_de_sitter()
        inv = tensors.curvature_invariants(metric)
        R = inv['ricci_scalar']
        assert R is not None
        # OGRePy gives R = -4*Lambda for 4D AdS (R/Lambda = -4)
        Lambda_sym = params['Lambda']
        ratio = sp.simplify(R / Lambda_sym)
        assert ratio == -4, f"R/Lambda = {ratio} (expected -4)"

    def test_metric_signature(self):
        """AdS should have Lorentzian signature (-,+,+,+)."""

        from sct_tools import tensors
        metric, params = tensors.anti_de_sitter()
        comp = metric.components()
        g_tt = comp[0, 0]
        # At r=0, g_tt should be -1 (for |Lambda|*r^2 -> 0)
        r_sym = params['symbols'][1]  # r coordinate
        g_tt_r0 = g_tt.subs(r_sym, 0)
        assert float(g_tt_r0) == pytest.approx(-1.0)


# ============================================================================
# tensors.reissner_nordstrom
# ============================================================================

@pytest.mark.skipif(not HAS_OGREPY, reason="OGRePy not installed")
class TestReissnerNordstrom:
    """Test Reissner-Nordstrom metric construction."""

    def test_creates_metric(self):
        """reissner_nordstrom() should return (metric, params) tuple."""
        from sct_tools import tensors
        result = tensors.reissner_nordstrom()
        assert isinstance(result, tuple) and len(result) == 2
        metric, params = result
        assert 'M' in params and 'Q' in params

    def test_reduces_to_schwarzschild(self):
        """With Q=0, Reissner-Nordstrom should reduce to Schwarzschild.
        g_tt(Q=0) = -(1-2M/r) = Schwarzschild."""
        import sympy as sp

        from sct_tools import tensors
        metric, params = tensors.reissner_nordstrom()
        comp = metric.components()
        g_tt = comp[0, 0]
        Q_sym = params['Q']
        g_tt_Q0 = g_tt.subs(Q_sym, 0)
        r_sym = params['symbols'][1]
        M_sym = params['M']
        expected = -(1 - 2 * M_sym / r_sym)
        diff = sp.simplify(g_tt_Q0 - expected)
        assert diff == 0

    def test_curvature_ricci_scalar_zero(self):
        """RN is vacuum+EM: Ricci scalar R = 0 (traceless EM stress tensor)."""
        import sympy as sp

        from sct_tools import tensors
        metric, params = tensors.reissner_nordstrom()
        inv = tensors.curvature_invariants(metric)
        R = inv['ricci_scalar']
        assert R is not None
        R_simplified = sp.simplify(R)
        assert R_simplified == 0, f"R = {R_simplified} (expected 0)"


# ============================================================================
# entanglement.mps_bond_dimensions
# ============================================================================

@pytest.mark.skipif(not HAS_QUIMB, reason="quimb not installed")
class TestMPSBondDimensions:
    """Test MPS bond dimension extraction."""

    def test_random_mps(self):
        """random_mps should have expected bond dimensions."""
        from sct_tools import entanglement
        L = 8
        bond_dim = 4
        mps = entanglement.random_mps(L, bond_dim=bond_dim)
        bonds = entanglement.mps_bond_dimensions(mps)
        assert len(bonds) == L
        # Interior bonds should be <= bond_dim
        for b in bonds[1:-1]:
            assert 1 <= b <= bond_dim

    def test_small_mps(self):
        """L=2 MPS should have 2 bond dimensions."""
        from sct_tools import entanglement
        mps = entanglement.random_mps(2, bond_dim=2)
        bonds = entanglement.mps_bond_dimensions(mps)
        assert len(bonds) == 2
        assert all(b >= 1 for b in bonds)

    def test_returns_list_of_ints(self):
        """Bond dimensions should be positive integers."""
        from sct_tools import entanglement
        mps = entanglement.random_mps(6, bond_dim=8)
        bonds = entanglement.mps_bond_dimensions(mps)
        assert isinstance(bonds, list)
        for b in bonds:
            assert isinstance(b, (int, np.integer))
            assert b >= 1


# ============================================================================
# compute.cached and clear_cache
# ============================================================================

class TestCached:
    """Test the @cached decorator (joblib disk cache)."""

    def test_caching_returns_same_result(self):
        """Cached function should return same result on repeated calls."""
        from sct_tools import compute

        @compute.cached
        def slow_square(x):
            return x ** 2

        r1 = slow_square(7)
        r2 = slow_square(7)
        assert r1 == 49
        assert r2 == 49

    def test_clear_cache_no_error(self):
        """clear_cache() should not raise."""
        from sct_tools import compute
        compute.clear_cache()  # Should complete without error


# ============================================================================
# compute.vegas_integrate
# ============================================================================

@pytest.mark.skipif(not HAS_VEGAS, reason="vegas not installed")
class TestVegasIntegrate:
    """Test adaptive Monte Carlo integration."""

    def test_1d_gaussian(self):
        """Integrate exp(-x^2) from -3 to 3.
        Exact: sqrt(pi) * erf(3) ~ 1.7320...(approx sqrt(pi)*0.99998)."""
        import math

        from sct_tools import compute

        def integrand(x):
            return math.exp(-x[0] ** 2)

        result = compute.vegas_integrate(integrand, [(-3, 3)],
                                         nitn=10, neval=5000)
        exact = np.sqrt(np.pi) * 1.0  # erf(3) ~ 0.99998 ~ 1
        assert result.mean == pytest.approx(exact, rel=0.01)

    def test_2d_integral(self):
        """Integrate 1 over [0,1]^2 = 1.0."""
        from sct_tools import compute

        def integrand(x):
            return 1.0

        result = compute.vegas_integrate(integrand, [(0, 1), (0, 1)],
                                         nitn=5, neval=1000)
        assert result.mean == pytest.approx(1.0, rel=0.01)

    def test_quality_factor(self):
        """Well-behaved integral should have Q > 0.01."""
        import math

        from sct_tools import compute

        def integrand(x):
            return math.exp(-x[0] ** 2)

        result = compute.vegas_integrate(integrand, [(-2, 2)],
                                         nitn=10, neval=5000)
        assert result.Q > 0.01  # Good convergence


# ============================================================================
# compute.jax_grad
# ============================================================================

@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJaxGrad:
    """Test JAX automatic differentiation wrapper."""

    def test_square_gradient(self):
        """d/dx (x^2) = 2x."""
        import jax.numpy as jnp

        from sct_tools import compute

        def f(x):
            return x ** 2

        grad_f = compute.jax_grad(f)
        assert float(grad_f(jnp.float32(3.0))) == pytest.approx(6.0, rel=1e-5)

    def test_sin_gradient(self):
        """d/dx sin(x) = cos(x)."""
        import jax.numpy as jnp

        from sct_tools import compute

        def f(x):
            return jnp.sin(x)

        grad_f = compute.jax_grad(f)
        x = jnp.float32(1.0)
        assert float(grad_f(x)) == pytest.approx(float(jnp.cos(x)), rel=1e-5)

    def test_multivariate(self):
        """d/dx (x^2 + y) at x=2, y=3 should be 4 (= 2x)."""
        import jax.numpy as jnp

        from sct_tools import compute

        def f(x, y):
            return x ** 2 + y

        grad_f = compute.jax_grad(f, argnums=0)
        result = float(grad_f(jnp.float32(2.0), jnp.float32(3.0)))
        assert result == pytest.approx(4.0, rel=1e-5)
