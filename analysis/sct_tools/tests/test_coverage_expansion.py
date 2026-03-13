"""
Coverage expansion tests — fills gaps identified in deep audit.

Covers:
  - dphi_dx, dphi_dx_fast (derivative of master function)
  - dhC_scalar_dx, dhC_dirac_dx, dhR_dirac_dx, dhC_vector_dx, dhR_vector_dx
  - get_taylor_coefficients (Taylor coefficient export)
  - asymptotic_expansion (UV asymptotics)
  - _mp form factor functions (mpmath high-precision)
  - graphs: spectral_action_on_graph, zeta_function_graph
  - tensors: weyl_tensor
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import form_factors as ff

# =============================================================================
# dphi_dx and dphi_dx_fast
# =============================================================================

class TestDphiDx:
    """Test derivative of master function phi(x)."""

    def test_dphi_dx_at_zero(self):
        """phi'(0) = -1/6."""
        assert ff.dphi_dx(0.0) == pytest.approx(-1.0 / 6.0, rel=1e-12)

    def test_dphi_dx_fast_at_zero(self):
        """phi_fast'(0) = -1/6."""
        assert ff.dphi_dx_fast(0.0) == pytest.approx(-1.0 / 6.0, rel=1e-12)

    def test_dphi_dx_negative(self):
        """phi'(x) should be negative for x > 0 (phi is decreasing)."""
        for x in [0.1, 1.0, 5.0, 50.0]:
            assert ff.dphi_dx_fast(x) < 0

    def test_dphi_dx_fast_vs_quad(self):
        """dphi_dx_fast matches dphi_dx (quad) at several points."""
        for x in [0.5, 2.0, 5.0, 20.0, 100.0]:
            fast = ff.dphi_dx_fast(x)
            quad_val = ff.dphi_dx(x)
            assert fast == pytest.approx(quad_val, rel=1e-8), \
                f"Mismatch at x={x}: fast={fast}, quad={quad_val}"

    def test_dphi_dx_fast_raises_on_negative(self):
        """dphi_dx_fast requires x >= 0."""
        with pytest.raises(ValueError):
            ff.dphi_dx_fast(-1.0)

    def test_dphi_dx_fast_raises_on_nan(self):
        """dphi_dx_fast requires finite x."""
        with pytest.raises(ValueError):
            ff.dphi_dx_fast(float('nan'))


# =============================================================================
# Derivative functions: dhC_scalar_dx, dhC_dirac_dx, dhR_dirac_dx
# =============================================================================

class TestFormFactorDerivatives:
    """Test d/dx of form factors against numerical finite differences."""

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 10.0, 50.0])
    def test_dhC_scalar_dx(self, x):
        """dhC_scalar_dx matches numerical derivative."""
        h = 1e-6
        numerical = (ff.hC_scalar_fast(x + h) - ff.hC_scalar_fast(x - h)) / (2 * h)
        analytic = ff.dhC_scalar_dx(x)
        assert analytic == pytest.approx(numerical, rel=1e-4)

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 10.0, 50.0])
    def test_dhC_dirac_dx(self, x):
        """dhC_dirac_dx matches numerical derivative."""
        h = 1e-6
        numerical = (ff.hC_dirac_fast(x + h) - ff.hC_dirac_fast(x - h)) / (2 * h)
        analytic = ff.dhC_dirac_dx(x)
        assert analytic == pytest.approx(numerical, rel=1e-4)

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 10.0, 50.0])
    def test_dhR_dirac_dx(self, x):
        """dhR_dirac_dx matches numerical derivative."""
        h = 1e-6
        numerical = (ff.hR_dirac_fast(x + h) - ff.hR_dirac_fast(x - h)) / (2 * h)
        analytic = ff.dhR_dirac_dx(x)
        assert analytic == pytest.approx(numerical, rel=1e-4)

    def test_dhC_scalar_dx_at_zero(self):
        """dhC_scalar_dx(0) = first Taylor coefficient = -1/1680."""
        # c_1 of hC_scalar = a_3 / 2 = [(-1)^3 * 3! / 7!] / 2 = -6/(5040*2) = -1/1680
        # Actually c_1 = _HC0_TAYLOR[1]
        expected = ff._HC0_TAYLOR[1]
        val = ff.dhC_scalar_dx(0.0)
        assert val == pytest.approx(expected, rel=1e-12)

    def test_derivative_raises_on_negative_x(self):
        """All derivative functions raise on x < 0."""
        for func in [ff.dhC_scalar_dx, ff.dhC_dirac_dx, ff.dhR_dirac_dx]:
            with pytest.raises(ValueError):
                func(-1.0)


# =============================================================================
# Vector derivative functions
# =============================================================================

class TestVectorDerivatives:
    """Test d/dx of vector form factors."""

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 10.0])
    def test_dhC_vector_dx(self, x):
        """dhC_vector_dx matches numerical derivative."""
        h = 1e-6
        numerical = (ff.hC_vector_fast(x + h) - ff.hC_vector_fast(x - h)) / (2 * h)
        analytic = ff.dhC_vector_dx(x)
        assert analytic == pytest.approx(numerical, rel=1e-4)

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 10.0])
    def test_dhR_vector_dx(self, x):
        """dhR_vector_dx matches numerical derivative."""
        h = 1e-6
        numerical = (ff.hR_vector_fast(x + h) - ff.hR_vector_fast(x - h)) / (2 * h)
        analytic = ff.dhR_vector_dx(x)
        assert analytic == pytest.approx(numerical, rel=1e-4)


# =============================================================================
# get_taylor_coefficients
# =============================================================================

class TestGetTaylorCoefficients:
    """Test Taylor coefficient export function."""

    def test_available_form_factors(self):
        """All listed form factors should work."""
        for name in ['hC_scalar', 'hC_dirac', 'hR_dirac', 'hC_vector',
                      'hR_vector', 'hR_scalar_A', 'hR_scalar_B', 'hR_scalar_C',
                      'phi']:
            coeffs = ff.get_taylor_coefficients(name)
            assert isinstance(coeffs, np.ndarray)
            assert len(coeffs) > 0

    def test_unknown_form_factor_raises(self):
        """Unknown form factor name should raise ValueError."""
        with pytest.raises(ValueError, match="unknown form factor"):
            ff.get_taylor_coefficients('nonexistent')

    def test_n_terms_truncation(self):
        """n_terms parameter truncates output."""
        full = ff.get_taylor_coefficients('hC_scalar')
        trunc = ff.get_taylor_coefficients('hC_scalar', n_terms=5)
        assert len(trunc) == 5
        np.testing.assert_array_equal(full[:5], trunc)

    def test_phi_coefficients(self):
        """phi Taylor coefficients: a_n = (-1)^n * n! / (2n+1)!"""
        coeffs = ff.get_taylor_coefficients('phi', n_terms=5)
        assert coeffs[0] == pytest.approx(1.0, rel=1e-14)  # a_0 = 1
        assert coeffs[1] == pytest.approx(-1.0 / 6.0, rel=1e-14)  # a_1 = -1/6
        assert coeffs[2] == pytest.approx(1.0 / 60.0, rel=1e-14)  # a_2 = 2/120 = 1/60

    def test_hC_scalar_c0(self):
        """hC_scalar c_0 = 1/120."""
        coeffs = ff.get_taylor_coefficients('hC_scalar', n_terms=1)
        assert coeffs[0] == pytest.approx(1.0 / 120.0, rel=1e-14)


# =============================================================================
# asymptotic_expansion
# =============================================================================

class TestAsymptoticExpansion:
    """Test UV asymptotic expansion function."""

    def test_hC_scalar_asymptotic(self):
        """asymptotic_expansion('hC_scalar', x) matches exact at large x."""
        x = 1000.0
        exact = ff.hC_scalar_fast(x)
        asymp = ff.asymptotic_expansion('hC_scalar', x)
        assert asymp == pytest.approx(exact, rel=0.01)

    def test_hC_dirac_asymptotic(self):
        """asymptotic_expansion('hC_dirac', x) matches exact at large x."""
        x = 1000.0
        exact = ff.hC_dirac_fast(x)
        asymp = ff.asymptotic_expansion('hC_dirac', x)
        assert asymp == pytest.approx(exact, rel=0.01)

    def test_hR_dirac_asymptotic(self):
        """asymptotic_expansion('hR_dirac', x) matches exact at large x."""
        x = 1000.0
        exact = ff.hR_dirac_fast(x)
        asymp = ff.asymptotic_expansion('hR_dirac', x)
        # hR_dirac has slow convergence; 2% tolerance at x=1000
        assert asymp == pytest.approx(exact, rel=0.02)

    def test_hC_vector_asymptotic(self):
        """asymptotic_expansion('hC_vector', x) matches exact at large x."""
        x = 1000.0
        exact = ff.hC_vector_fast(x)
        asymp = ff.asymptotic_expansion('hC_vector', x)
        assert asymp == pytest.approx(exact, rel=0.01)

    def test_hR_vector_asymptotic(self):
        """asymptotic_expansion('hR_vector', x) matches exact at large x."""
        x = 1000.0
        exact = ff.hR_vector_fast(x)
        asymp = ff.asymptotic_expansion('hR_vector', x)
        assert asymp == pytest.approx(exact, rel=0.01)

    def test_unknown_raises(self):
        """Unknown form factor in asymptotic_expansion raises ValueError."""
        with pytest.raises(ValueError):
            ff.asymptotic_expansion('unknown', 100.0)

    def test_negative_x_raises(self):
        """asymptotic_expansion requires x > 0."""
        with pytest.raises(ValueError):
            ff.asymptotic_expansion('hC_scalar', -1.0)

    def test_n_terms_deprecation_warning(self):
        """n_terms != 5 emits DeprecationWarning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ff.asymptotic_expansion('hC_scalar', 100.0, n_terms=3)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)


# =============================================================================
# mpmath form factor functions
# =============================================================================

class TestMpmathFormFactors:
    """Test _mp variants (high-precision mpmath evaluation)."""

    @pytest.fixture(autouse=True)
    def check_mpmath(self):
        try:
            import mpmath  # noqa: F401
        except ImportError:
            pytest.skip("mpmath not available")

    def test_phi_mp_at_zero(self):
        """phi_mp(0) = 1."""
        val = float(ff.phi_mp(0.0))
        assert val == pytest.approx(1.0, rel=1e-30)

    def test_phi_mp_vs_fast(self):
        """phi_mp matches phi_fast to float64 accuracy."""
        for x in [1.0, 5.0, 50.0]:
            mp_val = float(ff.phi_mp(x))
            fast_val = ff.phi_fast(x)
            assert mp_val == pytest.approx(fast_val, rel=1e-12)

    def test_hC_scalar_mp_local_limit(self):
        """hC_scalar_mp(0) = 1/120."""
        val = float(ff.hC_scalar_mp(0.0))
        assert val == pytest.approx(1.0 / 120.0, rel=1e-20)

    def test_hR_scalar_mp_conformal(self):
        """hR_scalar_mp(0, xi=1/6) = 0."""
        val = float(ff.hR_scalar_mp(0.0, xi=1.0 / 6.0))
        assert abs(val) < 1e-25

    def test_hC_dirac_mp_local_limit(self):
        """hC_dirac_mp(0) = -1/20."""
        val = float(ff.hC_dirac_mp(0.0))
        assert val == pytest.approx(-1.0 / 20.0, rel=1e-20)

    def test_hR_dirac_mp_local_limit(self):
        """hR_dirac_mp(0) = 0."""
        val = float(ff.hR_dirac_mp(0.0))
        assert abs(val) < 1e-25

    def test_hC_vector_mp_local_limit(self):
        """hC_vector_mp(0) = 1/10."""
        val = float(ff.hC_vector_mp(0.0))
        assert val == pytest.approx(1.0 / 10.0, rel=1e-20)

    def test_hR_vector_mp_local_limit(self):
        """hR_vector_mp(0) = 0."""
        val = float(ff.hR_vector_mp(0.0))
        assert abs(val) < 1e-25

    def test_hC_scalar_mp_vs_fast(self):
        """hC_scalar_mp matches hC_scalar_fast at x=5."""
        mp_val = float(ff.hC_scalar_mp(5.0))
        fast_val = ff.hC_scalar_fast(5.0)
        assert mp_val == pytest.approx(fast_val, rel=1e-12)

    def test_hC_dirac_mp_vs_fast(self):
        """hC_dirac_mp matches hC_dirac_fast at x=10."""
        mp_val = float(ff.hC_dirac_mp(10.0))
        fast_val = ff.hC_dirac_fast(10.0)
        assert mp_val == pytest.approx(fast_val, rel=1e-12)


# =============================================================================
# Graph functions
# =============================================================================

class TestGraphFunctions:
    """Test spectral action and zeta function on graphs."""

    def test_spectral_action_cycle_graph(self):
        """Spectral action on C_4 cycle graph should be real and finite."""
        from sct_tools.graphs import spectral_action_on_graph
        # C_4 adjacency matrix
        adj = np.array([[0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [1, 0, 1, 0]], dtype=float)
        result = spectral_action_on_graph(adj)
        assert isinstance(result, dict)
        assert 'action' in result
        assert np.isfinite(result['action'])
        assert 'eigenvalues' in result

    def test_spectral_action_with_coefficients(self):
        """Spectral action with custom polynomial coefficients."""
        from sct_tools.graphs import spectral_action_on_graph
        adj = np.eye(3)  # trivial graph
        result = spectral_action_on_graph(adj, coefficients=[1.0, -0.5, 0.1])
        assert isinstance(result, dict)
        assert np.isfinite(result['action'])

    def test_zeta_function_graph_basic(self):
        """Zeta function on K_3 complete graph."""
        from sct_tools.graphs import zeta_function_graph
        # K_3 adjacency
        adj = np.array([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]], dtype=float)
        s_vals = np.array([1.0, 2.0, 3.0])
        result = zeta_function_graph(adj, s_vals)
        assert len(result) == 3
        assert all(np.isfinite(result))

    def test_zeta_function_graph_positive_s(self):
        """Zeta function should converge for s > 0 on non-zero eigenvalues."""
        from sct_tools.graphs import zeta_function_graph
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        result = zeta_function_graph(adj, np.array([2.0]))
        assert np.isfinite(result[0])


# =============================================================================
# Weyl tensor
# =============================================================================

class TestWeylTensor:
    """Test Weyl tensor computation."""

    def test_weyl_tensor_schwarzschild(self):
        """Weyl tensor of Schwarzschild should be non-zero."""
        try:
            import OGRePy  # noqa: F401

            from sct_tools.tensors import schwarzschild, weyl_tensor
        except ImportError:
            pytest.skip("OGRePy not available")
        # schwarzschild() returns (metric, coords, symbols) tuple
        metric_data = schwarzschild()
        if isinstance(metric_data, tuple):
            metric = metric_data[0]
        else:
            metric = metric_data
        W = weyl_tensor(metric)
        assert W is not None
