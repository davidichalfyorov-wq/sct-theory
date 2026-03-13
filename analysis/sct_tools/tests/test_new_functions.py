"""
Tests for ALL new sct_tools functions added in the enhancement pass.

Covers: constants (NaturalUnits), compute (benchmark, cache_info),
        form_factors (derivatives, Taylor export, asymptotics),
        fitting (MINOS, chi2_cov, bayesian_limit, discovery_significance),
        data_io (read_csv), graphs (spectral_action, zeta_function),
        verification (quiet mode, check_numerical_stability).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import compute, data_io, fitting, graphs, verification
from sct_tools import constants as const
from sct_tools import form_factors as ff

# =============================================================================
# constants.py — NaturalUnits, conversion factors
# =============================================================================

class TestNaturalUnitConversions:
    def test_inv_GeV_to_m(self):
        # 1 GeV^{-1} ~ 1.97e-16 m
        assert const.inv_GeV_to_m == pytest.approx(1.97e-16, rel=0.01)

    def test_inv_GeV_to_s(self):
        # 1 GeV^{-1} ~ 6.58e-25 s
        assert const.inv_GeV_to_s == pytest.approx(6.58e-25, rel=0.01)

    def test_GeV_to_K(self):
        # 1 GeV ~ 1.16e13 K
        assert const.GeV_to_K == pytest.approx(1.16e13, rel=0.01)

    def test_inv_GeV2_to_pb(self):
        # 1 GeV^{-2} ~ 0.3894 mb = 3.894e8 pb
        assert const.inv_GeV2_to_pb == pytest.approx(3.894e8, rel=0.01)

    def test_planck_natural(self):
        assert const.M_Pl_natural == const.M_Pl_GeV
        assert const.l_Pl_natural == pytest.approx(1.0 / const.M_Pl_GeV, rel=1e-14)


class TestNaturalUnitsContext:
    def test_context_manager(self):
        with const.NaturalUnits() as nu:
            assert nu.M_Pl == pytest.approx(1.22e19, rel=0.01)
            assert nu.alpha_em == pytest.approx(1 / 137.036, rel=1e-4)

    def test_to_meters(self):
        with const.NaturalUnits() as nu:
            # Planck length: 1/M_Pl in GeV^{-1}
            l_pl = 1.0 / nu.M_Pl
            l_si = nu.to_meters(l_pl)
            assert l_si == pytest.approx(1.616e-35, rel=0.01)

    def test_to_kelvin(self):
        with const.NaturalUnits() as nu:
            T = nu.to_kelvin(1.0)  # 1 GeV in Kelvin
            assert T == pytest.approx(1.16e13, rel=0.01)

    def test_to_pb(self):
        with const.NaturalUnits() as nu:
            sigma = nu.to_pb(1.0)  # 1 GeV^{-2} in pb
            assert sigma == pytest.approx(3.894e8, rel=0.01)


# =============================================================================
# compute.py — benchmark, cache_info
# =============================================================================

class TestBenchmark:
    def test_context_manager(self):
        with compute.benchmark("test") as b:
            _ = sum(range(1000))
        assert b.elapsed > 0
        assert b.label == "test"

    def test_decorator(self):
        @compute.benchmark("decorated")
        def f():
            return sum(range(100))
        result = f()
        assert result == 4950


class TestCacheInfo:
    def test_returns_dict(self):
        info = compute.cache_info()
        assert 'path' in info
        assert 'size_mb' in info
        assert 'exists' in info
        assert isinstance(info['size_mb'], float)


# =============================================================================
# form_factors.py — derivatives, Taylor export, asymptotics
# =============================================================================

class TestPhiDerivative:
    def test_dphi_at_zero(self):
        # phi'(0) = -1/6 (from Taylor: phi ~ 1 - x/6 + ...)
        assert ff.dphi_dx_fast(0) == pytest.approx(-1.0 / 6, rel=1e-10)

    def test_dphi_negative(self):
        # phi is decreasing, so phi' < 0 for x > 0
        for x in [0.1, 1, 10]:
            assert ff.dphi_dx_fast(x) < 0

    def test_dphi_quad_vs_fast(self):
        # Quadrature and fast (Dawson) should agree
        for x in [0.5, 2.0, 10.0]:
            quad_val = ff.dphi_dx(x)
            fast_val = ff.dphi_dx_fast(x)
            assert quad_val == pytest.approx(fast_val, rel=1e-8)


class TestFormFactorDerivatives:
    def test_dhC_scalar_finite(self):
        for x in [0.0, 0.1, 1.0, 10.0]:
            val = ff.dhC_scalar_dx(x)
            assert np.isfinite(val)

    def test_dhC_dirac_finite(self):
        for x in [0.0, 0.1, 1.0, 10.0]:
            val = ff.dhC_dirac_dx(x)
            assert np.isfinite(val)

    def test_dhR_dirac_finite(self):
        for x in [0.0, 0.1, 1.0, 10.0]:
            val = ff.dhR_dirac_dx(x)
            assert np.isfinite(val)

    def test_derivative_numerical(self):
        # Compare analytic derivative with finite difference
        x = 1.0
        h = 1e-6
        fd = (ff.hC_scalar(x + h) - ff.hC_scalar(x - h)) / (2 * h)
        analytic = ff.dhC_scalar_dx(x)
        assert fd == pytest.approx(analytic, rel=1e-4)

    def test_derivative_dirac_numerical(self):
        x = 2.0
        h = 1e-6
        fd = (ff.hC_dirac(x + h) - ff.hC_dirac(x - h)) / (2 * h)
        analytic = ff.dhC_dirac_dx(x)
        assert fd == pytest.approx(analytic, rel=1e-4)


class TestTaylorExport:
    def test_phi_coefficients(self):
        coeffs = ff.get_taylor_coefficients('phi', n_terms=5)
        assert len(coeffs) == 5
        assert coeffs[0] == pytest.approx(1.0, abs=1e-14)
        assert coeffs[1] == pytest.approx(-1.0 / 6, rel=1e-10)

    def test_hC_scalar_coefficients(self):
        coeffs = ff.get_taylor_coefficients('hC_scalar', n_terms=3)
        assert len(coeffs) == 3
        # First coefficient = h_C^(0)(0) = 1/120
        assert coeffs[0] == pytest.approx(1.0 / 120, rel=1e-10)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            ff.get_taylor_coefficients('nonexistent')


class TestAsymptoticExpansion:
    def test_hC_scalar_large_x(self):
        x = 10000.0
        approx_val = ff.asymptotic_expansion('hC_scalar', x, n_terms=3)
        exact_val = ff.hC_scalar(x)
        assert approx_val == pytest.approx(exact_val, rel=0.02)

    def test_hC_dirac_large_x(self):
        x = 1000.0
        approx_val = ff.asymptotic_expansion('hC_dirac', x, n_terms=3)
        exact_val = ff.hC_dirac(x)
        assert approx_val == pytest.approx(exact_val, rel=0.01)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            ff.asymptotic_expansion('nonexistent', 10.0)


# =============================================================================
# fitting.py — chi2_cov, bayesian_limit, discovery_significance
# =============================================================================

class TestChi2Cov:
    def test_diagonal_case(self):
        # With diagonal covariance, should equal standard chi2
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.1, 3.1])
        cov = np.diag([0.01, 0.01, 0.01])
        chi2_val, ndof, chi2_red, p_val = fitting.chi2_cov(obs, exp, cov)
        expected = np.sum((obs - exp)**2 / 0.01)
        assert chi2_val == pytest.approx(expected, rel=1e-10)
        assert ndof == 3

    def test_correlated(self):
        obs = np.array([1.0, 2.0])
        exp = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        chi2_val, ndof, chi2_red, p_val = fitting.chi2_cov(obs, exp, cov)
        assert chi2_val == pytest.approx(0.0, abs=1e-14)


class TestBayesianLimit:
    def test_upper_limit(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        result = fitting.bayesian_limit(samples, cl=0.95, side='upper')
        assert 'limit' in result
        # 95% upper limit of N(0,1) should be ~1.645
        assert result['limit'] == pytest.approx(1.645, rel=0.1)

    def test_lower_limit(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        result = fitting.bayesian_limit(samples, cl=0.95, side='lower')
        assert result['limit'] == pytest.approx(-1.645, rel=0.1)

    def test_hdi(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        result = fitting.bayesian_limit(samples, cl=0.68, side='hdi')
        assert 'lower' in result
        assert 'upper' in result
        assert result['lower'] < 0 < result['upper']


class TestDiscoverySignificance:
    def test_known_values(self):
        # Z value should be ~5 for s=25, b=1
        result = fitting.discovery_significance(25, 1)
        assert result['Z'] > 4.0
        assert 'p_value' in result

    def test_zero_signal(self):
        result = fitting.discovery_significance(0, 10)
        assert result['Z'] == pytest.approx(0.0, abs=1e-10)

    def test_returns_p_value(self):
        result = fitting.discovery_significance(100, 10)
        assert result['p_value'] < 1e-5


# =============================================================================
# data_io.py — read_csv
# =============================================================================

class TestReadCSV:
    def test_csv_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        result = data_io.read_csv(str(f), delimiter=',')
        assert result['n_rows'] == 2
        assert 'a' in result['columns']
        assert result['data']['a'][0] == 1

    def test_tsv_file(self, tmp_path):
        f = tmp_path / "test.tsv"
        f.write_text("x\ty\n1.0\t2.0\n3.0\t4.0\n")
        result = data_io.read_csv(str(f), delimiter='\t')
        assert result['n_rows'] == 2

    def test_comment_lines(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_text("# Header comment\n# Another comment\ncol1 col2\n1.0 2.0\n3.0 4.0\n")
        result = data_io.read_csv(str(f), delimiter=r'\s+')
        assert result['n_rows'] == 2

    def test_auto_delimiter_warns(self, tmp_path):
        """Auto-detected delimiter emits a warning."""
        import warnings
        f = tmp_path / "test.dat"
        f.write_text("col1 col2\n1.0 2.0\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_io.read_csv(str(f))
            assert len(w) == 1
            assert "auto-detected" in str(w[0].message)


# =============================================================================
# graphs.py — spectral_action_on_graph, zeta_function_graph
# =============================================================================

class TestSpectralAction:
    def test_identity_action(self):
        # f(x) = 1, Tr(f(L)) = N
        A = np.ones((4, 4)) - np.eye(4)  # K4
        result = graphs.spectral_action_on_graph(A, f=lambda x: 1.0)
        assert result['action'] == pytest.approx(4.0, abs=1e-12)

    def test_heat_kernel_consistency(self):
        # f(x) = e^{-t*x} should give same as heat_kernel_trace
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        t = 1.0
        result = graphs.spectral_action_on_graph(A, f=lambda x: np.exp(-t * x))
        hk = graphs.heat_kernel_trace(A, [t])
        assert result['action'] == pytest.approx(hk[0], rel=1e-10)

    def test_polynomial_coefficients(self):
        # f(x) = 1 - x => Tr(I - L) = N - Tr(L) = N - 2E/N... test just runs
        A = np.eye(3, k=1) + np.eye(3, k=-1)
        result = graphs.spectral_action_on_graph(A, coefficients=[1.0, -1.0])
        assert np.isfinite(result['action'])

    def test_n_eigenvalues(self):
        A = np.ones((10, 10)) - np.eye(10)
        result = graphs.spectral_action_on_graph(A, f=lambda x: 1.0, n_eigenvalues=3)
        assert result['action'] == pytest.approx(3.0, abs=1e-12)
        assert len(result['eigenvalues']) == 3


class TestZetaFunction:
    def test_positive_s(self):
        A = np.ones((4, 4)) - np.eye(4)  # K4: eigenvalues 0, 4, 4, 4
        zeta = graphs.zeta_function_graph(A, [1.0])
        # zeta(1) = 3 * (1/4) = 0.75
        assert zeta[0] == pytest.approx(0.75, rel=1e-10)

    def test_multiple_s(self):
        A = np.eye(3, k=1) + np.eye(3, k=-1)
        zeta = graphs.zeta_function_graph(A, [1.0, 2.0])
        assert len(zeta) == 2
        assert all(np.isfinite(zeta))


# =============================================================================
# verification.py — quiet mode, check_numerical_stability
# =============================================================================

class TestVerifierQuietMode:
    def test_quiet_no_output(self, capsys):
        v = verification.Verifier("test", quiet=True)
        v.check_value("silent check", 1.0, 1.0)
        captured = capsys.readouterr()
        assert "PASS" not in captured.out

    def test_normal_has_output(self, capsys):
        v = verification.Verifier("test", quiet=False)
        v.check_value("loud check", 1.0, 1.0)
        captured = capsys.readouterr()
        assert "PASS" in captured.out


class TestNumericalStability:
    def test_stable_function(self):
        # Use a gently varying function — small relative changes between consecutive points
        result = verification.check_numerical_stability(
            lambda x: 100 + x, np.linspace(0, 3, 50), label="100+x"
        )
        assert result['stable'] is True
        assert len(result['issues']) == 0

    def test_detects_nan(self):
        def bad_func(x):
            if x > 5:
                return float('nan')
            return x

        result = verification.check_numerical_stability(
            bad_func, np.linspace(0, 10, 20), label="bad"
        )
        assert result['stable'] is False
        assert any("NaN" in desc for _, desc in result['issues'])

    def test_detects_overflow(self):
        result = verification.check_numerical_stability(
            lambda x: np.exp(x), np.linspace(0, 800, 50), label="exp"
        )
        # exp(800) ~ 2.7e347, should trigger near-overflow or Inf
        assert result['stable'] is False


# =============================================================================
# WSL INTEGRATION TESTS
# =============================================================================

class TestWSLIntegration:
    """Tests for WSL2 helper functions (wsl_run, wsl_run_script, wsl_check)."""

    def test_wsl_check(self):
        """Verify wsl_check() returns expected structure."""
        result = compute.wsl_check()
        assert 'available' in result
        assert 'packages' in result
        assert 'errors' in result
        assert isinstance(result['packages'], dict)
        assert isinstance(result['errors'], list)

    @pytest.mark.skipif(
        not compute.wsl_check()['available'],
        reason="WSL2 not available"
    )
    def test_wsl_run_basic(self):
        """Test basic Python execution in WSL."""
        result = compute.wsl_run(
            'import json; print(json.dumps({"x": 42}))',
            return_json=True
        )
        assert result == {"x": 42}

    @pytest.mark.skipif(
        not compute.wsl_check()['available'],
        reason="WSL2 not available"
    )
    def test_wsl_run_numpy(self):
        """Test numpy is available in WSL venv."""
        result = compute.wsl_run(
            'import json, numpy as np; print(json.dumps({"pi": float(np.pi)}))',
            return_json=True
        )
        assert abs(result['pi'] - 3.14159265) < 1e-6

    @pytest.mark.skipif(
        not compute.wsl_check()['available'],
        reason="WSL2 not available"
    )
    def test_wsl_run_raw_output(self):
        """Test return_json=False returns raw string."""
        result = compute.wsl_run('print("hello from wsl")', return_json=False)
        assert result == "hello from wsl"

    @pytest.mark.skipif(
        not compute.wsl_check()['available'],
        reason="WSL2 not available"
    )
    def test_wsl_packages_available(self):
        """Verify all 4 WSL-specific packages are importable."""
        status = compute.wsl_check()
        assert 'healpy' in status['packages']
        assert 'classy' in status['packages']
        assert 'cadabra2' in status['packages']
        assert 'pySecDec' in status['packages']
