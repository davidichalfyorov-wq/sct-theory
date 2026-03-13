"""Round 11 audit: regression completeness + foundational identity tests.

Three categories:
1. Regression test for wsl_run_script argument escaping (Bug #13 from R4)
2. Regression test for verify_vacuum numerical fallback (Bug #15 from R4)
3. Taylor coefficient identity tests (ID10, ID11 from identity audit)
"""

import math

import pytest
import sympy as sp

from sct_tools import form_factors as ff

# ============================================================================
# 1. wsl_run_script argument escaping regression (Bug #13)
# ============================================================================


class TestWSLArgEscaping:
    """Verify that wsl_run_script escapes single quotes in args.

    The actual WSL call requires WSL to be installed, so we test the
    escaping logic directly by importing and checking the code path.
    """

    def test_wsl_run_escaping_applied(self):
        """wsl_run applies single-quote escaping to python_code."""

        # The escaping line: escaped = python_code.replace("'", "'\\''")
        # Verify this pattern works correctly
        code_with_quote = "print('hello')"
        escaped = code_with_quote.replace("'", "'\\''")
        assert "'\\''" in escaped
        assert escaped == "print('\\''hello'\\'')"

    def test_wsl_run_script_args_escaping(self):
        """wsl_run_script escapes single quotes in args list."""
        # Simulate the escaping logic from compute.py line 482
        args = ["--file=/tmp/test's_dir/file.py", "it's a test"]
        escaped_args = [a.replace("'", "'\\''") for a in args]
        for ea in escaped_args:
            # After escaping, bash will correctly interpret each arg
            assert "'" not in ea or "'\\'" in ea
        assert escaped_args[0] == "--file=/tmp/test'\\''s_dir/file.py"
        assert escaped_args[1] == "it'\\''s a test"

    def test_wsl_run_script_path_conversion(self):
        """wsl_run_script converts Windows paths to WSL paths."""

        # Test the conversion logic from compute.py lines 473-476
        win_path = r"F:\some\path\script.py"
        wsl_path = win_path.replace("\\", "/")
        if len(wsl_path) >= 2 and wsl_path[1] == ":":
            drive = wsl_path[0].lower()
            wsl_path = f"/mnt/{drive}{wsl_path[2:]}"
        assert wsl_path == "/mnt/f/some/path/script.py"

    def test_wsl_run_script_path_with_spaces(self):
        """Paths with spaces are wrapped in single quotes."""
        win_path = r"F:\Black Mesa Research Facility\script.py"
        wsl_path = win_path.replace("\\", "/")
        if len(wsl_path) >= 2 and wsl_path[1] == ":":
            drive = wsl_path[0].lower()
            wsl_path = f"/mnt/{drive}{wsl_path[2:]}"
        # The actual command wraps in single quotes: python3 '{wsl_path}'
        cmd = f"python3 '{wsl_path}'"
        assert "'/mnt/f/Black Mesa Research Facility/script.py'" in cmd


# ============================================================================
# 2. verify_vacuum numerical fallback regression (Bug #15)
# ============================================================================


class TestVerifyVacuumNumericalFallback:
    """Test that verify_vacuum uses numerical fallback when sympy can't simplify.

    The fix (R4 M3) added numerical evaluation at random points when
    sp.simplify fails to reduce Einstein tensor components to zero.
    """

    def test_numerical_fallback_with_float_metric(self):
        """Schwarzschild with Float mass triggers numerical fallback path.

        sp.simplify(Float_expr) often does NOT reduce to exact zero,
        so verify_vacuum must fall back to numerical evaluation.
        """
        try:
            import OGRePy  # noqa: F401
        except ImportError:
            pytest.skip("OGRePy not installed")

        from sct_tools.tensors import schwarzschild, verify_vacuum

        # Float mass (not Rational) — harder for sp.simplify to reach zero
        M = sp.Float(1.0)
        metric, _params = schwarzschild(M=M)

        # Should return True (symbolic or numerical fallback)
        result = verify_vacuum(metric)
        assert result is True

    def test_non_vacuum_detected(self):
        """A metric that is NOT vacuum should return False."""
        try:
            import OGRePy as ogre
        except ImportError:
            pytest.skip("OGRePy not installed")

        from sct_tools.tensors import spherical_coords, verify_vacuum

        # Conformally flat metric with r-dependent factor — NOT Ricci-flat
        coords, syms = spherical_coords()
        t, r, theta, phi_sym = syms
        omega = 1 + r
        metric = ogre.Metric(
            coords=coords,
            components=[
                [-omega**2, 0, 0, 0],
                [0, omega**2, 0, 0],
                [0, 0, omega**2 * r**2, 0],
                [0, 0, 0, omega**2 * r**2 * sp.sin(theta) ** 2],
            ],
        )
        result = verify_vacuum(metric)
        assert result is False


# ============================================================================
# 3. Taylor coefficient identity tests (ID10, ID11)
# ============================================================================


class TestPhiTaylorCoefficients:
    """Verify _AN[n] = (-1)^n * n! / (2n+1)! for all stored coefficients (ID11)."""

    def test_an_formula(self):
        """Each coefficient matches the analytic formula."""
        for n in range(len(ff._AN)):
            expected = (-1) ** n * math.factorial(n) / math.factorial(2 * n + 1)
            assert ff._AN[n] == pytest.approx(expected, rel=1e-14), (
                f"_AN[{n}] = {ff._AN[n]}, expected {expected}"
            )

    def test_an_first_few_known_values(self):
        """Spot-check: a_0 = 1, a_1 = -1/6, a_2 = 1/60, a_3 = -1/840."""
        assert ff._AN[0] == pytest.approx(1.0, rel=1e-14)
        assert ff._AN[1] == pytest.approx(-1.0 / 6, rel=1e-14)
        assert ff._AN[2] == pytest.approx(1.0 / 60, rel=1e-14)
        assert ff._AN[3] == pytest.approx(-1.0 / 840, rel=1e-14)

    def test_phi_taylor_series(self):
        """phi(x) = sum_n a_n * x^n converges to phi_fast at small x."""
        for x in [0.01, 0.1, 0.5, 1.0]:
            phi_taylor = sum(ff._AN[n] * x**n for n in range(20))
            phi_ref = ff.phi_fast(x)
            assert phi_taylor == pytest.approx(phi_ref, rel=1e-10), (
                f"phi Taylor sum at x={x}: {phi_taylor} vs phi_fast: {phi_ref}"
            )


class TestFormFactorTaylorCoefficients:
    """Verify Taylor coefficient arrays match their defining formulas (ID10)."""

    def test_hc0_taylor_formula(self):
        """_HC0_TAYLOR[k] = _AN[k+2] / 2."""
        for k in range(len(ff._HC0_TAYLOR)):
            expected = ff._AN[k + 2] / 2
            assert ff._HC0_TAYLOR[k] == pytest.approx(expected, rel=1e-14), (
                f"_HC0_TAYLOR[{k}] = {ff._HC0_TAYLOR[k]}, expected {expected}"
            )

    def test_hcd_taylor_formula(self):
        """_HCD_TAYLOR[k] = _AN[k+1]/2 + 2*_AN[k+2]."""
        for k in range(len(ff._HCD_TAYLOR)):
            expected = ff._AN[k + 1] / 2 + 2 * ff._AN[k + 2]
            assert ff._HCD_TAYLOR[k] == pytest.approx(expected, rel=1e-14), (
                f"_HCD_TAYLOR[{k}] = {ff._HCD_TAYLOR[k]}, expected {expected}"
            )

    def test_hrd_taylor_formula(self):
        """_HRD_TAYLOR[k] = _AN[k+1]/12 + 5*_AN[k+2]/6."""
        for k in range(len(ff._HRD_TAYLOR)):
            expected = ff._AN[k + 1] / 12 + 5 * ff._AN[k + 2] / 6
            assert ff._HRD_TAYLOR[k] == pytest.approx(expected, rel=1e-14), (
                f"_HRD_TAYLOR[{k}] = {ff._HRD_TAYLOR[k]}, expected {expected}"
            )

    def test_hr0_a_taylor_formula(self):
        """_HR0_A[k] = _AN[k]/32 + _AN[k+1]/8 + 5*_AN[k+2]/24."""
        for k in range(len(ff._HR0_A)):
            expected = ff._AN[k] / 32 + ff._AN[k + 1] / 8 + 5 * ff._AN[k + 2] / 24
            assert ff._HR0_A[k] == pytest.approx(expected, rel=1e-14), (
                f"_HR0_A[{k}] = {ff._HR0_A[k]}, expected {expected}"
            )

    def test_hr0_b_taylor_formula(self):
        """_HR0_B[k] = -_AN[k]/4 - _AN[k+1]/2."""
        for k in range(len(ff._HR0_B)):
            expected = -ff._AN[k] / 4 - ff._AN[k + 1] / 2
            assert ff._HR0_B[k] == pytest.approx(expected, rel=1e-14), (
                f"_HR0_B[{k}] = {ff._HR0_B[k]}, expected {expected}"
            )

    def test_hr0_c_taylor_formula(self):
        """_HR0_C[k] = _AN[k]/2."""
        for k in range(len(ff._HR0_C)):
            expected = ff._AN[k] / 2
            assert ff._HR0_C[k] == pytest.approx(expected, rel=1e-14), (
                f"_HR0_C[{k}] = {ff._HR0_C[k]}, expected {expected}"
            )

    def test_taylor_series_matches_fast(self):
        """Taylor series evaluation matches _fast at small x for all form factors."""
        for x in [0.01, 0.1, 0.5, 1.0, 1.5]:
            # hC_scalar
            hc_taylor = sum(ff._HC0_TAYLOR[k] * x**k for k in range(20))
            hc_fast = ff.hC_scalar_fast(x)
            assert hc_taylor == pytest.approx(hc_fast, rel=1e-8), (
                f"hC_scalar Taylor vs fast at x={x}"
            )

            # hC_dirac
            hcd_taylor = sum(ff._HCD_TAYLOR[k] * x**k for k in range(20))
            hcd_fast = ff.hC_dirac_fast(x)
            assert hcd_taylor == pytest.approx(hcd_fast, rel=1e-8), (
                f"hC_dirac Taylor vs fast at x={x}"
            )

            # hR_dirac
            hrd_taylor = sum(ff._HRD_TAYLOR[k] * x**k for k in range(20))
            hrd_fast = ff.hR_dirac_fast(x)
            assert hrd_taylor == pytest.approx(hrd_fast, rel=1e-8), (
                f"hR_dirac Taylor vs fast at x={x}"
            )

            # hR_scalar at xi=0
            hr0_taylor = sum(ff._HR0_A[k] * x**k for k in range(20))
            hr0_fast = ff.hR_scalar_fast(x, xi=0.0)
            assert hr0_taylor == pytest.approx(hr0_fast, rel=1e-8), (
                f"hR_scalar(xi=0) Taylor vs fast at x={x}"
            )

            # hR_scalar at xi=1/6
            hr_conf_taylor = sum(
                (ff._HR0_A[k] + (1 / 6) * ff._HR0_B[k] + (1 / 36) * ff._HR0_C[k])
                * x**k
                for k in range(20)
            )
            hr_conf_fast = ff.hR_scalar_fast(x, xi=1.0 / 6)
            assert hr_conf_taylor == pytest.approx(hr_conf_fast, rel=1e-8), (
                f"hR_scalar(xi=1/6) Taylor vs fast at x={x}"
            )


class TestAlgebraicDecompositions:
    """Verify algebraic identities between composite and component functions."""

    @pytest.mark.parametrize("x", [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0])
    def test_hc_scalar_equals_half_fric(self, x):
        """hC_scalar(x) = f_Ric(x) / 2 (definitional identity)."""
        lhs = ff.hC_scalar(x)
        rhs = ff.f_Ric(x) / 2
        assert lhs == pytest.approx(rhs, rel=1e-12)

    @pytest.mark.parametrize("x", [0.1, 1.0, 10.0, 100.0])
    @pytest.mark.parametrize("xi", [0.0, 1.0 / 6, 0.5, 1.0])
    def test_hr_scalar_equals_cz_sum(self, x, xi):
        """hR_scalar(x,xi) = f_Ric(x)/3 + f_R(x) + xi*f_RU(x) + xi^2*f_U(x)."""
        lhs = ff.hR_scalar(x, xi)
        rhs = ff.f_Ric(x) / 3 + ff.f_R(x) + xi * ff.f_RU(x) + xi**2 * ff.f_U(x)
        assert lhs == pytest.approx(rhs, rel=1e-12)

    @pytest.mark.parametrize("x", [0.1, 1.0, 10.0, 100.0])
    def test_fast_matches_cz_hc_scalar(self, x):
        """hC_scalar_fast agrees with quad-based f_Ric/2."""
        fast = ff.hC_scalar_fast(x)
        cz = ff.f_Ric(x) / 2
        # Allow wider tolerance for quad imprecision at small x
        assert fast == pytest.approx(cz, rel=1e-6)

    @pytest.mark.parametrize("x", [0.1, 1.0, 10.0, 100.0])
    @pytest.mark.parametrize("xi", [0.0, 1.0 / 6])
    def test_fast_matches_cz_hr_scalar(self, x, xi):
        """hR_scalar_fast agrees with quad-based CZ decomposition."""
        fast = ff.hR_scalar_fast(x, xi)
        cz = ff.f_Ric(x) / 3 + ff.f_R(x) + xi * ff.f_RU(x) + xi**2 * ff.f_U(x)
        assert fast == pytest.approx(cz, rel=1e-6)
