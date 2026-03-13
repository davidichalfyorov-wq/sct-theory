"""Tests for sct_tools.form_interface — FORM 5.0 subprocess interface.

Most tests mock the FORM binary since it may not be available in CI.
Integration tests with actual FORM are marked with @pytest.mark.form.
"""

from unittest.mock import MagicMock, patch

import pytest

from sct_tools import form_interface


class TestBackendDetection:
    def test_check_form_returns_dict(self):
        result = form_interface.check_form()
        assert isinstance(result, dict)
        assert 'available' in result
        assert 'wsl' in result
        assert 'windows' in result
        assert 'recommended' in result

    def test_recommended_is_valid(self):
        result = form_interface.check_form()
        assert result['recommended'] in ('wsl', 'windows', None)


class TestOutputParsing:
    """Test _parse_form_expression on synthetic FORM output."""

    def test_parse_simple_expression(self):
        output = """
FORM by J.Vermaseren,version 4.2
   F =
      + 4*d_(mu,nu)*d_(rho,sigma)
      - 4*d_(mu,rho)*d_(nu,sigma)
      + 4*d_(mu,sigma)*d_(nu,rho);

  Time =  0.00 sec
"""
        result = form_interface._parse_form_expression(output, "F")
        assert "d_(mu,nu)" in result
        assert "d_(mu,rho)" in result

    def test_parse_zero_result(self):
        output = """
   F = 0;

  Time =  0.00 sec
"""
        result = form_interface._parse_form_expression(output, "F")
        assert result == "0"

    def test_parse_single_term(self):
        output = """
   F =
      4*d_(mu,nu);
"""
        result = form_interface._parse_form_expression(output, "F")
        assert "4*d_(mu,nu)" in result

    def test_parse_different_variable(self):
        output = """
   G =
      12*e_(mu,nu,rho,sigma);
"""
        result = form_interface._parse_form_expression(output, "G")
        assert "12*e_(mu,nu,rho,sigma)" in result

    def test_parse_empty_output(self):
        """Empty output should return empty string."""
        result = form_interface._parse_form_expression("", "F")
        assert result == ""


class TestFormToSympy:
    def test_kronecker_delta(self):
        result = form_interface.parse_form_to_sympy("4*d_(mu,nu)")
        assert "KroneckerDelta" in result

    def test_levi_civita(self):
        result = form_interface.parse_form_to_sympy("4*e_(mu,nu,rho,sigma)")
        assert "LeviCivita" in result


class TestTraceGamma:
    """Test trace_gamma() logic — mocked FORM execution."""

    def test_zero_indices(self):
        """Tr[I] = d = 4."""
        result = form_interface.trace_gamma()
        assert result == "4"

    def test_odd_indices(self):
        """Trace of odd number of gamma matrices vanishes."""
        result = form_interface.trace_gamma("mu", "nu", "rho")
        assert result == "0"

    @patch('sct_tools.form_interface.run_form_script')
    def test_two_indices(self, mock_run):
        """Tr[gamma^mu gamma^nu] = 4 * g^{mu nu}."""
        mock_run.return_value = """
   F =
      4*d_(mu,nu);
"""
        result = form_interface.trace_gamma("mu", "nu")
        assert "4*d_(mu,nu)" in result
        mock_run.assert_called_once()

    @patch('sct_tools.form_interface.run_form_script')
    def test_four_indices_calls_form(self, mock_run):
        """Tr[gamma^mu gamma^nu gamma^rho gamma^sigma] uses FORM."""
        mock_run.return_value = """
   F =
      + 4*d_(mu,nu)*d_(rho,sigma)
      - 4*d_(mu,rho)*d_(nu,sigma)
      + 4*d_(mu,sigma)*d_(nu,rho);
"""
        result = form_interface.trace_gamma("mu", "nu", "rho", "sigma")
        assert "d_(mu,nu)*d_(rho,sigma)" in result
        assert "d_(mu,rho)*d_(nu,sigma)" in result


class TestTraceGammaWithGamma5:
    def test_too_few_indices(self):
        """Need at least 4 gammas with gamma_5."""
        result = form_interface.trace_gamma_with_gamma5("mu", "nu")
        assert result == "0"

    @patch('sct_tools.form_interface.run_form_script')
    def test_four_indices(self, mock_run):
        mock_run.return_value = """
   F =
      + 4*i_*e_(mu,nu,rho,sigma);
"""
        result = form_interface.trace_gamma_with_gamma5("mu", "nu", "rho", "sigma")
        assert "e_(mu,nu,rho,sigma)" in result


class TestSeeleyDeWitt:
    def test_scalar_coefficients(self):
        result = form_interface.seeley_dewitt_a2("scalar")
        assert 'R2' in result
        assert 'RicciSq' in result
        assert 'RiemannSq' in result
        assert 'BoxR' in result
        assert 'OmegaSq' in result

    def test_dirac_coefficients(self):
        result = form_interface.seeley_dewitt_a2("dirac")
        assert 'OmegaSq' in result
        # Dirac has non-zero gauge connection (Omega)
        assert result['OmegaSq'] != '0'

    def test_vector_coefficients(self):
        result = form_interface.seeley_dewitt_a2("vector")
        assert 'OmegaSq' in result

    def test_unknown_field_type(self):
        with pytest.raises(ValueError, match="Unknown field_type"):
            form_interface.seeley_dewitt_a2("tensor")


class TestVerifyTrace:
    @patch('sct_tools.form_interface.trace_gamma')
    def test_verify_two_gammas(self, mock_trace):
        mock_trace.return_value = "4*d_(mu,nu)"
        result = form_interface.verify_trace("mu", "nu")
        assert 'form_result' in result
        assert 'sympy_result' in result
        assert 'agree' in result
        assert 'label' in result

    def test_verify_empty(self):
        """Tr[I] = 4: both FORM and SymPy should agree."""
        result = form_interface.verify_trace()
        assert result['form_result'] == '4'
        assert result['sympy_result'] == '4'
        assert result['agree'] is True

    def test_verify_odd(self):
        """Odd number of gammas: both should give 0."""
        result = form_interface.verify_trace("mu")
        assert result['form_result'] == '0'
        assert result['sympy_result'] == '0'
        assert result['agree'] is True


class TestFormSession:
    def test_context_manager(self):
        with form_interface.FormSession(dim=4) as fs:
            assert fs.dim == 4
            assert isinstance(fs.header_lines, list)

    def test_declare_indices(self):
        with form_interface.FormSession() as fs:
            fs.declare_indices("mu", "nu", "rho")
            assert "mu" in fs._indices
            assert "nu" in fs._indices
            assert "rho" in fs._indices

    def test_declare_vectors(self):
        with form_interface.FormSession() as fs:
            fs.declare_vectors("p", "q")
            assert "p" in fs._vectors
            assert "q" in fs._vectors

    def test_declare_symbols(self):
        with form_interface.FormSession() as fs:
            fs.declare_symbols("m", "M")
            assert "m" in fs._symbols
            assert "M" in fs._symbols

    def test_add_line(self):
        with form_interface.FormSession() as fs:
            fs.add_line("Local F = 1;")
            assert "Local F = 1;" in fs.body_lines

    @patch('sct_tools.form_interface.run_form_script')
    def test_execute_builds_script(self, mock_run):
        mock_run.return_value = "dummy output"
        with form_interface.FormSession(dim=4) as fs:
            fs.declare_indices("mu", "nu")
            fs.add_line("Local F = 1;")
            fs.add_line(".end")
            fs.execute()
        # Check the script was built and passed to run_form_script
        call_args = mock_run.call_args
        script = call_args[0][0]
        assert "Dimension 4;" in script
        assert "Indices mu, nu;" in script
        assert "Local F = 1;" in script

    @patch('sct_tools.form_interface.run_form_script')
    def test_compute_trace(self, mock_run):
        mock_run.return_value = """
   F =
      4*d_(mu,nu);
"""
        with form_interface.FormSession(dim=4) as fs:
            fs.declare_indices("mu", "nu")
            result = fs.compute_trace("g_(1, mu, nu)")
        assert "4*d_(mu,nu)" in result


class TestRunFormScript:
    @patch('sct_tools.form_interface.check_form')
    def test_no_backend_raises(self, mock_check):
        mock_check.return_value = {
            'available': False, 'wsl': False, 'windows': False,
            'recommended': None,
        }
        with pytest.raises(FileNotFoundError, match="FORM not found"):
            form_interface.run_form_script("dummy script", backend='auto')

    @patch('subprocess.run')
    @patch('sct_tools.form_interface.check_form')
    def test_form_error_raises(self, mock_check, mock_subprocess):
        mock_check.return_value = {
            'available': True, 'wsl': False, 'windows': True,
            'recommended': 'windows',
        }
        mock_subprocess.return_value = MagicMock(
            returncode=1, stdout="", stderr="FORM error"
        )
        with pytest.raises(RuntimeError, match="FORM exited with code 1"):
            form_interface.run_form_script("bad script", backend='windows')
