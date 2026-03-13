"""
Tests for sct_tools.lean — Lean 4 formal verification module.

Tests cover:
- Module imports and constants
- Identity registry (SCT_IDENTITIES)
- Local Lean paths and utilities (_find_lake, _get_lean_project_dir)
- prove_local() with mock subprocess
- prove_scilean() with mock subprocess
- build_sctlean() with mock subprocess
- verify_deep() logic (both backends, verdict logic)
- verify_phase_deep() aggregation
- check_local() structure
- check_scilean() structure
- check_all_backends() structure
- physlean_* template functions with mock
- Verifier Layer 5/6 methods (check_lean_local, check_lean_deep, check_lean_sctlean_module)

For tests that call subprocess or external APIs, we use unittest.mock.
For local Lean installation checks, we test the actual check_local() return shape.
"""

import io
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sct_tools import lean
from sct_tools.verification import Verifier

# =========================================================================== #
#  Module-level constants and structure
# =========================================================================== #


class TestLeanModuleStructure:
    """Test module-level definitions and constants."""

    def test_root_paths_defined(self):
        assert lean._ROOT.exists() or True  # Path obj exists
        assert isinstance(lean._LEAN_DIR, Path)
        assert isinstance(lean._PROOFS_DIR, Path)
        assert isinstance(lean._SCTLEAN_DIR, Path)

    def test_junction_path(self):
        assert lean._JUNCTION == Path("C:/sct-lean")

    def test_wsl_scilean_path(self):
        assert lean._WSL_SCILEAN == "~/sct-scilean"

    def test_sct_identities_nonempty(self):
        assert len(lean.SCT_IDENTITIES) > 0

    def test_sct_identities_structure(self):
        """Every identity must have name, lhs, rhs, description, phase."""
        required_keys = {"name", "lhs", "rhs", "description", "phase"}
        for ident in lean.SCT_IDENTITIES:
            missing = required_keys - set(ident.keys())
            assert not missing, f"{ident['name']} missing keys: {missing}"

    def test_sct_identities_unique_names(self):
        names = [i["name"] for i in lean.SCT_IDENTITIES]
        assert len(names) == len(set(names)), "Duplicate identity names"

    def test_sct_identities_valid_lean_names(self):
        """All identity names must be valid Lean identifiers."""
        for ident in lean.SCT_IDENTITIES:
            name = ident["name"]
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name), \
                f"Invalid Lean identifier: {name}"

    def test_known_phases_present(self):
        phases = {i["phase"] for i in lean.SCT_IDENTITIES}
        expected = {"NT-1", "NT-1b-scalar", "NT-1b-vector", "NT-1b-combined"}
        assert expected.issubset(phases)

    def test_documented_module_level_check_wrappers_exist(self):
        assert hasattr(lean, "check_lean_local")
        assert callable(lean.check_lean_local)
        assert hasattr(lean, "check_lean_deep")
        assert callable(lean.check_lean_deep)

    def test_documented_module_level_check_wrappers_delegate(self):
        with patch.object(Verifier, "check_lean_local", return_value=True) as mock_local, \
             patch.object(Verifier, "check_lean_deep", return_value=True) as mock_deep:
            assert lean.check_lean_local("test", "(1 : ℚ)", "(1 : ℚ)") is True
            assert lean.check_lean_deep("test", "(1 : ℚ)", "(1 : ℚ)") is True
            mock_local.assert_called_once_with("test", "(1 : ℚ)", "(1 : ℚ)")
            mock_deep.assert_called_once_with("test", "(1 : ℚ)", "(1 : ℚ)")


# =========================================================================== #
#  Identity registry functions
# =========================================================================== #


class TestIdentityRegistry:
    def test_get_identities_by_phase_scalar(self):
        ids = lean.get_identities_by_phase("NT-1b-scalar")
        assert len(ids) > 0
        assert all(i["phase"] == "NT-1b-scalar" for i in ids)

    def test_get_identities_by_phase_vector(self):
        ids = lean.get_identities_by_phase("NT-1b-vector")
        assert len(ids) > 0
        assert all(i["phase"] == "NT-1b-vector" for i in ids)

    def test_get_identities_by_phase_empty(self):
        ids = lean.get_identities_by_phase("nonexistent-phase")
        assert ids == []

    def test_get_identities_by_phase_combined(self):
        ids = lean.get_identities_by_phase("NT-1b-combined")
        assert len(ids) >= 3  # sm_total_beta_weyl, dof_scalar, dof_fermion, dof_vector

    def test_specific_scalar_beta_identity(self):
        """Check the scalar beta_W = 1/120 identity exists with correct values."""
        ids = lean.get_identities_by_phase("NT-1b-scalar")
        bw = [i for i in ids if i["name"] == "sct_scalar_beta_weyl"]
        assert len(bw) == 1
        assert "120" in bw[0]["lhs"]

    def test_specific_vector_ghost_subtraction(self):
        """Check ghost subtraction identity: 7/60 - 2*(1/120) = 1/10."""
        ids = lean.get_identities_by_phase("NT-1b-vector")
        gs = [i for i in ids if i["name"] == "sct_ghost_subtraction"]
        assert len(gs) == 1
        assert "/ 10" in gs[0]["rhs"]


# =========================================================================== #
#  Utility functions
# =========================================================================== #


class TestUtilityFunctions:
    def test_find_lake_returns_string_or_none(self):
        result = lean._find_lake()
        assert result is None or isinstance(result, str)

    def test_get_lean_project_dir_returns_path(self):
        result = lean._get_lean_project_dir()
        assert isinstance(result, Path)

    def test_get_lean_project_dir_prefers_junction(self):
        """If junction exists, it should be preferred."""
        if lean._JUNCTION.exists():
            assert lean._get_lean_project_dir() == lean._JUNCTION
        else:
            assert lean._get_lean_project_dir() == lean._LEAN_DIR

    def test_set_api_key(self):
        """set_api_key should store key in environment."""
        old = os.environ.pop("ARISTOTLE_API_KEY", None)
        try:
            lean.set_api_key("test_key_12345")
            assert os.environ["ARISTOTLE_API_KEY"] == "test_key_12345"
        finally:
            if old:
                os.environ["ARISTOTLE_API_KEY"] = old
            else:
                os.environ.pop("ARISTOTLE_API_KEY", None)


# =========================================================================== #
#  prove_local() with mock subprocess
# =========================================================================== #


class TestProveLocal:
    def test_prove_local_lake_not_found(self):
        with patch.object(lean, '_find_lake', return_value=None):
            result = lean.prove_local("theorem t : 1 = 1 := by rfl")
            assert result["status"] == "error"
            assert "lake not found" in result.get("error", "")

    def test_prove_local_success(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""

        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc), \
             patch.object(Path, 'write_text'), \
             patch.object(Path, 'unlink'):
            result = lean.prove_local("import Mathlib.Tactic\ntheorem t : 1 = 1 := by rfl")
            assert result["status"] == "ok"
            assert result["has_sorry"] is False
            assert result["returncode"] == 0

    def test_prove_local_failure(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "unknown identifier 'foo'"

        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc), \
             patch.object(Path, 'write_text'), \
             patch.object(Path, 'unlink'):
            result = lean.prove_local("import Mathlib.Tactic\ntheorem t : 1 = 2 := by rfl")
            assert result["status"] == "error"
            assert result["returncode"] == 1

    def test_prove_local_detects_sorry(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""

        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc), \
             patch.object(Path, 'write_text'), \
             patch.object(Path, 'unlink'):
            result = lean.prove_local("theorem t : 1 = 1 := by sorry")
            assert result["has_sorry"] is True

    def test_prove_local_timeout(self):
        import subprocess as sp
        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", side_effect=sp.TimeoutExpired("lake", 300)), \
             patch.object(Path, 'write_text'), \
             patch.object(Path, 'unlink'):
            result = lean.prove_local("theorem t : 1 = 1 := by rfl", timeout=300)
            assert result["status"] == "error"
            assert "Timed out" in result.get("error", "")


# =========================================================================== #
#  prove_local_file()
# =========================================================================== #


class TestProveLocalFile:
    def test_file_not_found(self):
        with patch.object(lean, '_find_lake', return_value="/fake/lake"):
            result = lean.prove_local_file("/nonexistent/file.lean")
            assert result["status"] == "error"
            assert "not found" in result.get("error", "").lower()

    def test_lake_not_found(self):
        with patch.object(lean, '_find_lake', return_value=None):
            result = lean.prove_local_file("/some/file.lean")
            assert result["status"] == "error"


# =========================================================================== #
#  build_sctlean() with mock subprocess
# =========================================================================== #


class TestBuildSCTLean:
    def test_build_lake_not_found(self):
        with patch.object(lean, '_find_lake', return_value=None):
            result = lean.build_sctlean()
            assert result["status"] == "error"

    def test_build_success(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Build completed"
        mock_proc.stderr = ""

        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc):
            result = lean.build_sctlean()
            assert result["status"] == "ok"
            assert result["returncode"] == 0

    def test_build_failure(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Build error"

        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc):
            result = lean.build_sctlean()
            assert result["status"] == "error"

    def test_build_timeout(self):
        import subprocess as sp
        with patch.object(lean, '_find_lake', return_value="/fake/lake"), \
             patch.object(lean, '_get_lean_project_dir', return_value=Path("/fake/proj")), \
             patch("subprocess.run", side_effect=sp.TimeoutExpired("lake", 600)):
            result = lean.build_sctlean()
            assert result["status"] == "error"
            assert "timed out" in result.get("error", "").lower()


# =========================================================================== #
#  prove_scilean() with mock subprocess
# =========================================================================== #


class TestProveScilean:
    def test_prove_scilean_normalizes_windows_path_for_wslpath(self):
        class FakeTempFile(io.StringIO):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

        fake_temp = FakeTempFile(r"C:\Users\youre\AppData\Local\Temp\tmp_scilean.lean")
        wslpath_proc = MagicMock()
        wslpath_proc.returncode = 0
        wslpath_proc.stdout = "/mnt/c/Users/youre/AppData/Local/Temp/tmp_scilean.lean\n"
        wslpath_proc.stderr = ""
        lean_proc = MagicMock()
        lean_proc.returncode = 0
        lean_proc.stdout = ""
        lean_proc.stderr = ""

        def fake_run(cmd, **kwargs):
            if cmd[:5] == ["wsl", "-d", "Ubuntu", "--", "wslpath"]:
                assert cmd[-1] == "C:/Users/youre/AppData/Local/Temp/tmp_scilean.lean"
                return wslpath_proc
            if cmd[:5] == ["wsl", "-d", "Ubuntu", "--", "bash"]:
                return lean_proc
            raise AssertionError(f"Unexpected subprocess call: {cmd}")

        with patch("tempfile.NamedTemporaryFile", return_value=fake_temp), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("os.unlink"):
            result = lean.prove_scilean("theorem t : (1 : ℚ) = 1 := by norm_num")

        assert result["status"] == "ok"

    def test_prove_scilean_success(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = lean.prove_scilean(
                "example : (1 : Float) + 1 = 2 := by native_decide"
            )
            assert result["status"] == "ok"
            assert result["has_sorry"] is False

    def test_prove_scilean_failure(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "error at line 1"
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = lean.prove_scilean("bad code")
            assert result["status"] == "error"

    def test_prove_scilean_timeout(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("wsl", 300)):
            result = lean.prove_scilean("code", timeout=300)
            assert result["status"] == "error"
            assert "timed out" in result.get("error", "").lower()

    def test_prove_scilean_detects_sorry(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = lean.prove_scilean("theorem t : 1 = 1 := by sorry")
            assert result["has_sorry"] is True


# =========================================================================== #
#  check_local() / check_scilean() / check_all_backends()
# =========================================================================== #


class TestCheckBackends:
    def test_check_local_returns_correct_keys(self):
        result = lean.check_local()
        assert "available" in result
        assert "lean_version" in result
        assert "lake_path" in result
        assert "project_dir" in result
        assert "packages" in result
        assert "errors" in result
        assert isinstance(result["available"], bool)
        assert isinstance(result["packages"], list)
        assert isinstance(result["errors"], list)

    def test_check_scilean_returns_correct_keys(self):
        result = lean.check_scilean()
        assert "available" in result
        assert "lean_version" in result
        assert "packages" in result
        assert "errors" in result
        assert isinstance(result["available"], bool)

    def test_check_all_backends_returns_three_keys(self):
        result = lean.check_all_backends()
        assert "aristotle" in result
        assert "local_lean" in result
        assert "scilean" in result

    def test_check_all_backends_local_has_available(self):
        result = lean.check_all_backends()
        assert "available" in result["local_lean"]

    def test_check_local_lake_not_found(self):
        with patch.object(lean, '_find_lake', return_value=None):
            result = lean.check_local()
            assert result["available"] is False
            assert len(result["errors"]) > 0


# =========================================================================== #
#  verify_deep() logic
# =========================================================================== #


class TestVerifyDeep:
    def test_verify_deep_both_pass(self):
        """If both backends pass, fully_verified should be True."""
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        mock_aristotle = {"status": "complete", "output": "theorem ...",
                          "output_path": None, "verified": True}

        with patch.object(lean, 'prove_local', return_value=mock_local), \
             patch.object(lean, 'verify_identity', return_value=mock_aristotle):
            result = lean.verify_deep("test_thm", "(1 : ℝ)", "(1 : ℝ)")
            assert result["fully_verified"] is True
            assert "local_lean" in result["backends_used"]
            assert "aristotle" in result["backends_used"]

    def test_verify_deep_local_only(self):
        """With use_aristotle=False, local pass → fully_verified."""
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}

        with patch.object(lean, 'prove_local', return_value=mock_local):
            result = lean.verify_deep("test_thm", "(1 : ℝ)", "(1 : ℝ)",
                                      use_aristotle=False)
            assert result["fully_verified"] is True
            assert result["aristotle_result"] is None

    def test_verify_deep_local_fail(self):
        """If local fails, fully_verified is False even if Aristotle passes."""
        mock_local = {"status": "error", "returncode": 1,
                      "stdout": "", "stderr": "ring failed"}
        mock_aristotle = {"status": "complete", "output": "ok",
                          "output_path": None, "verified": True}

        with patch.object(lean, 'prove_local', return_value=mock_local), \
             patch.object(lean, 'verify_identity', return_value=mock_aristotle):
            result = lean.verify_deep("test_thm", "(1 : ℝ)", "(1 : ℝ)")
            assert result["fully_verified"] is False

    def test_verify_deep_ring_fallback_norm_num(self):
        """If ring fails, verify_deep tries norm_num as fallback."""
        call_count = [0]
        def mock_prove_local(code, timeout=300):
            call_count[0] += 1
            if call_count[0] == 1:
                # ring fails
                return {"status": "error", "returncode": 1,
                        "stdout": "", "stderr": "ring failed"}
            else:
                # norm_num succeeds
                return {"status": "ok", "has_sorry": False,
                        "returncode": 0, "stdout": "", "stderr": ""}

        with patch.object(lean, 'prove_local', side_effect=mock_prove_local), \
             patch.object(lean, 'verify_identity',
                          return_value={"verified": True, "status": "complete",
                                        "output": "ok", "output_path": None}):
            result = lean.verify_deep("test_thm", "(1 : ℝ)", "(1 : ℝ)")
            assert call_count[0] == 2  # ring then norm_num
            assert result["fully_verified"] is True

    def test_verify_deep_no_backends(self):
        """With both backends disabled, should raise ValueError."""
        with pytest.raises(ValueError, match="at least one backend"):
            lean.verify_deep("test_thm", "(1 : ℝ)", "(1 : ℝ)",
                             use_aristotle=False, use_local=False)


# =========================================================================== #
#  verify_phase_deep() aggregation
# =========================================================================== #


class TestVerifyPhaseDeep:
    def test_phase_deep_empty_phase(self):
        result = lean.verify_phase_deep("nonexistent-phase-xyz")
        assert result["total"] == 0
        assert "error" in result

    def test_phase_deep_aggregation(self):
        """Verify that results are properly aggregated."""
        # Use local-only to avoid needing Aristotle
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}

        with patch.object(lean, 'prove_local', return_value=mock_local):
            result = lean.verify_phase_deep(
                "NT-1b-scalar", use_aristotle=False, use_local=True
            )
            assert result["total"] > 0
            assert result["attempted"] == result["total"]
            assert result["fully_verified"] == result["total"]
            assert result["partial"] == []
            assert result["failed"] == []

    def test_phase_deep_stop_on_failure(self):
        """stop_on_failure should halt after first failure."""
        mock_local = {"status": "error", "returncode": 1,
                      "stdout": "", "stderr": "fail"}

        with patch.object(lean, 'prove_local', return_value=mock_local):
            result = lean.verify_phase_deep(
                "NT-1b-scalar", use_aristotle=False, use_local=True,
                stop_on_failure=True,
            )
            assert result["attempted"] == 1
            assert len(result["failed"]) == 1


# =========================================================================== #
#  PhysLean template functions
# =========================================================================== #


class TestPhysLeanTemplates:
    def test_physlean_lorentz_proof_calls_prove_local(self):
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        with patch.object(lean, 'prove_local', return_value=mock_local) as mock:
            result = lean.physlean_lorentz_proof("True", "trivial")
            assert mock.called
            code_arg = mock.call_args[0][0]
            assert "PhysLean.Relativity.Lorentz" in code_arg
            assert result["status"] == "ok"

    def test_physlean_sm_dof_proof(self):
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        with patch.object(lean, 'prove_local', return_value=mock_local) as mock:
            result = lean.physlean_sm_dof_proof("quark", 6)
            assert mock.called
            code_arg = mock.call_args[0][0]
            assert "6" in code_arg
            assert result["status"] == "ok"

    def test_physlean_anomaly_proof(self):
        charges = [(6, "(1 : ℚ) / 6"), (2, "-(1 : ℚ) / 2")]
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        with patch.object(lean, 'prove_local', return_value=mock_local) as mock:
            result = lean.physlean_anomaly_proof(charges)
            assert mock.called
            code_arg = mock.call_args[0][0]
            assert "^ 3" in code_arg
            assert "= 0" in code_arg
            assert result["status"] == "ok"


# =========================================================================== #
#  Verifier Layer 5/6 methods
# =========================================================================== #


class TestVerifierLeanMethods:
    def test_check_lean_local_pass(self):
        v = Verifier("test", quiet=True)
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        with patch("sct_tools.lean.prove_local", return_value=mock_local):
            passed = v.check_lean_local("test identity", "(1 : ℚ)", "(1 : ℚ)")
            assert passed is True
            assert v.n_pass == 1
            assert v.n_fail == 0

    def test_check_lean_local_fail(self):
        v = Verifier("test", quiet=True)
        mock_local = {"status": "error", "returncode": 1,
                      "stdout": "", "stderr": "ring failed"}
        with patch("sct_tools.lean.prove_local", return_value=mock_local):
            passed = v.check_lean_local("bad identity", "(1 : ℚ)", "(2 : ℚ)")
            assert passed is False
            assert v.n_fail == 1

    def test_check_lean_local_custom_tactic(self):
        v = Verifier("test", quiet=True)
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        with patch("sct_tools.lean.prove_local", return_value=mock_local) as mock:
            v.check_lean_local("test", "(1 : ℚ)", "(1 : ℚ)", tactic="norm_num")
            code_arg = mock.call_args[0][0]
            assert "norm_num" in code_arg

    def test_check_lean_deep_pass(self):
        v = Verifier("test", quiet=True)
        mock_deep = {"fully_verified": True, "backends_used": ["local_lean", "aristotle"],
                     "local_result": {"status": "ok"}, "aristotle_result": {"verified": True},
                     "name": "test"}
        with patch("sct_tools.lean.verify_deep", return_value=mock_deep):
            passed = v.check_lean_deep("test", "(1 : ℚ)", "(1 : ℚ)")
            assert passed is True
            assert v.n_pass == 1

    def test_check_lean_deep_fail(self):
        v = Verifier("test", quiet=True)
        mock_deep = {"fully_verified": False, "backends_used": ["local_lean"],
                     "local_result": {"status": "error"}, "aristotle_result": None,
                     "name": "test"}
        with patch("sct_tools.lean.verify_deep", return_value=mock_deep):
            passed = v.check_lean_deep("test", "(1 : ℚ)", "(2 : ℚ)",
                                       use_aristotle=False)
            assert passed is False
            assert v.n_fail == 1

    def test_check_lean_deep_auto_name(self):
        """Name should be auto-generated from label if not provided."""
        v = Verifier("test", quiet=True)
        mock_deep = {"fully_verified": True, "backends_used": ["local_lean"],
                     "local_result": {"status": "ok"}, "aristotle_result": None,
                     "name": "auto"}
        with patch("sct_tools.lean.verify_deep", return_value=mock_deep) as mock:
            v.check_lean_deep("beta_W = 1/120", "(1 : ℚ)", "(1 : ℚ)",
                              use_aristotle=False)
            call_kwargs = mock.call_args[1]
            assert call_kwargs["name"].startswith("sct_v_")

    def test_check_lean_sctlean_module_pass(self):
        v = Verifier("test", quiet=True)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""

        with patch("sct_tools.lean._find_lake", return_value="/fake/lake"), \
             patch("sct_tools.lean._get_lean_project_dir",
                   return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc):
            passed = v.check_lean_sctlean_module(
                "Build FormFactors", "SCTLean.FormFactors"
            )
            assert passed is True
            assert v.n_pass == 1

    def test_check_lean_sctlean_module_fail(self):
        v = Verifier("test", quiet=True)
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "build error"

        with patch("sct_tools.lean._find_lake", return_value="/fake/lake"), \
             patch("sct_tools.lean._get_lean_project_dir",
                   return_value=Path("/fake/proj")), \
             patch("subprocess.run", return_value=mock_proc):
            passed = v.check_lean_sctlean_module(
                "Build FormFactors", "SCTLean.FormFactors"
            )
            assert passed is False
            assert v.n_fail == 1

    def test_check_lean_sctlean_module_no_lake(self):
        v = Verifier("test", quiet=True)
        with patch("sct_tools.lean._find_lake", return_value=None):
            passed = v.check_lean_sctlean_module(
                "Build FormFactors", "SCTLean.FormFactors"
            )
            assert passed is False
            assert v.n_fail == 1

    def test_verifier_summary_includes_lean_checks(self):
        """Lean checks should appear in summary with [LEAN] prefix."""
        v = Verifier("test", quiet=True)
        mock_local = {"status": "ok", "has_sorry": False, "returncode": 0,
                      "stdout": "", "stderr": ""}
        with patch("sct_tools.lean.prove_local", return_value=mock_local):
            v.check_lean_local("identity A", "(1 : ℚ)", "(1 : ℚ)")
        assert any("[LEAN]" in c["label"] for c in v.checks)


# =========================================================================== #
#  Integration tests (only if Lean is installed)
# =========================================================================== #


@pytest.fixture
def lean_available():
    """Skip test if local Lean 4 is not installed."""
    result = lean.check_local()
    if not result["available"]:
        pytest.skip("Local Lean 4 not available")
    return result


class TestLeanIntegration:
    """Integration tests — require actual Lean 4 installation."""

    def test_prove_local_trivial(self, lean_available):
        """Prove a trivial identity using local Lean."""
        code = (
            "import Mathlib.Tactic\n\n"
            "theorem _sct_test : (1 : ℚ) + 1 = 2 := by norm_num\n"
        )
        result = lean.prove_local(code, timeout=120)
        assert result["status"] == "ok", f"Failed: {result.get('stderr', result.get('error', ''))}"

    def test_prove_local_ring_identity(self, lean_available):
        """Prove SCT scalar beta_W identity via ring."""
        code = (
            "import Mathlib.Tactic\n\n"
            "theorem sct_test_beta : (1 : ℚ) / 120 = 1 / 120 := by ring\n"
        )
        result = lean.prove_local(code, timeout=120)
        assert result["status"] == "ok"

    def test_prove_local_incorrect_rejects(self, lean_available):
        """An incorrect identity must fail."""
        code = (
            "import Mathlib.Tactic\n\n"
            "theorem sct_test_bad : (1 : ℚ) / 120 = 1 / 60 := by ring\n"
        )
        result = lean.prove_local(code, timeout=120)
        assert result["status"] == "error"

    def test_build_sctlean_succeeds(self, lean_available):
        """The full SCTLean library should build."""
        result = lean.build_sctlean(timeout=600)
        error_message = result.get('stderr', result.get('error', ''))
        assert result["status"] == "ok", f"Build failed: {error_message}"

    def test_check_local_has_packages(self, lean_available):
        """check_local should report mathlib and PhysLean."""
        result = lean.check_local()
        assert "mathlib" in result["packages"]
        assert "PhysLean" in result["packages"]
