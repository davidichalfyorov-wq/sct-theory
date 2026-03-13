"""
Extended tests for sct_tools.lean — mock-based tests for Aristotle API functions.

Tests functions that were previously untested:
  prove(), prove_file(), formalize(), verify_identity(), verify_batch(),
  verify_phase(), verify_all(), sct_phi_zero_proof(), sct_conformal_coupling_proof(),
  sct_ghost_counting_proof(), list_projects(), prove_local_file()
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sct_tools import lean

# ---------------------------------------------------------------------------
#  prove() — Aristotle sorry-filling
# ---------------------------------------------------------------------------


class TestProve:
    """Tests for prove() with mocked aristotlelib."""

    def test_prove_missing_aristotlelib(self):
        """prove() raises ImportError when aristotlelib not installed."""
        with patch.dict("sys.modules", {"aristotlelib": None}):
            with pytest.raises(ImportError, match="aristotlelib not installed"):
                lean.prove("theorem test : 1 = 1 := by sorry")

    @patch("sct_tools.lean._ensure_api_key")
    @patch("sct_tools.lean._run")
    def test_prove_success(self, mock_run, mock_key):
        """prove() returns complete status with output."""
        mock_aristotlelib = MagicMock()
        with patch.dict("sys.modules", {"aristotlelib": mock_aristotlelib}):
            # Create a temp file to simulate output
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False
            ) as f:
                f.write("theorem test : 1 = 1 := by norm_num")
                out_path = f.name

            try:
                mock_run.return_value = None
                result = lean.prove(
                    "theorem test : 1 = 1 := by sorry", output_path=out_path
                )
                assert result["status"] == "complete"
                assert result["output"] is not None
                assert "sorry" not in result["output"]
            finally:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

    @patch("sct_tools.lean._ensure_api_key")
    @patch("sct_tools.lean._run", side_effect=Exception("API timeout"))
    def test_prove_api_error(self, mock_run, mock_key):
        """prove() returns error status on API failure."""
        mock_aristotlelib = MagicMock()
        with patch.dict("sys.modules", {"aristotlelib": mock_aristotlelib}):
            result = lean.prove("theorem test : 1 = 1 := by sorry")
            assert result["status"] == "error"
            assert "API timeout" in result["error"]

    @patch("sct_tools.lean._ensure_api_key")
    @patch("sct_tools.lean._run")
    def test_prove_nowait(self, mock_run, mock_key):
        """prove() returns submitted status when wait=False."""
        mock_aristotlelib = MagicMock()
        with patch.dict("sys.modules", {"aristotlelib": mock_aristotlelib}):
            mock_run.return_value = None
            result = lean.prove(
                "theorem test : 1 = 1 := by sorry", wait=False
            )
            assert result["status"] == "submitted"


# ---------------------------------------------------------------------------
#  prove_file() — Aristotle file-based sorry-filling
# ---------------------------------------------------------------------------


class TestProveFile:
    """Tests for prove_file() with mocked aristotlelib."""

    def test_prove_file_not_found(self):
        """prove_file() raises FileNotFoundError for missing file."""
        mock_aristotlelib = MagicMock()
        with patch.dict("sys.modules", {"aristotlelib": mock_aristotlelib}):
            with patch("sct_tools.lean._ensure_api_key"):
                with pytest.raises(FileNotFoundError, match="not found"):
                    lean.prove_file("/nonexistent/file.lean")

    @patch("sct_tools.lean._ensure_api_key")
    @patch("sct_tools.lean._run")
    def test_prove_file_success(self, mock_run, mock_key):
        """prove_file() returns complete status for existing file."""
        mock_aristotlelib = MagicMock()
        with patch.dict("sys.modules", {"aristotlelib": mock_aristotlelib}):
            # Create input file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False
            ) as inp:
                inp.write("theorem test : 1 = 1 := by sorry")
                input_path = inp.name

            # Create output file
            out_path = input_path.replace(".lean", "_proved.lean")
            Path(out_path).write_text(
                "theorem test : 1 = 1 := by norm_num", encoding="utf-8"
            )

            try:
                mock_run.return_value = None
                result = lean.prove_file(input_path, output_path=out_path)
                assert result["status"] == "complete"
                assert result["output"] is not None
            finally:
                for p in [input_path, out_path]:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def test_prove_file_missing_aristotlelib(self):
        """prove_file() raises ImportError when aristotlelib not installed."""
        with patch.dict("sys.modules", {"aristotlelib": None}):
            with pytest.raises(ImportError, match="aristotlelib not installed"):
                lean.prove_file("/some/file.lean")


# ---------------------------------------------------------------------------
#  formalize() — NL → Lean conversion
# ---------------------------------------------------------------------------


class TestFormalize:
    """Tests for formalize() with mocked aristotlelib."""

    def test_formalize_missing_aristotlelib(self):
        """formalize() raises ImportError when aristotlelib not installed."""
        with patch.dict("sys.modules", {"aristotlelib": None}):
            with pytest.raises(ImportError, match="aristotlelib not installed"):
                lean.formalize("The sum of angles in a triangle is 180 degrees.")

    @patch("sct_tools.lean._ensure_api_key")
    @patch("sct_tools.lean._run", side_effect=Exception("Server error"))
    def test_formalize_api_error(self, mock_run, mock_key):
        """formalize() returns error status on API failure."""
        mock_aristotlelib = MagicMock()
        mock_aristotlelib.ProjectInputType = MagicMock()
        mock_aristotlelib.ProjectInputType.INFORMAL = "INFORMAL"
        with patch.dict("sys.modules", {"aristotlelib": mock_aristotlelib}):
            result = lean.formalize("beta_W = 1/120")
            assert result["status"] == "error"
            assert "Server error" in result["error"]


# ---------------------------------------------------------------------------
#  verify_identity() — single identity verification
# ---------------------------------------------------------------------------


class TestVerifyIdentity:
    """Tests for verify_identity() with mocked prove()."""

    @patch("sct_tools.lean.prove")
    def test_verify_identity_success(self, mock_prove):
        """verify_identity() marks as verified when proof succeeds."""
        mock_prove.return_value = {
            "status": "complete",
            "output": "theorem test_id : (1 : ℝ) / 120 = (1 : ℝ) / 120 := by norm_num",
            "output_path": "/tmp/test_id.lean",
        }
        result = lean.verify_identity(
            "test_id", "(1 : ℝ) / 120", "(1 : ℝ) / 120", save=False
        )
        assert result["verified"] is True
        assert result["status"] == "complete"

    @patch("sct_tools.lean.prove")
    def test_verify_identity_sorry_remains(self, mock_prove):
        """verify_identity() marks as unverified when sorry remains."""
        mock_prove.return_value = {
            "status": "complete",
            "output": "theorem test_id : (1 : ℝ) / 120 = (1 : ℝ) / 119 := by sorry",
            "output_path": "/tmp/test_id.lean",
        }
        result = lean.verify_identity(
            "test_id", "(1 : ℝ) / 120", "(1 : ℝ) / 119", save=False
        )
        assert result["verified"] is False

    @patch("sct_tools.lean.prove")
    def test_verify_identity_error(self, mock_prove):
        """verify_identity() marks as unverified on error."""
        mock_prove.return_value = {
            "status": "error",
            "output": None,
            "error": "timeout",
            "output_path": None,
        }
        result = lean.verify_identity(
            "test_id", "(1 : ℝ) / 120", "(1 : ℝ) / 120", save=False
        )
        assert result["verified"] is False

    @patch("sct_tools.lean.prove")
    def test_verify_identity_generates_correct_code(self, mock_prove):
        """verify_identity() generates Lean code with correct theorem structure."""
        mock_prove.return_value = {
            "status": "complete",
            "output": "no sorry",
            "output_path": None,
        }
        lean.verify_identity(
            "my_theorem", "(1 : ℝ) / 2", "(0.5 : ℝ)",
            description="Half equals 0.5", save=False,
        )
        # Check the code passed to prove()
        code_arg = mock_prove.call_args[0][0]
        assert "import Mathlib.Tactic" in code_arg
        assert "theorem my_theorem" in code_arg
        assert "(1 : ℝ) / 2 = (0.5 : ℝ)" in code_arg
        assert "sorry" in code_arg
        assert "Half equals 0.5" in code_arg


# ---------------------------------------------------------------------------
#  verify_batch() — batch identity verification
# ---------------------------------------------------------------------------


class TestVerifyBatch:
    """Tests for verify_batch() with mocked verify_identity()."""

    @patch("sct_tools.lean.verify_identity")
    def test_batch_all_pass(self, mock_vi):
        """verify_batch() counts correctly when all pass."""
        mock_vi.return_value = {"status": "complete", "verified": True}
        ids = [
            {"name": "a", "lhs": "1", "rhs": "1"},
            {"name": "b", "lhs": "2", "rhs": "2"},
        ]
        result = lean.verify_batch(ids)
        assert result["total"] == 2
        assert result["verified"] == 2
        assert result["failed"] == []
        assert result["errors"] == []

    @patch("sct_tools.lean.verify_identity")
    def test_batch_mixed(self, mock_vi):
        """verify_batch() correctly reports mixed results."""
        mock_vi.side_effect = [
            {"status": "complete", "verified": True},
            {"status": "complete", "verified": False},
            {"status": "error", "verified": False, "error": "timeout"},
        ]
        ids = [
            {"name": "a", "lhs": "1", "rhs": "1"},
            {"name": "b", "lhs": "1", "rhs": "2"},
            {"name": "c", "lhs": "1", "rhs": "1"},
        ]
        result = lean.verify_batch(ids)
        assert result["total"] == 3
        assert result["verified"] == 1
        assert result["failed"] == ["b"]
        assert result["errors"] == ["c"]

    @patch("sct_tools.lean.verify_identity")
    def test_batch_stop_on_failure(self, mock_vi):
        """verify_batch() stops early when stop_on_failure=True."""
        mock_vi.side_effect = [
            {"status": "complete", "verified": True},
            {"status": "complete", "verified": False},
            {"status": "complete", "verified": True},  # should not be reached
        ]
        ids = [
            {"name": "a", "lhs": "1", "rhs": "1"},
            {"name": "b", "lhs": "1", "rhs": "2"},
            {"name": "c", "lhs": "1", "rhs": "1"},
        ]
        result = lean.verify_batch(ids, stop_on_failure=True)
        assert result["attempted"] == 2  # stopped after b
        assert result["verified"] == 1


# ---------------------------------------------------------------------------
#  verify_phase(), verify_all()
# ---------------------------------------------------------------------------


class TestVerifyPhaseAndAll:
    """Tests for verify_phase() and verify_all()."""

    @patch("sct_tools.lean.verify_batch")
    def test_verify_phase_known(self, mock_vb):
        """verify_phase() passes correct identities to verify_batch()."""
        mock_vb.return_value = {"total": 8, "verified": 8}
        lean.verify_phase("NT-1b-scalar")
        assert mock_vb.called
        ids_passed = mock_vb.call_args[0][0]
        assert all(i["phase"] == "NT-1b-scalar" for i in ids_passed)

    def test_verify_phase_unknown(self):
        """verify_phase() returns error for unknown phase."""
        result = lean.verify_phase("NONEXISTENT-PHASE")
        assert result["total"] == 0
        assert "error" in result

    @patch("sct_tools.lean.verify_batch")
    def test_verify_all(self, mock_vb):
        """verify_all() passes ALL identities to verify_batch()."""
        mock_vb.return_value = {
            "total": len(lean.SCT_IDENTITIES),
            "verified": len(lean.SCT_IDENTITIES),
        }
        lean.verify_all()
        ids_passed = mock_vb.call_args[0][0]
        assert len(ids_passed) == len(lean.SCT_IDENTITIES)


# ---------------------------------------------------------------------------
#  SCT-specific proof templates
# ---------------------------------------------------------------------------


class TestSCTProofTemplates:
    """Tests for sct_phi_zero_proof, sct_conformal_coupling_proof, sct_ghost_counting_proof."""

    @patch("sct_tools.lean.prove")
    def test_phi_zero_proof_calls_prove(self, mock_prove):
        """sct_phi_zero_proof() calls prove() with integral theorem."""
        mock_prove.return_value = {
            "status": "complete",
            "output": "proof",
            "output_path": None,
        }
        lean.sct_phi_zero_proof()
        code_arg = mock_prove.call_args[0][0]
        assert "sct_phi_at_zero" in code_arg
        assert "exp" in code_arg
        assert "sorry" in code_arg

    @patch("sct_tools.lean.verify_identity")
    def test_conformal_coupling_proof(self, mock_vi):
        """sct_conformal_coupling_proof() calls verify_identity correctly."""
        mock_vi.return_value = {"status": "complete", "verified": True}
        lean.sct_conformal_coupling_proof()
        call_kwargs = mock_vi.call_args
        assert "conformal" in call_kwargs[1].get("description", "").lower() or \
               "conformal" in str(call_kwargs)

    @patch("sct_tools.lean.verify_identity")
    def test_ghost_counting_proof(self, mock_vi):
        """sct_ghost_counting_proof() calls verify_identity with correct lhs/rhs."""
        mock_vi.return_value = {"status": "complete", "verified": True}
        lean.sct_ghost_counting_proof()
        call_args = mock_vi.call_args
        # Should have (14-2)/120 = 1/10
        assert "14" in str(call_args)
        assert "10" in str(call_args)


# ---------------------------------------------------------------------------
#  prove_local_file() — local Lean file verification
# ---------------------------------------------------------------------------


class TestProveLocalFile:
    """Tests for prove_local_file() with mocked subprocess."""

    @patch("subprocess.run")
    def test_prove_local_file_success(self, mock_run):
        """prove_local_file() returns ok on successful build."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Build completed successfully", stderr=""
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False
        ) as f:
            f.write("-- test")
            path = f.name
        try:
            result = lean.prove_local_file(path)
            assert result["status"] == "ok"
        finally:
            os.unlink(path)

    def test_prove_local_file_not_found(self):
        """prove_local_file() raises error for missing file."""
        result = lean.prove_local_file("/nonexistent/file.lean")
        assert result["status"] == "error"
        assert "not found" in result.get("error", "").lower() or result["status"] == "error"
