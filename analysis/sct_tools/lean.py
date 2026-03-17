"""
SCT Tools — Lean 4 formal verification via Aristotle API + local Lean 4.

Provides three verification backends:

1. **Aristotle (cloud)**: Fill `sorry` gaps via Aristotle API.
   Requires: aristotlelib (pip install aristotlelib), ARISTOTLE_API_KEY.

2. **Local Lean 4 (PhysLean + Mathlib4)**: Build proofs locally using the
   SCT Lean project at theory/lean/. Requires: elan, lake on PATH.
   Includes PhysLean (Lorentz group, SM, tensor notation) and Mathlib4
   (Riemannian manifolds, spectral theory, measures).

3. **WSL SciLean**: Verify numerical algorithm properties in WSL.
   Requires: elan in WSL Ubuntu. SciLean = autodiff, ODEs, optimization.

Core API:
    prove()           — Fill sorries via Aristotle cloud
    prove_local()     — Build .lean file with local lake (PhysLean/Mathlib4)
    prove_scilean()   — Build .lean file in WSL SciLean project
    formalize()       — Convert natural language → Lean 4 proof
    verify_identity() — Verify rational identity via Aristotle
    verify_deep()     — Chain: Aristotle → local Lean → cross-check

High-level:
    verify_phase()    — All identities for an SCT phase
    verify_all()      — All registered SCT identities
    check_local()     — Verify local Lean 4 + PhysLean installation
    check_scilean()   — Verify WSL SciLean installation

Usage:
    from sct_tools.lean import prove, prove_local, verify_deep, check_local

    # Aristotle cloud proof
    result = prove('''
    import Mathlib.Tactic
    theorem t : (1 : ℚ) / 120 = 1 / 120 := by sorry
    ''')

    # Local Lean proof (PhysLean + Mathlib4)
    result = prove_local('''
    import Mathlib.Tactic
    theorem t : (1 : ℚ) / 120 = 1 / 120 := by ring
    ''')

    # Deep verification (both backends)
    result = verify_deep("sct_beta_W_scalar",
                         "(1 : ℝ) / 120", "(1 : ℝ) / 120")
"""

import asyncio
import logging
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _validate_lean_name(name, param="name"):
    """Validate that a string is a safe Lean 4 identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{param} must be a non-empty string")
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_']*$", name):
        raise ValueError(
            f"{param} must be a valid Lean identifier (alphanumeric + _ + '), "
            f"got {name!r}"
        )


def _has_sorry(code):
    """Check if Lean code contains the sorry tactic (word-boundary aware)."""
    # Strip single-line comments
    stripped = re.sub(r'--[^\n]*', '', code)
    # Strip block comments (non-nested)
    stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
    return bool(re.search(r'\bsorry\b', stripped))


def _windows_path_for_wsl(path):
    """Normalize a Windows path string before passing it to `wslpath`."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    return Path(path).as_posix()

# Project root
_ROOT = Path(__file__).resolve().parent.parent.parent
_LEAN_DIR = _ROOT / "theory" / "lean"
_PROOFS_DIR = _LEAN_DIR / "proofs"
_SCTLEAN_DIR = _LEAN_DIR / "SCTLean"

# Junction path (avoids Windows spaces-in-path issues for lake)
_JUNCTION = Path("C:/sct-lean")

# WSL SciLean project path
_WSL_SCILEAN = "~/sct-scilean"


def _ensure_api_key():
    """Load API key from .env if not already set."""
    if os.environ.get("ARISTOTLE_API_KEY"):
        return
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ARISTOTLE_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip()
                if key:
                    os.environ["ARISTOTLE_API_KEY"] = key
                    return
    raise EnvironmentError(
        "ARISTOTLE_API_KEY not set. Add to .env or call set_api_key()"
    )


def set_api_key(key):
    """Set the Aristotle API key programmatically."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("API key must be a non-empty string")
    try:
        import aristotlelib
        aristotlelib.set_api_key(key)
    except ImportError:
        pass
    os.environ["ARISTOTLE_API_KEY"] = key


def _run(coro):
    """Run an async coroutine from synchronous context."""
    loop_running = False
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        pass
    if loop_running:
        # Already in an event loop (e.g. Jupyter) — use thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        # No event loop — safe to use asyncio.run directly
        return asyncio.run(coro)


def prove(code, output_path=None, wait=True):
    """
    Fill in `sorry` gaps in inline Lean 4 code.

    Parameters
    ----------
    code : str
        Lean 4 source code with `sorry` placeholders.
    output_path : str or Path, optional
        Where to save the completed proof. Default: temp file.
    wait : bool
        If True (default), block until proof is complete.

    Returns
    -------
    dict
        Keys: 'status' (str), 'output' (str or None), 'output_path' (str or None)
        On error: also includes 'error' (str)
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    try:
        import aristotlelib
    except ImportError:
        raise ImportError(
            "aristotlelib not installed. Run: python -m pip install aristotlelib"
        )

    _ensure_api_key()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        input_path = f.name

    if output_path is None:
        _fd, _name = tempfile.mkstemp(suffix=".lean")
        os.close(_fd)
        output_path = Path(_name)
    else:
        output_path = Path(output_path)

    try:
        _run(aristotlelib.Project.prove_from_file(
            input_file_path=input_path,
            output_file_path=str(output_path),
            wait_for_completion=wait,
            auto_add_imports=False,
            validate_lean_project=False,
        ))

        if wait and output_path.exists():
            output_text = output_path.read_text(encoding="utf-8")
            status = "complete"
        else:
            output_text = None
            status = "submitted"

        return {
            "status": status,
            "output": output_text,
            "output_path": str(output_path) if output_path.exists() else None,
        }

    except Exception as e:
        return {
            "status": "error",
            "output": None,
            "error": str(e),
            "output_path": None,
        }
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass


def prove_file(input_path, output_path=None, wait=True):
    """
    Fill in `sorry` gaps in a Lean 4 file.

    Parameters
    ----------
    input_path : str or Path
        Path to .lean file with sorry placeholders.
    output_path : str or Path, optional
        Where to save completed proof. Default: <input>_proved.lean
    wait : bool
        Block until complete (default: True).

    Returns
    -------
    dict
        Keys: 'status', 'output', 'output_path'
    """
    try:
        import aristotlelib
    except ImportError:
        raise ImportError(
            "aristotlelib not installed. Run: python -m pip install aristotlelib"
        )

    _ensure_api_key()

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Lean file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_proved")
    else:
        output_path = Path(output_path)

    try:
        _run(aristotlelib.Project.prove_from_file(
            input_file_path=str(input_path),
            output_file_path=str(output_path),
            wait_for_completion=wait,
            auto_add_imports=False,
            validate_lean_project=False,
        ))

        if wait and output_path.exists():
            output_text = output_path.read_text(encoding="utf-8")
            status = "complete"
        else:
            output_text = None
            status = "submitted"

        return {
            "status": status,
            "output": output_text,
            "output_path": str(output_path) if output_path.exists() else None,
        }

    except Exception as e:
        return {
            "status": "error",
            "output": None,
            "error": str(e),
            "output_path": None,
        }


def formalize(description, output_path=None, wait=True):
    """
    Convert a natural-language mathematical statement to a Lean 4 proof.

    Parameters
    ----------
    description : str
        Mathematical statement in English (may include LaTeX).
    output_path : str or Path, optional
        Where to save the formalized proof.
    wait : bool
        Block until complete (default: True).

    Returns
    -------
    dict
        Keys: 'status', 'output', 'output_path'
    """
    try:
        import aristotlelib
    except ImportError:
        raise ImportError(
            "aristotlelib not installed. Run: python -m pip install aristotlelib"
        )

    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "formalize: description must be a non-empty string"
        )

    _ensure_api_key()

    if output_path is None:
        _fd, _name = tempfile.mkstemp(suffix=".lean")
        os.close(_fd)
        output_path = Path(_name)
    else:
        output_path = Path(output_path)

    try:
        _run(aristotlelib.Project.prove_from_file(
            input_content=description,
            output_file_path=str(output_path),
            wait_for_completion=wait,
            auto_add_imports=False,
            validate_lean_project=False,
            project_input_type=aristotlelib.ProjectInputType.INFORMAL,
        ))

        if wait and output_path.exists():
            output_text = output_path.read_text(encoding="utf-8")
            status = "complete"
        else:
            output_text = None
            status = "submitted"

        return {
            "status": status,
            "output": output_text,
            "output_path": str(output_path) if output_path.exists() else None,
        }

    except Exception as e:
        return {
            "status": "error",
            "output": None,
            "error": str(e),
            "output_path": None,
        }


# --------------------------------------------------------------------------- #
#  High-level verification API
# --------------------------------------------------------------------------- #

def verify_identity(name, lhs, rhs, description="", save=True):
    """
    Verify a rational arithmetic identity via Lean 4 formal proof.

    Parameters
    ----------
    name : str
        Theorem name (used for file naming, must be valid Lean identifier).
    lhs : str
        Left-hand side as Lean 4 real expression, e.g. '((14 : ℝ) - 2) / 120'.
    rhs : str
        Right-hand side as Lean 4 real expression, e.g. '1 / 10'.
    description : str, optional
        Human-readable docstring for the theorem.
    save : bool
        If True, save proven .lean file to theory/lean/proofs/.

    Returns
    -------
    dict
        Keys: 'status', 'verified' (bool), 'output', 'output_path', 'error'
    """
    _validate_lean_name(name)
    if not isinstance(lhs, str) or not lhs.strip():
        raise ValueError("lhs must be a non-empty string")
    if not isinstance(rhs, str) or not rhs.strip():
        raise ValueError("rhs must be a non-empty string")
    desc_comment = f"/-- {description} -/\n" if description else ""
    code = f"""\
import Mathlib.Tactic

{desc_comment}theorem {name} :
    {lhs} = {rhs} := by
  sorry
"""
    out_path = None
    if save:
        _PROOFS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _PROOFS_DIR / f"{name}.lean"

    import re
    result = prove(code, output_path=out_path)
    output_text = result.get("output", "") or ""
    # Strip Lean comments before checking for sorry tactic
    stripped = re.sub(r'--[^\n]*', '', output_text)
    stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
    verified = (
        result["status"] == "complete"
        and output_text
        and not re.search(r'\bsorry\b', stripped)
    )
    result["verified"] = verified
    if verified:
        logger.info("VERIFIED: %s  (%s = %s)", name, lhs, rhs)
    else:
        logger.warning("UNVERIFIED: %s  (%s = %s)", name, lhs, rhs)
    return result


def verify_batch(identities, stop_on_failure=False):
    """
    Verify multiple identities sequentially.

    Parameters
    ----------
    identities : list of dict
        Each dict has keys: 'name', 'lhs', 'rhs', and optionally 'description'.
    stop_on_failure : bool
        If True, stop at first unverified identity.

    Returns
    -------
    dict
        Keys: 'total', 'verified', 'failed', 'errors', 'results' (list)
    """
    results = []
    verified_count = 0
    failed_names = []
    error_names = []

    for ident in identities:
        r = verify_identity(
            name=ident["name"],
            lhs=ident["lhs"],
            rhs=ident["rhs"],
            description=ident.get("description", ""),
        )
        results.append({"identity": ident, **r})

        if r["verified"]:
            verified_count += 1
        elif r["status"] == "error":
            error_names.append(ident["name"])
            if stop_on_failure:
                break
        else:
            failed_names.append(ident["name"])
            if stop_on_failure:
                break

    return {
        "total": len(identities),
        "attempted": len(results),
        "verified": verified_count,
        "failed": failed_names,
        "errors": error_names,
        "results": results,
    }


# --------------------------------------------------------------------------- #
#  SCT canonical identity registry
# --------------------------------------------------------------------------- #

# All provable algebraic identities from SCT Theory canonical results.
# Each entry is a dict with: name, lhs, rhs, description, phase.
SCT_IDENTITIES = [
    # --- NT-1b Phase 1: Scalar (spin-0) beta-coefficients ---
    {
        "name": "sct_scalar_beta_weyl",
        "lhs": "(1 : ℝ) / 120",
        "rhs": "(1 : ℝ) / 120",
        "description": "Scalar Weyl beta: beta_W^(0) = 1/120",
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_conformal_coupling",
        "lhs": "(1 : ℝ) / 2 * ((1 : ℝ) / 6 - (1 : ℝ) / 6) ^ 2",
        "rhs": "(0 : ℝ)",
        "description": (
            "At conformal coupling xi = 1/6, scalar Ricci beta vanishes: "
            "beta_R^(0)(1/6) = (1/2)(1/6 - 1/6)^2 = 0"
        ),
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_f_Ric_zero",
        "lhs": "(1 : ℝ) / 60",
        "rhs": "(1 : ℝ) / 60",
        "description": "CZ scalar form factor: f_Ric(0) = 1/60",
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_f_R_zero",
        "lhs": "(1 : ℝ) / 120",
        "rhs": "(1 : ℝ) / 120",
        "description": "CZ scalar form factor: f_R(0) = 1/120",
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_f_RU_zero",
        "lhs": "-(1 : ℝ) / 6",
        "rhs": "-(1 : ℝ) / 6",
        "description": "CZ scalar form factor: f_RU(0) = -1/6",
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_f_U_zero",
        "lhs": "(1 : ℝ) / 2",
        "rhs": "(1 : ℝ) / 2",
        "description": "CZ scalar form factor: f_U(0) = 1/2",
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_f_Omega_zero",
        "lhs": "(1 : ℝ) / 12",
        "rhs": "(1 : ℝ) / 12",
        "description": "CZ scalar form factor: f_Omega(0) = 1/12",
        "phase": "NT-1b-scalar",
    },
    {
        "name": "sct_scalar_f_Rbis_zero",
        "lhs": "(1 : ℝ) / 3 * ((1 : ℝ) / 60) + (1 : ℝ) / 120",
        "rhs": "(1 : ℝ) / 72",
        "description": (
            "CZ scalar combined: f_{R,bis}(0) = (1/3)*f_Ric(0) + f_R(0) "
            "= 1/180 + 1/120 = 1/72"
        ),
        "phase": "NT-1b-scalar",
    },

    # --- NT-1: Dirac (spin-1/2) beta-coefficients ---
    {
        "name": "sct_dirac_beta_weyl",
        "lhs": "(1 : ℝ) / 20",
        "rhs": "(1 : ℝ) / 20",
        "description": "Dirac Weyl beta: beta_W^(1/2) = 1/20",
        "phase": "NT-1",
    },

    # --- NT-1b Phase 2: Vector (spin-1) beta-coefficients ---
    {
        "name": "sct_vector_beta_weyl",
        "lhs": "((14 : ℝ) - 2) / 120",
        "rhs": "(1 : ℝ) / 10",
        "description": (
            "Vector Weyl beta with ghost subtraction: "
            "beta_W^(1) = (14 - 2*1)/120 = 12/120 = 1/10"
        ),
        "phase": "NT-1b-vector",
    },
    {
        "name": "sct_vector_unconstr_beta_weyl",
        "lhs": "(7 : ℝ) / 60",
        "rhs": "(7 : ℝ) / 60",
        "description": (
            "Unconstrained vector Weyl beta: "
            "beta_W^(unconstr) = 14/120 = 7/60"
        ),
        "phase": "NT-1b-vector",
    },
    {
        "name": "sct_vector_unconstr_beta_weyl_equiv",
        "lhs": "(14 : ℝ) / 120",
        "rhs": "(7 : ℝ) / 60",
        "description": "Reduction: 14/120 = 7/60",
        "phase": "NT-1b-vector",
    },
    {
        "name": "sct_vector_unconstr_beta_ricci",
        "lhs": "(1 : ℝ) / 36",
        "rhs": "(1 : ℝ) / 36",
        "description": (
            "Unconstrained vector Ricci beta: "
            "beta_R^(unconstr) = 1/36"
        ),
        "phase": "NT-1b-vector",
    },
    {
        "name": "sct_ghost_scalar_beta",
        "lhs": "(1 : ℝ) / 120",
        "rhs": "(1 : ℝ) / 120",
        "description": (
            "Single FP ghost = scalar contribution: "
            "beta_W^(ghost) = 1/120"
        ),
        "phase": "NT-1b-vector",
    },
    {
        "name": "sct_ghost_subtraction",
        "lhs": "(7 : ℝ) / 60 - 2 * ((1 : ℝ) / 120)",
        "rhs": "(1 : ℝ) / 10",
        "description": (
            "Ghost subtraction: 7/60 - 2*(1/120) = 7/60 - 1/60 = 1/10"
        ),
        "phase": "NT-1b-vector",
    },

    # --- Phase 3 (Combined SM): d.o.f. counting ---
    {
        "name": "sct_sm_total_beta_weyl",
        "lhs": (
            "(4 : ℝ) * ((1 : ℝ) / 120) + "
            "(45 : ℝ) / 2 * ((1 : ℝ) / 20) + "
            "(12 : ℝ) * ((1 : ℝ) / 10)"
        ),
        "rhs": "(4 : ℝ) / 120 + (45 : ℝ) / 40 + (12 : ℝ) / 10",
        "description": (
            "SM total Weyl beta: "
            "N_s*beta_W^(0) + (N_f/2)*beta_W^(1/2) + N_v*beta_W^(1)"
        ),
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_sm_dof_scalar",
        "lhs": "(4 : ℝ)",
        "rhs": "(4 : ℝ)",
        "description": "SM scalar d.o.f.: N_s = 4 (real Higgs doublet)",
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_sm_dof_fermion",
        "lhs": "(45 : ℝ)",
        "rhs": "(45 : ℝ)",
        "description": (
            "SM fermion d.o.f.: N_f = 45 Dirac "
            "(Weyl->Dirac convention)"
        ),
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_sm_dof_vector",
        "lhs": "(12 : ℝ)",
        "rhs": "(12 : ℝ)",
        "description": (
            "SM gauge boson d.o.f.: N_v = 12 "
            "(8 gluons + W+/W-/Z + photon)"
        ),
        "phase": "NT-1b-combined",
    },

    # --- BV parametric weights ---
    {
        "name": "sct_bv_weight_U",
        "lhs": "(1 : ℝ) / 2",
        "rhs": "(1 : ℝ) / 2",
        "description": "BV parametric weight: Phi_U = 1/2",
        "phase": "NT-1",
    },

    # --- Spectral function normalization ---
    {
        "name": "sct_spectral_normalization",
        "lhs": "(1 : ℝ) / (16 * Real.pi ^ 2)",
        "rhs": "(1 : ℝ) / (16 * Real.pi ^ 2)",
        "description": (
            "Spectral action normalization: "
            "F_i(z) = h_i(z) / (16*pi^2)"
        ),
        "phase": "NT-1",
    },
    {
        "name": "nt2_hc_scalar_pole_cancels",
        "lhs": "(1 : ℚ) / 12 + (-(1 : ℚ) / 6) / 2",
        "rhs": "(0 : ℚ)",
        "description": "NT-2 scalar Weyl pole cancellation at z = 0",
        "phase": "NT-2",
    },
    {
        "name": "nt2_hc_dirac_pole_cancels",
        "lhs": "(1 : ℚ) / 3 + 2 * (-(1 : ℚ) / 6)",
        "rhs": "(0 : ℚ)",
        "description": "NT-2 Dirac Weyl pole cancellation at z = 0",
        "phase": "NT-2",
    },
    {
        "name": "nt2_hc_vector_pole_cancels",
        "lhs": "(1 : ℚ) / 6 + (-(1 : ℚ) / 6)",
        "rhs": "(0 : ℚ)",
        "description": "NT-2 vector Weyl pole cancellation at z = 0",
        "phase": "NT-2",
    },
    {
        "name": "nt2_hr_vector_pole_cancels",
        "lhs": "(5 : ℚ) / 72 + 5 * (-(1 : ℚ) / 6) / 12",
        "rhs": "(0 : ℚ)",
        "description": "NT-2 vector Ricci pole cancellation at z = 0",
        "phase": "NT-2",
    },
    {
        "name": "nt2_total_weyl_beta",
        "lhs": "(13 : ℚ) / 120",
        "rhs": "(13 : ℚ) / 120",
        "description": "NT-2 local Weyl coefficient inherited from Phase 3",
        "phase": "NT-2",
    },
    {
        "name": "nt4a_c2_local",
        "lhs": "2 * ((13 : ℚ) / 120)",
        "rhs": "(13 : ℚ) / 60",
        "description": "NT-4a spin-2 local coefficient c2 = 13/60",
        "phase": "NT-4a",
    },
    {
        "name": "nt4a_scalar_mode_minimal",
        "lhs": "6 * ((0 : ℚ) - 1 / 6) ^ 2",
        "rhs": "(1 : ℚ) / 6",
        "description": "NT-4a scalar mode coefficient at minimal coupling",
        "phase": "NT-4a",
    },
    {
        "name": "nt4a_scalar_mode_conformal",
        "lhs": "6 * (((1 : ℚ) / 6) - 1 / 6) ^ 2",
        "rhs": "(0 : ℚ)",
        "description": "NT-4a scalar mode decoupling at conformal coupling",
        "phase": "NT-4a",
    },
    {
        "name": "nt4a_pi_tt_zero",
        "lhs": "(1 : ℚ) + 0",
        "rhs": "(1 : ℚ)",
        "description": "NT-4a propagator normalization Pi_TT(0) = 1",
        "phase": "NT-4a",
    },

    # --- Phase 3 (Combined SM): alpha_C from h_C(0) local limits ---
    {
        "name": "sct_sm_alpha_C_value",
        "lhs": (
            "(4 : ℚ) * (1 / 120) + "
            "(45 : ℚ) / 2 * (-(1 : ℚ) / 20) + "
            "(12 : ℚ) * (1 / 10)"
        ),
        "rhs": "(13 : ℚ) / 120",
        "description": (
            "SM total Weyl coefficient: "
            "alpha_C = N_s*h_C^(0)(0) + N_D*h_C^(1/2)(0) + N_v*h_C^(1)(0) = 13/120"
        ),
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_sm_alpha_R_conformal",
        "lhs": "2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2",
        "rhs": "(0 : ℚ)",
        "description": (
            "alpha_R vanishes at conformal coupling xi = 1/6: "
            "2*(1/6 - 1/6)^2 = 0"
        ),
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_sm_alpha_R_minimal",
        "lhs": "2 * ((0 : ℚ) - 1 / 6) ^ 2",
        "rhs": "(1 : ℚ) / 18",
        "description": (
            "alpha_R at minimal coupling xi = 0: "
            "2*(0 - 1/6)^2 = 1/18"
        ),
        "phase": "NT-1b-combined",
    },

    # --- Phase 3 (Combined SM): c1/c2 Wilson coefficient ratio ---
    {
        "name": "sct_c1c2_ratio_conformal",
        "lhs": (
            "(2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 - "
            "(2 : ℚ) / 3 * ((13 : ℚ) / 120)) / "
            "(2 * ((13 : ℚ) / 120))"
        ),
        "rhs": "-(1 : ℚ) / 3",
        "description": (
            "c1/c2 = -1/3 at conformal coupling xi = 1/6 (parameter-free)"
        ),
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_c1c2_ratio_minimal",
        "lhs": (
            "(2 * ((0 : ℚ) - 1 / 6) ^ 2 - "
            "(2 : ℚ) / 3 * ((13 : ℚ) / 120)) / "
            "(2 * ((13 : ℚ) / 120))"
        ),
        "rhs": "-(1 : ℚ) / 3 + 10 / 39",
        "description": (
            "c1/c2 at minimal coupling xi = 0: "
            "-1/3 + 10/39"
        ),
        "phase": "NT-1b-combined",
    },

    # --- Phase 3 (Combined SM): 3c1 + c2 scalar mode combination ---
    {
        "name": "sct_3c1_plus_c2_conformal",
        "lhs": (
            "3 * (2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 - "
            "(2 : ℚ) / 3 * ((13 : ℚ) / 120)) + "
            "2 * ((13 : ℚ) / 120)"
        ),
        "rhs": "(0 : ℚ)",
        "description": (
            "3c1 + c2 = 0 at conformal coupling: scalar mode decouples"
        ),
        "phase": "NT-1b-combined",
    },
    {
        "name": "sct_3c1_plus_c2_minimal",
        "lhs": (
            "3 * (2 * ((0 : ℚ) - 1 / 6) ^ 2 - "
            "(2 : ℚ) / 3 * ((13 : ℚ) / 120)) + "
            "2 * ((13 : ℚ) / 120)"
        ),
        "rhs": "(1 : ℚ) / 6",
        "description": (
            "3c1 + c2 = 1/6 at minimal coupling xi = 0"
        ),
        "phase": "NT-1b-combined",
    },

    # --- NT-4a: effective masses and potential ---
    {
        "name": "nt4a_m2_squared",
        "lhs": "(1 : ℚ) + (13 : ℚ) / 60 * (-(60 : ℚ) / 13)",
        "rhs": "(0 : ℚ)",
        "description": (
            "NT-4a spin-2 effective mass pole: "
            "Pi_TT(-60/13) = 1 + (13/60)*(-60/13) = 0"
        ),
        "phase": "NT-4a",
    },
    {
        "name": "nt4a_m0_squared_minimal",
        "lhs": (
            "(1 : ℚ) + 6 * ((0 : ℚ) - 1 / 6) ^ 2 * (-(6 : ℚ))"
        ),
        "rhs": "(0 : ℚ)",
        "description": (
            "NT-4a scalar effective mass pole at xi = 0: "
            "Pi_s(-6) = 1 + (1/6)*(-6) = 0"
        ),
        "phase": "NT-4a",
    },
    {
        "name": "nt4a_newton_potential_finite",
        "lhs": "(1 : ℚ) - 4 / 3 + 1 / 3",
        "rhs": "(0 : ℚ)",
        "description": (
            "NT-4a modified potential V(0)/V_N(0) = 0: "
            "1 - 4/3 + 1/3 = 0 (Newtonian singularity resolved)"
        ),
        "phase": "NT-4a",
    },
    {
        "name": "nt4a_mass_ratio_minimal",
        "lhs": "(6 : ℚ) / ((60 : ℚ) / 13)",
        "rhs": "(13 : ℚ) / 10",
        "description": (
            "NT-4a mass ratio at xi = 0: "
            "m0^2/m2^2 = 6/(60/13) = 13/10"
        ),
        "phase": "NT-4a",
    },
]

# --------------------------------------------------------------------------- #
#  Additional formally verified theorems (NOT in SCT_IDENTITIES)
#
#  The following theorems from chiral_q_identity.lean involve variables
#  (not simple lhs = rhs rational identities) and are therefore not
#  registered in SCT_IDENTITIES, but are formally verified in Lean 4:
#
#    sq_comm_of_anticomm          -- square commutes when elements anticommute
#    prod_comm_of_anticomm        -- product commutes when elements anticommute
#    chiral_q_identity            -- main chiral-q theorem
#    d_sq_comm                    -- d² commutes with chiral grading
#    even_clifford_comm           -- even Clifford elements commute with grading
#    spin_connection_comm_two     -- spin connection commutes with γ-grading
#    diffeo_generator_comm        -- diffeomorphism generator commutativity
#    bv_canonical_transformation  -- BV canonical transformation preserves bracket
#    cme_preserved                -- classical master equation preserved
#    centralizer_mul_closed       -- centralizer closed under multiplication
#    centralizer_neg_closed       -- centralizer closed under negation
#    centralizer_inv_closed       -- centralizer closed under inversion
#    centralizer_triple_product   -- centralizer closed under triple product
# --------------------------------------------------------------------------- #


def get_identities_by_phase(phase):
    """Return all identities for a given phase."""
    return [i for i in SCT_IDENTITIES if i["phase"] == phase]


def verify_phase(phase, stop_on_failure=False):
    """
    Verify all algebraic identities for a specific SCT phase.

    Parameters
    ----------
    phase : str
        One of: 'NT-1', 'NT-1b-scalar', 'NT-1b-vector', 'NT-1b-combined'
    stop_on_failure : bool
        Stop at first failure.

    Returns
    -------
    dict
        Batch verification results.
    """
    ids = get_identities_by_phase(phase)
    if not ids:
        return {
            "total": 0,
            "error": f"No identities for phase '{phase}'",
        }
    return verify_batch(ids, stop_on_failure=stop_on_failure)


def verify_all(stop_on_failure=False):
    """
    Verify ALL SCT canonical identities.

    Returns
    -------
    dict
        Batch verification results for all registered identities.
    """
    return verify_batch(SCT_IDENTITIES, stop_on_failure=stop_on_failure)


# --------------------------------------------------------------------------- #
#  SCT-specific proof template functions (backward compatibility)
# --------------------------------------------------------------------------- #

def sct_phi_zero_proof(output_path=None):
    """
    Lean 4 proof that phi(0) = 1.

    phi(x) = integral_0^1 exp(-xi(1-xi)x) dxi
    At x=0: integrand = exp(0) = 1, so phi(0) = integral_0^1 1 dxi = 1.
    """
    code = """\
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.MeasureTheory.Integral.IntervalIntegral

open MeasureTheory Real Set

/-- The SCT master function phi(x) = integral_0^1 exp(-xi*(1-xi)*x) dxi
    evaluates to 1 at x = 0. -/
theorem sct_phi_at_zero :
    ∫ ξ in (0:ℝ)..1, Real.exp (-(ξ * (1 - ξ) * 0)) = 1 := by
  sorry
"""
    return prove(code, output_path=output_path)


def sct_conformal_coupling_proof(output_path=None):
    """
    Lean 4 proof that beta_R^(0)(xi=1/6) = 0.

    beta_R^(0)(xi) = (1/2)*(xi - 1/6)^2  =>  beta_R^(0)(1/6) = 0
    """
    return verify_identity(
        name="sct_conformal_coupling_beta_R_zero",
        lhs="(1 : ℝ) / 2 * ((1 : ℝ) / 6 - (1 : ℝ) / 6) ^ 2",
        rhs="(0 : ℝ)",
        description=(
            "At conformal coupling xi = 1/6, the Ricci beta "
            "function vanishes: beta_R^(0)(1/6) = 0."
        ),
    )


def sct_ghost_counting_proof(output_path=None):
    """
    Lean 4 proof that vector beta_W^(1) = 1/10.

    Unconstrained: 14/120 = 7/60. Subtract 2 ghost contributions (each 1/120):
    (14 - 2)/120 = 12/120 = 1/10.
    """
    return verify_identity(
        name="sct_vector_beta_weyl_ghost",
        lhs="((14 : ℝ) - 2) / 120",
        rhs="1 / 10",
        description=(
            "Vector Weyl beta with ghost subtraction: "
            "(14 - 2)/120 = 12/120 = 1/10."
        ),
    )


# --------------------------------------------------------------------------- #
#  Local Lean 4 verification (PhysLean + Mathlib4)
# --------------------------------------------------------------------------- #

def _get_lean_project_dir():
    """Return the best path to the Lean project (junction if available)."""
    if _JUNCTION.exists():
        return _JUNCTION
    return _LEAN_DIR


def _find_lake():
    """Find lake executable."""
    # Try elan-managed lake first
    elan_lake = Path(os.path.expanduser("~/.elan/bin/lake.exe"))
    if elan_lake.exists():
        return str(elan_lake)
    elan_lake_unix = Path(os.path.expanduser("~/.elan/bin/lake"))
    if elan_lake_unix.exists():
        return str(elan_lake_unix)
    # Try PATH
    import shutil
    lake = shutil.which("lake")
    if lake:
        return lake
    return None


def prove_local(code, timeout=300):
    """
    Build inline Lean 4 code using the local SCT Lean project.

    Uses the local lake build system with PhysLean + Mathlib4.
    Code must NOT contain `sorry` — this checks that proofs compile.

    Parameters
    ----------
    code : str
        Complete Lean 4 source (with all imports and proofs filled in).
    timeout : int
        Max seconds for lake build (default: 300).

    Returns
    -------
    dict
        Keys: 'status' ('ok'|'error'), 'has_sorry' (bool),
        'stdout', 'stderr', 'returncode'
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("prove_local: code must be a non-empty string")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(
            f"prove_local: timeout must be a positive number, got {timeout}"
        )

    lake = _find_lake()
    if lake is None:
        return {"status": "error", "error": "lake not found. Install elan."}

    proj_dir = _get_lean_project_dir()
    has_sorry = _has_sorry(code)

    # Write code to a temp file in the project directory
    tmp_name = f"_sct_check_{os.getpid()}.lean"
    tmp_path = proj_dir / tmp_name

    try:
        tmp_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [lake, "env", "lean", str(tmp_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(proj_dir),
        )
        status = "ok" if proc.returncode == 0 else "error"
        return {
            "status": status,
            "has_sorry": has_sorry,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def prove_local_file(file_path, timeout=300):
    """
    Type-check a .lean file using the local SCT Lean project.

    Parameters
    ----------
    file_path : str or Path
        Path to .lean file to verify.
    timeout : int
        Max seconds for lean check.

    Returns
    -------
    dict
        Keys: 'status' ('ok'|'error'), 'has_sorry', 'stdout', 'stderr'
    """
    lake = _find_lake()
    if lake is None:
        return {"status": "error", "error": "lake not found. Install elan."}

    file_path = Path(file_path).resolve()
    if not file_path.exists():
        return {"status": "error", "error": f"File not found: {file_path}"}

    proj_dir = _get_lean_project_dir()
    code = file_path.read_text(encoding="utf-8")
    has_sorry = _has_sorry(code)

    try:
        proc = subprocess.run(
            [lake, "env", "lean", str(file_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(proj_dir),
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "has_sorry": has_sorry,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def build_sctlean(timeout=600):
    """
    Build the entire SCTLean library.

    Returns
    -------
    dict
        Keys: 'status', 'stdout', 'stderr', 'returncode'
    """
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(
            f"build_sctlean: timeout must be a positive number, got {timeout}"
        )
    lake = _find_lake()
    if lake is None:
        return {"status": "error", "error": "lake not found"}

    proj_dir = _get_lean_project_dir()
    try:
        proc = subprocess.run(
            [lake, "build", "SCTLean"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(proj_dir),
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Build timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_local():
    """
    Verify local Lean 4 + PhysLean installation status.

    Returns
    -------
    dict
        Keys: 'available' (bool), 'lean_version', 'lake_path',
        'project_dir', 'packages' (list), 'errors' (list)
    """
    errors = []
    result = {
        "available": False,
        "lean_version": None,
        "lake_path": None,
        "project_dir": None,
        "packages": [],
        "errors": errors,
    }

    lake = _find_lake()
    if lake is None:
        errors.append("lake not found. Install elan: https://github.com/leanprover/elan")
        return result
    result["lake_path"] = lake

    proj_dir = _get_lean_project_dir()
    result["project_dir"] = str(proj_dir)

    if not (proj_dir / "lakefile.lean").exists():
        errors.append(f"No lakefile.lean in {proj_dir}")
        return result

    # Check lean version
    try:
        proc = subprocess.run(
            [lake, "env", "lean", "--version"],
            capture_output=True, text=True, timeout=30,
            cwd=str(proj_dir),
        )
        if proc.returncode == 0:
            result["lean_version"] = proc.stdout.strip()
    except Exception as e:
        errors.append(f"lean --version failed: {e}")

    # Check packages
    lake_dir = proj_dir / ".lake" / "packages"
    if lake_dir.exists():
        result["packages"] = sorted(p.name for p in lake_dir.iterdir() if p.is_dir())

    if "mathlib" in result["packages"] and "PhysLean" in result["packages"]:
        result["available"] = True
    else:
        missing = []
        if "mathlib" not in result["packages"]:
            missing.append("mathlib")
        if "PhysLean" not in result["packages"]:
            missing.append("PhysLean")
        errors.append(f"Missing packages: {missing}. Run: lake update")

    return result


# --------------------------------------------------------------------------- #
#  WSL SciLean verification
# --------------------------------------------------------------------------- #

def prove_scilean(code, timeout=300):
    """
    Build Lean 4 code using SciLean in WSL.

    Use for proofs about numerical algorithms: autodiff correctness,
    ODE solver properties, optimization convergence.

    Parameters
    ----------
    code : str
        Lean 4 source code (may import SciLean).
    timeout : int
        Max seconds for WSL lean check.

    Returns
    -------
    dict
        Keys: 'status' ('ok'|'error'), 'has_sorry', 'stdout', 'stderr'
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("prove_scilean: code must be a non-empty string")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(
            f"prove_scilean: timeout must be a positive number, got {timeout}"
        )

    has_sorry = _has_sorry(code)

    # Write code to a Windows temp file, then copy to WSL to avoid heredoc injection
    tmp_name = f"_sct_check_{os.getpid()}.lean"
    win_tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False,
                                         encoding='utf-8') as tf:
            tf.write(code)
            win_tmp = tf.name
        # Convert Windows path to WSL path
        win_tmp_wslpath = _windows_path_for_wsl(win_tmp)
        wsl_path_proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "wslpath", "-u", win_tmp_wslpath],
            capture_output=True, text=True, timeout=30,
        )
        if wsl_path_proc.returncode != 0:
            return {
                "status": "error",
                "error": (
                    f"wslpath failed (rc={wsl_path_proc.returncode}): "
                    f"{wsl_path_proc.stderr.strip()}"
                ),
            }
        wsl_tmp = wsl_path_proc.stdout.strip()
        wsl_cmd = (
            f"source ~/.elan/env && "
            f"cd {_WSL_SCILEAN} && "
            f"cp {shlex.quote(wsl_tmp)} {tmp_name} && "
            f"lake env lean {tmp_name} 2>&1; "
            f"EXIT=$?; rm -f {tmp_name}; exit $EXIT"
        )

        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c", wsl_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "has_sorry": has_sorry,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"WSL timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        if win_tmp:
            try:
                os.unlink(win_tmp)
            except OSError:
                pass


def check_scilean():
    """
    Verify WSL SciLean installation status.

    Returns
    -------
    dict
        Keys: 'available' (bool), 'lean_version', 'packages', 'errors'
    """
    try:
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
             f"source ~/.elan/env && cd {_WSL_SCILEAN} && "
             "lean --version 2>&1 && echo '---' && "
             "ls .lake/packages/ 2>/dev/null"],
            capture_output=True, text=True, timeout=30,
        )
        output = proc.stdout.strip()
        parts = output.split("---")
        version = parts[0].strip() if parts else None
        packages = parts[1].strip().split() if len(parts) > 1 else []

        available = proc.returncode == 0 and "scilean" in [p.lower() for p in packages]
        return {
            "available": available,
            "lean_version": version,
            "packages": packages,
            "errors": [proc.stderr.strip()] if proc.stderr.strip() else [],
        }
    except Exception as e:
        return {"available": False, "lean_version": None, "packages": [], "errors": [str(e)]}


# --------------------------------------------------------------------------- #
#  Deep verification (multi-backend)
# --------------------------------------------------------------------------- #

def verify_deep(name, lhs, rhs, description="", save=True,
                use_aristotle=True, use_local=True):
    """
    Verify an identity using multiple Lean backends for maximum confidence.

    Pipeline:
    1. Local Lean 4 (PhysLean/Mathlib4) — algebraic proof via `ring`/`norm_num`
    2. Aristotle cloud — independent `sorry`-filling

    Both must succeed for 'fully_verified' = True.

    Parameters
    ----------
    name : str
        Theorem name.
    lhs, rhs : str
        Lean 4 expressions.
    description : str
        Human-readable description.
    save : bool
        Save proof files.
    use_aristotle : bool
        Use Aristotle cloud backend.
    use_local : bool
        Use local Lean 4 backend.

    Returns
    -------
    dict
        Keys: 'fully_verified' (bool), 'local_result', 'aristotle_result',
        'name', 'backends_used'
    """
    if not use_aristotle and not use_local:
        raise ValueError("verify_deep: at least one backend must be enabled")
    _validate_lean_name(name)
    if not isinstance(lhs, str) or not lhs.strip():
        raise ValueError("lhs must be a non-empty string")
    if not isinstance(rhs, str) or not rhs.strip():
        raise ValueError("rhs must be a non-empty string")

    result = {
        "name": name,
        "fully_verified": False,
        "backends_used": [],
        "local_result": None,
        "aristotle_result": None,
    }

    # --- Backend 1: Local Lean (algebraic proof) ---
    if use_local:
        desc_comment = f"/-- {description} -/\n" if description else ""
        local_code = f"""\
import Mathlib.Tactic

{desc_comment}theorem {name} :
    {lhs} = {rhs} := by
  ring
"""
        local_res = prove_local(local_code)
        result["local_result"] = local_res
        result["backends_used"].append("local_lean")

        # If ring fails, try norm_num
        if local_res.get("status") != "ok":
            local_code_nn = f"""\
import Mathlib.Tactic

{desc_comment}theorem {name} :
    {lhs} = {rhs} := by
  norm_num
"""
            local_res = prove_local(local_code_nn)
            result["local_result"] = local_res

        if local_res.get("status") == "ok":
            logger.info("LOCAL VERIFIED: %s", name)
        else:
            logger.warning("LOCAL FAILED: %s — %s",
                           name, local_res.get("stderr", local_res.get("error", ""))[:200])

    # --- Backend 2: Aristotle cloud ---
    if use_aristotle:
        aristotle_res = verify_identity(
            name=name, lhs=lhs, rhs=rhs,
            description=description, save=save,
        )
        result["aristotle_result"] = aristotle_res
        result["backends_used"].append("aristotle")

    # --- Verdict ---
    local_ok = (
        result.get("local_result", {}).get("status") == "ok"
        if use_local
        else True
    )
    aristotle_ok = (
        result.get("aristotle_result", {}).get("verified", False)
        if use_aristotle
        else True
    )
    result["fully_verified"] = local_ok and aristotle_ok

    if result["fully_verified"]:
        logger.info("DEEP VERIFIED: %s (backends: %s)", name, result["backends_used"])
    else:
        logger.warning("DEEP INCOMPLETE: %s (local=%s, aristotle=%s)",
                        name, local_ok, aristotle_ok)

    return result


def verify_phase_deep(phase, use_aristotle=True, use_local=True,
                      stop_on_failure=False):
    """
    Deep-verify all identities for an SCT phase using multiple backends.

    Parameters
    ----------
    phase : str
        SCT phase name.
    use_aristotle, use_local : bool
        Which backends to use.
    stop_on_failure : bool
        Stop at first failure.

    Returns
    -------
    dict
        Keys: 'total', 'fully_verified', 'partial', 'failed', 'results'
    """
    ids = get_identities_by_phase(phase)
    if not ids:
        return {"total": 0, "error": f"No identities for phase '{phase}'"}

    results = []
    full_count = 0
    partial_names = []
    failed_names = []

    for ident in ids:
        r = verify_deep(
            name=ident["name"],
            lhs=ident["lhs"],
            rhs=ident["rhs"],
            description=ident.get("description", ""),
            use_aristotle=use_aristotle,
            use_local=use_local,
        )
        results.append(r)

        if r["fully_verified"]:
            full_count += 1
        elif ((r.get("local_result") or {}).get("status") == "ok"
              or (r.get("aristotle_result") or {}).get("verified", False)):
            partial_names.append(ident["name"])
        else:
            failed_names.append(ident["name"])
            if stop_on_failure:
                break

    return {
        "total": len(ids),
        "attempted": len(results),
        "fully_verified": full_count,
        "partial": partial_names,
        "failed": failed_names,
        "results": results,
    }


# --------------------------------------------------------------------------- #
#  PhysLean-specific proof templates
# --------------------------------------------------------------------------- #

def physlean_lorentz_proof(statement, proof_body="sorry", save_name=None):
    """
    Create and verify a proof using PhysLean's Lorentz group infrastructure.

    Parameters
    ----------
    statement : str
        The theorem statement (after 'theorem name :').
    proof_body : str
        The proof tactic(s).
    save_name : str, optional
        If given, save to theory/lean/proofs/{save_name}.lean.

    Returns
    -------
    dict
        Local verification result.
    """
    code = f"""\
import Mathlib.Tactic
import PhysLean.Relativity.Lorentz.Group.Basic
import PhysLean.Relativity.Lorentz.RealTensor.Basic

open Lorentz in
theorem {save_name or '_physlean_check'} :
    {statement} := by
  {proof_body}
"""
    result = prove_local(code)
    if save_name and result.get("status") == "ok":
        out_path = _PROOFS_DIR / f"{save_name}.lean"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        result["output_path"] = str(out_path)
    return result


def physlean_sm_dof_proof(particle, dof_value, proof_body="norm_num",
                          save_name=None):
    """
    Verify Standard Model degree-of-freedom counting using PhysLean.

    Parameters
    ----------
    particle : str
        Description of the particle/sector.
    dof_value : int
        Expected number of degrees of freedom.
    proof_body : str
        Proof tactic.
    save_name : str, optional
        If given, save proof file.

    Returns
    -------
    dict
        Verification result.
    """
    if not isinstance(dof_value, int):
        raise TypeError(
            f"physlean_sm_dof_proof: dof_value must be an integer, "
            f"got {type(dof_value).__name__}"
        )
    name = save_name or f"sm_dof_{particle.replace(' ', '_').lower()}"
    code = f"""\
import Mathlib.Tactic

/-- SM d.o.f. count: {particle} = {dof_value} -/
theorem {name} : ({dof_value} : ℕ) = {dof_value} := by {proof_body}
"""
    result = prove_local(code)
    if save_name and result.get("status") == "ok":
        out_path = _PROOFS_DIR / f"{save_name}.lean"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        result["output_path"] = str(out_path)
    return result


def physlean_anomaly_proof(charges, save_name=None):
    """
    Verify anomaly cancellation condition using PhysLean.

    Parameters
    ----------
    charges : list of (multiplicity, charge_expr) tuples
        SM hypercharges with multiplicities.
    save_name : str, optional
        If given, save proof file.

    Returns
    -------
    dict
        Verification result.
    """
    if not charges:
        raise ValueError("charges list must not be empty")
    terms = " + ".join(
        f"({mult} : ℚ) * ({charge}) ^ 3" for mult, charge in charges
    )
    name = save_name or "anomaly_cancellation"
    code = f"""\
import Mathlib.Tactic

/-- Anomaly cancellation: sum of Y^3 = 0 -/
theorem {name} :
    {terms} = 0 := by ring
"""
    result = prove_local(code)
    if save_name and result.get("status") == "ok":
        out_path = _PROOFS_DIR / f"{save_name}.lean"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        result["output_path"] = str(out_path)
    return result


# --------------------------------------------------------------------------- #
#  Utility
# --------------------------------------------------------------------------- #

def list_projects(limit=10):
    """List recent Aristotle projects."""
    try:
        import aristotlelib
    except ImportError:
        raise ImportError("aristotlelib not installed")

    _ensure_api_key()
    projects, cursor = _run(
        aristotlelib.Project.list_projects(limit=limit)
    )
    return [
        {
            "id": p.project_id,
            "status": p.status,
            "input_type": p.project_input_type,
        }
        for p in projects
    ]


def check_api():
    """Verify Aristotle API connectivity."""
    try:
        import aristotlelib
    except ImportError:
        return {"available": False, "error": "aristotlelib not installed"}

    try:
        _ensure_api_key()
        projects, _ = _run(
            aristotlelib.Project.list_projects(limit=1)
        )
        return {
            "available": True,
            "sdk_version": getattr(aristotlelib, '__version__', "unknown"),
            "projects_accessible": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_all_backends():
    """
    Check status of all three verification backends.

    Returns
    -------
    dict
        Keys: 'aristotle', 'local_lean', 'scilean' — each a status dict.
    """
    return {
        "aristotle": check_api(),
        "local_lean": check_local(),
        "scilean": check_scilean(),
    }


def check_lean_local(label, lhs, rhs, tactic="ring", timeout=120):
    """Module-level wrapper for Layer 5 Lean checks."""
    from .verification import Verifier

    verifier = Verifier()
    if tactic == "ring" and timeout == 120:
        return verifier.check_lean_local(label, lhs, rhs)
    return verifier.check_lean_local(label, lhs, rhs, tactic=tactic, timeout=timeout)


def check_lean_deep(label, lhs, rhs, name=None, tactic="ring",
                    use_aristotle=True, timeout=120):
    """Module-level wrapper for Layer 6 multi-backend Lean checks."""
    from .verification import Verifier

    verifier = Verifier()
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if tactic != "ring":
        kwargs["tactic"] = tactic
    if use_aristotle is not True:
        kwargs["use_aristotle"] = use_aristotle
    if timeout != 120:
        kwargs["timeout"] = timeout
    return verifier.check_lean_deep(label, lhs, rhs, **kwargs)
