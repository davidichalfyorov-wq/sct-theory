# ruff: noqa: E402, I001
"""Phase 4 Aristotle-backed Lean verification runner for NT-2 and NT-4a."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from aristotlelib import Project, set_api_key

from sct_tools.lean import get_identities_by_phase

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"


def _load_aristotle_key() -> str:
    for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ARISTOTLE_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise RuntimeError("ARISTOTLE_API_KEY not found in .env")


def _build_code(identity: dict[str, str]) -> str:
    description = identity.get("description", "")
    desc_comment = f"/-- {description} -/\n" if description else ""
    return (
        "import Mathlib.Tactic\n\n"
        f"{desc_comment}theorem {identity['name']} :\n"
        f"    {identity['lhs']} = {identity['rhs']} := by\n"
        "  sorry\n"
    )


def _verify_output_has_no_sorry(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    text = output_path.read_text(encoding="utf-8")
    stripped = re.sub(r"--[^\n]*", "", text)
    stripped = re.sub(r"/-.*?-/", "", stripped, flags=re.DOTALL)
    return "sorry" not in stripped


async def _run_identity(identity: dict[str, str], phase_slug: str, polling_seconds: int) -> dict[str, object]:
    out_dir = RESULTS_ROOT / phase_slug / "aristotle"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{identity['name']}.lean"

    project_id = await Project.prove_from_file(
        input_content=_build_code(identity),
        auto_add_imports=False,
        validate_lean_project=False,
        wait_for_completion=False,
        output_file_path=output_path,
    )
    project = await Project.from_id(project_id)
    completed_path = await project.wait_for_completion(
        output_path,
        polling_interval_seconds=polling_seconds,
        max_polling_failures=10,
    )
    completed_path = Path(completed_path)
    verified = _verify_output_has_no_sorry(completed_path)
    return {
        "name": identity["name"],
        "description": identity.get("description", ""),
        "project_id": project_id,
        "output_path": str(completed_path),
        "verified": verified,
    }


async def run_phase_cloud_verification(phase: str, polling_seconds: int = 10) -> dict[str, object]:
    identities = get_identities_by_phase(phase)
    phase_slug = "nt2" if phase == "NT-2" else "nt4a" if phase == "NT-4a" else phase.lower()
    tasks = [_run_identity(identity, phase_slug, polling_seconds) for identity in identities]
    results = await asyncio.gather(*tasks)
    verified = sum(1 for result in results if result["verified"])
    return {
        "phase": phase,
        "total": len(results),
        "verified": verified,
        "failed": [result["name"] for result in results if not result["verified"]],
        "results": results,
    }


async def run_all(polling_seconds: int = 10) -> dict[str, dict[str, object]]:
    set_api_key(_load_aristotle_key())
    nt2, nt4a = await asyncio.gather(
        run_phase_cloud_verification("NT-2", polling_seconds=polling_seconds),
        run_phase_cloud_verification("NT-4a", polling_seconds=polling_seconds),
    )
    return {"NT-2": nt2, "NT-4a": nt4a}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Aristotle-backed Phase 4 verification.")
    parser.add_argument("--polling-seconds", type=int, default=10)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = asyncio.run(run_all(polling_seconds=args.polling_seconds))
    (RESULTS_ROOT / "nt2" / "nt2_verify_phase_cloud.json").write_text(
        json.dumps(payload["NT-2"], indent=2),
        encoding="utf-8",
    )
    (RESULTS_ROOT / "nt4a" / "nt4a_verify_phase_cloud.json").write_text(
        json.dumps(payload["NT-4a"], indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
