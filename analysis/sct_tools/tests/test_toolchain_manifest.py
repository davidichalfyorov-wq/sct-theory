"""Tests that dependency manifests cover the active SCT toolchain."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCT_TOOLS_DIR = ROOT / "analysis" / "sct_tools"
RUNTIME_REQUIREMENTS = ROOT / "requirements.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"

LOCAL_MODULES = {
    "cas_backends",
    "compute",
    "constants",
    "data_io",
    "entanglement",
    "entire_function",
    "fitting",
    "form_factors",
    "form_interface",
    "graphs",
    "lean",
    "plotting",
    "propagator",
    "scripts",
    "tensors",
    "verification",
    "sct_tools",
}

PACKAGE_NAME_MAP = {
    "scienceplots": "scienceplots",
}


def _read_requirements(path):
    pkgs = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = re.split(r"[<>=\\[]", line, maxsplit=1)[0].strip().lower()
        pkgs.add(pkg)
    return pkgs


def _third_party_import_roots():
    roots = set()
    stdlib = set(sys.stdlib_module_names)
    for path in SCT_TOOLS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
    normalized = set()
    for root in roots:
        if root in stdlib or root in LOCAL_MODULES or root.startswith("_"):
            continue
        normalized.add(PACKAGE_NAME_MAP.get(root.lower(), root.lower()))
    return normalized


def test_runtime_requirements_cover_sct_tools_imports():
    declared = _read_requirements(RUNTIME_REQUIREMENTS)
    imported = _third_party_import_roots()
    missing = sorted(imported - declared)
    assert not missing, f"requirements.txt missing runtime packages: {missing}"


def test_dev_requirements_cover_tooling():
    assert DEV_REQUIREMENTS.exists(), "requirements-dev.txt must exist"
    declared = _read_requirements(DEV_REQUIREMENTS)
    expected = {"pytest", "ruff", "notebook", "jupyterlab", "ipykernel"}
    missing = sorted(expected - declared)
    assert not missing, f"requirements-dev.txt missing tooling packages: {missing}"


def test_package_all_exports_are_resolvable():
    if str(ROOT / "analysis") not in sys.path:
        sys.path.insert(0, str(ROOT / "analysis"))

    import sct_tools

    missing = [name for name in sct_tools.__all__ if not hasattr(sct_tools, name)]
    assert not missing, f"sct_tools.__all__ contains unresolved exports: {missing}"
