"""
SCT Theory — Structured Computation Metadata.

Provides a standard envelope for all computation results, ensuring
reproducibility and traceability. Every numerical experiment should
produce a ComputationRecord that captures:
  - What was computed (task_id, parameters)
  - When (timestamp)
  - On what (hardware, software versions, random seed)
  - What came out (results, verification status, duration)

Usage:
    from sct_tools.metadata import record_computation, save_record

    rec = record_computation("FND1-EXP14", params={"N": 10000, "M": 30},
                             seed=42)
    # ... run computation ...
    rec.results = {"alpha_C": 13/120, "n_checks": 88}
    rec.complete()
    save_record(rec, "analysis/fnd1_data/exp14_results.json")
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _auto_hardware() -> dict:
    """Detect hardware: CPU, RAM, GPU."""
    hw = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
        "gpu": None,
        "vram_gb": None,
    }
    # RAM detection
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    # GPU detection via CuPy
    try:
        _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
        if os.path.isdir(_cuda):
            os.add_dll_directory(_cuda)
        import cupy as cp
        dev = cp.cuda.Device(0)
        hw["gpu"] = dev.attributes.get("DeviceName", str(dev))
        hw["vram_gb"] = round(dev.mem_info[1] / (1024**3), 1)
    except Exception:
        pass
    return hw


def _auto_software() -> dict:
    """Capture software versions."""
    sw = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        from sct_tools import __version__
        sw["sct_tools"] = __version__
    except ImportError:
        pass
    for pkg in ("numpy", "scipy", "mpmath", "sympy"):
        try:
            mod = __import__(pkg)
            sw[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return sw


@dataclass
class ComputationRecord:
    """Standard metadata envelope for SCT computations.

    Attributes:
        task_id: Identifier (e.g. "FND1-EXP14", "PPN1-MCMC", "NT2-CHECK").
        timestamp: ISO 8601 creation time (auto-filled).
        software: Dict of package versions (auto-filled).
        hardware: Dict of CPU/GPU/RAM info (auto-filled).
        seed: Random seed used (None if deterministic).
        parameters: Experiment-specific input parameters.
        results: Numerical outputs (filled after computation).
        verification: Layer verdicts {layer_name: {status, n_checks, details}}.
        duration_s: Wall-clock time in seconds (filled by complete()).
        status: "RUNNING" | "COMPLETE" | "PARTIAL" | "FAILED".
        notes: Free-form notes.
    """

    task_id: str
    timestamp: str = ""
    software: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    seed: int | None = None
    parameters: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    duration_s: float = 0.0
    status: str = "RUNNING"
    notes: str = ""
    _start_time: float = field(default=0.0, repr=False)

    def complete(self, status: str = "COMPLETE") -> None:
        """Mark computation as finished, record duration."""
        self.duration_s = round(time.time() - self._start_time, 3)
        self.status = status

    def fail(self, error_msg: str = "") -> None:
        """Mark computation as failed."""
        self.duration_s = round(time.time() - self._start_time, 3)
        self.status = "FAILED"
        if error_msg:
            self.notes = (self.notes + "\n" + error_msg).strip()

    def attach_verification(self, verifier) -> None:
        """Copy check results from a Verifier instance.

        Args:
            verifier: sct_tools.verification.Verifier instance.
        """
        self.verification = {
            "label": verifier.name,
            "n_pass": verifier.n_pass,
            "n_fail": verifier.n_fail,
            "all_passed": verifier.all_passed,
            "checks": [
                {k: v for k, v in c.items() if k != "error" or v}
                for c in verifier.checks
            ],
        }

    def to_dict(self) -> dict:
        """Serialize to dict (excludes private fields)."""
        d = asdict(self)
        d.pop("_start_time", None)
        return d


def record_computation(
    task_id: str,
    parameters: dict | None = None,
    seed: int | None = None,
    detect_hardware: bool = True,
    notes: str = "",
) -> ComputationRecord:
    """Create a pre-filled ComputationRecord.

    Args:
        task_id: Identifier for the computation.
        parameters: Experiment-specific input parameters.
        seed: Random seed (None if deterministic).
        detect_hardware: Auto-detect CPU/GPU/RAM (adds ~0.1s).
        notes: Free-form notes.

    Returns:
        ComputationRecord ready for use.
    """
    return ComputationRecord(
        task_id=task_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        software=_auto_software(),
        hardware=_auto_hardware() if detect_hardware else {},
        seed=seed,
        parameters=parameters or {},
        notes=notes,
        _start_time=time.time(),
    )


class _RecordEncoder(json.JSONEncoder):
    """Handle non-serializable types in records."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "__float__"):
            return float(obj)
        if hasattr(obj, "__int__"):
            return int(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_record(record: ComputationRecord, path: str | Path) -> Path:
    """Save a ComputationRecord as JSON.

    Args:
        record: The record to save.
        path: Output file path.

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False,
                  cls=_RecordEncoder)
    return path.resolve()


def load_record(path: str | Path) -> ComputationRecord:
    """Load a ComputationRecord from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        ComputationRecord instance.

    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove private fields that may have leaked
    data.pop("_start_time", None)

    return ComputationRecord(**data)


def validate_record(record: ComputationRecord) -> list[str]:
    """Check a record for completeness issues.

    Returns:
        List of warning strings (empty = all good).
    """
    warnings = []
    if not record.task_id:
        warnings.append("task_id is empty")
    if not record.timestamp:
        warnings.append("timestamp is empty")
    if not record.parameters:
        warnings.append("parameters dict is empty")
    if not record.results and record.status == "COMPLETE":
        warnings.append("status=COMPLETE but results dict is empty")
    if record.status == "RUNNING":
        warnings.append("status still RUNNING — call complete() or fail()")
    if record.seed is None and record.status == "COMPLETE":
        warnings.append("no seed recorded — results may not be reproducible")
    if not record.software:
        warnings.append("software versions not recorded")
    return warnings
