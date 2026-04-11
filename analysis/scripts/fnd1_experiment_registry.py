"""
FND-1 Experiment Registry — Auto-discovery infrastructure.

Provides:
1. Standard metadata schema for all experiments
2. Auto-scanner that discovers all experiment JSONs
3. Progress file mechanism for running experiments
4. Migration for existing result files

Usage in experiment scripts:
    from fnd1_experiment_registry import ExperimentMeta, save_experiment, update_progress

    meta = ExperimentMeta(
        route=3, name="commutator_hm",
        description="Commutator [H,M] curvature test with mediation analysis",
        N=3000, M=80, status="running",
    )
    update_progress(meta, step="Part 1: eps=-0.5", pct=0.15)
    ...
    meta.status = "completed"
    meta.verdict = "BREAKTHROUGH"
    save_experiment(meta, results_dict, output_path)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "speculative" / "numerics" / "ensemble_results"
PROGRESS_FILE = RESULTS_DIR / "_progress.json"

ROUTE_INFO = {
    1: {
        "name": "Ensemble Spectral Observables",
        "status": "CLOSED",
        "color": "red",
        "description": "Symmetrized BD operator, Euclidean heat trace. No curvature sensitivity found.",
    },
    2: {
        "name": "Coarse-Grained / Emergent Spectral Triple",
        "status": "OPEN",
        "color": "gray",
        "description": "Emergent reconstruction step. Mathematically unbuilt. Reserved as fallback.",
    },
    3: {
        "name": "Lorentzian/Krein Reformulation",
        "status": "IN PROGRESS",
        "color": "blue",
        "description": "Retarded BD operator, SVD, commutator [H,M]. Currently testing causal asymmetry.",
    },
}


@dataclass
class ExperimentMeta:
    route: int
    name: str
    description: str = ""
    N: int = 0
    M: int = 0
    status: str = "running"  # running, completed, failed
    verdict: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    wall_time_sec: float = 0.0
    parameters: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)


def save_experiment(meta: ExperimentMeta, results: dict, path: Path):
    """Save experiment with standard metadata header."""
    data = {
        "_meta": asdict(meta),
        **results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)


def update_progress(meta: ExperimentMeta, step: str = "", pct: float = 0.0,
                    eta_min: float = 0.0):
    """Write a progress file for the dashboard to read."""
    progress = {
        "route": meta.route,
        "name": meta.name,
        "description": meta.description,
        "status": "running",
        "step": step,
        "pct": pct,
        "eta_min": eta_min,
        "N": meta.N,
        "M": meta.M,
        "timestamp": datetime.now().isoformat(),
    }
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
        f.flush()


def clear_progress():
    """Remove progress file when experiment completes."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def scan_experiments() -> list[dict]:
    """
    Auto-discover all experiment JSONs in the results directory.
    Returns list of dicts with metadata + file info.
    """
    experiments = []

    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue  # skip progress files

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Extract metadata (from _meta if present, or infer from content)
        if "_meta" in data:
            meta = data["_meta"]
        else:
            meta = _infer_meta(path.name, data)

        meta["file"] = path.name
        meta["file_path"] = str(path)
        meta["file_size"] = path.stat().st_size
        meta["file_mtime"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        meta["data"] = data

        experiments.append(meta)

    return experiments


def _infer_meta(filename: str, data: dict) -> dict:
    """Infer metadata from legacy files without _meta header."""
    meta = {
        "route": 0,
        "name": filename.replace(".json", ""),
        "description": "",
        "status": "completed",
        "verdict": data.get("verdict", ""),
        "wall_time_sec": data.get("wall_time_sec", 0),
        "N": 0,
        "M": 0,
        "tags": [],
    }

    # Infer route from filename
    if "route3" in filename:
        meta["route"] = 3
    elif "gate" in filename:
        meta["route"] = 1

    # Infer parameters
    params = data.get("parameters", {})
    meta["N"] = params.get("N", 0)
    meta["M"] = params.get("M", 0)

    # Infer description from filename
    name_map = {
        "gate0_gate1_N200_M50": "Gate 0+1: Null model + Quick Kill (N=200)",
        "gate3_gate4_results": "Gate 3+4: Finite-size scaling + Ensemble stability",
        "gate5_curvature_results": "Gate 5: Curvature sensitivity (SDW extraction)",
        "gate5_matched_pairs": "Gate 5: Matched-pairs curvature test (N=1000)",
        "gate5_n5000_followup": "Gate 5: N=5000 follow-up",
        "gate5_triple_verification": "Gate 5: Triple verification (reproducibility + null + dose)",
        "route3_quickkill": "Route 3 Quick Kill: SVD discrimination + curvature + DW zeta",
        "route3_verification": "Route 3 Verification: Multi-epsilon + reproducibility",
        "route3_commutator": "Route 3 Commutator [H,M]: Mediation analysis + reproducibility",
    }
    base = filename.replace(".json", "")
    meta["description"] = name_map.get(base, base)

    # Infer tags
    if "gate5" in filename:
        meta["tags"].append("curvature")
    if "verification" in filename or "triple" in filename:
        meta["tags"].append("verification")
    if "quickkill" in filename:
        meta["tags"].append("quickkill")
    if "commutator" in filename:
        meta["tags"].append("commutator")

    return meta


def get_progress() -> dict | None:
    """Read current progress file, if any."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def get_route_experiments(experiments: list[dict], route: int) -> list[dict]:
    """Filter experiments by route."""
    return [e for e in experiments if e.get("route") == route]


def _clean(obj):
    """Clean NaN/inf/numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj
