"""
FND-1 Deep Cross-Experiment Analysis Pipeline.

Loads all experiment JSONs, normalizes schemas, computes cross-experiment
statistics, scaling laws, and generates publication-quality figures.

Usage:
    python analysis/scripts/fnd1_deep_analysis.py

Requires: numpy, scipy, matplotlib (+ scienceplots optional)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "speculative" / "numerics" / "ensemble_results"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "fnd1_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Unified experiment record
# ---------------------------------------------------------------------------

@dataclass
class ExpRecord:
    """Normalized record for one experiment result."""
    exp_id: str             # e.g. "exp1", "exp7"
    name: str               # e.g. "d4_link_verification"
    route: int
    dimension: int          # 2, 3, or 4
    operator: str           # "link_laplacian", "bd", "sj", "commutator", "magnetic", "ollivier_ricci"
    N_values: list[int]
    N_primary: int
    M: int
    verdict: str

    # Primary signal: partial r at N_primary (quadrupole preferred)
    r_partial: float = 0.0
    p_partial: float = 1.0
    predictor: str = ""     # "linear" or "quadratic"

    # Effect size
    cohen_d: float = 0.0

    # Geometry recovery
    rho_spearman: float = 0.0

    # Scaling data: r_partial at each N
    scaling_N: list[int] = field(default_factory=list)
    scaling_r: list[float] = field(default_factory=list)
    scaling_rho: list[float] = field(default_factory=list)
    signal_grows: bool = False

    # Per-profile breakdown
    r_partial_quadrupole: float = 0.0
    r_partial_coscosh: float = 0.0

    # Heat trace specific
    plateau_value: float = 0.0
    bd_goe_ratio: float = 0.0
    frac_significant: float = 0.0  # EXP-9: fraction of tau values significant

    # Raw data ref
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders: one per experiment family
# ---------------------------------------------------------------------------

def _safe_get(d, *keys, default=0.0):
    """Navigate nested dict safely. Returns default for None (NaN in JSON)."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return default if d is None else d


def load_exp1(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    N_primary = params.get("N_primary", 2000)

    # Get quadrupole mediation at N_primary
    configs = data.get("configs", {})
    quad = configs.get("quadrupole", {})
    tidal_primary = configs.get(f"tidal_N{N_primary}", {})

    med_q = quad.get("mediation", {}) or {}
    med_t = tidal_primary.get("mediation", {}) or {}

    # Scaling across N for tidal
    scaling = data.get("scaling", {})

    return ExpRecord(
        exp_id="exp1", name="d4_link_verification", route=2, dimension=4,
        operator="link_laplacian",
        N_values=params.get("N_values", []),
        N_primary=N_primary, M=params.get("M", 80),
        verdict=data.get("verdict", ""),
        r_partial=float(med_q.get("fiedler_r_partial", 0)),
        p_partial=float(med_q.get("fiedler_p_partial", 1)),
        r_partial_quadrupole=float(med_q.get("fiedler_r_partial", 0)),
        r_partial_coscosh=float(med_t.get("fiedler_r_partial", 0)),
        rho_spearman=float(_safe_get(quad, "geometry_flat", "rho_spearman_mean")),
        scaling_N=scaling.get("N", []),
        scaling_r=scaling.get("partial_r", []),
        scaling_rho=scaling.get("rho_flat", []),
        signal_grows=scaling.get("signal_grows", False),
        raw=data,
    )


def load_exp2(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    N_primary = params.get("N_primary", 3000)
    results = data.get("results", {})

    # Quadrupole at N_primary
    q_key = f"quadrupole_N{N_primary}"
    c_key = f"coscosh_N{N_primary}"
    med_q = _safe_get(results, q_key, "mediation", default={})
    med_c = _safe_get(results, c_key, "mediation", default={})

    summary = data.get("summary", {})

    return ExpRecord(
        exp_id="exp2", name="d4_sj_verification", route=2, dimension=4,
        operator="sj_vacuum",
        N_values=params.get("N_values", []),
        N_primary=N_primary, M=params.get("M", 80),
        verdict=data.get("verdict", ""),
        r_partial=float(_safe_get(med_q, "best_r_partial")),
        p_partial=float(_safe_get(med_q, "best_p_partial", default=1.0)),
        r_partial_quadrupole=float(_safe_get(med_q, "best_r_partial")),
        r_partial_coscosh=float(_safe_get(med_c, "best_r_partial")),
        raw=data,
    )


def load_exp3(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    N_primary = params.get("N_primary", 2000)
    results = data.get("results", {})

    q_key = f"quadrupole_N{N_primary}"
    c_key = f"coscosh_N{N_primary}"
    med_q = _safe_get(results, q_key, "mediation", default={})
    med_c = _safe_get(results, c_key, "mediation", default={})

    return ExpRecord(
        exp_id="exp3", name="d4_commutator", route=3, dimension=4,
        operator="commutator",
        N_values=params.get("N_values", []),
        N_primary=N_primary, M=params.get("M", 100),
        verdict=data.get("verdict", ""),
        r_partial=float(_safe_get(med_q, "best_r_partial")),
        p_partial=float(_safe_get(med_q, "best_p_partial", default=1.0)),
        r_partial_quadrupole=float(_safe_get(med_q, "best_r_partial")),
        r_partial_coscosh=float(_safe_get(med_c, "best_r_partial")),
        raw=data,
    )


def load_exp4(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    N_primary = params.get("N_primary", 2000)
    results = data.get("results", {})

    q_key = f"quadrupole_N{N_primary}"
    c_key = f"coscosh_N{N_primary}"

    # EXP-4 has mediation_spectral_diff etc.
    med_q = _safe_get(results, q_key, "mediation_spectral_diff", default={})
    med_c = _safe_get(results, c_key, "mediation_spectral_diff", default={})

    return ExpRecord(
        exp_id="exp4", name="d4_magnetic_phase", route=7, dimension=4,
        operator="magnetic_laplacian",
        N_values=params.get("N_values", []),
        N_primary=N_primary, M=params.get("M", 100),
        verdict=data.get("verdict", ""),
        r_partial=float(_safe_get(med_q, "best_r_partial")),
        p_partial=float(_safe_get(med_q, "best_p_partial", default=1.0)),
        r_partial_quadrupole=float(_safe_get(med_q, "best_r_partial")),
        r_partial_coscosh=float(_safe_get(med_c, "best_r_partial")),
        raw=data,
    )


def load_exp5(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})

    # Find highest N
    N_vals = sorted([int(k) for k in rbn.keys()])
    N_primary = N_vals[-1] if N_vals else 3000
    best = rbn.get(str(N_primary), {})

    return ExpRecord(
        exp_id="exp5", name="d4_spectral_action", route=2, dimension=4,
        operator="link_laplacian",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 60),
        verdict=data.get("verdict", ""),
        r_partial=float(best.get("curvature_r_partial", 0)),
        p_partial=float(best.get("curvature_p_partial", 1)),
        bd_goe_ratio=float(best.get("bd_goe_ratio", 0)),
        raw=data,
    )


def load_exp6(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})
    N_primary = params.get("N_primary", 10000)
    best = rbn.get(str(N_primary), {})

    # Scaling
    N_vals = sorted([int(k) for k in rbn.keys()])
    scaling_r = [rbn[str(n)].get("fiedler_r_partial", 0) for n in N_vals]
    scaling_rho = [rbn[str(n)].get("rho_flat", 0) for n in N_vals]

    r_p = float(best.get("fiedler_r_partial") or 0)
    return ExpRecord(
        exp_id="exp6", name="d4_higher_N", route=2, dimension=4,
        operator="link_laplacian",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 30),
        verdict=data.get("verdict", ""),
        r_partial=r_p,
        p_partial=float(best.get("fiedler_p_partial") or 1),
        cohen_d=float(best.get("fiedler_cohen_d") or 0),
        rho_spearman=float(best.get("rho_flat") or 0),
        r_partial_quadrupole=r_p,  # EXP-6 mediation uses quadrupole only
        scaling_N=N_vals,
        scaling_r=scaling_r,
        scaling_rho=scaling_rho,
        signal_grows=data.get("scaling", {}).get("signal_grows", False),
        raw=data,
    )


def load_exp7(data: dict) -> ExpRecord:
    rbn = data.get("results_by_N", {})
    params = data.get("parameters", {})
    N_vals = sorted([int(k) for k in rbn.keys()])
    N_primary = N_vals[-1] if N_vals else 5000
    best = rbn.get(str(N_primary), {})

    plateau = _safe_get(best, "bd_corrected", "plateau", default={})

    return ExpRecord(
        exp_id="exp7", name="d2_dw_a0", route=3, dimension=2,
        operator="bd",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 100),
        verdict=data.get("verdict", ""),
        plateau_value=float(plateau.get("value", 0)),
        bd_goe_ratio=float(best.get("bd_goe_ratio", 0)),
        raw=data,
    )


def load_exp8(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})
    N_primary = params.get("N_primary", 1000)
    best = rbn.get(str(N_primary), {})
    med = best.get("mediation", {})

    return ExpRecord(
        exp_id="exp8", name="d2_hasse_ricci", route=4, dimension=2,
        operator="ollivier_ricci",
        N_values=sorted([int(k) for k in rbn.keys()]),
        N_primary=N_primary, M=params.get("M", 80),
        verdict=data.get("verdict", ""),
        r_partial=float(med.get("r_partial_extended", 0)),
        p_partial=float(med.get("p_partial_extended", 1)),
        raw=data,
    )


def load_exp9(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})
    N_vals = sorted([int(k) for k in rbn.keys()])
    N_primary = N_vals[-1] if N_vals else 1000
    best = rbn.get(str(N_primary), {})

    frac_sig = float(best.get("frac_significant", 0))
    tK_max_z = float(best.get("tK_max_z", 0))

    return ExpRecord(
        exp_id="exp9", name="d2_large_ensemble", route=2, dimension=2,
        operator="link_laplacian",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 5000),
        verdict=data.get("verdict", ""),
        frac_significant=frac_sig,
        # Use tK_max_z as pseudo-r_partial for d=2 null check
        r_partial=min(frac_sig, 1.0),  # >0.3 means "detected"
        raw=data,
    )


def load_exp11(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})
    N_primary = params.get("N_primary", 2000)
    best = rbn.get(str(N_primary), {})
    med = best.get("mediation", {})

    N_vals = sorted([int(k) for k in rbn.keys()])
    scaling_r = [_safe_get(rbn[str(n)], "mediation", "best_r_partial") for n in N_vals]
    scaling_rho = [rbn[str(n)].get("rho_flat", 0) for n in N_vals]

    return ExpRecord(
        exp_id="exp11", name="d3_intermediate", route=2, dimension=3,
        operator="link_laplacian",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 80),
        verdict=data.get("verdict", ""),
        r_partial=float(med.get("best_r_partial", 0)),
        p_partial=float(med.get("best_p_partial", 1)),
        rho_spearman=float(best.get("rho_flat", 0)),
        scaling_N=N_vals,
        scaling_r=scaling_r,
        scaling_rho=scaling_rho,
        raw=data,
    )


def load_exp12(data: dict) -> ExpRecord:
    params = data.get("parameters", {})
    rf = data.get("results_flat", {})
    scaling = data.get("scaling_k2", {})

    N_vals = sorted([int(k) for k in rf.keys()])
    N_primary = N_vals[-1] if N_vals else 5000
    best = rf.get(str(N_primary), {})

    rho_primary = _safe_get(best, "summary_by_k", "2", "rho_spearman_mean")

    return ExpRecord(
        exp_id="exp12", name="d2_link_scaling", route=2, dimension=2,
        operator="link_laplacian",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 80),
        verdict=data.get("verdict", ""),
        rho_spearman=float(rho_primary),
        scaling_N=scaling.get("N", []),
        scaling_rho=scaling.get("rho_spearman", []),
        signal_grows=scaling.get("is_increasing", False),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

def load_exp5b(data: dict) -> ExpRecord:
    """EXP-5b: high-N extension of EXP-5. Same schema."""
    rec = load_exp5(data)
    rec.exp_id = "exp5b"
    rec.name = "d4_spectral_highN"
    return rec


def load_exp13(data: dict) -> ExpRecord:
    """EXP-13: d=3 commutator [H,M] Weyl null test."""
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})
    N_primary = params.get("N_primary", 2000)
    best = rbn.get(str(N_primary), {})
    med = best.get("mediation", {})

    N_vals = sorted([int(k) for k in rbn.keys()])
    scaling_r = [_safe_get(rbn[str(n)], "mediation", "best_r_partial") for n in N_vals]

    return ExpRecord(
        exp_id="exp13", name="d3_commutator", route=3, dimension=3,
        operator="commutator",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 100),
        verdict=data.get("verdict", ""),
        r_partial=float(med.get("best_r_partial", 0) or 0),
        p_partial=float(med.get("best_p_partial", 1) or 1),
        scaling_N=N_vals,
        scaling_r=scaling_r,
        raw=data,
    )


def load_exp14(data: dict) -> ExpRecord:
    """EXP-14: d=4 commutator [H,M] high-N scaling."""
    params = data.get("parameters", {})
    rbn = data.get("results_by_N", {})
    N_primary = params.get("N_primary", 10000)
    best = rbn.get(str(N_primary), {})
    med = best.get("mediation", {})

    N_vals = sorted([int(k) for k in rbn.keys()])
    scaling = data.get("scaling", {})
    scaling_r = scaling.get("r_partial_best_of_3", scaling.get("r_partial", []))

    return ExpRecord(
        exp_id="exp14", name="d4_commutator_highN", route=3, dimension=4,
        operator="commutator",
        N_values=N_vals,
        N_primary=N_primary, M=params.get("M", 80),
        verdict=data.get("verdict", ""),
        r_partial=float(med.get("best_r_partial", 0) or 0),
        p_partial=float(med.get("best_p_partial", 1) or 1),
        scaling_N=N_vals,
        scaling_r=[float(x) if x is not None else 0 for x in scaling_r],
        signal_grows=scaling.get("signal_grows", False),
        raw=data,
    )


LOADERS = {
    "exp1_d4_link_verification": load_exp1,
    "exp2_d4_sj_verification": load_exp2,
    "exp3_d4_commutator": load_exp3,
    "exp4_d4_magnetic_phase": load_exp4,
    "exp5_d4_spectral_action": load_exp5,
    "exp5b_d4_spectral_highN": load_exp5b,
    "exp6_d4_higher_N": load_exp6,
    "exp7_dw_a0": load_exp7,
    "exp8_d2_hasse_ricci": load_exp8,
    "exp9_d2_large_ensemble": load_exp9,
    "exp11_d3_intermediate": load_exp11,
    "exp12_link_scaling": load_exp12,
    "exp13_d3_commutator": load_exp13,
    "exp14_d4_commutator_highN": load_exp14,
}


def load_all() -> list[ExpRecord]:
    """Load all available experiment JSONs and normalize to ExpRecords."""
    records = []
    for json_path in sorted(RESULTS_DIR.glob("exp*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {json_path.name}: {e}")
            continue

        meta = data.get("_meta", {})
        name = meta.get("name", json_path.stem)

        if name in LOADERS:
            try:
                rec = LOADERS[name](data)
                records.append(rec)
                print(f"  Loaded {name}: d={rec.dimension}, "
                      f"r_partial={rec.r_partial:+.4f}")
            except Exception as e:
                print(f"  ERROR loading {name}: {e}")
        else:
            print(f"  SKIP {json_path.name}: no loader for '{name}'")

    return records


# ---------------------------------------------------------------------------
# Analysis 1: Summary table
# ---------------------------------------------------------------------------

def summary_table(records: list[ExpRecord]) -> str:
    """Generate markdown summary table of all experiments."""
    lines = [
        "| Exp | Dimension | Operator | N_primary | M | |r_partial| | p | Verdict |",
        "|-----|-----------|----------|-----------|---|------------|---|---------|",
    ]
    for r in sorted(records, key=lambda x: (x.dimension, x.exp_id)):
        # Approximate 95% CI for r_partial (1/sqrt(M) heuristic, accurate for |r|<0.5)
        se_r = 1.0 / max(1, r.M) ** 0.5
        ci_lo = r.r_partial - 1.96 * se_r
        ci_hi = r.r_partial + 1.96 * se_r
        v_short = r.verdict[:50] + "..." if len(r.verdict) > 50 else r.verdict
        lines.append(
            f"| {r.exp_id} | d={r.dimension} | {r.operator} | {r.N_primary} "
            f"| {r.M} | {abs(r.r_partial):.4f} [{ci_lo:+.2f},{ci_hi:+.2f}] "
            f"| {r.p_partial:.2e} | {v_short} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 2: Operator comparison (d=4 only)
# ---------------------------------------------------------------------------

def operator_comparison(records: list[ExpRecord]) -> dict:
    """Compare operators in d=4: which detects curvature best?"""
    d4 = [r for r in records if r.dimension == 4]
    if not d4:
        return {}

    result = {}
    for r in sorted(d4, key=lambda x: abs(x.r_partial), reverse=True):
        result[r.exp_id] = {
            "operator": r.operator,
            "r_partial": r.r_partial,
            "p_partial": r.p_partial,
            "r_quad": r.r_partial_quadrupole,
            "r_cosc": r.r_partial_coscosh,
            "N": r.N_primary,
            "M": r.M,
        }
    return result


# ---------------------------------------------------------------------------
# Analysis 3: Dimension comparison (d=2 vs d=3 vs d=4)
# ---------------------------------------------------------------------------

def dimension_comparison(records: list[ExpRecord]) -> dict:
    """Compare link_laplacian signal across dimensions.
    Separates curvature detection (r_partial) from geometry recovery (rho_spearman).
    """
    link_recs = [r for r in records if r.operator == "link_laplacian"]
    by_dim = {}
    for r in link_recs:
        by_dim.setdefault(r.dimension, []).append(r)

    result = {}
    for dim in sorted(by_dim.keys()):
        recs = by_dim[dim]
        # Best curvature detection (exclude EXP-12 which has no mediation)
        curv_recs = [r for r in recs if r.exp_id != "exp12"]
        best_curv = max(curv_recs, key=lambda x: abs(x.r_partial)) if curv_recs else None
        # Best geometry recovery
        best_geom = max(recs, key=lambda x: x.rho_spearman)
        result[f"d={dim}"] = {
            "curvature_exp": best_curv.exp_id if best_curv else "N/A",
            "curvature_r_partial": best_curv.r_partial if best_curv else 0.0,
            "curvature_p_partial": best_curv.p_partial if best_curv else 1.0,
            "geometry_exp": best_geom.exp_id,
            "geometry_rho": best_geom.rho_spearman,
            "n_experiments": len(recs),
        }
    return result


# ---------------------------------------------------------------------------
# Analysis 4: Profile comparison (quadrupole vs coscosh)
# ---------------------------------------------------------------------------

def profile_comparison(records: list[ExpRecord]) -> dict:
    """Compare quadrupole (pure Weyl) vs coscosh (Weyl + monopole) across d=4.
    This is the KEY falsification test: if coscosh signal >> quadrupole signal,
    the coscosh result is density-driven, not curvature-driven.

    NOTE: Classification uses |r|>0.10 threshold (exploratory, not formally tested).
    A proper comparison requires bootstrap CI on (r_cosc - r_quad), which needs
    per-sprinkling data (see fnd1_per_sprinkling.py).
    """
    d4 = [r for r in records if r.dimension == 4
          and (r.r_partial_quadrupole != 0 or r.r_partial_coscosh != 0)]
    if not d4:
        return {}

    result = {}
    for r in d4:
        rq = r.r_partial_quadrupole
        rc = r.r_partial_coscosh
        # Interpretation
        if abs(rq) > 0.10 and abs(rc) > 0.10:
            interpretation = "BOTH (genuine + density)"
        elif abs(rq) > 0.10 and abs(rc) < 0.10:
            interpretation = "QUADRUPOLE ONLY (pure Weyl)"
        elif abs(rq) < 0.10 and abs(rc) > 0.10:
            interpretation = "COSCOSH ONLY (density artifact)"
        else:
            interpretation = "NEITHER (no signal)"

        result[r.exp_id] = {
            "operator": r.operator,
            "r_quadrupole": rq,
            "r_coscosh": rc,
            "ratio": round(abs(rc) / abs(rq), 4) if abs(rq) > 1e-6 else 9999.0,
            "interpretation": interpretation,
        }
    return result


# ---------------------------------------------------------------------------
# Analysis 5b: Cross-validation (EXP-1 vs EXP-6 consistency)
# ---------------------------------------------------------------------------

def cross_validation(records: list[ExpRecord]) -> dict:
    """Check consistency between overlapping experiments."""
    results = {}

    # EXP-1 and EXP-6 both measure link Fiedler in d=4
    exp1 = next((r for r in records if r.exp_id == "exp1"), None)
    exp6 = next((r for r in records if r.exp_id == "exp6"), None)

    if exp1 and exp6 and exp1.scaling_N and exp6.scaling_N:
        # Find overlapping N values
        overlap = set(exp1.scaling_N) & set(exp6.scaling_N)
        if overlap:
            checks = []
            for N in sorted(overlap):
                i1 = exp1.scaling_N.index(N) if N in exp1.scaling_N else -1
                i6 = exp6.scaling_N.index(N) if N in exp6.scaling_N else -1
                if (i1 >= 0 and i6 >= 0
                        and i1 < len(exp1.scaling_r)
                        and i6 < len(exp6.scaling_r)):
                    r1 = exp1.scaling_r[i1]
                    r6 = exp6.scaling_r[i6]
                    # SE approximation: 1/sqrt(M). Formal test needs per-ensemble CIs.
                    se1 = 1.0 / max(1, exp1.M) ** 0.5
                    se6 = 1.0 / max(1, exp6.M) ** 0.5
                    se_diff = (se1 ** 2 + se6 ** 2) ** 0.5
                    checks.append({
                        "N": N, "exp1_r": round(r1, 4), "exp6_r": round(r6, 4),
                        "diff": round(abs(r1 - r6), 4),
                        "se_diff": round(se_diff, 4),
                        "within_2se": abs(r1 - r6) < 2 * se_diff,
                        "note": "qualitative (formal test needs per-ensemble SEs)",
                    })
            results["exp1_vs_exp6"] = {
                "overlapping_N": sorted(overlap),
                "checks": checks,
                "all_within_2se": all(c["within_2se"] for c in checks),
            }

    # EXP-12 (d=2 geometry) vs EXP-11 (d=3 geometry): rho should improve d=3 > d=2 at same N
    exp12 = next((r for r in records if r.exp_id == "exp12"), None)
    exp11 = next((r for r in records if r.exp_id == "exp11"), None)
    if exp12 and exp11 and exp12.rho_spearman > 0 and exp11.rho_spearman > 0:
        results["geometry_d2_vs_d3"] = {
            "rho_d2": exp12.rho_spearman,
            "rho_d3": exp11.rho_spearman,
            "d2_better": exp12.rho_spearman > exp11.rho_spearman,
        }

    return results


# ---------------------------------------------------------------------------
# Analysis 6: Scaling laws
# ---------------------------------------------------------------------------

def scaling_analysis(records: list[ExpRecord]) -> dict:
    """Extract scaling laws from experiments with multiple N values."""
    results = {}

    for r in records:
        if len(r.scaling_N) < 3:
            continue

        Ns = np.array(r.scaling_N, dtype=float)

        # Geometry scaling: (1 - rho) vs N — guard against length mismatch
        if r.scaling_rho and len(r.scaling_rho) == len(Ns) and all(x is not None and x > 0 for x in r.scaling_rho):
            one_minus_rho = np.array([1 - x if x is not None else 1.0 for x in r.scaling_rho])
            mask = one_minus_rho > 1e-10
            if np.sum(mask) >= 3:
                lr = stats.linregress(np.log(Ns[mask]), np.log(one_minus_rho[mask]))
                results[f"{r.exp_id}_geometry"] = {
                    "exponent": float(lr.slope),
                    "R2": float(lr.rvalue ** 2),
                    "description": f"(1-rho) ~ N^{{{lr.slope:.2f}}}",
                }

        # Signal scaling: |r_partial| vs N (log-log, power law) — guard against length mismatch
        clean_r = [x for x in (r.scaling_r or []) if x is not None]
        if clean_r and len(r.scaling_r) == len(Ns) and any(abs(x) > 0 for x in clean_r):
            abs_r = np.array([abs(x) if x is not None else 0.0 for x in r.scaling_r])
            mask = abs_r > 1e-10
            n_pts = int(np.sum(mask))
            if n_pts >= 3:
                lr = stats.linregress(np.log(Ns[mask]), np.log(abs_r[mask]))
                results[f"{r.exp_id}_signal"] = {
                    "exponent": float(lr.slope),
                    "R2": float(lr.rvalue ** 2),
                    "n_points": n_pts,
                    "description": f"|r_partial| ~ N^{{{lr.slope:.2f}}}",
                    "grows": bool(lr.slope > 0 and lr.rvalue ** 2 > 0.7 and n_pts >= 4),
                }

    return results


# ---------------------------------------------------------------------------
# Analysis 7: Per-experiment detailed report
# ---------------------------------------------------------------------------

def per_experiment_report(records: list[ExpRecord]) -> str:
    """Detailed breakdown for each experiment."""
    lines = []
    for r in sorted(records, key=lambda x: x.exp_id):
        lines.append(f"\n--- {r.exp_id}: {r.name} (d={r.dimension}, {r.operator}) ---")
        lines.append(f"  N_primary={r.N_primary}, M={r.M}, N_values={r.N_values}")
        lines.append(f"  r_partial={r.r_partial:+.4f}, p={r.p_partial:.2e}")
        if r.r_partial_quadrupole != 0 or r.r_partial_coscosh != 0:
            lines.append(f"  r_quad={r.r_partial_quadrupole:+.4f}, r_cosc={r.r_partial_coscosh:+.4f}")
        if r.rho_spearman > 0:
            lines.append(f"  rho_spearman={r.rho_spearman:.4f}")
        if r.cohen_d != 0:
            lines.append(f"  cohen_d={r.cohen_d:+.2f}")
        if r.plateau_value > 0:
            lines.append(f"  plateau={r.plateau_value:.6f}")
        if r.bd_goe_ratio > 0:
            lines.append(f"  BD/GOE={r.bd_goe_ratio:.2f}")
        if r.frac_significant > 0:
            lines.append(f"  frac_significant={r.frac_significant:.3f}")
        if r.signal_grows:
            lines.append(f"  SIGNAL GROWS WITH N")
        v_short = r.verdict[:100] if r.verdict else "(no verdict)"
        lines.append(f"  Verdict: {v_short}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 8: Meta-analysis (Fisher's method)
# ---------------------------------------------------------------------------

def meta_analysis(records: list[ExpRecord]) -> dict:
    """Combine p-values across d=4 experiments using Fisher's method.
    Selects ONE experiment per operator (highest N) to ensure approximate independence.
    """
    d4 = [r for r in records if r.dimension == 4 and r.p_partial < 1.0 and r.p_partial > 0]
    if len(d4) < 2:
        return {"n_experiments": len(d4), "note": "insufficient experiments"}

    # One per operator (highest N_primary wins)
    by_op = {}
    for r in d4:
        if r.operator not in by_op or r.N_primary > by_op[r.operator].N_primary:
            by_op[r.operator] = r
    d4_indep = list(by_op.values())

    if len(d4_indep) < 2:
        return {"n_experiments": len(d4_indep), "note": "insufficient independent operators"}

    p_values = [r.p_partial for r in d4_indep]
    # Fisher's combined test: -2 * sum(log(p)) ~ chi^2(2k)
    chi2_stat = -2 * sum(np.log(p) for p in p_values)
    df = 2 * len(p_values)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df)

    # Check if one p-value dominates
    log_contribs = [-2 * np.log(p) for p in p_values]
    max_contrib = max(log_contribs)
    dominant_frac = max_contrib / chi2_stat if chi2_stat > 0 else 0
    dominant_exp = d4_indep[log_contribs.index(max_contrib)]

    return {
        "n_experiments": len(d4_indep),
        "experiments": [f"{r.exp_id}({r.operator})" for r in d4_indep],
        "p_values": p_values,
        "chi2_stat": float(chi2_stat),
        "df": df,
        "combined_p": float(combined_p),
        "significant": combined_p < 0.01,
        "dominant_operator": f"{dominant_exp.exp_id}({dominant_exp.operator})",
        "dominant_fraction": round(dominant_frac, 2),
        "caveat": ("Experiments share underlying causal-set code; "
                    "combined p is a lower bound, not exact. "
                    f"Driven {dominant_frac*100:.0f}% by {dominant_exp.exp_id}."),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _init_style():
    """Initialize publication-ready matplotlib style."""
    try:
        plt.style.use(["science", "high-vis"])
    except Exception:
        plt.rcParams.update({
            "font.size": 10, "axes.linewidth": 0.8,
            "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        })


def fig_operator_comparison(records: list[ExpRecord]):
    """Bar chart: |r_partial| by operator in d=4."""
    d4 = [r for r in records if r.dimension == 4]
    if not d4:
        return

    _init_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    names = [f"{r.exp_id}\n{r.operator}" for r in d4]
    r_quad = [abs(r.r_partial_quadrupole) for r in d4]
    r_cosc = [abs(r.r_partial_coscosh) for r in d4]

    x = np.arange(len(d4))
    w = 0.35
    ax.bar(x - w / 2, r_quad, w, label="Quadrupole (pure Weyl)", color="#2196F3")
    ax.bar(x + w / 2, r_cosc, w, label="Coscosh (+ monopole)", color="#FF9800", alpha=0.7)
    ax.axhline(0.10, color="red", linestyle="--", linewidth=0.8, label="r=0.10 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("|r_partial|")
    ax.set_title("d=4 Curvature Detection: Operator Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "operator_comparison.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "operator_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: operator_comparison.pdf")


def fig_dimension_comparison(records: list[ExpRecord]):
    """Bar chart: |r_partial| by dimension for link_laplacian."""
    link_recs = [r for r in records if r.operator == "link_laplacian"]
    if len(link_recs) < 2:
        return

    _init_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: curvature signal by dimension
    by_dim = {}
    for r in link_recs:
        by_dim.setdefault(r.dimension, []).append(r)

    dims = sorted(by_dim.keys())
    best_r = [max(abs(r.r_partial) for r in by_dim[d]) for d in dims]
    best_rho = [max(r.rho_spearman for r in by_dim[d]) for d in dims]
    colors = {2: "#4CAF50", 3: "#FF9800", 4: "#2196F3"}

    ax1.bar(range(len(dims)), best_r,
            color=[colors.get(d, "gray") for d in dims])
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels([f"d={d}" for d in dims])
    ax1.set_ylabel("|r_partial|")
    ax1.set_title("Curvature Signal by Dimension")
    ax1.axhline(0.10, color="red", linestyle="--", linewidth=0.8)

    # Right: geometry recovery by dimension
    ax2.bar(range(len(dims)), best_rho,
            color=[colors.get(d, "gray") for d in dims])
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels([f"d={d}" for d in dims])
    ax2.set_ylabel("Spearman rho")
    ax2.set_title("Geometry Recovery by Dimension")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dimension_comparison.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "dimension_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: dimension_comparison.pdf")


def fig_scaling(records: list[ExpRecord]):
    """Scaling plots for experiments with multiple N."""
    scaling_recs = [r for r in records if len(r.scaling_N) >= 3 and r.scaling_rho]
    if not scaling_recs:
        return

    _init_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in scaling_recs:
        Ns = np.array(r.scaling_N)
        rhos = np.array(r.scaling_rho)
        if all(x is not None and x > 0 for x in rhos):
            ax.plot(Ns, rhos, "o-", label=f"{r.exp_id} (d={r.dimension})", markersize=5)

    ax.set_xlabel("N (causal set size)")
    ax.set_ylabel("Spearman rho (geometry recovery)")
    ax.set_xscale("log")
    ax.set_title("Geometry Recovery Scaling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_geometry.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "scaling_geometry.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: scaling_geometry.pdf")


def fig_heat_trace(records: list[ExpRecord]):
    """Heat trace comparison for EXP-5 and EXP-7."""
    for r in records:
        if r.exp_id not in ("exp5", "exp7"):
            continue

        rbn = r.raw.get("results_by_N", {})
        if not rbn:
            continue

        _init_style()
        fig, ax = plt.subplots(figsize=(7, 5))
        ylabel = r"$K(\tau)$"
        title = f"{r.exp_id}: Heat Trace"
        has_data = False

        for N_str in sorted(rbn.keys(), key=int):
            block = rbn[N_str]

            # Find the curve data
            if r.exp_id == "exp7":
                curve = _safe_get(block, "bd_corrected", "tK_curve", default={})
                ylabel = r"$\tau \cdot K(\tau)$"
                title = f"EXP-7: BD Heat Trace (d=2)"
            else:
                # EXP-5: look in by_eps for the flat case
                by_eps = block.get("by_eps", {})
                flat_key = [k for k in by_eps if "0.0" in k or k.endswith("_0.0")]
                if flat_key:
                    curve = by_eps[flat_key[0]].get("t2K_curve", {})
                else:
                    continue
                ylabel = r"$\tau^2 \cdot K(\tau)$"
                title = f"EXP-5: Link Laplacian Heat Trace (d=4)"

            if curve:
                taus = sorted(curve.keys(), key=float)
                x = [float(t) for t in taus]
                y = [curve[t] for t in taus]
                ax.plot(x, y, label=f"N={N_str}", linewidth=1)
                has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"heat_trace_{r.exp_id}.pdf", dpi=150)
        fig.savefig(FIGURES_DIR / f"heat_trace_{r.exp_id}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: heat_trace_{r.exp_id}.pdf")


def fig_profile_comparison(records: list[ExpRecord]):
    """Paired bar chart: quadrupole vs coscosh |r_partial| for each d=4 experiment."""
    d4 = [r for r in records if r.dimension == 4
          and (r.r_partial_quadrupole != 0 or r.r_partial_coscosh != 0)]
    if not d4:
        return

    _init_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    names = [f"{r.exp_id}\n{r.operator}" for r in d4]
    rq = [abs(r.r_partial_quadrupole) for r in d4]
    rc = [abs(r.r_partial_coscosh) for r in d4]

    x = np.arange(len(d4))
    w = 0.35
    bars_q = ax.bar(x - w / 2, rq, w, label="Quadrupole (pure Weyl)", color="#2196F3")
    bars_c = ax.bar(x + w / 2, rc, w, label="Coscosh (+ monopole)", color="#FF9800", alpha=0.7)
    ax.axhline(0.10, color="red", linestyle="--", linewidth=0.8, label="|r|=0.10 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("|r_partial| (mediated)")
    ax.set_title("Profile Falsification Test: Pure Weyl vs Density-Contaminated")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profile_comparison.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "profile_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: profile_comparison.pdf")


def fig_signal_scaling(records: list[ExpRecord]):
    """Signal strength |r_partial| vs N for experiments with scaling data."""
    scaling_recs = [r for r in records if len(r.scaling_N) >= 3 and r.scaling_r
                    and any(x is not None and abs(x) > 0 for x in r.scaling_r)]
    if not scaling_recs:
        return

    _init_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in scaling_recs:
        Ns = np.array(r.scaling_N)
        rs = np.array([abs(x) if x is not None else 0.0 for x in r.scaling_r])
        if len(Ns) != len(rs):
            continue
        ax.plot(Ns, rs, "o-", label=f"{r.exp_id} (d={r.dimension})", markersize=5)

    ax.axhline(0.10, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="|r|=0.10")
    ax.set_xlabel("N (causal set size)")
    ax.set_ylabel("|r_partial| (curvature signal)")
    ax.set_xscale("log")
    ax.set_title("Curvature Signal Scaling with N")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_signal.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "scaling_signal.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: scaling_signal.pdf")


def fig_summary_heatmap(records: list[ExpRecord]):
    """Heatmap: experiment × metric overview."""
    if len(records) < 3:
        return

    _init_style()
    names = [r.exp_id for r in records]
    metrics = {
        "|r_partial|": [abs(r.r_partial) for r in records],
        "rho_geom": [r.rho_spearman for r in records],
        "Cohen d": [abs(r.cohen_d) for r in records],
    }

    data = np.array(list(metrics.values()))
    fig, ax = plt.subplots(figsize=(max(6, len(records) * 0.8), 3))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(list(metrics.keys()), fontsize=9)

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val > 0.001:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.3 else "black")

    ax.set_title("FND-1 Cross-Experiment Overview")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "summary_heatmap.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "summary_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: summary_heatmap.pdf")


# ---------------------------------------------------------------------------
# Advanced Analysis A: Spectral dimension from heat trace
# ---------------------------------------------------------------------------

def spectral_dimension_analysis(records: list[ExpRecord]) -> dict:
    """Extract spectral dimension d_S(tau) = -2 d ln K / d ln tau from heat traces.

    Uses EXP-5 (d=4 link Laplacian) and EXP-7 (d=2 BD) t2K/tK curves.
    d_S should converge to embedding dimension at intermediate tau.
    """
    results = {}

    for r in records:
        if r.exp_id not in ("exp5", "exp7"):
            continue

        rbn = r.raw.get("results_by_N", {})
        for N_str in sorted(rbn.keys(), key=int):
            block = rbn[N_str]

            # Find flat heat trace curve
            if r.exp_id == "exp7":
                curve = _safe_get(block, "bd_corrected", "tK_curve", default={})
                power = 1  # tau*K in EXP-7
                expected_d = 2
            else:
                by_eps = block.get("by_eps", {})
                flat_keys = [k for k in by_eps if k.endswith("_0.0") or "0.0" in k]
                if not flat_keys:
                    continue
                curve = by_eps[flat_keys[0]].get("t2K_curve", {})
                power = 2  # tau^2*K in EXP-5
                expected_d = 4

            if not curve or len(curve) < 5:
                continue

            pairs = sorted([(float(k), v) for k, v in curve.items()], key=lambda x: x[0])
            taus = np.array([p[0] for p in pairs])
            vals = np.array([p[1] for p in pairs])

            # K(tau) = val / tau^power
            K = vals / np.maximum(taus ** power, 1e-30)
            # Filter positive K
            mask = K > 1e-20
            if np.sum(mask) < 5:
                continue
            taus_f = taus[mask]
            K_f = K[mask]

            # d_S(tau) = -2 * d(ln K) / d(ln tau)
            log_tau = np.log(taus_f)
            log_K = np.log(K_f)
            # Numerical derivative (central difference)
            d_S = np.zeros(len(log_tau))
            for i in range(1, len(log_tau) - 1):
                d_S[i] = -2 * (log_K[i + 1] - log_K[i - 1]) / (log_tau[i + 1] - log_tau[i - 1])
            d_S[0] = d_S[1]
            d_S[-1] = d_S[-2]

            # Find plateau region (intermediate tau)
            mid = (taus_f > taus_f[2]) & (taus_f < taus_f[-3])
            if np.sum(mid) > 0:
                d_S_plateau = float(np.median(d_S[mid]))
            else:
                d_S_plateau = float(np.median(d_S))

            key = f"{r.exp_id}_N{N_str}"
            results[key] = {
                "expected_d": expected_d,
                "d_S_plateau": round(d_S_plateau, 2),
                "d_S_range": [round(float(np.min(d_S[1:-1])), 2),
                              round(float(np.max(d_S[1:-1])), 2)],
                "n_tau_points": len(taus_f),
                "matches_expected": abs(d_S_plateau - expected_d) < expected_d * 0.5,
            }

    return results


def fig_spectral_dimension(records: list[ExpRecord]):
    """Plot d_S(tau) from heat traces."""
    have_data = False
    for r in records:
        if r.exp_id in ("exp5", "exp7"):
            have_data = True
            break
    if not have_data:
        return

    _init_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in records:
        if r.exp_id not in ("exp5", "exp7"):
            continue

        rbn = r.raw.get("results_by_N", {})
        if not rbn:
            continue
        N_max = max(rbn.keys(), key=int)
        block = rbn[N_max]

        if r.exp_id == "exp7":
            curve = _safe_get(block, "bd_corrected", "tK_curve", default={})
            power, expected_d, label = 1, 2, f"EXP-7 BD (d=2, N={N_max})"
        else:
            by_eps = block.get("by_eps", {})
            flat_keys = [k for k in by_eps if k.endswith("_0.0") or "0.0" in k]
            if not flat_keys:
                continue
            curve = by_eps[flat_keys[0]].get("t2K_curve", {})
            power, expected_d, label = 2, 4, f"EXP-5 Link (d=4, N={N_max})"

        if not curve or len(curve) < 5:
            continue

        pairs = sorted([(float(k), v) for k, v in curve.items()], key=lambda x: x[0])
        taus = np.array([p[0] for p in pairs])
        vals = np.array([p[1] for p in pairs])
        K = vals / np.maximum(taus ** power, 1e-30)
        mask = K > 1e-20
        if np.sum(mask) < 5:
            continue

        log_tau = np.log(taus[mask])
        log_K = np.log(K[mask])
        d_S = np.zeros(len(log_tau))
        for i in range(1, len(log_tau) - 1):
            d_S[i] = -2 * (log_K[i + 1] - log_K[i - 1]) / (log_tau[i + 1] - log_tau[i - 1])
        d_S[0] = d_S[1]
        d_S[-1] = d_S[-2]

        ax.plot(taus[mask][1:-1], d_S[1:-1], "-", label=label, linewidth=1.5)
        ax.axhline(expected_d, linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$d_S(\tau)$")
    ax.set_title("Spectral Dimension from Heat Trace")
    ax.set_ylim(0, 8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spectral_dimension.pdf", dpi=150)
    fig.savefig(FIGURES_DIR / "spectral_dimension.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: spectral_dimension.pdf")


# ---------------------------------------------------------------------------
# Advanced Analysis B: Heat trace curvature sensitivity
# ---------------------------------------------------------------------------

def heat_trace_curvature(records: list[ExpRecord]) -> dict:
    """Compare heat trace curves between flat and curved for each N.

    If curvature shifts the heat trace, Delta(t^d/2 * K) should be nonzero.
    """
    results = {}

    for r in records:
        if r.exp_id != "exp5":
            continue

        rbn = r.raw.get("results_by_N", {})
        for N_str in sorted(rbn.keys(), key=int):
            by_eps = rbn[N_str].get("by_eps", {})

            flat_key = None
            curved_keys = []
            for k in by_eps:
                if k.endswith("_0.0") or "_0.0" in k:
                    flat_key = k
                elif "quadrupole" in k:
                    curved_keys.append(k)

            if not flat_key or not curved_keys:
                continue

            flat_curve = by_eps[flat_key].get("t2K_curve", {})
            if not flat_curve:
                continue

            for ck in curved_keys:
                curved_curve = by_eps[ck].get("t2K_curve", {})
                if not curved_curve:
                    continue

                # Compute Delta at matching tau values
                common_taus = set(flat_curve.keys()) & set(curved_curve.keys())
                if len(common_taus) < 3:
                    continue

                deltas = []
                for t in sorted(common_taus, key=float):
                    deltas.append(curved_curve[t] - flat_curve[t])

                results[f"N{N_str}_{ck}"] = {
                    "mean_delta": float(np.mean(deltas)),
                    "max_delta": float(np.max(np.abs(deltas))),
                    "n_tau": len(deltas),
                    "sign_consistent": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
                }

    return results


# ---------------------------------------------------------------------------
# Advanced Analysis C: Operator ranking across all metrics
# ---------------------------------------------------------------------------

def operator_deep_ranking(records: list[ExpRecord]) -> dict:
    """Rank d=4 operators by multiple criteria, not just r_partial."""
    d4 = [r for r in records if r.dimension == 4]
    if not d4:
        return {}

    ranking = {}
    for r in d4:
        score = 0
        reasons = []

        # 1. Quadrupole signal strength
        rq = abs(r.r_partial_quadrupole)
        if rq > 0.10:
            score += 2
            reasons.append(f"quad_r={rq:.2f}")
        elif rq > 0.05:
            score += 1
            reasons.append(f"quad_r={rq:.2f}(weak)")

        # 2. Profile falsification: coscosh >> quadrupole = density artifact
        rc = abs(r.r_partial_coscosh)
        if rq > 0.10 and rc > 0.10 and rc / max(rq, 0.01) < 2:
            score += 1
            reasons.append("profile_consistent")
        elif rq > 0.10 and rc < rq:
            score += 2
            reasons.append("quad>cosc(clean)")
        elif rc > 0.10 and rq < 0.05:
            score -= 1
            reasons.append("cosc_only(artifact)")

        # 3. p-value
        if r.p_partial < 0.001:
            score += 2
            reasons.append(f"p={r.p_partial:.1e}")
        elif r.p_partial < 0.01:
            score += 1
            reasons.append(f"p={r.p_partial:.2e}")

        # 4. Scaling (if available)
        if r.signal_grows:
            score += 3
            reasons.append("signal_grows")

        # 5. Geometry recovery
        if r.rho_spearman > 0.3:
            score += 1
            reasons.append(f"rho={r.rho_spearman:.2f}")

        ranking[r.exp_id] = {
            "operator": r.operator,
            "score": score,
            "reasons": reasons,
            "r_quad": rq,
            "r_cosc": rc,
            "p": r.p_partial,
        }

    return dict(sorted(ranking.items(), key=lambda x: x[1]["score"], reverse=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FND-1 DEEP CROSS-EXPERIMENT ANALYSIS")
    print("=" * 70)
    print()

    # Load
    print("Loading experiments...")
    records = load_all()
    print(f"\nLoaded {len(records)} experiments")

    if not records:
        print("No experiment JSONs found. Exiting.")
        return

    # Per-experiment detailed report
    print("\n" + "=" * 70)
    print("PER-EXPERIMENT DETAILS")
    print("=" * 70)
    details = per_experiment_report(records)
    print(details)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    table = summary_table(records)
    print(table)

    # Operator comparison
    print("\n" + "=" * 70)
    print("OPERATOR COMPARISON (d=4)")
    print("=" * 70)
    op_comp = operator_comparison(records)
    for eid, info in op_comp.items():
        print(f"  {eid} ({info['operator']}): r_partial={info['r_partial']:+.4f}"
              f"  [quad={info['r_quad']:+.4f}, cosc={info['r_cosc']:+.4f}]")

    # Dimension comparison
    print("\n" + "=" * 70)
    print("DIMENSION COMPARISON (link_laplacian)")
    print("=" * 70)
    dim_comp = dimension_comparison(records)
    for dim, info in dim_comp.items():
        print(f"  {dim}: curvature r={info['curvature_r_partial']:+.4f} ({info['curvature_exp']}), "
              f"geometry rho={info['geometry_rho']:.4f} ({info['geometry_exp']})")

    # Profile comparison
    print("\n" + "=" * 70)
    print("PROFILE COMPARISON (quadrupole vs coscosh, d=4)")
    print("=" * 70)
    prof_comp = profile_comparison(records)
    for eid, info in prof_comp.items():
        print(f"  {eid} ({info['operator']}): quad={info['r_quadrupole']:+.4f}, "
              f"cosc={info['r_coscosh']:+.4f} → {info['interpretation']}")

    # Cross-validation
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)
    xval = cross_validation(records)
    for key, info in xval.items():
        print(f"  {key}: {info}")
    if not xval:
        print("  (insufficient overlapping experiments)")

    # Scaling
    print("\n" + "=" * 70)
    print("SCALING LAWS")
    print("=" * 70)
    scaling = scaling_analysis(records)
    for key, info in scaling.items():
        print(f"  {key}: {info}")

    # Meta-analysis
    print("\n" + "=" * 70)
    print("META-ANALYSIS (Fisher's method, d=4, one per operator)")
    print("=" * 70)
    meta = meta_analysis(records)
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Advanced: Spectral dimension
    print("\n" + "=" * 70)
    print("SPECTRAL DIMENSION d_S(tau)")
    print("=" * 70)
    spec_dim = spectral_dimension_analysis(records)
    for key, info in spec_dim.items():
        status = "MATCH" if info["matches_expected"] else "MISMATCH"
        print(f"  {key}: d_S={info['d_S_plateau']:.1f} (expected {info['expected_d']}) [{status}]")
    if not spec_dim:
        print("  (no heat trace data available)")

    # Advanced: Heat trace curvature sensitivity
    print("\n" + "=" * 70)
    print("HEAT TRACE CURVATURE SENSITIVITY")
    print("=" * 70)
    ht_curv = heat_trace_curvature(records)
    for key, info in ht_curv.items():
        sign = "consistent" if info["sign_consistent"] else "mixed"
        print(f"  {key}: mean_delta={info['mean_delta']:+.6f}, "
              f"max={info['max_delta']:.6f}, sign={sign}")
    if not ht_curv:
        print("  (no EXP-5 quadrupole heat trace data)")

    # Advanced: Operator deep ranking
    print("\n" + "=" * 70)
    print("OPERATOR DEEP RANKING (d=4)")
    print("=" * 70)
    ranking = operator_deep_ranking(records)
    for eid, info in ranking.items():
        print(f"  #{info['score']:+d} {eid} ({info['operator']}): {', '.join(info['reasons'])}")
    if not ranking:
        print("  (no d=4 experiments)")

    # Figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    fig_operator_comparison(records)
    fig_dimension_comparison(records)
    fig_profile_comparison(records)
    fig_scaling(records)
    fig_signal_scaling(records)
    fig_heat_trace(records)
    fig_spectral_dimension(records)
    fig_summary_heatmap(records)

    # Save analysis JSON
    analysis_output = {
        "n_experiments": len(records),
        "experiments": [r.exp_id for r in records],
        "operator_comparison": op_comp,
        "dimension_comparison": dim_comp,
        "profile_comparison": prof_comp,
        "cross_validation": xval,
        "scaling": scaling,
        "meta_analysis": meta,
        "spectral_dimension": spec_dim,
        "heat_trace_curvature": ht_curv,
        "operator_ranking": ranking,
        "summary": table,
    }

    out_path = RESULTS_DIR / "fnd1_cross_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)
    print(f"\nSaved analysis: {out_path}")

    # Overall verdict
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    d4_signals = [r for r in records if r.dimension == 4 and abs(r.r_partial) > 0.10]
    d3_signal = any(r.dimension == 3 and abs(r.r_partial) > 0.10 for r in records)
    # d=2 null: check each experiment's actual metric (not just r_partial)
    d2_recs = [r for r in records if r.dimension == 2]
    d2_null = (
        len(d2_recs) > 0
        # Mediation-based experiments: r_partial should be small
        and all(abs(r.r_partial) < 0.10 for r in d2_recs
                if r.exp_id not in ("exp12", "exp9"))
        # EXP-9 uses frac_significant as signal metric
        and all(r.frac_significant < 0.3 for r in d2_recs if r.exp_id == "exp9")
    )
    geometry_works = any(r.rho_spearman > 0.5 for r in records)

    print(f"  d=4 curvature signals (|r|>0.10): {len(d4_signals)}")
    if d4_signals:
        for s in d4_signals:
            se = 1.0 / max(1, s.M) ** 0.5
            print(f"    {s.exp_id} ({s.operator}): r={s.r_partial:+.4f} "
                  f"[{s.r_partial-1.96*se:+.2f},{s.r_partial+1.96*se:+.2f}]")
    print(f"  d=3 curvature signal: {'YES' if d3_signal else 'NO'}")
    print(f"  d=2 mediation null: {'YES' if d2_null else 'NO'}")
    # EXP-7 separately (different metric: DW a0 prediction, not mediation)
    exp7 = next((r for r in records if r.exp_id == "exp7"), None)
    if exp7:
        print(f"  d=2 DW-a0 prediction: {'PASS' if abs(exp7.plateau_value - 0.159) < 0.02 else 'FAIL'}"
              f" (plateau={exp7.plateau_value:.4f}, BD/GOE={exp7.bd_goe_ratio:.2f})")
    print(f"  Geometry recovery: {'YES' if geometry_works else 'NO'}")

    if meta.get("significant"):
        print(f"  Fisher combined p = {meta['combined_p']:.2e}"
              f" ({meta.get('caveat', '')})")
    else:
        print(f"  Fisher combined p = {meta.get('combined_p', 'N/A')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
