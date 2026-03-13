# ruff: noqa: E402, I001
"""Standalone verification runner for NT-4a."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.verification import Verifier

from scripts.nt4a_linearize import (
    linearized_curvature_identities,
    random_k_vectors,
    random_symmetric_tensors,
)
from scripts.nt4a_newtonian import V_modified, generate_newtonian_report, small_r_limit_potential
from scripts.nt4a_propagator import (
    Pi_TT,
    Pi_scalar,
    check_bianchi_identity,
    check_gauge_invariance,
    find_first_positive_real_tt_zero,
    generate_propagator_figures,
    scalar_mode_coefficient,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "nt4a"


def run_nt4a_verification(*, xi: float = 0.0, dps: int = 100, output_path: Path | None = None) -> dict[str, object]:
    mp.mp.dps = dps
    verifier = Verifier("NT-4a Linearized Field Equations", quiet=True)

    identities = linearized_curvature_identities()
    verifier.check_value("Ricci scalar vanishes in TT gauge", identities["Ricci_scalar_TT"], 0)

    reference_pi_tt = {
        0.1: mp.mpf("1.0180639123257555"),
        0.5: mp.mpf("1.0232520423925255"),
        1.0: mp.mpf("0.8994734409"),
    }
    reference_pi_scalar = {
        0.1: mp.mpf("1.0170949568754593"),
        0.5: mp.mpf("1.0934412650431455"),
        1.0: mp.mpf("1.2043061095"),
    }

    for z, expected in reference_pi_tt.items():
        verifier.check_value_mp(
            f"Re Pi_TT({z}) reference",
            mp.re(Pi_TT(z, xi=xi, dps=dps)),
            expected,
            tol_digits=8,
        )
    for z, expected in reference_pi_scalar.items():
        verifier.check_value_mp(
            f"Re Pi_scalar({z}) reference",
            mp.re(Pi_scalar(z, xi=xi, dps=dps)),
            expected,
            tol_digits=8,
        )

    verifier.check_value("Pi_TT(0)=1", mp.re(Pi_TT(0, xi=xi, dps=dps)), 1.0)
    verifier.check_value("Pi_scalar(0)=1", mp.re(Pi_scalar(0, xi=xi, dps=dps)), 1.0)
    verifier.check_value("scalar mode coefficient", scalar_mode_coefficient(xi), 6 * (xi - 1 / 6) ** 2)

    for index, (k_vec, h_tensor) in enumerate(
        zip(random_k_vectors(seed=42, n_vectors=5), random_symmetric_tensors(seed=43, n_tensors=5), strict=True),
        start=1,
    ):
        verifier.check_value(f"Gauge invariance sample {index}", check_gauge_invariance(k_vec, xi=xi), True)
        verifier.check_value(f"Bianchi identity sample {index}", check_bianchi_identity(k_vec, xi=xi, h_tensor=h_tensor), True)

    v_large = V_modified(1e4, xi=xi, dps=dps)
    verifier.check_value("Large-r Newtonian asymptotic ratio", v_large / (-(1 / 1e4)), 1.0, rtol=1e-2)
    verifier.check_value("Small-r potential is finite", mp.isfinite(V_modified(1e-6, xi=xi, dps=dps)), True)
    verifier.check_value(
        "Conformal local potential reopens the short-distance divergence",
        mp.isinf(small_r_limit_potential(xi=1 / 6, dps=dps)),
        True,
    )
    verifier.check_value_mp(
        "Positive-real TT zero location",
        find_first_positive_real_tt_zero(xi=xi, dps=dps),
        mp.mpf("2.4148388898653689"),
        tol_digits=10,
    )

    report = {
        "phase": "NT-4a",
        "xi": xi,
        "checks_passed": verifier.n_pass,
        "checks_failed": verifier.n_fail,
        "propagator": generate_propagator_figures(dps=dps),
        "newtonian": generate_newtonian_report(output_path=RESULTS_DIR / "nt4a_newtonian.json"),
    }
    if output_path is None:
        output_path = RESULTS_DIR / "nt4a_verify.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verifier.n_fail:
        raise AssertionError(f"NT-4a verification failed with {verifier.n_fail} failed checks")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone NT-4a verification.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt4a_verify.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_nt4a_verification(xi=args.xi, dps=args.dps, output_path=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
