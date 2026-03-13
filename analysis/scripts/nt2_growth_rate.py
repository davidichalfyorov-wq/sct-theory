# ruff: noqa: E402, I001
"""Growth-rate scans and figures for NT-2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import create_figure, init_style, save_figure

from scripts.nt2_entire_function import FIGURES_DIR, RESULTS_DIR, F1_total_complex, F2_total_complex, estimate_growth_rate


def generate_growth_report(
    *,
    xi: float = 0.0,
    radii: list[float] | None = None,
    angles: list[float] | None = None,
    dps: int = 100,
    output_path: Path | None = None,
    figure_name: str = "nt2_growth_rate",
) -> dict[str, object]:
    if radii is None:
        radii = [50.0, 100.0, 200.0, 400.0, 800.0]
    if angles is None:
        angles = [0.0, mp.pi / 4, mp.pi / 2, 3 * mp.pi / 4, mp.pi]

    report = {
        "phase": "NT-2",
        "xi": xi,
        "F1": estimate_growth_rate(lambda z: F1_total_complex(z, xi=xi, dps=dps), radii=radii, angles=angles, dps=dps),
        "F2": estimate_growth_rate(lambda z: F2_total_complex(z, xi=xi, dps=dps), radii=radii, angles=angles, dps=dps),
    }

    init_style()
    fig, ax = create_figure(figsize=(4.5, 3.2))
    ax.plot(
        [sample["radius"] for sample in report["F1"]["samples"]],
        [sample["max_modulus"] for sample in report["F1"]["samples"]],
        marker="o",
        label=rf"$|F_1|_{{\max}}$, $\rho \approx {report['F1']['order']:.3f}$",
    )
    ax.plot(
        [sample["radius"] for sample in report["F2"]["samples"]],
        [sample["max_modulus"] for sample in report["F2"]["samples"]],
        marker="s",
        label=rf"$|F_2|_{{\max}}$, $\rho \approx {report['F2']['order']:.3f}$",
    )
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$M(r) = \max_{|z|=r} |F_k(z)|$")
    ax.set_title("NT-2 Growth-Rate Scan")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figure_name, fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)

    if output_path is None:
        output_path = RESULTS_DIR / "nt2_growth_rate.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NT-2 growth-rate report and figure.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt2_growth_rate.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = generate_growth_report(xi=args.xi, dps=args.dps, output_path=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
