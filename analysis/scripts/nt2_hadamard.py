# ruff: noqa: E402, I001
"""Hadamard-style zero analysis for NT-2."""

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

from scripts.nt2_entire_function import FIGURES_DIR, RESULTS_DIR, ALPHA_C, F1_total_complex, F2_total_complex, alpha_R, find_real_axis_zeros


def Pi_entire(z: complex | float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Legacy NT-2 proxy built from the total SM form factors."""
    return 1 + ALPHA_C * F1_total_complex(z, xi=xi, dps=dps) + alpha_R(xi) * F2_total_complex(z, xi=xi, dps=dps)


def find_complex_zeros(
    func,
    *,
    re_bounds: tuple[float, float] = (-40.0, 10.0),
    im_bounds: tuple[float, float] = (-20.0, 20.0),
    grid_size: int = 10,
    dps: int = 100,
) -> list[tuple[float, float]]:
    """Find complex zeros from a deterministic seed grid."""
    mp.mp.dps = dps
    roots: list[tuple[float, float]] = []
    re_values = [re_bounds[0] + (re_bounds[1] - re_bounds[0]) * idx / (grid_size - 1) for idx in range(grid_size)]
    im_values = [im_bounds[0] + (im_bounds[1] - im_bounds[0]) * idx / (grid_size - 1) for idx in range(grid_size)]

    for re_value in re_values:
        for im_value in im_values:
            seed = mp.mpc(re_value, im_value)
            try:
                root = mp.findroot(func, (seed, seed + 0.25 + 0.25j))
            except (ValueError, ZeroDivisionError):
                continue
            root_pair = (round(float(mp.re(root)), 8), round(float(mp.im(root)), 8))
            if not any(abs(root_pair[0] - existing[0]) < 1e-6 and abs(root_pair[1] - existing[1]) < 1e-6 for existing in roots):
                roots.append(root_pair)
    roots.sort()
    return roots


def generate_hadamard_report(
    *,
    xi: float = 0.0,
    dps: int = 100,
    output_path: Path | None = None,
    figure_name: str = "nt2_complex_plane",
) -> dict[str, object]:
    real_roots = find_real_axis_zeros(lambda z: Pi_entire(z, xi=xi, dps=dps), interval=(-200.0, 200.0), dps=dps)
    complex_roots = find_complex_zeros(lambda z: Pi_entire(z, xi=xi, dps=dps), dps=dps)

    report = {
        "phase": "NT-2",
        "xi": xi,
        "real_axis_zeros": real_roots,
        "complex_zeros": complex_roots,
        "hadamard_form": "Pi(z) = exp(az+b) * prod_n E1(z/z_n)",
    }

    init_style()
    fig, ax = create_figure(figsize=(4.5, 3.2))
    if complex_roots:
        ax.scatter([root[0] for root in complex_roots], [root[1] for root in complex_roots], label="Complex zeros")
    if real_roots:
        ax.scatter(real_roots, [0.0] * len(real_roots), marker="x", s=60, label="Real-axis zeros")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$\Re z$")
    ax.set_ylabel(r"$\Im z$")
    ax.set_title("NT-2 Zero Search")
    if complex_roots or real_roots:
        ax.legend()
    fig.tight_layout()
    save_figure(fig, figure_name, fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)

    if output_path is None:
        output_path = RESULTS_DIR / "nt2_hadamard.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NT-2 Hadamard/zero report.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt2_hadamard.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = generate_hadamard_report(xi=args.xi, dps=args.dps, output_path=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
