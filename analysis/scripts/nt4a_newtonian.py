# ruff: noqa: E402, I001
"""Local Yukawa approximation to the NT-4a Newtonian sector."""

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

from scripts.nt4a_propagator import (
    scalar_local_mass,
    spin2_local_mass,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "nt4a"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"


def _simpson_integral(func, a: float, b: float, n_steps: int) -> mp.mpf:
    if n_steps % 2 == 1:
        n_steps += 1
    h = (b - a) / n_steps
    total = func(a) + func(b)
    for idx in range(1, n_steps):
        x_val = a + idx * h
        total += (4 if idx % 2 else 2) * func(x_val)
    return total * h / 3


def effective_masses(*, Lambda: float = 1.0, xi: float = 0.0) -> tuple[mp.mpf, mp.mpf | None]:
    """Return local Yukawa masses induced by the TT and scalar sectors."""
    return spin2_local_mass(Lambda), scalar_local_mass(Lambda, xi)


def phi_local_ratio(
    r: float | mp.mpf,
    *,
    Lambda: float = 1.0,
    xi: float = 0.0,
    dps: int = 80,
) -> mp.mpf:
    """Return the local Yukawa approximation for Phi / Phi_Newton."""
    radius = mp.mpf(r)
    if radius <= 0:
        raise ValueError(f"r must be positive, got {r}")
    mp.mp.dps = dps
    m2, m0 = effective_masses(Lambda=Lambda, xi=xi)
    ratio = 1 - mp.mpf(4) / 3 * mp.e ** (-m2 * radius)
    if m0 is not None:
        ratio += mp.mpf(1) / 3 * mp.e ** (-m0 * radius)
    return ratio


def psi_local_ratio(
    r: float | mp.mpf,
    *,
    Lambda: float = 1.0,
    xi: float = 0.0,
    dps: int = 80,
) -> mp.mpf:
    """Return the local Yukawa approximation for Psi / Psi_Newton."""
    radius = mp.mpf(r)
    if radius <= 0:
        raise ValueError(f"r must be positive, got {r}")
    mp.mp.dps = dps
    m2, m0 = effective_masses(Lambda=Lambda, xi=xi)
    ratio = 1 - mp.mpf(2) / 3 * mp.e ** (-m2 * radius)
    if m0 is not None:
        ratio -= mp.mpf(1) / 3 * mp.e ** (-m0 * radius)
    return ratio


def gamma_local_ratio(
    r: float | mp.mpf,
    *,
    Lambda: float | mp.mpf = 1.0,
    xi: float | mp.mpf = 0.0,
    dps: int = 80,
) -> mp.mpf:
    """Return gamma = Psi / Phi in the local Yukawa approximation."""
    radius = mp.mpf(r)
    if radius == 0:
        m2, m0 = effective_masses(Lambda=float(Lambda), xi=float(xi))
        if m0 is None:
            return -1
        return (mp.mpf(2) * m2 + m0) / (mp.mpf(4) * m2 - m0)
    phi = phi_local_ratio(radius, Lambda=float(Lambda), xi=float(xi), dps=dps)
    psi = psi_local_ratio(radius, Lambda=float(Lambda), xi=float(xi), dps=dps)
    return psi / phi


def potential_ratio(
    r: float,
    *,
    Lambda: float = 1.0,
    xi: float = 0.0,
    dps: int = 80,
) -> mp.mpf:
    """Backward-compatible alias for the local Phi/Newton ratio."""
    return phi_local_ratio(r, Lambda=Lambda, xi=xi, dps=dps)


def small_r_limit_potential(
    *,
    Lambda: float = 1.0,
    xi: float = 0.0,
    G: float = 1.0,
    M: float = 1.0,
    dps: int = 80,
) -> mp.mpf:
    """Return the finite ``r -> 0`` limit when the scalar Yukawa mode is present."""
    mp.mp.dps = dps
    m2, m0 = effective_masses(Lambda=Lambda, xi=xi)
    if m0 is None:
        return mp.inf
    finite_limit = mp.mpf(4) / 3 * m2
    finite_limit -= mp.mpf(1) / 3 * m0
    return -mp.mpf(G) * mp.mpf(M) * finite_limit


def V_modified(
    r: float,
    *,
    Lambda: float = 1.0,
    xi: float = 0.0,
    G: float = 1.0,
    M: float = 1.0,
    dps: int = 80,
) -> mp.mpf:
    _, m0 = effective_masses(Lambda=Lambda, xi=xi)
    if r < 1e-5 / Lambda and m0 is not None:
        return small_r_limit_potential(Lambda=Lambda, xi=xi, G=G, M=M, dps=dps)
    ratio = potential_ratio(r, Lambda=Lambda, xi=xi, dps=dps)
    return -(mp.mpf(G) * mp.mpf(M) / mp.mpf(r)) * ratio


def sample_potential_curve(
    radii: list[float],
    *,
    Lambda: float = 1.0,
    xi: float = 0.0,
    G: float = 1.0,
    M: float = 1.0,
    dps: int = 80,
) -> list[dict[str, float]]:
    samples = []
    for radius in radii:
        value = V_modified(radius, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps)
        newton = -(mp.mpf(G) * mp.mpf(M) / mp.mpf(radius))
        samples.append(
            {
                "r": float(radius),
                "V": float(value),
                "ratio": float(value / newton),
            }
        )
    return samples


def generate_newtonian_report(
    *,
    xi_values: list[float] | None = None,
    Lambda: float = 1.0,
    output_path: Path | None = None,
    figure_name: str = "nt4a_newtonian_potential",
) -> dict[str, object]:
    if xi_values is None:
        xi_values = [0.0, 1 / 6, 1.0]
    radii = [10 ** exponent for exponent in (-1, 0, 1, 2, 3, 4, 5, 6)]
    report = {
        "phase": "NT-4a",
        "Lambda": Lambda,
        "curves": {
            str(xi): sample_potential_curve(radii, Lambda=Lambda, xi=xi)
            for xi in xi_values
        },
    }

    init_style()
    fig, ax = create_figure(figsize=(4.8, 3.3))
    for xi, samples in report["curves"].items():
        ax.plot(
            [sample["r"] for sample in samples],
            [sample["ratio"] for sample in samples],
            marker="o",
            label=rf"$\xi = {float(xi):.3g}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$r / \ell_P$")
    ax.set_ylabel(r"$V(r) / V_{\mathrm{Newton}}(r)$")
    ax.set_title("NT-4a Local Yukawa Approximation")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figure_name, fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)

    if output_path is None:
        output_path = RESULTS_DIR / "nt4a_newtonian.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NT-4a Newtonian-potential report.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt4a_newtonian.json")
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = generate_newtonian_report(Lambda=args.lambda_scale, output_path=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
