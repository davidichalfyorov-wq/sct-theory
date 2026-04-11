# ruff: noqa: E402, I001
"""NT-4a propagator and kinetic-operator utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import create_figure, init_style, save_figure

from scripts.nt2_entire_function import F1_total_complex, F2_total_complex
from scripts.nt4a_linearize import (
    check_off_shell_bianchi_identity,
    check_off_shell_gauge_invariance,
    contract_first_index_with_k,
    default_symmetric_tensor,
    scalar_projector,
    tt_projector,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "nt4a"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"

ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60


def spin2_local_coefficient() -> mp.mpf:
    """Return the local spin-2 coefficient c2 inherited from Phase 3."""
    return LOCAL_C2


def alpha_R(xi: float | mp.mpf) -> mp.mpf:
    xi_mp = mp.mpf(xi)
    if abs(xi_mp - mp.mpf(1) / 6) < mp.mpf("1e-14"):
        return mp.mpf(0)
    return 2 * (xi_mp - mp.mpf(1) / 6) ** 2


def scalar_mode_coefficient(xi: float | mp.mpf) -> mp.mpf:
    xi_mp = mp.mpf(xi)
    if abs(xi_mp - mp.mpf(1) / 6) < mp.mpf("1e-14"):
        return mp.mpf(0)
    return 6 * (xi_mp - mp.mpf(1) / 6) ** 2


def spin2_local_mass(Lambda: float | mp.mpf) -> mp.mpf:
    """Return the local Yukawa spin-2 mass m2 = Lambda / sqrt(c2)."""
    return mp.mpf(Lambda) / mp.sqrt(spin2_local_coefficient())


def scalar_local_mass(Lambda: float | mp.mpf, xi: float | mp.mpf) -> mp.mpf | None:
    """Return the local Yukawa scalar mass, or ``None`` at conformal coupling."""
    coeff = scalar_mode_coefficient(xi)
    if abs(coeff) < mp.mpf("1e-40"):
        return None
    return mp.mpf(Lambda) / mp.sqrt(coeff)


def _normalize_shape(value: mp.mpc, zero_value: mp.mpc) -> mp.mpc:
    if abs(zero_value) < mp.mpf("1e-40"):
        return mp.mpc(1)
    return value / zero_value


def F1_shape(z: complex | float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    return _normalize_shape(F1_total_complex(z, xi=xi, dps=dps), F1_total_complex(0, xi=xi, dps=dps))


def F2_shape(z: complex | float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    zero_value = F2_total_complex(0, xi=xi, dps=dps)
    if abs(zero_value) < mp.mpf("1e-40"):
        return mp.mpc(1)
    return _normalize_shape(F2_total_complex(z, xi=xi, dps=dps), zero_value)


def Pi_TT(z: complex | float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Spin-2 denominator normalized to Pi_TT(0)=1."""
    z_mp = mp.mpc(z)
    return 1 + LOCAL_C2 * z_mp * F1_shape(z_mp, xi=xi, dps=dps)


def Pi_scalar(z: complex | float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Spin-0 denominator: Pi_s = 1 + 3*z*alpha_R(z,xi).

    Valid for ALL xi, including xi = 1/6 where the local R^2 coefficient
    vanishes but the nonlocal form factor is nonzero.

    NOTE (2026-04-07): Previous version force-returned 1 at xi=1/6,
    based on the incorrect assumption that the nonlocal R^2 sector
    decouples at conformal coupling. The correct formula is
    Pi_s = 1 + 3*z*alpha_R(z,xi), which gives Pi_s > 1 for all z > 0
    and all xi (the no-scalaron theorem).
    """
    z_mp = mp.mpc(z)
    # Use the universal formula: Pi_s = 1 + 3*z*alpha_R(z,xi)
    # alpha_R(z) = F2_total * 16*pi^2 (undo the 1/(16pi^2) normalization)
    alpha_R_z = F2_total_complex(z_mp, xi=xi, dps=dps) * 16 * mp.pi**2
    return 1 + 3 * z_mp * alpha_R_z


def G_TT(k2: float | complex, Lambda2: float = 1.0, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    k2_mp = mp.mpc(k2)
    if k2_mp == 0:
        raise ZeroDivisionError("massless pole at k^2 = 0")
    z = k2_mp / mp.mpf(Lambda2)
    return 1 / (k2_mp * Pi_TT(z, xi=xi, dps=dps))


def G_scalar(k2: float | complex, Lambda2: float = 1.0, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    k2_mp = mp.mpc(k2)
    if k2_mp == 0:
        raise ZeroDivisionError("massless pole at k^2 = 0")
    z = k2_mp / mp.mpf(Lambda2)
    return 1 / (k2_mp * Pi_scalar(z, xi=xi, dps=dps))


def effective_newton_kernel(z: complex | float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Newtonian kernel with spin-2 and scalar weights."""
    return 4 / (3 * Pi_TT(z, xi=xi, dps=dps)) - 1 / (3 * Pi_scalar(z, xi=xi, dps=dps))


def find_first_positive_real_tt_zero(
    *,
    z_min: float = 0.0,
    z_max: float = 10.0,
    step: float = 0.05,
    xi: float = 0.0,
    dps: int = 100,
) -> mp.mpf:
    """Locate the first positive real zero of the TT denominator."""
    mp.mp.dps = dps
    z_left = mp.mpf(z_min)
    value_left = mp.re(Pi_TT(z_left, xi=xi, dps=dps))
    z_right = z_left + mp.mpf(step)

    while z_right <= mp.mpf(z_max):
        value_right = mp.re(Pi_TT(z_right, xi=xi, dps=dps))
        if value_left == 0:
            return z_left
        if value_left * value_right < 0:
            return mp.findroot(lambda t: mp.re(Pi_TT(t, xi=xi, dps=dps)), (z_left, z_right))
        z_left = z_right
        value_left = value_right
        z_right += mp.mpf(step)

    raise ValueError(f"no positive-real TT zero found in [{z_min}, {z_max}]")


def kinetic_operator(k_vec: np.ndarray, xi: float = 0.0, Lambda2: float = 1.0, dps: int = 100) -> np.ndarray:
    """Return K_{mu nu rho sigma}(k) in a projector decomposition."""
    k_arr = np.asarray(k_vec, dtype=float)
    k2 = float(np.dot(k_arr, k_arr))
    if k2 <= 0:
        raise ValueError("k_vec must have positive Euclidean norm")
    spin2 = float(mp.re(Pi_TT(k2 / Lambda2, xi=xi, dps=dps)))
    spin0 = float(mp.re(Pi_scalar(k2 / Lambda2, xi=xi, dps=dps)))
    return k2 * spin2 * tt_projector(k_arr) + k2 * spin0 * scalar_projector(k_arr)


def check_projector_transverse(k_vec: np.ndarray, tol: float = 1e-10) -> bool:
    contracted_tt = contract_first_index_with_k(tt_projector(k_vec), k_vec)
    contracted_scalar = contract_first_index_with_k(scalar_projector(k_vec), k_vec)
    return float(np.max(np.abs(contracted_tt))) < tol and float(np.max(np.abs(contracted_scalar))) < tol


def check_gauge_invariance(k_vec: np.ndarray, xi_vec: np.ndarray | None = None, xi: float = 0.0, tol: float = 1e-10) -> bool:
    del xi  # Phase-local denominators do not affect the off-shell Einstein-gauge identity.
    return check_off_shell_gauge_invariance(k_vec, xi_vec=xi_vec, tol=tol)


def check_bianchi_identity(
    k_vec: np.ndarray,
    xi: float = 0.0,
    h_tensor: np.ndarray | None = None,
    tol: float = 1e-10,
) -> bool:
    del xi  # The off-shell linearized Bianchi identity is checked before the nonlocal gauge fixing step.
    if h_tensor is None:
        h_tensor = default_symmetric_tensor()
    return check_off_shell_bianchi_identity(k_vec, h_tensor=h_tensor, tol=tol)


def export_propagator_snapshot(
    output_path: Path | None = None,
    *,
    xi: float = 0.0,
    dps: int = 100,
) -> Path:
    if output_path is None:
        output_path = RESULTS_DIR / "nt4a_propagator_snapshot.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_points = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    payload = {
        "phase": "NT-4a",
        "xi": xi,
        "Pi_TT": {str(z): [float(mp.re(Pi_TT(z, xi=xi, dps=dps))), float(mp.im(Pi_TT(z, xi=xi, dps=dps)))] for z in z_points},
        "Pi_scalar": {str(z): [float(mp.re(Pi_scalar(z, xi=xi, dps=dps))), float(mp.im(Pi_scalar(z, xi=xi, dps=dps)))] for z in z_points},
        "scalar_mode_coefficient": float(scalar_mode_coefficient(xi)),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def generate_propagator_figures(
    *,
    xi_values: list[float] | None = None,
    dps: int = 100,
) -> dict[str, list[dict[str, float]]]:
    if xi_values is None:
        xi_values = [0.0, 1 / 6, 1.0]
    z_points = np.linspace(0.0, 10.0, 80)

    report = {
        "tt": [],
        "scalar": [],
    }

    init_style()
    fig_tt, ax_tt = create_figure(figsize=(4.8, 3.3))
    fig_scalar, ax_scalar = create_figure(figsize=(4.8, 3.3))

    for xi in xi_values:
        tt_values = [float(mp.re(Pi_TT(z, xi=xi, dps=dps))) for z in z_points]
        scalar_values = [float(mp.re(Pi_scalar(z, xi=xi, dps=dps))) for z in z_points]
        report["tt"].append({"xi": xi, "values": tt_values})
        report["scalar"].append({"xi": xi, "values": scalar_values})

        ax_tt.plot(z_points, tt_values, label=rf"$\xi = {xi:.3g}$")
        ax_scalar.plot(z_points, scalar_values, label=rf"$\xi = {xi:.3g}$")

    ax_tt.set_xlabel(r"$z = k^2 / \Lambda^2$")
    ax_tt.set_ylabel(r"$\Pi_{\mathrm{TT}}(z)$")
    ax_tt.set_title("NT-4a TT Propagator Denominator")
    ax_tt.legend()
    fig_tt.tight_layout()

    ax_scalar.set_xlabel(r"$z = k^2 / \Lambda^2$")
    ax_scalar.set_ylabel(r"$\Pi_{\mathrm{s}}(z)$")
    ax_scalar.set_title("NT-4a Scalar Mode")
    ax_scalar.legend()
    fig_scalar.tight_layout()

    save_figure(fig_tt, "nt4a_propagator", fmt="pdf", directory=FIGURES_DIR)
    save_figure(fig_scalar, "nt4a_scalar_mode", fmt="pdf", directory=FIGURES_DIR)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NT-4a propagator reference data.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt4a_propagator_snapshot.json")
    parser.add_argument("--dps", type=int, default=100)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    path = export_propagator_snapshot(args.output, xi=args.xi, dps=args.dps)
    generate_propagator_figures(dps=args.dps)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
