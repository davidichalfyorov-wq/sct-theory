"""
NT-2 entire-function analysis for SCT spectral form factors.

This module is the phase-local source of truth for complex-domain evaluations
used during Phase 4. It intentionally avoids the real-domain restrictions in
analysis/sct_tools/form_factors.py and provides standalone evaluators,
serialization helpers, and numerical diagnostics for NT-2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "nt2"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"

N_S = 4
N_F = 45
N_V = 12
N_D = mp.mpf(N_F) / 2
ALPHA_C = mp.mpf(13) / 120

SMALL_Z_THRESHOLD = mp.mpf("0.5")
DEFAULT_DPS = 100


def _set_dps(dps: int) -> None:
    if dps <= 0:
        raise ValueError(f"dps must be positive, got {dps}")
    mp.mp.dps = dps


def _to_mpc(value: complex | float | int | mp.mpf | mp.mpc) -> mp.mpc:
    return mp.mpc(value)


def phi_series_coefficient(n: int) -> mp.mpf:
    """Return the exact Taylor coefficient of phi(z)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return (-1) ** n * mp.factorial(n) / mp.factorial(2 * n + 1)


def phi_complex_mp(z: complex | float | mp.mpc, dps: int = DEFAULT_DPS) -> mp.mpc:
    """Complex-domain master function phi(z)."""
    _set_dps(dps)
    z = _to_mpc(z)
    if z == 0:
        return mp.mpc(1)
    root = mp.sqrt(z)
    return mp.e ** (-z / 4) * mp.sqrt(mp.pi / z) * mp.erfi(root / 2)


def phi_series(z: complex | float | mp.mpc, n_terms: int = 40, dps: int = DEFAULT_DPS) -> mp.mpc:
    """Evaluate phi(z) via its convergent Taylor series."""
    _set_dps(dps)
    z = _to_mpc(z)
    total = mp.mpc(0)
    for n in range(n_terms):
        total += phi_series_coefficient(n) * z**n
    return total


def _hr_scalar_series_coeff(k: int) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    a_k = phi_series_coefficient(k)
    a_k1 = phi_series_coefficient(k + 1)
    a_k2 = phi_series_coefficient(k + 2)
    coeff_a = a_k / 32 + a_k1 / 8 + 5 * a_k2 / 24
    coeff_b = -a_k / 4 - a_k1 / 2
    coeff_c = a_k / 2
    return coeff_a, coeff_b, coeff_c


def _series_sum(coeffs: list[mp.mpf], z: mp.mpc) -> mp.mpc:
    total = mp.mpc(0)
    for n, coeff in enumerate(coeffs):
        total += coeff * z**n
    return total


def hC_scalar_complex(z: complex | float | mp.mpc, dps: int = DEFAULT_DPS, n_terms: int = 32) -> mp.mpc:
    _set_dps(dps)
    z = _to_mpc(z)
    if abs(z) < SMALL_Z_THRESHOLD:
        coeffs = [phi_series_coefficient(k + 2) / 2 for k in range(n_terms)]
        return _series_sum(coeffs, z)
    phi_val = phi_complex_mp(z, dps=dps)
    return 1 / (12 * z) + (phi_val - 1) / (2 * z**2)


def hR_scalar_complex(
    z: complex | float | mp.mpc,
    xi: float | mp.mpf = 0.0,
    dps: int = DEFAULT_DPS,
    n_terms: int = 32,
) -> mp.mpc:
    _set_dps(dps)
    z = _to_mpc(z)
    xi_mp = mp.mpf(xi)
    if abs(z) < SMALL_Z_THRESHOLD:
        coeffs = []
        for k in range(n_terms):
            coeff_a, coeff_b, coeff_c = _hr_scalar_series_coeff(k)
            coeffs.append(coeff_a + xi_mp * coeff_b + xi_mp**2 * coeff_c)
        return _series_sum(coeffs, z)
    phi_val = phi_complex_mp(z, dps=dps)
    f_ric = 1 / (6 * z) + (phi_val - 1) / z**2
    f_r = phi_val / 32 + phi_val / (8 * z) - mp.mpf(7) / (48 * z) - (phi_val - 1) / (8 * z**2)
    f_ru = -phi_val / 4 - (phi_val - 1) / (2 * z)
    f_u = phi_val / 2
    return f_ric / 3 + f_r + xi_mp * f_ru + xi_mp**2 * f_u


def hC_dirac_complex(z: complex | float | mp.mpc, dps: int = DEFAULT_DPS, n_terms: int = 32) -> mp.mpc:
    _set_dps(dps)
    z = _to_mpc(z)
    if abs(z) < SMALL_Z_THRESHOLD:
        coeffs = [phi_series_coefficient(k + 1) / 2 + 2 * phi_series_coefficient(k + 2) for k in range(n_terms)]
        return _series_sum(coeffs, z)
    phi_val = phi_complex_mp(z, dps=dps)
    return (3 * phi_val - 1) / (6 * z) + 2 * (phi_val - 1) / z**2


def hR_dirac_complex(z: complex | float | mp.mpc, dps: int = DEFAULT_DPS, n_terms: int = 32) -> mp.mpc:
    _set_dps(dps)
    z = _to_mpc(z)
    if abs(z) < SMALL_Z_THRESHOLD:
        coeffs = [phi_series_coefficient(k + 1) / 12 + 5 * phi_series_coefficient(k + 2) / 6 for k in range(n_terms)]
        return _series_sum(coeffs, z)
    phi_val = phi_complex_mp(z, dps=dps)
    return (3 * phi_val + 2) / (36 * z) + 5 * (phi_val - 1) / (6 * z**2)


def hC_vector_complex(z: complex | float | mp.mpc, dps: int = DEFAULT_DPS, n_terms: int = 32) -> mp.mpc:
    _set_dps(dps)
    z = _to_mpc(z)
    if abs(z) < SMALL_Z_THRESHOLD:
        coeffs = [phi_series_coefficient(k) / 4 + phi_series_coefficient(k + 1) + phi_series_coefficient(k + 2) for k in range(n_terms)]
        return _series_sum(coeffs, z)
    phi_val = phi_complex_mp(z, dps=dps)
    return phi_val / 4 + (6 * phi_val - 5) / (6 * z) + (phi_val - 1) / z**2


def hR_vector_complex(z: complex | float | mp.mpc, dps: int = DEFAULT_DPS, n_terms: int = 32) -> mp.mpc:
    _set_dps(dps)
    z = _to_mpc(z)
    if abs(z) < SMALL_Z_THRESHOLD:
        coeffs = [-phi_series_coefficient(k) / 48 - phi_series_coefficient(k + 1) / 12 + 5 * phi_series_coefficient(k + 2) / 12 for k in range(n_terms)]
        return _series_sum(coeffs, z)
    phi_val = phi_complex_mp(z, dps=dps)
    return -phi_val / 48 + (11 - 6 * phi_val) / (72 * z) + 5 * (phi_val - 1) / (12 * z**2)


def F1_total_complex(
    z: complex | float | mp.mpc,
    xi: float | mp.mpf = 0.0,
    dps: int = DEFAULT_DPS,
    N_s: int = N_S,
    N_f: int = N_F,
    N_v: int = N_V,
) -> mp.mpc:
    _set_dps(dps)
    n_dirac = mp.mpf(N_f) / 2
    numerator = (
        mp.mpf(N_s) * hC_scalar_complex(z, dps=dps)
        + n_dirac * hC_dirac_complex(z, dps=dps)
        + mp.mpf(N_v) * hC_vector_complex(z, dps=dps)
    )
    return numerator / (16 * mp.pi**2)


def F2_total_complex(
    z: complex | float | mp.mpc,
    xi: float | mp.mpf = 0.0,
    dps: int = DEFAULT_DPS,
    N_s: int = N_S,
    N_f: int = N_F,
    N_v: int = N_V,
) -> mp.mpc:
    _set_dps(dps)
    n_dirac = mp.mpf(N_f) / 2
    numerator = (
        mp.mpf(N_s) * hR_scalar_complex(z, xi=xi, dps=dps)
        + n_dirac * hR_dirac_complex(z, dps=dps)
        + mp.mpf(N_v) * hR_vector_complex(z, dps=dps)
    )
    return numerator / (16 * mp.pi**2)


def alpha_R(xi: float | mp.mpf) -> mp.mpf:
    xi_mp = mp.mpf(xi)
    return 2 * (xi_mp - mp.mpf(1) / 6) ** 2


@dataclass(frozen=True)
class PoleCancellationRecord:
    name: str
    local_limit: complex
    series_value: complex
    absolute_error: float


def pole_cancellation_report(xi: float = 0.0, dps: int = DEFAULT_DPS) -> list[PoleCancellationRecord]:
    _set_dps(dps)
    sample = mp.mpf("1e-12")
    records: list[PoleCancellationRecord] = []
    evaluators = [
        ("hC_scalar", lambda z: hC_scalar_complex(z, dps=dps), mp.mpf(1) / 120),
        ("hR_scalar", lambda z: hR_scalar_complex(z, xi=xi, dps=dps), (mp.mpf(xi) - mp.mpf(1) / 6) ** 2 / 2),
        ("hC_dirac", lambda z: hC_dirac_complex(z, dps=dps), -mp.mpf(1) / 20),
        ("hR_dirac", lambda z: hR_dirac_complex(z, dps=dps), mp.mpf(0)),
        ("hC_vector", lambda z: hC_vector_complex(z, dps=dps), mp.mpf(1) / 10),
        ("hR_vector", lambda z: hR_vector_complex(z, dps=dps), mp.mpf(0)),
        ("F1_total", lambda z: F1_total_complex(z, dps=dps), ALPHA_C / (16 * mp.pi**2)),
        ("F2_total", lambda z: F2_total_complex(z, xi=xi, dps=dps), alpha_R(xi) / (16 * mp.pi**2)),
    ]
    for name, func, expected in evaluators:
        series_value = func(sample)
        records.append(
            PoleCancellationRecord(
                name=name,
                local_limit=complex(expected),
                series_value=complex(series_value),
                absolute_error=float(abs(series_value - expected)),
            )
        )
    return records


def sample_modulus_on_circles(
    func,
    radii: list[float],
    angles: list[float],
    *,
    dps: int = DEFAULT_DPS,
) -> list[dict[str, float]]:
    _set_dps(dps)
    samples = []
    for radius in radii:
        values = []
        for angle in angles:
            z = mp.mpf(radius) * mp.e ** (1j * mp.mpf(angle))
            values.append(abs(func(z)))
        max_value = max(values)
        samples.append({"radius": float(radius), "max_modulus": float(max_value)})
    return samples


def estimate_growth_rate(
    func,
    radii: list[float] | None = None,
    angles: list[float] | None = None,
    *,
    dps: int = DEFAULT_DPS,
) -> dict[str, float | list[dict[str, float]]]:
    """
    Estimate finite-radius growth proxies from circle maxima.

    The returned ``order`` and ``type`` are effective finite-radius proxies,
    not standalone proofs of the exact entire-function order/type. For NT-2
    those exact asymptotics are controlled analytically by the closed-form
    master function ``phi(z)``.
    """
    if radii is None:
        radii = [50.0, 100.0, 200.0, 400.0, 800.0]
    if angles is None:
        angles = [0.0, mp.pi / 4, mp.pi / 2, 3 * mp.pi / 4, mp.pi]

    samples = sample_modulus_on_circles(func, radii, angles, dps=dps)
    filtered = [sample for sample in samples if sample["max_modulus"] > 1.0000001]
    if len(filtered) < 2:
        raise ValueError("not enough growth samples above unity to estimate order")

    xs = [mp.log(sample["radius"]) for sample in filtered]
    ys = [mp.log(mp.log(sample["max_modulus"])) for sample in filtered]
    n = mp.mpf(len(filtered))
    sum_x = mp.fsum(xs)
    sum_y = mp.fsum(ys)
    sum_xx = mp.fsum(x * x for x in xs)
    sum_xy = mp.fsum(x * y for x, y in zip(xs, ys, strict=True))
    denom = n * sum_xx - sum_x**2
    order = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else mp.nan

    type_samples = [sample["max_modulus"] for sample in filtered]
    type_values = [mp.log(value) / radius for value, radius in zip(type_samples, [sample["radius"] for sample in filtered], strict=True)]
    return {
        "order": float(order),
        "type": float(max(type_values)),
        "order_proxy": float(order),
        "type_proxy": float(max(type_values)),
        "analytic_order_claim": 1.0,
        "samples": samples,
    }


def find_real_axis_zeros(
    func,
    interval: tuple[float, float] = (-200.0, 200.0),
    *,
    n_samples: int = 800,
    dps: int = DEFAULT_DPS,
) -> list[float]:
    """Find simple real-axis zeros of a real-valued function."""
    _set_dps(dps)
    a, b = interval
    if a >= b:
        raise ValueError(f"invalid interval {interval}")
    xs = [a + (b - a) * i / n_samples for i in range(n_samples + 1)]
    values = [mp.re(func(x)) for x in xs]
    roots: list[float] = []

    for idx in range(n_samples):
        left_x = xs[idx]
        right_x = xs[idx + 1]
        left_v = values[idx]
        right_v = values[idx + 1]
        if left_v == 0:
            roots.append(float(left_x))
            continue
        if left_v * right_v > 0:
            continue
        try:
            root = mp.findroot(lambda t: func(t), (left_x, right_x))
        except (ValueError, ZeroDivisionError):
            continue
        root_f = float(mp.re(root))
        if not any(abs(root_f - existing) < 1e-6 for existing in roots):
            roots.append(root_f)
    return sorted(roots)


def serialize_nt2_snapshot(
    output_path: Path | None = None,
    *,
    xi: float = 0.0,
    dps: int = DEFAULT_DPS,
) -> Path:
    """Write machine-readable NT-2 reference data for later review steps."""
    _set_dps(dps)
    if output_path is None:
        output_path = RESULTS_DIR / "nt2_snapshot.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": "NT-2",
        "xi": xi,
        "dps": dps,
        "pole_cancellation": [
            {
                "name": record.name,
                "local_limit": [record.local_limit.real, record.local_limit.imag],
                "series_value": [record.series_value.real, record.series_value.imag],
                "absolute_error": record.absolute_error,
            }
            for record in pole_cancellation_report(xi=xi, dps=dps)
        ],
        "growth_F1": estimate_growth_rate(lambda z: F1_total_complex(z, xi=xi, dps=dps), dps=dps),
        "growth_F2": estimate_growth_rate(lambda z: F2_total_complex(z, xi=xi, dps=dps), dps=dps),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NT-2 complex-domain reference data.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=DEFAULT_DPS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt2_snapshot.json")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    path = serialize_nt2_snapshot(output_path=args.output, xi=args.xi, dps=args.dps)
    print(f"Wrote NT-2 snapshot to {path}")


if __name__ == "__main__":
    main()
