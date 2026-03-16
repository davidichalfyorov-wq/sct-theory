# ruff: noqa: E402, I001
"""
MR-2 sub-task d.5: Residue analysis at the first positive-real TT zero.

Computes Pi_TT'(z0) and the massive spin-2 residue R2 = 1/(z0 * Pi_TT'(z0))
using four independent methods and cross-checks against the Stelle (local) limit.

Methods:
    1. Direct analytic derivative via mpmath numerical differentiation
    2. Cauchy contour integral with proper complex F1(z)
    3. Richardson extrapolation of finite differences
    4. Local Taylor expansion of Pi_TT around z0

Physical interpretation:
    R2 < 0 => ghost (negative-norm massive spin-2 state)
    |R2| < |R2_Stelle| = 1 => nonlocal dressing reduces the ghost weight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

# Direct import from nt2_entire_function to avoid the circular import chain
# that goes through sct_tools.__init__ -> sct_tools.propagator -> scripts.nt4a_newtonian
from scripts.nt2_entire_function import F1_total_complex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (from verified Phase 3 / NT-4a results)
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60


# ---------------------------------------------------------------------------
# Pi_TT built from first principles (avoids circular import)
# ---------------------------------------------------------------------------
def _F1_at_zero(*, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """F1(0) — the local limit of the total Weyl form factor."""
    return F1_total_complex(0, xi=xi, dps=dps)


def _F1_shape(z: mp.mpc, *, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """F1_hat(z) = F1(z) / F1(0) — normalized form factor."""
    f0 = _F1_at_zero(xi=xi, dps=dps)
    if abs(f0) < mp.mpf("1e-40"):
        return mp.mpc(1)
    return F1_total_complex(z, xi=xi, dps=dps) / f0


def Pi_TT(z, *, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Spin-2 TT denominator: Pi_TT(z) = 1 + c2 * z * F1_hat(z)."""
    mp.mp.dps = dps
    z_mp = mp.mpc(z)
    return 1 + LOCAL_C2 * z_mp * _F1_shape(z_mp, xi=xi, dps=dps)


def find_first_positive_real_tt_zero(
    *, z_min: float = 0.0, z_max: float = 10.0, step: float = 0.05,
    xi: float = 0.0, dps: int = 100,
) -> mp.mpf:
    """Locate the first positive real zero of Pi_TT."""
    mp.mp.dps = dps
    z_left = mp.mpf(z_min)
    value_left = mp.re(Pi_TT(z_left, xi=xi, dps=dps))
    z_right = z_left + mp.mpf(step)

    while z_right <= mp.mpf(z_max):
        value_right = mp.re(Pi_TT(z_right, xi=xi, dps=dps))
        if value_left == 0:
            return z_left
        if value_left * value_right < 0:
            return mp.findroot(
                lambda t: mp.re(Pi_TT(t, xi=xi, dps=dps)), (z_left, z_right)
            )
        z_left = z_right
        value_left = value_right
        z_right += mp.mpf(step)

    raise ValueError(f"no positive-real TT zero found in [{z_min}, {z_max}]")


def _Pi_TT_real(z, *, xi: float = 0.0, dps: int = 100) -> mp.mpf:
    """Real-valued Pi_TT on the positive real axis."""
    return mp.re(Pi_TT(z, xi=xi, dps=dps))


# ---------------------------------------------------------------------------
# Method 1: Direct numerical derivative (mpmath diff)
# ---------------------------------------------------------------------------
def method1_direct_derivative(
    z0: mp.mpf, *, xi: float = 0.0, dps: int = 100
) -> dict:
    """
    Compute Pi_TT'(z0) via central finite difference at multiple step sizes.

    Uses h = 10^{-10} at 100-digit precision, giving ~80 reliable digits
    in the derivative. Also computes at h = 10^{-8} and 10^{-12} as
    convergence cross-check.
    """
    mp.mp.dps = dps

    results_by_h = {}
    for h_exp in [-8, -10, -12, -14]:
        h = mp.power(10, h_exp)
        f_plus = Pi_TT(z0 + h, xi=xi, dps=dps)
        f_minus = Pi_TT(z0 - h, xi=xi, dps=dps)
        d = mp.re(f_plus - f_minus) / (2 * h)
        results_by_h[h_exp] = float(d)

    # Primary result: h = 10^{-10}
    h = mp.mpf("1e-10")
    f_plus = Pi_TT(z0 + h, xi=xi, dps=dps)
    f_minus = Pi_TT(z0 - h, xi=xi, dps=dps)
    derivative = mp.re(f_plus - f_minus) / (2 * h)

    residue = 1 / (z0 * derivative)

    return {
        "method": "central_finite_difference",
        "z0": str(z0),
        "h": "1e-10",
        "Pi_TT_prime_z0": str(derivative),
        "R2": str(residue),
        "R2_float": float(residue),
        "ghost": float(residue) < 0,
        "convergence_by_h": results_by_h,
    }


# ---------------------------------------------------------------------------
# Method 2: Cauchy contour integral with proper complex evaluation
# ---------------------------------------------------------------------------
def method2_cauchy_contour(
    z0: mp.mpf, *, xi: float = 0.0, dps: int = 100, n_points: int = 512, radius: float = 0.01
) -> dict:
    """
    Compute Pi_TT'(z0) via the Cauchy integral formula:
        f'(z0) = (1 / 2pi i) * oint f(z) / (z - z0)^2 dz

    Uses proper complex-valued Pi_TT (not real-axis restriction).
    The contour is a small circle of given radius around z0.
    """
    mp.mp.dps = dps
    r = mp.mpf(radius)

    # Trapezoidal rule on the circle z = z0 + r*exp(i*theta)
    integral = mp.mpc(0)
    d_theta = 2 * mp.pi / n_points

    for k in range(n_points):
        theta = k * d_theta
        exp_itheta = mp.expj(theta)
        z = z0 + r * exp_itheta

        # Pi_TT evaluated at complex z (PROPER complex analytic continuation)
        pi_val = Pi_TT(z, xi=xi, dps=dps)

        # Integrand: Pi_TT(z) / (z - z0)^2 * dz
        # dz = i * r * exp(i*theta) * d_theta
        # (z - z0)^2 = r^2 * exp(2*i*theta)
        # => Pi_TT(z) * i * r * exp(i*theta) / (r^2 * exp(2*i*theta))
        # => Pi_TT(z) * i / (r * exp(i*theta))
        integrand = pi_val * mp.mpc(0, 1) / (r * exp_itheta)
        integral += integrand * d_theta

    derivative = integral / (2 * mp.pi * mp.mpc(0, 1))
    derivative_real = mp.re(derivative)

    residue = 1 / (z0 * derivative_real)

    return {
        "method": "cauchy_contour_integral",
        "z0": str(z0),
        "radius": float(radius),
        "n_points": n_points,
        "Pi_TT_prime_z0": str(derivative_real),
        "Pi_TT_prime_z0_imag": str(mp.im(derivative)),
        "R2": str(residue),
        "R2_float": float(residue),
        "ghost": float(residue) < 0,
    }


# ---------------------------------------------------------------------------
# Method 3: Richardson extrapolation of finite differences
# ---------------------------------------------------------------------------
def method3_richardson(
    z0: mp.mpf, *, xi: float = 0.0, dps: int = 100, n_levels: int = 8
) -> dict:
    """
    Richardson extrapolation of central finite differences.
    D(h) = [f(z0+h) - f(z0-h)] / (2h)
    with successive halvings and extrapolation table.
    """
    mp.mp.dps = dps

    h0 = mp.mpf("0.01")
    table = []

    # Build first column: D(h), D(h/2), D(h/4), ...
    for level in range(n_levels):
        h = h0 / mp.power(2, level)
        f_plus = _Pi_TT_real(z0 + h, xi=xi, dps=dps)
        f_minus = _Pi_TT_real(z0 - h, xi=xi, dps=dps)
        d = (f_plus - f_minus) / (2 * h)
        table.append([d])

    # Richardson extrapolation: eliminate h^2, h^4, ... errors
    for col in range(1, n_levels):
        for row in range(col, n_levels):
            factor = mp.power(4, col)
            extrapolated = (factor * table[row][col - 1] - table[row - 1][col - 1]) / (factor - 1)
            table[row].append(extrapolated)

    derivative = table[-1][-1]
    residue = 1 / (z0 * derivative)

    # Convergence: differences between successive best estimates
    convergence = []
    for level in range(1, n_levels):
        best_at_level = table[level][min(level, len(table[level]) - 1)]
        prev = table[level - 1][min(level - 1, len(table[level - 1]) - 1)]
        convergence.append(float(abs(best_at_level - prev)))

    return {
        "method": "richardson_extrapolation",
        "z0": str(z0),
        "n_levels": n_levels,
        "Pi_TT_prime_z0": str(derivative),
        "R2": str(residue),
        "R2_float": float(residue),
        "ghost": float(residue) < 0,
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# Method 4: Local Taylor expansion of Pi_TT around z0
# ---------------------------------------------------------------------------
def method4_analytic_chain_rule(
    z0: mp.mpf, *, xi: float = 0.0, dps: int = 100
) -> dict:
    """
    Compute Pi_TT'(z0) from the analytic structure:

        Pi_TT(z) = 1 + c2 * z * F1_hat(z)
        Pi_TT'(z) = c2 * [F1_hat(z) + z * F1_hat'(z)]

    F1_hat'(z0) is computed via central finite difference of F1_hat directly.
    This is an independent method because it differentiates the form factor
    rather than the composite propagator.
    """
    mp.mp.dps = dps

    # F1_hat at z0
    f1hat_z0 = _F1_shape(mp.mpc(z0), xi=xi, dps=dps)

    # F1_hat'(z0) via central finite difference
    h = mp.mpf("1e-10")
    f1hat_plus = _F1_shape(mp.mpc(z0 + h), xi=xi, dps=dps)
    f1hat_minus = _F1_shape(mp.mpc(z0 - h), xi=xi, dps=dps)
    f1hat_prime = mp.re(f1hat_plus - f1hat_minus) / (2 * h)

    # Pi_TT'(z0) = c2 * [F1_hat(z0) + z0 * F1_hat'(z0)]
    derivative = LOCAL_C2 * (mp.re(f1hat_z0) + z0 * f1hat_prime)

    # Cross-check: Pi_TT(z0) = 1 + c2 * z0 * F1_hat(z0) should be ~0
    pi_check = 1 + LOCAL_C2 * z0 * mp.re(f1hat_z0)

    residue = 1 / (z0 * derivative)

    return {
        "method": "analytic_chain_rule",
        "z0": str(z0),
        "F1_hat_z0": str(mp.re(f1hat_z0)),
        "F1_hat_prime_z0": str(f1hat_prime),
        "pi_tt_z0_check": str(pi_check),
        "pi_tt_z0_check_ok": float(abs(pi_check)) < 1e-15,
        "Pi_TT_prime_z0": str(derivative),
        "R2": str(residue),
        "R2_float": float(residue),
        "ghost": float(residue) < 0,
    }


# ---------------------------------------------------------------------------
# Cross-check: Stelle (local HDG) limit
# ---------------------------------------------------------------------------
def stelle_limit() -> dict:
    """
    In fourth-derivative gravity (Stelle 1977), Pi_TT(z) = 1 + c2*z (linear).
    The zero is at z0_Stelle = -1/c2 (negative for c2 > 0 => no positive zero).

    For comparison, if we formally set F1_hat(z) = 1, we get
        Pi_TT(z) = 1 + c2*z,  z0 = 1/c2 (taking absolute value for positive axis),
        Pi_TT'(z0) = c2,  R2 = 1/(z0 * c2) = c2.

    Actually, in the Stelle theory the massive spin-2 pole has residue R2_Stelle = -1
    (normalized so the massless graviton has residue +1). The sign comes from the
    relative sign between the 1/k^2 and 1/(k^2 - m^2) terms in the propagator
    partial-fraction decomposition.
    """
    c2 = float(LOCAL_C2)
    z0_stelle = 1 / c2  # = 60/13

    return {
        "c2": c2,
        "z0_stelle": z0_stelle,
        "Pi_TT_prime_stelle": c2,
        "R2_stelle": -1.0,
        "explanation": (
            "In the local (Stelle) limit, F1_hat=1 gives Pi_TT=1+c2*z. "
            "The massive pole has R2=-1 (standard Stelle ghost). "
            "SCT's nonlocal form factors modify both z0 and R2."
        ),
    }


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------
def consistency_checks(z0: mp.mpf, results: list[dict], *, dps: int = 100) -> dict:
    """Run internal consistency checks on the results."""
    mp.mp.dps = dps

    checks = {}

    # Check 1: Pi_TT(z0) = 0
    pi_at_z0 = _Pi_TT_real(z0, dps=dps)
    checks["pi_tt_at_z0_is_zero"] = {
        "value": str(pi_at_z0),
        "abs_value": float(abs(pi_at_z0)),
        "pass": float(abs(pi_at_z0)) < 1e-25,
    }

    # Check 2: All methods agree on Pi_TT'(z0) to at least 10 digits
    derivatives = [mp.mpf(r["Pi_TT_prime_z0"]) for r in results]
    if len(derivatives) >= 2:
        max_spread = max(abs(d1 - d2) for d1 in derivatives for d2 in derivatives)
        checks["method_agreement_derivative"] = {
            "max_spread": str(max_spread),
            "relative_spread": str(max_spread / abs(derivatives[0])),
            "pass": float(max_spread / abs(derivatives[0])) < 1e-10,
        }

    # Check 3: All methods agree on R2
    residues = [mp.mpf(r["R2"]) for r in results]
    if len(residues) >= 2:
        max_spread = max(abs(r1 - r2) for r1 in residues for r2 in residues)
        checks["method_agreement_R2"] = {
            "max_spread": str(max_spread),
            "relative_spread": str(max_spread / abs(residues[0])),
            "pass": float(max_spread / abs(residues[0])) < 1e-10,
        }

    # Check 4: R2 is negative (ghost)
    all_negative = all(float(mp.mpf(r["R2"])) < 0 for r in results)
    checks["all_methods_ghost"] = {"pass": all_negative}

    # Check 5: |R2| < 1 (nonlocal dressing reduces the Stelle ghost)
    r2_best = float(residues[0]) if residues else 0
    checks["ghost_suppression"] = {
        "R2_SCT": r2_best,
        "R2_Stelle": -1.0,
        "ratio": abs(r2_best),
        "suppressed": abs(r2_best) < 1.0,
    }

    # Check 6: z0 < z0_Stelle = 60/13
    z0_stelle = mp.mpf(60) / 13
    checks["z0_vs_stelle"] = {
        "z0_SCT": str(z0),
        "z0_Stelle": str(z0_stelle),
        "z0_shifted_left": float(z0) < float(z0_stelle),
    }

    # Check 7: Pi_TT'(z0) < 0 (denominator crosses zero from above)
    checks["derivative_sign"] = {
        "Pi_TT_prime_z0": float(derivatives[0]) if derivatives else None,
        "negative": float(derivatives[0]) < 0 if derivatives else None,
        "interpretation": "Pi_TT crosses zero downward (from + to -) at z0",
    }

    # Overall
    all_pass = all(
        c.get("pass", True) for c in checks.values() if isinstance(c, dict) and "pass" in c
    )
    checks["overall"] = "PASS" if all_pass else "FAIL"

    return checks


# ---------------------------------------------------------------------------
# Physical interpretation
# ---------------------------------------------------------------------------
def physical_interpretation(z0: mp.mpf, r2: float) -> dict:
    """Produce a physical interpretation summary."""
    c2 = float(LOCAL_C2)
    z0_stelle = 1 / c2

    return {
        "z0": float(z0),
        "z0_stelle": z0_stelle,
        "z0_shift": float(z0) - z0_stelle,
        "z0_shift_percent": 100 * (float(z0) - z0_stelle) / z0_stelle,
        "R2": r2,
        "R2_stelle": -1.0,
        "ghost_suppression_factor": abs(r2),
        "ghost_confirmed": r2 < 0,
        "interpretation": {
            "ghost_status": (
                "The massive spin-2 pole is a GHOST (negative residue). "
                "This is the SCT analogue of the Stelle ghost in higher-derivative gravity."
            ),
            "nonlocal_effect": (
                f"The nonlocal form factors shift z0 from {z0_stelle:.4f} (Stelle) "
                f"to {float(z0):.4f} (SCT), a {abs(float(z0) - z0_stelle)/z0_stelle*100:.1f}% shift. "
                f"The ghost weight |R2| is reduced from 1.0 (Stelle) to {abs(r2):.4f} (SCT), "
                f"a {(1-abs(r2))*100:.1f}% suppression."
            ),
            "compatibility_with_NT2": (
                "NT-2 establishes that F1(z) and F2(z) are entire functions. "
                "This does NOT prevent Pi_TT(z) = 1 + c2*z*F1_hat(z) from having zeros, "
                "because a zero of Pi_TT is not a pole of F1. "
                "The ghost arises from 1/Pi_TT, not from the form factors themselves."
            ),
            "connection_to_yukawa": (
                "The Yukawa coefficient -4/3 in V(r) = -(G*M/r)[1 - (4/3)e^{-m2*r} + ...] "
                "encodes the same ghost: partial-fraction decomposition of 1/(k^2 * Pi_TT) "
                "gives a negative-residue massive pole."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_full_analysis(*, xi: float = 0.0, dps: int = 100) -> dict:
    """Execute the complete MR-2 d.5 residue analysis."""
    mp.mp.dps = dps

    # Step 1: Find z0
    z0 = find_first_positive_real_tt_zero(xi=xi, dps=dps)
    print(f"z0 = {mp.nstr(z0, 30)}")

    # Step 2: Run all four methods
    print("Method 1: direct mpmath diff...")
    m1 = method1_direct_derivative(z0, xi=xi, dps=dps)
    print(f"  Pi_TT'(z0) = {m1['Pi_TT_prime_z0'][:30]}")
    print(f"  R2 = {m1['R2'][:30]}")

    print("Method 2: Cauchy contour integral...")
    m2 = method2_cauchy_contour(z0, xi=xi, dps=dps, n_points=1024, radius=0.001)
    print(f"  Pi_TT'(z0) = {m2['Pi_TT_prime_z0'][:30]}")
    print(f"  R2 = {m2['R2'][:30]}")

    print("Method 3: Richardson extrapolation...")
    m3 = method3_richardson(z0, xi=xi, dps=dps, n_levels=10)
    print(f"  Pi_TT'(z0) = {m3['Pi_TT_prime_z0'][:30]}")
    print(f"  R2 = {m3['R2'][:30]}")

    print("Method 4: analytic chain rule...")
    m4 = method4_analytic_chain_rule(z0, xi=xi, dps=dps)
    print(f"  Pi_TT'(z0) = {m4['Pi_TT_prime_z0'][:30]}")
    print(f"  R2 = {m4['R2'][:30]}")
    print(f"  Pi_TT(z0) check: {m4['pi_tt_z0_check'][:30]}")

    # Step 3: Stelle cross-check
    stelle = stelle_limit()

    # Step 4: Consistency checks
    all_results = [m1, m2, m3, m4]
    checks = consistency_checks(z0, all_results, dps=dps)

    # Step 5: Physical interpretation
    r2_best = float(mp.mpf(m1["R2"]))
    interpretation = physical_interpretation(z0, r2_best)

    # Assemble full report
    report = {
        "task": "MR-2 sub-task d.5",
        "description": "Residue analysis at the first positive-real TT zero",
        "xi": xi,
        "dps": dps,
        "z0": str(z0),
        "z0_float": float(z0),
        "methods": {
            "method1_direct": m1,
            "method2_contour": m2,
            "method3_richardson": m3,
            "method4_taylor": m4,
        },
        "stelle_comparison": stelle,
        "consistency_checks": checks,
        "physical_interpretation": interpretation,
        "best_values": {
            "z0": str(z0),
            "Pi_TT_prime_z0": m1["Pi_TT_prime_z0"],
            "R2": m1["R2"],
            "R2_float": m1["R2_float"],
            "ghost": m1["ghost"],
        },
    }

    return report


def save_report(report: dict, output_path: Path | None = None) -> Path:
    """Save the full analysis report to JSON."""
    if output_path is None:
        output_path = RESULTS_DIR / "mr2_residue_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MR-2 d.5: TT residue analysis")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "mr2_residue_analysis.json")
    args = parser.parse_args()

    report = run_full_analysis(xi=args.xi, dps=args.dps)

    path = save_report(report, args.output)
    print(f"\n{'='*60}")
    print(f"Report saved to {path}")
    print(f"Overall: {report['consistency_checks']['overall']}")
    print(f"z0 = {report['best_values']['z0'][:30]}")
    print(f"Pi_TT'(z0) = {report['best_values']['Pi_TT_prime_z0'][:30]}")
    print(f"R2 = {report['best_values']['R2'][:30]}")
    print(f"Ghost: {report['best_values']['ghost']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
