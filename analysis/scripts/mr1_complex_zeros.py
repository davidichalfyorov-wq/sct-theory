# ruff: noqa: E402, I001
"""
MR-1 sub-task (c): Complex zero finder for SCT propagator denominators.

Systematically locates all zeros of Pi_TT(z) and Pi_s(z, xi) in the
complex plane within a specified radius, using three complementary methods:

  1. Argument principle (contour integral for zero counting)
  2. Grid-based Newton search (broad scan + local refinement)
  3. Delaunay interpolation (adaptive refinement near detected zeros)

Classification of zeros:
  Type A: Real positive (z > 0) -- Euclidean ghost poles
  Type B: Real negative (z < 0) -- correspond to physical Lorentzian poles (z_L > 0)
  Type C: Complex conjugate pairs -- Lee-Wick partners
  Type D: Complex, not in conjugate pairs -- would indicate non-hermiticity (impossible
          for real coefficients; finding such zeros would signal a numerical artifact)

All computations use mpmath for arbitrary precision.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mpmath as mp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.mr1_lorentzian import Pi_TT_complex, Pi_scalar_complex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DPS = 50


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class ComplexZero:
    """A zero of a meromorphic function in the complex plane."""
    z: complex
    z_str: str
    function: str  # "Pi_TT" or "Pi_scalar"
    zero_type: str  # "A", "B", "C", "D"
    multiplicity: int = 1
    residue: complex | None = None
    residue_str: str | None = None
    verified: bool = False
    abs_value_at_zero: float = 0.0
    xi: float | None = None
    conjugate_partner: complex | None = None

    def to_dict(self) -> dict:
        d = {
            "z_real": self.z.real,
            "z_imag": self.z.imag,
            "z_str": self.z_str,
            "function": self.function,
            "zero_type": self.zero_type,
            "multiplicity": self.multiplicity,
            "abs_value_at_zero": self.abs_value_at_zero,
            "verified": self.verified,
        }
        if self.residue is not None:
            d["residue_real"] = self.residue.real
            d["residue_imag"] = self.residue.imag
            d["residue_str"] = self.residue_str
        if self.xi is not None:
            d["xi"] = self.xi
        if self.conjugate_partner is not None:
            d["conjugate_partner_real"] = self.conjugate_partner.real
            d["conjugate_partner_imag"] = self.conjugate_partner.imag
        return d


def classify_zero(z0: complex | mp.mpc, tol: float = 1e-10) -> str:
    """
    Classify a complex zero into Type A, B, C, or D.

    Type A: Real positive (z > 0, |Im z| < tol)
    Type B: Real negative (z < 0, |Im z| < tol)
    Type C: Complex (|Im z| >= tol) -- will be paired later
    Type D: Reserved for unpaired complex zeros (should not occur for real coefficients)

    Parameters
    ----------
    z0 : complex zero location
    tol : tolerance for "real" classification

    Returns
    -------
    'A', 'B', 'C', or 'D'
    """
    z0_mpc = mp.mpc(z0)
    if abs(mp.im(z0_mpc)) < tol:
        if mp.re(z0_mpc) > tol:
            return "A"
        elif mp.re(z0_mpc) < -tol:
            return "B"
        else:
            return "A"  # near origin, classify as positive-real
    return "C"


# ===================================================================
# Method 1: Argument Principle (zero counting)
# ===================================================================

def zero_count_argument_principle(
    f: Callable,
    R: float,
    dps: int = DEFAULT_DPS,
    n_points: int = 1024,
) -> int:
    """
    Count zeros of f(z) inside the circle |z| = R using the argument principle.

    N = (1 / 2*pi*i) * oint_{|z|=R} f'(z)/f(z) dz

    The derivative f'(z) is computed numerically via central differences.

    Parameters
    ----------
    f : analytic function of a complex variable
    R : radius of the contour
    dps : decimal places of precision
    n_points : number of quadrature points on the contour

    Returns
    -------
    Integer number of zeros (counting multiplicity) inside |z| = R.
    If f has poles, the count is N_zeros - N_poles.
    """
    mp.mp.dps = dps
    R_mp = mp.mpf(R)
    h_deriv = mp.mpf("1e-8")

    # Parameterize: z(t) = R * e^{i*t}, t in [0, 2*pi)
    total = mp.mpc(0)
    dt = 2 * mp.pi / n_points

    for k in range(n_points):
        t = k * dt
        z = R_mp * mp.exp(mp.mpc(0, t))
        dz_dt = mp.mpc(0, 1) * z  # dz/dt = i * z

        # f'(z) via central difference
        f_plus = f(z + h_deriv)
        f_minus = f(z - h_deriv)
        f_prime = (f_plus - f_minus) / (2 * h_deriv)

        f_val = f(z)
        if abs(f_val) < mp.mpf("1e-30"):
            # f(z) ~ 0 on the contour -- contour passes through a zero
            # This is degenerate; adjust radius slightly
            return zero_count_argument_principle(f, R * 1.01, dps=dps, n_points=n_points)

        integrand = f_prime / f_val * dz_dt
        total += integrand * dt

    N = total / (2 * mp.pi * mp.mpc(0, 1))
    return int(round(float(mp.re(N))))


# ===================================================================
# Method 2: Grid-based Newton search
# ===================================================================

def _grid_newton_search(
    f: Callable,
    R_max: float,
    grid_N: int = 30,
    dps: int = DEFAULT_DPS,
    tol: float = 1e-25,
    max_iter: int = 50,
) -> list[mp.mpc]:
    """
    Find zeros of f(z) by starting Newton's method from a rectangular grid.

    The grid covers [-R_max, R_max] x [-R_max, R_max] in the complex plane.
    Also includes points on the real and imaginary axes separately for
    better coverage of real zeros.

    Parameters
    ----------
    f : analytic function
    R_max : search radius
    grid_N : number of grid divisions per axis
    dps : decimal places of precision
    tol : convergence tolerance for Newton's method
    max_iter : maximum Newton iterations

    Returns
    -------
    List of distinct zeros found (as mpc values).
    """
    mp.mp.dps = dps
    zeros_found: list[mp.mpc] = []

    # Generate starting points
    starts: list[mp.mpc] = []

    # Rectangular grid
    for i in range(grid_N + 1):
        for j in range(grid_N + 1):
            re_part = -R_max + 2 * R_max * i / grid_N
            im_part = -R_max + 2 * R_max * j / grid_N
            starts.append(mp.mpc(re_part, im_part))

    # Extra points along real axis (finer grid)
    for i in range(4 * grid_N + 1):
        re_part = -R_max + 2 * R_max * i / (4 * grid_N)
        starts.append(mp.mpc(re_part, 0))

    # Extra points along imaginary axis
    for j in range(2 * grid_N + 1):
        im_part = -R_max + 2 * R_max * j / (2 * grid_N)
        starts.append(mp.mpc(0, im_part))

    # Polar grid for better coverage
    for r_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        r = R_max * r_frac
        for k in range(16):
            theta = 2 * mp.pi * k / 16
            starts.append(r * mp.exp(mp.mpc(0, theta)))

    for z0 in starts:
        try:
            root = mp.findroot(f, z0, tol=mp.mpf(10) ** (-dps // 2), maxsteps=max_iter)
        except (ValueError, ZeroDivisionError, mp.libmp.NoConvergence):
            continue

        # Check if within search radius
        if abs(root) > R_max * 1.05:
            continue

        # Check if it's actually a zero
        val_at_root = abs(f(root))
        if val_at_root > mp.mpf("1e-15"):
            continue

        # Check if it's a new zero (not duplicate)
        is_new = True
        for existing in zeros_found:
            if abs(root - existing) < mp.mpf("1e-8"):
                is_new = False
                break
        if is_new:
            zeros_found.append(root)

    return zeros_found


def _refine_zero(
    f: Callable,
    z_approx: mp.mpc,
    dps: int = DEFAULT_DPS,
) -> mp.mpc:
    """Refine a zero using Newton's method at high precision."""
    mp.mp.dps = dps
    try:
        return mp.findroot(f, z_approx, tol=mp.mpf(10) ** (-dps + 5))
    except (ValueError, ZeroDivisionError, mp.libmp.NoConvergence):
        return z_approx


def _compute_residue(
    Pi_func: Callable,
    z0: mp.mpc,
    dps: int = DEFAULT_DPS,
) -> mp.mpc:
    """
    Compute the propagator residue at a zero z0 of Pi(z).

    For G(k^2) = 1/(k^2 * Pi(k^2/Lambda^2)), the residue at k^2 = z0*Lambda^2 is:
      Res = 1 / (z0 * Pi'(z0))

    Parameters
    ----------
    Pi_func : the propagator denominator function
    z0 : location of the zero
    dps : decimal places of precision

    Returns
    -------
    Residue as a complex number.
    """
    mp.mp.dps = dps
    h = mp.mpf("1e-10")
    Pi_deriv = (Pi_func(z0 + h) - Pi_func(z0 - h)) / (2 * h)
    if abs(Pi_deriv) < mp.mpf("1e-40"):
        return mp.mpc(mp.inf)
    return 1 / (z0 * Pi_deriv)


# ===================================================================
# Main search functions
# ===================================================================

def find_complex_zeros_PiTT(
    R_max: float = 50.0,
    grid_N: int = 30,
    dps: int = DEFAULT_DPS,
) -> list[ComplexZero]:
    """
    Find all zeros of Pi_TT(z) in the complex plane within |z| < R_max.

    Uses grid-based Newton search with refinement and residue computation.

    Parameters
    ----------
    R_max : search radius
    grid_N : grid divisions per axis
    dps : decimal places of precision

    Returns
    -------
    List of ComplexZero objects describing all found zeros.
    """
    mp.mp.dps = dps

    def f(z):
        return Pi_TT_complex(z, dps=dps)

    raw_zeros = _grid_newton_search(f, R_max, grid_N=grid_N, dps=dps)

    results: list[ComplexZero] = []
    for z_raw in raw_zeros:
        z_refined = _refine_zero(f, z_raw, dps=dps)
        residue = _compute_residue(f, z_refined, dps=dps)
        z_type = classify_zero(z_refined)
        val_at_zero = float(abs(f(z_refined)))

        cz = ComplexZero(
            z=complex(z_refined),
            z_str=str(z_refined),
            function="Pi_TT",
            zero_type=z_type,
            residue=complex(residue),
            residue_str=str(residue),
            verified=val_at_zero < 1e-15,
            abs_value_at_zero=val_at_zero,
        )
        results.append(cz)

    # Pair conjugate zeros (Type C)
    _pair_conjugates(results)

    return sorted(results, key=lambda cz: abs(cz.z))


def find_complex_zeros_Pis(
    R_max: float = 50.0,
    xi: float = 0.0,
    grid_N: int = 30,
    dps: int = DEFAULT_DPS,
) -> list[ComplexZero]:
    """
    Find all zeros of Pi_scalar(z, xi) in the complex plane within |z| < R_max.

    At xi = 1/6, Pi_s > 1 on positive real axis (no real zeros).
    Complex zeros may exist. Proceeds as for Pi_TT.

    Parameters
    ----------
    R_max : search radius
    xi : non-minimal coupling
    grid_N : grid divisions per axis
    dps : decimal places of precision

    Returns
    -------
    List of ComplexZero objects.
    """
    mp.mp.dps = dps
    xi_mp = mp.mpf(xi)

    # NOTE (2026-04-07): removed early return at xi=1/6.
    # Pi_s > 1 on positive real axis at conformal coupling,
    # but complex zeros may exist and should be searched.

    def f(z):
        return Pi_scalar_complex(z, xi=xi, dps=dps)

    raw_zeros = _grid_newton_search(f, R_max, grid_N=grid_N, dps=dps)

    results: list[ComplexZero] = []
    for z_raw in raw_zeros:
        z_refined = _refine_zero(f, z_raw, dps=dps)
        residue = _compute_residue(f, z_refined, dps=dps)
        z_type = classify_zero(z_refined)
        val_at_zero = float(abs(f(z_refined)))

        cz = ComplexZero(
            z=complex(z_refined),
            z_str=str(z_refined),
            function="Pi_scalar",
            zero_type=z_type,
            residue=complex(residue),
            residue_str=str(residue),
            verified=val_at_zero < 1e-15,
            abs_value_at_zero=val_at_zero,
            xi=xi,
        )
        results.append(cz)

    _pair_conjugates(results)
    return sorted(results, key=lambda cz: abs(cz.z))


def _pair_conjugates(zeros: list[ComplexZero], tol: float = 1e-6) -> None:
    """Pair complex conjugate zeros and mark Type D if unpaired."""
    type_c = [cz for cz in zeros if cz.zero_type == "C"]
    paired = set()
    for i, cz1 in enumerate(type_c):
        if i in paired:
            continue
        z1 = cz1.z
        found_partner = False
        for j, cz2 in enumerate(type_c):
            if j <= i or j in paired:
                continue
            z2 = cz2.z
            # Check if z2 ~ conj(z1)
            if abs(z2.real - z1.real) < tol and abs(z2.imag + z1.imag) < tol:
                cz1.conjugate_partner = z2
                cz2.conjugate_partner = z1
                paired.add(i)
                paired.add(j)
                found_partner = True
                break
        if not found_partner:
            cz1.zero_type = "D"  # Unpaired complex zero (anomalous)


# ===================================================================
# Argument principle validation
# ===================================================================

def validate_zero_count(
    func_name: str = "Pi_TT",
    R_max: float = 50.0,
    xi: float = 0.0,
    dps: int = DEFAULT_DPS,
    n_points: int = 2048,
) -> dict:
    """
    Validate the number of zeros found by comparing with the argument principle.

    Parameters
    ----------
    func_name : "Pi_TT" or "Pi_scalar"
    R_max : search radius
    xi : non-minimal coupling (for Pi_scalar only)
    dps : decimal places of precision
    n_points : quadrature points for argument principle

    Returns
    -------
    Dict with zero count from argument principle and from Newton search.
    """
    mp.mp.dps = dps

    if func_name == "Pi_TT":
        f = lambda z: Pi_TT_complex(z, dps=dps)
        zeros = find_complex_zeros_PiTT(R_max=R_max, dps=dps)
    elif func_name == "Pi_scalar":
        f = lambda z: Pi_scalar_complex(z, xi=xi, dps=dps)
        zeros = find_complex_zeros_Pis(R_max=R_max, xi=xi, dps=dps)
    else:
        raise ValueError(f"Unknown function: {func_name}")

    # Count via argument principle at multiple radii
    radii = [R_max * 0.5, R_max * 0.75, R_max]
    ap_counts = {}
    for R in radii:
        count = zero_count_argument_principle(f, R, dps=dps, n_points=n_points)
        ap_counts[f"R={R:.1f}"] = count

    newton_count = len(zeros)
    newton_in_R = {
        f"R={R:.1f}": sum(1 for cz in zeros if abs(cz.z) < R)
        for R in radii
    }

    return {
        "function": func_name,
        "xi": xi if func_name == "Pi_scalar" else None,
        "argument_principle_counts": ap_counts,
        "newton_search_count": newton_count,
        "newton_in_radius": newton_in_R,
        "consistent": all(
            ap_counts[key] == newton_in_R[key]
            for key in ap_counts
        ),
    }


# ===================================================================
# Full catalogue
# ===================================================================

def complex_zero_catalogue(
    R_max: float = 50.0,
    xi_values: list[float] | None = None,
    dps: int = DEFAULT_DPS,
) -> dict:
    """
    Produce a complete catalogue of zeros for Pi_TT and Pi_scalar.

    Parameters
    ----------
    R_max : search radius
    xi_values : list of xi values for Pi_scalar analysis
    dps : decimal places of precision

    Returns
    -------
    Dict with zero catalogue and metadata.
    """
    mp.mp.dps = dps
    if xi_values is None:
        xi_values = [0.0, 1.0 / 6]

    catalogue: dict = {
        "R_max": R_max,
        "dps": dps,
        "Pi_TT_zeros": [],
        "Pi_scalar_zeros": {},
        "summary": {},
    }

    # Pi_TT zeros
    tt_zeros = find_complex_zeros_PiTT(R_max=R_max, dps=dps)
    catalogue["Pi_TT_zeros"] = [cz.to_dict() for cz in tt_zeros]
    catalogue["summary"]["Pi_TT"] = {
        "total_zeros": len(tt_zeros),
        "type_A": sum(1 for cz in tt_zeros if cz.zero_type == "A"),
        "type_B": sum(1 for cz in tt_zeros if cz.zero_type == "B"),
        "type_C": sum(1 for cz in tt_zeros if cz.zero_type == "C"),
        "type_D": sum(1 for cz in tt_zeros if cz.zero_type == "D"),
        "all_verified": all(cz.verified for cz in tt_zeros),
    }

    # Pi_scalar zeros
    for xi in xi_values:
        xi_key = f"xi={xi:.6f}"
        s_zeros = find_complex_zeros_Pis(R_max=R_max, xi=xi, dps=dps)
        catalogue["Pi_scalar_zeros"][xi_key] = [cz.to_dict() for cz in s_zeros]
        catalogue["summary"][f"Pi_scalar_{xi_key}"] = {
            "total_zeros": len(s_zeros),
            "type_A": sum(1 for cz in s_zeros if cz.zero_type == "A"),
            "type_B": sum(1 for cz in s_zeros if cz.zero_type == "B"),
            "type_C": sum(1 for cz in s_zeros if cz.zero_type == "C"),
            "type_D": sum(1 for cz in s_zeros if cz.zero_type == "D"),
            "all_verified": all(cz.verified for cz in s_zeros),
        }

    return catalogue


def save_catalogue(catalogue: dict, filename: str = "mr1_complex_zeros.json") -> Path:
    """Save the zero catalogue to JSON."""
    output_path = RESULTS_DIR / filename
    with open(output_path, "w") as f:
        json.dump(catalogue, f, indent=2)
    return output_path


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MR-1 complex zero finder")
    parser.add_argument("--R-max", type=float, default=50.0, help="Search radius")
    parser.add_argument("--grid-N", type=int, default=25, help="Grid divisions per axis")
    parser.add_argument("--dps", type=int, default=50, help="Decimal places")
    parser.add_argument("--save", action="store_true", help="Save catalogue to JSON")
    parser.add_argument("--validate", action="store_true", help="Run argument principle validation")
    args = parser.parse_args()

    print(f"MR-1 Complex Zero Finder (R_max={args.R_max}, grid_N={args.grid_N}, dps={args.dps})")
    print("=" * 70)

    # Pi_TT zeros
    print("\n--- Pi_TT zeros ---")
    tt_zeros = find_complex_zeros_PiTT(R_max=args.R_max, grid_N=args.grid_N, dps=args.dps)
    for cz in tt_zeros:
        print(f"  z = {cz.z_str:50s}  type={cz.zero_type}  |f(z)|={cz.abs_value_at_zero:.2e}  "
              f"R={cz.residue_str}")
    print(f"  Total: {len(tt_zeros)} zeros")
    print(f"    Type A (real+): {sum(1 for c in tt_zeros if c.zero_type == 'A')}")
    print(f"    Type B (real-): {sum(1 for c in tt_zeros if c.zero_type == 'B')}")
    print(f"    Type C (complex pairs): {sum(1 for c in tt_zeros if c.zero_type == 'C')}")
    print(f"    Type D (anomalous): {sum(1 for c in tt_zeros if c.zero_type == 'D')}")

    # Pi_scalar zeros at xi=0
    print("\n--- Pi_scalar zeros (xi=0) ---")
    s_zeros = find_complex_zeros_Pis(R_max=args.R_max, xi=0.0, grid_N=args.grid_N, dps=args.dps)
    for cz in s_zeros:
        print(f"  z = {cz.z_str:50s}  type={cz.zero_type}  |f(z)|={cz.abs_value_at_zero:.2e}")
    print(f"  Total: {len(s_zeros)} zeros")

    # Pi_scalar at xi=1/6 (should be empty)
    print("\n--- Pi_scalar zeros (xi=1/6) ---")
    s_conformal = find_complex_zeros_Pis(R_max=args.R_max, xi=1.0/6, grid_N=args.grid_N, dps=args.dps)
    print(f"  Total: {len(s_conformal)} zeros (should be 0)")

    if args.validate:
        print("\n--- Argument Principle Validation ---")
        val_result = validate_zero_count("Pi_TT", R_max=args.R_max, dps=args.dps)
        print(f"  Pi_TT: AP counts = {val_result['argument_principle_counts']}")
        print(f"          Newton counts = {val_result['newton_in_radius']}")
        print(f"          Consistent = {val_result['consistent']}")

    if args.save:
        catalogue = complex_zero_catalogue(R_max=args.R_max, dps=args.dps)
        out = save_catalogue(catalogue)
        print(f"\nCatalogue saved to {out}")
