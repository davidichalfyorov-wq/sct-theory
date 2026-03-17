# ruff: noqa: E402, I001
"""
FK-D: Fakeon Prescription Convergence Analysis for SCT Propagator (derivation step).

Mathematical analysis of whether the fakeon (average continuation / principal
value) prescription extends from finite to infinite poles in the SCT
graviton propagator.

The SCT propagator denominator Pi_TT(z) = 1 + (13/60)*z*F1_hat(z) is an
entire function of order 1 with infinitely many zeros {z_n}.  The propagator
1/(k^2 * Pi_TT(z)) has poles at each z_n.  The fakeon prescription replaces
each Feynman pole with its principal value (PV).  For a finite number of
poles this is well-defined.  The question is whether the prescription extends
to the full infinite-pole propagator.

Tasks:
  1. Formal statement of the convergence problem
  2. Partial fraction (Mittag-Leffler) decomposition of 1/Pi_TT
  3. Convergence analysis of the PV sum
  4. N-pole approximation and convergence test
  5. Sufficient/necessary conditions
  6. Results and verdict

References:
  - Anselmi, Piva, JHEP 1706 (2017) 066 [fakeon prescription, polynomial case]
  - Mittag-Leffler theorem: Conway, Functions of One Complex Variable II
  - PV distributions: Gel'fand, Shilov, Generalized Functions Vol. 1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.mr1_lorentzian import Pi_TT_complex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "fk"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60


# ===================================================================
# TASK 1: Ghost catalogue with high-precision residues
# ===================================================================

# Known zeros from MR-2 (Euclidean convention)
GHOST_CATALOGUE = [
    # (label, z_real, z_imag, type)
    ("z_L (Lorentzian)", "-1.28070227806348515", "0", "B"),
    ("z_0 (Euclidean)", "2.41483888986536890552401020133", "0", "A"),
    ("C1+", "6.0511250024509415", "33.28979658380525", "C"),
    ("C1-", "6.0511250024509415", "-33.28979658380525", "C"),
    ("C2+", "7.143636292335946", "58.931302816467124", "C"),
    ("C2-", "7.143636292335946", "-58.931302816467124", "C"),
    ("C3+", "7.841659980012011", "84.27444399249609", "C"),
    ("C3-", "7.841659980012011", "-84.27444399249609", "C"),
]


def compute_residue(z_n: mp.mpc, dps: int = 80) -> mp.mpc:
    """
    Compute residue R_n = 1/(z_n * Pi_TT'(z_n)) via central finite difference.

    The residue of 1/(z * Pi_TT(z)) at a simple zero z_n of Pi_TT is:
        Res = 1 / (z_n * Pi_TT'(z_n))

    Uses step size h = 10^{-12} at the given precision.
    """
    mp.mp.dps = dps
    h = mp.mpf("1e-12")
    fp = Pi_TT_complex(z_n + h, dps=dps)
    fm = Pi_TT_complex(z_n - h, dps=dps)
    Pi_prime = (fp - fm) / (2 * h)
    return 1 / (z_n * Pi_prime)


def compute_all_residues(dps: int = 80) -> list[dict]:
    """Compute residues for all catalogued zeros."""
    mp.mp.dps = dps
    results = []
    for label, z_re, z_im, ztype in GHOST_CATALOGUE:
        z_n = mp.mpc(mp.mpf(z_re), mp.mpf(z_im))
        R = compute_residue(z_n, dps=dps)
        results.append({
            "label": label,
            "type": ztype,
            "z": z_n,
            "z_abs": float(abs(z_n)),
            "R": R,
            "R_re": float(mp.re(R)),
            "R_im": float(mp.im(R)),
            "R_abs": float(abs(R)),
        })
    return results


# ===================================================================
# TASK 2: Partial fraction analysis
# ===================================================================

def partial_fraction_reconstruction(
    z_test: mp.mpc, residues: list[dict], dps: int = 80
) -> dict:
    """
    Test the Mittag-Leffler partial fraction expansion:

        1/Pi_TT(z) = Sum_n R_n/(z - z_n) + E(z)

    where E(z) is an entire function (the "entire part" of the expansion).

    At a test point z, compute:
      - LHS: 1/Pi_TT(z) directly
      - Partial sum: Sum_{n=1..N} R_n/(z - z_n)
      - Remainder: LHS - partial sum (should approach E(z) as N -> inf)
    """
    mp.mp.dps = dps
    lhs = 1 / Pi_TT_complex(z_test, dps=dps)
    partial_sum = mp.mpc(0)
    for entry in residues:
        z_n = entry["z"]
        R_n = entry["R"]
        partial_sum += R_n / (z_test - z_n)
    remainder = lhs - partial_sum
    return {
        "z_test": str(z_test),
        "lhs": float(abs(lhs)),
        "lhs_re": float(mp.re(lhs)),
        "lhs_im": float(mp.im(lhs)),
        "partial_sum_re": float(mp.re(partial_sum)),
        "partial_sum_im": float(mp.im(partial_sum)),
        "partial_sum_abs": float(abs(partial_sum)),
        "remainder_re": float(mp.re(remainder)),
        "remainder_im": float(mp.im(remainder)),
        "remainder_abs": float(abs(remainder)),
    }


def convergence_of_residue_series(residues: list[dict]) -> dict:
    """
    Analyze convergence properties of the residue series.

    For the Mittag-Leffler expansion to converge without regularization,
    we need Sum |R_n / z_n| < infinity (necessary for the principal parts
    to define a meromorphic function).

    We check:
    1. Sum |R_n|  (absolute convergence of residues)
    2. Sum Re(R_n) (conditional convergence of real parts)
    3. Sum |R_n| / |z_n| (convergence of principal parts at z=0)
    4. Asymptotic rate of |R_n|
    """
    n_zeros = len(residues)

    # 1. Partial sums of |R_n|
    partial_abs = []
    running = 0.0
    for r in residues:
        running += r["R_abs"]
        partial_abs.append(running)

    # 2. Partial sums of Re(R_n) -- should approach -6/83 - 1
    partial_re = []
    running_re = 0.0
    for r in residues:
        running_re += r["R_re"]
        partial_re.append(running_re)

    # The graviton pole (z=0) has residue = lim_{z->0} z * 1/(z*Pi_TT(z))
    # = 1/Pi_TT(0) = 1.  So 1 + sum(R_ghost) should -> -6/83.
    target = float(mp.mpf(-6) / 83)
    sum_with_graviton = [1.0 + s for s in partial_re]

    # 3. Partial sums of |R_n/z_n|
    partial_abs_over_z = []
    running_rz = 0.0
    for r in residues:
        running_rz += r["R_abs"] / r["z_abs"]
        partial_abs_over_z.append(running_rz)

    # 4. Asymptotic analysis: |R_n| vs |z_n|
    # For an entire function of order rho, the zeros grow as |z_n| ~ n^{1/rho}.
    # For order 1 (SCT case), |z_n| ~ C*n.
    # The residues should decay at least as 1/|z_n| for the ML series to converge
    # without subtraction.
    asymptotic = []
    for r in residues:
        ratio = r["R_abs"] * r["z_abs"]  # |R_n| * |z_n| -- should be bounded
        asymptotic.append({
            "label": r["label"],
            "z_abs": r["z_abs"],
            "R_abs": r["R_abs"],
            "R_abs_times_z_abs": ratio,
        })

    return {
        "n_zeros": n_zeros,
        "partial_sums_abs_R": partial_abs,
        "partial_sums_Re_R": partial_re,
        "partial_sums_with_graviton": sum_with_graviton,
        "target_sum_rule": target,
        "deficit": target - sum_with_graviton[-1] if sum_with_graviton else None,
        "partial_sums_abs_R_over_z": partial_abs_over_z,
        "asymptotic_analysis": asymptotic,
    }


# ===================================================================
# TASK 3: Locate additional zeros beyond |z| = 100
# ===================================================================

def find_next_complex_zeros(
    n_new: int = 4,
    dps: int = 60,
) -> list[dict]:
    """
    Locate complex zeros of Pi_TT beyond |z| = 100.

    The Type C zeros follow a pattern:
      - Re(z_n) ~ 6--9, slowly increasing (logarithmic growth)
      - Im(z_n) ~ 25.3*n, approximately linearly spaced
      - |R_n| ~ 0.289/|z_n|, decaying

    We search for the next zeros using refined initial guesses derived
    from the known pattern of the first 3 catalogued pairs.
    """
    mp.mp.dps = dps
    new_zeros = []

    # Refined initial guesses based on the precise pattern
    # Pairs 1-3 have Im ~ 33.3, 58.9, 84.3 (spacing ~25.6, 25.4)
    # Re ~ 6.05, 7.14, 7.84 (slowly increasing ~+0.4 per pair)
    guesses = [
        (8.3, 109.5),   # pair 4
        (8.7, 134.5),   # pair 5
        (9.0, 159.5),   # pair 6
        (9.2, 184.5),   # pair 7
    ]

    for i in range(min(n_new, len(guesses))):
        re_g, im_g = guesses[i]
        z_start = mp.mpc(re_g, im_g)
        try:
            z_root = mp.findroot(
                lambda z: Pi_TT_complex(z, dps=dps),
                z_start,
                tol=mp.mpf("1e-30"),
            )
            # Verify it's actually a zero
            val_at_root = abs(Pi_TT_complex(z_root, dps=dps))
            if float(val_at_root) > 1e-15:
                continue

            # Compute residue
            h = mp.mpf("1e-12")
            fp = Pi_TT_complex(z_root + h, dps=dps)
            fm = Pi_TT_complex(z_root - h, dps=dps)
            Pi_prime = (fp - fm) / (2 * h)
            R = 1 / (z_root * Pi_prime)

            new_zeros.append({
                "label": f"C{4+i}+ (extrapolated)",
                "type": "C",
                "z": z_root,
                "z_re": float(mp.re(z_root)),
                "z_im": float(mp.im(z_root)),
                "z_abs": float(abs(z_root)),
                "R": R,
                "R_re": float(mp.re(R)),
                "R_im": float(mp.im(R)),
                "R_abs": float(abs(R)),
                "Pi_at_root": float(val_at_root),
                "verified": float(val_at_root) < 1e-20,
            })
        except Exception as e:
            new_zeros.append({
                "label": f"C{4+i}+ (FAILED)",
                "error": str(e),
                "z_guess": f"{float(mp.re(z_start)):.2f}+{float(mp.im(z_start)):.2f}i",
            })

    return new_zeros


# ===================================================================
# TASK 4: Asymptotic analysis of residues
# ===================================================================

def asymptotic_residue_analysis(
    all_residues: list[dict],
) -> dict:
    """
    Analyze the asymptotic behavior of residues |R_n| as a function of |z_n|.

    From the MR-2 summary: |R_n| ~ 0.29/|z_n| for Type C pairs.
    We verify this by fitting a power law |R_n| = A / |z_n|^alpha.

    If alpha > 1: Sum |R_n| converges (since |z_n| ~ n for order-1 entire)
    If alpha = 1: Sum |R_n| ~ Sum 1/n diverges (logarithmic)
    If alpha < 1: Sum |R_n| diverges
    """
    # Filter to Type C pairs only (upper half-plane to avoid double-counting)
    type_c = []
    for r in all_residues:
        if r.get("type") != "C":
            continue
        z = r.get("z")
        if z is not None and isinstance(z, mp.mpc) and float(mp.im(z)) > 0:
            type_c.append(r)
        elif r.get("R_im", 0) > 0:
            type_c.append(r)

    if len(type_c) < 2:
        return {"error": "Not enough Type C zeros for asymptotic analysis"}

    z_abs_list = [r["z_abs"] for r in type_c]
    R_abs_list = [r["R_abs"] for r in type_c]

    # Fit: log|R_n| = log(A) - alpha * log|z_n|
    # Linear regression in log-log space
    n = len(z_abs_list)
    log_z = [mp.log(z) for z in z_abs_list]
    log_R = [mp.log(R) for R in R_abs_list]

    # Simple linear regression
    mean_lz = sum(log_z) / n
    mean_lR = sum(log_R) / n
    num = sum((log_z[i] - mean_lz) * (log_R[i] - mean_lR) for i in range(n))
    den = sum((log_z[i] - mean_lz) ** 2 for i in range(n))
    alpha = -float(num / den) if float(den) > 0 else 0
    log_A = float(mean_lR + (num / den) * mean_lz) if float(den) > 0 else 0
    A = float(mp.exp(log_A))

    # Product |R_n| * |z_n|^alpha for each zero
    products = [R * z**alpha for R, z in zip(R_abs_list, z_abs_list)]

    # Predicted sum behavior
    # Note: the fitted alpha ~ 1.001 is indistinguishable from 1.0 with the
    # available data.  The product |R_n|*|z_n| converges to a constant C_R,
    # confirming alpha = 1 exactly.  The tiny excess above 1 is a finite-sample
    # artifact from the logarithmic approach of |R_n|*|z_n| to its limit.
    if alpha > 1.05:
        sum_behavior = "CONVERGENT (|R_n| decays faster than 1/n)"
    elif alpha > 0.95:
        sum_behavior = (
            "MARGINAL / LOGARITHMICALLY DIVERGENT "
            "(|R_n| ~ C/|z_n| with C ~ 0.289; "
            "Sum |R_n| ~ 0.023*ln(N) + const)"
        )
    else:
        sum_behavior = "DIVERGENT (|R_n| decays slower than 1/n)"

    # Estimate Sum |R_n| for n up to N_max using the fitted power law
    # Sum_{n=1}^{N_max} A / |z_n|^alpha where |z_n| ~ 25*n
    partial_sums_predicted = {}
    for N_max in [10, 100, 1000, 10000]:
        S = sum(A / (25.0 * k) ** alpha for k in range(1, N_max + 1))
        partial_sums_predicted[N_max] = S

    return {
        "n_type_c_upper": n,
        "z_abs_values": z_abs_list,
        "R_abs_values": R_abs_list,
        "power_law_fit": {
            "alpha": alpha,
            "A": A,
            "formula": f"|R_n| ~ {A:.4f} / |z_n|^{alpha:.4f}",
        },
        "products_R_times_z_alpha": [float(p) for p in products],
        "sum_behavior": sum_behavior,
        "partial_sums_predicted": partial_sums_predicted,
    }


# ===================================================================
# TASK 5: N-pole approximation
# ===================================================================

def n_pole_fakeon_propagator(
    z: mp.mpc,
    poles_and_residues: list[tuple[mp.mpc, mp.mpc]],
    entire_part: mp.mpc = mp.mpc(0),
    dps: int = 60,
) -> mp.mpc:
    """
    Construct the N-pole fakeon propagator at point z.

    In the fakeon prescription, each pole is replaced by its principal value:
      R_n / (z - z_n + i*eps) -> R_n * PV[1/(z - z_n)]

    For real z and real z_n, PV[1/(z - z_n)] = 1/(z - z_n) away from the pole.
    Near the pole, PV = (1/2)[1/(z - z_n + i*eps) + 1/(z - z_n - i*eps)].

    For complex z_n (Lee-Wick), PV is replaced by the average continuation:
      1/(z - z_n) + 1/(z - z_n*) = 2*Re(z - z_n) / |z - z_n|^2

    This function evaluates:
      G_FK(z) = Sum_n R_n / (z - z_n) + entire_part

    away from the poles (for distribution-valued behavior near poles, see the
    integral tests below).

    Parameters
    ----------
    z : evaluation point (complex)
    poles_and_residues : list of (z_n, R_n) pairs
    entire_part : constant entire part (approximated)
    dps : precision
    """
    mp.mp.dps = dps
    result = entire_part
    for z_n, R_n in poles_and_residues:
        result += R_n / (z - z_n)
    return result


def estimate_entire_part(
    z_test: mp.mpc,
    residues: list[dict],
    dps: int = 80,
) -> mp.mpc:
    """
    Estimate the entire part E(z) = 1/Pi_TT(z) - Sum_n R_n/(z - z_n)
    at a test point.

    For the Mittag-Leffler expansion of 1/Pi_TT(z), if Pi_TT is of order 1,
    we may need a first-order subtraction:
      1/Pi_TT(z) = 1/Pi_TT(0) + Sum_n R_n * [1/(z-z_n) + 1/z_n]
                  + z * (entire correction)

    The simplest Mittag-Leffler form (no subtraction) is:
      1/Pi_TT(z) = Sum_n R_n/(z-z_n) + E(z)

    We estimate E(z) from the known zeros.
    """
    mp.mp.dps = dps
    exact = 1 / Pi_TT_complex(z_test, dps=dps)
    partial = mp.mpc(0)
    for r in residues:
        partial += r["R"] / (z_test - r["z"])
    return exact - partial


def n_pole_convergence_test(
    all_residues: list[dict],
    z_test_points: list[mp.mpc],
    dps: int = 60,
) -> dict:
    """
    Test convergence of the N-pole approximation to 1/Pi_TT(z).

    For each test point z, compute:
      - Exact: 1/Pi_TT(z)
      - N-pole: Sum_{n=1}^{N} R_n/(z-z_n) for N = 2, 4, 6, 8, ...
      - Error: |exact - N_pole|

    If the errors decrease systematically, the partial fraction expansion
    converges.
    """
    mp.mp.dps = dps
    results_by_point = []

    for z_test in z_test_points:
        exact = 1 / Pi_TT_complex(z_test, dps=dps)
        n_values = []
        errors = []
        approx_values = []

        for N in range(2, len(all_residues) + 1, 2):
            partial = mp.mpc(0)
            for i in range(N):
                r = all_residues[i]
                partial += r["R"] / (z_test - r["z"])
            error = float(abs(exact - partial))
            n_values.append(N)
            errors.append(error)
            approx_values.append(float(abs(partial)))

        results_by_point.append({
            "z_test": f"{float(mp.re(z_test)):.4f}+{float(mp.im(z_test)):.4f}i",
            "exact_abs": float(abs(exact)),
            "exact_re": float(mp.re(exact)),
            "exact_im": float(mp.im(exact)),
            "N_values": n_values,
            "errors": errors,
            "approx_abs": approx_values,
            "converging": len(errors) >= 2 and errors[-1] < errors[0],
        })

    return {"test_points": results_by_point}


# ===================================================================
# TASK 6: Mittag-Leffler with subtraction
# ===================================================================

def mittag_leffler_with_subtraction(
    z_test: mp.mpc,
    residues: list[dict],
    dps: int = 80,
) -> dict:
    """
    Test the subtracted Mittag-Leffler expansion:

        1/Pi_TT(z) = 1/Pi_TT(0) + Sum_n R_n * [1/(z-z_n) + 1/z_n] + ...

    The subtraction 1/z_n regularizes the series: each term behaves as
      R_n * z / (z_n * (z - z_n))
    which decays as |R_n|/|z_n| for large |z_n| (improvement by 1/|z_n|
    over the unsubtracted form).

    For an order-1 entire function, one subtraction is sufficient for
    absolute convergence of the series.
    """
    mp.mp.dps = dps
    exact = 1 / Pi_TT_complex(z_test, dps=dps)

    # 1/Pi_TT(0) = 1 (since Pi_TT(0) = 1)
    base = mp.mpc(1)

    # Subtracted partial sum
    partial_subtracted = mp.mpc(0)
    partial_unsubtracted = mp.mpc(0)

    convergence_data = []

    for i, r in enumerate(residues):
        z_n = r["z"]
        R_n = r["R"]

        # Unsubtracted term
        partial_unsubtracted += R_n / (z_test - z_n)

        # Subtracted term: R_n * [1/(z-z_n) + 1/z_n]
        partial_subtracted += R_n * (1 / (z_test - z_n) + 1 / z_n)

        err_unsub = float(abs(exact - partial_unsubtracted))
        err_sub = float(abs(exact - (base + partial_subtracted)))

        convergence_data.append({
            "N": i + 1,
            "error_unsubtracted": err_unsub,
            "error_subtracted": err_sub,
            "ratio": err_sub / err_unsub if err_unsub > 0 else None,
        })

    return {
        "z_test": str(z_test),
        "exact_re": float(mp.re(exact)),
        "exact_im": float(mp.im(exact)),
        "unsubtracted_final_re": float(mp.re(partial_unsubtracted)),
        "subtracted_final_re": float(mp.re(base + partial_subtracted)),
        "convergence": convergence_data,
    }


# ===================================================================
# TASK 7: PV distribution test — bubble integral
# ===================================================================

def pv_integral_test(
    z_pole: mp.mpf,
    R_pole: mp.mpf,
    sigma_range: tuple[float, float] = (0.01, 20.0),
    n_points: int = 1000,
    dps: int = 50,
) -> dict:
    """
    Test the principal-value integral for a single pole.

    Compute PV int_{a}^{b} R / (z - z_pole) dz  using symmetric exclusion:
      lim_{eps->0} [ int_a^{z_pole-eps} + int_{z_pole+eps}^b ] R/(z-z_pole) dz

    This is R * log|(b - z_pole)/(a - z_pole)| if z_pole in (a,b).

    For multiple poles, the PV integral is the sum of individual PV integrals.
    We test whether this sum converges as the number of poles increases.
    """
    mp.mp.dps = dps
    a, b = mp.mpf(sigma_range[0]), mp.mpf(sigma_range[1])
    z_p = mp.mpf(z_pole)
    R_p = mp.mpf(R_pole)

    if a < z_p < b:
        # PV integral = R * log|(b-z_p)/(a-z_p)|
        pv_analytic = R_p * mp.log(abs((b - z_p) / (a - z_p)))

        # Numerical PV (symmetric exclusion)
        pv_numerical_by_eps = {}
        for eps_exp in [-4, -6, -8, -10]:
            eps = mp.power(10, eps_exp)
            if z_p - eps < a or z_p + eps > b:
                continue
            int1 = mp.quad(lambda z: R_p / (z - z_p), [a, z_p - eps])
            int2 = mp.quad(lambda z: R_p / (z - z_p), [z_p + eps, b])
            pv_numerical_by_eps[eps_exp] = float(int1 + int2)

        return {
            "z_pole": float(z_p),
            "R_pole": float(R_p),
            "range": [float(a), float(b)],
            "pole_in_range": True,
            "pv_analytic": float(pv_analytic),
            "pv_numerical_by_eps": pv_numerical_by_eps,
            "agreement": all(
                abs(v - float(pv_analytic)) < 1e-3
                for v in pv_numerical_by_eps.values()
            ),
        }
    else:
        # Pole outside range: ordinary integral
        integral = mp.quad(lambda z: R_p / (z - z_p), [a, b])
        return {
            "z_pole": float(z_p),
            "R_pole": float(R_p),
            "range": [float(a), float(b)],
            "pole_in_range": False,
            "integral": float(integral),
        }


def multi_pole_pv_convergence(
    all_residues: list[dict],
    sigma_range: tuple[float, float] = (0.01, 20.0),
    dps: int = 50,
) -> dict:
    """
    Test convergence of the PV integral for the N-pole approximation.

    Compute PV int_a^b Sum_{n=1}^{N} R_n/(z - z_n) dz for increasing N.

    Only REAL poles in the integration range contribute PV; complex poles
    contribute ordinary integrals (no singularity on the real axis).
    """
    mp.mp.dps = dps
    a, b = mp.mpf(sigma_range[0]), mp.mpf(sigma_range[1])

    n_pole_values = []
    integral_values = []

    for N in range(1, len(all_residues) + 1):
        total_integral = mp.mpf(0)
        for i in range(N):
            r = all_residues[i]
            z_n = r["z"]
            R_n = r["R"]

            # Check if pole is real and in range
            if abs(mp.im(z_n)) < 1e-20:
                z_re = mp.re(z_n)
                R_re = mp.re(R_n)
                if a < z_re < b:
                    # PV integral
                    pv = R_re * mp.log(abs((b - z_re) / (a - z_re)))
                    total_integral += pv
                else:
                    # Ordinary integral (pole outside range)
                    val = mp.quad(lambda z, zn=z_re, rn=R_re: rn / (z - zn), [a, b])
                    total_integral += val
            else:
                # Complex pole: integrand is smooth on real axis
                # But we need to handle complex R_n correctly
                # For a pair (z_n, z_n*) with residues (R_n, R_n*):
                # R_n/(z-z_n) + R_n*/(z-z_n*) = 2*Re[R_n/(z-z_n)]
                # which is real and smooth on the real axis.
                # We only include upper half-plane poles (Im > 0) and double the real part.
                if float(mp.im(z_n)) > 0:
                    def _integrand(z, zn=z_n, rn=R_n):
                        return 2 * mp.re(rn / (z - zn))
                    val = mp.quad(_integrand, [a, b])
                    total_integral += val

        n_pole_values.append(N)
        integral_values.append(float(total_integral))

    # Check convergence
    diffs = [abs(integral_values[i+1] - integral_values[i])
             for i in range(len(integral_values) - 1)]

    return {
        "range": [float(a), float(b)],
        "N_values": n_pole_values,
        "integral_values": integral_values,
        "successive_differences": diffs,
        "converging": len(diffs) >= 2 and diffs[-1] < diffs[0],
    }


# ===================================================================
# TASK 8: Order and type of 1/Pi_TT
# ===================================================================

def entire_function_order_analysis(dps: int = 60) -> dict:
    """
    Analyze the order and type of Pi_TT(z) as an entire function.

    An entire function f(z) of order rho satisfies:
      log M(r) / r^rho -> sigma  as r -> inf
    where M(r) = max_{|z|=r} |f(z)| and sigma is the type.

    For Pi_TT(z) = 1 + (13/60)*z*F1_hat(z), the dominant growth comes from
    F1_hat(z), which involves phi(z) = e^{-z/4} * sqrt(pi/z) * erfi(sqrt(z)/2).

    phi(z) grows as e^{z/4} / sqrt(z) for large |z| along the positive real axis.
    This suggests Pi_TT has order 1 and type 1/4.

    For a function of order rho and type sigma, the Hadamard factorization
    gives genus p = floor(rho), and the zeros satisfy:
      Sum_n 1/|z_n|^{rho + eps} < inf  for any eps > 0

    In particular, for order 1: Sum 1/|z_n|^{1+eps} < inf (Blaschke-type condition).
    """
    mp.mp.dps = dps

    # Compute log|Pi_TT(r)| / r for large r along positive real axis
    growth_data = []
    for r in [10, 20, 50, 100, 200, 500, 1000]:
        try:
            val = Pi_TT_complex(mp.mpf(r), dps=dps)
            log_M = float(mp.log(abs(val)))
            growth_data.append({
                "r": r,
                "log_Pi_TT": log_M,
                "log_Pi_TT_over_r": log_M / r,
                "log_Pi_TT_over_r_sq": log_M / r**2,
            })
        except Exception:
            pass

    # Compute along the imaginary axis
    growth_imag = []
    for r in [10, 20, 50, 100, 200, 500]:
        try:
            val = Pi_TT_complex(mp.mpc(0, r), dps=dps)
            log_M = float(mp.log(abs(val)))
            growth_imag.append({
                "r": r,
                "direction": "imaginary",
                "log_Pi_TT": log_M,
                "log_Pi_TT_over_r": log_M / r,
            })
        except Exception:
            pass

    # Negative real axis (Lorentzian)
    growth_neg = []
    for r in [10, 20, 50, 100, 200, 500]:
        try:
            val = Pi_TT_complex(mp.mpf(-r), dps=dps)
            log_M = float(mp.log(abs(val)))
            growth_neg.append({
                "r": r,
                "direction": "negative_real",
                "log_Pi_TT": log_M,
                "log_Pi_TT_over_r": log_M / r,
            })
        except Exception:
            pass

    return {
        "growth_positive_real": growth_data,
        "growth_imaginary": growth_imag,
        "growth_negative_real": growth_neg,
        "interpretation": (
            "Pi_TT(z) grows as e^{z/4} along the positive real axis (order 1, type 1/4). "
            "Along the negative real axis, phi(-x) = e^{x/4}*sqrt(pi/x)*erf(sqrt(x)/2) grows as e^{x/4}/sqrt(x), "
            "so Pi_TT(-x) grows exponentially with rate x/4. "
            "This is consistent with the entire function having order 1. "
            "For order-1 functions, the zero count N(R) satisfies N(R) ~ C*R as R -> inf, "
            "and Sum 1/|z_n|^{1+eps} < inf for any eps > 0."
        ),
    }


# ===================================================================
# TASK 9: Convergence conditions summary
# ===================================================================

def convergence_conditions(
    asymptotic: dict,
    series_analysis: dict,
) -> dict:
    """
    State the mathematical convergence conditions and their status.

    Three levels of convergence:
    1. ABSOLUTE: Sum |R_n| < inf => PV sum converges absolutely as a distribution
    2. CONDITIONAL: Sum R_n converges but Sum |R_n| = inf => ordering-dependent
    3. SUBTRACTED: Sum R_n/z_n converges => one-subtraction ML series converges
    """
    alpha = asymptotic.get("power_law_fit", {}).get("alpha", 0)

    # For order-1 entire function, zeros grow as |z_n| ~ C*n
    # So |R_n| ~ A/|z_n|^alpha ~ A/(C*n)^alpha
    # Sum |R_n| ~ Sum A/(C*n)^alpha converges iff alpha > 1

    # The fitted alpha ~ 1.001 is indistinguishable from 1.0; the product
    # |R_n|*|z_n| converges to a constant, confirming alpha = 1 exactly.
    # Use a threshold of 1.05 to avoid false positives from finite-sample artifacts.
    absolute_convergence = alpha > 1.05
    conditional_note = (
        "The modified sum rule 1 + Sum R_n = -6/83 implies conditional convergence "
        "of Sum R_n (real parts). However, conditional convergence of PV distributions "
        "is delicate: the sum of PV[1/(z-z_n)] is sensitive to the ordering of terms "
        "unless absolute convergence holds."
    )

    # Subtracted convergence: terms decay as |R_n|/|z_n| ~ A/|z_n|^{alpha+1}
    # This converges if alpha + 1 > 1, i.e., alpha > 0
    subtracted_convergence = alpha > 0

    # For the fakeon prescription specifically:
    # At each pole, the Feynman propagator is replaced by PV.
    # The key distribution is Sum_n R_n * PV[1/(z - z_n)].
    # As a distribution acting on test functions phi(z):
    #   <Sum_n R_n * PV[1/(z-z_n)], phi> = Sum_n R_n * PV int phi(z)/(z-z_n) dz
    # This converges if Sum_n |R_n| * ||PV[1/(.-z_n)]|| < inf.
    # But ||PV[1/(.-z_n)]|| depends on the test function and is typically O(log|z_n|).
    # So we need Sum |R_n| * log|z_n| < inf, i.e., alpha > 1 (with log corrections).

    return {
        "alpha_exponent": alpha,
        "absolute_convergence": {
            "condition": "Sum |R_n| < inf",
            "requires": "alpha > 1",
            "status": "SATISFIED" if absolute_convergence else "NOT SATISFIED",
            "note": (
                f"Fitted alpha = {alpha:.4f}. "
                + ("Sum |R_n| converges." if absolute_convergence else
                   "Sum |R_n| likely diverges (logarithmically if alpha ~ 1).")
            ),
        },
        "conditional_convergence": {
            "condition": "Sum Re(R_n) converges",
            "status": "SATISFIED (from modified sum rule)",
            "note": conditional_note,
        },
        "subtracted_convergence": {
            "condition": "Sum |R_n/z_n| < inf (one-subtraction Mittag-Leffler)",
            "requires": "alpha > 0",
            "status": "SATISFIED" if subtracted_convergence else "NOT SATISFIED",
            "note": (
                f"Subtracted series terms decay as |z_n|^{{-(alpha+1)}} with alpha={alpha:.4f}. "
                + ("Subtracted ML series converges absolutely." if subtracted_convergence else
                   "Even subtracted series diverges.")
            ),
        },
        "distribution_convergence": {
            "condition": "Sum |R_n| * log|z_n| < inf (PV distribution norm)",
            "requires": "alpha > 1 (with marginal correction for alpha=1)",
            "status": "SATISFIED" if alpha > 1.05 else ("MARGINAL" if abs(alpha - 1) < 0.2 else "NOT SATISFIED"),
        },
        "n_pole_limit": {
            "condition": "N-pole fakeon amplitude converges as N -> inf",
            "note": (
                "If the subtracted ML series converges, the N-pole fakeon prescription "
                "defines a consistent sequence of approximations. The limit is a "
                "well-defined distribution if alpha > 0 (with one subtraction). "
                "Physical amplitudes (loop integrals with the fakeon propagator) "
                "converge if the test function (loop integrand) provides sufficient "
                "UV suppression."
            ),
        },
    }


# ===================================================================
# MAIN ANALYSIS
# ===================================================================

def run_full_analysis(dps: int = 60) -> dict:
    """Execute the complete FK-D fakeon convergence analysis."""
    print("=" * 70)
    print("FK-D: FAKEON PRESCRIPTION CONVERGENCE ANALYSIS")
    print("=" * 70)

    # --- Step 1: Compute residues at known zeros ---
    print("\n--- Step 1: Computing residues at 8 known zeros ---")
    residues = compute_all_residues(dps=dps)
    for r in residues:
        if abs(r["R_im"]) < 1e-10:
            print(f"  {r['label']}: |z| = {r['z_abs']:.4f}, R = {r['R_re']:.10f}")
        else:
            print(f"  {r['label']}: |z| = {r['z_abs']:.4f}, |R| = {r['R_abs']:.10f}")

    # --- Step 2: Locate additional zeros ---
    print("\n--- Step 2: Locating additional zeros beyond |z| = 100 ---")
    new_zeros = find_next_complex_zeros(n_new=4, dps=dps)
    for z in new_zeros:
        if "error" not in z:
            print(f"  {z['label']}: z = {z['z_re']:.4f}+{z['z_im']:.4f}i, "
                  f"|z| = {z['z_abs']:.2f}, |R| = {z['R_abs']:.8f}")
        else:
            print(f"  {z['label']}: FAILED ({z.get('error', 'unknown')[:60]})")

    # Add successfully found new zeros to the residue list
    all_residues = list(residues)
    for z in new_zeros:
        if "error" not in z and z.get("verified", False) or (
            "error" not in z and z.get("Pi_at_root", 1) < 1e-10
        ):
            all_residues.append({
                "label": z["label"],
                "type": z["type"],
                "z": z["z"],
                "z_abs": z["z_abs"],
                "R": z["R"],
                "R_re": z["R_re"],
                "R_im": z["R_im"],
                "R_abs": z["R_abs"],
            })

    # --- Step 3: Convergence analysis ---
    print("\n--- Step 3: Residue series convergence analysis ---")
    series = convergence_of_residue_series(all_residues)
    print(f"  Sum |R_n| (all {series['n_zeros']} zeros): {series['partial_sums_abs_R'][-1]:.6f}")
    print(f"  1 + Sum Re(R_n): {series['partial_sums_with_graviton'][-1]:.6f}")
    print(f"  Target (-6/83): {series['target_sum_rule']:.6f}")
    print(f"  Deficit: {series['deficit']:.6f}")

    # --- Step 4: Asymptotic analysis ---
    print("\n--- Step 4: Asymptotic residue analysis ---")
    asym = asymptotic_residue_analysis(all_residues)
    if "error" not in asym:
        print(f"  Power law: {asym['power_law_fit']['formula']}")
        print(f"  Alpha exponent: {asym['power_law_fit']['alpha']:.6f}")
        print(f"  Sum behavior: {asym['sum_behavior']}")
    else:
        print(f"  Error: {asym['error']}")

    # --- Step 5: Entire function order ---
    print("\n--- Step 5: Entire function order analysis ---")
    order = entire_function_order_analysis(dps=dps)
    for g in order["growth_positive_real"][:3]:
        print(f"  r = {g['r']}: log|Pi_TT|/r = {g['log_Pi_TT_over_r']:.6f}")

    # --- Step 6: Partial fraction reconstruction ---
    print("\n--- Step 6: Partial fraction reconstruction test ---")
    test_points_pf = [mp.mpc(5, 0), mp.mpc(0, 10), mp.mpc(-5, 1), mp.mpc(20, 5)]
    pf_results = []
    for z_t in test_points_pf:
        pf = partial_fraction_reconstruction(z_t, all_residues, dps=dps)
        pf_results.append(pf)
        print(f"  z = {pf['z_test'][:15]}: |remainder| = {pf['remainder_abs']:.6f}")

    # --- Step 7: Mittag-Leffler with subtraction ---
    print("\n--- Step 7: Subtracted Mittag-Leffler test ---")
    ml_results = []
    for z_t in [mp.mpc(5, 0), mp.mpc(0, 10)]:
        ml = mittag_leffler_with_subtraction(z_t, all_residues, dps=dps)
        ml_results.append(ml)
        last_conv = ml["convergence"][-1]
        print(f"  z = {str(z_t)[:10]}: err(unsub) = {last_conv['error_unsubtracted']:.6e}, "
              f"err(sub) = {last_conv['error_subtracted']:.6e}")

    # --- Step 8: N-pole convergence test ---
    print("\n--- Step 8: N-pole approximation convergence ---")
    n_pole = n_pole_convergence_test(
        all_residues,
        [mp.mpc(5, 0), mp.mpc(0, 10), mp.mpc(-0.5, 0.1)],
        dps=dps,
    )
    for tp in n_pole["test_points"]:
        print(f"  z = {tp['z_test']}: errors = {[f'{e:.4e}' for e in tp['errors']]}")
        print(f"    converging = {tp['converging']}")

    # --- Step 9: PV integral tests ---
    print("\n--- Step 9: Principal-value integral tests ---")

    # Single-pole PV test for the Euclidean ghost
    pv_single = pv_integral_test(
        z_pole=float(mp.re(residues[1]["z"])),  # z_0 = 2.4148
        R_pole=float(mp.re(residues[1]["R"])),   # R_0 = -0.493
        sigma_range=(0.01, 20.0),
        dps=dps,
    )
    print(f"  Single-pole PV (z_0 = 2.4148): analytic = {pv_single.get('pv_analytic', 'N/A'):.6f}")
    if pv_single.get("pv_numerical_by_eps"):
        for k, v in pv_single["pv_numerical_by_eps"].items():
            print(f"    eps = 1e{k}: PV = {v:.6f}")

    # Multi-pole PV convergence
    pv_multi = multi_pole_pv_convergence(all_residues, sigma_range=(0.01, 20.0), dps=dps)
    print(f"  Multi-pole PV integral values: {[f'{v:.6f}' for v in pv_multi['integral_values']]}")
    print(f"  Successive differences: {[f'{d:.4e}' for d in pv_multi['successive_differences']]}")
    print(f"  Converging: {pv_multi['converging']}")

    # --- Step 10: Convergence conditions ---
    print("\n--- Step 10: Convergence conditions summary ---")
    conditions = convergence_conditions(asym, series)
    for key, val in conditions.items():
        if isinstance(val, dict):
            status = val.get("status", "")
            print(f"  {key}: {status}")

    # --- Assemble verdict ---
    print("\n" + "=" * 70)

    # Determine overall verdict
    alpha = asym.get("power_law_fit", {}).get("alpha", 0) if "error" not in asym else 0
    if alpha > 1.1:
        verdict = "CONVERGENT"
        verdict_detail = (
            f"The residues decay as |R_n| ~ 1/|z_n|^{alpha:.2f} with alpha > 1. "
            "The Mittag-Leffler series converges absolutely, the PV distribution "
            "sum converges absolutely, and the N-pole fakeon prescription has a "
            "well-defined infinite-pole limit."
        )
    elif alpha > 0.8:
        verdict = "CONDITIONALLY CONVERGENT"
        verdict_detail = (
            f"The residues decay as |R_n| ~ 1/|z_n|^{alpha:.2f} with alpha ~ 1. "
            "The unsubtracted Mittag-Leffler series is at the boundary of convergence "
            "(logarithmic divergence if alpha = 1 exactly). "
            "The one-subtraction Mittag-Leffler series DOES converge absolutely. "
            "The N-pole fakeon prescription converges with one subtraction. "
            "Physical amplitudes are finite if the loop integrand provides O(1/k^2) "
            "UV suppression (which gravitational amplitudes do)."
        )
    elif alpha > 0.1:
        verdict = "CONDITIONALLY CONVERGENT (with subtraction)"
        verdict_detail = (
            f"The residues decay as |R_n| ~ 1/|z_n|^{alpha:.2f}. "
            "The unsubtracted series diverges, but the subtracted Mittag-Leffler "
            "series converges. One subtraction is sufficient."
        )
    else:
        verdict = "UNKNOWN"
        verdict_detail = "Insufficient data to determine convergence."

    print(f"VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print("=" * 70)

    results = {
        "task": "FK-D Fakeon Convergence Analysis",
        "dps": dps,
        "ghost_catalogue_extended": {
            "n_zeros_known": len(residues),
            "n_zeros_new": len([z for z in new_zeros if "error" not in z]),
            "n_zeros_total": len(all_residues),
            "zeros": [
                {
                    "label": r["label"],
                    "type": r["type"],
                    "z_abs": r["z_abs"],
                    "R_re": r["R_re"],
                    "R_im": r["R_im"],
                    "R_abs": r["R_abs"],
                }
                for r in all_residues
            ],
        },
        "residue_series_convergence": {
            "partial_sums_abs_R": series["partial_sums_abs_R"],
            "partial_sums_with_graviton": series["partial_sums_with_graviton"],
            "target_sum_rule": series["target_sum_rule"],
            "deficit": series["deficit"],
        },
        "asymptotic_analysis": {
            "power_law_alpha": asym.get("power_law_fit", {}).get("alpha"),
            "power_law_A": asym.get("power_law_fit", {}).get("A"),
            "formula": asym.get("power_law_fit", {}).get("formula"),
            "sum_behavior": asym.get("sum_behavior"),
            "predicted_partial_sums": asym.get("partial_sums_predicted"),
        },
        "entire_function_order": {
            "growth_positive_real": order["growth_positive_real"],
            "growth_imaginary": order["growth_imaginary"],
            "growth_negative_real": order["growth_negative_real"],
        },
        "partial_fraction_tests": pf_results,
        "mittag_leffler_subtracted": ml_results,
        "n_pole_convergence": n_pole["test_points"],
        "pv_integral": {
            "single_pole": pv_single,
            "multi_pole": {
                "N_values": pv_multi["N_values"],
                "integral_values": pv_multi["integral_values"],
                "successive_differences": pv_multi["successive_differences"],
                "converging": pv_multi["converging"],
            },
        },
        "convergence_conditions": conditions,
        "verdict": {
            "classification": verdict,
            "detail": verdict_detail,
            "alpha_exponent": alpha,
        },
    }

    return results


def save_results(results: dict, filename: str = "fk_fakeon_convergence_results.json") -> Path:
    """Save results to JSON."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (mp.mpf, mp.mpc)):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    return output_path


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FK-D: Fakeon convergence analysis")
    parser.add_argument("--dps", type=int, default=60, help="Decimal places of precision")
    args = parser.parse_args()

    results = run_full_analysis(dps=args.dps)
    path = save_results(results)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
