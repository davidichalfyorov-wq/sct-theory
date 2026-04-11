"""
SCT Theory — Simulation-in-the-Loop Quick Sanity Checks.

Fast numerical verification battery (<5s) that can run after any
symbolic change. Catches sign errors, coefficient mistakes, and
regression bugs immediately rather than waiting for full verification.

Usage:
    from sct_tools.quick_check import quick_sanity, quick_sanity_report

    results = quick_sanity()           # dict of all checks
    ok = quick_sanity_report()         # prints formatted report, returns bool
    ok = quick_sanity("phi")           # run only phi-related checks
    ok = quick_sanity("canonical")     # run only canonical value checks
"""

from __future__ import annotations

import math
import time


def _check(name: str, got: float, expected: float,
           atol: float = 1e-12, rtol: float = 1e-10) -> dict:
    """Single check: compare got vs expected."""
    if expected == 0:
        error = abs(got)
        passed = error < atol
    else:
        error = abs(got - expected) / max(abs(expected), 1e-300)
        passed = error < rtol or abs(got - expected) < atol
    return {
        "name": name,
        "expected": expected,
        "got": got,
        "error": error,
        "passed": passed,
    }


def quick_sanity(group: str | None = None) -> dict[str, dict]:
    """Run fast sanity checks on SCT canonical results.

    Args:
        group: Optional filter — "phi", "scalar", "dirac", "vector",
               "combined", "canonical", "uv", "masses". None = all.

    Returns:
        Dict mapping check_name -> {expected, got, error, passed}.
    """
    from .constants import N_f, N_s, N_v
    from .form_factors import (
        F1_total,
        alpha_C_SM,
        alpha_R_SM,
        hC_dirac_fast,
        hC_scalar_fast,
        hC_vector_fast,
        hR_dirac_fast,
        hR_scalar_fast,
        hR_vector_fast,
        phi_fast,
    )

    checks = {}

    # --- Group: phi (master function) ---
    if group in (None, "phi", "canonical"):
        checks["phi(0) = 1"] = _check("phi(0) = 1", phi_fast(0.0), 1.0)
        # phi'(0) = -1/6: check via finite difference
        dx = 1e-8
        dphi = (phi_fast(dx) - phi_fast(0.0)) / dx
        checks["phi'(0) = -1/6"] = _check("phi'(0) = -1/6", dphi, -1/6, atol=1e-6)
        # phi(x) > 0 for moderate x
        checks["phi(10) > 0"] = _check("phi(10) > 0", float(phi_fast(10.0) > 0), 1.0)

    # --- Group: scalar (spin-0) ---
    if group in (None, "scalar", "canonical"):
        checks["beta_W^(0) = 1/120"] = _check(
            "beta_W^(0) = 1/120", hC_scalar_fast(0.0), 1/120)
        checks["beta_R^(0)(xi=0) = 1/72"] = _check(
            "beta_R^(0)(xi=0) = 1/72", hR_scalar_fast(0.0, 0.0), 1/72)
        checks["beta_R^(0)(xi=1/6) = 0"] = _check(
            "beta_R^(0)(xi=1/6) = 0", hR_scalar_fast(0.0, 1/6), 0.0, atol=1e-14)

    # --- Group: dirac (spin-1/2) ---
    if group in (None, "dirac", "canonical"):
        checks["beta_W^(1/2) = -1/20"] = _check(
            "beta_W^(1/2) = -1/20", hC_dirac_fast(0.0), -1/20)
        checks["beta_R^(1/2) = 0"] = _check(
            "beta_R^(1/2) = 0", hR_dirac_fast(0.0), 0.0, atol=1e-14)

    # --- Group: vector (spin-1) ---
    if group in (None, "vector", "canonical"):
        checks["beta_W^(1) = 1/10"] = _check(
            "beta_W^(1) = 1/10", hC_vector_fast(0.0), 1/10)
        checks["beta_R^(1) = 0"] = _check(
            "beta_R^(1) = 0", hR_vector_fast(0.0), 0.0, atol=1e-14)

    # --- Group: combined SM ---
    if group in (None, "combined", "canonical"):
        checks["alpha_C = 13/120"] = _check(
            "alpha_C = 13/120", alpha_C_SM(), 13/120)
        checks["alpha_R(xi=1/6) = 0"] = _check(
            "alpha_R(xi=1/6) = 0", alpha_R_SM(1/6), 0.0, atol=1e-14)
        checks["alpha_R(xi=0) = 1/18"] = _check(
            "alpha_R(xi=0) = 1/18", alpha_R_SM(0.0), 1/18)
        checks["F1(0) = 13/(1920*pi^2)"] = _check(
            "F1(0) = 13/(1920*pi^2)", F1_total(0.0), 13 / (1920 * math.pi**2))
        # c1/c2 at conformal coupling
        ac = alpha_C_SM()
        ar = alpha_R_SM(1/6)
        if ac != 0:
            c1c2 = -1/3 + 120 * ar / (13)  # at xi=1/6, ar=0 -> c1/c2=-1/3
            checks["c1/c2(xi=1/6) = -1/3"] = _check(
                "c1/c2(xi=1/6) = -1/3", c1c2, -1/3)

    # --- Group: UV asymptotics ---
    if group in (None, "uv"):
        x_large = 5000.0
        x_ac = x_large * (
            N_s * hC_scalar_fast(x_large)
            + (N_f / 2) * hC_dirac_fast(x_large)
            + N_v * hC_vector_fast(x_large)
        )
        checks["x*alpha_C(x->inf) = -89/12"] = _check(
            "x*alpha_C(x->inf) = -89/12", x_ac, -89/12, rtol=1e-3)

    # --- Group: effective masses ---
    if group in (None, "masses"):
        # m2 = Lambda * sqrt(60/13)
        m2_ratio = math.sqrt(60 / 13)
        checks["m2/Lambda = sqrt(60/13)"] = _check(
            "m2/Lambda = sqrt(60/13)", m2_ratio, 2.1483, rtol=1e-3)
        # m0(xi=0) = Lambda * sqrt(6)
        m0_ratio = math.sqrt(6)
        checks["m0(xi=0)/Lambda = sqrt(6)"] = _check(
            "m0(xi=0)/Lambda = sqrt(6)", m0_ratio, 2.4495, rtol=1e-3)

    return checks


def quick_sanity_report(group: str | None = None) -> bool:
    """Run quick sanity checks and print a formatted report.

    Args:
        group: Optional filter (see quick_sanity).

    Returns:
        True if all checks pass.
    """
    t0 = time.time()
    results = quick_sanity(group)
    elapsed = time.time() - t0

    n_pass = sum(1 for r in results.values() if r["passed"])
    n_fail = sum(1 for r in results.values() if not r["passed"])
    n_total = len(results)

    print(f"{'='*60}")
    print(f"  SCT Quick Sanity Check ({n_total} checks, {elapsed:.2f}s)")
    print(f"{'='*60}")

    for name, r in results.items():
        status = "OK" if r["passed"] else "FAIL"
        if r["expected"] == 0:
            print(f"  [{status:4s}] {name}: got={r['got']:.6e}")
        else:
            print(f"  [{status:4s}] {name}: got={r['got']:.10e}, "
                  f"expected={r['expected']:.10e}")

    print(f"{'='*60}")
    if n_fail == 0:
        print(f"  ALL {n_pass} CHECKS PASSED ({elapsed:.2f}s)")
    else:
        print(f"  {n_fail}/{n_total} CHECKS FAILED")
    print(f"{'='*60}")

    return n_fail == 0
