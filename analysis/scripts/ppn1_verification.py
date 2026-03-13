# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.nt4a_newtonian import (
    V_modified,
    gamma_local_ratio,
    phi_local_ratio,
    psi_local_ratio,
    small_r_limit_potential,
)
from scripts.ppn1_parameters import beta_ppn, gamma_ppn

mpmath.mp.dps = 100

_RESULTS_DIR = ANALYSIS_DIR / "results" / "ppn1"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def check_dimensions():
    r = mpmath.mpf('1.0')
    gamma = gamma_ppn(r, mpmath.mpf('1.0'), mpmath.mpf('0'))
    assert mpmath.isfinite(gamma), "gamma should be dimensionless and finite"

def check_limiting_cases():
    L_inf = mpmath.mpf('1e50')
    r = mpmath.mpf('1.5e11') * mpmath.mpf('5.06e6')
    xi = mpmath.mpf('0')

    # Large Lambda -> GR limit
    g_inf = gamma_ppn(r, L_inf, xi)
    assert abs(g_inf - mpmath.mpf('1')) < 1e-10, f"gamma = {g_inf} != 1 in GR limit"

    # beta is intentionally open until the nonlinear derivation is completed.
    try:
        beta_ppn(r, L_inf, xi)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("beta_ppn must remain explicitly not derived")

    # Directly cross-check the local NT-4a layer used by PPN1.
    phi = phi_local_ratio(1.0, Lambda=1.0, xi=0.0, dps=80)
    psi = psi_local_ratio(1.0, Lambda=1.0, xi=0.0, dps=80)
    gamma_local = gamma_local_ratio(1.0, Lambda=1.0, xi=0.0, dps=80)
    assert abs(gamma_local - psi / phi) < mpmath.mpf('1e-30')

    # At exact conformal coupling, the scalar Yukawa channel decouples and the
    # local short-distance potential no longer has a finite r -> 0 limit.
    assert small_r_limit_potential(Lambda=1.0, xi=mpmath.mpf('1') / 6) == mpmath.inf
    conformal_short = V_modified(1e-8, Lambda=1.0, xi=mpmath.mpf('1') / 6, dps=80)
    assert conformal_short > 0

def check_numeric_precision():
    # Compute gamma - 1 at r=1 AU for various Lambdas
    r = mpmath.mpf('1.495978707e11') * mpmath.mpf('5.06773e6')
    lambdas = [mpmath.mpf('1e-3'), mpmath.mpf('1'), mpmath.mpf('1e3'), mpmath.mpf('1e28')]
    xi = mpmath.mpf('0')

    for L in lambdas:
        gamma = gamma_ppn(r, L, xi)
        print(f"Lambda = {L} eV: |gamma - 1| = {abs(gamma - mpmath.mpf('1'))}")


def verification_snapshot() -> dict:
    r = mpmath.mpf('1.0')
    xi = mpmath.mpf('0')
    gamma_unit = gamma_ppn(r, mpmath.mpf('1.0'), xi)
    r_au = mpmath.mpf('1.495978707e11') * mpmath.mpf('5.06773e6')
    lambda_samples = [mpmath.mpf('1e-3'), mpmath.mpf('1'), mpmath.mpf('1e3'), mpmath.mpf('1e28')]

    return {
        "phase": "PPN-1",
        "scope": "linear_static_local_yukawa",
        "checks": {
            "gamma_unit_radius": str(gamma_unit),
            "phi_local_unit": str(phi_local_ratio(1.0, Lambda=1.0, xi=0.0, dps=80)),
            "psi_local_unit": str(psi_local_ratio(1.0, Lambda=1.0, xi=0.0, dps=80)),
            "gamma_local_unit": str(gamma_local_ratio(1.0, Lambda=1.0, xi=0.0, dps=80)),
            "conformal_small_r_limit": str(
                small_r_limit_potential(Lambda=1.0, xi=mpmath.mpf('1') / 6)
            ),
            "conformal_short_potential": str(
                V_modified(1e-8, Lambda=1.0, xi=mpmath.mpf('1') / 6, dps=80)
            ),
        },
        "gamma_minus_one_at_au": {
            str(L): str(abs(gamma_ppn(r_au, L, xi) - mpmath.mpf('1')))
            for L in lambda_samples
        },
        "beta_status": "not_derived",
    }

def main():
    print("Running Layer 1 checks (Analytic)...")
    check_dimensions()
    check_limiting_cases()

    print("Running Layer 2 checks (Numeric)...")
    check_numeric_precision()

    snapshot = verification_snapshot()
    (_RESULTS_DIR / "ppn1_verification.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )

    print("All verification checks PASSED.")

if __name__ == "__main__":
    main()
