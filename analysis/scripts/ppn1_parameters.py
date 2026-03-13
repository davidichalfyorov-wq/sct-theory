# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.nt4a_newtonian import effective_masses, gamma_local_ratio

# Set mpmath precision
mpmath.mp.dps = 100

_RESULTS_DIR = ANALYSIS_DIR / "results" / "ppn1"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def c2() -> mpmath.mpf:
    return mpmath.mpf('13') / 60

def alpha_R(xi: mpmath.mpf) -> mpmath.mpf:
    return mpmath.mpf('2') * (xi - mpmath.mpf('1') / 6)**2

def m2_mass(Lambda: mpmath.mpf) -> mpmath.mpf:
    m2, _ = effective_masses(Lambda=float(Lambda), xi=0.0)
    return m2

def m0_mass(Lambda: mpmath.mpf, xi: mpmath.mpf) -> mpmath.mpf:
    _, m0 = effective_masses(Lambda=float(Lambda), xi=float(xi))
    if m0 is None:
        return mpmath.inf
    return m0

def gamma_ppn(r: mpmath.mpf, Lambda: mpmath.mpf, xi: mpmath.mpf) -> mpmath.mpf:
    """
    Compute the linear PPN parameter gamma in the local Yukawa approximation
    inherited from the NT-4a propagator layer.
    """
    return gamma_local_ratio(r, Lambda=Lambda, xi=xi, dps=mpmath.mp.dps)

def beta_ppn(r: mpmath.mpf, Lambda: mpmath.mpf, xi: mpmath.mpf) -> mpmath.mpf:
    """
    beta is not yet derived from the nonlinear SCT field equations.
    """
    raise NotImplementedError("PPN beta is not derived yet beyond local linear order")

def ppn_table(Lambda: mpmath.mpf, xi: mpmath.mpf) -> dict:
    r_au = mpmath.mpf('1.495978707e11') * mpmath.mpf('5.06773e6') # AU in eV^-1
    gamma_val = gamma_ppn(r_au, Lambda, xi)

    return {
        "scope": "linear_static_local_yukawa",
        "gamma": str(gamma_val),
        "beta": "not_derived",
        "beta_status": "open_nonlinear_problem",
        "xi_ppn": "not_derived",
        "alpha1": "not_derived",
        "alpha2": "not_derived",
        "alpha3": "not_derived",
        "zeta1": "not_derived",
        "zeta2": "not_derived",
        "zeta3": "not_derived",
        "zeta4": "not_derived"
    }


def ppn_snapshot(Lambda: mpmath.mpf, xi: mpmath.mpf) -> dict:
    return {
        "phase": "PPN-1",
        "scope": "linear_static_local_yukawa",
        "Lambda_eV": str(Lambda),
        "xi": str(xi),
        "m2_eV": str(m2_mass(Lambda)),
        "m0_eV": str(m0_mass(Lambda, xi)),
        "bounds_eV": {
            "cassini": lower_bound_Lambda("cassini"),
            "eot_wash": lower_bound_Lambda("eot-wash"),
        },
        "parameters": ppn_table(Lambda, xi),
    }

def lower_bound_Lambda(experiment: str) -> float:
    """
    Returns lower bound on Lambda in eV from various experiments.
    """
    if experiment == "cassini":
        # 1 m = 5.07e6 eV^-1
        # 1 AU = 1.5e11 m = 7.6e17 eV^-1
        # m_2 * r > ln(2/3 / 2.3e-5) ~ 10.27
        # m_2 > 10.27 / 7.6e17 = 1.35e-17 eV
        # Lambda > m_2 / sqrt(60/13) = 1.35e-17 / 2.148 ~ 6.2e-18 eV
        return 6.2e-18
    elif experiment == "eot-wash":
        # 1 m = 5.07e6 eV^-1
        # 50 um = 5e-5 * 5.07e6 = 253.5 eV^-1
        # m_2 > 1 / 253.5 = 3.94e-3 eV
        # Lambda > 3.94e-3 / 2.148 ~ 1.84e-3 eV
        return 1.84e-3
    return 0.0

def exclusion_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.fill_betweenx([1e-6, 1], 1e-19, 1.84e-3, color='red', alpha=0.3, label='Eöt-Wash Excluded')
    plt.fill_betweenx([1e-6, 1], 1e-19, 6.2e-18, color='blue', alpha=0.3, label='Cassini Excluded')
    plt.axvline(x=1.84e-3, color='red', linestyle='--')
    plt.axvline(x=6.2e-18, color='blue', linestyle='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Lambda$ [eV]')
    plt.ylabel(r'$|\gamma - 1|$')
    plt.title('SCT PPN Exclusion Plot')
    plt.legend()
    plt.savefig('analysis/figures/ppn1_exclusion.pdf')

def main():
    parser = argparse.ArgumentParser(description='Compute PPN parameters for SCT.')
    parser.add_argument('--Lambda', type=float, default=1e-3, help='Cutoff scale in eV')
    parser.add_argument('--xi', type=float, default=0.0, help='Non-minimal coupling')
    args = parser.parse_args()

    L_mp = mpmath.mpf(str(args.Lambda))
    xi_mp = mpmath.mpf(str(args.xi))

    snapshot = ppn_snapshot(L_mp, xi_mp)
    print(json.dumps(snapshot, indent=2))
    (_RESULTS_DIR / "ppn1_snapshot.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )
    exclusion_plot()

if __name__ == "__main__":
    main()
