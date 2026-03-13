from __future__ import annotations

import json
from pathlib import Path

from sympy import Rational, exp, simplify, symbols

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ppn1"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def verify_local_ppn_structure() -> dict:
    r, m2, m0 = symbols("r m2 m0", positive=True, real=True)

    phi = 1 - Rational(4, 3) * exp(-m2 * r) + Rational(1, 3) * exp(-m0 * r)
    psi = 1 - Rational(2, 3) * exp(-m2 * r) - Rational(1, 3) * exp(-m0 * r)
    gamma = simplify(psi / phi)
    conformal_phi = 1 - Rational(4, 3) * exp(-m2 * r)

    print("Phi_local =", phi)
    print("Psi_local =", psi)
    print("gamma_local =", gamma)

    phi_gr = simplify(phi.subs({exp(-m2 * r): 0, exp(-m0 * r): 0}))
    psi_gr = simplify(psi.subs({exp(-m2 * r): 0, exp(-m0 * r): 0}))
    short_distance_generic = simplify(phi.subs({exp(-m2 * r): 1, exp(-m0 * r): 1}))
    short_distance_conformal = simplify(conformal_phi.subs({exp(-m2 * r): 1}))
    gamma_conformal_short = simplify((1 - Rational(2, 3)) / (1 - Rational(4, 3)))

    assert phi_gr == 1
    assert psi_gr == 1
    assert short_distance_generic == 0
    assert short_distance_conformal == -Rational(1, 3)
    assert gamma_conformal_short == -1

    print("GR limit Phi =", phi_gr)
    print("GR limit Psi =", psi_gr)
    print("Generic short-distance cancellation coefficient =", short_distance_generic)
    print("Conformal short-distance coefficient =", short_distance_conformal)
    print("Conformal short-distance gamma =", gamma_conformal_short)

    return {
        "phase": "PPN-1",
        "scope": "linear_static_local_yukawa",
        "phi_local": str(phi),
        "psi_local": str(psi),
        "gamma_local": str(gamma),
        "gr_limit_phi": str(phi_gr),
        "gr_limit_psi": str(psi_gr),
        "generic_short_distance_coefficient": str(short_distance_generic),
        "conformal_short_distance_coefficient": str(short_distance_conformal),
        "conformal_short_distance_gamma": str(gamma_conformal_short),
    }


if __name__ == "__main__":
    snapshot = verify_local_ppn_structure()
    (_RESULTS_DIR / "ppn1_symbolic.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )
