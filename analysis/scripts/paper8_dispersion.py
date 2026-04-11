"""
Graviton dispersion relation in Spectral Causal Theory (SCT).

Physics: The linearized equation of motion for the transverse-traceless
graviton mode reads

    Pi_TT(Box/Lambda^2) Box h_{mu nu} = 0

In Fourier space with z = k^2/Lambda^2 (Euclidean signature), the
on-shell condition factorizes:

    Pi_TT(z) * z = 0

yielding two branches:
  (1) z = 0  -->  omega = |k|  (massless graviton, speed = c, always present)
  (2) Pi_TT(z_0) = 0  -->  massive mode at m_2 = Lambda * sqrt(z_0) (fakeon)

Key results:
  - On-shell gravitons propagate at exactly c (z=0 branch is unmodified)
  - No birefringence (Pi_TT is helicity-independent)
  - Signal velocity = c (Paley-Wiener + entireness of form factors)
  - Massive branch is a Lee-Wick ghost (fakeon prescription eliminates it)
  - GW170817 compatible: |v - c|/c = 0 exactly

SM field content: N_s = 4, N_D = N_f/2 = 22.5, N_v = 12.
alpha_C = 13/120 (parameter-free, xi-independent).
Pi_TT(z) = 1 + (13/60) * z * F_hat_1(z), where F_hat_1 = alpha_C(z)/alpha_C(0).

David Alfyorov, Igor Shnyukov
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DPS = 50
mp.mp.dps = DPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"

# SM multiplicities (CPR 0805.2909)
N_S = mp.mpf(4)       # real scalar d.o.f. (Higgs doublet)
N_D = mp.mpf("22.5")  # Dirac fermions = N_f/2 = 45/2
N_V = mp.mpf(12)      # vector d.o.f.
ALPHA_C_0 = mp.mpf(13) / 120  # verified canonical value
LOCAL_C2 = 2 * ALPHA_C_0       # 13/60, spin-2 local coefficient


# ---------------------------------------------------------------------------
# Master function phi(z) = e^{-z/4} sqrt(pi/z) erfi(sqrt(z)/2)
# ---------------------------------------------------------------------------
def phi(z: mp.mpc | mp.mpf) -> mp.mpc:
    """Master function, entire, phi(0) = 1, phi'(0) = -1/6."""
    z = mp.mpc(z)
    if z == 0:
        return mp.mpc(1)
    return mp.exp(-z / 4) * mp.sqrt(mp.pi / z) * mp.erfi(mp.sqrt(z) / 2)


def phi_series(z: mp.mpc, n_terms: int = 60) -> mp.mpc:
    """Evaluate phi(z) via convergent Taylor series: sum_n a_n z^n,
    a_n = (-1)^n n! / (2n+1)!."""
    z = mp.mpc(z)
    total = mp.mpc(0)
    for n in range(n_terms):
        total += (-1) ** n * mp.factorial(n) / mp.factorial(2 * n + 1) * z ** n
    return total


# ---------------------------------------------------------------------------
# Individual spin form factors h_C^(s)(z)
# ---------------------------------------------------------------------------
def _phi_coeff(n: int) -> mp.mpf:
    """Taylor coefficient a_n of phi(z): a_n = (-1)^n n! / (2n+1)!."""
    return (-1) ** n * mp.factorial(n) / mp.factorial(2 * n + 1)


def hC_scalar(z: mp.mpc) -> mp.mpc:
    """h_C^{(0)}(z) = 1/(12z) + (phi-1)/(2z^2). Local limit: 1/120.

    Series: h_C^(0)(z) = sum_{n>=0} a_{n+2}/2 * z^n
    (the 1/z poles cancel: 1/12 + a_1/2 = 1/12 - 1/12 = 0).
    """
    z = mp.mpc(z)
    if abs(z) < mp.mpf("0.1"):
        total = mp.mpc(0)
        for n in range(50):
            total += _phi_coeff(n + 2) / 2 * z ** n
        return total
    p = phi(z)
    return mp.mpf(1) / (12 * z) + (p - 1) / (2 * z ** 2)


def hC_dirac(z: mp.mpc) -> mp.mpc:
    """h_C^{(1/2)}(z) = (3*phi-1)/(6z) + 2(phi-1)/z^2. Local limit: -1/20.

    Series: h_C^(1/2)(z) = sum_{n>=0} [a_{n+1}/2 + 2*a_{n+2}] * z^n
    (1/z poles cancel: 1/3 + 2*a_1 = 1/3 - 1/3 = 0).
    """
    z = mp.mpc(z)
    if abs(z) < mp.mpf("0.1"):
        total = mp.mpc(0)
        for n in range(50):
            total += (_phi_coeff(n + 1) / 2 + 2 * _phi_coeff(n + 2)) * z ** n
        return total
    p = phi(z)
    return (3 * p - 1) / (6 * z) + 2 * (p - 1) / z ** 2


def hC_vector(z: mp.mpc) -> mp.mpc:
    """h_C^{(1)}(z) = phi/4 + (6*phi-5)/(6z) + (phi-1)/z^2. Local limit: 1/10.

    Series: h_C^(1)(z) = sum_{n>=0} [a_n/4 + a_{n+1} + a_{n+2}] * z^n
    (1/z poles cancel: 1/6 + a_1 = 1/6 - 1/6 = 0).
    """
    z = mp.mpc(z)
    if abs(z) < mp.mpf("0.1"):
        total = mp.mpc(0)
        for n in range(50):
            total += (_phi_coeff(n) / 4 + _phi_coeff(n + 1) + _phi_coeff(n + 2)) * z ** n
        return total
    p = phi(z)
    return p / 4 + (6 * p - 5) / (6 * z) + (p - 1) / z ** 2


# ---------------------------------------------------------------------------
# Combined SM alpha_C(z) and Pi_TT(z)
# ---------------------------------------------------------------------------
def alpha_C(z: mp.mpc) -> mp.mpc:
    """Total SM Weyl coefficient: alpha_C(z) = N_s*hC0 + N_D*hC1/2 + N_v*hC1.
    alpha_C(0) = 13/120."""
    return N_S * hC_scalar(z) + N_D * hC_dirac(z) + N_V * hC_vector(z)


def Pi_TT(z: mp.mpc) -> mp.mpc:
    """Spin-2 kinetic denominator.

    Pi_TT(z) = 1 + c_2 * z * [alpha_C(z) / alpha_C(0)]
             = 1 + 2 * z * alpha_C(z)

    where c_2 = 2*alpha_C(0) = 13/60.
    Pi_TT(0) = 1 by construction (canonical normalization).
    """
    z = mp.mpc(z)
    return 1 + 2 * z * alpha_C(z)


# ---------------------------------------------------------------------------
# Root finder for Pi_TT(z) = 0 on positive real axis
# ---------------------------------------------------------------------------
def find_first_positive_zero(z_max: float = 20.0, step: float = 0.02) -> mp.mpf:
    """Locate the first positive real zero of Pi_TT."""
    z_left = mp.mpf("0.1")
    val_left = mp.re(Pi_TT(z_left))
    z = z_left + mp.mpf(step)
    while z <= mp.mpf(z_max):
        val_right = mp.re(Pi_TT(z))
        if val_left * val_right < 0:
            return mp.findroot(lambda t: mp.re(Pi_TT(t)), (z_left, z))
        z_left = z
        val_left = val_right
        z += mp.mpf(step)
    raise ValueError(f"No positive-real zero found in (0.1, {z_max}]")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def main() -> dict:
    mp.mp.dps = DPS
    results = {}

    # ------------------------------------------------------------------
    # 1. Verify alpha_C(0) = 13/120
    # ------------------------------------------------------------------
    a0 = alpha_C(mp.mpf(0))
    a0_exact = mp.mpf(13) / 120
    a0_err = abs(a0 - a0_exact)
    print(f"alpha_C(0) = {mp.nstr(a0, 20)}")
    print(f"  exact 13/120 = {mp.nstr(a0_exact, 20)}")
    print(f"  |error| = {float(a0_err):.2e}")
    assert a0_err < mp.mpf("1e-25"), f"alpha_C(0) mismatch: {a0}"
    results["alpha_C_0"] = float(mp.re(a0))

    # Verify Pi_TT(0) = 1
    pi0 = Pi_TT(mp.mpf(0))
    print(f"\nPi_TT(0) = {mp.nstr(pi0, 20)}")
    assert abs(pi0 - 1) < mp.mpf("1e-25"), f"Pi_TT(0) != 1: {pi0}"
    results["Pi_TT_0"] = float(mp.re(pi0))

    # ------------------------------------------------------------------
    # 2. Compute Pi_TT(z) over z in [0.01, 10] at 200 points
    #    (z > 0: physical Euclidean region; z < 0 diverges due to 1/z poles)
    # ------------------------------------------------------------------
    N_points = 200
    z_min, z_max = 0.01, 10.0
    z_values = [mp.mpf(z_min) + mp.mpf(i) * mp.mpf(z_max - z_min) / (N_points - 1)
                for i in range(N_points)]
    pi_values = []
    for z in z_values:
        val = Pi_TT(z)
        pi_values.append(float(mp.re(val)))

    results["z_scan"] = [float(z) for z in z_values]
    results["Pi_TT_scan"] = pi_values

    print(f"\nPi_TT(z) computed at {N_points} points over [{z_min}, {z_max}]")

    # Also compute extended range [-10, 10] with appropriate care
    N_ext = 200
    z_ext = [mp.mpf(-10) + mp.mpf(i) * mp.mpf(20) / (N_ext - 1) for i in range(N_ext)]
    pi_ext = []
    for z in z_ext:
        val = Pi_TT(z)
        pi_ext.append(float(mp.re(val)))
    results["z_extended"] = [float(z) for z in z_ext]
    results["Pi_TT_extended"] = pi_ext

    # ------------------------------------------------------------------
    # 3. Check Pi_TT(z) > 0 for 0 < z < z_0 (no tachyonic instability
    #    in the physical region between massless and massive poles)
    # ------------------------------------------------------------------
    # First find z_0
    z0 = find_first_positive_zero()
    m2_over_Lambda = mp.sqrt(z0)

    # Check positivity for 0 < z < z_0
    z_check = [mp.mpf(i) * z0 / 100 for i in range(1, 100)]
    pi_check = [float(mp.re(Pi_TT(z))) for z in z_check]
    min_pi_physical = min(pi_check)
    all_positive_physical = all(p > 0 for p in pi_check)
    print(f"\nPhysical region positivity check (0 < z < z_0 = {float(z0):.6f}):")
    print(f"  min Pi_TT = {min_pi_physical:.10f}")
    print(f"  Pi_TT > 0 in (0, z_0)?  {'YES' if all_positive_physical else 'NO'}")
    results["physical_positivity"] = all_positive_physical
    results["min_Pi_TT_physical_region"] = min_pi_physical

    # ------------------------------------------------------------------
    # 4. Ghost mass from first positive zero
    # ------------------------------------------------------------------
    pi_at_z0 = Pi_TT(z0)
    print(f"\nFirst positive zero of Pi_TT:")
    print(f"  z_0 = {mp.nstr(z0, 20)}")
    print(f"  |Pi_TT(z_0)| = {float(abs(pi_at_z0)):.2e}")
    print(f"  m_2/Lambda = sqrt(z_0) = {mp.nstr(m2_over_Lambda, 15)}")

    # Compare with local approximation: at linear order,
    # Pi_TT(z) ~ 1 + (13/60)*z, so z_0^{local} = 60/13
    z0_local = mp.mpf(60) / 13
    m2_local = mp.sqrt(z0_local)
    print(f"\n  z_0 (local approx) = 60/13 = {mp.nstr(z0_local, 15)}")
    print(f"  m_2/Lambda (local approx) = sqrt(60/13) = {mp.nstr(m2_local, 15)}")
    print(f"  ratio z_0/z_0^local = {float(z0 / z0_local):.10f}")
    print(f"  ratio m_2/m_2^local = {float(m2_over_Lambda / m2_local):.10f}")

    results["z0_ghost"] = float(z0)
    results["z0_local_approx"] = float(z0_local)
    results["m2_over_Lambda"] = float(m2_over_Lambda)
    results["m2_over_Lambda_local"] = float(m2_local)
    results["Pi_TT_at_z0"] = float(abs(pi_at_z0))

    # ------------------------------------------------------------------
    # 5. Pi_TT at specific z values
    # ------------------------------------------------------------------
    test_points = {
        "z=-5":  mp.mpf(-5),
        "z=-1":  mp.mpf(-1),
        "z=0":   mp.mpf(0),
        "z=0.5": mp.mpf("0.5"),
        "z=1":   mp.mpf(1),
        "z=2":   mp.mpf(2),
        "z=z0":  z0,
        "z=5":   mp.mpf(5),
        "z=10":  mp.mpf(10),
    }

    print("\nPi_TT at selected points:")
    print(f"  {'z':>10s}  {'Pi_TT(z)':>22s}")
    print("  " + "-" * 35)
    pi_table = {}
    for label, z in test_points.items():
        val = Pi_TT(z)
        re_val = float(mp.re(val))
        im_val = float(mp.im(val))
        if abs(im_val) < 1e-40:
            print(f"  {label:>10s}  {re_val:>+22.15f}")
        else:
            print(f"  {label:>10s}  {re_val:>+22.15f} + {im_val:.2e}i")
        pi_table[label] = {"Re": re_val, "Im": im_val}
    results["Pi_TT_table"] = pi_table

    # ------------------------------------------------------------------
    # 6. Derivative at zero: Pi_TT'(0) = c_2 = 13/60
    # ------------------------------------------------------------------
    # Use symmetric finite difference at small h
    h = mp.mpf("1e-6")
    pi_prime_0 = mp.re(Pi_TT(h) - Pi_TT(-h)) / (2 * h)
    pi_prime_exact = LOCAL_C2  # 13/60
    print(f"\nPi_TT'(0):")
    print(f"  numerical  = {mp.nstr(pi_prime_0, 15)}")
    print(f"  exact 13/60 = {mp.nstr(pi_prime_exact, 15)}")
    err = abs(pi_prime_0 - pi_prime_exact)
    print(f"  |error| = {float(err):.2e}")
    results["Pi_TT_prime_0_numerical"] = float(pi_prime_0)
    results["Pi_TT_prime_0_exact"] = float(pi_prime_exact)

    # ------------------------------------------------------------------
    # 7. UV asymptotic: z * alpha_C(z) -> -89/12 as z -> inf
    # ------------------------------------------------------------------
    z_large = mp.mpf(1000)
    z_alpha = mp.re(z_large * alpha_C(z_large))
    uv_exact = mp.mpf(-89) / 12
    print(f"\nUV asymptotic x*alpha_C(x->inf):")
    print(f"  z=1000: z*alpha_C = {mp.nstr(z_alpha, 15)}")
    print(f"  exact = -89/12 = {mp.nstr(uv_exact, 15)}")
    print(f"  ratio = {float(z_alpha / uv_exact):.10f}")
    results["uv_asymptotic_z1000"] = float(z_alpha)
    results["uv_asymptotic_exact"] = float(uv_exact)

    # ------------------------------------------------------------------
    # 8. Additional zeros (Lee-Wick spectrum)
    # ------------------------------------------------------------------
    print("\nSearching for additional positive-real zeros of Pi_TT...")
    zeros_found = [z0]
    z_search = z0 + mp.mpf("0.1")
    val_prev = mp.re(Pi_TT(z_search))
    step = mp.mpf("0.05")
    while z_search <= 50:
        z_next = z_search + step
        val_next = mp.re(Pi_TT(z_next))
        if val_prev * val_next < 0:
            zn = mp.findroot(lambda t: mp.re(Pi_TT(t)), (z_search, z_next))
            zeros_found.append(zn)
            print(f"  z_{len(zeros_found)} = {mp.nstr(zn, 15)}, m/Lambda = {mp.nstr(mp.sqrt(zn), 10)}")
        z_search = z_next
        val_prev = val_next

    results["positive_real_zeros"] = [float(z) for z in zeros_found]
    results["masses_over_Lambda"] = [float(mp.sqrt(z)) for z in zeros_found]
    print(f"  Total positive-real zeros found in (0, 50]: {len(zeros_found)}")

    # ------------------------------------------------------------------
    # 9. Print physical conclusions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GRAVITON DISPERSION RELATION IN SCT")
    print("=" * 60)
    print()
    print("Linearized EOM:  Pi_TT(Box/Lambda^2) Box h_{mu nu} = 0")
    print("Momentum space:  Pi_TT(z) * z = 0,   z = k^2/Lambda^2")
    print()
    print("Branch 1:  z = 0  (massless graviton)")
    print("  --> omega = |k|  (unmodified dispersion relation)")
    print()
    print("Branch 2:  Pi_TT(z_n) = 0  (massive Lee-Wick modes)")
    for i, zn in enumerate(zeros_found):
        print(f"  --> z_{i + 1} = {float(zn):.10f},  m_{i + 1}/Lambda = {float(mp.sqrt(zn)):.10f}")
    print()
    print("Physical conclusions:")
    print()
    print("  Dispersion modified for on-shell gravitons?  NO")
    print("    (omega^2 = k^2 is always a solution via the z=0 branch)")
    print()
    print("  Birefringence?  NO")
    print("    (Pi_TT is helicity-independent: same for h_+ and h_x)")
    print()
    print("  Signal velocity = c?  YES")
    print("    (Paley-Wiener theorem + entireness of form factors)")
    print()
    print("  GW170817 compatible?  YES  (exact, |v - c|/c = 0)")
    print("    (massless branch travels at c; massive modes are fakeons)")
    print()
    print(f"  First ghost mass:  m_2/Lambda = {float(m2_over_Lambda):.10f}")
    print(f"  Local approximation: m_2/Lambda = sqrt(60/13) = {float(m2_local):.10f}")
    print(f"  Physical region (0 < z < z_0): Pi_TT > 0?  {'YES' if all_positive_physical else 'NO'}")
    print()
    print("  Pi_TT values at selected points:")
    for label, z in test_points.items():
        val = Pi_TT(z)
        print(f"    {label:>10s}:  Pi_TT = {float(mp.re(val)):+.15f}")

    # Collect summary
    results["conclusions"] = {
        "dispersion_modified": False,
        "birefringence": False,
        "signal_velocity_c": True,
        "GW170817_compatible": True,
        "delta_v_over_c": 0.0,
        "physical_region_positive": all_positive_physical,
        "ghost_mass_m2_over_Lambda": float(m2_over_Lambda),
        "ghost_mass_local_approx": float(m2_local),
        "ghost_type": "Lee-Wick (fakeon prescription)",
        "total_positive_real_zeros_0_to_50": len(zeros_found),
    }
    results["parameters"] = {
        "N_s": int(N_S),
        "N_D": float(N_D),
        "N_v": int(N_V),
        "alpha_C_0": float(ALPHA_C_0),
        "c2_local": float(LOCAL_C2),
        "dps": DPS,
        "N_scan_points": N_points,
        "z_scan_range": [z_min, z_max],
    }

    return results


if __name__ == "__main__":
    results = main()

    # Save to JSON
    out_path = RESULTS_DIR / "paper8_dispersion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, (mp.mpf, mp.mpc)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_serialize)

    print(f"\nResults saved to {out_path}")
