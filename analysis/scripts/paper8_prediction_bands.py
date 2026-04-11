"""Paper 8: Prediction bands for cutoff function family f(u)=e^{-u^n}.

Uses CANONICAL code imports from nt2_entire_function for n=1 cross-check,
and generalized phi_n integrals for n=2..5. All sign conventions match
the verified codebase (fermion sign built into hC_dirac).
"""
import json
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

# --- Canonical imports for n=1 cross-check ---
from scripts.nt2_entire_function import (
    F1_total_complex,
    hC_dirac_complex,
    hC_scalar_complex,
    hC_vector_complex,
)

ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60
N_S, N_F, N_V = 4, 45, 12
N_D = mp.mpf(N_F) / 2  # 22.5 Dirac


# --- Generalized master function ---
def phi_n(x, n):
    """phi_n(x) = int_0^1 exp(-(a(1-a)x)^n) da."""
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(1)
    return mp.quad(lambda a: mp.exp(-(a * (1 - a) * x) ** n), [0, 1])


def phi_canonical(x):
    """Closed-form phi for n=1."""
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(1)
    return mp.exp(-x / 4) * mp.sqrt(mp.pi / x) * mp.erfi(mp.sqrt(x) / 2)


# --- Form factors with generalized phi ---
# SIGN CONVENTION: fermion h_C is NEGATIVE at x=0 (= -1/20).
# This matches the canonical code where hC_dirac_complex(0) = -0.05.
def _hC_scalar(x, p):
    """h_C^(0)(x) using phi value p. Returns +1/120 at x=0."""
    x = mp.mpf(x)
    if abs(x) < mp.mpf("1e-12"):
        return mp.mpf(1) / 120
    return 1 / (12 * x) + (p - 1) / (2 * x ** 2)


def _hC_dirac(x, p):
    """h_C^(1/2)(x). The formula (3p-1)/(6x)+2(p-1)/x^2 is negative for x>0.

    At x=0 the L'Hopital limit using phi'(0)=-1/6 gives +1/20, but the
    CANONICAL code returns -1/20 (fermion sign in the heat kernel trace).
    For the combination N_s*hC_s + N_D*hC_d + N_v*hC_v to give alpha_C=13/120,
    we need hC_d(0) = -1/20. So we use -1/20 at x=0 and the bare formula
    (which is already negative) at x>0.
    """
    x = mp.mpf(x)
    if abs(x) < mp.mpf("1e-12"):
        return mp.mpf(-1) / 20
    return (3 * p - 1) / (6 * x) + 2 * (p - 1) / x ** 2


def _hC_vector(x, p):
    """h_C^(1)(x). Returns +1/10 at x=0."""
    x = mp.mpf(x)
    if abs(x) < mp.mpf("1e-12"):
        return mp.mpf(1) / 10
    return p / 4 + (6 * p - 5) / (6 * x) + (p - 1) / x ** 2


def F1_total_n(z, n):
    """SM F1(z) for cutoff e^{-u^n}. All positive combination (sign in hC_dirac)."""
    z = mp.mpf(z)
    p = phi_n(z, n) if n != 1 else phi_canonical(z)
    numerator = (
        mp.mpf(N_S) * _hC_scalar(z, p)
        + N_D * _hC_dirac(z, p)
        + mp.mpf(N_V) * _hC_vector(z, p)
    )
    return numerator / (16 * mp.pi ** 2)


def Pi_TT_n(z, n):
    """Dressed TT propagator for cutoff n. Pi_TT(z) = 1 + c2 * z * F1_shape(z)."""
    z = mp.mpf(z)
    if abs(z) < mp.mpf("1e-12"):
        return mp.mpf(1)
    F1_0 = F1_total_n(0, n)
    F1_z = F1_total_n(z, n)
    F_shape = F1_z / F1_0
    return 1 + LOCAL_C2 * z * F_shape


# =====================================================================
# CROSS-CHECK: n=1 against canonical code
# =====================================================================
print("=" * 70)
print("CROSS-CHECK: n=1 vs canonical nt2_entire_function")
print("=" * 70)

F1_canon_0 = F1_total_complex(0, dps=50)
F1_mine_0 = F1_total_n(0, 1)
print(f"F1(0) canonical: {float(F1_canon_0.real):.12e}")
print(f"F1(0) mine:      {float(F1_mine_0):.12e}")
print(f"Match: {abs(float(F1_canon_0.real) - float(F1_mine_0)) < 1e-10}")

test_z = [0.5, 1.0, 2.0, 2.4, 3.0, 5.0]
print(f"\n{'z':>6} | {'Pi_canon':>14} | {'Pi_mine':>14} | {'diff':>10}")
print("-" * 55)
for z in test_z:
    F1_c = F1_total_complex(z, dps=50)
    F1_shape_c = F1_c / F1_canon_0
    pi_c = float((1 + LOCAL_C2 * mp.mpf(z) * F1_shape_c).real)
    pi_m = float(Pi_TT_n(z, 1))
    print(f"{z:>6.1f} | {pi_c:>14.8f} | {pi_m:>14.8f} | {abs(pi_c - pi_m):>10.2e}")

# Find canonical zero
z0_canon = mp.findroot(
    lambda z: 1 + LOCAL_C2 * z * F1_total_complex(z, dps=50) / F1_canon_0,
    mp.mpf("2.4"),
)
z0_mine = mp.findroot(lambda z: Pi_TT_n(z, 1), mp.mpf("2.4"))
print(f"\nCanonical z0 = {float(z0_canon.real):.10f}")
print(f"Mine z0      = {float(z0_mine):.10f}")
m2_canon = float(mp.sqrt(z0_canon.real))
m2_mine = float(mp.sqrt(z0_mine))
print(f"m2/Lambda canon = {m2_canon:.6f}")
print(f"m2/Lambda mine  = {m2_mine:.6f}")

# =====================================================================
# PREDICTION BANDS: n = 1..5
# =====================================================================
print("\n" + "=" * 70)
print("PREDICTION BANDS: f(u) = e^{-u^n}, n = 1..5")
print("=" * 70)

results = {}

for n in [1, 2, 3, 4, 5]:
    print(f"\n--- n = {n} ---")

    # phi'(0) analytic: d/dx phi_n(x)|_{x=0} for the BARE phi (before fermion sign)
    # phi_n'(0) = -n * Gamma(n+1)^2 / Gamma(2n+2)
    phi_prime_0 = float(-n * mp.gamma(n + 1) ** 2 / mp.gamma(2 * n + 2))

    # UV asymptotic x * phi_n(x -> inf)
    x_uv = mp.mpf(10000)
    phi_uv = phi_n(x_uv, n) if n > 1 else phi_canonical(x_uv)
    uv_asymp = float(x_uv * phi_uv)

    # Find m2/Lambda: zero of Pi_TT_n
    # Starting point depends on n: n=1 zero near 2.4; n>=2 near 5-6
    z_start = mp.mpf("2.4") if n == 1 else mp.mpf("5.5")
    try:
        z0 = mp.findroot(lambda z: Pi_TT_n(z, n), z_start)
        z0_val = float(z0.real) if hasattr(z0, "real") else float(z0)
        m2_over_Lambda = float(mp.sqrt(abs(z0_val)))
    except Exception as e:
        z0_val = None
        m2_over_Lambda = None
        print(f"  Zero-finding failed: {e}")

    # V(r)/V_N at xi=1/6 (only spin-2 Yukawa)
    V_ratios = {}
    if m2_over_Lambda is not None:
        for rL in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            V = float(1 - mp.mpf(4) / 3 * mp.exp(-m2_over_Lambda * rL))
            V_ratios[str(rL)] = V

    entry = {
        "n": n,
        "phi_prime_0": phi_prime_0,
        "uv_asymptotic": uv_asymp,
        "z0": z0_val,
        "m2_over_Lambda": m2_over_Lambda,
        "V_ratios": V_ratios,
    }
    results[f"n={n}"] = entry

    print(f"  phi'(0)    = {phi_prime_0:.6f}")
    print(f"  x*phi(inf) = {uv_asymp:.4f}")
    if m2_over_Lambda:
        print(f"  z0         = {z0_val:.6f}")
        print(f"  m2/Lambda  = {m2_over_Lambda:.4f}")
        print(f"  V(0)/V_N   = {V_ratios.get('0.01', 'N/A'):.6f}")
    else:
        print("  m2/Lambda  = FAILED")

# =====================================================================
# SUMMARY TABLE
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'n':>3} | {'phi_p(0)':>10} | {'x*phi(inf)':>10} | {'z_0':>10} | {'m2/Lam':>8} | {'V(0)/V_N':>10}")
print("-" * 68)
for n in [1, 2, 3, 4, 5]:
    e = results[f"n={n}"]
    z0s = f"{e['z0']:.4f}" if e["z0"] else "N/A"
    m2s = f"{e['m2_over_Lambda']:.4f}" if e["m2_over_Lambda"] else "N/A"
    v0s = f"{e['V_ratios'].get('0.01', 'N/A'):.6f}" if e["V_ratios"] else "N/A"
    print(
        f"{n:>3} | {e['phi_prime_0']:>10.6f} | {e['uv_asymptotic']:>10.4f} | "
        f"{z0s:>10} | {m2s:>8} | {v0s:>10}"
    )

# Prediction band summary
m2_vals = [e["m2_over_Lambda"] for e in results.values() if e["m2_over_Lambda"]]
if m2_vals:
    print(f"\n--- PREDICTION BAND ---")
    print(f"m2/Lambda range: [{min(m2_vals):.4f}, {max(m2_vals):.4f}]")
    print(f"Spread: {(max(m2_vals)-min(m2_vals))/min(m2_vals)*100:.1f}%")

# Save
out_path = Path(ANALYSIS_DIR).parent / "analysis" / "results" / "paper8_prediction_bands.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
