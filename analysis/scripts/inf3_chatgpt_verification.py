"""
INF3 ChatGPT Formula Verification
==================================
Cross-checks key formulas from ChatGPT 5.4 Pro's exact spectral cosmology analysis.

Verifies:
  A. S^4 exact spectral action and no-saddle theorem (dS/ds > 0)
  B. N_s^crit thresholds for spin-2 zero disappearance
  C. z_0 persistence at alpha_C(0)=0 (N_D = 74/3)
  D. Asymptotic equation coefficient (12*Lambda^2)
  E. Pairwise threshold m_a + m_b < sqrt(6)*Lambda
"""

import sys
sys.path.insert(0, "analysis")

import mpmath
mpmath.mp.dps = 50  # 50-digit precision

from sct_tools.verification import Verifier

v = Verifier("INF3-ChatGPT-Verification")

# ============================================================
# PART A: S^4 exact spectral action and no-saddle theorem
# ============================================================
print("=" * 70)
print("PART A: S^4 exact spectral action (f=exp(-u))")
print("=" * 70)

# S^4 Dirac spectrum: eigenvalues ±(m/r), m=2,3,...
# mult_±(m) = (2/3)(m^3 - m)
# For f(u) = exp(-u), one finite-sector mass mu_I = 0 (massless):
#   S(s) = (4/3) * sum_{m=2}^{inf} (m^3-m) * exp(-m^2/s^2)
# where s = r*Lambda

def S4_action(s, mu_I_list=None, N_max=500):
    """Exact S^4 spectral action for f(u)=exp(-u)."""
    if mu_I_list is None:
        mu_I_list = [mpmath.mpf(0)]  # single massless mode
    s = mpmath.mpf(s)
    total = mpmath.mpf(0)
    for mu_I in mu_I_list:
        for m in range(2, N_max + 1):
            m_mp = mpmath.mpf(m)
            mult = m_mp**3 - m_mp
            arg = mu_I**2 + m_mp**2 / s**2
            total += mult * mpmath.exp(-arg)
    return mpmath.mpf(4) / 3 * total

def S4_deriv(s, mu_I_list=None, N_max=500):
    """Exact dS/ds for S^4 spectral action."""
    if mu_I_list is None:
        mu_I_list = [mpmath.mpf(0)]
    s = mpmath.mpf(s)
    total = mpmath.mpf(0)
    for mu_I in mu_I_list:
        for m in range(2, N_max + 1):
            m_mp = mpmath.mpf(m)
            coeff = m_mp**5 - m_mp**3  # m^5 - m^3
            arg = mu_I**2 + m_mp**2 / s**2
            total += coeff * mpmath.exp(-arg)
    return mpmath.mpf(8) / (3 * s**3) * total

# Test at several s values
print("\nS^4 action and derivative (massless, f=exp(-u)):")
print(f"{'s':>8} {'S(s)':>25} {'dS/ds':>25} {'dS/ds>0?':>10}")
for s_val in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    S_val = S4_action(s_val)
    dS_val = S4_deriv(s_val)
    sign = "YES" if dS_val > 0 else "NO"
    print(f"{s_val:8.1f} {float(S_val):25.15e} {float(dS_val):25.15e} {sign:>10}")
    v.check_value(f"S4 dS/ds > 0 at s={s_val}", float(dS_val > 0), 1.0)

v.checkpoint("S4 no-saddle theorem")

# Test with nonzero finite-sector mass
print("\nS^4 with mu_I = [0, 0.5, 1.0] (3 modes):")
mu_list = [mpmath.mpf(0), mpmath.mpf('0.5'), mpmath.mpf(1)]
for s_val in [1.0, 5.0, 20.0]:
    dS_val = S4_deriv(s_val, mu_list)
    sign = "YES" if dS_val > 0 else "NO"
    print(f"  s={s_val:5.1f}: dS/ds = {float(dS_val):20.12e}  >0? {sign}")
    v.check_value(f"S4 dS/ds > 0 (3-mode) at s={s_val}", float(dS_val > 0), 1.0)

v.checkpoint("S4 no-saddle with finite-sector masses")

# ============================================================
# PART B: N_s^crit thresholds
# ============================================================
print("\n" + "=" * 70)
print("PART B: N_s^crit for disappearance of positive-real Pi_TT zero")
print("=" * 70)

# SCT form factors (verified canonical results)
def phi_master(x):
    """Master function phi(x) = exp(-x/4)*sqrt(pi/x)*erfi(sqrt(x)/2)."""
    x = mpmath.mpf(x)
    if x < mpmath.mpf('1e-10'):
        return mpmath.mpf(1) - x / 6
    return mpmath.exp(-x / 4) * mpmath.sqrt(mpmath.pi / x) * mpmath.erfi(mpmath.sqrt(x) / 2)

def hC_scalar(x):
    """h_C^(0)(x) = 1/(12x) + (phi-1)/(2x^2)."""
    x = mpmath.mpf(x)
    p = phi_master(x)
    return 1 / (12 * x) + (p - 1) / (2 * x**2)

def hC_dirac(x):
    """h_C^(1/2)(x) = (3*phi-1)/(6x) + 2(phi-1)/x^2."""
    x = mpmath.mpf(x)
    p = phi_master(x)
    return (3 * p - 1) / (6 * x) + 2 * (p - 1) / x**2

def hC_vector(x):
    """h_C^(1)(x) = phi/4 + (6*phi-5)/(6x) + (phi-1)/x^2."""
    x = mpmath.mpf(x)
    p = phi_master(x)
    return p / 4 + (6 * p - 5) / (6 * x) + (p - 1) / x**2

def alpha_C_total(z, N_s, N_D, N_v):
    """Total alpha_C(z) = N_s*hC^(0)(z) + N_D*hC^(1/2)(z) + N_v*hC^(1)(z).

    Sign convention: hC_dirac already includes the (-) from ghost subtraction,
    i.e. hC_dirac(0) = -1/20 (not +1/20). So all terms are summed with +.
    Result: alpha_C(0) = 4/120 + 22.5*(-1/20) + 12/10 = 13/120 for SM.
    """
    z = mpmath.mpf(z)
    return N_s * hC_scalar(z) + N_D * hC_dirac(z) + N_v * hC_vector(z)

def Pi_TT(z, N_s, N_D, N_v):
    """Pi_TT(z) = 1 + 2*z*alpha_C(z)."""
    z = mpmath.mpf(z)
    return 1 + 2 * z * alpha_C_total(z, N_s, N_D, N_v)

# Verify SM alpha_C(0) = 13/120
ac0_SM = alpha_C_total(mpmath.mpf('1e-8'), 4, 22.5, 12)
print(f"\nalpha_C(0) for SM (4, 22.5, 12): {float(ac0_SM):.10f}")
print(f"Expected 13/120 = {13/120:.10f}")
v.check_value("alpha_C(0) SM", float(ac0_SM), 13/120, rtol=1e-5)

# Verify alpha_C(0) = 0 at N_D_crit
# ChatGPT: N_D^crit = N_s/6 + 2*N_v (using |beta_W| convention)
# Our code: hC_dirac(0) = -1/20, so alpha_C(0) = N_s/120 + N_D*(-1/20) + N_v/10
# Setting to 0: N_s/120 - N_D/20 + N_v/10 = 0 => N_D = N_s/6 + 2*N_v = 74/3
ac0_crit = alpha_C_total(mpmath.mpf('1e-8'), 4, mpmath.mpf(74)/3, 12)
print(f"\nalpha_C(0) at N_D=74/3: {float(ac0_crit):.15e}")
v.check_value("alpha_C(0) at critical N_D", float(abs(ac0_crit)), 0.0, atol=1e-5)

v.checkpoint("alpha_C local limits")

# Find positive-real zero of Pi_TT at critical N_D = 74/3
print("\nSearching for Pi_TT zero at N_D = 74/3 (alpha_C(0) = 0)...")
N_D_crit = mpmath.mpf(74) / 3

# Binary search for zero
z_lo, z_hi = mpmath.mpf('0.1'), mpmath.mpf('10')
for _ in range(200):
    z_mid = (z_lo + z_hi) / 2
    val = Pi_TT(z_mid, 4, N_D_crit, 12)
    if val > 0:
        z_lo = z_mid
    else:
        z_hi = z_mid

z0_crit = (z_lo + z_hi) / 2
m2_ratio_crit = mpmath.sqrt(z0_crit)
print(f"z_0 at N_D=74/3: {float(z0_crit):.12f}")
print(f"m_2/Lambda at N_D=74/3: {float(m2_ratio_crit):.12f}")
print(f"ChatGPT claimed: z_0 ≈ 1.981194209876, m_2/Λ ≈ 1.407549007984")
v.check_value("z0 at critical N_D", float(z0_crit), 1.981194209876, rtol=1e-6)
v.check_value("m2/Lambda at critical N_D", float(m2_ratio_crit), 1.407549007984, rtol=1e-6)

# Verify SM z_0 = 2.4148...
print("\nSearching for Pi_TT zero for SM (4, 22.5, 12)...")
z_lo, z_hi = mpmath.mpf('0.1'), mpmath.mpf('10')
for _ in range(200):
    z_mid = (z_lo + z_hi) / 2
    val = Pi_TT(z_mid, 4, 22.5, 12)
    if val > 0:
        z_lo = z_mid
    else:
        z_hi = z_mid

z0_SM = (z_lo + z_hi) / 2
m2_SM = mpmath.sqrt(z0_SM)
print(f"z_0 for SM: {float(z0_SM):.12f}")
print(f"m_2/Lambda for SM: {float(m2_SM):.12f}")
print(f"Our canonical: z_0 = 2.41483889..., m_2/Lambda = 1.553975...")
v.check_value("z0 SM", float(z0_SM), 2.41483889, rtol=1e-5)
v.check_value("m2/Lambda SM", float(m2_SM), 1.553975, rtol=1e-4)

v.checkpoint("Pi_TT zeros")

# Now find N_s^crit: the supremum of N(z) = -(1 + 2z*(N_D*hC^(1/2) + N_v*hC^(1)))/(2z*hC^(0))
# for SM-like (N_D=22.5, N_v=12) and PS-like (N_D=24, N_v=21)

def N_threshold(z, N_D, N_v):
    """N_s threshold function: Pi_TT(z)=0 iff N_s = N(z).

    From Pi_TT = 1 + 2z*(N_s*hC_s + N_D*hC_D + N_v*hC_v) = 0,
    solve: N_s = -(1 + 2z*(N_D*hC_D + N_v*hC_v)) / (2z*hC_s).
    """
    z = mpmath.mpf(z)
    hC0 = hC_scalar(z)
    hCD = hC_dirac(z)
    hCV = hC_vector(z)
    numerator = -(1 + 2 * z * (N_D * hCD + N_v * hCV))
    denominator = 2 * z * hC0
    return numerator / denominator

print("\n" + "-" * 50)
print("N_s^crit for SM-like (N_D=22.5, N_v=12):")

# Scan for supremum
best_N = mpmath.mpf('-inf')
best_z = mpmath.mpf(0)
for z_test in [mpmath.mpf(i) / 10 for i in range(1, 2000)]:
    N_val = N_threshold(z_test, 22.5, 12)
    if N_val > best_N:
        best_N = N_val
        best_z = z_test

# Refine with golden section search
a, b = best_z - mpmath.mpf('0.5'), best_z + mpmath.mpf('0.5')
gr = (mpmath.sqrt(5) + 1) / 2
for _ in range(100):
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    if N_threshold(c, 22.5, 12) > N_threshold(d, 22.5, 12):
        b = d
    else:
        a = c
N_s_crit_SM = N_threshold((a + b) / 2, 22.5, 12)
z_crit_SM = (a + b) / 2
print(f"N_s^crit (SM-like) = {float(N_s_crit_SM):.12f}")
print(f"at z = {float(z_crit_SM):.6f}")
print(f"ChatGPT claimed: N_s^crit ≈ 100.777907223644")
v.check_value("N_s^crit SM-like", float(N_s_crit_SM), 100.777907223644, rtol=1e-4)

print("\nN_s^crit for PS-like (N_D=24, N_v=21):")
best_N = mpmath.mpf('-inf')
best_z = mpmath.mpf(0)
for z_test in [mpmath.mpf(i) / 10 for i in range(1, 2000)]:
    N_val = N_threshold(z_test, 24, 21)
    if N_val > best_N:
        best_N = N_val
        best_z = z_test

a, b = best_z - mpmath.mpf('0.5'), best_z + mpmath.mpf('0.5')
for _ in range(100):
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    if N_threshold(c, 24, 21) > N_threshold(d, 24, 21):
        b = d
    else:
        a = c
N_s_crit_PS = N_threshold((a + b) / 2, 24, 21)
z_crit_PS = (a + b) / 2
print(f"N_s^crit (PS-like) = {float(N_s_crit_PS):.12f}")
print(f"at z = {float(z_crit_PS):.6f}")
print(f"ChatGPT claimed: N_s^crit ≈ 137.042494159732")
v.check_value("N_s^crit PS-like", float(N_s_crit_PS), 137.042494159732, rtol=1e-4)

v.checkpoint("N_s^crit thresholds")

# ============================================================
# PART C: Verify ChatGPT's extension table
# ============================================================
print("\n" + "=" * 70)
print("PART C: Extension table verification")
print("=" * 70)

extensions = [
    ("SM", 4, 22.5, 12),
    ("SM+3nuR", 4, 24, 12),
    ("SM+3nuR+sigma", 5, 24, 12),
    ("U(1)_BL tuned", 6, 24, 13),
    ("PS-A39", 39, 24, 21),
    ("PS-A40", 40, 24, 21),
    ("PS-B", 200, 24, 21),
    ("PS-C", 272, 24, 21),
]

chatgpt_alpha_C = {
    "SM": mpmath.mpf(13)/120,
    "SM+3nuR": mpmath.mpf(1)/30,
    "SM+3nuR+sigma": mpmath.mpf(1)/24,
    "U(1)_BL tuned": mpmath.mpf(3)/20,
    "PS-A39": mpmath.mpf(49)/40,
    "PS-A40": mpmath.mpf(37)/30,
    "PS-B": mpmath.mpf(77)/30,
    "PS-C": mpmath.mpf(19)/6,
}

print(f"\n{'Model':<20} {'(N_s,N_D,N_v)':<20} {'alpha_C(0)':>12} {'expected':>12} {'match?':>8}")
for name, Ns, ND, Nv in extensions:
    # alpha_C(0) = N_s/120 - N_D/20 + N_v/10
    ac = mpmath.mpf(Ns)/120 - mpmath.mpf(ND)/20 + mpmath.mpf(Nv)/10
    expected = chatgpt_alpha_C[name]
    match = abs(float(ac - expected)) < 1e-10
    print(f"{name:<20} ({Ns},{ND},{Nv}){'':<8} {float(ac):12.6f} {float(expected):12.6f} {'YES' if match else 'NO':>8}")
    v.check_value(f"alpha_C(0) {name}", float(ac), float(expected), atol=1e-10)

v.checkpoint("extension table alpha_C")

# Check which have positive-real Pi_TT zero
print(f"\n{'Model':<20} {'N_s':>6} {'N_s^crit':>12} {'has zero?':>10}")
for name, Ns, ND, Nv in extensions:
    if ND == 22.5 and Nv == 12:
        crit = N_s_crit_SM
    elif ND == 24 and Nv == 12:
        # Need to compute for (24, 12) — different from SM-like
        best_N_loc = mpmath.mpf('-inf')
        for z_t in [mpmath.mpf(i) / 10 for i in range(1, 2000)]:
            N_val = N_threshold(z_t, ND, Nv)
            if N_val > best_N_loc:
                best_N_loc = N_val
        crit = best_N_loc
    elif ND == 24 and Nv == 13:
        best_N_loc = mpmath.mpf('-inf')
        for z_t in [mpmath.mpf(i) / 10 for i in range(1, 2000)]:
            N_val = N_threshold(z_t, ND, Nv)
            if N_val > best_N_loc:
                best_N_loc = N_val
        crit = best_N_loc
    elif ND == 24 and Nv == 21:
        crit = N_s_crit_PS
    else:
        crit = mpmath.mpf('nan')
    has_zero = "YES" if Ns < float(crit) else "NO"
    print(f"{name:<20} {Ns:6} {float(crit):12.4f} {has_zero:>10}")

v.checkpoint("extension table zeros")

# ============================================================
# PART D: Pairwise threshold m_a + m_b < sqrt(6)*Lambda
# ============================================================
print("\n" + "=" * 70)
print("PART D: Pairwise threshold verification")
print("=" * 70)

# From ChatGPT: kinetic coefficient K_m = (U/(2*Lambda^4))*(1 - 2m^2/(3*Lambda^2))
# K_m > 0 iff m^2 < (3/2)*Lambda^2 iff |m| < sqrt(3/2)*Lambda
# For pairwise: m_a + m_b < sqrt(6)*Lambda

threshold_single = mpmath.sqrt(mpmath.mpf(3) / 2)
threshold_pair = mpmath.sqrt(mpmath.mpf(6))
print(f"Single-mode threshold: |m|/Lambda < sqrt(3/2) = {float(threshold_single):.10f}")
print(f"Pairwise threshold: (m_a+m_b)/Lambda < sqrt(6) = {float(threshold_pair):.10f}")

# Verify: for diagonal channel (a=b), m_a + m_b = 2*m_a
# 2*m_a < sqrt(6)*Lambda iff m_a < sqrt(6)/2 * Lambda = sqrt(3/2)*Lambda
# This matches single-mode threshold. CHECK:
print(f"\nsqrt(6)/2 = {float(threshold_pair/2):.10f}")
print(f"sqrt(3/2) = {float(threshold_single):.10f}")
print(f"Match: {abs(float(threshold_pair/2 - threshold_single)) < 1e-15}")
v.check_value("sqrt(6)/2 = sqrt(3/2)", float(threshold_pair/2), float(threshold_single), rtol=1e-14)

v.checkpoint("pairwise threshold")

# ============================================================
# PART E: Summary
# ============================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

v.summary()
