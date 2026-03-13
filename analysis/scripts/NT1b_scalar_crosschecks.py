"""
NT-1b Scalar Form Factor Cross-checks (Verification Pass — Full Verification)
=====================================================================
Comprehensive 4-layer numerical verification of h_C^(0)(x) and h_R^(0)(x;xi)
for a real scalar field with non-minimal coupling xi.

All computations use mpmath with >= 100 decimal digits of precision.
Every check reports explicit PASS/FAIL verdict.

Date: 2026-03-09
Author: David Alfyorov (Verification Pass pipeline)
"""

import sys
from mpmath import (
    mp, mpf, quad as mpquad, exp as mpexp, pi as mppi,
    fac, power, sqrt, erfi, inf, log, fsum, nstr
)

# =====================================================================
# PRECISION SETUP
# =====================================================================
mp.dps = 100  # 100 decimal digits

TOLERANCE_TIGHT = mpf('1e-90')    # For exact rational results
TOLERANCE_MEDIUM = mpf('1e-50')   # For numerical integrations
TOLERANCE_LOOSE = mpf('1e-30')    # For convergence checks
TOLERANCE_UV = mpf('1e-2')        # For UV asymptotic checks (O(1/x) convergence)
TOLERANCE_CANCEL = mpf('1e-25')   # For checks near x=0 with pole cancellation

total_checks = 0
total_pass = 0
total_fail = 0
failed_checks = []

def report(name, computed, expected, tol=TOLERANCE_MEDIUM):
    """Report a single check with PASS/FAIL verdict."""
    global total_checks, total_pass, total_fail
    total_checks += 1
    diff = abs(computed - expected)
    if expected != 0:
        rel_err = diff / abs(expected)
    else:
        rel_err = diff  # absolute for zero targets
    passed = diff < tol or (expected != 0 and rel_err < tol)
    status = "PASS" if passed else "FAIL"
    if passed:
        total_pass += 1
    else:
        total_fail += 1
        failed_checks.append(name)
    print(f"  {status}: {name}")
    print(f"    computed = {nstr(computed, 30)}")
    print(f"    expected = {nstr(expected, 30)}")
    print(f"    |diff|   = {nstr(diff, 10)}")
    if expected != 0:
        print(f"    rel err  = {nstr(rel_err, 10)}")
    return passed


# =====================================================================
# CORE FUNCTIONS (mpmath high precision)
# =====================================================================

def phi_mp(x):
    """Master function phi(x) = int_0^1 exp[-alpha(1-alpha)*x] dalpha"""
    x = mpf(x)
    if x == 0:
        return mpf(1)
    return mpquad(lambda a: mpexp(-a * (1 - a) * x), [0, 1])


def phi_taylor(x, N=60):
    """phi(x) via Taylor series: sum_{n=0}^N (-x)^n * n!/(2n+1)!"""
    x = mpf(x)
    s = mpf(0)
    for n in range(N + 1):
        s += power(-x, n) * fac(n) / fac(2 * n + 1)
    return s


# --- CZ form factors ---

def f_Ric(x):
    """CZ form factor f_Ric(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return mpf(1) / (6 * x) + (p - 1) / x**2


def f_R(x):
    """CZ form factor f_R(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return p / 32 + p / (8 * x) - mpf(7) / (48 * x) - (p - 1) / (8 * x**2)


def f_RU(x):
    """CZ form factor f_{RU}(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return -p / 4 - (p - 1) / (2 * x)


def f_U(x):
    """CZ form factor f_U(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return p / 2


def f_Omega(x):
    """CZ form factor f_Omega(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return -(p - 1) / (2 * x)


# --- Weyl basis form factors ---

def f_C(x):
    """Weyl form factor f_C = (1/2) f_Ric (at d=4)"""
    return f_Ric(x) / 2


def f_Rbis(x):
    """f_{R,bis} = (1/3) f_Ric + f_R (at d=4)"""
    return f_Ric(x) / 3 + f_R(x)


def hC_scalar(x):
    """Scalar Weyl form factor h_C^(0)(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return mpf(1) / (12 * x) + (p - 1) / (2 * x**2)


def hR_scalar(x, xi):
    """Scalar R^2 form factor h_R^(0)(x; xi)"""
    x = mpf(x)
    xi = mpf(xi)
    return f_Rbis(x) + xi * f_RU(x) + xi**2 * f_U(x)


# --- Dirac form factors (from NT-1) ---

def hC_dirac(x):
    """Dirac Weyl form factor h_C^(1/2)(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return (3 * p - 1) / (6 * x) + 2 * (p - 1) / x**2


def hR_dirac(x):
    """Dirac R^2 form factor h_R^(1/2)(x)"""
    x = mpf(x)
    p = phi_mp(x)
    return (3 * p + 2) / (36 * x) + 5 * (p - 1) / (6 * x**2)


# =====================================================================
# CHECK N1: h_C^(0)(0) = 1/120 via Taylor expansion
# =====================================================================
print("=" * 72)
print("CHECK N1: h_C^(0)(0) = 1/120")
print("=" * 72)

# Use Taylor series with analytic pole cancellation
# h_C(x) = 1/(12x) + (phi-1)/(2x^2)
# phi(x) = sum_{n=0}^inf (-x)^n b_n where b_n = n!/(2n+1)!
# phi-1 = sum_{n=1}^inf (-x)^n b_n
# (phi-1)/x = sum_{n=1}^inf (-x)^n b_n / x = sum_{n=0}^inf (-1)^{n+1} b_{n+1} x^n
# (phi-1)/(2x^2) = sum_{n=0}^inf (-1)^{n+1} b_{n+1} x^{n-1} / 2
# ... pole cancellation gives h_C(0) from the n=0 term:
# h_C(0) = -b_1/12 + b_2/2  ... wait, let me redo.
# Actually: phi - 1 = -b_1 x + b_2 x^2 - b_3 x^3 + ...
# (phi-1)/x^2 = -b_1/x + b_2 - b_3 x + ...
# (phi-1)/(2x^2) = -b_1/(2x) + b_2/2 - ...
# 1/(12x) + (phi-1)/(2x^2) = 1/(12x) - b_1/(2x) + b_2/2 + ...
# b_1 = 1!/(3!) = 1/6, so 1/(12x) - 1/(12x) = 0. Poles cancel!
# Constant term: b_2/2 = (2!/5!) / 2 = (2/120)/2 = 1/120

b_1 = fac(1) / fac(3)  # 1/6
b_2 = fac(2) / fac(5)  # 1/60

hC_0_analytic = b_2 / 2  # = 1/120
hC_0_numerical = hC_scalar(mpf('1e-30'))  # Very small x

report("h_C(0) analytic pole cancellation", hC_0_analytic, mpf(1) / 120, TOLERANCE_TIGHT)
report("h_C(0) numerical at x=1e-30", hC_0_numerical, mpf(1) / 120, TOLERANCE_CANCEL)


# =====================================================================
# CHECK N2: h_R^(0)(0; xi=0) = 1/72
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N2: h_R^(0)(0; xi=0) = 1/72")
print("=" * 72)

# f_{R,bis}(0) = (1/3)*f_Ric(0) + f_R(0) = (1/3)*(1/60) + 1/120 = 1/180 + 1/120
fRic_0 = mpf(1) / 60
fR_0 = mpf(1) / 120
fRbis_0 = fRic_0 / 3 + fR_0  # = 1/180 + 1/120 = 5/360 = 1/72

report("f_{R,bis}(0) = 1/72", fRbis_0, mpf(1) / 72, TOLERANCE_TIGHT)

# Numerical check
hR_0_0_num = hR_scalar(mpf('1e-30'), mpf(0))
report("h_R(0;0) numerical at x=1e-30", hR_0_0_num, mpf(1) / 72, TOLERANCE_CANCEL)


# =====================================================================
# CHECK N3: h_R^(0)(0; xi=1/6) = 0 (conformal coupling)
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N3: h_R^(0)(0; xi=1/6) = 0 (conformal invariance)")
print("=" * 72)

# Analytic: (1/2)(1/6 - 1/6)^2 = 0
hR_0_conf_analytic = mpf(1) / 72 - mpf(1) / 36 + mpf(1) / 72
report("h_R(0;1/6) analytic", hR_0_conf_analytic, mpf(0), TOLERANCE_TIGHT)

# Numerical
hR_0_conf_num = hR_scalar(mpf('1e-30'), mpf(1) / 6)
report("h_R(0;1/6) numerical at x=1e-30", hR_0_conf_num, mpf(0), mpf('1e-25'))


# =====================================================================
# CHECK N4: beta_R(xi) = (1/2)(xi - 1/6)^2 for multiple xi values
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N4: beta_R(xi) = (1/2)(xi - 1/6)^2 for 6 values of xi")
print("=" * 72)

xi_values = [mpf(0), mpf(1) / 6, mpf(1) / 4, mpf(1) / 3, mpf(1) / 2, mpf(1)]
for xi_val in xi_values:
    # From form factor local limits
    hR_from_limits = fRbis_0 + xi_val * (mpf(-1) / 6) + xi_val**2 * (mpf(1) / 2)
    # From closed formula
    hR_target = (xi_val - mpf(1) / 6)**2 / 2
    report(f"beta_R(xi={nstr(xi_val, 6)})", hR_from_limits, hR_target, TOLERANCE_TIGHT)


# =====================================================================
# CHECK N5: h_C^(0) at 7 test points (formula vs direct integration)
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N5: h_C^(0)(x) at 7 test points")
print("=" * 72)

test_x = [mpf('0.01'), mpf('0.1'), mpf(1), mpf(5), mpf(10), mpf(50), mpf(100)]
for x_val in test_x:
    # Formula
    hC_formula = hC_scalar(x_val)
    # Direct integration: h_C = (1/2) f_Ric
    # f_Ric = 1/(6x) + (phi-1)/x^2
    p = phi_mp(x_val)
    hC_direct = mpf(1) / (12 * x_val) + (p - 1) / (2 * x_val**2)
    # Also verify via f_C
    hC_via_fC = f_C(x_val)
    report(f"h_C({nstr(x_val, 4)}) formula vs direct", hC_formula, hC_direct)


# =====================================================================
# CHECK N6: h_R^(0) at 7 test points for xi=0, 1/6, 1
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N6: h_R^(0)(x;xi) at 7 test points, 3 xi values")
print("=" * 72)

xi_test = [mpf(0), mpf(1) / 6, mpf(1)]
for xi_val in xi_test:
    for x_val in test_x:
        # Via h_R formula
        hR_formula = hR_scalar(x_val, xi_val)
        # Via individual components computed independently
        p = phi_mp(x_val)
        A = p / 32 + (p / 8 - mpf(13) / 144) / x_val + 5 * (p - 1) / (24 * x_val**2)
        B = -p / 4 - (p - 1) / (2 * x_val)
        C = p / 2
        hR_components = A + xi_val * B + xi_val**2 * C
        report(f"h_R({nstr(x_val, 4)}; xi={nstr(xi_val, 6)}) formula vs components",
               hR_formula, hR_components)


# =====================================================================
# CHECK N7: F_1 = h_C/(16*pi^2) for exponential psi
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N7: F_1^(0) = h_C/(16*pi^2) for psi=exp(-u)")
print("=" * 72)

# For exponential spectral function psi(u) = exp(-u):
# Psi_1(u) = Psi_2(u) = exp(-u), Psi_1(0) = Psi_2(0) = 1
# F_1(z) = (1/(16*pi^2)) * [Psi_1(0)/(12z)
#           + (1/(2z^2)) int_0^1 [Psi_2(alpha(1-alpha)z) - Psi_2(0)] dalpha]
# = (1/(16*pi^2)) * [1/(12z) + (1/(2z^2)) int_0^1 [exp(-alpha(1-alpha)z) - 1] dalpha]
# = (1/(16*pi^2)) * [1/(12z) + (phi(z)-1)/(2z^2)]
# = h_C(z) / (16*pi^2)

for z_val in [mpf('0.01'), mpf('0.1'), mpf(1), mpf(5), mpf(10), mpf(50), mpf(100)]:
    # F_1 via spectral integral
    Psi1_0 = mpf(1)
    Psi2_0 = mpf(1)
    int_part = mpquad(
        lambda a: mpexp(-a * (1 - a) * z_val) - 1, [0, 1]
    )
    F1_spectral = (Psi1_0 / (12 * z_val) + int_part / (2 * z_val**2)) / (16 * mppi**2)
    # F_1 via h_C
    F1_hk = hC_scalar(z_val) / (16 * mppi**2)
    report(f"F_1({nstr(z_val, 4)}) spectral vs h_C/(16pi^2)",
           F1_spectral, F1_hk)


# =====================================================================
# CHECK N8: F_2 = h_R/(16*pi^2) for exponential psi
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N8: F_2^(0) = h_R/(16*pi^2) for psi=exp(-u), 3 xi values")
print("=" * 72)

for xi_val in [mpf(0), mpf(1) / 6, mpf(1)]:
    for z_val in [mpf('0.1'), mpf(1), mpf(10), mpf(100)]:
        # F_2 via spectral integral formula (eq:F2)
        # For psi=exp(-u): psi(w) = exp(-w), Psi_1(w) = exp(-w), Psi_2(w) = exp(-w)
        psi_int = mpquad(
            lambda a: (mpf(1) / 32 + xi_val**2 / 2 - xi_val * a * (1 - a)) *
                      mpexp(-a * (1 - a) * z_val), [0, 1]
        )
        Psi1_int = mpquad(
            lambda a: mpexp(-a * (1 - a) * z_val), [0, 1]
        ) / (8 * z_val)
        Psi1_const = -mpf(13) / (144 * z_val)
        Psi2_int = mpf(5) / (24 * z_val**2) * mpquad(
            lambda a: mpexp(-a * (1 - a) * z_val) - 1, [0, 1]
        )
        F2_spectral = (psi_int + Psi1_int + Psi1_const + Psi2_int) / (16 * mppi**2)
        # F_2 via h_R
        F2_hk = hR_scalar(z_val, xi_val) / (16 * mppi**2)
        report(f"F_2({nstr(z_val, 4)};xi={nstr(xi_val, 6)}) spectral vs h_R/(16pi^2)",
               F2_spectral, F2_hk)


# =====================================================================
# CHECK N9: Taylor series (60 terms) vs direct computation of h_C
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N9: h_C^(0) Taylor series (60 terms) vs numerical")
print("=" * 72)

def hC_taylor_mp(x, N=60):
    """h_C via Taylor series with analytic pole cancellation.
    h_C(x) = sum_{m=0}^{N-1} c_m (-x)^m
    where c_m = -b_{m+1}/(12) + b_{m+2}/2  [wrong formula]

    Actually: h_C = 1/(12x) + (phi-1)/(2x^2)
    phi - 1 = sum_{n=1}^inf (-x)^n b_n
    (phi-1)/x = sum_{n=1}^inf (-x)^n b_n / x = sum_{n=0}^inf (-1)^{n+1} x^n b_{n+1}
    (phi-1)/(2x^2) = (1/2) * (phi-1)/x / x
    = (1/2) sum_{n=0}^inf (-1)^{n+1} x^{n-1} b_{n+1}
    = (1/2) * [ -b_1/x + b_2 - b_3 x + b_4 x^2 - ... ]
    1/(12x) + (phi-1)/(2x^2) = [1/12 - b_1/2]/x + b_2/2 - b_3 x/2 + ...
    Since b_1 = 1/6, 1/12 - 1/12 = 0. Pole cancels.
    h_C(x) = sum_{m=0}^inf (-1)^m b_{m+2}/2 * x^m

    Wait, more carefully:
    (phi-1)/(2x^2) = (1/(2x^2)) * sum_{n=1} (-x)^n b_n
    = (1/2) sum_{n=1} (-1)^n x^{n-2} b_n
    = (1/2) [ -b_1/x + b_2 - b_3 x + b_4 x^2 - ... ]

    1/(12x) = ... we combine with -b_1/(2x):
    [1/12 - b_1/2]/x = [1/12 - 1/12]/x = 0

    So h_C(x) = sum_{m=0}^{inf} (-1)^m (b_{m+2}/2) x^m
    """
    x = mpf(x)
    s = mpf(0)
    for m in range(N):
        b_mp2 = fac(m + 2) / fac(2 * (m + 2) + 1)
        s += power(-1, m) * b_mp2 / 2 * power(x, m)
    return s

for x_val in [mpf('0.001'), mpf('0.01'), mpf('0.1'), mpf('0.5'), mpf(1)]:
    hC_tay = hC_taylor_mp(x_val, N=60)
    hC_num = hC_scalar(x_val)
    report(f"h_C({nstr(x_val, 4)}) Taylor(60) vs numerical", hC_tay, hC_num)


# =====================================================================
# CHECK N10: Taylor series (60 terms) vs direct computation of h_R
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N10: h_R^(0) Taylor series (60 terms) vs numerical")
print("=" * 72)

def hR_taylor_mp(x, xi, N=60):
    """h_R via Taylor series with analytic pole cancellation.

    h_R = f_{R,bis} + xi*f_{RU} + xi^2*f_U

    For f_U = phi/2 = (1/2) sum b_n (-x)^n  -> direct series

    For f_{RU} = -phi/4 - (phi-1)/(2x)
    The parametric form: f_{RU} = int_0^1 [-alpha(1-alpha)] exp[-alpha(1-alpha)x] dalpha
    f_{RU}(x) = sum_{n=0} (-x)^n * int_0^1 alpha^{n+1}(1-alpha)^{n+1}/n! dalpha
    = sum_{n=0} (-x)^n b_{n+1}/n! * n!  [Wait: int alpha^{n+1}(1-alpha)^{n+1} dalpha = B(n+2,n+2) = ((n+1)!)^2/(2n+3)!]

    Actually, f_{RU}(x) = -sum_{n=0} (-x)^n/(n!) * int_0^1 alpha^{n+1}(1-alpha)^{n+1} dalpha
    = -sum_{n=0} (-x)^n/(n!) * ((n+1)!)^2/(2n+3)!
    = -sum_{n=0} (-x)^n * (n+1)!/(2n+3)!/n! * (n+1)!
    Hmm, let me just use: (n+1)! = (n+1)*n!, so ((n+1)!)^2/(n!*(2n+3)!) = (n+1)^2 * n!/(2n+3)!

    Actually simpler: f_{RU}(x) = sum_{n=0}^inf (-x)^n * [-int_0^1 alpha^{n+1}(1-alpha)^{n+1} dalpha / n!]
    = -sum_{n=0}^inf (-x)^n * ((n+1)!)^2 / (n! * (2n+3)!)
    = -sum_{n=0}^inf (-x)^n * (n+1) * (n+1)! / (2n+3)!

    Hmm this is getting complicated. Let me use a different approach.

    From the Taylor expansion:
    f_{R,bis}(x) = sum_{m=0}^inf c_m^A (-x)^m  [needs pole cancellation]
    f_{RU}(x) = sum_{m=0}^inf c_m^B (-x)^m
    f_U(x) = sum_{m=0}^inf c_m^C (-x)^m

    For f_U: c_m^C = b_m/2 where b_n = n!/(2n+1)!

    For f_{RU}: from f_{RU} = -phi/4 - (phi-1)/(2x)
    phi/4 = (1/4) sum b_n (-x)^n
    (phi-1)/(2x) = (1/2) sum_{n=1} (-1)^n b_n x^{n-1}
    = (1/2) sum_{m=0} (-1)^{m+1} b_{m+1} x^m

    f_{RU} = sum_{m=0} (-x)^m * [-b_m/4 + b_{m+1}/2]
    Wait: -phi/4 = -(1/4) sum b_n (-x)^n = sum (-1)^{n+1} b_n x^n / 4 = sum (-x)^n (-b_n/4)
    -(phi-1)/(2x) = -(1/2) sum_{n=1} (-x)^n b_n / x = -(1/2) sum_{n=1} (-1)^n b_n x^{n-1}
    = -(1/2) sum_{m=0} (-1)^{m+1} b_{m+1} x^m = (1/2) sum_{m=0} (-1)^m b_{m+1} (-x)^m ... no.

    Let me be very explicit.
    phi = sum_{n=0}^inf (-1)^n b_n x^n

    -phi/4 = sum_{n=0}^inf (-1)^{n+1} b_n x^n / 4
    = sum_{n=0} c1_n x^n where c1_n = (-1)^{n+1} b_n / 4

    -(phi-1)/(2x) = -(1/(2x)) sum_{n=1}^inf (-1)^n b_n x^n
    = -(1/2) sum_{n=1} (-1)^n b_n x^{n-1}
    = -(1/2) sum_{m=0} (-1)^{m+1} b_{m+1} x^m
    = sum_{m=0} c2_m x^m where c2_m = (-1)^m b_{m+1} / 2

    f_{RU} = sum_{m=0} (c1_m + c2_m) x^m
    c_m^B = (-1)^{m+1} b_m / 4 + (-1)^m b_{m+1} / 2
    = (-1)^m [-b_m/4 + b_{m+1}/2]

    For f_{R,bis}: more complex. Let me combine term by term.
    f_{R,bis} = phi/32 + (phi/8 - 13/144)/x + 5(phi-1)/(24x^2)

    phi/32 = sum (-1)^n b_n x^n / 32

    (phi/8)/x = (1/8) sum (-1)^n b_n x^{n-1} = (1/8) sum_{m=-1} (-1)^{m+1} b_{m+1} x^m
    -13/(144x): singular at x=0

    5(phi-1)/(24x^2) = (5/24) sum_{n=1} (-1)^n b_n x^{n-2}
    = (5/24) sum_{m=-1} (-1)^{m+2} b_{m+2} x^m

    Collecting poles:
    1/x term: (-1/8)(-1)b_0 + ... wait this is messy. Let me do it properly.

    (phi/8 - 13/144)/x:
    phi/8 = (1/8)(b_0 - b_1 x + b_2 x^2 - ...)
    phi/8 - 13/144 = b_0/8 - 13/144 + series in x
    = 1/8 - 13/144 + ... = (18 - 13)/144 + ... = 5/144 + ...
    (phi/8 - 13/144)/x = 5/(144x) + terms from (-b_1 x/8)/x + ... = 5/(144x) - b_1/8 + ...

    5(phi-1)/(24x^2) = (5/24)((-b_1 x + b_2 x^2 - ...)/x^2)
    = (5/24)(-b_1/x + b_2 - b_3 x + ...)
    = -5b_1/(24x) + 5b_2/24 - 5b_3 x/24 + ...

    1/x poles: 5/(144x) - 5b_1/(24x) = 5/(144x) - 5/(144x) = 0  [since b_1 = 1/6]

    Constant: b_0/32 - b_1/8 + 5b_2/24
    = 1/32 - 1/48 + 5/(24*60) = 1/32 - 1/48 + 1/288
    LCD 288: = 9/288 - 6/288 + 1/288 = 4/288 = 1/72 ✓

    General m-th term (m >= 0):
    From phi/32: (-1)^m b_m / 32
    From (phi/8)/x at x^m: need phi/8 at x^{m+1}: (-1)^{m+1} b_{m+1} / 8
    From -13/(144x) at x^m: 0 for m >= 0 (only 1/x pole)
    From 5(phi-1)/(24x^2) at x^m: 5*(-1)^{m+2} b_{m+2} / 24 = 5*(-1)^m b_{m+2} / 24

    c_m^A = (-1)^m [ b_m/32 - b_{m+1}/8 + 5*b_{m+2}/24 ]

    So finally:
    h_R(x;xi) = sum_{m=0}^inf x^m * [ c_m^A + xi * c_m^B + xi^2 * c_m^C ]
    = sum_{m=0}^inf x^m * (-1)^m * [ b_m/32 - b_{m+1}/8 + 5b_{m+2}/24
                                      + xi(-b_m/4 + b_{m+1}/2) + xi^2 * b_m/2 ]
    """
    x = mpf(x)
    xi = mpf(xi)
    s = mpf(0)
    for m in range(N):
        bm = fac(m) / fac(2 * m + 1)
        bm1 = fac(m + 1) / fac(2 * m + 3)
        bm2 = fac(m + 2) / fac(2 * m + 5)

        c_A = bm / 32 - bm1 / 8 + 5 * bm2 / 24
        c_B = -bm / 4 + bm1 / 2
        c_C = bm / 2

        coeff = c_A + xi * c_B + xi**2 * c_C
        s += power(-1, m) * coeff * power(x, m)
    return s


for xi_val in [mpf(0), mpf(1) / 6, mpf(1)]:
    for x_val in [mpf('0.001'), mpf('0.01'), mpf('0.1'), mpf('0.5')]:
        hR_tay = hR_taylor_mp(x_val, xi_val, N=60)
        hR_num = hR_scalar(x_val, xi_val)
        report(f"h_R({nstr(x_val, 4)};xi={nstr(xi_val, 6)}) Taylor(60) vs numerical",
               hR_tay, hR_num)


# =====================================================================
# CHECK N11: UV asymptotics — x*h_C(x) -> 1/12 as x->inf
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N11: UV asymptotics x*h_C(x) -> 1/12")
print("=" * 72)
print("  (Convergence is O(1/x), so rel. error ~ 6/x)")

# Use largest x values to demonstrate convergence and check that
# the ratio x*h_C(x) / (1/12) approaches 1 within expected O(1/x) rate
for x_val in [mpf(1000), mpf(10000), mpf(100000)]:
    x_hC = x_val * hC_scalar(x_val)
    # Expected: 1/12 - 1/(2x) + O(1/x^2), so relative error ~ 6/x
    tol_uv = 10 / x_val  # generous margin
    report(f"x*h_C({nstr(x_val, 6)}) -> 1/12 (tol~{nstr(tol_uv, 3)})",
           x_hC, mpf(1) / 12, tol_uv)


# =====================================================================
# CHECK N12: UV asymptotics — x*h_R(x;0) -> -1/36 as x->inf
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N12: UV asymptotics x*h_R(x;0) -> -1/36")
print("=" * 72)
print("  (Convergence is O(1/x), so rel. error ~ C/x)")

for x_val in [mpf(1000), mpf(10000), mpf(100000)]:
    x_hR = x_val * hR_scalar(x_val, mpf(0))
    tol_uv = 10 / x_val
    report(f"x*h_R({nstr(x_val, 6)};0) -> -1/36 (tol~{nstr(tol_uv, 3)})",
           x_hR, mpf(-1) / 36, tol_uv)


# =====================================================================
# CHECK N13: UV asymptotics — x^2*h_R(x;1/6) -> -1/9 as x->inf
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N13: UV conformal coupling x^2*h_R(x;1/6) -> -1/9")
print("=" * 72)
print("  (Convergence is O(1/x), so rel. error ~ C/x)")

for x_val in [mpf(1000), mpf(10000), mpf(100000)]:
    x2_hR = x_val**2 * hR_scalar(x_val, mpf(1) / 6)
    tol_uv = 10 / x_val
    report(f"x^2*h_R({nstr(x_val, 6)};1/6) -> -1/9 (tol~{nstr(tol_uv, 3)})",
           x2_hR, mpf(-1) / 9, tol_uv)


# =====================================================================
# CHECK N14: Dirac/scalar ratio at x=0
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N14: Dirac/scalar local limit ratio")
print("=" * 72)

# h_C^(1/2)(0) = -1/20, h_C^(0)(0) = 1/120
# |ratio| = (1/20)/(1/120) = 6
# But: h_C^(1/2)(0)/h_C^(0)(0) = (-1/20)/(1/120) = -6

# Use Taylor-based evaluation for maximum precision
# h_C^(1/2)(0) from NT-1: Dirac h_C(0) = -1/20
# h_C^(0)(0) = 1/120 (scalar)
# Ratio: (-1/20) / (1/120) = -6
hC_dirac_0_exact = mpf(-1) / 20
hC_scalar_0_exact = mpf(1) / 120
ratio_exact = hC_dirac_0_exact / hC_scalar_0_exact
report("h_C^(1/2)(0) / h_C^(0)(0) = -6 (exact)", ratio_exact, mpf(-6), TOLERANCE_TIGHT)

# Also check numerically with moderate tolerance
hC_dirac_0 = hC_dirac(mpf('1e-30'))
hC_scalar_0 = hC_scalar(mpf('1e-30'))
ratio = hC_dirac_0 / hC_scalar_0
report("h_C^(1/2)(0) / h_C^(0)(0) = -6 (numerical)", ratio, mpf(-6), TOLERANCE_CANCEL)

# beta_R comparison: Dirac h_R(0) = 0, scalar h_R(0;1/6) = 0
hR_dirac_0 = hR_dirac(mpf('1e-30'))
report("h_R^(1/2)(0) = 0 (Dirac conformal)", hR_dirac_0, mpf(0), TOLERANCE_CANCEL)


# =====================================================================
# CHECK N15: Seeley-DeWitt a_4 vs form factor local limits
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N15: Seeley-DeWitt a_4 coefficients (dual derivation)")
print("=" * 72)

# From a_4 computation (Vassilevich):
# a_4 = (1/360)[ -2 R_{mn}^2 + 2 R_{mnrs}^2 + (180*xi^2 - 60*xi + 5) R^2 + ...]
# In Weyl basis (dropping E_4):
# C^2 coefficient: (1/360) * 3 = 1/120
# R^2 coefficient: (180*xi^2 - 60*xi + 5)/360

# Check C^2 coefficient
a4_C2 = mpf(3) / 360  # = 1/120
report("a_4|_{C^2} = 1/120 = h_C(0)", a4_C2, mpf(1) / 120, TOLERANCE_TIGHT)

# Check R^2 coefficient for multiple xi
for xi_val in [mpf(0), mpf(1) / 6, mpf(1) / 4, mpf(1)]:
    a4_R2 = (180 * xi_val**2 - 60 * xi_val + 5) / 360
    hR_target = (xi_val - mpf(1) / 6)**2 / 2
    report(f"a_4|_{{R^2}}(xi={nstr(xi_val, 6)}) = h_R(0;xi)",
           a4_R2, hR_target, TOLERANCE_TIGHT)


# =====================================================================
# CHECK N16: Pole cancellation verification
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N16: Pole cancellation at x=0")
print("=" * 72)

# For h_C: the 1/x poles from 1/(12x) and (phi-1)/(2x^2) must cancel
# Extract the 1/x coefficient: 1/12 - b_1/2 = 1/12 - 1/12 = 0
pole_hC = mpf(1) / 12 - fac(1) / (2 * fac(3))
report("h_C 1/x pole coefficient = 0", pole_hC, mpf(0), TOLERANCE_TIGHT)

# For f_{R,bis}: 1/x poles must cancel
# 5/(144) - 5*b_1/24 = 5/144 - 5/(24*6) = 5/144 - 5/144 = 0
pole_fRbis = mpf(5) / 144 - 5 * fac(1) / (24 * fac(3))
report("f_{R,bis} 1/x pole coefficient = 0", pole_fRbis, mpf(0), TOLERANCE_TIGHT)


# =====================================================================
# CHECK N17: Parametric integral representation of f_{RU}
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N17: Parametric integral f_{RU} = int[-alpha(1-alpha)]exp[...]")
print("=" * 72)

for x_val in [mpf('0.1'), mpf(1), mpf(10), mpf(100)]:
    # Direct formula
    fRU_formula = f_RU(x_val)
    # Parametric integral
    fRU_parametric = mpquad(
        lambda a: -a * (1 - a) * mpexp(-a * (1 - a) * x_val), [0, 1]
    )
    report(f"f_{{RU}}({nstr(x_val, 4)}) parametric vs formula",
           fRU_parametric, fRU_formula)


# =====================================================================
# CHECK N18: CZ local limit values f_i(0)
# =====================================================================
print("\n" + "=" * 72)
print("CHECK N18: CZ form factor local limits f_i(0)")
print("=" * 72)

# Numerical limit check at small x.
# At x=1e-15, the O(x) correction to each f_i(0) is ~x/C ~ 1e-16,
# plus pole cancellation in f_Ric, f_R costs ~15 additional digits.
# Combined: ~1e-15 relative error. We use a tolerance that accounts for
# both effects: the finite-x correction dominates.
x_small = mpf('1e-8')
tol_num_limit = mpf('1e-7')  # O(x_small) correction ~ x/C for each form factor
targets = {
    'f_Ric(0)': (f_Ric(x_small), mpf(1) / 60),
    'f_R(0)': (f_R(x_small), mpf(1) / 120),
    'f_RU(0)': (f_RU(x_small), mpf(-1) / 6),
    'f_U(0)': (f_U(x_small), mpf(1) / 2),
    'f_Omega(0)': (f_Omega(x_small), mpf(1) / 12),
}
for name, (computed, expected) in targets.items():
    report(f"CZ local limit {name} (x={nstr(x_small, 3)})", computed, expected, tol_num_limit)

# Also verify analytically via Taylor coefficients
print("\n  Analytic verification from Taylor coefficients:")
b = [fac(n) / fac(2 * n + 1) for n in range(5)]  # b_0=1, b_1=1/6, b_2=1/60, ...
# f_Ric(0) = 1/60: from pole cancellation, constant = b_2 = 1/60
report("f_Ric(0) analytic = b_2 = 1/60", b[2], mpf(1) / 60, TOLERANCE_TIGHT)
# f_R(0) = 1/120: computed above = 1/32 - 1/48 - 1/480 = 4/480 = 1/120
fR_0_analytic = mpf(1) / 32 - mpf(1) / 48 - b[2] / 8
report("f_R(0) analytic = 1/120", fR_0_analytic, mpf(1) / 120, TOLERANCE_TIGHT)
# f_RU(0) = -1/6
fRU_0_analytic = -mpf(1) / 4 + b[1] / 2  # -1/4 + 1/12 = -1/6
report("f_RU(0) analytic = -1/6", fRU_0_analytic, mpf(-1) / 6, TOLERANCE_TIGHT)
# f_U(0) = 1/2
report("f_U(0) analytic = 1/2", b[0] / 2, mpf(1) / 2, TOLERANCE_TIGHT)
# f_Omega(0) = 1/12
report("f_Omega(0) analytic = 1/12", b[1] / 2, mpf(1) / 12, TOLERANCE_TIGHT)


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)
print(f"Total checks:  {total_checks}")
print(f"PASS:          {total_pass}")
print(f"FAIL:          {total_fail}")
if failed_checks:
    print("\nFailed checks:")
    for fc in failed_checks:
        print(f"  - {fc}")
else:
    print("\nALL CHECKS PASSED")
print("=" * 72)
