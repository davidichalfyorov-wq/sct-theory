"""
Frobenius subleading coefficients for SCT localized equation near r=0
on Schwarzschild background (M=1).

ODE (homogeneous part, after multiplying by r^3):
    r^2(r-2) u'' + 2r(r-1) u' - 6(r-2) u = 0

where H = 1 - 2M/r with M=1.

The full ODE with source:
    H u'' + (H' + 2H/r) u' - 6H/r^2 u = (Lambda^2/Dtau)(u - W)
    W = -2M/r^3

Frobenius ansatz: u = r^s (1 + a_1 r + a_2 r^2 + ...)
Indicial equation: s^2 - 6 = 0 => s = +/- sqrt(6)

This script computes:
1. Subleading coefficients a_1, a_2, a_3 for regular branch s=+sqrt(6)
2. Subleading coefficients b_1, b_2, b_3 for singular branch s=-sqrt(6)
3. Source term contribution at each order
4. Mass function m(r) behavior near r=0
5. Numerical verification with mpmath
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import sympy as sp
from sympy import sqrt, Rational, symbols, expand, simplify, oo, pi
import mpmath

mpmath.mp.dps = 50

# =====================================================================
# PART 1: Homogeneous Frobenius analysis
# =====================================================================

print("=" * 70)
print("FROBENIUS ANALYSIS: SCT Localized Eq. on Schwarzschild (M=1)")
print("=" * 70)
print()

s = sp.Symbol('s')
a1, a2, a3, a4, a5 = sp.symbols('a1 a2 a3 a4 a5')

# ODE after multiplying by r^3/r^s:
# r^2(r-2) u'' + 2r(r-1) u' - 6(r-2) u = 0
#
# With u = r^s * sum_{n>=0} a_n r^n (a_0=1):
# u'' / r^{s-2} has nth coeff = (s+n)(s+n-1) a_n
# u'  / r^{s-1} has nth coeff = (s+n) a_n
# u   / r^s     has nth coeff = a_n
#
# Multiplying:
# r^2(r-2) * r^{s-2} * sum P_n a_n r^n  =>  r^s * (r-2) * sum P_n a_n r^n
# 2r(r-1) * r^{s-1} * sum Q_n a_n r^n   =>  r^s * 2(r-1) * sum Q_n a_n r^n
# -6(r-2) * r^s * sum a_n r^n            =>  r^s * (-6)(r-2) * sum a_n r^n
#
# where P_n = (s+n)(s+n-1), Q_n = (s+n)
#
# Expanding (r-2)*f = r*f - 2*f  and  (r-1)*f = r*f - f:
#
# Coeff of r^n (after factoring r^s):
# = [P_{n-1} a_{n-1} - 2 P_n a_n] + 2[Q_{n-1} a_{n-1} - Q_n a_n] + (-6)[a_{n-1} - 2 a_n]
#   (with a_{-1} = 0)
#
# = (P_{n-1} + 2Q_{n-1} - 6) a_{n-1} + (-2P_n - 2Q_n + 12) a_n


def P(n):
    return (s + n) * (s + n - 1)


def Q(n):
    return (s + n)


def indicial(n_val):
    """Coefficient of a_n in the r^n equation."""
    return -2 * P(n_val) - 2 * Q(n_val) + 12


def coupling(n_val):
    """Coefficient of a_{n-1} in the r^n equation."""
    return P(n_val - 1) + 2 * Q(n_val - 1) - 6


# Indicial equation: n=0, coeff of a_0 = 0
ind_eq = expand(indicial(0))
print(f"Indicial equation: {ind_eq} = 0")
print(f"  => -2s^2 + 12 = 0  =>  s^2 = 6")
roots = sp.solve(ind_eq, s)
print(f"  Roots: s = {roots}")
assert set(roots) == {sqrt(6), -sqrt(6)}
print()

s_reg = sqrt(6)
s_sing = -sqrt(6)

# =====================================================================
# PART 2: Regular branch s = +sqrt(6)
# =====================================================================

print("=" * 70)
print("REGULAR BRANCH: s = sqrt(6)")
print("=" * 70)
print()

def solve_frobenius_coeffs(s_val, n_max=5):
    """Solve Frobenius recurrence for coefficients a_1, ..., a_{n_max}."""
    a_syms = [sp.Symbol(f'a{k}') for k in range(n_max + 1)]
    a_vals = {a_syms[0]: sp.Integer(1)}

    results = []
    for n in range(1, n_max + 1):
        # Equation: indicial(n)*a_n + coupling(n)*a_{n-1} = 0
        # (plus terms from a_{n-2}, etc. if n > 1 due to higher order couplings)
        # Actually, our recurrence is 2-term: only a_n and a_{n-1} contribute.

        ind_n = expand(indicial(n).subs(s, s_val))
        coup_n = expand(coupling(n).subs(s, s_val))

        # a_n = -coupling(n)/indicial(n) * a_{n-1}
        a_nm1_val = a_vals.get(a_syms[n-1], a_syms[n-1])
        a_n_val = simplify(-coup_n / ind_n * a_nm1_val)
        a_vals[a_syms[n]] = a_n_val

        results.append({
            'n': n,
            'indicial_coeff': ind_n,
            'coupling_coeff': coup_n,
            'ratio': simplify(-coup_n / ind_n),
            'a_n': a_n_val,
            'a_n_float': float(a_n_val.evalf(30)),
        })

        print(f"  n={n}:")
        print(f"    indicial coeff = {ind_n}")
        print(f"    coupling coeff = {coup_n}")
        print(f"    a_{n}/a_{n-1} = {simplify(-coup_n / ind_n)}")
        print(f"    a_{n} = {a_n_val}")
        print(f"    a_{n} (float) = {float(a_n_val.evalf(30)):.15f}")
        print()

    return results, a_vals


print("Regular branch coefficients:")
print()
reg_results, reg_vals = solve_frobenius_coeffs(s_reg, n_max=5)

print()
print("=" * 70)
print("SINGULAR BRANCH: s = -sqrt(6)")
print("=" * 70)
print()

print("Singular branch coefficients:")
print()
sing_results, sing_vals = solve_frobenius_coeffs(s_sing, n_max=5)

# =====================================================================
# PART 3: Source term analysis
# =====================================================================

print()
print("=" * 70)
print("PART 3: SOURCE TERM CONTRIBUTION")
print("=" * 70)
print()

# The full ODE with source (multiplied by r^3):
# r^2(r-2) u'' + 2r(r-1) u' - 6(r-2) u = r^3 * (Lambda^2/Dtau)(u - W)
#
# W = -2/r^3, so u - W = u + 2/r^3
# RHS = r^3 * mu * (u + 2/r^3) = mu * (r^3 u + 2)
#   where mu = Lambda^2/Dtau
#
# For u ~ r^{sqrt(6)} near r=0:
# r^3 * u ~ r^{3+sqrt(6)} ~ r^{5.449}
# The constant term 2*mu contributes at r^0
#
# So the source modifies the Frobenius structure at SPECIFIC orders.

print("Source: RHS = mu * r^3 * u + 2*mu")
print(f"  where mu = Lambda^2/Dtau")
print()
print("For u = r^s (1 + a1*r + ...) with s = sqrt(6) ~ 2.449:")
print(f"  r^3 * u ~ r^{{3+s}} = r^{{{3+float(sqrt(6).evalf()):.4f}}}")
print(f"  This is NON-INTEGER power, so it does NOT couple to")
print(f"  the integer-step Frobenius series unless we include it as")
print(f"  a separate series.")
print()
print("The constant source term 2*mu modifies the particular solution.")
print("For the HOMOGENEOUS Frobenius coefficients computed above,")
print("the source does not affect them -- it generates a separate")
print("particular solution branch.")
print()
print("Particular solution from the constant source:")
print("  u_p = const => 0 = -6(-2)*const + 2*mu")
print("  => 12*u_p = -2*mu => u_p = -mu/6")
print("  Check: H*0 + 0 - 6H/r^2 * u_p = -6H u_p/r^2 = 6*(2/r)*(mu/6)/r^2 = 2*mu/r^3")
print("  RHS: mu*(u_p - W) = mu*(-mu/6 + 2/r^3)")
print("  These don't match at leading order in 1/r => particular solution is more complex.")
print()

# More careful particular solution
print("--- Particular solution (dominant balance near r=0) ---")
print()
print("Near r=0, the dominant terms of the homogeneous ODE are:")
print("  -2/r * u'' + (-2/r^2)*u' + 12/r^2 * u   (from the -2 parts of (r-2), etc.)")
print("  => -2r u'' - 2 u' + 12 u = RHS * r^2")
print()
print("For the source W = -2/r^3:")
print("  mu*(u - W) = mu*u + 2mu/r^3")
print("  Multiplied by r^3 (as in our standard form):")
print("  RHS_std = mu*r^3*u + 2*mu")
print()
print("At r->0, the 2*mu constant source generates a particular solution")
print("that enters as a correction to the homogeneous solution.")
print("Since 2*mu is O(r^0) and the homogeneous solutions are r^{+/-sqrt(6)},")
print("the particular solution for the constant forcing is:")
print("  u_p = (2*mu) / indicial(0) evaluated at s=0")
ind_at_0 = expand(indicial(0).subs(s, 0))
print(f"  indicial(s=0) = {ind_at_0}")
print(f"  u_p = 2*mu / {ind_at_0} = mu/6")
print(f"  (This is a constant particular solution for the r^0 forcing.)")
print()

# =====================================================================
# PART 4: Mass function behavior
# =====================================================================

print("=" * 70)
print("PART 4: MASS FUNCTION m(r) NEAR r=0")
print("=" * 70)
print()

print("Weyl scalar W related to mass function m(r) by:")
print("  W = (-r^2 m'' + 4r m' - 6m) / (6 r^3)")
print()
print("If u ~ r^{sqrt(6)}, then J = sum(c_k u_k) ~ r^{sqrt(6)}.")
print("The Weyl amplitude correction dW ~ J ~ r^{sqrt(6)}.")
print("Total W_total = W_background + dW ~ -2/r^3 + A*r^{sqrt(6)}")
print()
print("For the correction part: W_corr = A*r^{sqrt(6)} = (-r^2 m_corr'' + 4r m_corr' - 6m_corr)/(6r^3)")
print()
print("Try m_corr = B * r^alpha:")
print("  m_corr'' = B*alpha*(alpha-1)*r^{alpha-2}")
print("  m_corr'  = B*alpha*r^{alpha-1}")
print("  W_corr = B*(-alpha*(alpha-1)*r^alpha + 4*alpha*r^alpha - 6*r^alpha) / (6*r^3)")
print("         = B * r^{alpha-3} * (-alpha^2 + alpha + 4alpha - 6) / 6")
print("         = B * r^{alpha-3} * (-alpha^2 + 5alpha - 6) / 6")
print()

alpha = sp.Symbol('alpha')
mass_indicial = -alpha**2 + 5*alpha - 6
print(f"  Mass indicial polynomial: {mass_indicial}")
print(f"  Factor: {sp.factor(mass_indicial)} = -(alpha-2)(alpha-3)")
print()
print("  Setting W_corr ~ r^{sqrt(6)}: alpha - 3 = sqrt(6) => alpha = 3 + sqrt(6)")
print(f"  alpha = 3 + sqrt(6) = {float(3 + sqrt(6).evalf()):.10f}")
print()
print("  Check: mass_indicial(3+sqrt(6)) = -(3+sqrt(6)-2)(3+sqrt(6)-3) = -(1+sqrt(6))*sqrt(6)")
mass_ind_val = mass_indicial.subs(alpha, 3 + sqrt(6))
mass_ind_simplified = simplify(mass_ind_val)
print(f"  = {mass_ind_simplified} = {float(mass_ind_simplified.evalf()):.10f}")
print()
print(f"  So m_corr = B * r^{{3+sqrt(6)}} with B = 6A / ({mass_ind_simplified})")
print(f"  B = -6A / (sqrt(6)*(1+sqrt(6))) = -6A / (sqrt(6) + 6)")
B_factor = simplify(-6 / (sqrt(6) * (1 + sqrt(6))))
print(f"  B/A = {B_factor} = {float(B_factor.evalf()):.10f}")
print()

print("  RESULT: m(r) ~ m_bg(r) + B*r^{3+sqrt(6)} near r=0")
print(f"  Exponent: 3 + sqrt(6) = {float((3 + sqrt(6)).evalf()):.10f}")
print(f"  This is > 3, so the correction is SUBLEADING to the")
print(f"  background m_bg = M = 1 (Schwarzschild constant mass).")
print()

# Also: metric function behavior
print("  Metric: f(r) = 1 - 2m(r)/r")
print("  f_bg = 1 - 2/r (Schwarzschild)")
print(f"  f_corr ~ -2B*r^{{2+sqrt(6)}} = -2B*r^{{{float((2+sqrt(6)).evalf()):.4f}}}")
print(f"  This goes to ZERO as r->0, so the correction is subdominant")
print(f"  to the 1/r divergence in f_bg.")
print()

# =====================================================================
# PART 5: Effective E^2 behavior
# =====================================================================

print("=" * 70)
print("PART 5: EFFECTIVE E^2 (Kretschner proxy) NEAR r=0")
print("=" * 70)
print()

print("E^2_eff = (3/2)(W + dW)^2")
print("W_bg = -2/r^3,  dW ~ A*r^{sqrt(6)}")
print()
print("E^2_eff = (3/2)(W_bg + dW)^2")
print("        = (3/2)[W_bg^2 + 2*W_bg*dW + dW^2]")
print("        = (3/2)[4/r^6 - 4A*r^{sqrt(6)-3} + A^2*r^{2*sqrt(6)}]")
print()
print(f"Leading: (3/2)*4/r^6 = 6/r^6  (Schwarzschild Kretschner)")
print(f"Cross term: -(3/2)*4A*r^{{sqrt(6)-3}} = -6A*r^{{sqrt(6)-3}}")
print(f"  exponent sqrt(6)-3 = {float((sqrt(6)-3).evalf()):.10f}")
print(f"  This is NEGATIVE, so the cross term DIVERGES as r->0,")
print(f"  but SLOWER than the 1/r^6 background.")
print()
print(f"  Ratio: dE^2/E^2_bg ~ A*r^{{sqrt(6)}} * r^3 = A*r^{{3+sqrt(6)}}")
print(f"  This goes to ZERO as r->0.")
print(f"  => The SCT correction to Kretschner is PERTURBATIVE near r=0.")
print()

# =====================================================================
# PART 6: Numerical verification via direct ODE substitution
# =====================================================================

print("=" * 70)
print("PART 6: NUMERICAL VERIFICATION")
print("=" * 70)
print()

# Verify by substituting the truncated series back into the ODE
# and checking that the residual is of higher order.

s6 = mpmath.sqrt(6)

# Regular branch coefficients
a1_f = float(reg_results[0]['a_n'].evalf(30))
a2_f = float(reg_results[1]['a_n'].evalf(30))
a3_f = float(reg_results[2]['a_n'].evalf(30))

print(f"Regular branch: u = r^sqrt(6) * (1 + {a1_f:.12f}*r + {a2_f:.12f}*r^2 + {a3_f:.12f}*r^3 + ...)")
print()

def u_series(r_val, s_val, coeffs, n_terms):
    """Evaluate truncated Frobenius series."""
    result = mpmath.power(r_val, s_val)
    series_sum = mpmath.mpf(1)
    for k in range(len(coeffs)):
        series_sum += coeffs[k] * mpmath.power(r_val, k + 1)
    return result * series_sum


def ode_residual(r_val, s_val, coeffs):
    """Compute ODE residual: H*u'' + (H'+2H/r)*u' - 6H/r^2*u."""
    h = 1 / mpmath.mpf(10000)  # finite difference step

    u0 = u_series(r_val, s_val, coeffs, len(coeffs))
    up = u_series(r_val + h, s_val, coeffs, len(coeffs))
    um = u_series(r_val - h, s_val, coeffs, len(coeffs))

    u_prime = (up - um) / (2 * h)
    u_double_prime = (up - 2 * u0 + um) / (h * h)

    H_val = 1 - 2 / r_val
    H_prime = 2 / (r_val * r_val)

    lhs = H_val * u_double_prime + (H_prime + 2 * H_val / r_val) * u_prime - 6 * H_val / (r_val * r_val) * u0
    return lhs


# Test at several small r values
test_points = [mpmath.mpf('0.01'), mpmath.mpf('0.05'), mpmath.mpf('0.1'), mpmath.mpf('0.2')]

print("Verification: ODE residual at test points (should decrease with more terms)")
print()

for r_test in test_points:
    # With 0 corrections
    res0 = ode_residual(r_test, s6, [])
    # With 1 correction
    res1 = ode_residual(r_test, s6, [a1_f])
    # With 2 corrections
    res2 = ode_residual(r_test, s6, [a1_f, a2_f])
    # With 3 corrections
    res3 = ode_residual(r_test, s6, [a1_f, a2_f, a3_f])

    # Normalize by u value
    u_val = u_series(r_test, s6, [a1_f, a2_f, a3_f], 3)

    print(f"  r = {float(r_test):.3f}:")
    print(f"    |residual(0 corr)|/|u| = {float(abs(res0/u_val)):.6e}")
    print(f"    |residual(1 corr)|/|u| = {float(abs(res1/u_val)):.6e}")
    print(f"    |residual(2 corr)|/|u| = {float(abs(res2/u_val)):.6e}")
    print(f"    |residual(3 corr)|/|u| = {float(abs(res3/u_val)):.6e}")

print()

# =====================================================================
# PART 7: Check for resonance (s+ - s- = integer?)
# =====================================================================

print("=" * 70)
print("PART 7: RESONANCE CHECK")
print("=" * 70)
print()

diff = 2 * sqrt(6)
print(f"s+ - s- = 2*sqrt(6) = {float(diff.evalf()):.10f}")
print(f"This is IRRATIONAL => NO RESONANCE.")
print(f"Both branches are independent. No logarithmic terms needed.")
print(f"General solution: u = C_1 * r^{{sqrt(6)}} (1 + a1*r + ...) + C_2 * r^{{-sqrt(6)}} (1 + b1*r + ...)")
print()

# =====================================================================
# PART 8: Convergence radius of the Frobenius series
# =====================================================================

print("=" * 70)
print("PART 8: CONVERGENCE RADIUS")
print("=" * 70)
print()

print("The ODE r^2(r-2)u'' + 2r(r-1)u' - 6(r-2)u = 0 has singular points at:")
print("  r = 0 (regular singular point)")
print("  r = 2 (regular singular point = horizon)")
print("  r = infinity (irregular singular point)")
print()
print("Frobenius theory guarantees convergence for |r| < distance to next singularity.")
print(f"  Convergence radius = |2 - 0| = 2 = 2M")
print(f"  (Series valid for 0 < r < 2M, i.e., INSIDE the horizon)")
print()

# Verify with ratio test
print("Ratio test (regular branch):")
for k in range(len(reg_results) - 1):
    an = reg_results[k]['a_n_float']
    anp1 = reg_results[k+1]['a_n_float']
    if abs(an) > 1e-30:
        ratio = abs(anp1 / an)
        print(f"  |a_{k+2}/a_{k+1}| = {ratio:.10f}  (limit should -> 1/R = 0.5)")

print()

# =====================================================================
# SAVE RESULTS
# =====================================================================

print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)
print()

output = {
    'computation': 'Frobenius subleading coefficients for SCT localized equation',
    'background': 'Schwarzschild (M=1)',
    'ode_standard_form': 'r^2(r-2)u\'\' + 2r(r-1)u\' - 6(r-2)u = 0',
    'indicial_equation': 's^2 - 6 = 0',
    'singular_points': {
        'r=0': 'regular singular (Frobenius center)',
        'r=2M': 'regular singular (horizon)',
        'r=inf': 'irregular singular',
    },
    'convergence_radius': 2.0,
    'resonance': False,
    'resonance_note': 's+ - s- = 2*sqrt(6) is irrational, no log terms needed',
    'regular_branch': {
        'exponent_symbolic': 'sqrt(6)',
        'exponent_numerical': float(mpmath.sqrt(6)),
        'series': 'u_reg = r^{sqrt(6)} * (1 + a1*r + a2*r^2 + a3*r^3 + a4*r^4 + a5*r^5 + ...)',
        'coefficients': {},
    },
    'singular_branch': {
        'exponent_symbolic': '-sqrt(6)',
        'exponent_numerical': float(-mpmath.sqrt(6)),
        'series': 'u_sing = r^{-sqrt(6)} * (1 + b1*r + b2*r^2 + b3*r^3 + b4*r^4 + b5*r^5 + ...)',
        'coefficients': {},
    },
    'mass_function': {
        'correction_exponent_symbolic': '3 + sqrt(6)',
        'correction_exponent_numerical': float(3 + mpmath.sqrt(6)),
        'm_corr': 'B * r^{3+sqrt(6)}',
        'B_over_A': float(B_factor.evalf(30)),
        'note': 'Correction is subleading to constant m_bg = M at r=0',
    },
    'E_squared': {
        'background': '6/r^6 (standard Schwarzschild Kretschner)',
        'cross_term_exponent': float(mpmath.sqrt(6) - 3),
        'relative_correction': 'dE^2/E^2 ~ A*r^{3+sqrt(6)} -> 0 as r->0',
        'conclusion': 'SCT correction is perturbative near singularity',
    },
    'source_term': {
        'W': '-2/r^3',
        'mu': 'Lambda^2 / Dtau',
        'particular_solution_constant': 'mu/6 (from r^0 forcing)',
        'coupling_note': 'Source r^3*u term generates non-integer power r^{3+sqrt(6)}, separate from Frobenius series',
    },
    'recurrence_relation': {
        'formula': 'a_n = -(P_{n-1} + 2Q_{n-1} - 6) / (-2P_n - 2Q_n + 12) * a_{n-1}',
        'P_n': '(s+n)(s+n-1)',
        'Q_n': '(s+n)',
        'indicial_factor': '-2(s+n)^2 + 12',
        'coupling_factor': '(s+n-1)(s+n-2) + 2(s+n-1) - 6 = (s+n-1)^2 + (s+n-1) - 6',
    },
}

for i, res in enumerate(reg_results):
    n = res['n']
    output['regular_branch']['coefficients'][f'a{n}'] = {
        'symbolic': str(res['a_n']),
        'numerical': res['a_n_float'],
        'ratio_to_previous': str(res['ratio']),
        'ratio_numerical': float(res['ratio'].evalf(30)),
    }

for i, res in enumerate(sing_results):
    n = res['n']
    output['singular_branch']['coefficients'][f'b{n}'] = {
        'symbolic': str(res['a_n']),
        'numerical': res['a_n_float'],
        'ratio_to_previous': str(res['ratio']),
        'ratio_numerical': float(res['ratio'].evalf(30)),
    }

outpath = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'gap_g1', 'frobenius_subleading.json')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {outpath}")
print()

# =====================================================================
# SUMMARY TABLE
# =====================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"{'Branch':<12} {'Exponent':<20} {'a1/b1':<20} {'a2/b2':<20} {'a3/b3':<20}")
print("-" * 92)
print(f"{'Regular':<12} {'sqrt(6)='+str(round(float(s6),6)):<20} "
      f"{reg_results[0]['a_n_float']:<20.12f} "
      f"{reg_results[1]['a_n_float']:<20.12f} "
      f"{reg_results[2]['a_n_float']:<20.12f}")
print(f"{'Singular':<12} {'-sqrt(6)='+str(round(float(-s6),6)):<20} "
      f"{sing_results[0]['a_n_float']:<20.12f} "
      f"{sing_results[1]['a_n_float']:<20.12f} "
      f"{sing_results[2]['a_n_float']:<20.12f}")
print()
print("General solution near r=0:")
print(f"  u(r) = C_1 r^{{+sqrt(6)}} [1 + {reg_results[0]['a_n_float']:.8f} r + {reg_results[1]['a_n_float']:.8f} r^2 + ...]")
print(f"       + C_2 r^{{-sqrt(6)}} [1 + {sing_results[0]['a_n_float']:.8f} r + {sing_results[1]['a_n_float']:.8f} r^2 + ...]")
print()
print(f"Mass function: m(r) = M + B*r^{{3+sqrt(6)}} + ...  (exponent = {float(3+s6):.6f})")
print(f"  B/A = {float(B_factor.evalf()):.10f}")
print()
print(f"Convergence radius: R = 2M = 2 (to horizon)")
print(f"No resonance (2*sqrt(6) irrational)")
print()
print("DONE.")
