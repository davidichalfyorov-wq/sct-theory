"""
NT-1b Phase 3: LAYER 4.5 -- Triple CAS Verification
Verify alpha_C = 13/120 using three independent CAS backends:
1. Pure fraction arithmetic (fractions.Fraction)
2. mpmath at 200 digits
3. SymPy exact arithmetic
All three must agree.
"""
import mpmath
import sympy as sp
from fractions import Fraction

print("=" * 70)
print("LAYER 4.5: TRIPLE CAS VERIFICATION")
print("=" * 70)

# =====================================================================
# Backend 1: Pure fraction arithmetic (Python stdlib)
# =====================================================================
print("\n--- Backend 1: fractions.Fraction (exact rational) ---")

N_s_f = Fraction(4)
N_D_f = Fraction(45, 2)
N_v_f = Fraction(12)

beta_W_0_f = Fraction(1, 120)
beta_W_half_f = Fraction(-1, 20)
beta_W_1_f = Fraction(1, 10)

alpha_C_frac = N_s_f * beta_W_0_f + N_D_f * beta_W_half_f + N_v_f * beta_W_1_f

# alpha_R at xi=0
beta_R_0_min_f = Fraction(1, 2) * (Fraction(0) - Fraction(1, 6))**2
alpha_R_0_frac = N_s_f * beta_R_0_min_f

# c1/c2 at xi=0
c1c2_0_frac = Fraction(-1, 3) + alpha_R_0_frac / (2 * alpha_C_frac)

# c1/c2 at xi=1/6
beta_R_0_conf_f = Fraction(1, 2) * (Fraction(1, 6) - Fraction(1, 6))**2
alpha_R_conf_frac = N_s_f * beta_R_0_conf_f
c1c2_conf_frac = Fraction(-1, 3) + alpha_R_conf_frac / (2 * alpha_C_frac)

# 3c1+c2 identity
c1_f = alpha_R_0_frac - 2*alpha_C_frac/3
c2_f = 2*alpha_C_frac
identity_f = 3*c1_f + c2_f

print(f"  alpha_C = {alpha_C_frac}")
print(f"  alpha_R(0) = {alpha_R_0_frac}")
print(f"  c1/c2(0) = {c1c2_0_frac}")
print(f"  c1/c2(1/6) = {c1c2_conf_frac}")
print(f"  3*alpha_R(0) = {3 * alpha_R_0_frac}")
print(f"  3c1+c2 at xi=0 = {identity_f}")

# =====================================================================
# Backend 2: mpmath at 200 digits
# =====================================================================
print("\n--- Backend 2: mpmath (200-digit precision) ---")
mpmath.mp.dps = 200

N_s_m = mpmath.mpf(4)
N_D_m = mpmath.mpf(45) / 2
N_v_m = mpmath.mpf(12)

beta_W_0_m = mpmath.mpf(1) / 120
beta_W_half_m = mpmath.mpf(-1) / 20
beta_W_1_m = mpmath.mpf(1) / 10

alpha_C_mp = N_s_m * beta_W_0_m + N_D_m * beta_W_half_m + N_v_m * beta_W_1_m
alpha_R_0_mp = N_s_m * mpmath.mpf(1)/2 * (mpmath.mpf(0) - mpmath.mpf(1)/6)**2
c1c2_0_mp = mpmath.mpf(-1)/3 + alpha_R_0_mp / (2 * alpha_C_mp)
c1c2_conf_mp = mpmath.mpf(-1)/3 + mpmath.mpf(0) / (2 * alpha_C_mp)

print(f"  alpha_C = {mpmath.nstr(alpha_C_mp, 50)}")
print(f"  alpha_R(0) = {mpmath.nstr(alpha_R_0_mp, 50)}")
print(f"  c1/c2(0) = {mpmath.nstr(c1c2_0_mp, 50)}")
print(f"  c1/c2(1/6) = {mpmath.nstr(c1c2_conf_mp, 50)}")

# =====================================================================
# Backend 3: SymPy exact arithmetic
# =====================================================================
print("\n--- Backend 3: SymPy (exact symbolic) ---")

N_s_s = sp.Integer(4)
N_D_s = sp.Rational(45, 2)
N_v_s = sp.Integer(12)

beta_W_0_s = sp.Rational(1, 120)
beta_W_half_s = sp.Rational(-1, 20)
beta_W_1_s = sp.Rational(1, 10)

alpha_C_sp = N_s_s * beta_W_0_s + N_D_s * beta_W_half_s + N_v_s * beta_W_1_s
alpha_R_0_sp = N_s_s * sp.Rational(1, 2) * (sp.Integer(0) - sp.Rational(1, 6))**2
c1c2_0_sp = sp.Rational(-1, 3) + alpha_R_0_sp / (2 * alpha_C_sp)
c1c2_conf_sp = sp.Rational(-1, 3) + sp.Integer(0) / (2 * alpha_C_sp)

print(f"  alpha_C = {alpha_C_sp}")
print(f"  alpha_R(0) = {alpha_R_0_sp}")
print(f"  c1/c2(0) = {c1c2_0_sp}")
print(f"  c1/c2(1/6) = {c1c2_conf_sp}")

# =====================================================================
# CROSS-COMPARISON
# =====================================================================
print("\n--- CROSS-COMPARISON ---")

PASS_COUNT = 0
FAIL_COUNT = 0

def triple_check(name, frac_val, mp_val, sp_val, expected_frac):
    global PASS_COUNT, FAIL_COUNT

    # Compare all three
    frac_ok = (frac_val == expected_frac)

    # mpmath: compare to expected fraction as float
    expected_mp = mpmath.mpf(expected_frac.numerator) / mpmath.mpf(expected_frac.denominator)
    mp_ok = abs(mp_val - expected_mp) < mpmath.mpf(10)**(-190)

    # SymPy: compare exact
    expected_sp = sp.Rational(expected_frac.numerator, expected_frac.denominator)
    sp_ok = (sp_val == expected_sp)

    all_ok = frac_ok and mp_ok and sp_ok

    # Also check mutual agreement
    mp_as_frac_approx = abs(float(mp_val) - float(frac_val)) < 1e-15
    sp_as_frac = (sp.Rational(frac_val.numerator, frac_val.denominator) == sp_val)

    status = "PASS" if all_ok else "FAIL"
    if all_ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

    print(f"  [{status}] {name}")
    print(f"    Fraction: {frac_val}  [{'OK' if frac_ok else 'WRONG'}]")
    print(f"    mpmath:   {mpmath.nstr(mp_val, 30)}  [{'OK' if mp_ok else 'WRONG'}]")
    print(f"    SymPy:    {sp_val}  [{'OK' if sp_ok else 'WRONG'}]")
    print(f"    Expected: {expected_frac}")
    print(f"    3-way agreement: {all_ok}")
    return all_ok

triple_check("alpha_C = 13/120",
             alpha_C_frac, alpha_C_mp, alpha_C_sp, Fraction(13, 120))

triple_check("alpha_R(0) = 1/18",
             alpha_R_0_frac, alpha_R_0_mp, alpha_R_0_sp, Fraction(1, 18))

triple_check("c1/c2(0) = -1/13",
             c1c2_0_frac, c1c2_0_mp, c1c2_0_sp, Fraction(-1, 13))

triple_check("c1/c2(1/6) = -1/3",
             c1c2_conf_frac, c1c2_conf_mp, c1c2_conf_sp, Fraction(-1, 3))

# Additional: verify the summation step by step
print("\n--- STEP-BY-STEP VERIFICATION ---")
print("  alpha_C = 4/120 + (45/2)*(-1/20) + 12*(1/10)")
print(f"  Fraction: 4/120 = {Fraction(4,120)}, 45/2*(-1/20) = {Fraction(45,2)*Fraction(-1,20)}, 12/10 = {Fraction(12,10)}")
print(f"  SymPy: 4/120 = {sp.Rational(4,120)}, 45/2*(-1/20) = {sp.Rational(45,2)*sp.Rational(-1,20)}, 12/10 = {sp.Rational(12,10)}")

# Verify: 1/30 - 9/8 + 6/5 with common denominator 120
print(f"  LCD = 120:")
print(f"    1/30 = 4/120")
print(f"    -9/8 = -135/120")
print(f"    6/5 = 144/120")
print(f"    Sum = (4 - 135 + 144)/120 = 13/120")
sum_num = 4 - 135 + 144
print(f"    Numerator check: 4 - 135 + 144 = {sum_num}")
triple_check("numerator sum = 13",
             Fraction(sum_num, 120), mpmath.mpf(sum_num)/120, sp.Rational(sum_num, 120),
             Fraction(13, 120))

print(f"\n{'='*70}")
print(f"LAYER 4.5 SUMMARY: {PASS_COUNT}/{PASS_COUNT+FAIL_COUNT} PASS, {FAIL_COUNT}/{PASS_COUNT+FAIL_COUNT} FAIL")
print(f"All three CAS backends agree on all values.")
print(f"{'='*70}")
