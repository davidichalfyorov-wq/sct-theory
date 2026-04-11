"""
Independent verification of Theta^{R^2}_{tt} = 0 on f = 1 - 2ar background.

Claim (analytical): For the R^2 variation tensor
  Theta^{(R^2)}_{tt} = 2 nabla_t nabla_t R - 2 g_tt box R + 2 R R_tt - (1/2) g_tt R^2
this vanishes identically when f(r) = 1 - 2*c2*r.

Metric: ds^2 = -f dt^2 + dr^2/f + r^2 (dtheta^2 + sin^2 theta dphi^2)
"""

import sympy as sp
from sympy import symbols, Rational, simplify, diff, factor, expand, cancel, Symbol
import json, os

r, a = symbols("r a", positive=True)

f = 1 - 2*a*r
fp = diff(f, r)    # = -2a
fpp = diff(f, r, 2)  # = 0

print("=" * 60)
print("METRIC: f(r) = 1 - 2ar")
print("=" * 60)
print(f"f   = {f}")
print(f"f'  = {fp}")
print(f"f'' = {fpp}")
print()

# =============================================
# RICCI TENSOR (standard SSS formulas)
# =============================================
# R_tt = f (f''/2 + f'/r)
# R_rr = -(f''/2 + f'/r)/f
# R_thth = 1 - f - r f'

R_tt = f * (fpp/2 + fp/r)
R_rr = -(fpp/2 + fp/r) / f
R_thth = 1 - f - r*fp

R_tt = simplify(R_tt)
R_rr = simplify(R_rr)
R_thth = simplify(R_thth)

print("=== RICCI TENSOR ===")
print(f"R_tt   = {R_tt}")
print(f"R_rr   = {R_rr}")
print(f"R_thth = {R_thth}")

# Cross-check: R_rr = -R_tt/f^2
check_rr = simplify(R_rr + R_tt/f**2)
print(f"Check R_rr + R_tt/f^2 = {check_rr}")
assert check_rr == 0, "R_rr consistency FAILED"
print()

# =============================================
# RICCI SCALAR
# =============================================
# R = g^{mu nu} R_{mu nu} = -R_tt/f + f R_rr + 2 R_thth/r^2
R_scalar = -R_tt/f + f*R_rr + 2*R_thth/r**2
R_scalar = simplify(expand(R_scalar))

print(f"R (Ricci scalar) = {R_scalar}")
print(f"R factored       = {factor(R_scalar)}")
print()

# =============================================
# DERIVATIVES OF R
# =============================================
dR = diff(R_scalar, r)
dR = simplify(dR)
d2R = diff(R_scalar, r, 2)
d2R = simplify(d2R)

print(f"dR/dr   = {dR}")
print(f"d^2R/dr^2 = {d2R}")
print()

# =============================================
# BOX R (covariant d'Alembertian of scalar)
# =============================================
# For scalar S(r) in SSS:
# box S = f S'' + (f' + 2f/r) S'
box_R = f * d2R + (fp + 2*f/r) * dR
box_R = simplify(box_R)

print(f"box R = {box_R}")
print(f"box R expanded = {expand(box_R)}")
print()

# =============================================
# nabla_t nabla_t R
# =============================================
# For scalar R(r):
# nabla_t R = 0 (no t-dependence)
# nabla_t nabla_t R = -Gamma^r_{tt} partial_r R
# Gamma^r_{tt} = f f'/2
Gamma_r_tt = f * fp / 2
nab_t_nab_t_R = -Gamma_r_tt * dR
nab_t_nab_t_R = simplify(nab_t_nab_t_R)

print(f"Gamma^r_tt = {simplify(Gamma_r_tt)}")
print(f"nabla_t nabla_t R = {nab_t_nab_t_R}")
print()

# =============================================
# THETA^{R^2}_{tt}
# =============================================
# Theta_{tt} = 2 nabla_t nabla_t R - 2 g_tt box R + 2 R R_tt - (1/2) g_tt R^2
# g_tt = -f

g_tt = -f
Theta_tt = (2*nab_t_nab_t_R
            - 2*g_tt*box_R
            + 2*R_scalar*R_tt
            - Rational(1, 2)*g_tt*R_scalar**2)

print("=== THETA^{R^2}_{tt} ===")
print(f"Term 1: 2 nabla_t nabla_t R = {simplify(2*nab_t_nab_t_R)}")
print(f"Term 2: -2 g_tt box R = {simplify(-2*g_tt*box_R)}")
print(f"Term 3: 2 R R_tt = {simplify(2*R_scalar*R_tt)}")
print(f"Term 4: -(1/2) g_tt R^2 = {simplify(-Rational(1,2)*g_tt*R_scalar**2)}")
print()

Theta_simplified = simplify(Theta_tt)
Theta_expanded = expand(Theta_tt)
Theta_cancelled = cancel(Theta_tt)
Theta_factored = factor(Theta_tt)

print(f"Theta_tt (simplified) = {Theta_simplified}")
print(f"Theta_tt (expanded)   = {Theta_expanded}")
print(f"Theta_tt (cancelled)  = {Theta_cancelled}")
print(f"Theta_tt (factored)   = {Theta_factored}")
print()

is_zero = (Theta_simplified == 0) or (Theta_cancelled == 0) or (simplify(Theta_expanded) == 0)
print(f"*** Theta^(R^2)_tt = 0?  {is_zero} ***")
print()

# =============================================
# EINSTEIN TENSOR G_tt
# =============================================
# G_tt = R_tt - (1/2) g_tt R = R_tt + (1/2) f R
G_tt = R_tt + Rational(1, 2) * f * R_scalar
G_tt = simplify(G_tt)

# Cross-check via G^t_t = -R_thth/r^2 => G_tt = f R_thth/r^2
G_tt_check = f * R_thth / r**2
assert simplify(G_tt - G_tt_check) == 0, "G_tt consistency FAILED"

print(f"G_tt = {G_tt}")
print(f"G_tt = {expand(G_tt)}")
print(f"G_tt (from R_thth) = {simplify(G_tt_check)}  [MATCHES]")

# Series expansion
G_tt_series = sp.series(G_tt, a, 0, n=3)
print(f"G_tt series in a = {G_tt_series}")
print()

# =============================================
# FIELD EQUATION ANALYSIS
# =============================================
print("=" * 60)
print("FIELD EQUATION: G_tt + alpha_R * Theta^(R^2)_tt = 0")
print("=" * 60)
if is_zero:
    print("Since Theta^(R^2)_tt = 0 identically:")
    print("  G_tt = 0")
    print(f"  {G_tt} = 0")
    print(f"  Leading order: 4a/r = 0  =>  a = c2 = 0")
    print()
    print("CONCLUSION: n=2 (f = 1 - 2c2*r) is EXCLUDED.")
    print("The R^2 sector provides NO correction to the tt field equation.")
    conclusion = "VERIFIED: Theta^(R^2)_tt = 0 identically"
    implication = "G_tt = 4a/r - 8a^2 = 0 => a = c2 = 0 => n=2 EXCLUDED"
else:
    print(f"Theta^(R^2)_tt = {Theta_simplified}")
    print("Need to include this in the field equation.")
    conclusion = f"NOT ZERO: Theta^(R^2)_tt = {Theta_simplified}"
    implication = "R^2 sector contributes; need full analysis"

print()

# =============================================
# NUMERICAL SPOT CHECK at specific r, a values
# =============================================
print("=== NUMERICAL SPOT CHECKS ===")
test_points = [(1, sp.Rational(1, 10)),
               (2, sp.Rational(1, 20)),
               (5, sp.Rational(1, 100)),
               (sp.Rational(1, 2), sp.Rational(1, 5))]

for r_val, a_val in test_points:
    subs = {r: r_val, a: a_val}
    theta_num = Theta_tt.subs(subs)
    G_num = G_tt.subs(subs)
    R_num = R_scalar.subs(subs)
    print(f"  r={r_val}, a={a_val}: f={f.subs(subs)}, R={R_num}, "
          f"G_tt={G_num}, Theta_tt={theta_num}")

print()

# =============================================
# SAVE RESULTS
# =============================================
out_dir = "analysis/results/gap_g1"
os.makedirs(out_dir, exist_ok=True)

results = {
    "task": "Verify Theta^{R^2}_{tt} on f = 1 - 2ar",
    "metric": "ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2",
    "f": str(f),
    "components": {
        "R_tt": str(R_tt),
        "R_rr": str(R_rr),
        "R_thth": str(R_thth),
        "R_scalar": str(R_scalar),
        "dR_dr": str(dR),
        "d2R_dr2": str(d2R),
        "box_R": str(box_R),
        "nabla_t_nabla_t_R": str(nab_t_nab_t_R),
        "G_tt": str(G_tt),
    },
    "Theta_R2_tt": {
        "term_1_2nab_t_nab_t_R": str(simplify(2*nab_t_nab_t_R)),
        "term_2_neg2gtt_boxR": str(simplify(-2*g_tt*box_R)),
        "term_3_2R_Rtt": str(simplify(2*R_scalar*R_tt)),
        "term_4_neg_half_gtt_R2": str(simplify(-Rational(1,2)*g_tt*R_scalar**2)),
        "total_simplified": str(Theta_simplified),
        "total_expanded": str(Theta_expanded),
        "total_factored": str(Theta_factored),
        "is_zero": is_zero,
    },
    "cross_checks": {
        "R_rr_consistency": True,
        "G_tt_two_methods": True,
        "numerical_spot_checks": [
            {"r": str(rv), "a": str(av),
             "Theta_tt": str(Theta_tt.subs({r: rv, a: av}))}
            for rv, av in test_points
        ],
    },
    "conclusion": conclusion,
    "implication": implication,
    "field_equation_consequence": (
        "G_tt + alpha_R Theta^(R^2)_tt = 0 with Theta=0 "
        "=> G_tt = 4a/r - 8a^2 = 0 => a=0 => c2=0 => n=2 excluded"
        if is_zero else "Theta nonzero; full analysis needed"
    ),
}

outpath = os.path.join(out_dir, "theta_r2_verification.json")
with open(outpath, "w") as fout:
    json.dump(results, fout, indent=2)

print(f"Results saved to {outpath}")
print()
print("=" * 60)
print("FINAL VERDICT:", conclusion)
print("=" * 60)
