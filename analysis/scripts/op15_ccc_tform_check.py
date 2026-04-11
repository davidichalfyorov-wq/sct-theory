"""
OP-15: Cross-check CCC coefficients against Decanini-Folacci.

Strategy: Compute a_6 on-shell CCC coefficient for scalar/Dirac/vector
using Vassilevich eq.(4.29) formula directly, in two normalizations:
  1. Our convention: (4pi)^2 * 7! * a_6
  2. DF convention: W = (1/(192 pi^2 m^2)) int sqrt(g) [...]

Then verify the conversion factor and resolve the discrepancy.

The key question: does our -109/3 (Dirac CCC) correctly account for
the Omega^3 contribution, and does it use the same sign convention
for fermion loops as Decanini-Folacci?
"""
import mpmath
mpmath.mp.dps = 50
mp = mpmath

from fractions import Fraction as F

print("=" * 70)
print("OP-15: CCC coefficient normalization check")
print("=" * 70)
print()

# ======================================================================
# SECTION 1: Our MR-5b coefficients
# ======================================================================
print("SECTION 1: Our MR-5b CCC coefficients")
print("Convention: (4pi)^2 * 7! * a_6, chain CCC contraction")
print()

our_scalar_CCC = F(-16, 3)
our_dirac_CCC = F(-109, 3)  # = -64/3 (geom*4) + (-45/3) (Omega^3)
our_vector_CCC = F(148, 3)   # = 116/3 (unconstrained) - 2*(-16/3) (ghosts)
our_vector_unconstrained = F(116, 3)

print(f"  Scalar (minimal):    {our_scalar_CCC} = {float(our_scalar_CCC):.6f}")
print(f"  Dirac:               {our_dirac_CCC} = {float(our_dirac_CCC):.6f}")
print(f"    decomposition: geometry = {F(-64,3)}, Omega^3 = {F(-45,3)}")
print(f"  Vector (with 2 FP): {our_vector_CCC} = {float(our_vector_CCC):.6f}")
print(f"    = unconstrained {our_vector_unconstrained} - 2*ghost {2*our_scalar_CCC}")
print()

# ======================================================================
# SECTION 2: Decanini-Folacci on-shell CCC (from 0706.0691)
# ======================================================================
print("SECTION 2: Decanini-Folacci on-shell CCC coefficients")
print("Convention: W = (1/(192 pi^2 m^2)) int sqrt(g) [...], I1 contraction")
print()

# DF eq.(3.13) scalar (xi=0), on Ricci-flat shell:
# Only terms 9,10 survive: (17/7560)*I1 + (-1/270)*I2
# In 4D: I1 = 2*I2 (Mistry identity), so I2 = I1/2
# Total = (17/7560)*I1 + (-1/270)*(I1/2) = (17/7560 - 1/540)*I1
df_scalar = F(17, 7560) - F(1, 540)
print(f"  DF scalar (xi=0): {df_scalar} = {float(df_scalar):.10f}")
print(f"    = {df_scalar.numerator}/{df_scalar.denominator}")

# DF eq.(5.21) Dirac, on Ricci-flat:
# Terms 9,10: (29/7560)*I1 + (-1/108)*I2 = (29/7560 - 1/216)*I1
df_dirac = F(29, 7560) - F(1, 216)
print(f"  DF Dirac:         {df_dirac} = {float(df_dirac):.10f}")
print(f"    = {df_dirac.numerator}/{df_dirac.denominator}")

# DF eq.(5.24) Vector (Proca), on Ricci-flat:
# Terms 9,10: (-67/2520)*I1 + (1/18)*I2 = (-67/2520 + 1/36)*I1
df_vector_proca = F(-67, 2520) + F(1, 36)
print(f"  DF vector (Proca):{df_vector_proca} = {float(df_vector_proca):.10f}")
print(f"    = {df_vector_proca.numerator}/{df_vector_proca.denominator}")

# DF vector with 2 FP ghost subtraction:
df_vector_gauge = df_vector_proca - 2 * df_scalar
print(f"  DF vector (gauge): {df_vector_gauge} = {float(df_vector_gauge):.10f}")
print(f"    = Proca - 2*scalar = {df_vector_proca} - 2*{df_scalar}")
print()

# ======================================================================
# SECTION 3: Derive conversion factor analytically
# ======================================================================
print("SECTION 3: Conversion factor analysis")
print()

# The effective action for a BOSONIC field:
# W_boson = -(1/2) Tr ln(D) = -(1/2) sum_n (-1)^{n+1}/(n) int a_{2n} t^{n-d/2} ...
# In dimensional regularization for the a_6 (n=3, d=4) term:
# W|_{a6} = -(1/2) (4pi)^{-2} int a_6 * (1/(m^2)) [the m^2 from the regulated integral]
#         = -1/(32 pi^2 m^2) int a_6

# Where a_6 in Vassilevich convention = (4pi)^{-2} tr{...}/7!
# So: a_6 = (4pi)^{-2} * (our_coeff / 7!) * CCC
# W|_{a6} = -1/(32 pi^2 m^2) * (4pi)^{-2} * (our_coeff / 5040) * int CCC
#         = -our_coeff / (32 * 16 * pi^4 * 5040 * m^2) * int CCC

# DF writes: W = 1/(192 pi^2 m^2) * df_coeff * int CCC
# For BOSONS: df_coeff = -our_coeff / (192 pi^2 * 32 * 16 * pi^4 * 5040) ???
# No, wait. Let me redo more carefully.

# Vassilevich eq.(4.29): a_6(1,D) = (4pi)^{-n/2} int sqrt(g) tr_V{1/7! * [...] }
# In d=4: a_6 = (4pi)^{-2} * (1/7!) * int sqrt(g) tr{...}
# Our convention: "coeff in tr{...}" = our_coeff, i.e.,
# tr{...}|_CCC = our_coeff * CCC (per field)

# For SCALAR (tr_V = tr over 1-dim space = 1):
# a_6(scalar) = (4pi)^{-2} * (1/7!) * our_scalar_CCC * int CCC

# Effective action for scalar (real):
# W_scalar = -(1/2) int_0^inf dt/t e^{-tm^2} (4pi t)^{-2} [t^3 * a_6_integrand + ...]
# The t^3 * (4pi)^{-2} * (1/7!) * our_coeff integral gives:
# W|_{a6} = -(1/2) * (4pi)^{-2} * (1/7!) * our_coeff * int CCC * integral_0^inf t^{3-1-2} e^{-tm^2} dt
# Wait, I need to be more careful with the heat kernel trace formula.

# Standard: Tr(e^{-tD}) = sum_n a_n(D) t^{(n-d)/2}  [d = dimension of manifold]
# For n=6, d=4: t^{(6-4)/2} = t^1
#
# The effective action: W = -(1/2) integral_0^inf dt/t e^{-tm^2} Tr(e^{-tD})
# a_6 contribution: -(1/2) integral_0^inf dt/t * t * e^{-tm^2} * a_6
#                  = -(1/2) * a_6 * integral_0^inf e^{-tm^2} dt
#                  = -(1/2) * a_6 * (1/m^2)
#                  = -a_6 / (2 m^2)

# So: W|_{a6} = -a_6 / (2 m^2)
# With a_6 = (4pi)^{-2} * (our_coeff/7!) * int CCC:
# W|_{a6} = -(4pi)^{-2} * (our_coeff/7!) / (2 m^2) * int CCC
#         = -our_coeff / (2 * 16pi^2 * 5040 * m^2) * int CCC
#         = -our_coeff / (161280 pi^2 m^2) * int CCC

# DF: W = (1/(192 pi^2 m^2)) * df_coeff * int CCC
# So: -our_coeff / (161280 pi^2 m^2) = df_coeff / (192 pi^2 m^2)
# df_coeff = -192 * our_coeff / 161280 = -our_coeff / 840

conv_boson = F(-1, 840)
print(f"Boson conversion: DF_coeff = {conv_boson} * our_coeff")
print()

# Test on scalar:
print("Test scalar:")
df_from_our = conv_boson * our_scalar_CCC
print(f"  {conv_boson} * ({our_scalar_CCC}) = {df_from_our} = {float(df_from_our):.10f}")
print(f"  DF scalar = {df_scalar} = {float(df_scalar):.10f}")
print(f"  Match: {df_from_our == df_scalar}")
print()

# For DIRAC fermion:
# W_fermion = +Tr ln(D_slash) = +(1/2) Tr ln(D^2)
# The sign is OPPOSITE to bosons! So:
# W|_{a6} = +a_6 / (2 m^2) = +(4pi)^{-2} * (our_coeff/7!) / (2 m^2) * int CCC
# df_coeff_fermion = +192 * our_coeff / 161280 = +our_coeff / 840

conv_fermion = F(1, 840)
print(f"Fermion conversion: DF_coeff = {conv_fermion} * our_coeff")
print()

# Test on Dirac:
print("Test Dirac:")
df_from_our_dirac = conv_fermion * our_dirac_CCC
print(f"  {conv_fermion} * ({our_dirac_CCC}) = {df_from_our_dirac} = {float(df_from_our_dirac):.10f}")
print(f"  DF Dirac = {df_dirac} = {float(df_dirac):.10f}")
print(f"  Match: {df_from_our_dirac == df_dirac}")
print()

# Test on vector (Proca = unconstrained, no ghosts):
print("Test vector (unconstrained/Proca):")
df_from_our_vector = conv_boson * our_vector_unconstrained
print(f"  {conv_boson} * ({our_vector_unconstrained}) = {df_from_our_vector} = {float(df_from_our_vector):.10f}")
print(f"  DF Proca = {df_vector_proca} = {float(df_vector_proca):.10f}")
print(f"  Match: {df_from_our_vector == df_vector_proca}")
print()

# Test on vector with ghosts:
print("Test vector (with 2 FP ghosts):")
df_from_our_vg = conv_boson * our_vector_CCC
print(f"  {conv_boson} * ({our_vector_CCC}) = {df_from_our_vg} = {float(df_from_our_vg):.10f}")
print(f"  DF gauge = {df_vector_gauge} = {float(df_vector_gauge):.10f}")
print(f"  Match: {df_from_our_vg == df_vector_gauge}")
print()

# ======================================================================
# SECTION 4: SM total cross-check
# ======================================================================
print("=" * 70)
print("SECTION 4: SM total CCC cross-check")
print("=" * 70)
print()

# In DF convention: W_SM = sum over species of DF_coeff * 1/(192 pi^2 m^2) int CCC
# Scalars (bosonic): N_s * conv_boson * our_scalar
# Dirac (fermionic): N_D * conv_fermion * our_dirac
# Vectors (bosonic, with ghosts): N_v * conv_boson * our_vector_CCC

N_s = 4
N_D = F(45, 2)
N_v = 12

sm_df = N_s * conv_boson * our_scalar_CCC + N_D * conv_fermion * our_dirac_CCC + N_v * conv_boson * our_vector_CCC
print(f"SM total in DF convention: {sm_df} = {float(sm_df):.10f}")
print(f"  = {sm_df.numerator}/{sm_df.denominator}")
print()

# And in our convention:
sm_our = N_s * our_scalar_CCC + N_D * our_dirac_CCC + N_v * our_vector_CCC
print(f"SM total in our convention: {sm_our} = {float(sm_our):.6f}")
print(f"  = {sm_our.numerator}/{sm_our.denominator} (expected: -1481/6)")
print(f"  Match: {sm_our == F(-1481, 6)}")
