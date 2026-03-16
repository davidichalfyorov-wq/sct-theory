# ruff: noqa: E402, I001
"""Debug the worldline a_2 computation step by step."""
import sys
from pathlib import Path
from itertools import product as iproduct
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

D = 4

def build_gamma():
    I2 = np.eye(2, dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    g = np.zeros((D, 4, 4), dtype=complex)
    g[0] = np.kron(s1, I2); g[1] = np.kron(s2, I2)
    g[2] = np.kron(s3, s1); g[3] = np.kron(s3, s2)
    return g

def build_sigma(g):
    s = np.zeros((D, D, 4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            s[a, b] = 0.5 * (g[a] @ g[b] - g[b] @ g[a])
    return s

gam = build_gamma()
sig = build_sigma(gam)

# Simple test: constant diagonal Omega (abelian case)
# Omega_{01} = lambda * sigma_{01}, all others zero
lam = 0.3
Omega = np.zeros((D, D, 4, 4), dtype=complex)
Omega[0, 1] = lam * sig[0, 1]
Omega[1, 0] = -lam * sig[0, 1]  # antisymmetric

# Omega_sq = sum_{mn} Omega[m,n] @ Omega[m,n]
Osq = np.zeros((4, 4), dtype=complex)
for a in range(D):
    for b in range(D):
        Osq += Omega[a, b] @ Omega[a, b]

tr_Osq = np.trace(Osq).real
print(f"tr(Omega_sq) = {tr_Osq:.6f}")
print(f"Expected: 2*lam^2*tr(sig01^2) = 2*{lam}^2*tr(sig01^2)")
print(f"  sig01^2 = {(sig[0,1]@sig[0,1]).diagonal()}")
print(f"  tr(sig01^2) = {np.trace(sig[0,1]@sig[0,1]).real}")
print(f"  2*lam^2*tr(sig01^2) = {2*lam**2*np.trace(sig[0,1]@sig[0,1]).real:.6f}")

tr_a2_exact = (1.0/12.0) * tr_Osq
print(f"\ntr(a_2^fiber) exact = (1/12)*tr(Osq) = {tr_a2_exact:.8f}")

# Now manually compute the worldline integral for a_2:
# a_2^{fiber} = (1/4) int_{0<u2<u1<1} du1 du2
#     sum_m1,n1,m2,n2 Omega[m1,n1] @ Omega[m2,n2]
#     * <z^{n1}(u1) dz^{m1}/du1 z^{n2}(u2) dz^{m2}/du2>

# Fields: 0=z^{n1}(u1), 1=dz^{m1}(u1), 2=z^{n2}(u2), 3=dz^{m2}(u2)
# Matchings of {0,1,2,3}:
# M1: (0,1)(2,3) - same vertex
# M2: (0,2)(1,3) - z1-z2, dz1-dz2
# M3: (0,3)(1,2) - z1-dz2, dz1-z2

def gb(u, v):
    return min(u, v) - u*v

def gd1(u, v):
    return (1.0-v if u < v else (-v if u > v else 0.5-v))

def gd2(u, v):
    return gd1(v, u)

# For M1: (0,1)(2,3)
# <z^{n1}(u1) dz^{m1}(u1)> = 2*Gd2(u1,u1)*delta^{n1,m1}
# <z^{n2}(u2) dz^{m2}(u2)> = 2*Gd2(u2,u2)*delta^{n2,m2}
# Index contraction: sum_{n1=m1, n2=m2} tr[Omega[m1,n1]@Omega[m2,n2]]
#   = sum_{a,b} tr[Omega[a,a]@Omega[b,b]]
# But Omega[a,a] = 0 (antisymmetric)! So M1 = 0.
print("\n--- Matching (0,1)(2,3): same-vertex contraction ---")
print("  Omega[a,a] = 0 (antisymmetric) => this matching gives 0. ✓")

# For M2: (0,2)(1,3)
# <z^{n1}(u1) z^{n2}(u2)> = 2*G(u1,u2)*delta^{n1,n2}
# <dz^{m1}(u1) dz^{m2}(u2)> = 2*Gdd(u1,u2)*delta^{m1,m2}
# Gdd(u1,u2) for u1 != u2 = -1
# Index contraction: sum_{n1=n2=n, m1=m2=m} tr[Omega[m,n]@Omega[m,n]]
#   = tr(Omega_sq)
# Propagator factor: 2*G(u1,u2) * 2*(-1) = -4*G(u1,u2)
print("\n--- Matching (0,2)(1,3): z1-z2, dz1-dz2 ---")
print("  Index contraction gives tr(Omega_sq)")
print(f"  tr(Omega_sq) = {tr_Osq:.6f}")

# Integral: int_{0<u2<u1<1} (-4)*G(u1,u2) du1 du2
# G(u1,u2) = u2 - u1*u2 = u2(1-u1) for u2 < u1
from numpy.polynomial.legendre import leggauss
nq = 30
nd, wt = leggauss(nq)
nd = 0.5*(nd+1); wt = 0.5*wt

I_M2 = 0.0
for i1 in range(nq):
    u1 = nd[i1]
    for i2 in range(nq):
        u2 = u1*nd[i2]  # 0 < u2 < u1
        jac = u1  # v1=u1, v2=u2/u1; jac = v1
        I_M2 += wt[i1]*wt[i2]*jac * (-4.0)*gb(u1, u2)

print(f"  Integral of (-4)*G over simplex = {I_M2:.8f}")
print(f"  Expected: -4 * int_0^1 du1 int_0^{u1} du2 u2(1-u1) = -4 * 1/12 = {-4.0/12:.8f}")

# For M3: (0,3)(1,2)
# <z^{n1}(u1) dz^{m2}(u2)> = 2*Gd2(u1,u2)*delta^{n1,m2}
# <dz^{m1}(u1) z^{n2}(u2)> = 2*Gd1(u1,u2)*delta^{m1,n2}
# Gd2(u1,u2) = dG/du2 at (u1,u2). For u1 > u2: dG/du2 = 1-u1.
# Gd1(u1,u2) = dG/du1. For u1 > u2: dG/du1 = -u2.
# Index contraction: n1=m2, m1=n2 =>
#   sum_{m2,n2} tr[Omega[n2,m2]@Omega[m2,n2]]
#   = sum_{a,b} tr[Omega[a,b]@Omega[b,a]]
#   = sum_{a,b} tr[-Omega[a,b]@Omega[a,b]]  (since Omega[b,a]=-Omega[a,b])
#   = -tr(Omega_sq)
# Propagator factor: 2*Gd2(u1,u2) * 2*Gd1(u1,u2) = 4*Gd2*Gd1
print("\n--- Matching (0,3)(1,2): z1-dz2, dz1-z2 ---")
print("  Index contraction: sum_{a,b} tr[Omega[a,b]@Omega[b,a]] = -tr(Omega_sq)")
print(f"  = {-tr_Osq:.6f}")

I_M3 = 0.0
for i1 in range(nq):
    u1 = nd[i1]
    for i2 in range(nq):
        u2 = u1*nd[i2]
        jac = u1
        I_M3 += wt[i1]*wt[i2]*jac * 4.0*gd2(u1,u2)*gd1(u1,u2)

print(f"  Integral of 4*Gd2*Gd1 over simplex = {I_M3:.8f}")
# For u1>u2: Gd2 = 1-u1, Gd1 = -u2
# 4*Gd2*Gd1 = 4*(1-u1)*(-u2) = -4*u2*(1-u1)
# Same integrand as M2! So I_M3 should equal I_M2.
print(f"  Expected: same as M2 = {-4.0/12:.8f}")

# Total: tr(a_2^fiber) = (1/4) * [0 + I_M2*tr(Osq) + I_M3*(-tr(Osq))]
# = (1/4) * tr(Osq) * [I_M2 - I_M3]
# If I_M2 == I_M3, then the total is ZERO. That explains the bug!

print(f"\n--- DIAGNOSIS ---")
print(f"  I_M2 = {I_M2:.8f}")
print(f"  I_M3 = {I_M3:.8f}")
print(f"  I_M2 - I_M3 = {I_M2 - I_M3:.8e}")
print(f"  Total = (1/4) * tr(Osq) * (I_M2 - I_M3) = {0.25*tr_Osq*(I_M2-I_M3):.8e}")

# The issue: M2 gives +tr(Osq) and M3 gives -tr(Osq), and the integrals
# are EQUAL, so they cancel! This means the worldline method gives zero
# at second order.
#
# But we KNOW that a_2^{fiber} = (1/12)*Osq is NONZERO. What went wrong?
#
# THE ANSWER: The expansion of the path-ordered exponential P exp(-int V)
# at second order is NOT simply (1/2)*<V*V>. The correct expansion is:
#
# P exp(-S) = Id - <S> + (1/2!)*<P(S^2)> - ...
#
# For the PATH-ORDERED product at second order:
# (1/2!) * int_0^1 int_0^1 P[V(u1)V(u2)] du1 du2
# = int_{u1>u2} V(u1)V(u2) du1 du2  [path ordering removes the 1/2!]
#
# The FIRST ORDER term <S> also contributes! We have:
# <S> = (1/2) int_0^1 Omega_{mn} <z^n(u) dz^m(u)/du> du
# = (1/2) int_0^1 Omega_{mn} * 2*Gd2(u,u) * delta^{nm} du
# = int_0^1 Omega_{nn} * Gd2(u,u) du
# = 0 (since Omega_{nn} = 0, antisymmetric)
#
# So <S> = 0. Good. The second order term is:
# <S^2> = path-ordered integral at 2nd order of the EXPONENTIAL.
# But for the PATH-ORDERED exponential:
# P exp(-S) = sum_n (-1)^n int_{u1>...>un} V(u1)...V(un) du1...dun
#
# The second-order term is:
# int_{u1>u2} V(u1) V(u2) du1 du2
# = int_{u1>u2} (1/4) Omega_{m1n1} Omega_{m2n2}
#   z^{n1}(u1) dz^{m1}/du1 z^{n2}(u2) dz^{m2}/du2  du1 du2
#
# After Wick contraction: this is what we computed above. The result is zero.
#
# BUT WAIT: the heat kernel for D = -nabla^2 is NOT the same as the
# Wilson line expectation! The heat kernel involves the PARTICLE propagator,
# not the straight Wilson line.
#
# The correct worldline representation of the heat kernel of D = -nabla^2 + E
# on a vector bundle is:
#
# K(t,x,y) = int [Dz] exp(-1/(4t) int |dz|^2 ds) * P exp(-int A(z) . dz)
#
# where z(s) for s in [0,1] with z(0)=y, z(1)=x.
# For the DIAGONAL (x=y), z(0)=z(1)=x, so the paths are LOOPS.
#
# The proper normalization gives:
# K(t,x,x) = (4pi t)^{-d/2} * <P exp(-int_0^1 A(x+sqrt(2t)*z(s)) . sqrt(2t)*dz)>
#
# where z(s) is a Brownian bridge (z(0)=z(1)=0) with <z^a(s)z^b(s')>=delta^{ab}*G_B(s,s').
#
# In Fock-Schwinger gauge: A_m(x+y) = (1/2) Omega_{mn} y^n (for constant Omega).
# So: A_m(x+sqrt(2t)*z(s)) = (1/2) Omega_{mn} * sqrt(2t) * z^n(s)
#
# The coupling term becomes:
# int_0^1 A_m * sqrt(2t) * dz^m = int_0^1 (1/2) Omega_{mn} * sqrt(2t)*z^n * sqrt(2t)*dz^m/ds ds
# = t * int_0^1 Omega_{mn} z^n dz^m/ds ds
#
# So the expansion is in powers of t (not 1):
# P exp(-t * int Omega z dz) = Id - t*<int V> + (t^2/...)*<...> + ...
#
# The a_2 coefficient comes from the t^2 term! Not the t^1 term.
# The t^1 term gives a_1.
#
# For a_2, we need the terms of order t^2 in the expansion.
# There are TWO contributions at order t^2:
# (i) (t^2/2) * (int V)^2 from the second-order expansion
# (ii) t * contribution from the O(t) correction to the gauge field
#      (beyond the linear Fock-Schwinger approximation)
#
# Actually, for CONSTANT Omega on flat space:
# The Fock-Schwinger gauge gives A_m exactly. There is no higher-order correction.
# So the heat kernel is EXACTLY:
#
# K(t) = (4pi t)^{-d/2} * <P exp(-t * int_0^1 Omega_{mn} z^n dz^m/ds ds)>
#
# And the a_n coefficient is the coefficient of t^n in the expansion.
# The second-order (t^2) term is:
# (1/2) * t^2 * <(int V)^2>_{path-ordered}
# = t^2 * int_{s1>s2} <V(s1)V(s2)> ds1 ds2
#
# which is EXACTLY what we computed (with the factor (1/2)^2 from the
# two Omega_{mn} z^n dz^m vertices).
#
# So the answer IS zero at second order? That contradicts the known result!
#
# THE RESOLUTION: I think the issue is with the coincident-point
# regularization of <dz^m(s) dz^n(s')> at s=s'. The Wick contraction
# <dz^a(s)/ds dz^b(s')/ds'> has a DELTA FUNCTION SINGULARITY at s=s'.
# This delta function contributes to the diagonal and affects the
# self-contraction terms.
#
# For matching M1: (0,1)(2,3), i.e., contracting within each vertex:
# <z^{n1}(u1) dz^{m1}(u1)> * <z^{n2}(u2) dz^{m2}(u2)>
# The contraction <z^n(u) dz^m(u)/du> at COINCIDENT points is:
# 2 * dG_B(u,v)/dv |_{v=u} = 2 * (0.5 - u)  [regularized]
#
# This gives delta^{n,m}, so Omega[m,n] delta^{n,m} = Omega[m,m] = 0.
# So M1 still gives zero. Good.
#
# The issue must be elsewhere. Let me reconsider.
#
# WAIT: I think the problem is that the worldline path integral gives
# the heat kernel for the SCALAR covariant Laplacian on the bundle,
# NOT the traced heat kernel. The path integral gives the MATRIX-valued
# heat kernel K(t,x,x) (an endomorphism), and then we take tr_spin.
#
# But that's what I computed -- the tr[Omega_sq] is the traced result.
#
# Let me check: is the worldline formula correct for a NON-ABELIAN connection?
#
# For a non-abelian connection, the path-ordered exponential is:
# P exp(-int A.dz) (matrix-valued)
# Its expansion at second order is:
# Id - int V + int_{s1>s2} V(s1)V(s2) + ...
# NOT Id - int V + (1/2)(int V)^2 + ... (that's the unordered version).
#
# The Wick contraction gives:
# <int_{s1>s2} V(s1) V(s2)>
# where V(s) = Omega_{mn} z^n(s) dz^m(s)/ds  (MATRIX-valued)
#
# The expectation is taken over the z fields, but the MATRIX ordering
# V(s1) V(s2) (with s1>s2) is preserved.
#
# After Wick contraction:
# <V(s1) V(s2)> = (1/4) sum_{m1n1 m2n2} Omega[m1,n1] @ Omega[m2,n2]
#                  * <z^{n1}(s1) dz^{m1}(s1) z^{n2}(s2) dz^{m2}(s2)>
#
# The Wick contraction of 4 Gaussian fields:
# <ABCD> = <AB><CD> + <AC><BD> + <AD><BC>
# where A=z^{n1}, B=dz^{m1}, C=z^{n2}, D=dz^{m2}
#
# <AB> = <z^{n1}(s1) dz^{m1}(s1)/ds1> = 2*Gd2(s1,s1)*d^{n1m1}
# <CD> = <z^{n2}(s2) dz^{m2}(s2)/ds2> = 2*Gd2(s2,s2)*d^{n2m2}
# Product: 4*Gd2(s1,s1)*Gd2(s2,s2) * d^{n1m1}*d^{n2m2}
# Omega contraction: Omega[m1,m1]@Omega[m2,m2] = 0
#
# <AC> = <z^{n1}(s1) z^{n2}(s2)> = 2*G(s1,s2)*d^{n1n2}
# <BD> = <dz^{m1}(s1) dz^{m2}(s2)> = 2*Gdd(s1,s2)*d^{m1m2}
# Product: 4*G*Gdd * d^{n1n2}*d^{m1m2}
# Omega contraction: sum_{n,m} Omega[m,n]@Omega[m,n] = Omega_sq
#
# <AD> = <z^{n1}(s1) dz^{m2}(s2)/ds2> = 2*Gd2(s1,s2)*d^{n1m2}
# <BC> = <dz^{m1}(s1)/ds1 z^{n2}(s2)> = 2*Gd1(s1,s2)*d^{m1n2}
# Wait: <dz^{m1}/ds1 z^{n2}(s2)> = d/ds1 <z^{m1}(s1) z^{n2}(s2)>
#      = 2 * dG/ds1(s1,s2) * d^{m1n2} = 2*Gd1(s1,s2)*d^{m1n2}
# Product: 4*Gd2(s1,s2)*Gd1(s1,s2)*d^{n1m2}*d^{m1n2}
# Omega contraction: sum_{n1=m2, m1=n2} Omega[m1,n1]@Omega[m2,n2]
#   = sum_{a,b} Omega[b,a]@Omega[a,b] = -sum_{a,b} Omega[a,b]@Omega[a,b] = -Omega_sq
# Wait no: Omega[m1,n1] with m1=n2, n1=m2. So:
#   Omega[n2,m2] @ Omega[m2,n2]
# = sum_{m2,n2} Omega[n2,m2] @ Omega[m2,n2]
# Let a=n2, b=m2: = sum_{a,b} Omega[a,b]@Omega[b,a]
# Since Omega[b,a] = -Omega[a,b]:
# = -sum_{a,b} Omega[a,b]@Omega[a,b] = -Omega_sq

# So the full second-order term (at order t^2, with the (1/4) prefactor) is:
# tr[a_2^{fiber}] = (1/4) * int_{s1>s2} ds1 ds2
#   [0 + 4*G(s1,s2)*Gdd(s1,s2)*tr(Osq) + 4*Gd2(s1,s2)*Gd1(s1,s2)*(-tr(Osq))]
# = tr(Osq) * int_{s1>s2} [G*Gdd - Gd2*Gd1] ds1 ds2

# For s1 > s2: G = s2(1-s1), Gd1 = -s2, Gd2 = 1-s1, Gdd = -1
# G*Gdd = -s2(1-s1)
# Gd2*Gd1 = (1-s1)*(-s2) = -s2(1-s1)
# Difference: G*Gdd - Gd2*Gd1 = -s2(1-s1) - (-s2(1-s1)) = 0 !!!

print("\n--- KEY FINDING ---")
print("  For s1 > s2:")
print("  G*Gdd = s2(1-s1)*(-1) = -s2(1-s1)")
print("  Gd2*Gd1 = (1-s1)*(-s2) = -s2(1-s1)")
print("  G*Gdd - Gd2*Gd1 = 0")
print()
print("  THE CANCELLATION IS EXACT. The worldline method gives ZERO for a_2^{fiber}.")
print()
print("  This means the worldline PATH-ORDERED EXPONENTIAL approach")
print("  does NOT directly reproduce the Seeley-DeWitt expansion.")
print()
print("  The reason: the heat kernel expansion K = (4pi t)^{-d/2} sum t^n a_n")
print("  involves the FULL heat kernel on the curved bundle, including the")
print("  parallel transport and the Van Vleck-Morette determinant.")
print("  The naive worldline P exp(-int A.dz) misses the CURVATURE CONTRIBUTION")
print("  to the particle propagator on the curved bundle.")
print()
print("  For a FLAT BASE with CONSTANT curvature Omega, the correct formula")
print("  involves the MATRIX-VALUED harmonic oscillator, not just the Wilson line.")
print()
print("  CONCLUSION: The worldline Wick contraction method as implemented")
print("  is INCORRECT for extracting the a_n coefficients. A different approach")
print("  is needed.")
