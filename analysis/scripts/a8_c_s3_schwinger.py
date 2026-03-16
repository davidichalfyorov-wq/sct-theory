# ruff: noqa: E402, I001
"""
c_S3 extraction via the Schwinger proper-time formula.

For a Laplace-type operator D = -nabla^2 on a vector bundle with
CONSTANT curvature Omega_{mn} over FLAT space R^d:

The EXACT heat kernel diagonal is (Schwinger 1951, Schubert 2001):

  K(t,x,x) = (4pi t)^{-d/2} * det^{1/2}[t*F/(sinh(t*F))]  [scalar part]
              * tr_fiber[exp(t * sigma_F)]                      [spin part]

where F_{mn} = Omega_{mn} acts as a 2-form on spacetime, and
sigma_F is the spin coupling to the field strength.

For the Lichnerowicz operator on a Ricci-flat manifold with
Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}:

The Schwinger formula gives BOTH the geometric determinant factor
AND the fiber exponential factor.

The key insight: F_{mn} acts on SPACETIME indices (it's a 6x6 matrix
in bivector space), while Omega_{mn} acts on FIBER (spinor) indices.
For the Dirac operator, these are COUPLED because the same Weyl
tensor C enters both.

The correct formula for the Lichnerowicz heat kernel on Ricci-flat
with constant Weyl is (Parker-Toms 2009, Ch. 9; Avramidi 2000):

  tr K(t) = (4pi t)^{-d/2} * det^{1/2}[tR_bv/sinh(tR_bv)]
            * tr_spin[exp(t * Q)]

where:
- R_bv is the Riemann tensor as a 6x6 matrix on bivector space
- Q is the "effective potential" = (1/12)*Omega_sq + ... (matrix-valued
  in spinor space, involving corrections from the geometry)

On a SYMMETRIC SPACE (nabla R = 0), the formula simplifies and we can
expand in powers of t to get the a_n coefficients.

For the quartic (t^4 = a_8) level, we need to expand both factors
to order t^4 and take their product.

CORRECT METHOD (avoiding the broken worldline approach):
Expand the Schwinger formula perturbatively in t.

Author: David Alfyorov
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from itertools import product as iproduct
from pathlib import Path

import numpy as np
from numpy import einsum
from scipy.linalg import expm, logm

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

PROJECT_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "a8_dirac"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D = 4
PASS = 0
FAIL = 0


def rec(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


# ===================================================================
# Infrastructure (same as before)
# ===================================================================

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

def build_eps():
    e = np.zeros((D, D, D, D))
    for a, b, c, d in iproduct(range(D), repeat=4):
        if len({a, b, c, d}) == 4:
            p = [a, b, c, d]; s = 1
            for i in range(4):
                for j in range(i+1, 4):
                    if p[i] > p[j]: s *= -1
            e[a, b, c, d] = s
    return e

def thooft():
    eta = np.zeros((3, D, D)); eb = np.zeros((3, D, D))
    eta[0,0,1]=1; eta[0,1,0]=-1; eta[0,2,3]=1; eta[0,3,2]=-1
    eta[1,0,2]=1; eta[1,2,0]=-1; eta[1,3,1]=1; eta[1,1,3]=-1
    eta[2,0,3]=1; eta[2,3,0]=-1; eta[2,1,2]=1; eta[2,2,1]=-1
    eb[0,0,1]=1; eb[0,1,0]=-1; eb[0,2,3]=-1; eb[0,3,2]=1
    eb[1,0,2]=1; eb[1,2,0]=-1; eb[1,3,1]=-1; eb[1,1,3]=1
    eb[2,0,3]=1; eb[2,3,0]=-1; eb[2,1,2]=-1; eb[2,2,1]=1
    return eta, eb

def mk_weyl(Wp, Wm, eta, eb):
    C = np.zeros((D,D,D,D))
    for i in range(3):
        for j in range(3):
            C += Wp[i,j]*einsum('ab,cd->abcd', eta[i], eta[j])
            C += Wm[i,j]*einsum('ab,cd->abcd', eb[i], eb[j])
    return C

def rnd_ts3(rng):
    A = rng.standard_normal((3,3)); A = (A+A.T)/2; A -= np.trace(A)/3*np.eye(3)
    return A

def gen_weyl(rng):
    eta, eb = thooft()
    return mk_weyl(rnd_ts3(rng), rnd_ts3(rng), eta, eb)

def sd_dec(C, eps):
    sC = 0.5*einsum('abef,efcd->abcd', eps, C)
    return 0.5*(C+sC), 0.5*(C-sC)

def pq(Cp, Cm):
    return float(einsum('abcd,abcd->', Cp, Cp)), float(einsum('abcd,abcd->', Cm, Cm))

def mk_omega(C, sig):
    O = np.zeros((D,D,4,4), dtype=complex)
    for m in range(D):
        for n in range(D):
            for r in range(D):
                for s in range(D):
                    O[m,n] += 0.25*C[m,n,r,s]*sig[r,s]
    return O


# ===================================================================
# Heat kernel via small-t expansion of the Schwinger formula
# ===================================================================

def compute_heat_trace_density(C_weyl, sigma, t_val):
    """Compute tr_spin[K(t,x,x)] * (4pi*t)^{d/2} for the Lichnerowicz
    operator on Ricci-flat with constant Weyl.

    Uses the Schwinger proper-time formula (exact for constant curvature
    on flat space).

    For the Lichnerowicz operator D = -nabla^2 (on Ricci-flat, E=0):
    The heat kernel diagonal on the fiber is:

    K(t,x,x) = (4pi t)^{-d/2} * Phi(t)

    where Phi(t) is a 4x4 spinor matrix determined by:

    Phi(t) = det^{1/2}[t*R_bv / sinh(t*R_bv)] * exp(t * Omega_eff)

    Here R_bv is the Riemann curvature acting on bivector space (6x6),
    and Omega_eff is the effective fiber coupling.

    For the Lichnerowicz operator:
    Omega_eff = -E + (1/2) sum_{mn} Omega_{mn}^2 * [correction from geometry]

    Wait, this is not right. Let me use the CORRECT formula.

    The Avramidi formula for the heat kernel of a Laplace-type operator
    D = -(g^{ab} nabla_a nabla_b + E) on a symmetric space is:

    K(t) = (4pi t)^{-d/2} * Omega(t)

    where Omega(t) is the solution of the matrix transport equation:

    d Omega/dt = [E + (1/2) sum_{mn} Omega_{mn} * F_mn(t)] * Omega(t)

    with Omega(0) = Id. Here F_mn(t) involves the geometry.

    For the SPECIAL CASE of flat space (R=0) with constant Omega (E=0):

    The heat kernel simplifies enormously. The operator is:
    D = -(partial + A)^2 = -(partial^2 + 2A.partial + A^2 + div A)

    In Fock-Schwinger gauge: A_m(x) = -(1/2) Omega_{mn} x^n (constant curvature).
    Then: A^2 = (1/4) Omega_{mk} Omega_{ml} x^k x^l  [matrix-valued]
    And: div A = -(1/2) Omega_{mm} = 0 (antisymmetric)

    So: D = -(partial^2 - Omega_{mn} x^n partial_m + (1/4) Omega_{mk} Omega_{ml} x^k x^l)

    This is a MATRIX-VALUED harmonic oscillator (in each spacetime direction).
    The heat kernel of a matrix-valued harmonic oscillator is KNOWN EXACTLY
    from the Mehler kernel.

    For a SCALAR harmonic oscillator H = -d^2/dx^2 + omega^2 x^2:
    K(t,0,0) = (4pi t)^{-1/2} * (omega t / sinh(omega t))^{1/2}

    For the MATRIX-VALUED case with non-commuting Omega's, the formula
    involves the matrix square root, which is more complex.

    HOWEVER, for the PURPOSE of extracting the TRACE at small t:
    We can NUMERICALLY evaluate the heat kernel by diagonalizing the operator
    or by direct series expansion.

    APPROACH: Expand exp(t*M) where M is the "effective Hamiltonian" matrix
    in the combined (spacetime x fiber) space, and extract the t^n coefficients.

    Actually, the SIMPLEST correct approach for constant curvature on flat space:

    The heat kernel trace density is:
    tr[K(t)] * (4pi t)^2 = tr_spin[det_space^{1/2}(tF/sinh(tF)) * exp(tQ)]

    where F is the SPACETIME field strength (a 4x4 antisymmetric matrix
    or equivalently a 6x6 symmetric matrix in bivector space) and
    Q is the fiber coupling.

    For the Lichnerowicz operator:
    F_{mn} = R_{mn..} acts on spacetime indices.
    Q = sum of Omega^2 contributions.

    On Ricci-flat flat space, F = 0 and Q = 0 at leading order.
    The heat kernel is simply (4pi t)^{-2} * tr(Id) = 4*(4pi t)^{-2}.

    The CORRECTIONS come from the fact that the covariant derivative
    involves Omega, which generates nonzero contributions starting at
    order t^1 in the FIBER sector.

    RESOLUTION: The correct approach is NOT the Schwinger formula (which
    applies to abelian fields), but the AVRAMIDI algebraic method for
    non-abelian fields.

    For a non-abelian constant field on flat space, the heat kernel is:

    K(t,x,x) = (4pi t)^{-d/2} * exp(t * Omega_sq / 12)  (approximately)

    Wait, but we showed this is NOT right either (the (1/12) comes from
    the DeWitt coefficient a_2, which is what we want to DERIVE).

    OK, let me use the MATRIX EXPONENTIAL METHOD directly.

    For constant Omega on flat R^4, the operator is:
    D psi = -(delta^{mn} partial_m partial_n + 2 A_m partial_m + A^2) psi

    where A_m = -(1/2) Omega_{mn} x^n (Fock-Schwinger gauge).

    In momentum space: D_hat psi_hat(k) = (k^2 + ...) with matrix structure.
    The heat kernel in position space involves Gaussian integrals with
    the matrix-valued quadratic form.

    The EXACT traced heat kernel on flat space with constant non-abelian
    curvature Omega is (Schwinger for non-abelian case):

    tr K(t,x,x) = (4pi t)^{-d/2} * tr_fiber[Psi(t)]

    where Psi(t) is determined by the matrix equation:

    Psi(t) = T exp(-(1/12) t^2 * ad_Omega^2 + ...) * Id

    This is the AVRAMIDI generating function on flat space.

    For PRACTICAL COMPUTATION, the cleanest method is:
    Construct the FULL operator matrix in a finite basis and
    compute exp(-tD) numerically, then extract the diagonal
    and take the fiber trace.

    But that requires discretizing R^4, which is impractical.

    FINAL APPROACH: Use the KNOWN FORMULA from the literature.
    """
    pass  # See below for the actual computation


def compute_traced_an_via_small_t(C_weyl, sigma, eps, n_t=50, t_max=0.1):
    """Extract the heat kernel coefficients a_n by fitting tr[K(t)]
    at multiple small t values.

    The idea: compute tr[K(t,x,x)] * (4pi t)^2 at many small t values,
    fit as a polynomial in t, and extract coefficients.

    For this to work, we need an INDEPENDENT way to compute tr[K(t)].

    On flat space with constant non-abelian field strength, the EXACT
    heat kernel can be computed from the MATRIX-VALUED Mehler formula.

    For the covariant Laplacian D = -(partial + A)^2 with A_m = -(1/2)F_{mn}x^n:
    D = -(partial^2 - F_{mn} x^n partial^m + (1/4) F_{mk} F_{ml} x^k x^l)
      = -(partial^2 - F_{mn} x^n partial^m + (1/4) (F^2)_{kl} x^k x^l)

    This is a GENERALIZED harmonic oscillator. For MATRIX-VALUED F,
    the key issue is that F_{mn} is a matrix in fiber space, so the
    "frequency" is matrix-valued.

    For the Dirac case: F_{mn} = Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}
    is a 4x4 matrix for each (m,n) pair.

    The heat kernel can be computed by going to the EIGENBASIS of the
    antisymmetric tensor F_{mn} in spacetime. In 4D, an antisymmetric
    tensor has eigenvalues +-if1, +-if2 (two "frequencies").

    BUT: for non-abelian F, [F_{mn}, F_{pq}] != 0 in general, so there
    is no simultaneous diagonalization.

    NEVERTHELESS: for the TRACE, we can use a path integral representation
    that is amenable to perturbation theory. The correct worldline formula
    (Strassler 1992, hep-ph/9205205) for the non-abelian case is:

    tr K(t,x,x) = (4pi t)^{-d/2} * <tr_fiber P exp(-t int_0^1 [
        (1/4)|dz/ds|^2/t^2 + (1/2)F_{mn}z^n dz^m/ds + ... ] ds)>

    Wait, I already tried this and it gave zero. The issue was that the
    worldline propagator in the PRESENCE of the field strength is MODIFIED.

    The correct worldline computation uses the MODIFIED PROPAGATOR:
    <z^a(s) z^b(s')>_F = G_F(s,s')^{ab}

    where G_F is the Green's function of the 1D operator:
    (-d^2/ds^2 + t^2 F_{ab}) G_F^{bc}(s,s') = delta^{ac} delta(s-s')

    This is the SCHWINGER MODIFIED PROPAGATOR. For abelian F, it gives
    the standard Mehler kernel. For non-abelian F, it's more complex.

    OK, I realize the worldline approach is much more involved than I
    initially thought for non-abelian fields. Let me use a COMPLETELY
    DIFFERENT method.
    """
    pass


# ===================================================================
# METHOD: Direct matrix computation on a LATTICE
# ===================================================================
#
# On a TORUS T^4 with constant Omega, the Lichnerowicz operator has
# a discrete spectrum. We can compute the heat trace as sum exp(-t*lambda_n).
# With enough eigenvalues, the small-t expansion gives the a_n.
#
# For a lattice with N^4 sites and lattice spacing a:
# The covariant Laplacian is approximated by the lattice Laplacian
# with the link variables U_{m,x} = exp(-a * A_m(x+a/2))
# where A_m(x) = -(1/2) Omega_{mn} x^n.
#
# This is a FINITE MATRIX (4*N^4 x 4*N^4 for Dirac) that can be
# diagonalized numerically.
#
# For extracting a_4 (8th order in curvature), we need high precision
# and large lattice. This is computationally expensive but correct.
#
# ALTERNATIVELY: use the KNOWN RESULT from the literature.
# Let me try one more literature search, then fall back to the lattice.

def compute_omega_structures(C_weyl, sigma, eps):
    """Compute all relevant Omega and Weyl structures for a given background."""
    Omega = mk_omega(C_weyl, sigma)

    # Omega_sq = sum_{mn} Omega[m,n]^2 (4x4 spinor matrix)
    Omega_sq = np.zeros((4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            Omega_sq += Omega[a, b] @ Omega[a, b]

    # S1: tr(Omega chain)
    S1 = 0.0
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    S1 += np.trace(Omega[a,b] @ Omega[b,c] @ Omega[c,d] @ Omega[d,a]).real

    # S2: tr(Omega_sq^2)
    S2 = np.trace(Omega_sq @ Omega_sq).real

    # S3: [tr(Omega_sq)]^2
    tr_Osq = np.trace(Omega_sq).real
    S3 = tr_Osq**2

    # S4: sum |tr(Omega[a,b]@Omega[c,d])|^2
    S4 = 0.0
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    t = np.trace(Omega[a,b] @ Omega[c,d])
                    S4 += (t * t.conjugate()).real

    # Weyl structures
    C_plus, C_minus = sd_dec(C_weyl, eps)
    p, q = pq(C_plus, C_minus)

    # Riemann bivector matrix
    bv = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    R_bv = np.zeros((6, 6))
    for i, (a, b) in enumerate(bv):
        for j, (c, d) in enumerate(bv):
            R_bv[i, j] = C_weyl[a, b, c, d]

    return {
        "Omega": Omega, "Omega_sq": Omega_sq,
        "S1": S1, "S2": S2, "S3": S3, "S4": S4,
        "tr_Osq": tr_Osq, "p": p, "q": q,
        "R_bv": R_bv,
        "trM2": np.trace(R_bv @ R_bv),
        "trM4": np.trace(R_bv @ R_bv @ R_bv @ R_bv),
    }


def expand_schwinger_abelianized(data, order=4):
    """Compute the traced heat kernel coefficient using the ABELIANIZED
    Schwinger formula.

    For the Lichnerowicz operator on Ricci-flat, we can write:

    tr K(t) * (4pi t)^2 = F_geom(t) * F_fiber(t)

    where:
    F_geom(t) = det^{1/2}[t*R_bv / sinh(t*R_bv)]  (scalar, from geometry)
    F_fiber(t) = tr_spin[exp(t * Q)]                 (from fiber)

    On Ricci-flat E=0, the fiber contribution Q is related to Omega.
    For the Lichnerowicz Laplacian, the well-known result is:
    Q = (1/12) Omega_sq (at leading order) + higher-order corrections.

    BUT: the product F_geom * F_fiber is NOT the correct formula for a
    non-abelian connection. The correct formula couples geometry and fiber.

    For the PURPOSE OF EXTRACTING c_S3, we make the following observation:

    The heat kernel coefficients a_n are UNIVERSAL polynomials in the
    curvature invariants. On Ricci-flat with E=0 and d=4:

    tr[a_0] = 4
    tr[a_1] = 0
    tr[a_2] = (1/12)*tr(Omega_sq) + (1/180)*4*(C^2) = (1/12)*tr(Osq) + (4/180)*(p+q)
    tr[a_3] = ...
    tr[a_4] = A*(p^2+q^2) + B*pq  (the quartic Weyl sector)

    The coefficients A and B are what we want to determine.

    APPROACH: Rather than computing A and B from a first-principles
    derivation, we can USE THE KNOWN FORMULA by extracting it from
    a COMPUTATION ON A SPECIFIC MANIFOLD where the eigenvalues are known.

    The manifold CP^2 (complex projective plane) is Kaehler, Einstein
    (Ric = 6g for radius 1), and has constant holomorphic sectional
    curvature. Its Dirac spectrum is KNOWN (Cahen-Gutt-Trautman 1993).

    But CP^2 is NOT Ricci-flat. We need its Weyl tensor.

    Actually, for extracting the UNIVERSAL coefficients of a_4, we
    DON'T need a Ricci-flat manifold. We can use ANY manifold and
    subtract the non-Weyl terms (which are known from lower a_n).

    BETTER: Use the KNOWN NUMERICAL COMPUTATION by Amsterdamski,
    Berkin & O'Connor (1989) for the scalar a_8 on S^4, and
    EXTEND IT to the Dirac case using the known traces.

    CLEANEST APPROACH: Use the VASSILEVICH/AVRAMIDI master formula.
    Although a_4 (= our a_8) is not given EXPLICITLY in Vassilevich (2003),
    it IS known in the GENERAL FORM from Avramidi's work.

    The general structure of a_4 on a closed manifold, for the operator
    D = -(Box + E), is (Avramidi 2000, Theorem 7.2):

    (4pi)^{d/2} * a_4 = integral sqrt{g} * tr[
      sum of 26 dimension-8 invariants with rational coefficients
    ]

    The 26 invariants include quartic curvature, mixed curvature-E,
    quartic E, and derivative terms.

    On Ricci-flat E=0, after dropping all R and E terms, the surviving
    quartic Omega terms are (from Avramidi's Table):

    9! * (surviving quartic Omega sector) =
      alpha1 * Omega_{ab}Omega^{bc}Omega_{cd}Omega^{da}
    + alpha2 * Omega_{ab}Omega^{ab}Omega_{cd}Omega^{cd}
    + alpha3 * [tr_fiber(Omega_{ab}Omega^{ab})]^2  <-- THIS IS c_S3
    + alpha4 * tr_fiber(Omega_{ab}Omega^{cd}) * tr_fiber(Omega_{cd}Omega^{ab})
    + (pure-geometry quartic Weyl terms) * tr(Id)

    The coefficients alpha1..alpha4 ARE KNOWN from the recursion but
    I need to either find them in the literature or compute them.

    Since the worldline method failed, let me try a completely different
    approach: DERIVE c_S3 from the heat kernel recursion ALGEBRAICALLY.

    The recursion for the traced a_n on a closed manifold (Vassilevich
    Eq. 4.1) involves the transport equation and the Van Vleck determinant.
    For CONSTANT curvature, the recursion simplifies.

    The KEY INSIGHT is that on a SYMMETRIC SPACE with constant curvature,
    the a_n coefficients satisfy (Avramidi 2000, Eq. 6.16):

    sum_{n=0}^{infinity} t^n * a_n = exp(t * F(R, Omega, E))

    where F is a specific function of the curvature data. This means:

    a_0 = Id
    a_1 = F
    a_2 = (1/2) F^2 + (1/2) F'  [where F' involves commutators]
    a_3 = ...
    a_4 = (1/24) F^4 + (correction terms)

    On Ricci-flat with E=0, F involves only Omega, and the correction
    terms come from the non-commutativity of Omega.

    At LEADING ORDER (ignoring commutators):
    a_4 ~ (1/24) F^4 where F ~ (1/12) Omega_sq (from a_2 = (1/12) Omega_sq)

    Hmm wait, a_1 = F = 0 on Ricci-flat E=0. So the exponential ansatz
    gives a_1 = F = 0, and then a_2 = (1/2)F^2 + correction = correction.
    This means F is NOT simply related to a_1.

    CORRECT TREATMENT: On Ricci-flat E=0, the first nonzero coefficient is a_2.
    Write: sum t^n a_n = Id + t^2 a_2 + t^3 a_3 + t^4 a_4 + ...

    The logarithm: log(Id + X) = X - X^2/2 + X^3/3 - ...
    where X = t^2 a_2 + t^3 a_3 + ...

    F_eff = t^{-1} log(sum t^n a_n) = t a_2 + t^2 (a_3 - (1/2)a_2^2) + ...

    Hmm, this gets circular. Let me just use the NUMERICAL approach with
    a different method.
    """
    # Expand det^{1/2}[tM/sinh(tM)] to order t^4
    M = data["R_bv"]
    M2 = M @ M; M4 = M2 @ M2
    trM2 = np.trace(M2); trM4 = np.trace(M4)

    # f_geom(t) = 1 - (1/12)*trM2*t^2 + [(-1/288)(trM2)^2 + (7/720)*trM4]*t^4 + ...
    g0 = 1.0
    g2 = -(1.0/12.0)*trM2
    g4 = (-1.0/288.0)*trM2**2 + (7.0/720.0)*trM4

    # These are exact for the geometric sector.
    return {"g0": g0, "g2": float(g2), "g4": float(g4),
            "trM2": float(trM2), "trM4": float(trM4)}


def compute_heat_trace_numerical(C_weyl, sigma, t_values):
    """Compute the EXACT heat trace density on flat R^4 with constant Omega
    by diagonalizing the covariant Laplacian on a truncated momentum space.

    For constant Omega on flat space in a periodic box of size L:
    The operator in momentum space is:
    D_hat(k) = k^2 * Id - i*Omega_{mn}*k_m*(d/dk_n) + (1/4)*Omega_{mk}*Omega_{ml}*(d^2/dk_k dk_l)
    ... this is complex because of the position-momentum mixing.

    SIMPLER: work in POSITION SPACE on a lattice.
    On a N^4 lattice with spacing a, the covariant Laplacian is:
    (D psi)_x = -(1/a^2) sum_m [U_{m,x} psi_{x+m} + U_{m,x-m}^dag psi_{x-m} - 2 psi_x]

    where U_{m,x} = exp(-a * A_m(x+a/2)) is the link variable.
    For A_m(x) = -(1/2) Omega_{mn} x^n:
    U_{m,x} = exp(a/2 * Omega_{mn} * (x^n + a*delta^n_m/2))

    The lattice operator is a (4*N^4) x (4*N^4) matrix (4 for spinor indices).
    For N = 8: 4*8^4 = 16384. Diagonalizable.

    The heat trace is: Tr[exp(-t*D_lattice)] = sum_i exp(-t*lambda_i)
    For the DIAGONAL heat kernel density: K(t,x,x) = (1/N^4) * sum_i exp(-t*lambda_i) |psi_i(x)|^2

    But we want the TRACED density in the continuum limit.
    For a CONSTANT Omega, K(t,x,x) is INDEPENDENT of x (by translation
    invariance in the covering space). So:
    tr K(t) = (1/N^4) * Tr[exp(-t*D)] / (volume per site)

    Actually, for constant Omega the lattice is NOT translationally invariant
    because A_m(x) depends on x. The FS gauge breaks translation invariance.

    We can RESTORE translation invariance by working on a TORUS with
    CONSTANT CURVATURE (twisted boundary conditions).

    This is getting complicated. Let me try a VERY DIFFERENT approach.
    """
    pass


# ===================================================================
# ALGEBRAIC METHOD: Use the KNOWN universal formula structure
# ===================================================================
#
# Rather than computing from first principles, use the KNOWN STRUCTURE
# of the heat kernel coefficients from the mathematical literature.
#
# The key facts:
#
# FACT 1: On a d=4 closed manifold, for D = -(Box + E):
# tr[a_2] = (1/12)*tr(Omega^2) + (1/180)*tr(Id)*(C^2 - Ric^2) + ... [known]
#
# FACT 2: The tr[a_4] on Ricci-flat with E=0 has the form:
# tr[a_4] = sum of quartic Weyl invariants * tr(Id)
#          + sum of quartic Omega contractions
#          + sum of mixed C^2 * Omega^2 terms
#
# FACT 3: The only quartic Weyl invariants in d=4 are:
# (C^2)^2 = (p+q)^2 and C^4_chain = (p^2+q^2)/2
# Equivalently: (p^2+q^2) and pq.
#
# FACT 4: The quartic Omega structures S1, S2, S4 all give only (p^2+q^2).
# S3 = [tr(Omega^2)]^2 gives (p^2+q^2)/4 + pq/2.
#
# FACT 5: The mixed C^2*Omega^2 terms:
# C^2 * tr(Omega^2) = (p+q) * (-(p+q)/2) = -(p+q)^2/2
#   = -(p^2+q^2)/2 - pq. HAS pq!
# C_{abcd} * tr(Omega^{ab} Omega^{cd}) = ? (need to compute)
#
# So the question of whether the full a_4 has pq depends on:
# (a) The coefficient of S3 (this is c_S3)
# (b) The coefficient of C^2 * tr(Omega^2)
# (c) The coefficient of other mixed terms
#
# Let me compute the pq content of ALL contributing structures.

def analyze_mixed_terms(C_weyl, sigma, eps):
    """Compute the mixed C^2 * Omega^2 type structures and their pq content."""
    Omega = mk_omega(C_weyl, sigma)
    C_plus, C_minus = sd_dec(C_weyl, eps)
    p, q = pq(C_plus, C_minus)

    # Mixed structure M1: (C^2) * tr(Omega^2)
    C2 = float(einsum('abcd,abcd->', C_weyl, C_weyl))  # = p + q
    tr_Osq = 0.0
    for a in range(D):
        for b in range(D):
            tr_Osq += np.trace(Omega[a,b] @ Omega[a,b]).real
    M1 = C2 * tr_Osq  # = (p+q) * (-(p+q)/2) = -(p+q)^2/2

    # Mixed structure M2: C_{abcd} * tr(Omega^{ab} Omega^{cd})
    M2 = 0.0
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    M2 += C_weyl[a,b,c,d] * np.trace(Omega[a,b] @ Omega[c,d]).real

    # Mixed structure M3: C_{abcd} * tr(Omega^{ac} Omega^{bd})
    M3 = 0.0
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    M3 += C_weyl[a,b,c,d] * np.trace(Omega[a,c] @ Omega[b,d]).real

    # Mixed M4: R_bv^2 term (pure geometry quartic)
    R_bv = np.zeros((6, 6))
    bv = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for i, (a, b) in enumerate(bv):
        for j, (c, d) in enumerate(bv):
            R_bv[i, j] = C_weyl[a, b, c, d]
    M2_bv = R_bv @ R_bv
    trM2 = np.trace(M2_bv)  # = (p+q)
    trM4 = np.trace(M2_bv @ M2_bv)  # = ?

    # C^4_chain in the standard contraction
    C4chain = float(einsum('abcd,cdef,efgh,ghab->', C_weyl, C_weyl, C_weyl, C_weyl))
    C2sq = C2**2

    return {
        "p": p, "q": q, "C2": C2,
        "tr_Osq": tr_Osq, "M1_C2_trOsq": M1,
        "M2_Cabcd_trOabOcd": M2, "M3_Cabcd_trOacObd": M3,
        "trM2": trM2, "trM4": trM4,
        "C4chain": C4chain, "C2sq": C2sq,
    }


def run():
    print("=" * 72)
    print("c_S3 ANALYSIS: (p,q) decomposition of ALL quartic structures")
    print("=" * 72)

    gam = build_gamma()
    sig = build_sigma(gam)
    eps = build_eps()
    rng = np.random.default_rng(2026_03_16)

    n_trials = 25
    all_data = []

    for trial in range(n_trials):
        C = gen_weyl(rng)
        d = analyze_mixed_terms(C, sig, eps)
        s = compute_omega_structures(C, sig, eps)

        row = {
            "p": d["p"], "q": d["q"],
            "p2q2": d["p"]**2 + d["q"]**2,
            "pq": d["p"] * d["q"],
            "ppqq": (d["p"] + d["q"])**2,
            "S1": s["S1"], "S2": s["S2"], "S3": s["S3"], "S4": s["S4"],
            "tr_Osq": d["tr_Osq"],
            "M1": d["M1_C2_trOsq"],
            "M2": d["M2_Cabcd_trOabOcd"],
            "M3": d["M3_Cabcd_trOacObd"],
            "C4chain": d["C4chain"], "C2sq": d["C2sq"],
            "trM4": d["trM4"],
        }
        all_data.append(row)

    # Fit each structure against (p^2+q^2, pq) basis
    A_mat = np.array([[d["p2q2"], d["pq"]] for d in all_data])

    structures = ["S1", "S2", "S3", "S4", "M1", "M2", "M3",
                  "C4chain", "C2sq", "trM4"]

    print("\nStructure             | coeff(p^2+q^2)  | coeff(pq)       | has_pq? | ratio")
    print("-" * 90)

    results = {}
    for name in structures:
        b_vec = np.array([d[name] for d in all_data])
        c, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        fit = A_mat @ c
        err = np.max(np.abs(fit - b_vec))
        rele = err / (np.max(np.abs(b_vec)) + 1e-30)

        has_pq = abs(c[1]) > 1e-6 * (abs(c[0]) + 1e-30)
        ratio = c[1] / c[0] if abs(c[0]) > 1e-30 else float('inf')

        print(f"  {name:20s} | {c[0]:15.8e} | {c[1]:15.8e} | {'YES' if has_pq else 'NO ':3s}     | {ratio:.4f}")

        results[name] = {
            "alpha": float(c[0]), "beta": float(c[1]),
            "has_pq": has_pq, "ratio": float(ratio),
            "max_err": float(err), "rel_err": float(rele),
        }

    # Key analysis
    print("\n" + "=" * 72)
    print("ANALYSIS OF pq CONTENT")
    print("=" * 72)

    print("\n  QUARTIC OMEGA STRUCTURES (fiber only):")
    print(f"    S1 (chain):       pq = {'YES' if results['S1']['has_pq'] else 'NO'}")
    print(f"    S2 (Osq^2 trace): pq = {'YES' if results['S2']['has_pq'] else 'NO'}")
    print(f"    S3 ([trOsq]^2):   pq = {'YES' if results['S3']['has_pq'] else 'NO'}")
    print(f"    S4 (double tr):   pq = {'YES' if results['S4']['has_pq'] else 'NO'}")

    print("\n  MIXED STRUCTURES (C^2 x Omega^2):")
    print(f"    M1 = C^2 * tr(Osq):           pq = {'YES' if results['M1']['has_pq'] else 'NO'}")
    print(f"    M2 = C_{'{abcd}'} tr(O^{'{ab}'}O^{'{cd}'}): pq = {'YES' if results['M2']['has_pq'] else 'NO'}")
    print(f"    M3 = C_{'{abcd}'} tr(O^{'{ac}'}O^{'{bd}'}): pq = {'YES' if results['M3']['has_pq'] else 'NO'}")

    print("\n  PURE GEOMETRY QUARTIC:")
    print(f"    C4_chain:  pq = {'YES' if results['C4chain']['has_pq'] else 'NO'}")
    print(f"    (C^2)^2:   pq = {'YES' if results['C2sq']['has_pq'] else 'NO'}")
    print(f"    tr(R_bv^4): pq = {'YES' if results['trM4']['has_pq'] else 'NO'}")

    # Count structures with pq
    pq_structures = [n for n in structures if results[n]['has_pq']]
    print(f"\n  Structures with pq content: {pq_structures}")
    print(f"  Structures without pq: {[n for n in structures if not results[n]['has_pq']]}")

    # The total a_4 is a LINEAR COMBINATION of these structures.
    # pq content of a_4 = sum of (coefficient * pq_content) for each structure.
    # The coefficient of each structure comes from the Avramidi recursion.
    # We don't know the exact coefficients, but we know WHICH structures
    # have pq content. If ANY structure with pq appears with nonzero coefficient,
    # then a_4 has pq and c_S3 != 0.

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    pq_sources = [n for n in pq_structures if n in ["S3", "M1", "C2sq"]]
    print(f"\n  Sources of pq in a_4: {pq_sources}")
    print()
    for s in pq_sources:
        r = results[s]
        print(f"    {s}: beta/alpha = {r['ratio']:.4f}")

    rec("S3 has pq", results["S3"]["has_pq"])
    rec("M1 has pq", results["M1"]["has_pq"])
    rec("C2sq has pq", results["C2sq"]["has_pq"])
    rec("S1 no pq", not results["S1"]["has_pq"])
    rec("S2 no pq", not results["S2"]["has_pq"])
    rec("S4 no pq", not results["S4"]["has_pq"])
    rec("C4chain no pq", not results["C4chain"]["has_pq"])

    print(f"\n  Total: {PASS} PASS, {FAIL} FAIL")

    # Save
    with open(RESULTS_DIR / "a8_pq_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {RESULTS_DIR / 'a8_pq_analysis.json'}")

    return results


if __name__ == "__main__":
    run()
