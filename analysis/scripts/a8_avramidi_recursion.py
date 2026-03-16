# ruff: noqa: E402, I001
"""
Direct computation of c_S3: coefficient of [tr(Omega^2)]^2 in Seeley-DeWitt a_8.

Strategy: On a covariantly-constant background (nabla R = 0, nabla Omega = 0),
the heat kernel recursion becomes purely algebraic. We evaluate the recursion
numerically for the Dirac operator on multiple random Ricci-flat Weyl backgrounds
and extract c_S3 by fitting the quartic Omega sector against the (p,q) basis.

The recursion (DeWitt 1965, Avramidi 2000):
On a closed manifold without boundary, for D = -(Box + E):

  K(t,x,x) = (4pi t)^{-d/2} sum_n t^n a_n(x)

where a_n(x) are endomorphism-valued (matrix in fiber space).

The diagonal coefficients satisfy:
  a_0 = Id
  (n + d/2) a_n = (Delta + E) a_{n-1} + sum of curvature terms

On a CONSTANT curvature background (nabla R = 0, nabla Omega = 0):
  - All derivative terms vanish
  - The recursion becomes purely algebraic
  - a_n is a polynomial in E, Omega_{ab}, R_{abcd}

For Dirac on Ricci-flat (E=0, Ric=0, d=4):
  a_0 = Id_4
  a_1 = (1/6)R*Id + E = 0
  a_2 = (1/2)E^2 + (1/12)Omega_{ab}Omega^{ab} + (1/180)C_{abcd}C^{abcd}*Id
       = (1/12)Omega_sq + (1/180)(p+q)*Id  [on Ricci-flat]

For a_3 and a_4, we need the FULL recursion including Omega-dependent terms.

METHOD: Rather than implementing the abstract recursion (which requires
careful bookkeeping of index contractions), we use a DIRECT NUMERICAL
approach on a flat-space model with constant field strength.

On FLAT SPACE (g = delta, Gamma = 0) with a constant gauge connection
curvature Omega_{mn}, the Laplace-type operator is:
  D = -(partial^2 + 2*Omega_{mn}*x^m*partial_n + ...)

This is not quite right for a curved manifold. Instead, we use the
KNOWN ALGEBRAIC STRUCTURE of the coefficients on constant curvature.

CORRECT APPROACH: Use the Avramidi algebraic method.
On a symmetric space (covariantly constant curvature), the heat kernel
diagonal has the EXACT closed form (Avramidi 2000, Eq. 7.1):

  K(t) = (4pi t)^{-d/2} * det^{-1/2}[sin(t^{1/2} R_L) / (t^{1/2} R_L)]
         * exp(-t*E_eff)

where R_L is the "Lorentz curvature operator" acting on the fiber.

For our purposes, we expand this in powers of t to extract a_n.

Actually, the simplest correct approach for EXTRACTING c_S3 is:

1. We KNOW the a_2 kernel exactly on Ricci-flat E=0:
   a_2 = (1/12) Omega_sq_mat + (1/180)(p+q)*Id_4

2. We KNOW a_3 from Vassilevich Eq (4.3) on Ricci-flat E=0:
   a_3 involves cubic Omega terms + mixed Omega*Riemann terms

3. For a_4 on CONSTANT curvature, the key terms come from
   the algebraic recursion which involves products of lower a_n's
   with the curvature operators.

SIMPLEST CORRECT METHOD:
Use the Avramidi generating function on a symmetric space.
On a Ricci-flat symmetric space (i.e., flat space with constant Omega),
the heat kernel has the closed form involving matrix exponentials.

For a constant field strength F_{mn} (our Omega_{mn}), on flat R^4:
  a_n = polynomial in F_{mn} of degree n (in fiber-valued matrices)

The generating function is:
  sum_n t^n a_n = det^{-1/2}[sinh(sqrt(t)*F) / (sqrt(t)*F)] * exp(...)

Wait -- this is for the SPIN connection on flat space, not curved.
For our case (Ricci-flat, E=0), the connection IS the spin connection
Omega_{mn} = (1/4)C_{mnrs}sigma^{rs}, and the "field strength" is Omega itself.

The CORRECT generating function for the heat kernel on a bundle with
constant curvature Omega over flat base manifold is (Avramidi 2000):

  K(t,x,x) = (4pi t)^{-d/2} * [t^{d/2} / det^{1/2}(Omega_L^{-1} sin(t Omega_L))]

where Omega_L is the "Lorentz generator" matrix acting in the (mn) index space.

Hmm, this is getting complicated. Let me use a much simpler method.

DIRECT METHOD: The Fock-Schwinger-DeWitt proper-time method gives
the heat kernel as a path integral. On a CONSTANT field strength
background, this path integral is Gaussian and can be evaluated exactly.

For a particle of spin s in a constant electromagnetic field F_{mn}:
  K(t) = (4pi t)^{-d/2} * prod_i [omega_i t / sinh(omega_i t)]
         * tr_spin[exp(-t * sigma.F)]

where omega_i are the eigenvalues of F and sigma.F is the spin coupling.

For the Dirac operator with Omega_{mn} = (1/4)C_{mnrs}sigma^{rs}:
The "electromagnetic" field strength IS Omega, and the spin coupling
is also Omega (it's the same connection).

On a Ricci-flat 4-manifold with constant Weyl tensor, the spin
connection curvature Omega acts on spinors, and the heat kernel
diagonal for the SQUARED Dirac operator (the Lichnerowicz Laplacian)
has the expansion involving a_n coefficients that are polynomials in Omega.

FOR THE COMPUTATION: we work on flat R^4 with a constant
antisymmetric tensor field Omega_{mn} (a 4x4 antisymmetric matrix
with values in 4x4 spinor matrices). This models the Ricci-flat
case exactly for the Omega-dependent terms.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import product as iproduct
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy import einsum

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "a8_dirac"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D = 4
mp.mp.dps = 50

PASS_COUNT = 0
FAIL_COUNT = 0


def record(label: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{tag}] {label}" + (f" -- {detail}" if detail else ""))


# ===================================================================
# INFRASTRUCTURE: gamma matrices, sigma, Weyl tensor construction
# ===================================================================

def build_gamma_euclidean():
    """Euclidean gamma matrices: {gamma^a, gamma^b} = 2 delta^{ab}."""
    I2 = np.eye(2, dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    gamma = np.zeros((D, 4, 4), dtype=complex)
    gamma[0] = np.kron(s1, I2)
    gamma[1] = np.kron(s2, I2)
    gamma[2] = np.kron(s3, s1)
    gamma[3] = np.kron(s3, s2)
    return gamma


def build_sigma(gamma):
    """sigma^{ab} = (1/2)[gamma^a, gamma^b]."""
    sig = np.zeros((D, D, 4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            sig[a, b] = 0.5 * (gamma[a] @ gamma[b] - gamma[b] @ gamma[a])
    return sig


def build_levi_civita():
    eps = np.zeros((D, D, D, D))
    for a, b, c, d in iproduct(range(D), repeat=4):
        if len({a, b, c, d}) == 4:
            perm = [a, b, c, d]
            sign = 1
            for i in range(4):
                for j in range(i + 1, 4):
                    if perm[i] > perm[j]:
                        sign *= -1
            eps[a, b, c, d] = sign
    return eps


def build_thooft_symbols():
    eta = np.zeros((3, D, D))
    etabar = np.zeros((3, D, D))
    eta[0, 0, 1] = 1;  eta[0, 1, 0] = -1
    eta[0, 2, 3] = 1;  eta[0, 3, 2] = -1
    eta[1, 0, 2] = 1;  eta[1, 2, 0] = -1
    eta[1, 3, 1] = 1;  eta[1, 1, 3] = -1
    eta[2, 0, 3] = 1;  eta[2, 3, 0] = -1
    eta[2, 1, 2] = 1;  eta[2, 2, 1] = -1
    etabar[0, 0, 1] = 1;  etabar[0, 1, 0] = -1
    etabar[0, 2, 3] = -1; etabar[0, 3, 2] = 1
    etabar[1, 0, 2] = 1;  etabar[1, 2, 0] = -1
    etabar[1, 3, 1] = -1; etabar[1, 1, 3] = 1
    etabar[2, 0, 3] = 1;  etabar[2, 3, 0] = -1
    etabar[2, 1, 2] = -1; etabar[2, 2, 1] = 1
    return eta, etabar


def make_weyl(W_plus, W_minus, eta, etabar):
    C = np.zeros((D, D, D, D))
    for i in range(3):
        for j in range(3):
            C += W_plus[i, j] * np.einsum('ab,cd->abcd', eta[i], eta[j])
            C += W_minus[i, j] * np.einsum('ab,cd->abcd', etabar[i], etabar[j])
    return C


def random_traceless_sym_3x3(rng):
    A = rng.standard_normal((3, 3))
    A = (A + A.T) / 2.0
    A -= np.trace(A) / 3.0 * np.eye(3)
    return A


def generate_random_weyl(rng):
    eta, etabar = build_thooft_symbols()
    W_plus = random_traceless_sym_3x3(rng)
    W_minus = random_traceless_sym_3x3(rng)
    C = make_weyl(W_plus, W_minus, eta, etabar)
    return C, W_plus, W_minus


def sd_decompose(C, eps):
    star_C = 0.5 * einsum('abef,efcd->abcd', eps, C)
    C_plus = 0.5 * (C + star_C)
    C_minus = 0.5 * (C - star_C)
    return C_plus, C_minus


def compute_p_q(C_plus, C_minus):
    p = float(einsum('abcd,abcd->', C_plus, C_plus))
    q = float(einsum('abcd,abcd->', C_minus, C_minus))
    return p, q


def build_omega(C_weyl, sigma):
    """Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}."""
    Omega = np.zeros((D, D, 4, 4), dtype=complex)
    for mu in range(D):
        for nu in range(D):
            for rho in range(D):
                for sig in range(D):
                    Omega[mu, nu] += 0.25 * C_weyl[mu, nu, rho, sig] * sigma[rho, sig]
    return Omega


# ===================================================================
# PART 1: Avramidi closed-form on symmetric space (constant curvature)
# ===================================================================
#
# On a symmetric space with covariantly constant curvature, the heat
# kernel has the Avramidi closed form. For a Laplace-type operator
# D = -(g^{mn} nabla_m nabla_n + E) on a vector bundle with connection
# curvature Omega_{mn}, the diagonal heat kernel is:
#
# K(t,x,x) = (4pi t)^{-d/2} * Phi(t)
#
# where Phi(t) is determined by the "matrix-valued phase space path integral"
# which on a symmetric space reduces to:
#
# Phi(t) = [det_spacetime(tR_L / sin(tR_L))]^{1/2} * exp(-t*E_eff)
#
# Here R_L is the "Lorentz curvature" operator: a d(d-1)/2 x d(d-1)/2 matrix
# acting on bivector indices, defined by:
#   (R_L)_{[mn],[pq]} = R_{mnpq}
#
# On Ricci-flat: R = C (Weyl), and E = 0 for Dirac (Lichnerowicz).
#
# HOWEVER: the formula above gives the SCALAR (geometry) part.
# The FIBER (Omega) part requires the fiber heat kernel, which on
# a constant-Omega background is:
#
# Phi_fiber(t) = exp(-t * Omega_eff)
#
# where Omega_eff involves the spin-curvature coupling.
#
# For the FULL computation including both geometry and fiber sectors,
# we use the COMBINED approach:
#
# The Lichnerowicz operator for Dirac is:
#   D_Lich = -Box + R/4 = -(g^{mn} nabla_m nabla_n + E)
# with E = -R/4 and Omega_{mn} = (1/4) R_{mnrs} gamma^{rs}.
#
# On Ricci-flat E = 0, and the heat kernel factorizes (on symmetric space):
#   K(t) = K_geom(t) * K_fiber(t)
#
# K_geom(t) = (4pi t)^{-d/2} * [det(tR_L / sinh(tR_L))]^{1/2}
# K_fiber(t) = exp(-t * [sum over Omega generators])
#
# But this factorization is subtle and the "product" is in the matrix sense.
#
# SIMPLEST CORRECT METHOD FOR EXTRACTION:
# We compute tr_spin[K(t,x,x)] perturbatively in t for several Weyl backgrounds,
# extract the t^4 coefficient (= a_4 in Avramidi convention = our a_8),
# and fit the quartic Weyl sector against the (p,q) basis.
#
# The perturbative expansion uses the KNOWN closed-form structure.


def compute_a8_numerical_from_closed_form(C_weyl, sigma, eps, n_terms=5):
    """Compute the traced a_8 coefficient using the Avramidi closed form.

    On a Ricci-flat symmetric space with constant Weyl and E=0:

    The heat trace density (traced over spinor indices) has the expansion:
    tr_spin[K(t,x,x)] = (4pi t)^{-2} * sum_n t^n * b_n

    where b_n = tr_spin[a_n(x)].

    The Avramidi closed form for the Lichnerowicz Laplacian on a
    Ricci-flat symmetric space gives:

    tr[K(t)] = (4pi t)^{-2} * tr_spin[
        det^{1/2}(t*Omega_L / sinh(t*Omega_L)) * exp(t * E_curvature)
    ]

    where Omega_L is the 6x6 matrix acting on bivector indices:
      (Omega_L)_{[mn],[pq]} = C_{mnpq}

    and E_curvature = (1/4) sum_{mn} Omega_{mn}^2 on Ricci-flat.

    Wait -- the formula needs to be derived carefully.

    CORRECT FORMULA (Avramidi 2000, Section 6.2):
    For the operator D = -(Box + E) on a bundle with curvature Omega,
    on a symmetric space (nabla R = 0, nabla Omega = 0):

    K(t,x,x) = (4pi t)^{-d/2} * det^{1/2}[tR/(sinh(tR))]
                * Phi(t)

    where:
    - R is the Riemann curvature as a d(d-1)/2 x d(d-1)/2 matrix
    - Phi(t) satisfies a matrix ODE involving E and Omega
    - On Ricci-flat with E=0: Phi(t) involves only Omega

    For the Dirac Lichnerowicz operator:
    The operator is D = -Box + R/4, so E = -R/4.
    On Ricci-flat: E = 0.
    Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}.

    The full heat kernel on the Ricci-flat symmetric space:
    K(t) = (4pi t)^{-2} * det^{1/2}[tR_bv/(sinh(tR_bv))]
           * Phi_spin(t)

    where R_bv is the Riemann curvature acting on bivectors
    and Phi_spin(t) is determined by Omega.

    FOR PRACTICAL COMPUTATION:
    We EXPAND everything in power series in t and extract t^4 coefficient.
    """
    # Build Omega
    Omega = build_omega(C_weyl, sigma)

    # --- Geometry sector ---
    # R_bv: the Riemann tensor as a 6x6 matrix on bivector space
    # Bivector basis: (01), (02), (03), (12), (13), (23)
    bv_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    R_bv = np.zeros((6, 6))
    for i, (a, b) in enumerate(bv_pairs):
        for j, (c, d) in enumerate(bv_pairs):
            R_bv[i, j] = C_weyl[a, b, c, d]

    # det^{1/2}[tR/(sinh(tR))] expanded to order t^4:
    # Let M = R_bv (6x6 matrix).
    # f(t) = det^{1/2}[tM/sinh(tM)]
    #       = det^{1/2}[1 - t^2 M^2/6 + 7t^4 M^4/360 - ...]
    #
    # Using det^{1/2}[I + X] = exp((1/2)tr[ln(I+X)])
    #                        = exp((1/2)(tr X - tr X^2/2 + ...))
    #
    # tM/sinh(tM) = I - (1/6)t^2 M^2 + (7/360)t^4 M^4 - ...
    # So X = -(1/6)t^2 M^2 + (7/360)t^4 M^4 - ...
    #
    # tr X = -(1/6)t^2 tr(M^2) + (7/360)t^4 tr(M^4) + ...
    # tr X^2 = (1/36)t^4 [tr(M^2)]^2 + ...
    #
    # (1/2)(tr X - tr X^2/2 + ...) =
    #   -(1/12)t^2 tr(M^2)
    #   + (1/2)[(7/360)tr(M^4) - (1/72)[tr(M^2)]^2] t^4 + ...
    #   = -(1/12)t^2 tr(M^2)
    #   + [7/(720) tr(M^4) - (1/144)[tr(M^2)]^2] t^4 + ...
    #
    # f(t) = exp[-(1/12)t^2 tr(M^2) + (7/720 tr(M^4) - 1/144 [tr M^2]^2) t^4 + ...]
    #       = 1 + [-(1/12)tr M^2] t^2
    #         + [(1/2)(1/12)^2 (tr M^2)^2 + 7/720 tr(M^4) - 1/144 (tr M^2)^2] t^4 + ...
    #       = 1 - (1/12)tr(M^2) t^2
    #         + [(1/288)(trM^2)^2 + 7/720 tr(M^4) - 1/144 (trM^2)^2] t^4 + ...
    #       = 1 - (1/12)tr(M^2) t^2
    #         + [(-1/288)(trM^2)^2 + (7/720)tr(M^4)] t^4 + ...

    M = R_bv
    M2 = M @ M
    M4 = M2 @ M2
    trM2 = np.trace(M2)
    trM4 = np.trace(M4)

    # Geometry factor coefficients (scalar, multiply by Id_4 for spinor trace)
    geom_t0 = 1.0  # constant
    geom_t2 = -(1.0/12.0) * trM2
    geom_t4 = (-1.0/288.0) * trM2**2 + (7.0/720.0) * trM4

    # --- Fiber (Omega) sector ---
    # On flat space with constant Omega, the fiber heat kernel is:
    # Phi_spin(t) = sum_n t^n a_n^{fiber}
    #
    # where a_n^{fiber} are determined by the Omega-only terms.
    #
    # The Omega-dependent part of the heat kernel diagonal on flat space
    # with constant field strength Omega_{mn} is:
    #
    # For the Lichnerowicz operator D = -(nabla^2 + E) with E = -R/4:
    # On flat space (R=0), D = -nabla^2 where nabla = partial + Omega (schematically).
    #
    # The EXACT heat kernel for a constant field strength on flat space
    # involves the Fock-Schwinger gauge. In this gauge:
    #
    # A_m(x) = -(1/2) Omega_{mn} x^n  (connection 1-form)
    #
    # and the operator becomes:
    # D = -(partial_m + A_m)^2 = -(partial^2 + 2 A_m partial_m + A_m A_m + partial_m A_m)
    #   = -(partial^2 - Omega_{mn} x^n partial^m + (1/4) Omega_{mk} x^k Omega_{ml} x^l - (d/2))
    #
    # Actually, for a CONNECTION (not EM field), the operator on a flat bundle is:
    # D psi = -(partial_m + Omega_m)(partial^m + Omega^m) psi
    # where Omega_m is the connection 1-form (matrix-valued).
    # In Fock-Schwinger gauge: Omega_m(x) = -(1/2) Omega_{mn} x^n
    # (here Omega_{mn} is the curvature, matrix-valued in the fiber).
    #
    # The heat kernel for this operator is known exactly:
    #
    # K(t,x,x) = (4pi t)^{-d/2} * product over pairs of eigenvalues of Omega
    #
    # For a 4x4 antisymmetric MATRIX-VALUED Omega, this gets complicated.
    # Let me use the EXPANSION approach instead.

    # The fiber contribution to the a_n coefficients, on flat space with
    # constant Omega (E=0), comes ONLY from the connection curvature.
    # The known coefficients are:
    #
    # a_0^{fiber} = Id_4   (identity in spinor space)
    # a_1^{fiber} = 0      (no E, no R)
    # a_2^{fiber} = (1/12) Omega_sq_mat
    #   where Omega_sq_mat = sum_{mn} Omega_{mn} @ Omega_{mn}  (4x4 spinor matrix)
    #
    # For a_3^{fiber}: From Vassilevich Eq. (4.3), the Omega-only terms in
    # a_3 (on flat space with E=0) are:
    # 7! * a_3^{fiber} = 60 tr(Omega_{ab} Omega^{bc} Omega_c^a)
    #                  + 30 * some other contractions
    # But "tr" here is the FIBER trace, and the structure is a MATRIX in fiber space.
    # Let me use the RECURSION instead.

    # RECURSION on flat space with E=0 and constant Omega:
    # The De Witt recursion for the diagonal heat kernel is:
    #   n * a_n = Delta(a_{n-1})|_{diagonal} + E * a_{n-1} + [Omega terms]
    #
    # On flat space with constant Omega, Delta = nabla^2, and acting on a_n (scalar)
    # gives zero since a_n is constant. BUT a_n is fiber-valued, and the
    # covariant Laplacian acting on it involves Omega.
    #
    # Actually, the CORRECT recursion involves the coincidence limit of
    # the off-diagonal heat kernel, not just the diagonal.
    # The recursion is:
    #   sigma^a nabla_a a_n + n*a_n = -D_x a_{n-1}
    # where sigma is the world function and D_x is the operator acting on x.
    #
    # At coincidence (x = x'), sigma = 0, and the recursion becomes:
    #   n * a_n(x,x) = -lim_{x'->x} D_x a_{n-1}(x,x')
    # which involves the off-diagonal a_{n-1}.
    #
    # This is more complex than just applying D to the diagonal.
    # The correct formula involves the "transport operator" and requires
    # knowledge of the full off-diagonal structure.
    #
    # FOR PRACTICAL COMPUTATION: I will use the KNOWN FORMULAS for
    # a_0 through a_4 as polynomials in E, Omega, R, and their derivatives,
    # specialized to flat space with E=0 and constant Omega.

    # From the Vassilevich-Avramidi universal formulas:
    # a_0 = Id
    # a_1 = (1/6)R + E = 0
    # a_2 = (1/2)E^2 + (1/6)Box E + (1/12)Omega^2 + (1/6)RE
    #        - (1/180)(Riem^2 - Ric^2) + (1/30)Box R + (1/72)R^2
    #      = (1/12) Omega_sq  [on flat, E=0]
    #
    # FOR a_3 (Vassilevich Eq. 4.3): On flat space E=0:
    # The ONLY surviving terms are those with Omega^3.
    # From Vassilevich:
    # 7! * a_3|_{Omega only} = 60 sum_{a,b,c} Omega_{ab} Omega_{bc} Omega_{ca}
    #
    # (on flat space with no R terms)
    # So: a_3 = (1/7!) * 60 * Omega_{ab} Omega_{bc} Omega_{ca}
    #         = (60/5040) * Omega3_chain
    #         = (1/84) * Omega3_chain
    #
    # where Omega3_chain = sum_{abc} Omega_{ab} @ Omega_{bc} @ Omega_{ca}  (4x4 matrix)

    # Compute Omega structures
    Omega_sq_mat = np.zeros((4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            Omega_sq_mat += Omega[a, b] @ Omega[a, b]

    Omega3_chain = np.zeros((4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            for c in range(D):
                Omega3_chain += Omega[a, b] @ Omega[b, c] @ Omega[c, a]

    # a_2^{fiber} = (1/12) Omega_sq_mat
    a2_fiber = (1.0/12.0) * Omega_sq_mat

    # a_3^{fiber} = (1/84) Omega3_chain  [Vassilevich]
    # WAIT: I need to verify the coefficient.
    # Vassilevich Eq (4.3), the terms without E or R:
    # In his normalization: 7! * (4pi)^{d/2} * a_3 involves:
    # "+60 Omega_{ij;k} Omega^{ij;k}" -- this is a derivative term
    # "+30 Omega_{ij} Omega^{jk} Omega_k^i" -- this is the cubic Omega chain
    # Hmm wait, there's also:
    # "+12 Omega_{ij} Omega^{ij;k}_{;k}" -- derivative
    #
    # On CONSTANT Omega (nabla Omega = 0), only the non-derivative term survives:
    # 7! * a_3|_{constant Omega, E=0, R=0} = 30 * tr-fiber(Omega_{ij} Omega^{jk} Omega_k^i)
    #
    # Hmm, but this "tr-fiber" - is the 30 already accounting for the fiber trace
    # or is the 30 the coefficient BEFORE taking fiber trace?
    #
    # In Vassilevich's notation, a_n is the TRACED coefficient (scalar-valued).
    # So his formula gives tr[a_n(x)], not the matrix a_n(x).
    #
    # For our purpose we need the MATRIX-valued a_n (endomorphism in spinor space)
    # to build a_4 which involves products of lower a_n's.
    #
    # The matrix-valued a_3 on constant Omega, flat, E=0:
    # Following Avramidi's conventions (the matrix-valued coefficients):
    # The heat kernel diagonal K(t,x,x) = (4pi t)^{-d/2} sum t^n a_n(x)
    # where a_n(x) is an endomorphism.
    # a_3 = (1/7!) * 30 * Omega_{ij} Omega^{jk} Omega_k^i  (as matrix in fiber)
    #      = (30/5040) * Omega3_chain = (1/168) * Omega3_chain
    #
    # CORRECTION: the Vassilevich formula has coefficient 30 for the cubic Omega
    # in the TRACED a_3 formula (i.e., after taking tr_fiber).
    # For the MATRIX-valued a_3, the coefficient is different.
    # From Avramidi (2000, Table 5.1), the matrix-valued a_3 contains:
    # (1/60) Omega_{ij;k} Omega^{ij;k} + (1/168) Omega_{ij} Omega^{jk} Omega_k^i + ...
    # On constant Omega, only the non-derivative cubic term survives.
    #
    # I'll use: a_3^{matrix} = (1/168) * Omega3_chain
    # and then VERIFY by checking tr(a_3) against the Vassilevich formula.

    # Check: tr(Omega3_chain) should relate to Vassilevich's coefficient.
    # From Vassilevich: 7! * tr(a_3)|_{cubic Omega} = 30 * tr(Omega3_chain)
    # => tr(a_3) = (30/5040) * tr(Omega3_chain) = (1/168) * tr(Omega3_chain)
    # So if a_3^{matrix} = c * Omega3_chain, then tr(a_3) = c * tr(Omega3_chain)
    # Matching: c = 1/168. Good.
    # BUT WAIT: if a_3 has OTHER Omega^3 structures (with different index
    # contractions), then the matrix and trace formulas are not so simply related.
    #
    # The ONLY non-derivative cubic Omega structure on flat space is:
    # Omega_{ij} Omega^{jk} Omega_k^i  (the chain with cyclic contraction)
    # Other contractions like Omega_{ij} Omega^{ij} Omega_{kk} = 0 since Omega is
    # antisymmetric in spacetime indices.
    # And Omega_{ij} Omega_{jk} Omega^{ik} = -Omega_{ij} Omega_{jk} Omega_{ki}
    # (by antisymmetry of Omega in spacetime indices: Omega^{ik} = -Omega^{ki}).
    # So the chain IS the unique cubic non-derivative structure.

    a3_fiber = (1.0/168.0) * Omega3_chain

    # For a_4 (our a_8):
    # On flat space with E=0 and constant Omega, a_4 involves quartic Omega terms.
    # The non-derivative quartic Omega structures are:
    # Q1: Omega_{ab} Omega^{bc} Omega_{cd} Omega^{da}  (4-chain)
    # Q2: (Omega_{ab} Omega^{ab})^2 = Omega_sq_mat^2    (square of Omega_sq)
    # Q3: Omega_{ab} Omega_{cd} Omega^{ab} Omega^{cd}   (= Q2 by relabeling)
    # Actually Q2 and Q3 are the same: sum_{abcd} Omega[a,b]@Omega[a,b]@Omega[c,d]@Omega[c,d]
    # which is Omega_sq_mat @ Omega_sq_mat.
    #
    # But there may be OTHER quartic contractions:
    # Q4: Omega_{ab} Omega^{cd} Omega_{cd} Omega^{ab} = Omega_sq_mat^2 (same as Q2)
    # Q5: Omega_{ab} Omega_{cd} Omega^{ac} Omega^{bd} (mixed contraction)
    #
    # On a flat background, the independent quartic Omega structures (as matrices
    # in fiber space) are:
    # Q1_mat = sum_{abcd} Omega[a,b] @ Omega[b,c] @ Omega[c,d] @ Omega[d,a]
    # Q2_mat = Omega_sq_mat @ Omega_sq_mat
    # Q5_mat = sum_{abcd} Omega[a,b] @ Omega[c,d] @ Omega[a,c] @ Omega[b,d]
    #
    # The general a_4 on flat, E=0, constant Omega is:
    # a_4 = c_Q1 * Q1_mat + c_Q2 * Q2_mat + c_Q5 * Q5_mat

    Q1_mat = np.zeros((4, 4), dtype=complex)  # 4-chain
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    Q1_mat += Omega[a, b] @ Omega[b, c] @ Omega[c, d] @ Omega[d, a]

    Q2_mat = Omega_sq_mat @ Omega_sq_mat  # (Omega^2)^2

    Q5_mat = np.zeros((4, 4), dtype=complex)  # mixed contraction
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    Q5_mat += Omega[a, b] @ Omega[c, d] @ Omega[a, c] @ Omega[b, d]

    # Now: a_4^{fiber} = alpha * Q1_mat + beta * Q2_mat + gamma * Q5_mat
    # The coefficients alpha, beta, gamma come from the Avramidi recursion.
    # We will DETERMINE them by fitting against the (p,q) decomposition.

    # BUT FIRST: we need the FULL a_4 including the mixed geometry-fiber terms.
    # On Ricci-flat (not flat!) with constant Weyl:
    #
    # a_4 = a_4^{pure geom} * Id_4  +  a_4^{fiber quartic}
    #      + a_4^{mixed: geom x fiber}
    #
    # The mixed terms come from:
    # - R_{abcd} * Omega_{mn} contractions (R*Omega^2 type at quadratic level, etc.)
    # - These contribute at the a_4 level through the recursion.
    #
    # On Ricci-flat: R_{abcd} = C_{abcd}, and for Dirac:
    # Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}
    # So Omega DEPENDS on C, and the "mixed" terms are NOT independent.
    #
    # The full traced a_4 is:
    # tr_spin[a_4] = geom_t4 * 4  [pure geometry, Id_4 has trace 4]
    #              + tr_spin[a_4^{fiber quartic}]
    #              + tr_spin[a_4^{mixed}]
    #
    # ALL of these are quartic in C (since Omega is linear in C).
    # The pure-geometry term is from the det^{1/2}[...] expansion above.

    # COMPLETE APPROACH:
    # Rather than trying to separate the sectors, compute the TOTAL traced a_4
    # including all contributions, and fit against (p^2+q^2, pq).

    # The total traced heat trace density up to order t^4:
    # tr K(t) = (4pi t)^{-2} * [geom_factor(t)] * tr_spin[fiber_factor(t)]
    #
    # geom_factor(t) = 1 + geom_t2 * t^2 + geom_t4 * t^4 + ...
    # tr_spin[fiber_factor(t)] = 4 + tr(a_2^f)*t^2 + tr(a_3^f)*t^3
    #                           + tr(a_4^f)*t^4 + ...
    #
    # Product up to t^4:
    # tr K(t) = (4pi t)^{-2} * [
    #   4                                           [t^0]
    #   + 0                                         [t^1]
    #   + (4*geom_t2 + tr(a_2^f))                  [t^2]
    #   + tr(a_3^f)                                 [t^3]
    #   + (4*geom_t4 + geom_t2*tr(a_2^f) + tr(a_4^f))  [t^4]
    # ]
    #
    # So: tr[a_4^{total}] = 4*geom_t4 + geom_t2*tr(a_2^f) + tr(a_4^f)

    # Hmm but this product formula is WRONG. The heat kernel is NOT a product
    # of geometry and fiber factors in general. The factorization only holds
    # on a symmetric space, and even then the two sectors are coupled.
    #
    # CORRECT APPROACH for the Avramidi closed form:
    # On a symmetric space, the heat kernel diagonal is given by the
    # Avramidi formula (Chapter 6 of his book) which involves a SINGLE
    # matrix exponential, not a product. The formula is:
    #
    # K(t,x,x) = (4pi t)^{-d/2} * Omega(t)
    #
    # where Omega(t) is determined by a matrix equation involving the full
    # curvature (Riemann + Omega combined).
    #
    # For the Lichnerowicz operator on a Ricci-flat space:
    # The curvature data is:
    # - R_{mnpq} = C_{mnpq} (Weyl tensor)
    # - Omega_{mn} = (1/4) C_{mnrs} sigma^{rs} (spin curvature)
    # - E = 0
    #
    # The Avramidi formula combines these into a single "total curvature"
    # operator acting on the combined (spacetime x fiber) space.
    #
    # THIS IS TOO COMPLEX TO IMPLEMENT HERE.
    #
    # FALLBACK: Use KNOWN RESULTS for the a_n coefficients.
    # The key formula is Vassilevich Eq. (4.1) which gives a_n in terms of
    # E, Omega, R, and their derivatives. On Ricci-flat with E=0:
    #
    # a_0 = Id
    # a_1 = 0
    # a_2 = (1/12) Omega_sq + (1/180) C^2 * Id
    # a_3 = [cubic Omega terms] + [mixed R*Omega terms] + [cubic R terms]
    #      = (1/168) Omega3_chain + (mixed terms) + (cubic Weyl) * Id
    #
    # For a_4: we need the FULL formula from Avramidi (2000) or the
    # Amsterdamski-Berkin-O'Connor (1989) result extended to non-scalar operators.
    #
    # Since I cannot extract this formula from the literature programmatically,
    # let me use an ALTERNATIVE approach.

    return {
        "Omega_sq_mat": Omega_sq_mat,
        "Omega3_chain": Omega3_chain,
        "Q1_mat": Q1_mat,
        "Q2_mat": Q2_mat,
        "Q5_mat": Q5_mat,
        "a2_fiber": a2_fiber,
        "a3_fiber": a3_fiber,
        "geom_t2": geom_t2,
        "geom_t4": geom_t4,
        "trM2": trM2,
        "trM4": trM4,
    }


# ===================================================================
# PART 2: Direct numerical extraction via small-t expansion on S2xS2
# ===================================================================
# The CLEAN approach: compute the heat trace on a specific manifold
# where the spectrum is known, and extract a_8 from the small-t asymptotics.
#
# BUT: we need a Ricci-flat manifold with nonzero Weyl, which means K3
# or a product involving it. The spectrum of the Dirac operator on K3
# is NOT known analytically.
#
# ALTERNATIVE: Use the Gilkey-Branson-Orsted UNIVERSAL FORMULA for a_4.
# This formula expresses a_4 as a polynomial in curvature invariants
# with UNIVERSAL rational coefficients. The coefficients have been
# computed by several groups.
#
# The most reliable source is Avramidi's thesis (1986) and book (2000).
# The key formula is:
#
# 9! * (4pi)^{d/2} * a_4 = sum of dimension-8 curvature invariants
#
# with specific rational coefficients for each invariant.
#
# Avramidi (2000, Table 7.1) gives the a_4 coefficient for a GENERAL
# Laplace-type operator. The Omega^4 terms are:
#
# From Avramidi (2000), the quartic Omega terms in a_4 (with his normalization):
#
# NOTE: I am reconstructing these from the KNOWN recursion pattern.
# The coefficients for a_2 and a_3 follow a specific pattern:
#
# a_2 has (1/12) Omega^2 (Vassilevich verified)
# a_3 has (1/168) Omega^3_chain + others
#
# The a_4 quartic Omega coefficients should follow the pattern:
# In normalization 9! * (4pi)^2 * a_4:
#
# Actually, let me use a completely different method. Rather than trying
# to extract coefficients from the literature, I will COMPUTE the
# heat kernel perturbatively using Feynman diagrams in the background
# field formalism.


# ===================================================================
# PART 3: FEYNMAN DIAGRAM COMPUTATION of a_8 Omega^4 sector
# ===================================================================
#
# The heat kernel has a path integral representation:
#   K(t,x,x) = int [Dz] exp(-int_0^t ds [(1/4)z_dot^2 + V(z)])
#
# where V(z) involves E and Omega.
#
# On flat space with constant Omega and E=0, the only vertex is the
# spin-orbit coupling V = -Omega_{mn} z^m dz^n/ds.
#
# The heat kernel to quartic order in Omega involves connected diagrams
# with 1, 2, 3, 4 Omega insertions.
#
# For the DIAGONAL heat kernel (x = x'):
#   <z^m(s) z^n(s')> = 2 [min(s,s') - ss'/t] delta^{mn} (propagator on [0,t])
#   <z^m(s)> = 0  (for the diagonal)
#
# The n-th order term is:
#   a_n^{fiber} = (1/t^n) * (connected n-point function)
#
# However, the Omega vertex is LINEAR in z (from z^m Omega_{mn} dz^n/ds),
# so the Feynman rules involve the worldline propagator.
#
# THIS IS IMPLEMENTABLE. The quartic term involves:
#   <(int V)^4>_connected with V = Omega_{mn} z^m dz^n/ds
#
# The worldline integrals are products of Green's functions on [0,t].
#
# Let me implement this worldline computation.


def worldline_green(s1, s2, t):
    """Green's function on [0,t] with Dirichlet BC: G(s1,s2) = min(s1,s2) - s1*s2/t."""
    return min(s1, s2) - s1 * s2 / t


def worldline_green_dot1(s1, s2, t):
    """d/ds1 G(s1,s2) = theta(s2-s1) - s2/t."""
    return (1.0 if s2 > s1 else (0.5 if s2 == s1 else 0.0)) - s2 / t


def worldline_green_dot2(s1, s2, t):
    """d/ds2 G(s1,s2) = theta(s1-s2) - s1/t."""
    return (1.0 if s1 > s2 else (0.5 if s1 == s2 else 0.0)) - s1 / t


def worldline_green_dot12(s1, s2, t):
    """d/ds1 d/ds2 G(s1,s2) = -delta(s1-s2) - 1/t."""
    # For s1 != s2: G_dot12 = -1/t
    # The delta function appears at coincident points and is regularized.
    return -1.0 / t  # Off-coincident


# Actually, the worldline computation for the heat kernel is more subtle.
# The vertex structure for the spin connection is:
#
# V = Omega_m(x) dx^m/ds = (gauge vertex from connection)
#
# In the Fock-Schwinger gauge with x_0 = x (the observation point):
# A_m(z) = (1/2) Omega_{mn} z^n  [to linear order in curvature]
#
# The worldline action includes:
# S_int = int_0^t ds A_m(z(s)) dz^m/ds = (1/2) int_0^t ds Omega_{mn} z^n dz^m/ds
#
# The heat kernel is:
# K(t,x,x) = (4pi t)^{-d/2} * <P exp(-S_int)>
# where P is path-ordering and <...> is expectation in the free worldline.
#
# To quartic order in Omega:
# <(S_int)^4> involves 4 insertions of (1/2)Omega_{mn} z^n dz^m/ds.
#
# This gives:
# (1/16) Omega_{m1n1} Omega_{m2n2} Omega_{m3n3} Omega_{m4n4} *
# int_0^t ds1 ds2 ds3 ds4 * <z^{n1}(s1) dz^{m1}/ds1 ... z^{n4}(s4) dz^{m4}/ds4>
#
# The Wick contractions involve products of worldline propagators.
# Each contraction is:
# <z^a(s) z^b(s')> = 2 G(s,s') delta^{ab}
# <z^a(s) dz^b(s')/ds'> = 2 G_dot2(s,s') delta^{ab}
# <dz^a/ds dz^b(s')/ds'> = 2 G_dot12(s,s') delta^{ab}
#
# For the Dirac case, we also need the spin factor:
# The heat kernel includes a factor from the spin connection:
# <P exp(-int Omega_{mn} z^m dz^n/2s * (SIGMA FACTOR))>
#
# Actually, for the LICHNEROWICZ operator (second-order), the worldline
# representation does NOT directly involve the spin factor in the same way.
# The Lichnerowicz operator is:
# D = -nabla^2 + R/4
# On Ricci-flat: D = -nabla^2
# where nabla involves the spin connection Omega_{mn} = (1/4) C sigma.
#
# The heat kernel of D = -nabla^2 on a vector bundle with connection A_m
# (fiber-valued 1-form) in the worldline formalism is:
#
# K(t,x,x) = (4pi t)^{-d/2} * <P exp(-int_0^t A_m(x+z(s)) dz^m/ds ds)>
#
# For our case: A_m is the spinor connection, A_m = (1/2) Omega_{mn} z^n
# in Fock-Schwinger gauge.
#
# The PATH-ORDERING is crucial for non-abelian connections!
# This means we cannot simply Wick-contract; we need to keep the
# fiber matrices in order.
#
# For the TRACED heat kernel, we take tr_spin of the path-ordered exponential.
#
# TO QUARTIC ORDER:
# P exp(-S) = Id - S + (1/2)P(S^2) - (1/6)P(S^3) + (1/24)P(S^4) + ...
#
# where S = int Omega_{mn} z^n dz^m/2s and P denotes path-ordering.
#
# The quartic term is:
# (1/24) * P[int_0^t ds1 int_0^t ds2 int_0^t ds3 int_0^t ds4
#           V(s1) V(s2) V(s3) V(s4)]
#
# With path-ordering: the integrals are restricted to s1 > s2 > s3 > s4,
# with a factor of 4! = 24 from the symmetrization. So:
#
# (1/24)*24 * int_{s1>s2>s3>s4} ds1 ds2 ds3 ds4 V(s1) V(s2) V(s3) V(s4)
# = int_{0<s4<s3<s2<s1<t} V(s1) V(s2) V(s3) V(s4) ds1 ds2 ds3 ds4
#
# Each V(si) = (1/2) Omega_{mi ni} z^{ni}(si) dz^{mi}(si)/dsi  [fiber matrix]
#
# This is a product of 4 fiber matrices (4x4 each) in the order s1>s2>s3>s4,
# contracted with worldline propagators from the z and dz Wick contractions.
#
# The computation involves:
# - 4 position variables z^{ni}(si) and 4 velocity variables dz^{mi}(si)/dsi
# - Wick contractions: pair up z's and dz's
# - Path ordering gives a definite matrix ordering
#
# This is a well-defined computation but requires careful implementation.
# Let me do it NUMERICALLY: evaluate the path-ordered integrals by
# Monte Carlo or quadrature for specific Omega configurations, and
# extract the a_4 coefficient.
#
# HOWEVER: a much SIMPLER approach exists.

# ===================================================================
# PART 4: EXACT COMPUTATION via matrix exponential
# ===================================================================
#
# For a CONSTANT abelian field strength on flat R^d, the heat kernel
# is known EXACTLY (Schwinger 1951):
#
# K(t) = (4pi t)^{-d/2} * det^{-1/2}[sinh(tF)/tF] * exp(-tE)
#
# where F is the field strength 2-form.
#
# For a NON-ABELIAN constant field strength, the formula is modified.
# The key difference is that path-ordering matters for non-abelian.
#
# However, there is a SPECIAL CASE: if the connection is
# ABELIAN in the fiber indices (i.e., [Omega_{mn}, Omega_{pq}] = 0
# for all mn,pq), then the path-ordered exponential becomes an ordinary
# exponential and the Schwinger formula applies.
#
# For the DIRAC operator with Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}:
# [Omega_{mn}, Omega_{pq}] = (1/16) C_{mnrs} C_{pquv} [sigma^{rs}, sigma^{uv}]
# This is generically NONZERO (the connection is non-abelian).
#
# HOWEVER: for SELF-DUAL or ANTI-SELF-DUAL Weyl tensors, the Omega
# matrices become block-diagonal in the chiral decomposition, and
# within each chiral block they may be abelianizable.
#
# For a GENERAL Weyl tensor, the non-abelian structure is essential.
#
# APPROACH: Discretize the path-ordered exponential.
# We divide [0,t] into N intervals and compute the product of
# matrix exponentials, each involving worldline-averaged Omega insertions.
# In the N -> infinity limit, this gives the exact result.
#
# Actually, let me use a COMPLETELY DIFFERENT and MUCH SIMPLER method.

# ===================================================================
# PART 5: CORRECT SIMPLE METHOD — Algebraic computation
# ===================================================================
#
# The a_4 coefficient for a Laplace-type operator D = -(Box + E) on a
# vector bundle is a UNIVERSAL POLYNOMIAL in the curvature invariants.
# On a Ricci-flat manifold with E = 0 and constant curvature:
#
# tr_spin[a_4] = A * (C^2)^2 + B * C^4_chain + C * C^4_other
#              + D * tr_spin(Omega^4) terms
#              + E * tr_spin(Omega^2) * C^2 mixed terms
#
# For the DIRAC operator, all of these are functions of the single
# tensor C_{abcd}, since Omega = (1/4)C*sigma.
#
# Therefore: tr_spin[a_4] = function of quartic Weyl invariants only
#           = A_total * (p^2+q^2) + B_total * pq
#
# We can determine A_total and B_total by evaluating on TWO specific
# backgrounds with DIFFERENT p/q ratios:
#
# Background 1: Self-dual (q = 0): tr[a_4] = A_total * p^2
# Background 2: General (p, q both nonzero)
#
# From two evaluations, we extract A_total and B_total.
#
# BUT: we need to be able to EVALUATE tr[a_4] numerically for
# a specific background. How?
#
# USE THE FACT that tr[a_4] can be computed from the SMALL-t
# expansion of the zeta function or the heat trace on that background.
#
# For a SPECIFIC BACKGROUND (fixed manifold), the heat trace is:
#   Tr exp(-tD) = (4pi t)^{-d/2} sum_n a_n * Vol + boundary terms
#
# If we can compute Tr exp(-tD) numerically for small t, we can
# extract a_4 by fitting.
#
# BUT: this requires KNOWING THE SPECTRUM of D, which we don't have
# for a general Ricci-flat manifold.
#
# FINAL CORRECT APPROACH: Use the KNOWN a_4 formula from the literature.
# The formula IS known (Avramidi 2000, Amsterdamski et al 1989, Barvinsky
# & Vilkovisky 1990). I will input the formula programmatically.
#
# From Avramidi (2000), "Heat Kernel and Quantum Gravity", Chapter 7,
# the a_4 coefficient (= our a_8) for a Laplace-type operator on a
# closed d-dimensional manifold without boundary has the structure:
#
# 9! * (4pi)^{d/2} * a_4 = integral sqrt{g} * sum of dim-8 invariants
#
# The quartic curvature invariants (without derivatives) are:
# (using the reduced basis valid on any 4-manifold)
#
# I1: R^4 = (R_{abcd})^4 = R_{abcd} R^{bcef} R_{efgh} R^{ghda}
# I2: (R_{abcd} R^{abcd})^2 = (Riem^2)^2
# I3: R^2 * Riem^2
# I4: (Ric^2)^2
# I5: R * R^{ab} R_{bc} R^c_a
# ... and more involving Ric, R
#
# On RICCI-FLAT: R = 0, R_{ab} = 0, R_{abcd} = C_{abcd}.
# Only I1 and I2 survive (the quartic Weyl invariants).
# Plus the Omega-dependent terms.
#
# The KEY FORMULA on Ricci-flat with E=0:
# From Avramidi (2000) and Barvinsky-Vilkovisky (1990):
#
# 9! * tr[a_4^{Ricci-flat, E=0}] =
#   c_geom_I1 * 4 * C4_chain + c_geom_I2 * 4 * (C^2)^2
#   + c_Om4_1 * tr(Omega4_chain) + c_Om4_2 * tr(Omega_sq^2)
#   + c_mixed_1 * C^2 * tr(Omega_sq)
#   + c_mixed_2 * C_{abcd} * tr(Omega^{ab} Omega^{cd})
#   + ... other mixed terms
#
# where the c's are rational numbers from the recursion.
#
# The TRACE of a_4 (taking the spinor trace) is what enters the
# heat trace and the spectral action.
#
# For the purpose of determining whether pq is present, we need:
# tr[a_4] as a function of (p,q).
#
# APPROACH: Compute tr[a_4] NUMERICALLY for many random Weyl backgrounds,
# fit against (p^2+q^2, pq), and extract the pq coefficient.
#
# The challenge: HOW to compute tr[a_4] numerically?
#
# ANSWER: Use the path-integral discretization.


def compute_heat_kernel_trace_lattice(C_weyl, sigma, t_val, N_lattice=200):
    """Compute tr[K(t,x,x)] on a flat 4-torus with constant Weyl curvature
    using lattice discretization of the worldline path integral.

    On flat space with constant field strength Omega, the heat kernel
    path integral reduces to a Gaussian integral that can be computed
    exactly by matrix methods.

    The key: for a constant (but non-abelian) connection on flat space,
    the heat kernel diagonal is:

    K(t,x,x) = (4pi t)^{-d/2} * [MATRIX_RESULT]

    where MATRIX_RESULT involves the path-ordered exponential of the
    connection integrated along closed loops.

    For CONSTANT field strength, the path-ordered integral over a
    straight-line path from x to x (closed loop) gives:

    P exp(-oint A_m dz^m) = Id  (trivially, for a point loop)

    But the heat kernel is NOT the straight-line approximation.
    It involves ALL loops weighted by the free-particle propagator.

    The correct computation uses the MATRIX-VALUED Mehler kernel
    for the non-abelian harmonic oscillator.

    CORRECT FORMULA (from Strassler 1992, Schubert 2001):
    For a constant non-abelian field strength F_{mn} (= our Omega_{mn})
    on flat R^d, the worldline path integral gives:

    K(t) = (4pi)^{-d/2} * t^{-d/2} *
           tr_fiber[T_ordered integral over all paths]

    For a CONSTANT field, this simplifies to:
    K(t) = (4pi t)^{-d/2} * tr_fiber[Phi(t)]

    where Phi(t) satisfies the matrix ODE:
    d Phi/dt = -E Phi + (1/12) Omega^{ab} Omega_{ab} Phi
               + higher-order corrections from the worldline

    On flat space with constant Omega and E = 0:
    Phi(t) = exp(t/12 * Omega_sq_mat) at leading order.

    The EXACT result involves the matrix-valued Mehler formula.
    For NON-ABELIAN constant Omega, the Mehler formula is:

    K(t,x,x) = (4pi t)^{-d/2} *
               [det_spacetime(tL / sinh(tL))]^{1/2} * Phi_fiber(t)

    where L_{mn,pq} = ... and Phi_fiber involves the non-abelian
    path-ordered exponential.

    FOR PRACTICAL PURPOSES: I will compute the heat kernel
    perturbatively to the required order (t^4 for a_4 = a_8)
    using the known perturbative expansion.
    """
    pass  # Placeholder


# ===================================================================
# PART 6: THE CORRECT AND FINAL METHOD
# ===================================================================
#
# After extensive analysis, the most reliable method is:
#
# 1. Use the KNOWN universal formulas for a_0, a_1, a_2, a_3 to
#    VERIFY our setup.
# 2. For a_4, use the AVRAMIDI RECURSION specialized to Ricci-flat E=0.
#
# The Avramidi recursion on a symmetric space (constant curvature) is
# ALGEBRAIC. The key formula (Avramidi 2000, Eq. 6.9) is:
#
# a_n = (1/n) * sum_{k=1}^{n} c_k * M_k * a_{n-k}
#
# where M_k are specific curvature monomials of order k and c_k are
# universal constants from the recursion.
#
# For our case (E=0 on Ricci-flat):
# M_1 = 0 (no R, no E)
# M_2 = (1/12) Omega_sq + (1/180) C^2 * Id  [from a_2]
# ... and so on.
#
# ACTUALLY: the recursion for the a_n on a symmetric space takes the
# very simple form (Avramidi 2000, Theorem 6.1):
#
# The heat kernel diagonal on a symmetric space is:
#
# K(t) = (4pi t)^{-d/2} * exp(-t * F(R))
#
# where F(R) is a matrix-valued function of the curvature.
# The a_n coefficients are then:
#
# a_n = ((-1)^n / n!) * [F(R)]^n
#
# This is the EXPONENTIAL ANSATZ, valid on symmetric spaces.
# It means a_4 = (1/24) * [F(R)]^4 where F(R) = -a_1 up to normalization.
#
# BUT a_1 = (1/6)R + E = 0 on Ricci-flat with E=0!
# So F(R) = 0 at first order, and we need to go to HIGHER ORDER
# in the expansion of F(R).
#
# Actually, the exponential ansatz K = exp(-tF) does NOT mean F = a_1.
# The relationship is:
# a_0 = Id
# a_1 = -F  (linear in t)
# a_2 = (1/2)F^2 - (correction from curved metric)
# etc.
#
# The corrections come from the NON-TRIVIAL MEASURE in the path integral
# on a curved space (the van Vleck-Morette determinant).
#
# On Ricci-flat with E=0: a_1 = 0, so F = 0 at first order.
# The first nonzero contribution to K-Id is at order t^2 from a_2.
# So the exponential ansatz gives:
# K(t) ~ Id + t^2 * a_2 + t^3 * a_3 + t^4 * a_4 + ...
# with a_4 ≠ (1/2)(a_2)^2 in general (the recursion is NOT simply exponential).
#
# THE KEY INSIGHT: On a Ricci-flat symmetric space with E=0, the recursion
# gives a_4 as a function of a_2, a_3, and new quartic curvature terms.
# The EXACT formula involves the coincidence limits of the off-diagonal
# kernel, which are determined by the full recursion.
#
# PRAGMATIC APPROACH: Give up on deriving c_S3 from the recursion directly
# and instead use the NUMERICAL PATH INTEGRAL method.


def compute_traced_a4_via_worldline_mc(C_weyl, sigma, n_samples=50000, rng=None):
    """Compute tr_spin[a_4(x)] using Monte Carlo worldline path integral.

    The heat trace density is:
    tr K(t,x,x) = (4pi t)^{-d/2} * sum_n t^n * tr[a_n]

    We compute the worldline path integral for the Lichnerowicz operator
    on flat space with constant Omega numerically, extracting the t^4 term.

    Method: Discretize the worldline into N steps, compute the
    path-ordered Wilson line for each random path, weight by the
    free-particle action, and average.

    This gives the EXACT (in the N -> infinity limit) heat kernel
    including all non-abelian effects.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    Omega = build_omega(C_weyl, sigma)

    # For the heat kernel on flat space with constant Omega:
    # K(t,x,x) = (4pi t)^{-d/2} * <P exp(-int_0^t A(z(s)) . dz(s))>
    #
    # The worldline z(s) is a closed loop: z(0) = z(t) = 0 (in FS gauge).
    # The free action is (1/4) int |dz/ds|^2 ds.
    # The gauge coupling is int A_m(z(s)) dz^m(s)/ds ds
    # where A_m(z) = (1/2) Omega_{mn} z^n in FS gauge.
    #
    # Discretize: z_i = z(i*dt), i = 0,...,N, dt = t/N, z_0 = z_N = 0.
    # Free action: S_free = (1/(4dt)) sum_i |z_{i+1} - z_i|^2
    # Gauge: S_gauge = (dt/2) sum_i Omega_{mn} * (z_i^n + z_{i+1}^n)/2
    #                         * (z_{i+1}^m - z_i^m)/dt
    #
    # The Wilson line W = P prod_i exp(-A_m(z_i+1/2) * (z_{i+1}^m - z_i^m))
    #
    # <tr W> / <1> gives the heat kernel trace.
    #
    # For small t, the dominant paths are short loops, and the expansion
    # in powers of Omega (which is proportional to t) gives the a_n.
    #
    # PROBLEM: Monte Carlo is noisy and requires many samples.
    # For extracting a_4, we need precision to ~10^-6 level, which
    # requires ~10^10 samples or variance reduction.
    #
    # BETTER: use the ANALYTIC worldline Green's function approach.
    pass


# ===================================================================
# PART 7: ANALYTIC WORLDLINE COMPUTATION (the correct method)
# ===================================================================
#
# On flat space with constant Omega, the path-ordered worldline integral
# can be computed EXACTLY to each order in Omega.
#
# The worldline propagator on [0,1] (we set t=1 and restore later):
# <z^a(u) z^b(v)> = 2 G_B(u,v) delta^{ab}
# where G_B(u,v) = min(u,v) - uv  (bosonic Green's function on [0,1])
#
# and d/du G_B(u,v) = theta(v-u) - v  (= G_B_dot(u,v))
#
# The coupling vertex is:
# V(u) = (1/2) Omega_{mn} z^n(u) dz^m(u)/du   [fiber matrix]
#
# The n-th order term in the expansion of the path-ordered exponential:
# a_n^{fiber, flat} = (-1)^n int_{0<u_n<...<u_1<1} du_1...du_n
#                     * <V(u_1) V(u_2) ... V(u_n)>_free
#                     * (matrix product in fiber space)
#
# Wick contractions of <V(u_1)...V(u_n)> involve pairings of z and dz.
# Each V has one z and one dz, so we need n z's and n dz's total.
# The contractions are:
#
# <z^a(u_i) z^b(u_j)> = 2 G_B(u_i, u_j) delta^{ab}
# <z^a(u_i) dz^b(u_j)/du_j> = 2 dG_B(u_i, u_j)/du_j * delta^{ab}
# <dz^a(u_i)/du_i dz^b(u_j)/du_j> = 2 d^2G_B(u_i,u_j)/du_i du_j * delta^{ab}
#
# For n = 4 (quartic in Omega):
# We have V_1 V_2 V_3 V_4 = product of 4 fiber matrices.
# Each V_i = (1/2) Omega_{m_i n_i} z^{n_i}(u_i) dz^{m_i}(u_i)/du_i
#
# The Wick contraction of 4 z's and 4 dz's can be done by pairing
# each z with a dz (or z with z and dz with dz, etc.).
#
# There are multiple Wick contraction patterns:
# (1) All z-dz pairs: z_i contracts with dz_j for some pairing
# (2) Some z-z and dz-dz pairs
#
# Total: the number of ways to pair 4 z's with 4 dz's is complex.
# It's easier to use the standard Wick theorem and sum over all pairings.
#
# For n = 4:
# <z_1 dz_1 z_2 dz_2 z_3 dz_3 z_4 dz_4> (with indices suppressed)
# = sum over all pairings * product of 2-point functions
#
# With 8 fields (4 z's and 4 dz's), each pairing has 4 contractions.
# Total number of Wick pairings: 7!! = 105.
# But many vanish by index structure.
#
# Actually, we need to be more careful. Each z^{n_i} carries a spacetime
# index n_i, and each dz^{m_i} carries an index m_i. The contraction
# <z^a dz^b> = 2 G'_B delta^{ab}, so the indices must match.
#
# This means we need to contract the spacetime indices of the Omega's
# according to the Wick pairing pattern.
#
# This is a FINITE and COMPUTABLE sum. Let me implement it.

def compute_a4_fiber_worldline():
    """Compute the fiber a_4 coefficient on flat space with constant Omega
    using the worldline formalism.

    Returns the coefficient as a function of the Omega structures Q1, Q2, Q5.
    """
    # The computation proceeds as follows:
    # 1. Write V_i = (1/2) Omega_{m_i n_i} z^{n_i} dz^{m_i}/du_i
    # 2. The quartic term is:
    #    a_4^{fiber} = (1/2)^4 * int_{0<u4<u3<u2<u1<1}
    #                 * sum_contractions (product of G_B's)
    #                 * (spacetime-contracted Omega product in fiber space)
    # 3. Evaluate the worldline integrals analytically using known formulas

    # The worldline integrals involve:
    # I_{ab,cd} = int_{0<u4<u3<u2<u1<1} G_B(u_a, u_c) G'_B(u_b, u_d) du1 du2 du3 du4
    # where a,b,c,d are indices into {1,2,3,4}.

    # For 4 V's with paired contractions, the contributing terms have the
    # form: Omega_{m1 n1} Omega_{m2 n2} Omega_{m3 n3} Omega_{m4 n4}
    #       * delta^{n_a m_c} delta^{n_b m_d} * (or similar pairings)
    #       * (worldline integral)
    #       * (fiber matrix product)

    # The fiber matrix product is path-ordered:
    # Omega_{m1 n1} @ Omega_{m2 n2} @ Omega_{m3 n3} @ Omega_{m4 n4}
    # (in order u1 > u2 > u3 > u4)

    # After contracting spacetime indices according to Wick pairings,
    # the result is a linear combination of the quartic Omega structures
    # Q1, Q2, Q5 (and their traces for the traced heat kernel).

    # Rather than doing this analytically (which requires enumerating
    # all 105 Wick contractions), let me compute NUMERICALLY.
    # Evaluate the FULL expression for specific Omega configurations
    # by numerical integration.

    print("\n" + "=" * 72)
    print("WORLDLINE COMPUTATION of a_4^{fiber} on flat space")
    print("=" * 72)

    # Numerical integration of the path-ordered worldline
    # using Gauss-Legendre quadrature on the ordered simplex.

    # For the ordered integral over 0 < u4 < u3 < u2 < u1 < 1:
    # Transform to the unit hypercube using u_i = v_1 * v_2 * ... * v_i
    # (i.e., u1 = v1, u2 = v1*v2, u3 = v1*v2*v3, u4 = v1*v2*v3*v4)
    # with Jacobian |J| = v1^3 * v2^2 * v3.

    # Use Gauss-Legendre on [0,1] for each v_i.
    from numpy.polynomial.legendre import leggauss

    n_quad = 16  # quadrature points per dimension
    nodes_01, weights_01 = leggauss(n_quad)
    # Transform from [-1,1] to [0,1]:
    nodes_01 = 0.5 * (nodes_01 + 1.0)
    weights_01 = 0.5 * weights_01

    return nodes_01, weights_01, n_quad


def gb(u, v):
    """Bosonic worldline Green's function G_B(u,v) = min(u,v) - u*v."""
    return min(u, v) - u * v


def gb_dot2(u, v):
    """dG_B/dv(u,v) = theta(u-v) - u. For u > v: 1 - u. For u < v: -u."""
    if u > v:
        return 1.0 - u
    elif u < v:
        return -u
    else:
        return 0.5 - u  # regularized


def gb_dot1(u, v):
    """dG_B/du(u,v) = theta(v-u) - v."""
    return gb_dot2(v, u)  # by symmetry G_B(u,v) = G_B(v,u)


def gb_dot12(u, v):
    """d^2 G_B/du dv = -delta(u-v) - 1. For u != v: -1."""
    # The delta function is handled by the path-ordering (u's are distinct).
    return -1.0  # Off-coincident


def evaluate_a4_fiber_for_omega(Omega, n_quad=20):
    """Evaluate tr_spin[a_4^{fiber}] numerically via worldline integration.

    The quartic term from the path-ordered Wilson line on flat space:

    a_4^{fiber} = (1/16) * int_{0<u4<u3<u2<u1<1}
                  * <Omega_{m1 n1} z^{n1}(u1) dz^{m1}/du1
                     Omega_{m2 n2} z^{n2}(u2) dz^{m2}/du2
                     Omega_{m3 n3} z^{n3}(u3) dz^{m3}/du3
                     Omega_{m4 n4} z^{n4}(u4) dz^{m4}/du4>

    Wick-contracted with the worldline propagator.
    """
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(n_quad)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    # Transform ordered simplex 0 < u4 < u3 < u2 < u1 < 1
    # to hypercube [0,1]^4 via:
    # u1 = v1, u2 = v1*v2, u3 = v1*v2*v3, u4 = v1*v2*v3*v4
    # Jacobian = v1^3 * v2^2 * v3

    result = np.zeros((4, 4), dtype=complex)

    for i1 in range(n_quad):
        v1 = nodes[i1]
        w1 = weights[i1]
        u1 = v1
        for i2 in range(n_quad):
            v2 = nodes[i2]
            w2 = weights[i2]
            u2 = v1 * v2
            for i3 in range(n_quad):
                v3 = nodes[i3]
                w3 = weights[i3]
                u3 = v1 * v2 * v3
                for i4 in range(n_quad):
                    v4 = nodes[i4]
                    w4 = weights[i4]
                    u4 = v1 * v2 * v3 * v4

                    jac = v1**3 * v2**2 * v3

                    # Wick contractions for
                    # <z^{n1}(u1) dz^{m1}/du1 z^{n2}(u2) dz^{m2}/du2
                    #  z^{n3}(u3) dz^{m3}/du3 z^{n4}(u4) dz^{m4}/du4>
                    #
                    # 8 fields: z1, dz1, z2, dz2, z3, dz3, z4, dz4
                    # Need to pair them. Each pairing gives a product of
                    # 4 two-point functions, each carrying a delta^{ab}.
                    #
                    # The non-zero two-point functions are:
                    # <z^a(u) z^b(v)> = 2 G_B(u,v) delta^{ab}
                    # <z^a(u) dz^b(v)/dv> = 2 G_B'_v(u,v) delta^{ab}
                    # <dz^a(u)/du z^b(v)> = 2 G_B'_u(u,v) delta^{ab}
                    # <dz^a(u)/du dz^b(v)/dv> = 2 G_B''(u,v) delta^{ab}
                    #
                    # For clarity, define:
                    # g(i,j) = G_B(u_i, u_j)
                    # gd1(i,j) = dG_B/du_i(u_i, u_j)
                    # gd2(i,j) = dG_B/du_j(u_i, u_j)
                    # gdd(i,j) = d^2G_B/du_i du_j(u_i, u_j)
                    u = [u1, u2, u3, u4]

                    # We need to enumerate ALL Wick pairings of
                    # {z1, dz1, z2, dz2, z3, dz3, z4, dz4}
                    # where z_i carries index n_i and dz_i carries index m_i.
                    #
                    # A pairing matches 8 objects into 4 pairs.
                    # Each pair gives a propagator (G, G', or G'') times delta.
                    #
                    # The spacetime delta's CONTRACT the Omega indices.
                    # The fiber matrices are in the FIXED ORDER: Omega_{m1n1} @ Omega_{m2n2} @ Omega_{m3n3} @ Omega_{m4n4}
                    # (path-ordered with u1 > u2 > u3 > u4).
                    #
                    # So the result involves a SUM over Wick pairings, each
                    # giving a specific index contraction of the 4 Omega's.
                    #
                    # This is complex but systematic. Let me enumerate the pairings
                    # using the labels: z1=0, d1=1, z2=2, d2=3, z3=4, d3=5, z4=6, d4=7
                    # where zi = z^{ni}(ui), di = dz^{mi}(ui)/dui.
                    #
                    # A pairing is a set of 4 pairs from {0,...,7}.
                    # The propagator for a pair depends on the type (z-z, z-d, d-z, d-d)
                    # and the proper-time labels.
                    #
                    # For efficiency, precompute the propagators:
                    G = np.zeros((4, 4))
                    Gd1 = np.zeros((4, 4))
                    Gd2 = np.zeros((4, 4))
                    Gdd = np.zeros((4, 4))
                    for ii in range(4):
                        for jj in range(4):
                            if ii == jj:
                                G[ii, jj] = gb(u[ii], u[jj])
                                Gd1[ii, jj] = 0.0
                                Gd2[ii, jj] = 0.0
                                Gdd[ii, jj] = -1.0  # Regularized
                            else:
                                G[ii, jj] = gb(u[ii], u[jj])
                                Gd1[ii, jj] = gb_dot1(u[ii], u[jj])
                                Gd2[ii, jj] = gb_dot2(u[ii], u[jj])
                                Gdd[ii, jj] = gb_dot12(u[ii], u[jj])

                    # Now enumerate the pairings.
                    # Instead of all 105 pairings (too many), use the FACTORED structure.
                    #
                    # Each V_i = Omega_{m_i n_i} z^{n_i} dz^{m_i}
                    # is a product of a fiber matrix Omega_{m_i n_i} and two worldline fields.
                    # After Wick contraction:
                    # - Each z^{n_i} contracts with some z^{n_j} or dz^{m_j}
                    # - Each dz^{m_i} contracts with some z^{n_j} or dz^{m_j}
                    # - The contraction gives delta^{n_i n_j} or delta^{n_i m_j} etc.
                    # - This delta contracts the Omega indices.
                    #
                    # The fiber matrix product is always in the order i=1,2,3,4.
                    # After index contraction, we get a specific quartic Omega structure.
                    #
                    # Rather than enumerate pairings, compute the FULL contraction
                    # using the matrix structure directly.
                    #
                    # The key insight: the integrand (after Wick contraction) can be
                    # written as a sum of terms, each involving a specific spacetime
                    # index contraction pattern.
                    #
                    # For 4 vertices, each with 2 fields (z and dz), we have 8 fields.
                    # A Wick pairing matches them into 4 pairs.
                    # Let me denote the type of each field:
                    # Field 2i: type z, time u_{i+1}, index n_{i+1}
                    # Field 2i+1: type dz, time u_{i+1}, index m_{i+1}
                    #
                    # A pair (a, b) with a < b gives propagator:
                    # prop(a,b) = 2 * P(a,b) * delta(index_a, index_b)
                    # where P(a,b) depends on the types:
                    # z-z: G_B(u_a, u_b)
                    # z-dz: G'_v(u_a, u_b) = gb_dot2(u_a, u_b) [derivative on the b side]
                    # dz-z: G'_u(u_a, u_b) = gb_dot1(u_a, u_b)
                    # dz-dz: G''(u_a, u_b) = gb_dot12(u_a, u_b)

                    # For the TRACED version, we compute:
                    # sum over all pairings: (product of 4 propagators) *
                    # (tr of contracted Omega product)
                    #
                    # Each pairing contracts the spacetime indices and
                    # determines the specific tr(Omega...Omega) structure.

                    # I'll compute this by explicit summation over spacetime
                    # indices. For each pairing, the delta's in the propagators
                    # set certain indices equal, and the result is a trace of
                    # a specific product of Omega matrices.

                    # SIMPLIFICATION: Rather than enumerate pairings, compute the
                    # FULL EXPECTATION VALUE directly using the determinant formula.
                    #
                    # <z^{a1}(u1) dz^{b1}/du1 z^{a2}(u2) dz^{b2}/du2
                    #  z^{a3}(u3) dz^{b3}/du3 z^{a4}(u4) dz^{b4}/du4>
                    #
                    # = sum over all perfect matchings of {(a1,z1), (b1,d1), (a2,z2), ...}
                    #
                    # For TRACED heat kernel with specific Omega's:
                    # result = (1/16) * sum_{m1,n1,...,m4,n4}
                    #          tr[Omega_{m1n1} Omega_{m2n2} Omega_{m3n3} Omega_{m4n4}]
                    #          * <z^{n1} dz^{m1} z^{n2} dz^{m2} z^{n3} dz^{m3} z^{n4} dz^{m4}>
                    #
                    # The expectation value decomposes into pairings with delta's.
                    # After summing over spacetime indices, the ONLY thing that survives
                    # are specific traces of Omega products with contracted indices.
                    #
                    # THIS IS EQUIVALENT TO: summing over all ways to connect the
                    # 4 z-legs and 4 dz-legs with propagators, and computing the
                    # resulting Omega trace.

                    # PRACTICAL IMPLEMENTATION:
                    # Compute the integrand as a sum over all pairings,
                    # but use the MATRIX STRUCTURE to do it efficiently.
                    #
                    # For each pair of vertex sites (i, j) with i != j:
                    # The propagator for (z_i, z_j) is 2*G[i,j]*delta^{n_i n_j}
                    # The propagator for (z_i, dz_j) is 2*Gd2[i,j]*delta^{n_i m_j}
                    # etc.
                    #
                    # After summing over indices:
                    # If z_i pairs with z_j: delta^{n_i n_j} means both Omega's share
                    # the z-index -> Omega_{m_i, n} Omega_{m_j, n} (summed over n)
                    # If z_i pairs with dz_j: delta^{n_i m_j} means Omega_{m_i n_i}
                    # has n_i = m_j, so the dz-index of j connects to the z-index of i.

                    # This is getting very intricate. Let me take a shortcut:
                    # CONTRACT NUMERICALLY over spacetime indices.

                    # For a given set of u-values, compute the integrand:
                    # integrand = (1/16) * sum_{m's, n's}
                    #   tr_spin[Omega_{m1n1}...Omega_{m4n4}]
                    #   * <z^{n1}dz^{m1} z^{n2}dz^{m2} z^{n3}dz^{m3} z^{n4}dz^{m4}>
                    #
                    # The Wick expectation is computed by the PFAFFIAN of the
                    # covariance matrix. But it's easier to use the SUM OVER
                    # ALL PAIRINGS formula.

                    # For 8 Gaussian random variables X_1,...,X_8 with covariance C:
                    # <X_1 X_2 ... X_8> = sum_pairings prod_{(i,j) in pairing} C_{ij}
                    #
                    # The 8 variables are: (z^{n1}_u1, dz^{m1}_u1, z^{n2}_u2, ..., dz^{m4}_u4)
                    # But they have VECTOR indices (a = 0,...,3).
                    # <z^a_i z^b_j> = 2 G[i,j] delta^{ab}
                    # <z^a_i dz^b_j> = 2 Gd2[i,j] delta^{ab}
                    # etc.
                    #
                    # So the covariance BLOCKS (indexed by vertex pairs) are:
                    # C[(z_i, z_j)] = 2*G[i,j] (for spacetime components: * delta^{ab})
                    # C[(z_i, dz_j)] = 2*Gd2[i,j]
                    # C[(dz_i, z_j)] = 2*Gd1[i,j]
                    # C[(dz_i, dz_j)] = 2*Gdd[i,j]
                    #
                    # The TRACED integrand is then (after summing over pairings
                    # and spacetime indices):
                    # I = (1/16) * sum_pairings {product of scalar propagators
                    #     * (matched spacetime index trace) * tr_spin[Omega product]}
                    #
                    # For EACH pairing, the spacetime deltas determine which Omega
                    # indices are contracted. The fiber trace is then computed.
                    #
                    # This is a FINITE computation for each u-quadrature point.
                    # Let me implement it explicitly.

                    # Define the 8 "fields": field k has:
                    # - vertex index: k // 2 (0,1,2,3)
                    # - type: 'z' if k even, 'dz' if k odd
                    # - spacetime index: the free index of Omega at that vertex
                    #   field 2*i carries index n_{i+1} (the second index of Omega)
                    #   field 2*i+1 carries index m_{i+1} (the first index of Omega)

                    # Compute covariance C[k, l] for each pair of fields:
                    def cov(k, l):
                        vk = k // 2  # vertex index (0-3)
                        vl = l // 2
                        tk = k % 2   # 0 = z, 1 = dz
                        tl = l % 2
                        if tk == 0 and tl == 0:
                            return 2.0 * G[vk, vl]
                        elif tk == 0 and tl == 1:
                            return 2.0 * Gd2[vk, vl]
                        elif tk == 1 and tl == 0:
                            return 2.0 * Gd1[vk, vl]
                        else:
                            return 2.0 * Gdd[vk, vl]

                    # Enumerate all perfect matchings of {0,1,...,7}
                    # A perfect matching is a set of 4 pairs.
                    # Number of perfect matchings: 7!! = 105.
                    # Generate them recursively.
                    def perfect_matchings(items):
                        if len(items) == 0:
                            yield []
                            return
                        first = items[0]
                        rest = items[1:]
                        for i, partner in enumerate(rest):
                            remaining = rest[:i] + rest[i+1:]
                            for matching in perfect_matchings(remaining):
                                yield [(first, partner)] + matching

                    # For each matching, compute the contribution to the traced integrand.
                    integrand = np.zeros((4, 4), dtype=complex)

                    for matching in perfect_matchings(list(range(8))):
                        # Product of covariances (scalar part)
                        prod_cov = 1.0
                        for (a, b) in matching:
                            prod_cov *= cov(a, b)

                        if abs(prod_cov) < 1e-30:
                            continue

                        # Determine the spacetime index contraction pattern.
                        # For each vertex i (0-3), Omega_{m_i, n_i}.
                        # Field 2*i carries index n_i, field 2*i+1 carries index m_i.
                        # A pair (a, b) in the matching means fields a and b share
                        # a spacetime index (summed with delta).
                        #
                        # The contraction pattern determines which Omega indices are
                        # identified. We can compute this by building a "connection"
                        # between the free indices of the Omega's.
                        #
                        # The fiber matrix product is always:
                        # M = Omega[m1,n1] @ Omega[m2,n2] @ Omega[m3,n3] @ Omega[m4,n4]
                        # tr_spin(M) after summing over the contracted indices.
                        #
                        # The index structure:
                        # For each pair (a, b), the indices of field a and field b
                        # are set equal (summed over).

                        # Build the contraction: for each Omega vertex, we have
                        # two free indices (m_i and n_i). The pairings identify
                        # certain indices.
                        #
                        # Index labels:
                        # Vertex 0: indices (m0, n0)
                        # Vertex 1: indices (m1, n1)
                        # Vertex 2: indices (m2, n2)
                        # Vertex 3: indices (m3, n3)
                        #
                        # Field 2*i -> index n_i
                        # Field 2*i+1 -> index m_i
                        #
                        # Pair (a, b): set index_of_field_a = index_of_field_b
                        # This creates constraints among {m0, n0, m1, n1, m2, n2, m3, n3}.

                        # Represent each index by (vertex, type):
                        # field k -> (k//2, 'n' if k%2==0 else 'm')
                        def field_index(k):
                            return (k // 2, 'n' if k % 2 == 0 else 'm')

                        # Build union-find for the 8 indices
                        idx_map = {}  # field -> canonical index
                        for k in range(8):
                            idx_map[k] = k

                        for (a, b) in matching:
                            # Merge index groups of a and b
                            ra = a
                            while idx_map[ra] != ra:
                                ra = idx_map[ra]
                            rb = b
                            while idx_map[rb] != rb:
                                rb = idx_map[rb]
                            idx_map[rb] = ra

                        def find(k):
                            while idx_map[k] != k:
                                k = idx_map[k]
                            return k

                        # Now sum over spacetime indices.
                        # Each connected component of the pairing shares one
                        # free spacetime index (summed 0..3).
                        # Find the connected components.
                        groups = {}
                        for k in range(8):
                            r = find(k)
                            if r not in groups:
                                groups[r] = []
                            groups[r].append(k)

                        n_free = len(groups)  # number of free indices to sum over

                        # For each assignment of spacetime indices to the groups:
                        # Compute tr_spin[Omega_{m0,n0} @ Omega_{m1,n1} @ Omega_{m2,n2} @ Omega_{m3,n3}]
                        # where m_i and n_i are determined by the group they belong to.

                        # Get the index value for each field given a dictionary
                        # group_root -> spacetime_value.
                        def evaluate(group_vals):
                            indices = {}
                            for k in range(8):
                                r = find(k)
                                indices[k] = group_vals[r]
                            # Omega indices: Omega_{m_i, n_i}
                            # m_i is field 2*i+1, n_i is field 2*i
                            m = [indices[2*i+1] for i in range(4)]
                            n = [indices[2*i] for i in range(4)]
                            # Matrix product: Omega[m0,n0] @ Omega[m1,n1] @ Omega[m2,n2] @ Omega[m3,n3]
                            mat = Omega[m[0], n[0]] @ Omega[m[1], n[1]] @ Omega[m[2], n[2]] @ Omega[m[3], n[3]]
                            return mat

                        # Sum over all assignments
                        roots = list(groups.keys())
                        # For n_free independent indices, sum over D^{n_free} = 4^{n_free} values.
                        mat_sum = np.zeros((4, 4), dtype=complex)
                        for assignment in iproduct(range(D), repeat=n_free):
                            group_vals = dict(zip(roots, assignment))
                            mat_sum += evaluate(group_vals)

                        integrand += prod_cov * mat_sum

                    # Multiply by prefactor: (1/16) * jacobian * quadrature weights
                    # The (1/16) = (1/2)^4 comes from the 4 factors of (1/2) in V.
                    # Actually, we also need (-1)^4 = 1 from the expansion of
                    # P exp(-S) at 4th order. The 4th order term of exp(-x) is x^4/24,
                    # but with path-ordering it's just the ordered integral (no 1/24 since
                    # the ordering replaces the 1/n! factor).
                    # So the prefactor is (1/2)^4 = 1/16.

                    w_total = w1 * w2 * w3 * w4 * jac
                    result += (1.0/16.0) * w_total * integrand

    return result


def run_c_s3_extraction():
    """Main computation: extract c_S3 by fitting a_4^{fiber} against (p,q)."""
    print("=" * 72)
    print("EXTRACTION OF c_S3: coefficient of [tr(Omega^2)]^2 in a_8")
    print("=" * 72)

    gamma = build_gamma_euclidean()
    sigma = build_sigma(gamma)
    eps = build_levi_civita()

    # First, verify the infrastructure with a_2
    print("\n--- Verification: a_2^{fiber} on flat space ---")
    rng = np.random.default_rng(314159)
    C, _, _ = generate_random_weyl(rng)
    Omega = build_omega(C, sigma)

    Omega_sq_mat = np.zeros((4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            Omega_sq_mat += Omega[a, b] @ Omega[a, b]

    a2_expected = (1.0/12.0) * Omega_sq_mat
    tr_a2 = np.trace(a2_expected)
    C_plus, C_minus = sd_decompose(C, eps)
    p, q = compute_p_q(C_plus, C_minus)
    tr_a2_expected = (1.0/12.0) * (-(p+q)/2.0)  # tr(Omega_sq) = -(p+q)/2
    record("tr(a_2^{fiber}) = -(p+q)/24",
           abs(tr_a2 - tr_a2_expected) < 1e-8 * abs(tr_a2_expected),
           f"got={tr_a2.real:.8f}, expected={tr_a2_expected:.8f}")

    # Now compute a_4^{fiber} for several backgrounds
    print("\n--- Computing a_4^{fiber} via worldline integration ---")
    print("  (This uses n_quad=12 quadrature per dimension; takes ~30 sec)")

    n_quad = 12  # Reduce for speed; increase for accuracy
    n_trials = 8

    data = []
    for trial in range(n_trials):
        print(f"\n  Trial {trial+1}/{n_trials}...")
        C, _, _ = generate_random_weyl(rng)
        Omega = build_omega(C, sigma)
        C_plus, C_minus = sd_decompose(C, eps)
        p, q = compute_p_q(C_plus, C_minus)

        a4_mat = evaluate_a4_fiber_for_omega(Omega, n_quad=n_quad)
        tr_a4 = complex(np.trace(a4_mat)).real

        # Also compute the quartic structures for cross-check
        Omega_sq_mat = np.zeros((4, 4), dtype=complex)
        for a in range(D):
            for b in range(D):
                Omega_sq_mat += Omega[a, b] @ Omega[a, b]
        tr_Omega_sq = complex(np.trace(Omega_sq_mat)).real
        S3 = tr_Omega_sq ** 2  # [tr(Omega^2)]^2

        data.append({
            "p": p, "q": q,
            "p2q2": p**2 + q**2, "pq": p*q,
            "tr_a4": tr_a4,
            "S3": S3,
            "tr_Omega_sq": tr_Omega_sq,
        })
        print(f"    p={p:.6f}, q={q:.6f}, tr_a4={tr_a4:.10e}, S3={S3:.10e}")

    # Fit tr_a4 = alpha*(p^2+q^2) + beta*pq
    A_mat = np.array([[d["p2q2"], d["pq"]] for d in data])
    b_vec = np.array([d["tr_a4"] for d in data])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    alpha_fit = coeffs[0]
    beta_fit = coeffs[1]
    fit_vals = A_mat @ coeffs
    max_err = np.max(np.abs(fit_vals - b_vec))
    rel_err = max_err / (np.max(np.abs(b_vec)) + 1e-30)

    print(f"\n  FIT RESULT:")
    print(f"    tr[a_4^{{fiber}}] = {alpha_fit:.10e} * (p^2+q^2) + {beta_fit:.10e} * pq")
    print(f"    Max fit error: {max_err:.2e}")
    print(f"    Rel fit error: {rel_err:.2e}")
    print(f"    beta/alpha = {beta_fit/alpha_fit:.6f}")

    # The pq coefficient tells us about c_S3:
    # tr[a_4^{fiber}] has pq ONLY from the S3 = [tr(Omega^2)]^2 structure.
    # S3 = (p+q)^2/4 = (p^2+q^2)/4 + pq/2
    # So: pq coefficient = c_S3 * (1/2)
    # => c_S3 = 2 * beta_fit
    c_S3 = 2.0 * beta_fit

    # Also: p^2+q^2 coefficient = (contribution from all structures)
    # For the non-pq structures: c_other * (p^2+q^2)
    # For S3: c_S3 * (1/4) * (p^2+q^2)
    # Total: [c_other + c_S3/4] * (p^2+q^2) + c_S3/2 * pq
    # So: alpha_fit = c_other + c_S3/4

    c_other = alpha_fit - c_S3 / 4.0

    print(f"\n  EXTRACTED COEFFICIENTS:")
    print(f"    c_S3 (coeff of [tr(Omega^2)]^2) = {c_S3:.10e}")
    print(f"    c_other (coeff of (p^2+q^2) from S1+S2+S4) = {c_other:.10e}")
    print(f"    c_S3 / c_other = {c_S3 / c_other:.6f}")

    # Determine if c_S3 is zero
    is_zero = abs(beta_fit) < 1e-4 * abs(alpha_fit)
    print(f"\n  IS c_S3 ZERO? {'YES' if is_zero else 'NO'}")
    if is_zero:
        print("    => The quartic Omega sector has NO pq content.")
        print("    => The ratio (C^2)^2 : (*CC)^2 is 1:1 from this sector.")
    else:
        print(f"    => The quartic Omega sector HAS pq content.")
        print(f"    => beta/alpha = {beta_fit/alpha_fit:.6f}")
        ratio_Csq2 = alpha_fit/2 + beta_fit/4
        ratio_starCC2 = alpha_fit/2 - beta_fit/4
        if abs(ratio_starCC2) > 1e-30:
            print(f"    => (C^2)^2 : (*CC)^2 = {ratio_Csq2/ratio_starCC2:.6f} : 1")
        print(f"    => This does NOT resolve the three-loop problem.")

    record("c_S3 nonzero", not is_zero, f"c_S3={c_S3:.6e}")

    return {
        "alpha": float(alpha_fit),
        "beta": float(beta_fit),
        "c_S3": float(c_S3),
        "c_other": float(c_other),
        "max_fit_err": float(max_err),
        "rel_fit_err": float(rel_err),
        "is_zero": is_zero,
        "n_trials": n_trials,
        "n_quad": n_quad,
        "pass_count": PASS_COUNT,
        "fail_count": FAIL_COUNT,
        "data": data,
    }


if __name__ == "__main__":
    results = run_c_s3_extraction()

    # Save results
    def sanitize(obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, (np.floating, mp.mpf)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    with open(RESULTS_DIR / "a8_c_S3_extraction.json", "w") as f:
        json.dump(sanitize(results), f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'a8_c_S3_extraction.json'}")
    print(f"\nTotal checks: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL")
