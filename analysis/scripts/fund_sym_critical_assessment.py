# ruff: noqa: E402, I001
"""
FUND-SYM: Critical assessment of the 1:1 ratio claim.

This script examines the key assumptions and potential failure modes
of the claim that the three-loop problem is resolved.

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from itertools import product as iproduct
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy import einsum

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

D = 4


def critical_check_1_ch_for_non_traceless():
    """Check: Does the CH identity hold for NON-traceless 3x3 matrices?

    The Weyl tensor is traceless, giving traceless 3x3 W+ and W-.
    For traceless A: Tr(A^4) = (1/2)(Tr A^2)^2.

    But the a_8 coefficient also receives contributions from terms
    involving the Ricci tensor. On shell, R_{ab} = 0 for vacuum gravity,
    but the three-loop counterterm is evaluated BEFORE going on-shell.

    If off-shell Ricci contributions modify the effective W+ and W-
    to be non-traceless, the CH identity changes.

    For a GENERAL 3x3 symmetric matrix A (with Tr A != 0):
    By Cayley-Hamilton in 3D:
    A^3 - (Tr A) A^2 + (1/2)((Tr A)^2 - Tr(A^2)) A - det(A) I = 0

    Taking trace:
    Tr(A^3) - (Tr A)(Tr A^2) + (1/2)((Tr A)^2 - Tr(A^2))(Tr A) - 3 det(A) = 0

    Then Tr(A^4) = (Tr A)(Tr A^3) - (1/2)((Tr A)^2 - Tr(A^2))(Tr A^2) + det(A)(Tr A)
    which is more complicated than the traceless case.

    Let's check: for the Weyl tensor on-shell, is the W+ matrix traceless?
    """
    print("=" * 72)
    print("CRITICAL CHECK 1: CH identity and tracelessness")
    print("=" * 72)

    # For the Weyl tensor, W+ is the self-dual part.
    # In the notation C+_{abcd} = W+_{ij} omega^i_{ab} omega^j_{cd}:
    # Tr(W+) = sum_i W+_{ii} corresponds to omega^i_{ab} omega^i_{cd} contracted with C+
    # = C+_{abab}??? No.
    # C+_{abcd} = C+_{cdab} (pair symmetry) and C+_{aacd} = 0 (traceless).
    # In the 3x3 matrix language: W+ is a 3x3 real symmetric matrix.
    # The tracelessness of the Weyl tensor C^a_{acd} = 0 translates to
    # Tr(W+) = 0 in the self-dual/anti-self-dual basis.
    #
    # PROOF: omega^i_{ab} omega^i_{cd} = (1/2)(delta_{ac} delta_{bd} - delta_{ad} delta_{bc})
    # + (1/2) epsilon_{abcd}  [for self-dual forms]
    # (This is the projector onto the self-dual subspace.)
    # Then: sum_i C+_{abia} (using omega^i contraction) relates to the trace of W+.
    # If C is Weyl (traceless on the first and third index), then
    # C+_{abia} = 0, which forces Tr(W+) = 0.
    #
    # CONCLUSION: For the Weyl tensor, W+ IS traceless. The CH identity holds.
    # But for contributions from R_{ab} (off-shell), the effective matrix
    # would NOT be traceless, and the CH identity would be modified.

    # Verify numerically
    rng = np.random.default_rng(100)
    for trial in range(5):
        # Random 3x3 symmetric traceless matrix
        A = rng.standard_normal((3, 3))
        A = (A + A.T) / 2.0
        A -= np.trace(A) / 3.0 * np.eye(3)

        tr_A4 = np.trace(np.linalg.matrix_power(A, 4))
        half_trA2_sq = 0.5 * np.trace(A @ A)**2
        err = abs(tr_A4 - half_trA2_sq)
        print(f"  Trial {trial}: Tr(A^4)={tr_A4:.6f}, (1/2)(TrA^2)^2={half_trA2_sq:.6f}, err={err:.2e}")

    print("\n  For traceless 3x3: CH gives Tr(A^4) = (1/2)(Tr A^2)^2 -- VERIFIED")
    print("  This holds for the Weyl tensor because C^a_{acd} = 0 => Tr(W+) = 0")

    # Now check what happens with trace (non-zero Ricci)
    print("\n  For NON-traceless 3x3 (off-shell with Ricci):")
    for trial in range(5):
        A = rng.standard_normal((3, 3))
        A = (A + A.T) / 2.0  # symmetric but NOT traceless

        tr_A4 = np.trace(np.linalg.matrix_power(A, 4))
        half_trA2_sq = 0.5 * np.trace(A @ A)**2
        err = abs(tr_A4 - half_trA2_sq)
        print(f"  Trial {trial}: Tr(A^4)={tr_A4:.6f}, (1/2)(TrA^2)^2={half_trA2_sq:.6f}, "
              f"ratio={tr_A4/half_trA2_sq if abs(half_trA2_sq) > 1e-10 else 'N/A':.6f}")

    print("\n  Off-shell: ratio varies! The CH identity BREAKS without tracelessness.")
    print("  However, for the three-loop counterterm, we evaluate on-shell")
    print("  (R_{ab} = 0 in vacuum), so the Weyl tensor IS traceless and CH holds.")


def critical_check_2_derivative_terms():
    """Check: Do derivative terms in a_8 spoil the ratio?

    The a_8 Seeley-DeWitt coefficient contains not just C^4 terms
    but also terms with covariant derivatives:
    - (nabla C)^2 C^0: C_{abcd;e} C^{abcd;e} (2 derivatives, quartic in Riemann)
    - nabla^2 C . C^2: terms with Box acting on Weyl times Weyl squared
    - etc.

    On RICCI-FLAT backgrounds, these terms still involve the Weyl tensor.
    The key question: do they contribute to the quartic Weyl invariants
    in a way that modifies the 1:1 ratio?

    For CONSTANT CURVATURE backgrounds (like S^4), nabla C = 0.
    But the three-loop counterterm is for GENERAL backgrounds.

    The answer depends on whether the derivative terms, when contracted
    with the propagator structure, generate independent quartic Weyl
    combinations not captured by (C^2)^2 + (*CC)^2.
    """
    print("\n" + "=" * 72)
    print("CRITICAL CHECK 2: Derivative terms in a_8")
    print("=" * 72)

    # Count of dimension-8 curvature invariants (Fulling et al. 1992):
    # Total: 26 independent invariants (for general d=4 Riemannian manifold)
    # On Ricci-flat: many reduce. The surviving purely-Weyl invariants:
    #
    # WITHOUT derivatives:
    #   1. (C^2)^2 = C_{abcd} C^{abcd} C_{efgh} C^{efgh}
    #   2. C^4_chain = C_{abcd} C^{cdef} C_{efgh} C^{ghab}
    #      (= C^4_cross by our CH analysis)
    #   These 2 are related by CH: C^4_chain = (1/2)(p^2+q^2)
    #   And they span a 2D space parametrized by (p^2+q^2, pq).
    #
    # WITH 2 derivatives:
    #   3. C_{abcd;e} C^{abcd;e}  (6 Weyl, 2 derivatives)
    #   4. C_{abcd;e} C^{abce;d}
    #   5. Box(C_{abcd}) C^{abcd}
    #   ... etc.
    #
    # WITH 4 derivatives:
    #   These would be dimension 8 but only quadratic in Weyl.
    #   They are relevant for the heat kernel but NOT for 3-loop counterterms
    #   (which are quartic in curvature at leading order).

    # For the THREE-LOOP COUNTERTERM specifically:
    # The 3-loop divergence in pure gravity was computed by Bern et al.
    # (arXiv: 1701.02422, 2017) and involves:
    # - R^4 type terms (quartic in Riemann, no derivatives)
    # - Lower terms are absorbed by field redefinitions

    # On shell (pure gravity, R_{ab} = 0):
    # The independent counterterm structures are:
    # K_1 = C^4_chain and K_2 = (C^2)^2
    # (or equivalently p^2+q^2 and pq)

    # The derivative terms (nabla C)^2 C^0 etc. are NOT independent
    # counterterms at three loops because:
    # 1. By integration by parts + Bianchi, many reduce to C^4 + total derivatives
    # 2. Field redefinitions g_{ab} -> g_{ab} + a C_{a}^{cde} C_{bcde} can
    #    remove some derivative structures

    # CONCLUSION: At three loops, the independent counterterms on-shell are
    # ONLY the non-derivative quartic Weyl invariants: K_1 and K_2.
    # The derivative terms are either redundant or removable.

    print("  Dimension-8 invariant count (FKWC): 26 total")
    print("  On Ricci-flat, non-derivative quartic Weyl: 2 (K_1, K_2)")
    print("  Derivative terms at dim-8:")
    print("    - (nabla C)^2 type: present in a_8 but reducible by IBP + Bianchi")
    print("    - Field redef g -> g + delta_g removes additional structures")
    print("  At three loops on shell: ONLY K_1 and K_2 survive as counterterms")
    print("  VERDICT: Derivative terms do NOT introduce new independent structures")
    print("           that would modify the 1:1 ratio.")


def critical_check_3_scalar_vector_ratio():
    """Check: Do scalar and vector fields also give 1:1 ratio?

    The spectral action sums over ALL fields in the SM.
    If scalars and vectors contribute different ratios,
    the total ratio would differ from 1:1.

    Scalar (spin 0): No spin connection. The quartic Weyl contribution
    comes from the universal b_8 coefficient of the scalar Laplacian.
    On Ricci-flat: Delta = -nabla^2 (minimal coupling).
    The a_8 coefficient for -nabla^2 involves only Riemann^4 terms,
    which on Ricci-flat are Weyl^4 = C^4_chain (by definition).
    C^4_chain = (1/2)(p^2 + q^2) by CH.
    So scalar gives ratio 1:1.

    Vector (spin 1): The connection is the Christoffel symbol in vector indices.
    Omega^{mu}_{nu rho sigma} = R^{mu}_{nu rho sigma}
    Tr_vec(Omega^4_chain) = R^a_{b m1n1} R^b_{c m2n2} R^c_{d m3n3} R^d_{a m4n4}
    On Ricci-flat: = C^a_b C^b_c C^c_d C^d_a = C^4_chain (in mixed indices)
    This is the SAME C^4_chain = (1/2)(p^2+q^2).
    So vector also gives ratio 1:1.

    But WAIT: for the vector, we need to be more careful about the trace.
    The vector trace is Tr_vec (over the mu index), which gives
    a factor of (D-2) = 2 for each vector component, but the RATIO
    p^2+q^2 vs pq does not depend on this factor.
    """
    print("\n" + "=" * 72)
    print("CRITICAL CHECK 3: Scalar and vector contributions")
    print("=" * 72)

    # For the scalar: verify that the scalar Laplacian a_8 gives C^4_chain
    # The scalar heat kernel has NO Omega term, so the quartic curvature
    # comes entirely from the "universal" part of the heat kernel expansion.
    # This universal part involves R_{abcd} R^{abcd} at a_4 and higher powers.
    #
    # At a_8 level, the scalar contribution is (schematically):
    # a_8^{scalar} = alpha_s * R^4 + beta_s * (nabla R)^2 R + ...
    # On Ricci-flat: = alpha_s * C^4 (specific contractions)
    #
    # The SPECIFIC contraction depends on the combinatorics of the heat kernel.
    # It is NOT simply C^4_chain. Let me think more carefully.
    #
    # Actually, for the scalar (minimal, no Omega, E = 0):
    # a_8 contains terms like:
    # - R_{abcd} R^{cdef} R_{efgh} R^{ghab} (chain)
    # - (R_{abcd} R^{abcd})^2 (square)
    # - Other contractions
    #
    # These are NOT all the same invariant! On Ricci-flat they become
    # C^4_chain and (C^2)^2, which are the two independent invariants.
    # So the scalar does NOT necessarily give ratio 1:1.
    #
    # CORRECTION: The scalar gives a SPECIFIC combination of (C^2)^2 and C^4_chain
    # (or equivalently p^2+q^2 and pq), with coefficients determined by
    # the combinatorics of the heat kernel expansion.
    #
    # For the Dirac spin-1/2: we showed that the Omega^4 piece gives 1:1.
    # But a_8 also has E^4, E^2 Omega^2, and other cross terms.
    # On Ricci-flat with E = -R/4 = 0 (since R = 0), the E terms vanish.
    # So the ONLY surviving contribution is Omega^4, which gives 1:1.
    # Plus the "universal" part (same as scalar).
    #
    # WAIT: The heat kernel for D^2 = -(nabla^2 + E) with E = -R/4
    # on Ricci-flat (R = 0) has E = 0. So all E-dependent terms vanish.
    # The surviving terms in a_8 are:
    # 1. Omega^4 terms (our computation, gives 1:1)
    # 2. "Universal" terms from the nabla^4 expansion (same as scalar)
    # 3. Cross terms Omega^2 * (nabla terms)
    #
    # The universal and cross terms need separate analysis.

    print("  IMPORTANT CORRECTION:")
    print("  The a_8 coefficient has MULTIPLE contributions beyond Tr(Omega^4):")
    print("  1. Tr(Omega^4_chain): proven 1:1 ratio (this work)")
    print("  2. Universal heat kernel terms (shared with scalar): NOT necessarily 1:1")
    print("  3. Cross terms (Omega^2 x nabla terms): on Ricci-flat, Omega^2 ~ C^2")
    print()
    print("  The universal terms are present for ALL fields and have the SAME")
    print("  ratio for all spins (since they come from the connection, not the bundle).")
    print("  The Omega^4 terms are spin-dependent but all give 1:1.")
    print("  The cross terms need explicit computation.")
    print()
    print("  KEY INSIGHT: The ratio question for the three-loop problem is about")
    print("  the TOTAL a_8, not just the Omega^4 piece. However:")
    print("  - On Ricci-flat, all terms reduce to Weyl^4 contractions")
    print("  - By our CH analysis, ALL Weyl^4 contractions live in the")
    print("    2D space spanned by (p^2+q^2) and pq")
    print("  - The question is whether the TOTAL coefficient of pq vanishes")
    print()
    print("  For the FULL a_8 on Ricci-flat, the terms are:")
    print("  (a) Pure Riemann^4 (from the scalar Laplacian sector)")
    print("  (b) Riemann^4 with spin trace (Omega^4, our computation)")
    print("  Term (a) involves the 3 contractions of R^4:")
    print("    (R^2)^2, R^4_chain, R^4_cross")
    print("  On Ricci-flat these become (C^2)^2, C^4_chain, C^4_cross")
    print("  By our CH identity: C^4_cross = C^4_chain, so STILL 2 invariants")
    print("  The RATIO of (C^2)^2 to C^4_chain in term (a) is NOT 1:1 in general")
    print("  -- it depends on the specific combinatorial coefficients in a_8.")


def critical_check_4_the_real_question():
    """The REAL question: does the full a_8 give a fixed ratio?

    Even if the ratio is NOT 1:1, the key point is:
    The spectral action generates a_8 with COMPLETELY FIXED coefficients
    (for a given particle content). The ratio c_1/c_2 is a SPECIFIC NUMBER.

    The three-loop counterterm ALSO has a specific ratio c_1^{ct}/c_2^{ct}.

    The question is: c_1/c_2 = c_1^{ct}/c_2^{ct} ?

    This is ONE equation with ZERO free parameters.
    It is either satisfied (by accident/symmetry) or not.

    The parameter f_8 controls the OVERALL scale, not the ratio.
    So f_8 can absorb the counterterm IF AND ONLY IF
    the ratio matches.

    Our analysis shows that Tr(Omega^4) contributes with ratio 1:1.
    But the TOTAL a_8 may have a different ratio due to other terms.

    HOWEVER: even without computing the exact ratio, the COUNTING
    changes from the V2 analysis:

    OLD (V2): "2 invariants, 1 parameter -> overdetermined (2:1)"
    NEW (this work): "2 invariants combine into 1 effective constraint
    (the ratio), plus 1 overall scale constraint -> 1+1 = 2 constraints,
    but f_8 absorbs the scale, leaving 1 constraint on the ratio.
    The ratio is FIXED by the spectral action (no free parameter).
    So it's 1 constraint with 0 parameters -> still 1:0 overdetermined,
    but it's a SINGLE numerical coincidence, not a structural obstruction."

    This is a WEAKER version of the three-loop problem: it requires
    ONE numerical coincidence instead of matching TWO independent numbers.
    """
    print("\n" + "=" * 72)
    print("CRITICAL CHECK 4: The real structure of the problem")
    print("=" * 72)

    print("""
    REVISED ANALYSIS of the three-loop problem:

    The V2/V3 analysis said:
      "2 quartic Weyl invariants K_1, K_2 vs 1 parameter f_8 -> 2:1 overdetermined"

    This was IMPRECISE. The correct statement:

    The spectral action generates: f_8 * a_8
    where a_8 = c_1 * K_1 + c_2 * K_2 + (derivative terms)

    Here c_1 and c_2 are FIXED NUMBERS determined by the SM content.
    The parameter f_8 scales the entire combination.

    The three-loop counterterm is: delta * [c_1^{ct} K_1 + c_2^{ct} K_2]
    where delta is the divergent part (needs absorption).

    For f_8 * a_8 to absorb the counterterm, we need:
    f_8 * c_1 = delta * c_1^{ct}  AND  f_8 * c_2 = delta * c_2^{ct}

    These TWO equations in ONE unknown (f_8/delta) are compatible iff:
    c_1 / c_2 = c_1^{ct} / c_2^{ct}   (RATIO CONDITION)

    If the ratio condition holds, then f_8 = delta * c_1^{ct} / c_1 solves both.
    If not, the system is inconsistent and D != 0 at three loops.

    WHAT WE PROVED:
    1. The CH identity reduces 3 quartic Weyl contractions to 2
    2. For the Tr(Omega^4) piece of a_8, the ratio IS 1:1 (i.e., (C^2)^2 = (*CC)^2)
    3. This is universal across all spins (a consequence of the chiral structure)

    WHAT WE DID NOT PROVE:
    4. That the FULL a_8 (including all terms, not just Omega^4) gives ratio 1:1
    5. That the counterterm ratio is also 1:1
    6. That these ratios match

    STATUS: The problem is REDUCED but NOT RESOLVED.
    The overdetermination is 1 ratio condition (not 2 independent numbers).
    This is a SIGNIFICANT improvement over the V2 analysis, but the
    three-loop problem remains open until the full ratio is computed.
    """)

    # Quantify the significance
    mp.mp.dps = 50

    print("  QUANTITATIVE SUMMARY:")
    print("  =====================")
    print()
    print("  V2 analysis: 2 constraints, 1 parameter -> P(match) ~ 0 (measure zero)")
    print("  This work:   1 ratio condition, 0 parameters -> P(match) = c_1/c_2 == c_1^ct/c_2^ct")
    print()
    print("  The probability of a 'random' ratio matching is indeed zero (measure zero),")
    print("  BUT the ratio could match due to a SYMMETRY or STRUCTURAL REASON.")
    print("  Our finding that the Omega^4 piece gives exactly 1:1 is suggestive")
    print("  of such a structural reason (the chiral decomposition of SO(4)).")
    print()
    print("  HOWEVER: the three-loop counterterm for pure gravity is known")
    print("  (Bern et al. 2017, arXiv:1701.02422). Its quartic Weyl structure")
    print("  involves a SPECIFIC combination that depends on the helicity of")
    print("  the internal gravitons. The self-dual decomposition of the")
    print("  counterterm would reveal whether its ratio is also 1:1.")
    print()
    print("  NEXT STEP: Compute the ratio c_1^{ct}/c_2^{ct} from Bern et al.")
    print("  and compare with the spectral action ratio c_1/c_2.")

    return {
        "revised_status": "REDUCED but NOT RESOLVED",
        "old_overdetermination": "2:1 (V2)",
        "new_overdetermination": "1 ratio condition (this work)",
        "omega4_ratio": "1:1 (proven)",
        "full_a8_ratio": "UNKNOWN (needs full computation)",
        "counterterm_ratio": "UNKNOWN (needs Bern et al. analysis)",
        "resolution_path": "Compute both ratios and compare",
    }


def main():
    critical_check_1_ch_for_non_traceless()
    critical_check_2_derivative_terms()
    critical_check_3_scalar_vector_ratio()
    assessment = critical_check_4_the_real_question()

    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print(f"\n  Status: {assessment['revised_status']}")
    print(f"  Omega^4 ratio: {assessment['omega4_ratio']}")
    print(f"  Full a_8 ratio: {assessment['full_a8_ratio']}")
    print(f"  Counterterm ratio: {assessment['counterterm_ratio']}")
    print(f"\n  The three-loop problem is REDUCED from 2:1 to 1:0 overdetermination.")
    print(f"  It now requires ONE numerical coincidence (ratio match)")
    print(f"  instead of matching two independent numbers.")
    print(f"  This is a significant theoretical improvement but not a resolution.")


if __name__ == "__main__":
    main()
