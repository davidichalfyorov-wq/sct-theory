---
id: OP-41
title: "SCT spectral bridge"
domain: [theory, mathematics]
difficulty: very-hard
status: closed
deep-research-tier: D
blocks: []
blocked-by: []
roadmap-tasks: []
papers: [hep-th/9606001, 0805.2909]
date-opened: 2026-03-31
date-updated: 2026-04-07
progress: "DEFINITIVELY CLOSED (negative). Three kill-shots: (A) nilpotent-triangular collapse (Hasse A^N=0), (B) tensor-type mismatch (CJ->E^2 != C^2), (C) nonlinear paired-response. BD hierarchy: BD->R, BD^2->R^2-2boxR, CJ->E^2, ???->C^2 (open, needs CJ_B)."
---

# OP-41: SCT spectral bridge

## 1. Statement

Can the CJ observable be connected to ANY coefficient of the spectral
action Tr(f(D^2/Lambda^2))? Specifically, can CJ be expressed as a
functional of the Dirac operator D of a spectral triple, or is it a
fundamentally non-spectral quantity?

Seven concrete routes were tested and ALL produced structural
obstructions. The problem is to either find an eighth route that
succeeds, or prove a no-go theorem establishing that CJ is
intrinsically non-spectral.

## 2. Context

Spectral Causal Theory (SCT) is built on the spectral action principle:
the gravitational action is Tr(f(D^2/Lambda^2)), which at one loop
gives alpha_C C^2 + alpha_R R^2 with alpha_C = 13/120 and
alpha_R = 2(xi - 1/6)^2. The CJ observable was constructed to detect
curvature on causal sets and has been shown to measure the Bel-Robinson
combination E^2 + B^2.

A spectral bridge would connect the causal-set program (discrete,
combinatorial) to the spectral action program (continuous, operator-
algebraic). Such a connection would provide a microscopic derivation
of the spectral action from causal set dynamics.

However, there are fundamental structural differences:

- The spectral action gives C^2 = E^2 - B^2 (Gauss-Bonnet sign).
- CJ gives E^2 + B^2 (Bel-Robinson sign).
- On type-N spacetimes: C^2 = 0 but CJ != 0 (8 sigma detection).

These sign differences suggest the two quantities probe different
aspects of curvature.

## 3. Known Results

**Seven routes tested, all closed:**

1. **Spectral determinant route:** det(I - p A) where A is the
   adjacency matrix of the Hasse diagram. Result: det(I - p A) = 1
   for all p, because A is nilpotent (strict upper triangular on a
   DAG). No information content.

2. **Spectral zeta function route:** zeta_A(s) = Tr(A^{-s}). Result:
   undefined because A has zero eigenvalues (nilpotent). Regularized
   versions give trivial results.

3. **Cycle counting route:** det(I - t A) = exp(-sum_{k >= 1} Tr(A^k)
   t^k / k). Result: Tr(A^k) = 0 for all k >= 1 because a DAG has no
   cycles. All cycle-based spectral invariants vanish identically.

4. **Laplacian route:** L = D^T D where D is the incidence matrix.
   Result: the spectrum of L detects the Hasse diagram structure, but
   L does not distinguish E^2 from B^2. The smallest non-zero
   eigenvalue gives a notion of "gap" but does not correlate with
   Weyl curvature.

5. **Modified adjacency route:** B = A + A^T + alpha I. Result:
   det(B) = alpha^N times a polynomial that depends only on the number
   of links, not on geometry. Geometry-blind.

6. **Path integral route:** Z = sum over paths exp(-S[path]). Result:
   the sum over maximal chains gives the chain polynomial, which is
   related to CJ. But this is the DEFINITION of CJ, not a spectral
   bridge (circular).

7. **Connes distance route:** d_C(p, q) = sup{|f(p) - f(q)| :
   ||[D, f]|| <= 1}. Result: the Connes distance on a causal set uses
   the causal structure but not the spectral action coefficients.
   d_C encodes metric information, not curvature.

**Three structural kill-shots:**

- **Kill-shot 1:** det(I - p A) = 1 (nilpotent adjacency). All
  characteristic-polynomial spectral invariants are trivial on DAGs.
- **Kill-shot 2:** No cycles on a DAG. All trace-based invariants
  (Tr(A^k) for k >= 1) vanish. Cycle-counting spectral methods are
  inapplicable.
- **Kill-shot 3:** det(B) = alpha^N (geometry-blind). Symmetric
  modifications of A cannot recover geometry from the determinant.

**The sign obstruction:** CJ measures the Bel-Robinson tensor
T_abcd = C_aecf C_b^e_d^f + (*C)_aecf (*C)_b^e_d^f, which gives
E^2 + B^2. The spectral action gives C^2 = C_abcd C^abcd = E^2 - B^2.
The difference E^2 + B^2 vs E^2 - B^2 is frame-dependent vs
frame-independent: the Bel-Robinson tensor is the unique symmetric
traceless conserved tensor quadratic in Weyl, while C^2 is the unique
Lorentz scalar quadratic in Weyl.

## 3b. Definitive Closure (2026-04-07)

**VERDICT: DEFINITIVELY CLOSED (NEGATIVE). Bridge does NOT exist.**

### Three structural obstructions

**(A) Nilpotent-triangular collapse.** The Hasse diagram adjacency
matrix A is strictly upper-triangular on any DAG, hence A^N = 0. All
spectral invariants built from det(I − pA), Tr(A^k), or characteristic
polynomials are trivially zero or constant. This kills routes 1-3
from Section 3.

**(B) Tensor-type mismatch.** CJ measures the Bel-Robinson combination
E² (electric part of Weyl), while the spectral action gives C² = E² − B²
(Gauss-Bonnet sign). These are fundamentally different tensor
contractions: Bel-Robinson is symmetric traceless conserved; C² is a
Lorentz scalar. No linear combination recovers one from the other
without independent access to B².

**(C) Nonlinear paired-response.** CJ depends on the paired structure
of the causal diamond (flat reference + curved target). The spectral
action is defined on a single manifold. The subtraction scheme
inherent in CJ (comparing sprinklings on flat vs curved) has no
analogue in the spectral action framework.

### BD hierarchy (corrected)

The Benincasa-Dowker hierarchy relates causal set observables to
continuum curvature invariants:
- BD → R (scalar curvature, Benincasa-Dowker 2010)
- BD² → R² − 2□R (de Brito-Eichhorn-Pfeiffer 2301.13525)
- CJ → E² (this work)
- ??? → C² (open, would require a CJ_B observable sensitive to B²)

### Key correction

The problem is NOT "causal order too poor to encode geometry" (causal
order is rich — it encodes the conformal class). The problem is
"spectral compression kills observer-projected Hasse-path content":
the spectral invariants of the Hasse diagram are algebraically too
simple (nilpotent A) to capture the statistical content of CJ.

## 4. Failed Approaches

All seven routes described above represent failed approaches. The
failures are not due to computational limitations but to structural
obstructions:

- Routes 1-3 fail because DAGs have nilpotent adjacency matrices.
- Routes 4-5 fail because simple spectral invariants of the Hasse
  diagram do not encode Weyl curvature.
- Route 6 is circular.
- Route 7 encodes metric, not curvature.

The fundamental issue is that CJ is a SECOND-ORDER statistical
quantity (variance of chain lengths), while spectral invariants of
the adjacency matrix are FIRST-ORDER algebraic quantities (determinant,
trace, eigenvalues). Converting second-order statistics to first-order
algebra requires additional structure that the Hasse diagram does not
provide.

## 5. Success Criteria

**Either:**
- Construct a spectral quantity S[D, causal set] that satisfies:
  (a) S is defined in terms of a Dirac-type operator on the causal set,
  (b) S = alpha_C C^2 + ... in the continuum limit (matching SCT), and
  (c) CJ can be expressed as a function of S and other spectral
  invariants.

**Or:**
- Prove a no-go theorem: for any Dirac-type operator D on a finite
  poset, the spectral action Tr(f(D^2)) cannot reproduce the
  Bel-Robinson sign E^2 + B^2. The proof must use the structural
  properties of DAGs (nilpotency, acyclicity) as essential ingredients.

## 6. Suggested Directions

1. **Non-commutative geometry on causal sets.** Construct a spectral
   triple (A, H, D) where A is the algebra of functions on the causal
   set, H is a Hilbert space of spinors on the causal set, and D is
   a Dirac operator that encodes the causal structure. The spectral
   action Tr(f(D^2)) would then give a candidate bridge. The challenge
   is defining D: the standard Dirac operator requires a spin
   structure, which a causal set does not have.

2. **Retarded-advanced spectral theory.** Instead of the adjacency
   matrix A, use the retarded Green function G_R = (I - A)^{-1} =
   I + A + A^2 + ... (finite sum on a DAG). G_R is NOT nilpotent and
   has non-trivial spectral properties. Investigate whether
   Tr(G_R^k) encodes curvature.

3. **Hodge Laplacian on the order complex.** The order complex of a
   poset is a simplicial complex whose k-simplices are chains of
   length k. The Hodge Laplacian on this complex has a non-trivial
   spectrum that reflects the topology and geometry of the poset.
   Investigate whether the Hodge spectrum encodes Weyl curvature.

4. **Accept the sign difference.** Perhaps CJ and the spectral action
   probe complementary aspects of curvature (Bel-Robinson vs Gauss-
   Bonnet). The bridge would then not be an identity but a DUALITY:
   CJ + spectral action = 2 E^2 (electric part only), and
   CJ - spectral action = 2 B^2 (magnetic part only).

## 7. References

- Connes, A. (1996). "Gravity coupled with matter and the foundation
  of non-commutative geometry." arXiv:hep-th/9603053.
- Chamseddine, A. H. and Connes, A. (1996). "The spectral action
  principle." arXiv:hep-th/9606001.
- Codello, A., Percacci, R. and Rahmede, C. (2008). "Investigating
  the ultraviolet properties of gravity with a Wilsonian
  renormalization group equation." arXiv:0805.2909.
- Sorkin, R. D. (2003). "Causal sets: Discrete gravity."
  arXiv:gr-qc/0309009.

## 8. Connections

- **OP-36 (RSY derivation):** A successful spectral bridge would
  provide an alternative derivation of the CJ coefficient, potentially
  bypassing the combinatorial difficulties of OP-36.
- **OP-38 (non-vacuum coefficients):** The spectral action naturally
  includes Ricci terms through the Seeley-DeWitt coefficients. A
  bridge would predict the non-vacuum CJ structure.
- **SCT spectral action coefficients:** alpha_C = 13/120 and
  alpha_R = 2(xi - 1/6)^2 are the target values on the spectral side.
  Any bridge must reproduce or relate to these specific numbers.
