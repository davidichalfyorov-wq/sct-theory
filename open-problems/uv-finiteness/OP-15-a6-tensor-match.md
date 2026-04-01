---
id: OP-15
title: "Explicit a_6 tensor structure match"
domain: [uv-finiteness]
difficulty: hard
status: partial
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [MR-4]
papers: ["hep-th/9209108", "0906.1572", "hep-th/0306138", "0706.0691", "2001.05975"]
date-opened: 2026-03-31
date-updated: 2026-04-01
progress: "Literature complete. DF coefficients available. CCC triplet normalization needs TFORM verification."
---

# OP-15: Explicit a_6 tensor structure match

## 1. Statement

Verify computationally that the full a_6 Seeley-DeWitt coefficient for
the squared Dirac operator D^2 on a general 4-manifold, when evaluated
with the SM field content (N_s = 4, N_D = 22.5, N_v = 12) and reduced
on-shell using the SCT equations of motion, gives a counterterm that is
exactly absorbable by the spectral function deformation delta(psi) with
coefficient delta f_6 = 4 c_2 (or its appropriate generalization).

## 2. Context

The MR-4 investigation established that the two-loop counterterm is
absorbable (Answer C: R^3 is PRESENT but ABSORBABLE). The argument
proceeds in two stages:

1. **On-shell reduction:** The equations of motion R_{mu nu} = O(alpha_C)
   reduce all dimension-6 curvature invariants to cubic Weyl
   contractions (CCC). In d = 4, there are 2 parity-even cubic Weyl
   invariants, but the parity-odd one (J_1 = C_{abcd} *C^{cdef}
   C_{ef}^{ab}) vanishes identically, leaving 1 effective invariant.

2. **delta(psi) absorption:** The deformation delta(psi)(u) =
   c_2 (2 - 4u + u^2) e^{-u} shifts the spectral moment f_6 while
   preserving f_2 (Einstein-Hilbert) and f_4 (one-loop). The ratio
   |R| = |delta f_6^{needed} / delta f_6^{available}| = 0.00469,
   confirming that absorption is possible.

However, the explicit tensor-level verification that the FULL a_6
coefficient (before on-shell reduction) has the correct structure has
not been performed. The a_6 coefficient for a general Laplace-type
operator -Box - E involves 8 independent dimension-6 curvature
invariants:

  R^3, R R_{mu nu}^2, R C^2, R_{mu nu} R^{nu rho} R_{rho}^{mu},
  R_{mu nu} R_{rho sigma} C^{mu rho nu sigma}, C^3 (CCC),
  nabla_mu R nabla^mu R, Box^2 R

plus the Goroff-Sagnotti term R_{abcd} R^{cdef} R_{ef}^{ab}.

The MR-4 result uses the on-shell reduction R_{mu nu} -> O(alpha_C)
to eliminate most of these, leaving only CCC. But the on-shell
reduction is itself part of the claim: the reduction is valid only if
the one-loop equations of motion correctly eliminate the off-shell
invariants, which requires matching the precise tensor contractions.

The chirality theorem provides an independent structural guarantee
(counterterms are block-diagonal) that supports the absorption claim.
But a direct computational verification of the tensor coefficients
would close the gap definitively.

## 3. Known Results

- **SM CCC coefficient: -1481/6** (summed over N_s = 4 scalars,
  N_D = 22.5 Dirac fermions, N_v = 12 gauge bosons). Individual
  contributions: scalar -16/3, Dirac -109/3, vector 148/3.

- **Deformation ratio |R| = 0.00469.** This is the ratio of the
  required delta f_6 to the available delta f_6, confirming that the
  deformation is small (0.47% adjustment to the spectral function).

- **Parker-Toms a_6 coefficient.** Parker and Toms (2009) computed
  the full a_6 for the squared Dirac operator. The scalar R^3
  coefficient is -14/(9 * 7!) = -14/45360. The full result involves
  all 8 dimension-6 invariants with known coefficients.

- **Goroff-Sagnotti verification.** The Goroff-Sagnotti two-loop
  divergence for pure GR is proportional to C^3 with coefficient
  209/2880 (in Goroff-Sagnotti conventions). This is a specific CCC
  contraction and has been verified by multiple independent groups
  (van de Ven 1992, Bern et al. 2015).

- **Spectral function deformation structure.** The deformation
  delta(psi)(u) = P(u) e^{-u} with polynomial P(u) shifts f_{2k} =
  integral u^{k-1} P(u) e^{-u} du. For the quadratic P = c_2(2-4u+u^2),
  we get delta f_2 = 0, delta f_4 = 0, delta f_6 = 4 c_2. The cubic
  extension P = -8 + 22u - 10u^2 + u^3 also preserves f_8.

## 3b. Partial Resolution (2026-04-01)

**STATUS: PARTIAL. Full literature base collected. Explicit 4D coefficients
available from Decanini-Folacci. CCC triplet normalization discrepancy
needs TFORM verification.**

### Structural correction: 10 invariants, not 8

Standard 4D integral basis after IBP has 10 independent dimension-6
invariants, not 8 as stated in our problem formulation. The full list
from Decanini-Folacci (0706.0691):

  R Box R, R_munu Box R^munu, R^3, R R_munu^2,
  R_mu^nu R_nu^rho R_rho^mu, R_munu R_rhosigma C^murhomusigma,
  R R_munurhosigma^2, R_munu R^mu_rhosigmatau R^nurhosigmatau,
  C^3 (type 1), C^3 (type 2)

In d=4, the two CCC types are related by identity (Mistry 2001.05975,
eq.(3.5)): C_abcd C^ab_ef C^cdef = 2 C_abcd C^ac_ef C^bedf.
So effectively 9 independent invariants, reducing to 1 on Ricci-flat shell.

### Universal a_6 formula: confirmed

Vassilevich (hep-th/0306138) eq.(4.29): full formula for a_6 of
Laplace-type operator D = -(g^munu nabla_mu nabla_nu + E). Also
Gilkey, Theorem 4.8.16.

### Explicit 4D coefficients per spin (Decanini-Folacci)

All 10 coefficients in the normalization W = (1/(192 pi^2 m^2)) int sqrt(g) [...]:

- **Scalar (xi=0):** eq.(3.13) — complete
- **Dirac:** eq.(5.21) — complete
- **Vector (Proca):** eq.(5.24) — complete

On-shell (Ricci-flat) CCC coefficients (I1 convention):
- Scalar: 1/2520
- Dirac: -1/1260
- Vector: 1/840
- Ratios: scalar/Dirac = -1/2, vector/Dirac = -3/2

### Goroff-Sagnotti 209/2880: confirmed

- GS eq.(1.2): Gamma^(2) = -(209/2880)(1/(4pi)^4 epsilon) int sqrt(g) C^3
- van de Ven (hep-th/9209108) eq.(6.30): same result independently

### CCC triplet normalization issue (OPEN)

Our MR-4 CCC coefficients (-16/3, -109/3, 148/3) are internally
consistent: 4*(-16/3) + 22.5*(-109/3) + 12*(148/3) = -1481/6.

However, the RATIOS do not match Decanini-Folacci on-shell ratios:
- Our: scalar/Dirac = 16/109 ~ 0.147
- DF: scalar/Dirac = -1/2

This indicates a normalization mismatch between our MR-4 convention
and the standard heat-kernel convention. The numbers may include
different FP ghost treatment, different spectral-action moment factors,
or combinatorial prefactors. This requires clarification through
explicit TFORM computation.

### Quartic Weyl invariants: 2 in d=4

Moura eqs.(3.5)-(3.6): two independent parity-even quartic Weyl
contractions in d=4 (W+^2 W-^2 and W+^4 + W-^4). Confirms OP-13
overdetermination: 2 quartic Weyl vs 1 spectral parameter delta f_8.

### Spectral action six-derivative sector

Mistry (2001.05975) eq.(3.7): entire 6-derivative gravitational sector
of the spectral action sits under ONE coefficient mu_3, confirming the
structural basis for absorption by spectral function deformation.

### Remaining task

TFORM computation needed to:
1. Verify our CCC triplet (-16/3, -109/3, 148/3) in explicit normalization
2. Map between our convention and DF convention
3. Confirm tensor-level (not just coefficient-counting) absorption

## 4. Failed Approaches

1. **Direct symbolic computation with SymPy.** Attempted to compute
   the full a_6 for the Dirac operator using SymPy tensor modules.
   The computation involves 4-index tensor contractions at cubic order
   in curvature, with endomorphism E = -R/4 and bundle curvature
   Omega_{mu nu} = (1/4) R_{mu nu rho sigma} gamma^{rho} gamma^{sigma}.
   The number of terms exceeds 10^4 after gamma-matrix trace evaluation.
   SymPy runs out of memory for the full computation on a general
   background (not a specific metric). The approach is in principle
   correct but computationally intractable without a specialized
   tensor algebra system.

2. **FORM computation (partial).** Used FORM to evaluate selected
   tensor contractions in a_6. Successfully reproduced the R^3, R C^2,
   and CCC coefficients individually. However, the full off-shell
   a_6 with all 8 invariants and their interactions under on-shell
   reduction has not been completed. The remaining contractions
   (R_{mu nu} R^{nu rho} R_{rho}^{mu}, mixed Ricci-Weyl terms)
   require additional FORM scripts.

3. **cadabra2 computation (partial).** Used cadabra2 for the covariant
   derivative terms (nabla R nabla R, Box^2 R). Successfully verified
   that these contribute only total derivatives on closed manifolds
   and can be dropped. But the interaction terms between Ricci and
   Weyl at cubic order were not completed.

## 5. Success Criteria

- Explicit computation of all 8 dimension-6 coefficients in a_6 for
  the squared Dirac operator D^2 with the full SM field content.
- Explicit on-shell reduction using R_{mu nu} = O(alpha_C), showing
  that the off-shell terms are correctly eliminated.
- Verification that the remaining CCC coefficient matches -1481/6.
- Confirmation that delta(psi) with delta f_6 = 4 c_2 absorbs the
  counterterm exactly, with the correct tensor structure (not just
  the scalar coefficient).

## 6. Suggested Directions

1. **Full TFORM computation.** Use TFORM (parallel FORM, 8 workers)
   to evaluate the complete a_6 coefficient. The computation breaks
   into 3 stages: (a) evaluate all gamma-matrix traces in
   Tr(a_6(x, D^2)) for each spin species; (b) contract all curvature
   tensors using the Bianchi identity and Ricci decomposition;
   (c) collect coefficients of the 8 independent invariants.
   Estimated computation time: 2-4 hours on 8 P-cores.

2. **Avramidi covariant technique.** Use Avramidi's covariant
   algebraic approach (math-ph/0107018) which expresses a_6 directly
   in terms of E, Omega, R and their covariant derivatives. This
   avoids the explicit gamma-matrix traces and reduces the problem
   to index contractions of curvature tensors with known combinatorial
   coefficients.

3. **Cross-check against Vassilevich tables.** Vassilevich
   (hep-th/0306138) tabulates a_6 coefficients for general Laplace-
   type operators. Compare the SCT a_6 (summed over SM fields) with
   the tabulated result and verify coefficient-by-coefficient
   agreement.

4. **On-shell reduction via background field method.** Instead of
   computing the full off-shell a_6 and then reducing on-shell,
   compute the two-loop effective action directly in the background
   field formalism with the background satisfying the one-loop EOM.
   This automatically eliminates the off-shell terms and gives the
   on-shell counterterm directly.

## 7. References

1. Goroff, M.H. and Sagnotti, A. "The ultraviolet behavior of
   Einstein gravity," Nucl. Phys. B 266 (1986) 709.
2. van de Ven, A.E.M. "Two-loop quantum gravity,"
   Nucl. Phys. B 378 (1992) 309, hep-th/9209108.
3. Parker, L. and Toms, D.J. "Quantum Field Theory in Curved
   Spacetime," Cambridge University Press, 2009.
4. Vassilevich, D.V. "Heat kernel expansion: user's manual,"
   Phys. Rep. 388 (2003) 279, hep-th/0306138.
5. Avramidi, I.G. "Covariant algebraic method for calculation of
   the low-energy heat kernel," J. Math. Phys. 36 (1995) 5055,
   hep-th/9403036.
6. Bern, Z. et al. "Ultraviolet properties of N=8 supergravity
   at five loops," Phys. Rev. D 98 (2018) 086021, arXiv:0906.1572.

## 8. Connections

- **Roadmap: MR-4** (two-loop effective action). OP-15 is the
  tensor-level completion of the MR-4 absorption argument. MR-4
  established absorption at the level of coefficient counting;
  OP-15 verifies it at the level of explicit tensor contractions.
- **Related to OP-13** (three-loop): The successful tensor match at
  L = 2 does not imply success at L = 3 (where the overdetermination
  appears), but the methodology developed for OP-15 would be directly
  applicable to the a_8 computation needed for OP-13.
- **Related to OP-14** (hidden principle): If the tensor match at
  L = 2 reveals unexpected cancellations or symmetries in the
  coefficient structure, these might hint at the hidden principle
  needed for L = 3.
- **Independent of all unitarity problems** (OP-07 through OP-12):
  The tensor match is a purely algebraic verification within the
  perturbative expansion.
