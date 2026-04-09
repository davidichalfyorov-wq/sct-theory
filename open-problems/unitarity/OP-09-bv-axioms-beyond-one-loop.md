---
id: OP-09
title: "BV axioms 3 and 4 beyond one loop"
domain: [unitarity, uv-finiteness]
difficulty: very-hard
status: open
deep-research-tier: B
blocks: [OP-06]
blocked-by: []
roadmap-tasks: [MR-2, MR-5]
papers: ["1704.07728"]
date-opened: 2026-03-31
date-updated: 2026-04-07
progress: "Axiom 3 YES at two loops (condition: 0 not in Spec(Pi+dPi)). Axiom 4 CONDITIONAL (two-loop QME plausible, infinite-pole Delta convergence NOT proven). Main obstruction: ANALYTIC (Delta in infinite-pole limit)."
---

# OP-09: BV axioms 3 and 4 beyond one loop

## 1. Statement

Prove or disprove BV axioms 3 and 4 at all loop orders:

- **BV-3:** The superdeterminant Sdet(delta(D^2) / delta(g_{mu nu}))
  is well-defined and spectral (its logarithm has a heat-kernel
  expansion whose coefficients are local spectral invariants).

- **BV-4:** No non-spectral Batalin-Vilkovisky anomaly arises in the
  D^2-quantization of SCT at any loop order (the BV master equation
  (S, S) = 0 holds with S defined through the spectral action).

## 2. Context

The CHIRAL-Q program establishes UV finiteness of SCT within the
D^2-quantization framework. The proof uses the chirality identity
{D, gamma_5} = 0, which forces all perturbative counterterms to be
block-diagonal in the chiral decomposition, guaranteeing that the
parity-mixed (pq) components of the counterterm vanish and that all
divergences are absorbable by the spectral function deformation
delta(psi).

This chirality argument is algebraic and holds at any loop order,
but it assumes five BV axioms about the field redefinition g -> D^2:

| Axiom | Statement | Status |
|-------|-----------|--------|
| BV-1 | g -> D^2 is a smooth Frechet map | PROVEN (Lichnerowicz) |
| BV-2 | On-shell invertible, kernel = gauge | PROVEN (Connes reconstruction) |
| BV-3 | Sdet well-defined, spectral | VERIFIED 1-LOOP |
| BV-4 | No non-spectral BV anomaly | VERIFIED 1-LOOP |
| BV-5 | Cutoff commutes with field redef | NATURAL (cutoff defined in D^2) |

BV-1, BV-2, and BV-5 are established on general grounds. BV-3 and
BV-4 have been verified numerically and symbolically at one loop
(matching the Seeley-DeWitt a_4 coefficient structure) but lack
all-orders proofs.

The two theorems in the CHIRAL-Q paper have different conditional
status:
- **Theorem 6.12 (UNCONDITIONAL):** D = 0 at L = 1 and L = 2.
- **Theorem 6.11 (CONDITIONAL on BV-3,4):** D = 0 at all loop orders.

## 3. Known Results

- **BV-3 at one loop:** The functional determinant
  det(delta(D^2)/delta(g)) reduces to the product of eigenvalue
  derivatives of D^2 with respect to metric perturbations. At one loop,
  this is the standard DeWitt-Seeley heat-kernel determinant, which is
  well-defined for elliptic operators. Verified by matching the a_4
  coefficient (alpha_C = 13/120, alpha_R = 2(xi-1/6)^2).

- **BV-4 at one loop:** The BV master equation at one loop reduces to
  the standard Ward identities for diffeomorphism invariance. These are
  satisfied by construction (the spectral action is a diffeomorphism
  scalar). No anomaly appears.

- **Higher-loop structure:** At L >= 2, BV-3 requires that the L-loop
  effective action, when expressed in D^2 variables, has counterterms
  that are spectral invariants of D^2 (not just of g). At L = 2, this
  has been verified through the CHIRAL-Q chirality argument (which
  shows the counterterm is absorbable by delta(psi)). At L >= 3, the
  three-loop overdetermination (OP-13) complicates the picture.

- **Survival probability: 82-90%** (CHIRAL-Q assessment, covering
  both BV-3 and BV-4).

## 3b. Two-Loop Assessment (2026-04-07)

**VERDICT: BV-3 YES at two loops. BV-4 CONDITIONAL.**

### BV-3 (Axiom 3) at two loops

BV-3 holds at two loops under the condition:
  0 ∉ Spec(Π_TT + δΠ_TT)
i.e., the one-loop corrected propagator has no new zero modes. This is
satisfied for SCT because Π_TT(z) = 1 + (13/60)z·F̂₁(z) has well-
separated zeros (minimum spacing Δz > 2 from the CL analysis), and
the two-loop correction δΠ is of order α²_C ≈ 10⁻⁴, which cannot
create new zeros in the gaps.

### BV-4 (Axiom 4) at two loops

The two-loop quantum master equation (QME) (S + ℏΓ₁, S + ℏΓ₁) = 0
is plausible but the infinite-pole BV antibracket operator Δ requires
convergence control. The issue is ANALYTIC, not structural:
- The antibracket Δ = ∂²/(∂Φ ∂Φ*) acts on functionals of the full
  infinite-pole propagator
- Convergence of Δ in the infinite-pole limit is NOT proven
- For finite poles (Anselmi's case), Δ is well-defined by construction

### Main obstruction classification

The obstruction to all-orders BV axioms is:
1. **Primary: ANALYTIC** — convergence of Δ in infinite-pole limit
2. **Secondary: STRUCTURAL** — D² embedding ≠ BV diffeomorphism
   (the map g → D² is not a field-space diffeomorphism in the BV sense)

### ABJ anomaly risk: LOW

No pure gravitational anomalies exist in d = 4 (all gravitational
anomalies require chiral matter in even d ≥ 6). The ABJ-type anomaly
from the standard model matter is already accounted for in the SM
spectral triple.

### Literature corrections

- The QME formula in the original prompt was corrected
- Reference 2006.06037 is an ML paper, not a fakeon paper
- Correct Anselmi references: 1801.00915, 1806.03605, 2201.00832
- BBH = Barnich-Brandt-Henneaux (hep-th/0002245) for local cohomology
- GPS = Gomis-Paris-Samuel for BV in higher-derivative gravity

## 4. Failed Approaches

1. **Index theory argument.** Attempted to prove BV-4 using the
   Atiyah-Singer index theorem, which controls chiral anomalies. The
   index theorem governs ABJ-type anomalies (axial U(1)), but BV
   anomalies are cohomological (BRST/antifield) and are not directly
   related to the index. The argument shows that the chiral anomaly
   is absent (since D^2 is chirality-even), but does not address the
   BV anomaly, which involves the antibracket structure.

2. **Explicit two-loop computation.** At two loops, the a_6 coefficient
   has 8 independent dimension-6 curvature invariants (including the
   Goroff-Sagnotti R^3 term). The counterterm was shown to be
   absorbable by delta(psi) (MR-4 result), confirming BV-3 at L = 2.
   But the method does not extend straightforwardly to L = 3 because
   the number of independent curvature invariants (P(4) = 3 quartic
   Weyl invariants from the Molien series) exceeds the number of
   spectral parameters (1 at each loop order). This is the content
   of the three-loop problem (OP-13).

3. **Formal power series in D^2.** Attempted to prove BV-3 by
   showing that the map g -> D^2 has a formal power series inverse
   (D^2 -> g) whose Jacobian has a well-defined formal determinant.
   The inverse exists formally by BV-2 (Connes reconstruction theorem),
   but the Jacobian determinant involves infinite sums over the
   spectrum of D^2, and the convergence of these sums at higher loops
   is not controlled.

## 5. Success Criteria

- A proof of BV-3 at all loop orders: the superdeterminant of the
  field redefinition g -> D^2 is well-defined as a formal power series
  in the coupling, with coefficients that are local spectral invariants.
- A proof of BV-4 at all loop orders: the BV antibracket anomaly
  vanishes to all orders in the spectral-action quantization scheme.
- Or: a specific loop order L_0 >= 3 where either BV-3 or BV-4 fails,
  with explicit identification of the non-spectral counterterm or
  anomaly.

## 6. Suggested Directions

1. **Cohomological argument for BV-4.** The BV anomaly lives in the
   local BRST cohomology H^1(s|d). For diffeomorphism-invariant
   theories in d = 4, this cohomology is known (Barnich-Brandt-Henneaux
   1994). Show that none of the nontrivial cohomology classes can arise
   from the spectral action, because the spectral action is a trace
   (hence invariant under inner automorphisms of the algebra A).

2. **Frechet regularity of g -> D^2.** BV-3 requires the map to be
   not just smooth but tame (in the sense of Nash-Moser), so that the
   implicit function theorem applies in the infinite-dimensional
   setting. Study the tame estimates for the Lichnerowicz formula
   D^2 = -(g^{mu nu} nabla_mu nabla_nu + E).

3. **Perturbative BV quantization in D^2 variables.** Formulate the
   BV quantization directly in D^2-space (bypassing the metric), using
   the antifield formalism of Batalin-Vilkovisky. In this setting,
   BV-3 is automatically satisfied (the variables ARE D^2), and BV-4
   reduces to the standard BV master equation for the spectral action.

4. **Numerical test at two loops.** Explicitly compute the two-loop
   effective action in D^2-variables on a non-trivial background
   (e.g., S^4 with radius a) and verify that the counterterm is a
   spectral invariant.

## 7. References

1. Anselmi, D. "On the quantum field theory of the gravitational
   interactions," JHEP 06 (2017) 086, arXiv:1704.07728.
2. Barnich, G., Brandt, F., and Henneaux, M. "Local BRST cohomology
   in gauge theories," Phys. Rep. 338 (2000) 439, hep-th/0002245.
3. Connes, A. "On the spectral characterization of manifolds,"
   J. Noncommut. Geom. 7 (2013) 1, arXiv:0810.2088.
4. Alfyorov, D. "Chirality of the Seeley-DeWitt coefficients and
   UV finiteness from D^2-quantization," DOI:10.5281/zenodo.19118075.

## 8. Connections

- **Blocks OP-06** (UV-completeness): All-orders UV finiteness
  (Theorem 6.11) is conditional on BV-3 and BV-4. Resolving OP-09
  would either establish or definitively exclude all-orders finiteness
  in the D^2 scheme.
- **Related to OP-10** (D^2 vs metric): If D^2 and metric
  quantization are inequivalent at loop level, OP-09 defines the
  self-consistency of the D^2 scheme as an independent theory.
- **Related to OP-13** (three-loop): At L = 3, BV-3 requires the
  three quartic Weyl counterterms to be expressible as spectral
  invariants of D^2. The overdetermination (3 invariants, 1 parameter)
  means BV-3 may fail at L = 3 unless a structural reduction exists.
- **Independent of OP-07, OP-08:** BV axioms concern the D^2-
  quantization route, not the fakeon route. If BV-3,4 hold but
  fakeon unitarity fails, D^2-quantization provides an alternative
  UV-finite theory.
