---
id: OP-50
title: "Remaining Lean 4 formalisation targets"
domain: [formal-verification, theory]
difficulty: medium-hard
status: open
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: []
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-50: Remaining Lean 4 formalisation targets

## 1. Statement

Complete the formal verification of five key SCT results that have
been proven analytically and verified numerically but not yet
formalised in Lean 4: (a) spectral positivity of Pi_entire, (b)
CHIRAL-Q chirality identity at higher loops, (c) NT-4b nonlinear
field equations, (d) angular integral over S^2, and (e) the two
remaining sorry statements in existing proof files.

## 2. Context

The SCT Lean 4 formalisation covers 41 canonical identities across
46 proof files, with 44 files sorry-free. This provides formal
verification for the algebraic core of the theory: coefficient
values, local limits, multiplicities, mass relations, and the CJ
bridge framework.

However, five categories of result remain unformalised. These range
from straightforward (closing 2 sorry statements in existing files)
to infrastructure-intensive (formalising functional derivatives on
the space of metrics). Each unformalised result is a potential point
of failure: an error in any of these would propagate through the
theory.

The formalisation is performed using three backends: local
(PhysLean + Mathlib4), Aristotle (cloud prover), and WSL SciLean.
All critical results require dual verification (at least two
backends confirming the proof).

## 3. Known Results

### (a) Spectral positivity: Pi_entire(z) > 0 for real z >= 0

- Verified numerically to z = 100 using 50-digit mpmath arithmetic.
- Pi_entire(z) is the product Pi_TT(z) x Pi_s(z, xi) restricted to
  the real positive axis. Its positivity ensures that the dressed
  graviton propagator has the correct sign for physical (on-shell)
  momenta.
- The analytic argument: Pi_TT(z) = 1 + (13/60) z F_hat_1(z) where
  F_hat_1 is positive for real z >= 0 (as a sum of positive parametric
  integrals). So Pi_TT(z) > 1 for z > 0. Similarly for Pi_s at
  xi = 1/6 (Pi_s = 1).
- Lean status: NOT formalised. The positivity of F_hat_1 for real
  z >= 0 requires formalising the parametric integral
  int_0^1 xi(1-xi) e^{-xi(1-xi) z} d xi, which involves
  transcendental functions not yet in Mathlib4.

### (b) CHIRAL-Q at higher loops

- The chirality identity {D, gamma_5} = 0 forces perturbative
  counterterms to be block-diagonal, giving the divergence degree
  D = 0 at every loop order in D^2-quantization.
- Proven unconditionally at L = 1 and L = 2. At all orders,
  conditional on BV axioms BV-3 and BV-4 (verified at one loop).
- Lean status: formalised at one loop only. The higher-loop extension
  requires formalising the BV-BRST cohomology (antifield formalism,
  graded differential algebras). PhysLean does not currently support
  this.

### (c) NT-4b nonlinear field equations

- The full nonlinear SCT field equations G_{mu nu} + Theta^{(R)}_{mu nu}
  + Theta^{(C)}_{mu nu} = 0 have been derived by dual methods (BV
  alpha-insertion and direct functional differentiation).
- Lean status: NOT formalised. Requires formalising:
  (i) functional derivatives delta/delta g^{mu nu} on the space of
  Lorentzian metrics,
  (ii) the heat kernel expansion Tr(f(D^2/Lambda^2)) as a functional
  of g,
  (iii) variation of the Weyl and Ricci tensors under metric
  perturbation.
  This is a major infrastructure project.

### (d) Angular integral over S^2

- The integral int_{S^2} d Omega f(theta, phi) appears in the
  derivation of form factor local limits, Newtonian potential, and
  PPN parameters.
- Only the rational skeleton has been formalised: combinatorial
  identities such as sum_{l=0}^L (2l+1) P_l(cos theta) = ...
  that involve only rational arithmetic.
- The full angular integral with transcendental integrands
  (exp, erfi, Gaussian) requires Mathlib4 integration theory for
  functions on the sphere, which is available in principle
  (MeasureTheory.Measure.haar) but has not been specialised to S^2.

### (e) Two remaining sorry statements

- Two proof files contain sorry at intermediate lemma steps.
  These are typically steps where the Lean elaborator times out
  on a large term or where a specific Mathlib4 lemma is missing.
- Aristotle has been unable to fill these sorry statements
  automatically, suggesting they require manual proof construction
  or a Mathlib4 contribution.

## 4. Failed Approaches

1. **Aristotle for spectral positivity.** Submitted the positivity
   statement to Aristotle. The prover timed out after several hours.
   The difficulty is that the statement involves a continuous function
   (parametric integral) and Aristotle's tactics are strongest for
   algebraic and combinatorial goals.

2. **SciLean for angular integrals.** SciLean has numerical integration
   but not symbolic integration over manifolds. The angular integral
   formalisation requires symbolic, not numerical, proof.

3. **Direct sorry elimination.** Attempted to fill the two remaining
   sorry statements by manual proof construction. One involves a
   chain of inequalities with 8 intermediate steps; the other
   involves a specialised Mathlib4 lemma about convergent series
   that does not exist yet.

## 5. Success Criteria

- **(a)** Lean 4 proof of Pi_entire(z) > 0 for all real z >= 0
  (or for z in [0, z_max] with z_max >= 100).
- **(b)** Lean 4 formalisation of the CHIRAL-Q identity at L = 2
  (extending the current L = 1 proof).
- **(c)** Lean 4 formalisation of the linearised field equations
  (the full nonlinear version is a long-term goal; the linearised
  version is a realistic milestone).
- **(d)** Lean 4 proof of at least one angular integral identity
  used in the form factor derivation (e.g., int_{S^2} n_i n_j dOmega
  = (4 pi / 3) delta_{ij}).
- **(e)** All 46 proof files sorry-free (0 remaining sorry statements).
- Dual verification: each new proof must be confirmed by at least
  two backends (local + Aristotle or local + SciLean).

## 6. Suggested Directions

1. **Spectral positivity via interval arithmetic.** Instead of
   proving positivity for all z >= 0 symbolically, partition [0, 100]
   into intervals and prove Pi_TT > 0 on each interval using
   rigorous interval arithmetic bounds. Mathlib4 has interval
   arithmetic support. This converts the transcendental problem
   into finitely many algebraic problems.

2. **CHIRAL-Q via graded algebra.** Formalise a minimal graded
   differential algebra sufficient for the BV-BRST argument. This
   does not require the full antifield formalism -- only the
   filtration by ghost number and the block-diagonal structure of
   the graded commutator {D, gamma_5} = 0.

3. **Linearised field equations.** Formalise the linearised version
   h_{mu nu} -> G^{lin}_{mu nu}[h] + Theta^{lin}_{mu nu}[h] = 0
   on flat background. This avoids the functional derivative
   infrastructure and reduces to algebraic manipulation of momentum-
   space expressions.

4. **Angular integral via Haar measure.** Use the Mathlib4
   MeasureTheory.Measure.haar formalism to define the uniform
   measure on S^2 and prove the identity int_{S^2} n_i n_j dOmega
   = (4 pi / 3) delta_{ij} by symmetry (rotation invariance implies
   the integral is proportional to delta_{ij}, and the trace
   determines the coefficient).

5. **Sorry elimination.** For the two remaining sorry statements,
   identify the exact Mathlib4 lemma gaps. If a missing lemma is
   small, contribute it to Mathlib4 upstream. If the sorry is due
   to elaboration timeout, refactor the proof into smaller lemmas
   with explicit type annotations.

## 7. References

1. Mathlib4 documentation. https://leanprover-community.github.io/mathlib4_docs/
2. PhysLean documentation. (project-internal)
3. Aristotle automated theorem prover documentation.
4. de Moura, L. and Ullrich, S. (2021). "The Lean 4 theorem prover
   and programming language." CADE-28, LNCS 12699, 625-635.
5. Alfyorov, D. "Nonlocal one-loop form factors from the spectral
   action." DOI:10.5281/zenodo.19098042.

## 8. Connections

- Spectral positivity **(a)** is the formal verification counterpart
  of the numerical verification in NT-2 and the unitarity analysis
  in MR-2.
- CHIRAL-Q **(b)** relates to **OP-09** (BV axiom verification) and
  **OP-10** (CHIRAL-Q extension to metric quantization).
- NT-4b field equations **(c)** relate to **OP-01** (Gap G1): the
  formal verification of the field equations would also formalise
  the structure of Theta^(C).
- Angular integral **(d)** is a prerequisite for formalising the
  local-limit derivations in NT-1 and NT-1b.
- Sorry elimination **(e)** is independent of all other problems.
