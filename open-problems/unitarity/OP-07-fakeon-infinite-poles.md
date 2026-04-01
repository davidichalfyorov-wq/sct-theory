---
id: OP-07
title: "Fakeon prescription for infinite-pole propagators"
domain: [unitarity]
difficulty: hard
status: open
deep-research-tier: A
blocks: [OP-22]
blocked-by: []
roadmap-tasks: [MR-2]
papers: ["1704.07728", "2308.09006", "1801.00915"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-07: Fakeon prescription for infinite-pole propagators

## 1. Statement

Prove that the fakeon (purely virtual particle) prescription for the
SCT graviton propagator 1/(k^2 Pi_TT(z)), which has infinitely many
poles from the entire function Pi_TT(z), yields a unitary S-matrix
at all loop orders. Alternatively, identify a concrete obstruction
that prevents such an extension.

## 2. Context

The fakeon prescription was introduced by Anselmi (2017) as an
alternative to the Feynman i*epsilon prescription at ghost poles.
Instead of propagating ghost particles as physical asymptotic states,
the fakeon prescription assigns zero imaginary part to the propagator
at ghost poles, effectively removing ghost pair production.

For polynomial propagators (finitely many poles), the prescription is
rigorously proven to preserve unitarity: the cutting equations for
the S-matrix are modified so that cuts through ghost lines give zero
contribution, and the optical theorem Im[T] >= 0 holds. This covers
Stelle gravity (fourth-derivative, 3 propagator poles) and its
higher-derivative extensions with any finite number of poles.

The SCT propagator is qualitatively different: Pi_TT(z) is entire of
order 1 and has infinitely many zeros. The Mittag-Leffler expansion
of 1/(z Pi_TT) is an infinite partial-fraction series. The fakeon
prescription must be applied to each pole individually, and the limit
as the number of poles N -> infinity must be shown to commute with
the prescription and preserve unitarity.

## 3. Known Results

- **CL commutativity (CERTIFIED):** The limit N -> infinity commutes
  with the fakeon principal-value prescription at the level of the
  one-loop amplitude. Proven via Weierstrass M-test with total bound
  Sigma M_n = 5.002e-4 (0.32% of total amplitude). Two-pole dominance
  holds: 99.68% of amplitude from z_L = -1.2807 and z_0 = 2.4148.

- **GZ constancy (CERTIFIED):** The entire part g_A(z) = -13/60 is
  constant (no polynomial growth), proven via Hadamard genus argument.
  Sum rule Sigma R_n / z_n = 13/60 verified to 99.95%.

- **One-loop optical theorem (CERTIFIED CONDITIONAL):** Im[T(s)] >= 0
  at all tested energies. Spectral positivity Im[G_dressed] > 0
  proven algebraically. Fakeon ghost cut removal verified.

- **Anselmi polynomial proof:** For polynomial propagators of any
  degree, the fakeon prescription gives unitary amplitudes (proven via
  modified cutting equations and the largest-time equation in
  Minkowski space). The key property used: finitely many poles allow
  explicit contour deformation and principal-value evaluation.

## 4. Failed Approaches

1. **Direct extension of Anselmi's proof.** The polynomial proof
   uses explicit contour deformation around a finite set of poles.
   For infinitely many poles accumulating at infinity, the contour
   deformation becomes an infinite sequence of deformations, and
   the convergence of the resulting series is not controlled by the
   existing proof strategy. The approach does not fail definitively
   but lacks the estimates needed for the infinite-pole case.

2. **Functional-integral derivation.** Attempting to define the
   fakeon prescription via a modified path integral measure
   (restricting the domain of integration to exclude ghost modes)
   encounters the problem that the ghost modes are dense in the
   spectrum of D^2 and cannot be cleanly separated from physical
   modes in a nonperturbative functional integral.

3. **Borel summation of the ML series.** The Mittag-Leffler series
   for 1/(z Pi_TT) converges absolutely (established by GZ), so
   Borel summation is not needed for convergence. However, the
   question is about unitarity of the S-matrix, not convergence of
   the propagator series -- the cutting equations involve products
   of propagators, and the convergence of product series is not
   automatic.

## 5. Success Criteria

- A rigorous proof that the S-matrix defined by the fakeon
  prescription on the full SCT propagator satisfies the optical
  theorem Im[T(s)] >= 0 at all orders in the loop expansion.
- Alternatively: a proof that the N-pole truncated S-matrix
  converges to the infinite-pole S-matrix uniformly in the
  external momenta, combined with the polynomial unitarity result.
- Or: identification of a specific loop order and kinematic
  configuration where ghost pair production is nonzero despite the
  fakeon prescription.

## 6. Suggested Directions

1. **Uniform convergence of cutting equations.** The cutting equations
   for the N-pole truncation are known to give Im[T_N] >= 0. Show
   that Im[T_N] -> Im[T_infinity] uniformly in external momenta,
   using the Weierstrass M-test bounds from CL as input.

2. **Spectral representation approach.** Since rho_TT is a pure sum
   of delta functions (no branch cuts from entire Pi_TT), attempt
   to construct a spectral representation for the full S-matrix and
   verify positivity of the spectral measure after fakeon exclusion.

3. **Axiomatic S-matrix route.** Use the Bogoliubov-Medvedev-Polivanov
   axioms (unitarity, causality, Lorentz invariance) and show that
   the fakeon-modified propagator satisfies all axioms. The acausal
   component has decay length 0.884/Lambda (from MR-3), which may
   be sufficient for the micro-local spectrum condition.

4. **Nonperturbative lattice test.** Discretize the theory on a
   4-torus, compute the partition function with fakeon-modified
   propagator, and verify reflection positivity (the Euclidean
   analogue of unitarity) numerically for small lattice sizes.

## 7. References

1. Anselmi, D. "On the quantum field theory of the gravitational
   interactions," JHEP 06 (2017) 086, arXiv:1704.07728.
2. Anselmi, D. "Fakeons and Lee-Wick models," JHEP 02 (2018) 141,
   arXiv:1801.00915.
3. Kubo, J. and Kugo, T. "Unitarity and higher-order gravitational
   scattering," arXiv:2308.09006.
4. Anselmi, D. and Piva, M. "A new formulation of Lee-Wick quantum
   field theory," JHEP 06 (2017) 066, arXiv:1703.04584.
5. Alfyorov, D. "Nonlocal one-loop form factors from the spectral
   action," DOI:10.5281/zenodo.19098042.

## 8. Connections

- **Blocks OP-22** (BH entropy second law): The fakeon prescription
  enters the definition of the graviton propagator near the horizon,
  and BH thermodynamics requires knowing whether ghost modes
  contribute to the entropy.
- **Blocked by nothing:** This is a root-level problem. Its resolution
  would upgrade MR-2 from CONDITIONAL to COMPLETE.
- **Related to OP-08** (all-orders KK resolution): OP-08 requires
  OP-07 as input because the operator-formalism proof of ghost
  decoupling depends on the fakeon prescription being well-defined.
- **Related to OP-09** (BV axioms): If fakeon unitarity fails, the
  CHIRAL-Q UV-finiteness result becomes the primary unitarity route,
  making OP-09 critical.
