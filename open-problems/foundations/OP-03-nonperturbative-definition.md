---
id: OP-03
title: "Non-perturbative definition of quantum SCT"
domain: [theory]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: [OP-06]
blocked-by: [OP-02]
roadmap-tasks: [MR-8]
papers: ["1510.01377", "2007.09726"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-03: Non-perturbative definition of quantum SCT

## 1. Statement

Define quantum SCT non-perturbatively via a path integral over Dirac
operators. This requires: (i) specifying the space of Dirac operators
that serve as gravitational configurations, (ii) constructing a
diffeomorphism-invariant measure on this space, and (iii) demonstrating
that discretized approximations (matrix geometries with finite-rank
Dirac operators D_N) converge to the continuum theory in a controlled
limit. The central test is whether Tr(f(D_N^2/Lambda^2)) reproduces
known causal set actions (Benincasa-Dowker) in the continuum limit.

## 2. Context

All current SCT results derive from the one-loop effective action
computed via heat-kernel expansion around a fixed background geometry.
This perturbative framework suffices for form factors, field equations,
and phenomenological predictions, but cannot address genuinely
non-perturbative questions: topology change, spacetime foam, the
measure problem, or the UV fate of the theory beyond perturbation
theory.

The spectral action Tr(f(D^2/Lambda^2)) is well-defined for compact
Riemannian spectral triples (Chamseddine-Connes 1996), and path
integrals over finite spectral triples have been studied by Barrett,
Glaser, and collaborators. The FND-1 numerical program has provided
evidence that discrete spectral operators can detect spacetime curvature
(commutator Cohen's d = +0.59 for pp-wave, link Laplacian d = -1.18),
and the CJ bridge formula has been verified to N = 15000. These results
suggest that a non-perturbative formulation based on matrix geometries
is viable, but no rigorous construction exists.

## 3. Known Results

- Path integrals over matrix geometries (N x N matrices) have been
  studied by Barrett-Glaser (2016), producing phase transitions and
  geometrically interesting configurations at large N.
- Glaser-Stern (2020-2021) studied convergence of spectral truncations,
  showing that finite matrix approximations to the Dirac operator on
  S^2 converge in a suitable sense.
- The Benincasa-Dowker action for causal sets in d = 2 and d = 4 is
  known explicitly. In d = 4 it involves specific combinatorial
  coefficients over order intervals.
- FND-1 numerical results: the commutator [C, C^T] of the causal
  matrix detects pp-wave curvature at Cohen's d = +0.59 (N = 5000,
  M = 40 sprinklings). Link-graph Laplacian eigenvalues separate
  pp-wave from flat at d = -1.18.
- The CJ bridge formula connecting path-kurtosis excess to curvature
  is verified numerically to N = 15000.
- The spectral action on S^4 has been computed exactly by Iochum,
  Levy, and Vassilevich, providing a benchmark for any discretization.

## 4. Failed Approaches

1. **Direct Lebesgue measure on the space of metrics.** The
   DeWitt-Faddeev-Popov construction requires gauge-fixing and
   produces a measure that is not diffeomorphism-invariant without
   ghost fields. In the spectral approach the analogous problem is
   defining a measure on the space of Dirac operators modulo unitary
   equivalence. No satisfactory resolution exists.

2. **Euclidean lattice discretization.** Standard lattice gravity
   (Regge calculus, dynamical triangulations) does not preserve the
   spectral triple structure. The Dirac operator on a simplicial
   complex does not have the same spectral properties as the continuum
   operator, and the convergence of spectral invariants is not
   established in d = 4.

3. **Random matrix theory.** Treating D_N as a random matrix from
   a standard ensemble (GUE, GOE) destroys the geometric content.
   The matrix entries of D_N are constrained by the spectral triple
   axioms (first-order condition, real structure), which reduce the
   effective dimension of the configuration space far below N^2.

## 5. Success Criteria

- A well-defined measure mu on the space of Dirac operators (or on a
  suitable discretized approximation thereof) that is invariant under
  the automorphism group of the spectral triple.
- Proof that expectation values of spectral invariants converge as
  N -> infinity.
- Demonstration that Tr(f(D_N^2/Lambda^2)) reproduces the Benincasa-Dowker
  action (or a known causal set action) in the continuum limit on
  Minkowski, de Sitter, and Schwarzschild backgrounds.
- Verification that the non-perturbative partition function
  Z = integral dmu(D) exp(-Tr(f(D^2/Lambda^2))) is well-defined
  (convergent) for suitable cutoff functions f.

## 6. Suggested Directions

1. **Barrett-Glaser matrix geometries with constraints.** Extend the
   Barrett-Glaser Monte Carlo to incorporate the full spectral triple
   axioms (real structure, first-order condition, grading) and measure
   convergence of spectral invariants as N increases.

2. **Spectral truncation approach (Connes-van Suijlekom).** Use the
   operator system framework to define truncations of the algebra A
   and study the induced truncation of the Dirac operator. The question
   is whether the truncated spectral action converges.

3. **Connection to causal set theory.** The FND-1 results suggest
   that the causal matrix C encodes geometric information. Investigate
   whether the Dirac operator can be reconstructed from the causal
   order (building on the work of Sorkin, Yazdi, and collaborators on
   the d'Alembertian on causal sets).

4. **Functional renormalization group (FRG).** Apply the Wetterass
   equation to the spectral action, using the eigenvalues of D^2 as
   the flow variable. If a UV fixed point exists, the non-perturbative
   theory is defined by the fixed-point action.

5. **Numerical convergence study.** Systematically compute
   Tr(f(D_N^2)) for matrix geometries on S^4 at N = 100, 500, 2000,
   10000 and compare with the exact Iochum-Levy-Vassilevich result.

## 7. References

1. Barrett, Glaser, "Monte Carlo simulations of random non-commutative
   geometries," J. Phys. A 49 (2016) 245001, arXiv:1510.01377.
2. Glaser, Stern, "Reconstructing manifolds from truncations of
   spectral triples," J. Geom. Phys. 159 (2021) 103921,
   arXiv:2007.09726.
3. Chamseddine, Connes, "The spectral action principle," Comm. Math.
   Phys. 186 (1997) 731, hep-th/9606001.
4. Benincasa, Dowker, "The scalar curvature of a causal set," Phys.
   Rev. Lett. 104 (2010) 181301, arXiv:1001.2725.
5. Connes, van Suijlekom, "Spectral truncations in noncommutative
   geometry and operator systems," arXiv:2004.14115.
6. Iochum, Levy, Vassilevich, "Spectral action beyond the weak-field
   approximation," Comm. Math. Phys. 316 (2012) 595, arXiv:1108.3749.

## 8. Connections

- **Blocked by OP-02** (Postulate 5): the non-perturbative definition
  requires choosing a dynamical principle (V3 -- path integral over
  spectral triples is the natural candidate).
- **Blocks OP-06** (UV-completeness): a non-perturbative definition
  is prerequisite for establishing UV finiteness beyond perturbation
  theory.
- Related to **OP-34** (N-scaling exponent): the scaling of discrete
  observables with N is a convergence diagnostic for the matrix
  geometry approximation.
- Independent of all perturbative results (NT-1 through NT-4, MR-2
  through MR-7), which do not require a non-perturbative formulation.
