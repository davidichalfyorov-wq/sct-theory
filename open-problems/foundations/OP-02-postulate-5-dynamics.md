---
id: OP-02
title: "Postulate 5: dynamical principle for spectral triples"
domain: [theory]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: [OP-03, OP-05, OP-06]
blocked-by: []
roadmap-tasks: [MT-3]
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-02: Postulate 5 -- dynamical principle for spectral triples

## 1. Statement

Formulate and investigate the fifth postulate of SCT, which specifies
how spectral data determine spacetime evolution. Three candidate
formulations exist:

- **V1 (spectral action extremization):** spacetime geometry extremizes
  S = Tr(f(D²/Λ²)), with the spectral triple as the dynamical variable.
- **V2 (spectral flow dynamics):** evolution is determined by spectral
  flow of D under a one-parameter family of geometries.
- **V3 (path integral over spectral triples):** the quantum theory is
  defined by a path integral over the space of Dirac operators.

None of these has been investigated beyond a schematic level.

## 2. Context

Postulates 1--4 of SCT specify the kinematic framework: what a spectral
triple is, how the Standard Model is encoded, what the product geometry
is, and what observables are. Postulate 5 is the dynamical content --
the analogue of the Einstein-Hilbert action principle in GR or the
Feynman path integral in QFT. Without it, the theory makes predictions
only at the level of effective field theory (perturbative expansion
around a given background).

All current SCT results (form factors, field equations, predictions)
follow from the one-loop effective action computed via heat-kernel
expansion. This does not require Postulate 5; it only requires the
spectral action as a classical starting point. But UV-completeness
(OP-06), the non-perturbative definition (OP-03), and QFT recovery
(OP-05) all require a specified dynamical principle.

## 3. Known Results

- The spectral action S = Tr(f(D²/Λ²)) is well-defined for compact
  Riemannian spectral triples (Chamseddine-Connes 1996).
- The heat-kernel expansion gives the correct classical GR + SM
  Lagrangian at order Λ⁰ (established in the NCG literature).
- Spectral flow has been studied in the context of the index theorem
  and anomalies (Atiyah-Patodi-Singer) but not as a dynamical principle.
- Path integrals over matrix geometries (finite spectral triples) have
  been studied by Barrett, Glaser, and collaborators.

## 4. Failed Approaches

No systematic investigation has been attempted. The problem is at the
level of formulation, not computation.

## 5. Success Criteria

For any candidate formulation:
1. **Well-posedness:** the variational/evolution/path-integral problem
   is mathematically well-defined.
2. **Classical limit:** reproduces the Einstein equations (with SM
   matter) in the classical limit.
3. **New predictions:** produces at least one prediction beyond the
   one-loop effective action.
4. **Compatibility:** consistent with unitarity (MR-2) and causality
   (MR-3) requirements.

## 6. Suggested Directions

1. For V1: study the moduli space of Dirac operators compatible with
   a given spectral triple. Is the extremization problem well-posed?
   What are the Euler-Lagrange equations?
2. For V2: define spectral flow for Lorentzian spectral triples
   (Strohmaier, Paschke-Verch). Is the flow equation hyperbolic?
3. For V3: extend Barrett-Glaser matrix geometry path integrals to
   include gravitational degrees of freedom. Study the measure problem.
4. Compare with other spectral approaches: Landi-Rovelli spectral
   spacetime, Connes-van Suijlekom recent work on spectral truncations.

## 7. References

1. Chamseddine, Connes, "The spectral action principle," Comm. Math.
   Phys. 186 (1997) 731, hep-th/9606001.
2. Barrett, Glaser, "Monte Carlo simulations of random non-commutative
   geometries," J. Phys. A 49 (2016) 245001, arXiv:1510.01377.
3. Paschke, Verch, "Local covariant quantum field theory over spectral
   geometries," Class. Quant. Grav. 21 (2004) 5299, gr-qc/0405057.
4. Connes, van Suijlekom, "Spectral truncations in noncommutative
   geometry and operator systems," arXiv:2004.14115.

## 8. Connections

- **Blocks OP-03** (non-perturbative definition): V3 is the natural
  route to non-perturbative quantum gravity.
- **Blocks OP-05** (QFT recovery): need dynamics to verify semiclassical
  limit reproduces standard QFT.
- **Blocks OP-06** (UV-completeness): UV behavior depends on the
  dynamical principle (renormalization group structure).
- Independent of all one-loop results (NT-1 through NT-4, MR-2 through
  MR-7), which use the effective action without Postulate 5.
