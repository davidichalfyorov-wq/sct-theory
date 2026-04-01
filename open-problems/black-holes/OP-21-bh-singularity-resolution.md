---
id: OP-21
title: "Black hole singularity resolution via nonlocal field equations"
domain: [black-holes, field-equations]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: []
blocked-by: [OP-01]
roadmap-tasks: [MR-9]
papers: ["1508.00869", "2012.11829"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-21: Black hole singularity resolution

## 1. Statement

Solve the nonlinear nonlocal SCT field equations

  G_{mu nu} + Theta^(R)_{mu nu} + Theta^(C)_{mu nu} = 0

on a static spherically symmetric ansatz

  ds^2 = -A(r) dt^2 + B(r) dr^2 + r^2 d Omega^2

and determine whether the Schwarzschild singularity at r = 0 is
resolved. Specifically: (a) determine the geometry near r = 0 (de
Sitter core, bounce, or other regular structure); (b) verify that the
Kretschmer scalar K = R_{mu nu rho sigma} R^{mu nu rho sigma} is finite
everywhere; (c) identify which Penrose energy condition is violated by
the nonlocal corrections.

## 2. Context

The modified Newtonian potential V(r)/V_N(r) = 1 - (4/3) e^{-m_2 r}
+ (1/3) e^{-m_0 r} is finite at r = 0, with V(0) = 0. This is a
linearised result (NT-4a), valid only for weak gravitational fields.
Near a black hole singularity, the curvature is strong and the full
nonlinear equations must be used.

The nonlinear field equations are known formally (NT-4b): the
Ricci-sector correction Theta^(R) has been computed explicitly, but
the Weyl-sector correction Theta^(C) has not been evaluated on
backgrounds with non-zero Weyl curvature. This is Gap G1 (OP-01).

Without Theta^(C), the field equations on Schwarzschild are incomplete.
Any claimed singularity resolution that ignores the Weyl sector is
unreliable, because Schwarzschild has maximal Weyl curvature
(C_{mu nu rho sigma} C^{mu nu rho sigma} = 48 M^2 / r^6).

## 3. Known Results

- **Linearised potential (NT-4a):** V(0) = 0, finite everywhere.
  Yukawa masses m_2 = 2.148 Lambda, m_0 = 2.449 Lambda.

- **Formal nonlinear equations (NT-4b):** G_{mu nu} + Theta^(R)
  + Theta^(C) = 0, where Theta^(R) involves F_2(Box/Lambda^2) acting
  on H_{mu nu}, and Theta^(C) involves F_1(Box/Lambda^2) acting on
  B_{mu nu} (Bach tensor).

- **FLRW limit (NT-4c):** on conformally flat backgrounds, Theta^(C)
  = 0 and the equations reduce to modified Friedmann equations with
  de Sitter as an exact solution. This provides no information about
  Schwarzschild.

- **Other nonlocal gravity approaches:** Modesto (2012) and
  Biswas-Mazumdar-Siegel (2012) showed that specific entire-function
  modifications of GR produce singularity-free Schwarzschild solutions
  with de Sitter cores. The SCT form factors F_1 and F_2 are also
  entire, suggesting a similar outcome, but the detailed structure
  differs.

- **Energy condition violation:** the linearised potential has
  V'(0) = 0, implying a repulsive gravitational force at short
  distances. This violates the strong energy condition. Whether the
  nonlinear solution violates only the SEC or also the NEC is unknown.

## 4. Failed Approaches

No explicit attempts have been made on the nonlinear problem because
Gap G1 (OP-01) blocks the computation. Any attempt that ignores
Theta^(C) is incomplete and potentially misleading.

A perturbative approach in M/r (post-Newtonian expansion) was
considered but abandoned because it breaks down precisely in the
strong-field region r ~ M where singularity resolution is relevant.

## 5. Success Criteria

- **Theta^(C) on Schwarzschild:** explicit evaluation of the
  Weyl-sector correction at r = 2M, 3M, 10M (OP-01 prerequisite).
- **Near-r=0 geometry:** determine whether the metric functions A(r)
  and B(r) approach de Sitter form (A ~ 1 - r^2/l^2, B ~ 1/(1 - r^2/l^2))
  or some other regular structure as r -> 0.
- **Kretschmer scalar:** verify K < infinity for all r >= 0.
- **Energy condition:** identify the minimal Penrose condition violated.
- **Mass dependence:** determine how the core radius (if de Sitter)
  depends on M and Lambda.

## 6. Suggested Directions

1. **Perturbative expansion in curvature.** Expand the nonlocal
   operator F_1(Box) on Schwarzschild using the Barvinsky-Vilkovisky
   covariant perturbation theory, keeping commutators [Box, C] to the
   necessary order. This is feasible but technically demanding.

2. **Numerical integration.** Discretize Box on a radial grid, compute
   F_1(Box) as a matrix function, and solve the resulting system of
   ODEs for A(r) and B(r) numerically. This requires a suitable
   regularization of the operator at r = 0.

3. **Comparison with Modesto-Biswas-Mazumdar-Siegel.** The BMS
   framework uses exp(-Box/Lambda^2) as the entire function. The SCT
   form factors have a different functional form (involving erfi). Study
   whether the qualitative features (de Sitter core, mass gap) survive
   the change of entire function.

4. **Effective metric approach.** Use the dressed propagator
   G_TT(k^2) = 1/(k^2 Pi_TT(k^2/Lambda^2)) to construct an effective
   metric in momentum space, Fourier transform to position space, and
   study the resulting geometry near r = 0. This bypasses the full
   nonlinear equations but captures the leading corrections.

## 7. References

1. Modesto, L. (2012). "Super-renormalizable quantum gravity." Phys.
   Rev. D 86, 044005. arXiv:1107.2403.
2. Biswas, T., Mazumdar, A. and Siegel, W. (2006). "Bouncing
   universes in string-inspired gravity." JCAP 03, 009.
   arXiv:hep-th/0508194.
3. Frolov, V. P. (2016). "Notes on nonsingular models of black
   holes." Phys. Rev. D 94, 104056. arXiv:1508.00869.
4. Alfyorov, D. (2026). "Nonlinear field equations and FLRW cosmology
   from the spectral action." DOI:10.5281/zenodo.19098027.
5. Chamseddine, A. H., Connes, A. and van Suijlekom, W. D. (2020).
   "Entropy and the spectral action." arXiv:2012.11829.

## 8. Connections

- **Blocked by OP-01 (Gap G1):** the Weyl-sector variational
  correction Theta^(C) must be computed on curved backgrounds before
  the nonlinear equations can be solved on Schwarzschild.
- **OP-22 (second law):** singularity resolution affects the causal
  structure of the black hole interior, which in turn affects the
  second law (area increase of the event horizon).
- **OP-23 (information paradox):** a regular interior changes the
  information loss problem qualitatively.
- **OP-07, OP-08 (ghost resolution):** the ghost sector contributes to
  the effective metric near r = 0 and may either help or hinder
  singularity resolution depending on the fakeon prescription.
