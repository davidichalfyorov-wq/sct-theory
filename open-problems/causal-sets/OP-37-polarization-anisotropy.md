---
id: OP-37
title: "Polarization anisotropy tensor of CJ"
domain: [theory, numerics]
difficulty: medium
status: open
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: []
papers: [1904.01034]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-37: Polarization anisotropy tensor of CJ

## 1. Statement

The CJ observable depends on the STRUCTURE of the electric Weyl tensor
E_ij, not merely on its squared magnitude E^2 = E_ij E^ij. For a
linearized gravitational wave with two polarization states:

  cross-polarization (h = 2 A x y):    CJ_cross = K_cross E^2
  plus-polarization  (h = A(x^2-y^2)): CJ_plus  = K_plus  E^2

the measured ratio is K_cross / K_plus = 3.0 +/- 0.15 (N = 5000,
M = 50, jet predicate). The angular integral (8 pi / 15) E^2 assumes
isotropy, predicting K_cross / K_plus = 1. Derive the anisotropy ratio
from the geometry of causal diamonds in anisotropic backgrounds, or
identify the tensor structure that replaces the isotropic angular
integral.

## 2. Context

The standard angular integral identity

  integral (E_ij n^i n^j)^2 d Omega = (8 pi / 15) E^2

averages over ALL directions n on the 2-sphere. This average is
appropriate when the causal diamond is spherically symmetric (as in
the isotropic limit). However, a causal diamond embedded in a
gravitational-wave background is NOT spherically symmetric: the wave
stretches the diamond along one axis and compresses it along the
perpendicular axis.

The anisotropy modifies the effective solid angle sampled by chains
in the diamond. Chains preferentially sample directions where the
diamond is wider, overweighting some components of E_ij relative to
others.

Numerically, the anisotropy tensor K_ijkl enters as:

  CJ = K_ijkl E^ij E^kl

with K_1212 / K_1111 = 1.5 (measured), compared to the isotropic
prediction K_1212 / K_1111 = 0.5. The factor-of-3 enhancement of
the cross-polarization comes from the additional off-diagonal
coupling.

## 3. Known Results

- **Isotropic limit:** K_ijkl = (8 pi / 15) (delta_ik delta_jl +
  delta_il delta_jk - (2/3) delta_ij delta_kl) / (2 E^2). This
  gives K_cross / K_plus = 1.0.
- **Measured ratio (jet predicate):** K_cross / K_plus = 3.0 +/- 0.15
  at N = 5000, M = 50. Stable across N = 2000 to 10000.
- **Measured ratio (ball predicate):** K_cross / K_plus = 1.15 +/- 0.08.
  Nearly isotropic. The ball predicate averages over all directions
  by construction, washing out the anisotropy.
- **K_1212 / K_1111 = 1.5:** Directly measured by comparing the
  kurtosis excess in the 12-plane vs the 11-direction.
- **Wang (1904.01034):** Area-counting in curved diamonds gives
  delta_A / A proportional to E_ij n^i n^j with direction-dependent
  coefficient. No treatment of the second moment (relevant for CJ).

## 4. Failed Approaches

1. **Perturbative deformation of the diamond boundary.** Expanded the
   boundary of the causal diamond to first order in h_ij (the metric
   perturbation). The first-order correction to the volume is
   proportional to R (Ricci), which vanishes in vacuum. The Weyl
   contribution appears only at second order, but the second-order
   boundary deformation requires solving the geodesic deviation
   equation, which was only done numerically.

2. **SO(3) decomposition of the jet predicate.** Decomposed the jet
   predicate (quadratic RNC polynomial) into l = 0, 2, 4 spherical
   harmonics. The l = 4 component is non-zero and contributes a
   polarization-dependent correction. However, the coefficient of the
   l = 4 term depends on the predicate choice, not on the geometry.
   This shows the ratio is predicate-dependent, but does not explain
   the specific value of 3.0.

3. **Symmetry argument.** Argued that cross and plus polarizations
   are related by a 45-degree rotation, which should leave E^2
   invariant. This is correct for the isotropic integral but does not
   apply to the diamond-averaged integral, because the diamond shape
   is NOT rotationally invariant in the transverse plane.

## 5. Success Criteria

- Derive K_ijkl as a function of the diamond geometry (proper-time
  height T, spatial extent, predicate choice).
- For the jet predicate: predict K_cross / K_plus = 3.0 +/- 0.3.
- For the ball predicate: predict K_cross / K_plus approximately 1.
- Explain the predicate dependence: why does the jet predicate
  introduce anisotropy while the ball predicate does not?
- Express the result as K_ijkl = K_iso_ijkl + K_aniso_ijkl(P) where
  P parametrizes the predicate.

## 6. Suggested Directions

1. **Effective solid-angle measure.** Define an effective measure
   d Omega_eff on the 2-sphere that weights directions by their
   representation in the jet predicate. The anisotropy ratio is then
   integral (E_ij n^i n^j)^2 d Omega_eff / integral E^2 d Omega.
   Compute d Omega_eff for the quadratic RNC jet.

2. **Eigenvalue decomposition.** Diagonalize E_ij. In the eigenframe,
   cross-polarization has E_12 = E_21 = A (off-diagonal), and
   plus-polarization has E_11 = -E_22 = A (diagonal). The diamond
   geometry in the eigenframe is different for the two cases. Compute
   the chain-length variance in each eigenframe.

3. **Numerical scan over diamond orientations.** Rotate the diamond
   axis relative to the wave propagation direction. Measure CJ as a
   function of the orientation angle theta. Fit K_ijkl from the
   angular dependence.

4. **2D model problem.** In d = 2, there is only one polarization
   direction. Solve the anisotropy problem exactly in 2D and lift
   the result to 4D using dimensional continuation.

## 7. References

- Wang, Z. (2019). "Causal set d'Alembertian in curved spacetime."
  arXiv:1904.01034.
- Dowker, F. and Glaser, L. (2013). "Causal set d'Alembertians for
  various dimensions." arXiv:1305.2588.
- Aslanbeigi, S. et al. (2014). "Generalized causal set
  d'Alembertians." arXiv:1403.1622.

## 8. Connections

- **OP-36 (RSY derivation):** The isotropic angular integral
  (8 pi / 15) E^2 used in OP-36 is the isotropic limit of the
  full anisotropy tensor K_ijkl. If K_ijkl is predicate-dependent,
  the RSY derivation must specify which predicate it uses.
- **OP-39 (exact Schwarzschild predicate):** An exact predicate
  would have different anisotropy properties from the jet predicate,
  potentially resolving the ratio discrepancy.
- **OP-42 (A_E universality):** The A_E ratio between geometries
  may depend on K_ijkl rather than on A_E alone.
