---
id: OP-47
title: "Third vacuum family and d = 3 dimensional test"
domain: [numerics, causal-sets]
difficulty: medium
status: open
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: []
papers: ["1212.0631"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-47: Third vacuum family and d = 3 dimensional test

## 1. Statement

Extend the CJ estimator tests to a third vacuum family beyond pp-wave
and Schwarzschild (candidates: Godel, Bianchi IX, or Kasner). Separately,
test the CJ bridge formula in d = 3 spacetime dimensions, where the
current d = 2 test shows a factor 3.6 discrepancy with the predicted
coefficient.

## 2. Context

The GTA framework has been validated on two vacuum families:

1. **PP-wave** (Type N in Petrov classification). Exact causal
   predicate available. T^4 scaling confirmed. A_E ~ 0.036.
   Purely electric Weyl.

2. **Schwarzschild** (Type D in Petrov classification). Jet predicate
   (order-2 RNC). T^4 scaling confirmed to 3.5 sigma (M = 200).
   A_E ~ 0.007. Purely electric Weyl (static observer).

A robust curvature detector must work on all vacuum families, not
just two. The Petrov classification has six types (O, D, II, III, N,
I). Type O is flat (CJ = 0). Types D and N are tested. The
remaining algebraically general types (I, II, III) are untested.

Additionally, the CJ bridge formula

  dk_vac = T^4 (A_E E^2 + A_B B^2)

was derived assuming d = 4 spacetime dimensions. In d = 2, a test
showed a factor 3.6 discrepancy between the measured CJ and the
predicted coefficient. This failure could indicate a dimension-
dependent correction or a fundamental limitation of the formula.
Testing in d = 3 would determine whether the discrepancy grows or
shrinks with dimension.

## 3. Known Results

- PP-wave (d = 4, Type N): CJ verified, T^4 scaling, A_E ~ 0.036.
- Schwarzschild (d = 4, Type D): CJ verified, T^4 scaling, A_E ~ 0.007.
- Flat spacetime (d = 4, Type O): CJ = 0 exactly (SO(3) selection rule).
- De Sitter (d = 4, conformally flat): CJ = 0 after Ricci subtraction.
- d = 2 test: CJ bridge formula FAILS by factor 3.6. The 2D causal
  diamond has fundamentally different combinatorics.
- d = 3: NOT TESTED. The 3D Hasse diagram construction and causal
  predicate are available in principle (sprinkle into 2+1 diamond).
- Godel spacetime: has both E and B nonzero, Type D. Exact causal
  relations are known but the causal structure is pathological
  (closed timelike curves outside a critical radius).
- Bianchi IX: anisotropic cosmological model, Type I in general.
  No exact causal predicate available; requires numerical integration.
- Kasner: anisotropic vacuum, Type I. Exact causal relations available
  in synchronous coordinates.

## 4. Failed Approaches

1. **d = 2 bridge formula.** The factor 3.6 discrepancy in d = 2
   suggests that the T^4 coefficient has a dimension-dependent
   prefactor. The derivation of the bridge formula uses 4D-specific
   angular integrals (integration over S^2) that have different
   values in lower dimensions.

2. **Godel causality.** The Godel spacetime has closed timelike curves,
   which break the partial order structure of the causal set. The CJ
   estimator assumes a well-defined causal ordering, so Godel is
   pathological as a test case unless the diamond is small enough to
   avoid CTCs.

## 5. Success Criteria

- CJ measurement on at least one Type I or Type II vacuum spacetime
  in d = 4, with N >= 2000 and M >= 50.
- Verification that CJ is nonzero and scales as T^4 on the new
  family.
- Extraction of A_E (and A_B if applicable) on the new family.
  Comparison with pp-wave and Schwarzschild values.
- CJ measurement in d = 3 with N >= 2000 and M >= 50 on a 2+1
  curved background (e.g., 3D Schwarzschild-BTZ or 3D pp-wave).
- Assessment of the dimensional dependence of the bridge formula
  coefficient: is the d = 2 failure a low-dimensional artifact or
  does it persist in d = 3?

## 6. Suggested Directions

1. Kasner spacetime: the Kasner metric ds^2 = -dt^2 + t^{2p_1} dx^2
   + t^{2p_2} dy^2 + t^{2p_3} dz^2 (with p_1 + p_2 + p_3 = 1,
   p_1^2 + p_2^2 + p_3^2 = 1) is a Type I vacuum. The causal
   structure is analytically tractable. Choose a diamond centered at
   a point with nonzero Weyl curvature.

2. BTZ black hole (d = 3): the 2+1 dimensional BTZ black hole has
   no propagating degrees of freedom but has nonzero curvature.
   It provides a clean d = 3 test case.

3. 3D pp-wave: construct a 2+1 pp-wave with h_{xx} = epsilon x^2.
   The causal predicate generalises straightforwardly from d = 4.

4. Godel with CTC-free diamond: choose a diamond center at r = 0
   in Godel spacetime and T small enough that the diamond lies
   entirely within the CTC-free region (r < r_CTC). This avoids
   the causality pathology.

## 7. References

1. Roy, M., Sinha, D. and Surya, S. (2013). "Discrete geometry of a
   small causal diamond." arXiv:1212.0631.
2. Banados, M., Teitelboim, C. and Zanelli, J. (1992). "The black
   hole in three-dimensional space-times." Phys. Rev. Lett. 69, 1849.
   arXiv:9204099.
3. Kasner, E. (1921). "Geometrical theorems on Einstein's cosmological
   equations." Am. J. Math. 43, 217.

## 8. Connections

- Related to **OP-34** (N-scaling exponent): if the exponent alpha
  differs between vacuum families, this would be visible in the
  third-family measurement.
- Related to **OP-45** (Kottler): Kottler is another extension
  beyond the two tested families, but with Ricci curvature.
- Related to **OP-46** (boosted Schwarzschild): both test the
  CJ response to non-electric Weyl components.
- Related to **OP-49** (blocked FND-1 tests): third vacuum family
  is test 5.2 in the FND-1 plan.
