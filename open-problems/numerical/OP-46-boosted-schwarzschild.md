---
id: OP-46
title: "Boosted Schwarzschild systematic test for magnetic Weyl"
domain: [numerics, causal-sets]
difficulty: medium
status: open
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: []
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-46: Boosted Schwarzschild systematic test for magnetic Weyl

## 1. Statement

Perform a systematic velocity scan of the CJ estimator on boosted
Schwarzschild sprinklings to quantify the magnetic Weyl contribution
B^2 to the GTA signal model. Determine whether the CJ response is
proportional to E^2 + B^2 (gravitational invariant) or to E^2 alone,
and extract the magnetic coupling coefficient a_B.

## 2. Context

The electric and magnetic parts of the Weyl tensor are defined
relative to an observer 4-velocity u^mu:

  E_{ij} = C_{0i0j},  B_{ij} = (1/2) epsilon_{ikl} C^{kl}{}_{0j}.

For a static observer in Schwarzschild, B = 0 (the spacetime is
purely electric). When the diamond is boosted with velocity v
relative to the static frame, both E and B become nonzero, with
B^2 proportional to v^2 E^2 at leading order.

The GTA signal model predicts CJ proportional to T^4 (A_E E^2 + A_B B^2).
Preliminary data from a boosted Schwarzschild run showed that
CJ tracks 24% of the E^2 + B^2 variation with a = 0.238. However,
this was a single-velocity measurement with limited statistics.

A systematic v-scan would determine A_B and test whether CJ is
sensitive to the full Kretschner invariant (E^2 + B^2 for vacuum)
or only to its electric part.

## 3. Known Results

- Static Schwarzschild CJ: well-characterized, dk = +0.030 at T = 1
  (9.3 sigma, M = 200). B = 0 in this configuration.
- PP-wave: also purely electric (B = 0 for standard orientation).
  CJ verified to scale as T^4 E^2.
- Boosted Schwarzschild pilot: a = 0.238, tracking 24% of E^2 + B^2
  variation. Single velocity v = 0.3c, M = 30.
- The boosted Schwarzschild predicate requires Lorentz-transforming
  the RNC coordinates, which introduces additional systematic errors
  in the jet predicate at large separations.

## 4. Failed Approaches

The pilot measurement was insufficient to distinguish between three
competing models:
(a) CJ proportional to E^2 (B^2 invisible),
(b) CJ proportional to E^2 + B^2 (Kretschner invariant),
(c) CJ proportional to alpha E^2 + beta B^2 (independent coefficients).

The single-velocity data point cannot discriminate. A v-scan is needed.

## 5. Success Criteria

- CJ measurements at 6+ boost velocities: v/c = {0, 0.1, 0.2, 0.3,
  0.4, 0.5}, with M >= 50 sprinklings per velocity and N >= 2000.
- Fit to the model CJ = T^4 (A_E E^2(v) + A_B B^2(v)) with E^2(v)
  and B^2(v) computed analytically from the Lorentz-boosted Weyl
  tensor.
- Extraction of A_B / A_E ratio with statistical uncertainty.
- Model comparison (chi-squared or Bayes factor) between
  E^2-only, E^2+B^2, and free alpha E^2 + beta B^2 models.
- Statement of whether B^2 contributes to CJ at all (A_B != 0?).

## 6. Suggested Directions

1. Velocity scan: fix M_BH and T, vary v. At each v, compute E^2(v)
   and B^2(v) analytically using the Lorentz transformation of the
   Weyl tensor. The electric and magnetic invariants for Schwarzschild
   boosted at velocity v are known in closed form.

2. Orientation scan: instead of boosting, rotate the diamond axis
   relative to the radial direction. At angle theta to the radial,
   the electric/magnetic decomposition changes. This provides an
   independent test of the E/B sensitivity.

3. PP-wave with B != 0: the sandwich pp-wave with h_{xy} != 0
   (cross-polarization) has B != 0. This gives a clean test on a
   spacetime with exact causal predicate, avoiding jet predicate
   systematics.

4. Predicate comparison: run the same v-scan with both the jet
   predicate and a higher-order (order-4) RNC expansion to quantify
   the systematic error from the predicate.

## 7. References

1. Stephani, H. et al. (2003). "Exact Solutions of Einstein's Field
   Equations." Cambridge University Press. Chapter 7 (Weyl tensor
   classification).
2. Cherubini, C. et al. (2002). "Second order scalar invariants of
   the Riemann tensor: Applications to black hole spacetimes." Int.
   J. Mod. Phys. D 11, 827. arXiv:0209028.

## 8. Connections

- Related to **OP-45** (Kottler test): both test specific curvature
  channels in the CJ response.
- Related to **OP-47** (third vacuum family): Godel spacetime has
  nonzero magnetic Weyl, providing another B != 0 test case.
- Related to **OP-34** (N-scaling): the scaling exponent alpha may
  differ between E^2 and B^2 channels, which would be visible in a
  boosted v-scan.
- Related to **OP-49** (blocked FND-1 tests): boosted Schwarzschild
  is test 5.1 in the FND-1 plan.
