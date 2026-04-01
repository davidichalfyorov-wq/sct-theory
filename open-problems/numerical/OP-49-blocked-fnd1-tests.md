---
id: OP-49
title: "Seven blocked FND-1 experimental tests"
domain: [numerics, causal-sets]
difficulty: easy-to-medium
status: open
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: [FND-1]
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-49: Seven blocked FND-1 experimental tests

## 1. Statement

Complete the seven experimental tests from the original FND-1 plan
that were designed but not executed due to prerequisites, computational
cost, or external dependencies. Each test addresses a specific aspect
of the CJ estimator's validity as a curvature detector.

## 2. Context

The FND-1 program (Recovery of QFT on Curved Spacetime from Causal
Sets) ran approximately 17 experiments across 9 routes. Seven tests
were designed with partial specifications but could not be completed:

| Test | Description | Blocking reason |
|------|-------------|----------------|
| 2.2 | Kottler (Schwarzschild-de Sitter) | Predicate modification needed |
| 2.3 | d = 3 CJ test | 3D Hasse builder not validated |
| 4.4 | Exact null geodesic predicate (Schwarzschild) | Numerical geodesic integrator |
| 5.1 | Boosted Schwarzschild (B != 0) | Lorentz-boosted predicate |
| 5.2 | Third vacuum family | Predicate for Type I/II spacetime |
| 7.2 | Holdout validation of GTA model | Requires tests 2.2, 5.1, 5.2 |
| 8.1 | External review by independent researcher | Requires collaborator |

Tests 2.2, 5.1, and 5.2 are directly addressed by OP-45, OP-46, and
OP-47 respectively. Tests 4.4 and 7.2 have unique aspects not covered
by other open problems. Test 8.1 requires an external collaborator.

## 3. Known Results

Each test has partial design from the FND-1 experimental plan:

**Test 2.2 (Kottler).** See OP-45 for details. Pilot showed 22%
CJ reduction.

**Test 2.3 (d = 3).** See OP-47 for details. The d = 2 test showed
a factor 3.6 failure of the bridge formula.

**Test 4.4 (Exact null predicate).** The Schwarzschild jet predicate
(order-2 RNC) introduces systematic errors at separations comparable
to the Schwarzschild radius. An exact predicate requires integrating
null geodesics numerically for each pair of sprinkled points. For
N = 5000 points, this means N(N-1)/2 ~ 12.5 million geodesic
integrations. Each geodesic integration requires solving the null
geodesic ODE in Schwarzschild coordinates (4 coupled first-order
ODEs, typically 50-100 RK4 steps). Estimated wall time: 2-8 hours
per sprinkling on 16-core CPU.

**Test 5.1 (Boosted Schwarzschild).** See OP-46 for details.
Preliminary a = 0.238 tracking.

**Test 5.2 (Third family).** See OP-47 for details.

**Test 7.2 (Holdout validation).** The GTA signal model
CJ = T^4 (A_E E^2 + A_B B^2) was calibrated on pp-wave and
Schwarzschild data. The holdout test reserves one or more geometries
(from tests 2.2, 5.1, 5.2) as blind predictions. The GTA model
predicts the CJ value on the holdout geometry using the calibrated
A_E and A_B coefficients, and the prediction is compared with the
measurement. This is a genuine out-of-sample test.

**Test 8.1 (External review).** An independent researcher (not
involved in the SCT project) replicates a subset of the FND-1
measurements using their own code and Hasse implementation. This
is the ultimate check against systematic bias in the sct_tools
pipeline. No collaborator has been identified.

## 4. Failed Approaches

Tests 2.2, 5.1, 5.2: see OP-45, OP-46, OP-47.

Test 4.4: a prototype exact null integrator was written but not
validated against known exact results. The geodesic ODE integration
converges slowly near the photon sphere (r = 3M), requiring adaptive
step size control.

Test 7.2: cannot proceed until at least one of tests 2.2, 5.1, 5.2
produces a measurement with sufficient precision.

Test 8.1: no external collaborator has been approached.

## 5. Success Criteria

- **Test 4.4:** CJ measurement on Schwarzschild with exact null
  predicate at N = 2000, M = 30. Comparison with jet-predicate
  result at the same N and M. Systematic error quantification:
  |CJ_exact - CJ_jet| / CJ_exact.
- **Test 7.2:** blind prediction of CJ on a holdout geometry
  using calibrated GTA coefficients. Prediction accuracy within
  2 sigma of measured value.
- **Test 8.1:** independent replication of at least one FND-1
  measurement (pp-wave CJ at N = 5000, M = 50) using code
  developed without access to sct_tools.
- All other tests: covered by OP-45, OP-46, OP-47.

## 6. Suggested Directions

1. **Test 4.4 implementation.** Use scipy.integrate.solve_ivp with
   the DOP853 method (high-order Runge-Kutta with adaptive step)
   for the null geodesic ODE. Validate against known Schwarzschild
   null geodesic solutions (light bending angle, photon sphere
   orbits). Parallelize over point pairs using multiprocessing
   (16 cores).

2. **Test 7.2 protocol.** Select the Kottler test (OP-45) as the
   holdout geometry. Calibrate A_E and A_B using pp-wave and
   Schwarzschild data only. Predict CJ(Kottler) at a specific
   Lambda_cc value. Run OP-45 measurement. Compare prediction vs
   measurement.

3. **Test 8.1 outreach.** Identify potential collaborators in the
   causal set community (Surya group at RRI Bangalore, Dowker group
   at Imperial, Glaser group). Provide them with the exact pp-wave
   causal predicate specification and diamond parameters, but not
   the sct_tools code. Request independent CJ measurement.

## 7. References

1. Benincasa, D. M. T. and Dowker, F. (2010). "The scalar curvature
   of a causal set." Phys. Rev. Lett. 104, 181301. arXiv:1001.2725.
2. Glaser, L. and Surya, S. (2013). "Towards a definition of locality
   in a Manifoldlike Causal Set." Phys. Rev. D 88, 124026.
   arXiv:1309.3403.

## 8. Connections

- Tests 2.2, 5.1, 5.2 are covered by **OP-45**, **OP-46**, **OP-47**.
- Test 7.2 depends on at least one of OP-45, OP-46, OP-47 completing.
- Test 4.4 is independent: it compares predicate quality, not
  geometry type.
- Test 8.1 is independent of all other problems: it is a social
  rather than computational challenge.
- All tests feed into the GTA framework validation and ultimately
  into the connection between causal sets and the continuum
  spectral action.
