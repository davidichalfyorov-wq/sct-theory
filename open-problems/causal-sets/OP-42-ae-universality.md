---
id: OP-42
title: "A_E universality across vacuum geometries"
domain: [numerics, theory]
difficulty: hard
status: open
deep-research-tier: B
blocks: []
blocked-by: [OP-39]
roadmap-tasks: []
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-42: A_E universality across vacuum geometries

## 1. Statement

The CJ observable in vacuum spacetimes satisfies CJ = A_E E^2 V_4
where E^2 = E_ij E^ij is the squared electric Weyl tensor and V_4 is
the diamond 4-volume. The coefficient A_E should be universal (geometry-
independent) if CJ is a true curvature detector. However, the measured
ratio is

  A_E(Schwarzschild) / A_E(pp-wave) = 0.508 +/- 0.03

at N = 10000, M = 50 (stable across N = 2000 to 10000). This is
significantly below 1.0. Determine whether the non-universality is
physical (reflecting genuine geometry-dependent corrections) or an
artifact of the jet predicate approximation.

## 2. Context

A_E was defined as the proportionality constant in the CJ bridge
formula. For a universal curvature detector, A_E should depend only on
the dimension d and the CJ construction (binning scheme, predicate
choice), not on the specific vacuum geometry.

Two vacuum geometries have been tested extensively:

1. **pp-wave:** h_+ = A (x^2 - y^2) f(u), exact plane gravitational
   wave. The jet predicate is exact for pp-waves (the metric is
   quadratic in transverse coordinates), so A_E(pp-wave) is the
   "gold standard" reference value.

2. **Schwarzschild:** ds^2 = -(1-2M/r) dt^2 + (1-2M/r)^{-1} dr^2 +
   r^2 d Omega^2. The jet predicate is approximate (quadratic RNC
   expansion), losing signal at the diamond boundary.

The ratio 0.508 has been investigated through 8 hypothesis tests:

- **H1 (orbit type):** Tested jet vs jet on both geometries.
  Jet/jet = 1.011 for pp-wave self-comparison. Cleared.
- **H2 (stratification bias):** Varied bin counts from 20 to 100.
  Ratio stable to 3%. Cleared.
- **H3 (depth bias):** Varied depth binning from 2 to 5 bins.
  Ratio stable to 4%. Cleared.
- **H4 (N-dependence):** Measured at N = 2000, 5000, 10000.
  Ratio stable: 0.51, 0.50, 0.51. Cleared.
- **H5 (boundary contamination):** Excluded boundary elements
  (outer 20%). Ratio changed from 0.51 to 0.54. Marginal.
- **H6 (proper-time range):** Varied T from 0.5 to 2.0 R_curv.
  Ratio varies from 0.53 (small T) to 0.48 (large T). Significant.
- **H7 (predicate order):** Compared order-2 vs order-1 jet.
  Order-1 gives ratio 0.38, order-2 gives 0.51. Strongly predicate-
  dependent.
- **H8 (exact predicate, partial):** Implemented exact predicate for
  N = 500 (feasible at small N). Ratio = 0.82 +/- 0.08. Significant
  improvement toward 1.0.

## 3. Known Results

- **A_E(pp-wave, jet, N=10000) = 0.0360 +/- 0.004:** Reference value.
- **A_E(Schwarzschild, jet, N=10000) = 0.0183 +/- 0.003:** Reduced
  by factor 0.508.
- **A_E(Schwarzschild, exact, N=500) = 0.0295 +/- 0.008:** Partial
  recovery (ratio 0.82). Statistical error is large due to small N.
- **Predicate quality metric:** At N = 5000, the jet predicate
  misclassifies 2.3% of causal relations on Schwarzschild (vs 0.0% on
  pp-wave). The misclassified pairs are concentrated at the diamond
  boundary, where the curvature signal is strongest.
- **Signal loss mechanism:** Misclassified pairs systematically reduce
  the measured kurtosis excess. A pair (p, q) that is truly causal but
  classified as spacelike is a "missed link." Missed links reduce the
  chain lengths, lowering the variance.
- **Missed-link fraction:** Approximately 66% of the signal loss
  (0.508 vs 1.0) can be attributed to missed links at the boundary.
  The remaining 34% is consistent with higher-order predicate errors
  in the diamond interior.

## 4. Failed Approaches

1. **Analytical correction factor.** Attempted to compute the fraction
   of missed links as a function of T / R_curv and apply a multiplicative
   correction to A_E. The correction depends on the geometry-dependent
   boundary shape, making it non-universal. A correction factor that
   depends on the answer defeats the purpose.

2. **Predicate-averaged universality.** Proposed that universality
   holds after averaging over a family of predicates (different orders,
   different coordinate choices). The average is not well-defined:
   there is no natural measure on the space of predicates, and the
   average over order-1 and order-2 jets gives 0.445, further from 1.0.

3. **Rescaling by effective volume.** Hypothesized that the jet
   predicate sees a smaller effective volume V_eff < V_4 on
   Schwarzschild. Measured V_eff / V_4 = 0.94, which gives a corrected
   ratio of 0.54 -- insufficient to explain the factor 2 discrepancy.

4. **Magnetic Weyl contribution.** On pp-waves, B = 0 (purely
   electric). On Schwarzschild, B != 0 in general frames. Hypothesized
   that the B contribution has opposite sign, reducing the total.
   Measured B^2 / E^2 = 0 in the static frame (Schwarzschild is type D,
   electrostatic). B = 0 in the natural frame, ruling out this
   explanation.

## 5. Success Criteria

- **Option A (universality confirmed):** Implement the exact
  Schwarzschild predicate (OP-39) at N >= 5000 and show
  A_E(Sch, exact) / A_E(ppw, jet) = 1.0 +/- 0.1. This would prove
  universality and attribute the 0.508 ratio entirely to predicate
  quality.

- **Option B (universality fails):** Show that even with the exact
  predicate, A_E(Sch) / A_E(ppw) deviates from 1.0 by more than 10%.
  In this case, identify the geometry-dependent correction term and
  show it has the form A_E = A_0 (1 + c_1 R_curv / T + ...) with
  computable c_1.

- **Either way:** the result must be validated on at least one
  additional vacuum geometry (e.g., Kerr, C-metric, or Kasner).

## 6. Suggested Directions

1. **Exact predicate at large N (primary route).** Implement the exact
   Schwarzschild predicate (see OP-39) using Hagihara elliptic
   integrals or GPU-parallel ODE integration. Measure A_E at N = 5000,
   10000. This directly resolves the question.

2. **Universality in d = 2.** In d = 2, the jet predicate is exact for
   ALL conformally flat spacetimes (which includes all 2D spacetimes).
   Measure A_E on multiple 2D geometries and check universality. If
   universality holds in d = 2 (where predicate quality is not an
   issue), this supports the predicate-artifact hypothesis for d = 4.

3. **Kerr test.** Kerr spacetime is type D (same Petrov type as
   Schwarzschild) but has non-zero angular momentum. If A_E(Kerr) /
   A_E(Sch) = 1.0, universality within a Petrov type is supported.
   If A_E(Kerr) != A_E(Sch), the angular momentum introduces a new
   parameter.

4. **Matched-predicate comparison.** Construct a pp-wave background
   with the SAME curvature profile as a Schwarzschild diamond (matching
   E_ij(x) at every point). Use the same jet predicate on both.
   Any remaining difference would be purely geometric (not predicate-
   related).

## 7. References

- Bollobas, B. and Brightwell, G. (1991). "The width of random graph
  orders." *Rand. Struct. Alg.* 2, 37-49.
- Sorkin, R. D. (2003). "Causal sets: Discrete gravity."
  arXiv:gr-qc/0309009.
- Benincasa, D. M. T. and Dowker, F. (2010). "Scalar curvature of a
  causal set." arXiv:1001.2725.

## 8. Connections

- **OP-39 (exact Schwarzschild predicate):** OP-42 is directly
  blocked by OP-39. Resolution of OP-39 provides the primary tool
  for resolving OP-42.
- **OP-34 (N-scaling):** If universality fails, A_E may have
  geometry-dependent N-scaling, complicating the continuum limit.
- **OP-37 (polarization anisotropy):** Different polarizations have
  different A_E within the same geometry. Universality across
  geometries is a stronger condition.
- **OP-36 (RSY derivation):** The RSY derivation assumes universality.
  If universality fails, the coefficient 8/(3 x 9!) acquires
  geometry-dependent corrections.
