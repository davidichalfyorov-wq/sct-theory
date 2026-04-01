---
id: OP-35
title: "Stratification measure closure"
domain: [mathematics, numerics]
difficulty: hard
status: open
deep-research-tier: B
blocks: [OP-36]
blocked-by: []
roadmap-tasks: []
papers: [0901.0240, 1212.0631]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-35: Stratification measure closure

## 1. Statement

The CJ bridge formula requires a stratification procedure: elements of
a causal diamond are binned by depth x position x proper-time into a
finite grid, producing bin-averaged kurtosis values. The resulting sum
approximates an integral over a "split measure" ds on s in [0, 1].
Prove that the specific binning scheme (5 proper-time x 3 radial x
3 depth = 45 bins with population-weighted averaging) converges to the
uniform split measure on the unit interval in the continuum limit
N -> infinity, or characterize the actual limiting measure.

## 2. Context

In causal set theory, any order-invariant quantity can be expressed as
a function of the rank coordinate r(x) = |past(x)| / N. Brightwell and
Luczak (0901.0240) proved that the rank coordinate converges to the
uniform distribution on [0, 1] for Poisson sprinklings of compact
Alexandrov intervals, giving a natural "split" parameter.

The CJ construction does not use the rank coordinate directly. Instead,
it assigns elements to strata based on three independent coordinates:

1. **Proper time** tau: the geodesic distance from the past tip,
   discretized into 5 equal bins.
2. **Radial position** rho: the spatial distance from the diamond axis
   in Riemann normal coordinates, discretized into 3 bins.
3. **Hasse depth** d: the length of the longest chain from the past tip
   to the element, discretized into 3 bins.

The resulting 45-bin grid is a non-trivial coordinate system on the
diamond. The question is whether the population-weighted sum over bins
converges to an integral with respect to the Brightwell-Luczak uniform
measure, or whether it introduces systematic distortions.

## 3. Known Results

- **Bulk integral value:** I_bulk = (3/10) V_4 / 9! verified by Monte
  Carlo simulation (MC value: 0.2999 +/- 0.0003 at N = 5000, M = 200).
  The factor 3/10 = 0.3 is the fraction of the uniform measure captured
  by the geometric binning scheme.
- **Brightwell-Luczak theorem:** For any order-invariant function f on
  a Poisson sprinkling, E[f(r(x))] -> integral_0^1 f(s) ds as
  N -> infinity, where r(x) is the rank coordinate.
- **Depth-rank correlation:** Hasse depth d and rank coordinate r are
  correlated (Spearman rho approximately 0.93 at N = 5000) but not
  identical. Depth measures the longest chain length, rank measures the
  past volume.
- **Population imbalance:** Bins near the diamond boundary have
  systematically fewer elements (by a factor of 3-5) than central bins.
  Population weighting compensates for this, but the weights themselves
  depend on N.

## 4. Failed Approaches

1. **Direct rank substitution.** Replacing the 45-bin grid with rank
   coordinate binning eliminates the stratification but loses the
   depth and radial information that separates bulk from boundary
   contributions. The boundary-bulk separation is essential: boundary
   elements contribute noise that is 10-50x larger than the curvature
   signal. Discarding the separation destroys the signal-to-noise ratio.

2. **Riemann-sum convergence argument.** Treated each bin as a Riemann
   sum cell with volume proportional to population. This gives
   convergence for smooth integrands, but the kurtosis excess is not
   smooth near the boundary (it has a jump discontinuity at the
   bulk-boundary transition). The jump introduces an O(1/sqrt(N))
   error that does not vanish with refinement of the bin count.

3. **Increasing bin count.** Tried 10 x 5 x 5 = 250 bins. Statistical
   noise per bin increases as 1/sqrt(population), making individual bin
   estimates unreliable at N = 5000. The optimal bin count is
   O(N^{1/3}) by the bias-variance tradeoff, but no proof exists that
   this optimal choice converges to the uniform measure.

## 5. Success Criteria

- A proof that the 45-bin stratification with population weighting
  converges to a well-defined measure mu on [0, 1] as N -> infinity.
- Identification of mu: either mu = ds (uniform) or mu = w(s) ds for
  some explicitly computable weight function w(s).
- If mu is not uniform, compute the correction factor mu/ds and verify
  it against the observed 0.2999/0.3333 = 0.900 ratio (if I_bulk were
  computed relative to the uniform expectation).
- Rate of convergence: ||mu_N - mu||_TV = O(N^{-gamma}) with gamma > 0
  explicitly determined.

## 6. Suggested Directions

1. **Coupling to Brightwell-Luczak.** Construct an explicit bijection
   between the 45-bin grid and rank-coordinate bins. If the bijection
   distorts the measure by a bounded Jacobian, convergence follows from
   the Brightwell-Luczak theorem plus bounded distortion.

2. **Stein's method for Poisson functionals.** The population counts
   in each bin are sums of indicator functions of a Poisson process.
   Stein's method gives quantitative convergence rates for such sums
   to their expectations, including the covariance structure.

3. **Numerical convergence study.** Run the stratification at
   N = 1000, 2000, 5000, 10000, 20000 on flat Minkowski diamonds
   (where the answer is known: CJ = 0). Measure the deviation of the
   bin-averaged measure from uniform as a function of N.

4. **Adaptive stratification.** Replace fixed bin boundaries with
   quantile-based boundaries (each bin has equal expected population).
   This automatically produces a measure that converges to uniform
   under the rank coordinate.

## 7. References

- Brightwell, G. and Luczak, M. (2011). "Order-invariant measures on
  causal sets." arXiv:0901.0240.
- Roy, M., Sinha, D. and Surya, S. (2013). "Discrete geometry of a
  small causal diamond." arXiv:1212.0631.
- Devroye, L. and Lugosi, G. (2001). *Combinatorial Methods in Density
  Estimation.* Springer.

## 8. Connections

- **OP-34 (N-scaling):** The measure determines the effective number of
  degrees of freedom in the CJ sum; incorrect measure normalization
  shifts the apparent N-scaling exponent.
- **OP-36 (RSY derivation):** Condition A in the RSY bridge formula is
  precisely the statement that the stratification measure converges to
  ds. Resolving OP-35 directly addresses Condition A.
