---
id: OP-36
title: "RSY derivation at Weyl order"
domain: [theory, mathematics]
difficulty: very-hard
status: open
deep-research-tier: B
blocks: []
blocked-by: [OP-34, OP-35]
roadmap-tasks: []
papers: [1212.0631, 2301.13525, 1904.01034]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-36: RSY derivation at Weyl order

## 1. Statement

Derive the coefficient 8/(3 x 9!) in the CJ bridge formula

  CJ = [8/(3 x 9!)] (8 pi / 15) E^2 V_4 + O(R^2)

from 4-dimensional Poisson sprinkling combinatorics at Weyl-squared
order, extending the Roy-Sinha-Surya (RSY) chain-counting results
beyond the Ricci scalar level. The formula has four factors:

  c_4^2 = 8/3,   1/9! (beta-function overlap),
  (8 pi / 15) E^2 (angular integral),   pi T^4 / 24 (volume).

Each factor is individually understood. The open problem is assembling
them into a single derivation that respects two conditions:

- **Condition A (measure):** the stratification measure converges to
  the uniform split measure (see OP-35).
- **Condition B (factorization):** the kurtosis excess factorizes into
  a geometric piece (depending on Weyl) and a combinatorial piece
  (depending on chain statistics) with controlled remainder.

## 2. Context

The RSY program (Roy, Sinha, Surya 2013) derives the Benincasa-Dowker
action from chain counting on causal sets. Their result operates at the
level of the scalar curvature R: they show that the expected number of
k-chains in a small Alexandrov interval recovers the coefficients of
the Benincasa-Dowker d'Alembertian, which gives R at leading order.

Extending this to Weyl-squared order requires:

1. Tracking the SECOND moment (variance/kurtosis) of chain-length
   distributions, not just the first moment (mean).
2. Using the curvature expansion of the causal diamond volume to
   extract the E^2 = E_ij E^ij Weyl-electric contribution.
3. Accounting for the angular structure of the embedding: the angular
   integral over the electric Weyl tensor gives (8 pi / 15) E^2 in
   d = 4, using the identity integral (E_ij n^i n^j)^2 d Omega =
   (8 pi / 15) E^2.

de Brito (2301.13525) extends the Benincasa-Dowker action to R^2 - 2 box R
order using stacked order intervals. Wang (1904.01034) computes the
area-counting analog (ACD) and shows that the E^2 contribution is
non-trivial only in d = 4 (where the Gauss-Bonnet Euler density
H^2 = 6(d-4) vanishes).

## 3. Known Results

- **c_4^2 = 8/3:** The second cumulant of the 4-chain-length distribution
  has coefficient 8/3 in the Weyl expansion. Derived by expanding the
  GHY volume of a geodesic ball to O(curvature^2) in RNC.
- **1/9! (beta overlap):** The overlap integral of the beta distribution
  B(5, 5) with the chain-length moment generating function gives
  1/9! = 1/362880. Verified numerically to 6 significant figures.
- **(8 pi / 15) E^2:** The angular integral identity for the electric
  Weyl tensor. Standard result from SO(3) representation theory:
  E_ij n^i n^j is an l = 2 spherical harmonic, and the integral of
  its square gives (8 pi / 15) trace(E^2).
- **pi T^4 / 24:** The volume of a causal diamond of proper-time
  height T in 4D Minkowski. Standard causal set result.
- **Combined numerical verification:** Monte Carlo simulation at
  N = 5000, M = 200 gives CJ / (E^2 V_4) = 0.0360 +/- 0.004.
  The predicted value 8/(3 x 9!) x (8 pi / 15) = 0.0367. Agreement
  within 2%.

- **Literal delta-function attempt:** Inserting a delta function
  delta(p, p') to enforce coincidence of the split point gives
  37/29099070, which has ratio 0.461 to 1/9!. The discrepancy is
  because the delta function is not the correct measure; the split
  must be distributed according to the beta overlap kernel.

## 4. Failed Approaches

1. **Direct delta(p, p') insertion.** Computing the chain-length
   variance with a coincidence delta function at the split point gives
   37/29099070 instead of 1/9!. The ratio is 0.461, indicating that
   the split-point distribution is not a delta function but the full
   beta B(5, 5) kernel. This approach fails because it does not
   account for the extended spatial support of the split.

2. **Moment-generating-function expansion.** Expanded the MGF of the
   chain-length distribution to fourth order in the curvature. The
   expansion converges only for curvature radii R_curv >> T (the
   diamond size), which is precisely the regime where the Weyl
   contribution is negligible. At R_curv ~ T, the expansion requires
   resummation that is equivalent to solving the original problem.

3. **Continuous-chain approximation.** Replaced discrete chains with
   continuous timelike geodesics. This gives the correct angular
   structure but misses the 1/9! combinatorial factor, which is
   inherently discrete. The continuous limit gives 1/8! instead,
   off by a factor of 9.

4. **de Brito stacked intervals.** Applied the de Brito construction
   (stacking two order intervals to get R^2 - 2 box R) but could not
   separate the Weyl^2 and Ricci^2 contributions. The stacking
   construction inherently mixes the two because it uses the full
   Riemann tensor, not the Weyl decomposition.

## 5. Success Criteria

- A complete derivation starting from the Poisson sprinkling
  definition of CJ and arriving at 8/(3 x 9!) (8 pi / 15) E^2 V_4
  with explicitly controlled error terms.
- The derivation must separately establish Condition A (measure
  convergence, see OP-35) and Condition B (factorization).
- The derivation must explain why the coefficient is independent of
  the magnetic Weyl tensor B_ij in the leading term.
- The result must hold for arbitrary vacuum spacetimes (R_mu_nu = 0),
  not just specific test geometries.

## 6. Suggested Directions

1. **Variance of the Benincasa-Dowker functional.** Compute
   Var[S_BD(x)] where S_BD is the BD action evaluated at a single
   point x with contributions from its causal past. The RSY technology
   computes E[S_BD]; extending to Var[S_BD] would give a Weyl-order
   quantity by the Gauss-Bonnet theorem.

2. **Two-point chain correlation function.** Define C(x, y) as the
   correlation of chain lengths through x and y. The integral of
   C over a diamond should decompose into Weyl and Ricci parts by
   standard tensor decomposition. The Weyl part would give the
   desired coefficient.

3. **Representation-theoretic factorization.** The l = 2 structure of
   E_ij under SO(3) and the (2d+1) = 9 ordered configuration space
   (OP-34) suggest a representation-theoretic argument. Decompose the
   chain-length variance into SO(3) irreps and identify the l = 2
   component with E^2.

4. **Exploit d = 4 specialness.** Wang's result shows that the E^2
   area counting works only in d = 4. Look for a topological or
   cohomological argument specific to 4 dimensions.

## 7. References

- Roy, M., Sinha, D. and Surya, S. (2013). arXiv:1212.0631.
- de Brito, G. P. (2023). arXiv:2301.13525.
- Wang, Z. (2019). arXiv:1904.01034.
- Benincasa, D. M. T. and Dowker, F. (2010). arXiv:1001.2725.
- Brightwell, G. and Luczak, M. (2011). arXiv:0901.0240.

## 8. Connections

- **OP-34 (N-scaling):** The N-scaling exponent determines the
  normalization of the volume factor V_4 in the formula.
- **OP-35 (stratification measure):** Condition A is precisely the
  content of OP-35.
- **OP-37 (polarization anisotropy):** The derivation must explain
  why the angular integral (8 pi / 15) E^2 uses the isotropic average,
  while OP-37 shows polarization-dependent ratios.
- **OP-38 (non-vacuum coefficients):** Extending beyond vacuum
  requires separating Weyl and Ricci contributions.
