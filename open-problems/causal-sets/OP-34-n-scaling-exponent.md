---
id: OP-34
title: "N-scaling exponent of the CJ observable"
domain: [numerics, mathematics]
difficulty: hard
status: open
deep-research-tier: A
blocks: [OP-36, OP-42]
blocked-by: []
roadmap-tasks: []
papers: [1212.0631, 1904.01034]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-34: N-scaling exponent of the CJ observable

## 1. Statement

The CJ (causal-jet) observable, defined as a stratified sum of local
path-kurtosis excess over causal diamonds in a Poisson sprinkling,
scales empirically as CJ proportional to N^alpha. Fits to data in the range
N = 500 to 10000 give alpha = 0.955 +/- 0.027, consistent with 8/9 = 0.889
but also consistent with 1. Derive the exact exponent from the combinatorics
of Poisson sprinklings on Hasse diagrams, or prove that no universal
power-law exponent exists.

## 2. Context

The CJ observable was constructed as a curvature detector for causal sets:
it measures the second cumulant (kurtosis excess) of maximal-chain-length
distributions in local order intervals, binned by depth, position, and
proper-time strata. The N-scaling determines whether CJ admits a finite
continuum limit (alpha < 1), diverges (alpha > 1), or sits at the marginal
case (alpha = 1).

A heuristic argument for alpha = 8/9 proceeds as follows. In d = 4
Lorentzian dimensions, a causal diamond has a (2d+1) = 9-dimensional
ordered configuration space (4 past lightcone coordinates, 1 splitting
surface, 4 future lightcone coordinates). The dominant contribution
to the path-kurtosis sum comes from chains whose length scales as
N^{2d/(2d+1)} = N^{8/9}. This 9-simplex interpretation is suggestive
but lacks rigorous justification.

## 3. Known Results

- **Numerical fit (N = 500-10000, M = 50):** alpha = 0.955 +/- 0.027
  (linear regression on log-log). Residuals show slight upward curvature.
- **Alternative model:** N / log(N)^{0.39} fits with comparable chi-squared,
  giving an effective exponent that approaches 1 from below.
- **Small-N vs large-N:** alpha increases from approximately 0.87 at
  N in [500, 2000] to approximately 1.00 at N in [5000, 10000]. This drift
  is consistent with either alpha = 1 with logarithmic corrections or
  alpha < 1 with slow convergence.
- **Flat spacetime (Minkowski diamond):** CJ = 0 exactly (by SO(3)
  selection rule). The scaling applies only to curved backgrounds.
- **pp-wave and Schwarzschild:** both show the same scaling exponent
  within statistical error.
- **Heuristic 9-simplex argument:** 4 past + 1 split + 4 future = 9
  ordered variables; Stirling approximation gives N^{8/9}. Not proven.

## 4. Failed Approaches

1. **Direct combinatorial counting.** Attempted to count the number of
   maximal antichains of length k in a Poisson-sprinkled diamond and
   extract the variance scaling. The number of antichains grows
   super-exponentially, making exact enumeration intractable beyond
   N approximately 200. The approach fails because it conflates maximal
   chains with maximal antichains in the growth estimate.

2. **Continuum limit via Kleitman-Rothschild.** The Kleitman-Rothschild
   theorem gives the asymptotic number of partial orders, but these are
   generic posets, not Poisson-sprinkled causal sets. The sprinkling
   constraint (faithful embedding into Lorentzian manifold) breaks the
   symmetry assumptions of the theorem.

3. **Mean-field chain-length model.** Modelled chain lengths as independent
   geometric random variables with parameter depending on N. Predicts
   alpha = 1 exactly, but misses the correlations induced by the causal
   structure. Correlations reduce the effective degrees of freedom,
   potentially lowering alpha.

## 5. Success Criteria

- A rigorous derivation of alpha from first principles (Poisson process
  on a Lorentzian manifold, Hasse diagram construction, chain-length
  distribution theory), valid in d = 4 dimensions.
- OR: a proof that no universal power-law alpha exists, with a
  characterization of the actual asymptotic form (e.g., N / log(N)^beta).
- The result must be consistent with the numerical data: alpha in
  [0.87, 1.03] across the tested N-range.
- If alpha = 8/9, the derivation must explain the dimensional origin
  of the 9-simplex structure.

## 6. Suggested Directions

1. **Longest-chain scaling in random DAGs.** The length of the longest
   chain in a Poisson-sprinkled d-dimensional diamond scales as
   c_d N^{1/(d+1)} (Bollobas-Brightwell). Investigate whether the
   second moment of the chain-length distribution follows a different
   exponent and whether stratification modifies it.

2. **Entropy of ordered configurations.** Compute the entropy of the
   space of maximal chains in an Alexandrov interval as a function of N.
   If the entropy scales as (8/9) log N, this would explain the exponent.

3. **Renormalization-group analysis.** Treat the stratification binning
   as a coarse-graining operation. Define a flow on the space of
   bin-averaged observables and look for fixed-point scaling.

4. **Exact results in d = 2.** In d = 2, the heuristic gives alpha = 4/5.
   This is numerically accessible at much larger N (up to 10^5) and
   could validate or invalidate the dimensional formula.

## 7. References

- Bollobas, B. and Brightwell, G. (1991). "The width of random graph
  orders." *Rand. Struct. Alg.* 2, 37-49.
- Brightwell, G. and Georgiou, N. (2010). "Continuum limits for
  classical sequential growth models." *Rand. Struct. Alg.* 36, 218-250.
- Roy, M., Sinha, D. and Surya, S. (2013). "Discrete geometry of a
  small causal diamond." arXiv:1212.0631.
- Wang, Z. (2019). "Causal set d'Alembertian in curved spacetime."
  arXiv:1904.01034.

## 8. Connections

- **OP-36 (RSY derivation):** The N-scaling exponent enters the
  normalization of the bulk integral; incorrect alpha propagates into
  the coefficient 8/(3 x 9!).
- **OP-42 (A_E universality):** If alpha differs between geometries,
  A_E ratios would acquire N-dependent corrections, complicating
  the universality test.
- **OP-43 (N-normalization divergence):** If alpha >= 1, the amplitude
  A_E diverges and OP-43 becomes a direct consequence of OP-34.
