---
id: OP-39
title: "Exact Schwarzschild predicate"
domain: [numerics, mathematics]
difficulty: medium
status: open
deep-research-tier: C
blocks: [OP-42]
blocked-by: []
roadmap-tasks: []
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-39: Exact Schwarzschild predicate

## 1. Statement

The "jet predicate" used to define local causal diamonds in curved
spacetimes is a quadratic expansion in Riemann normal coordinates
(RNC). On Schwarzschild backgrounds, this approximation retains only
approximately 34% of the CJ signal compared to an exact geodesic-based
predicate. Implement an exact predicate for Schwarzschild spacetime
that uses the true geodesic interval sigma(p, q) to determine causal
relations, and quantify the signal recovery.

## 2. Context

The CJ observable requires determining, for each pair of sprinkled
points (p, q), whether p causally precedes q. In flat spacetime, this
is trivial: check whether the interval (t_q - t_p)^2 - |x_q - x_p|^2
is positive and t_q > t_p.

In curved spacetime, the causal relation depends on the geodesic
structure. The jet predicate approximates the Synge world function
sigma(p, q) by its Taylor expansion to second order in the coordinate
separation:

  2 sigma_jet(p, q) = g_mu_nu(p) Delta x^mu Delta x^nu
                    + (1/3) R_mu_alpha_nu_beta(p) Delta x^mu Delta x^nu
                      Delta x^alpha Delta x^beta + ...

where Delta x = x_q - x_p in RNC centered at p. This approximation
is valid when Delta x << R_curv (the curvature radius). For
Schwarzschild, R_curv = (r^3 / (2M))^{1/2}, and the diamond size T
must satisfy T << R_curv.

The problem is that for detectably curved diamonds (where CJ >> 0),
one needs T not much smaller than R_curv, and the quadratic
approximation degrades.

## 3. Known Results

- **Signal loss quantification:** A_E(Sch, jet) / A_E(pp-wave, jet) =
  0.508 at N = 10000, M = 50. The pp-wave result is nearly exact (the
  jet predicate is exact for pp-waves to all orders in the transverse
  plane), so the ratio measures the Schwarzschild signal loss.
- **Order-3 divergence:** Adding the cubic term in the RNC expansion
  makes the predicate WORSE: the cubic correction is larger than the
  quadratic term at the diamond boundary, causing the predicate to
  misclassify approximately 12% of causal relations. The expansion is
  asymptotic, not convergent, at the boundary.
- **Exact interval (numerical):** For Schwarzschild, the geodesic
  interval sigma(p, q) can be computed by solving the geodesic ODE
  with boundary conditions x(0) = p, x(1) = q. This requires an
  iterative solver (shooting method or relaxation). Typical cost:
  50-200 ODE evaluations per pair, compared to O(1) for the jet.
- **Scaling cost:** For N points, there are N(N-1)/2 pairs. At N = 5000,
  this is 12.5 million pairs. At 100 ODE evaluations per pair, the
  total cost is 1.25 billion evaluations. In Python, this takes
  approximately 10^4 seconds. A C/Cython implementation could reduce
  this to approximately 100 seconds.

## 4. Failed Approaches

1. **Higher-order jet (order 3, 4, 5).** As noted above, the RNC
   expansion of the Synge function is asymptotic at the diamond
   boundary. The order-3 jet gives worse results than order 2, and
   order 4 is even worse (diverges at 15% of boundary points). The
   expansion radius of convergence is approximately 0.7 R_curv, but
   the diamond extends to approximately 1.0 R_curv in the radial
   direction. Asymptotic expansions cannot be fixed by adding terms.

2. **Pade resummation of the RNC series.** Constructed [2/2] and [3/3]
   Pade approximants of the Synge function. These improve convergence
   in the radial direction but introduce spurious poles in the angular
   directions, causing approximately 5% of pairs to have undefined
   causal relation. The poles cannot be removed without knowing the
   exact function.

3. **Lookup table interpolation.** Pre-computed sigma(p, q) on a grid
   and interpolated. The 4D nature of p and 4D nature of q means the
   grid is 8-dimensional. At 20 points per dimension, the grid has
   20^8 = 25.6 billion entries. Memory-prohibitive.

4. **Christoffel-symbol integration.** Integrated the geodesic equation
   dot{x}^mu + Gamma^mu_{alpha beta} dot{x}^alpha dot{x}^beta = 0
   using RK4 in Python/NumPy. Correct results, but 50x too slow for
   production use (approximately 3 hours for N = 2000).

## 5. Success Criteria

- A working implementation of the exact Schwarzschild predicate that
  correctly determines the causal relation for all pairs in a
  sprinkling of N = 5000 points.
- Runtime under 300 seconds for N = 5000 on a single CPU core, or
  under 30 seconds with GPU acceleration.
- Validation: on flat spacetime, the exact predicate and jet predicate
  must agree on > 99.9% of pairs.
- Signal recovery: A_E(Sch, exact) / A_E(pp-wave, jet) should be
  measured and compared to the theoretical prediction of 1.0.
- If A_E ratio approaches 1.0, this confirms the universality
  conjecture (OP-42).

## 6. Suggested Directions

1. **C/Cython geodesic solver.** Implement the Schwarzschild geodesic
   ODE in C with a 4th-order Runge-Kutta integrator. Use the Killing
   symmetries (energy E, angular momentum L, Carter constant Q) to
   reduce the 8D ODE to a 2D effective problem (r, theta). The
   effective potential approach reduces cost by approximately 10x.

2. **GPU-parallel ODE integration.** Each pair (p, q) is independent.
   Parallelize over pairs on the GPU using CuPy or CUDA. With N = 5000,
   there are 12.5 million independent ODEs. The RTX 3090 Ti has 10752
   CUDA cores; each core handles approximately 1200 pairs.

3. **Semi-analytical Schwarzschild geodesics.** Schwarzschild geodesics
   can be expressed in terms of elliptic integrals (Hagihara 1931).
   The Synge function is then a combination of elliptic K, E, and Pi.
   Evaluating elliptic integrals is O(1) per pair (no ODE), reducing
   the cost to O(N^2) with small constant.

4. **Hybrid approach.** Use the jet predicate for pairs with small
   separation (Delta x < 0.5 R_curv, where it is accurate) and the
   exact solver only for pairs near the diamond boundary (Delta x >
   0.5 R_curv). Approximately 70% of pairs fall in the "safe" zone,
   reducing the exact-solver workload by 3x.

## 7. References

- Hagihara, Y. (1931). "Theory of the relativistic trajectories in a
  gravitational field of Schwarzschild." *Jpn. J. Astron. Geophys.*
  8, 67-176.
- Poisson, E., Pound, A. and Vega, I. (2011). "The motion of point
  particles in curved spacetime." *Living Rev. Rel.* 14, 7.
  arXiv:1102.0529.
- Visser, M. (1993). "Van Vleck determinants: geodesic focusing
  and defocusing in Lorentzian spacetimes." arXiv:gr-qc/9303020.

## 8. Connections

- **OP-42 (A_E universality):** The exact predicate is the missing
  ingredient for a definitive universality test. If A_E(Sch, exact) /
  A_E(pp-wave) = 1.0, universality is confirmed. If it remains
  approximately 0.5, the non-universality is physical.
- **OP-37 (polarization anisotropy):** The exact predicate has
  different anisotropy properties from the jet predicate, affecting
  the polarization ratio.
- **OP-34 (N-scaling):** The exact predicate may modify the effective
  N-scaling exponent if boundary effects change.
