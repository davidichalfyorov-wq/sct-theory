# Numerical: Background Briefing

The numerical arm of the SCT project centers on causal set computations
that test the connection between the continuum spectral action and
discrete quantum gravity structures. The primary computational
framework is the Geodesic-Tidal-Alignment (GTA) approach, which uses
Poisson sprinklings on Lorentzian manifolds to probe curvature
signatures through path statistics on Hasse diagrams.

## Computational infrastructure

**Hardware.** Intel i9-12900KS (16 cores, 24 threads, 5.5 GHz boost),
64 GB DDR5-4000, NVIDIA RTX 3090 Ti (24 GB VRAM), approximately
13 TB NVMe storage. GPU acceleration via CuPy (CUDA 12.6) provides
3x speedup for matrix operations at N >= 5000.

**Software stack.** Python 3.12 with NumPy, SciPy, CuPy. The
sct_tools package (v0.7.0) provides:
- `sct_tools.hasse`: bitset-packed Hasse diagram construction (27x
  faster than dense C@C matrix), with functions `build_hasse_bitset`,
  `path_kurtosis_from_lists`, `crn_trial_bitset`, `run_aeff_ensemble`,
  and `sprinkle_diamond`.
- GPU-accelerated causal matrix construction and interval volume
  computation (C^2 = C @ C matmul).
- Lean 4 formal verification via 3-backend system.

**Performance benchmarks.**
- N = 10000 causal matrix + C^2: 1.0 s (GPU) vs 2.9 s (CPU).
- N = 20000: 7.6 s (GPU) vs 25 s (CPU).
- N = 50000: fits in 24 GB VRAM (two float32 matrices, 20 GB total).
- Bitset Hasse builder: 27x faster than dense approach for path
  kurtosis extraction.

## The CJ estimator and GTA framework

The CJ (stratified covariance) estimator is a causal set observable
designed to detect curvature through path statistics:

  CJ = sum_B w_B Cov_B(X, delta Y^2)

where B indexes strata (bins by ancestor/descendant count), w_B is the
stratum weight, X is a depth product proxy, and delta Y is the
link-score deviation within each stratum. CJ is zero on flat spacetime
(by the SO(3) selection rule, Lemma 5.1) and nonzero on curved
backgrounds.

The GTA framework extends CJ by introducing:
- Effective screening depth b_eff = 5 (from data).
- Noise amplitude sigma_0 ~ 0.299 x N^{1/4}.
- Signal model: dk ~ 0.019 q^2, noise ~ 0.016 q^{0.83}.
- Four vacuum families tested: pp-wave, Schwarzschild, de Sitter,
  Kottler (partial).

## Exact predicates

**PP-wave:** exact causal predicate available in Brinkmann coordinates
for h_{xx} = epsilon x_3^2. This gives exact causal relations without
numerical geodesic integration, enabling high-precision measurements.

**Schwarzschild:** only a jet predicate (order-2 RNC expansion) is
available. The jet predicate introduces systematic errors at large
separations. An exact Schwarzschild predicate requires numerical
null geodesic integration, which is computationally expensive but
feasible.

## Key numerical results (FND-1 program, closed)

- PP-wave T^4 scaling: ratio/T^4 ~ 1.3-1.7 (verified).
- Schwarzschild local T^4: ratio/T^4 = 0.998 at T = 0.70 (3.5 sigma,
  M = 200 sprinklings).
- De Sitter Ricci subtraction: exact zero (M = 50, all T).
- Schwarzschild production (M = 0.10): dk = +0.030 at T = 1 (9.3 sigma).
- N-scaling: CJ proportional to N^alpha with alpha = 0.955 +/- 0.027
  in [500, 10000].
- Link embedding: rho = 0.936, (1 - rho) ~ N^{-0.53} (strong geometry).
- A_E mismatch: A_E(pp-wave) ~ 0.036 vs A_E(Schwarzschild) ~ 0.007
  (factor 3-5x, attributed to predicate differences).

## Open numerical work

Five classes of numerical investigation remain: (1) Kottler
(Schwarzschild-de Sitter) test for R x Weyl cross-term; (2) boosted
Schwarzschild systematic for magnetic Weyl contribution; (3) third
vacuum family (Godel, Bianchi IX) and d = 3 test; (4) large-N
(N >= 20000) CJ data for exponent refinement; (5) seven blocked
FND-1 tests from the original experimental plan.
