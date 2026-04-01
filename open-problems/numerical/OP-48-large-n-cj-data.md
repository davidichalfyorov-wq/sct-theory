---
id: OP-48
title: "Large-N CJ data (N >= 20000) for scaling exponent refinement"
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

# OP-48: Large-N CJ data (N >= 20000) for scaling exponent refinement

## 1. Statement

Extend the CJ estimator measurements to N >= 20000 elements to
narrow the uncertainty on the N-scaling exponent alpha and
discriminate between the competing models alpha = 8/9 (power law)
and alpha = 1 with logarithmic correction (N / log(N)^{0.39}).
This requires optimizing the Hasse diagram builder for large N,
potentially through a C/Cython bitset implementation.

## 2. Context

The CJ estimator scales empirically as CJ proportional to N^alpha.
Current data in the range N = 500 to 10000 give
alpha = 0.955 +/- 0.027. Two models fit the data comparably:

1. **Pure power law:** CJ proportional to N^{8/9} ~ N^{0.889}.
   The heuristic 9-simplex argument predicts alpha = 8/9 in d = 4.

2. **Logarithmic correction:** CJ proportional to N / log(N)^{0.39}.
   This gives an effective exponent that approaches 1 from below as
   N increases.

The distinction between these models requires data at N >= 20000,
where the effective exponent alpha_eff(N) = d ln(CJ) / d ln(N)
differs by approximately 0.03 between the two models -- a difference
that becomes resolvable with M >= 50 sprinklings per N.

The computational bottleneck is the Hasse diagram construction. The
current Python bitset implementation in sct_tools.hasse runs at
O(N^2 x N/64) using uint64 ancestor bitsets. For N = 20000, this
requires approximately 20000^2 x 20000/64 ~ 1.25 x 10^{11} bitwise
operations -- feasible but slow in Python (estimated 30+ minutes per
sprinkling on CPU). A C or Cython implementation would reduce this
to under 1 minute.

## 3. Known Results

- N = 500-10000 data: alpha = 0.955 +/- 0.027 (M = 50 per N).
- N = 500-2000 sub-range: alpha ~ 0.87.
- N = 5000-10000 sub-range: alpha ~ 1.00.
- The upward drift in alpha with N-range is consistent with both
  models.
- GPU acceleration (CuPy) handles the causal matrix (C) and interval
  volume (C^2) at N = 20000 in 7.6 s. The bottleneck is the Hasse
  diagram (transitive reduction), not the causal matrix.
- N = 15000 has been achieved for isolated runs but not with M = 50
  statistical ensemble.
- The RTX 3090 Ti VRAM (24 GB) accommodates N = 50000 for the causal
  matrix but the Hasse builder is CPU-bound.

## 4. Failed Approaches

1. **Dense C@C approach.** The original method computed C^2 = C @ C
   as a dense matrix multiply to identify links (L_{ij} = C_{ij}
   if (C^2)_{ij} = 0). This is O(N^3) and becomes memory-prohibitive
   at N >= 30000 (two N x N float32 matrices = 7.2 GB at N = 30000).
   The bitset approach replaced this successfully up to N = 15000.

2. **GPU Hasse builder.** Attempted to port the bitset approach to
   CuPy. The per-element bitset operations do not parallelise
   efficiently on GPU because the Hasse update is inherently
   sequential (topological order traversal). Speedup was only 1.3x
   versus CPU, insufficient.

## 5. Success Criteria

- CJ data at N = {5000, 10000, 15000, 20000, 30000}, each with
  M >= 50 sprinklings, on pp-wave (exact predicate).
- Updated fit of alpha with the extended N-range.
- Model comparison (AIC or BIC) between pure power law and
  logarithmic correction models.
- If alpha = 8/9: the 9-simplex heuristic is supported.
  If alpha = 1: the logarithmic correction model is favored.
- Hasse builder performance: under 5 minutes per sprinkling at
  N = 30000 (current Python implementation extrapolation: ~90 min).

## 6. Suggested Directions

1. C/Cython bitset builder: port the critical inner loop of
   `build_hasse_bitset` to C via Cython. The bitwise AND and
   population count operations vectorise well. Expected speedup:
   20-50x, bringing N = 30000 to under 2 minutes.

2. Sparse Hasse builder: at large N, the Hasse diagram is sparse
   (average degree ~ c_4 N^{1/(d+1)} << N). Exploit sparsity by
   storing only the link list instead of the full ancestor bitset.
   This trades time for memory and may enable N = 50000.

3. Multi-GPU: distribute the causal matrix construction across
   multiple GPUs (if available) using CuPy. Each GPU handles a
   block of the N x N matrix.

4. Extrapolation method: instead of running at N = 30000 directly,
   fit the N-dependence of alpha_eff(N) using the existing data and
   extrapolate. This is less rigorous but gives a quick estimate.

## 7. References

1. Bollobas, B. and Brightwell, G. (1991). "The width of random graph
   orders." Rand. Struct. Alg. 2, 37-49.
2. Glaser, L. and Surya, S. (2013). "Towards a definition of locality
   in a Manifoldlike Causal Set." Phys. Rev. D 88, 124026.
   arXiv:1309.3403.

## 8. Connections

- **Directly addresses OP-34** (N-scaling exponent): large-N data
  is the primary method to resolve the exponent.
- Related to **OP-45** (Kottler), **OP-46** (boosted Schwarzschild),
  **OP-47** (third family): all require the same Hasse infrastructure.
- An optimized Hasse builder would benefit all numerical OP problems.
