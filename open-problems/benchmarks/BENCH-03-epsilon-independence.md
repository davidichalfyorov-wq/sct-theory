---
id: BENCH-03
title: "CJ proportionality to E² (epsilon independence)"
domain: [numerics, theory]
difficulty: easy
status: resolved
resolution-date: 2026-03-31
verification: "Numerical: CV(CJ/E²) = 1.4% for ε≥3, power law slope 1.97±0.01"
---

# BENCH-03: CJ proportionality to E²

## Problem

Verify that CJ/E² is independent of the curvature strength parameter ε,
confirming that CJ is proportional to the electric Weyl invariant E²
at leading order.

## Context

The CJ bridge formula predicts CJ ∝ E² × (other factors independent of
curvature strength). For the pp-wave metric with parameter ε, the
electric Weyl invariant scales as E² ∝ ε². Therefore CJ/ε² should be
constant across a range of ε values.

The validity range is ξ = ε/N^{1/4} ∈ [0.3, 0.9]. Below this range,
noise dominates; above, non-perturbative effects appear.

## Known Solution

At N=2000, T=1, with 20 seeds per ε value:

| ε | E² | CJ/E² |
|---|-----|-------|
| 1 | 0.5 | 6.88×10⁻³ (noise-dominated, ξ=0.15) |
| 2 | 2.0 | 5.39×10⁻³ |
| 3 | 4.5 | 4.86×10⁻³ |
| 4 | 8.0 | 4.72×10⁻³ |
| 5 | 12.5 | 4.73×10⁻³ |
| 6 | 18.0 | 4.72×10⁻³ |
| 8 | 32.0 | 4.70×10⁻³ |

For ε ≥ 3: coefficient of variation = 1.4%.
Power-law fit CJ ∝ ε^β: β = 1.97 ± 0.01 (consistent with β=2).

## Purpose as Benchmark

Tests ability to:
- Understand the validity range of the perturbative regime
- Recognise that the ε=1 outlier is a noise floor effect, not a failure
- Correctly identify the E² proportionality from numerical data
