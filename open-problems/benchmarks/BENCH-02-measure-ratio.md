---
id: BENCH-02
title: "Diamond midpoint measure discrepancy (3/10 ratio)"
domain: [mathematics]
difficulty: medium
status: resolved
resolution-date: 2026-03-31
verification: "Analytical + Monte Carlo (50 seeds, N=50000, result 0.2999±0.0003)"
---

# BENCH-02: Diamond midpoint measure discrepancy

## Problem

Compute the ratio

  I_bulk / (V₄/9!)

where I_bulk is the four-volume integral of the kernel product
k₄(τ₋/T) k₄(τ₊/T) over the interior of a 4D causal diamond, and
V₄ = πT⁴/24 is the diamond volume.

Here k₄(x) = x⁴/4!, and τ₋, τ₊ are proper times from the past
and future tips to a point inside the diamond.

## Context

This integral compares two measures for the beta-function overlap:
- **Uniform split measure:** ∫₀¹ k₄(s) k₄(1−s) ds = 1/9!
- **Geometric four-volume measure:** ∫_{D_T} k₄(τ₋/T) k₄(τ₊/T) dV₄

The ratio quantifies how much the CJ stratification scheme must
correct the geometric measure to recover the uniform split measure.

## Known Solution

In null coordinates u=t+r, v=t−r with a = u/T + 1/2, b = v/T + 1/2:

  τ₋² = T²ab,  τ₊² = T²(1−a)(1−b)

  I_bulk = [πT⁴/(2×24²)] ∫₀¹ da ∫₀ᵃ db (a−b)² a² b² (1−a)² (1−b)²
         = πT⁴/29030400

  V₄/9! = πT⁴/8709120

  **I_bulk / (V₄/9!) = 3/10 exactly.**

Verified by Monte Carlo: 50 independent realisations of 50000 points
each give 0.2999 ± 0.0003.

## Purpose as Benchmark

Tests ability to:
- Set up integrals in null coordinates for a causal diamond
- Evaluate nested polynomial integrals exactly
- Recognise that the geometric midpoint measure gives 30% of the
  uniform split measure (a non-obvious exact result)
