---
id: BENCH-01
title: "Beta-function overlap identity for all d"
domain: [mathematics]
difficulty: medium
status: resolved
resolution-date: 2026-03-31
verification: "Lean 4 sorry-free (cj_bridge_general_d.lean)"
---

# BENCH-01: Beta-function overlap identity for all d

## Problem

Prove that for all natural numbers d,

  (d!)² × C(2d,d) × (2d+1) = (2d+1)!

where C(2d,d) is the central binomial coefficient.

Equivalently, show that

  ∫₀¹ [s^d (1−s)^d / (d!)²] ds = 1/(2d+1)!

## Context

This identity encodes the beta-function overlap B(d+1,d+1) = (d!)²/(2d+1)!
and arises as the volume of the ordered (2d+1)-simplex formed by
concatenating d past, 1 split, and d future ordered coordinates.

In d=4, it gives the factor 1/9! = 1/362880 in the CJ bridge formula.

## Known Solution

The proof uses:
1. Mathlib's `Nat.choose_mul_factorial_mul_factorial`: C(2d,d)×d!×d! = (2d)!
2. Factorial recurrence: (2d+1)! = (2d+1)×(2d)!
3. Algebraic rearrangement via `calc` chain.

The full proof is in `theory/lean/proofs/cj_bridge_general_d.lean` and
is sorry-free. It has been independently verified by the Aristotle
automated theorem prover (two separate projects).

## Purpose as Benchmark

This problem tests the ability to:
- Work with combinatorial identities involving factorials and binomial coefficients
- Connect integral representations (beta function) to discrete identities
- Use the (2d+1)-simplex geometric interpretation

A successful solution should produce either a direct algebraic proof or
a proof via the beta integral B(d+1,d+1) = Γ(d+1)²/Γ(2d+2).
