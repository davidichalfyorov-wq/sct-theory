---
id: OP-45
title: "Kottler (Schwarzschild-de Sitter) cross-term test"
domain: [numerics, causal-sets]
difficulty: easy
status: open
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: [FND-1]
papers: ["1212.0631"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-45: Kottler (Schwarzschild-de Sitter) cross-term test

## 1. Statement

Run the full CJ estimator on Poisson sprinklings of the Kottler
(Schwarzschild-de Sitter) spacetime with varying cosmological constant
Lambda_cc. Quantify the coefficient of the R x Weyl cross-term in
the CJ response and determine whether the estimator decomposes
cleanly into additive Weyl and Ricci contributions.

## 2. Context

The Kottler metric is the unique static spherically symmetric vacuum
solution with cosmological constant:

  ds^2 = -(1 - 2M/r - Lambda_cc r^2/3) dt^2 + dr^2/(1 - 2M/r - Lambda_cc r^2/3) + r^2 dOmega^2.

This spacetime has both nonzero Weyl curvature (from the mass M) and
nonzero Ricci scalar (R = 4 Lambda_cc). It therefore provides a
controlled environment to test whether CJ responds additively to
Weyl and Ricci curvature or whether a cross-term R x C_{mu nu rho sigma}
is present.

Preliminary results from a pilot run showed a 22% reduction in the
CJ signal compared to the pure Schwarzschild case at the same M,
suggesting a nontrivial interaction between the R and C^2 channels.
However, this preliminary measurement used only a single Lambda_cc
value and small ensemble size (M < 20 sprinklings).

The GTA signal model predicts CJ proportional to T^4 (A_E E^2 + A_B B^2).
On Kottler, the Weyl invariants E^2 and B^2 are identical to those
on Schwarzschild (since Lambda_cc enters only through the Ricci tensor),
so any deviation from the pure Schwarzschild CJ must arise from a
Ricci-dependent correction to the estimator.

## 3. Known Results

- Schwarzschild CJ: well-characterized at N = 1000-10000, M = 50-200.
  dk ~ +0.030 at T = 1, quadratic in q_W.
- De Sitter CJ: exact zero after Ricci subtraction (M = 50, all T).
  The Ricci subtraction procedure removes the R contribution entirely.
- Kottler pilot: 22% reduction relative to Schwarzschild at
  Lambda_cc = 0.01 / r_s^2, N = 2000, M = 20. Not statistically
  definitive.
- The Kottler causal predicate requires modification of the
  Schwarzschild jet predicate to include the Lambda_cc correction
  to the metric functions.

## 4. Failed Approaches

The pilot run used too small an ensemble (M = 20) and a single
Lambda_cc value, making it impossible to determine whether the 22%
reduction is a systematic effect or a statistical fluctuation. No
Lambda_cc scan has been performed.

## 5. Success Criteria

- CJ measurement on Kottler at 5+ Lambda_cc values spanning
  Lambda_cc = 0 (Schwarzschild) to Lambda_cc = 0.1 / r_s^2, with
  M >= 50 sprinklings per point and N >= 2000.
- Extraction of the cross-term coefficient: CJ(Kottler) =
  CJ(Schwarzschild) + beta_cross R Lambda_cc + O(Lambda_cc^2).
  Determine beta_cross with statistical uncertainty.
- Verification that CJ(Lambda_cc = 0) reproduces the pure
  Schwarzschild result within errors.
- Statement of whether the cross-term is consistent with zero
  (additive decomposition holds) or significantly nonzero.

## 6. Suggested Directions

1. Lambda_cc scan: run CJ at Lambda_cc / (M^{-2}) = {0, 0.001,
   0.003, 0.01, 0.03, 0.1} with fixed M and N = 5000, M_spr = 50.
   Fit CJ(Lambda_cc) to a polynomial in Lambda_cc.

2. Ricci subtraction check: apply the de Sitter Ricci subtraction
   procedure to the Kottler CJ. If the subtracted signal equals the
   Schwarzschild CJ, the decomposition is additive. If not, the
   cross-term is physical.

3. Separate E^2 and R channels: on Kottler, E^2 and R are controlled
   by different parameters (M and Lambda_cc). Vary them independently
   to disentangle the two contributions.

## 7. References

1. Roy, M., Sinha, D. and Surya, S. (2013). "Discrete geometry of a
   small causal diamond." arXiv:1212.0631.
2. Kottler, F. (1918). "Uber die physikalischen Grundlagen der
   Einsteinschen Gravitationstheorie." Ann. Phys. 361, 401.

## 8. Connections

- Related to **OP-34** (N-scaling exponent): the Kottler test uses
  the same CJ estimator and Hasse infrastructure.
- Related to **OP-47** (third vacuum family): Kottler is a natural
  next step after Schwarzschild and pp-wave.
- Related to **OP-49** (blocked FND-1 tests): Kottler is test 2.2
  in the original FND-1 plan.
