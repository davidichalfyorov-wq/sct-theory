/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 18aaa4bc-0e03-4ab3-b76c-7f313ab1408c

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_conformal_coupling :
    (1 : ℝ) / 2 * ((1 : ℝ) / 6 - (1 : ℝ) / 6) ^ 2 = (0 : ℝ)
-/

import Mathlib.Tactic


/-- At conformal coupling xi = 1/6, scalar Ricci beta vanishes: beta_R^(0)(1/6) = (1/2)(1/6 - 1/6)^2 = 0 -/
theorem sct_scalar_conformal_coupling :
    (1 : ℝ) / 2 * ((1 : ℝ) / 6 - (1 : ℝ) / 6) ^ 2 = (0 : ℝ) := by
  grind