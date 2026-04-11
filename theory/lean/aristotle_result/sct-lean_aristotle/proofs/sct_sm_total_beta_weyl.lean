/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 3970478b-b4f3-4c28-b798-b00fe6322af2

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_sm_total_beta_weyl :
    (4 : ℝ) * ((1 : ℝ) / 120) + (45 : ℝ) / 2 * ((1 : ℝ) / 20) + (12 : ℝ) * ((1 : ℝ) / 10) = (4 : ℝ) / 120 + (45 : ℝ) / 40 + (12 : ℝ) / 10
-/

import Mathlib.Tactic


/-- SM total Weyl beta: N_s*beta_W^(0) + (N_f/2)*beta_W^(1/2) + N_v*beta_W^(1) -/
theorem sct_sm_total_beta_weyl :
    (4 : ℝ) * ((1 : ℝ) / 120) + (45 : ℝ) / 2 * ((1 : ℝ) / 20) + (12 : ℝ) * ((1 : ℝ) / 10) = (4 : ℝ) / 120 + (45 : ℝ) / 40 + (12 : ℝ) / 10 := by
  -- Combine like terms and simplify the expression.
  ring_nf at *