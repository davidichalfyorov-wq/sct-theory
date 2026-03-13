/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: ea7123b0-cab0-46ef-9c86-825edbccc801

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_beta_weyl :
    (1 : ℝ) / 120 = (1 : ℝ) / 120
-/

import Mathlib.Tactic


/-- Scalar Weyl beta: beta_W^(0) = 1/120 -/
theorem sct_scalar_beta_weyl :
    (1 : ℝ) / 120 = (1 : ℝ) / 120 := by
  norm_num +zetaDelta at *