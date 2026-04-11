/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: d7dc085d-77c5-41c1-a935-9c361785d9c1

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_vector_beta_weyl :
    ((14 : ℝ) - 2) / 120 = (1 : ℝ) / 10
-/

import Mathlib.Tactic


/-- Vector Weyl beta with ghost subtraction: beta_W^(1) = (14 - 2*1)/120 = 12/120 = 1/10 -/
theorem sct_vector_beta_weyl :
    ((14 : ℝ) - 2) / 120 = (1 : ℝ) / 10 := by
  norm_num +zetaDelta at *