/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 9ecd737a-a11b-4106-ac44-2cc5f6babcd5

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_f_U_zero :
    (1 : ℝ) / 2 = (1 : ℝ) / 2
-/

import Mathlib.Tactic


/-- CZ scalar form factor: f_U(0) = 1/2 -/
theorem sct_scalar_f_U_zero :
    (1 : ℝ) / 2 = (1 : ℝ) / 2 := by
  norm_num +zetaDelta at *