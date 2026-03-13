/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: f447f494-9701-4473-9066-b7aaec5d62f6

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_f_R_zero :
    (1 : ℝ) / 120 = (1 : ℝ) / 120
-/

import Mathlib.Tactic


/-- CZ scalar form factor: f_R(0) = 1/120 -/
theorem sct_scalar_f_R_zero :
    (1 : ℝ) / 120 = (1 : ℝ) / 120 := by
  norm_num +zetaDelta at *