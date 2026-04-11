/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 19feedb3-7f96-4ea6-8276-9cd9189305b1

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_f_Ric_zero :
    (1 : ℝ) / 60 = (1 : ℝ) / 60
-/

import Mathlib.Tactic


/-- CZ scalar form factor: f_Ric(0) = 1/60 -/
theorem sct_scalar_f_Ric_zero :
    (1 : ℝ) / 60 = (1 : ℝ) / 60 := by
  -- This is trivially true.
  norm_num