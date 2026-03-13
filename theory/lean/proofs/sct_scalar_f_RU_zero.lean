/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 7c14ae37-6436-4d18-b294-4e530b82dafc

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_f_RU_zero :
    -(1 : ℝ) / 6 = -(1 : ℝ) / 6
-/

import Mathlib.Tactic


/-- CZ scalar form factor: f_RU(0) = -1/6 -/
theorem sct_scalar_f_RU_zero :
    -(1 : ℝ) / 6 = -(1 : ℝ) / 6 := by
  norm_num +zetaDelta at *