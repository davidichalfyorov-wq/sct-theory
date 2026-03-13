/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 99ed5319-40cb-4e70-8c50-acb451934f18

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_f_Rbis_zero :
    (1 : ℝ) / 3 * ((1 : ℝ) / 60) + (1 : ℝ) / 120 = (1 : ℝ) / 72
-/

import Mathlib.Tactic


/-- CZ scalar combined: f_{R,bis}(0) = (1/3)*f_Ric(0) + f_R(0) = 1/180 + 1/120 = 1/72 -/
theorem sct_scalar_f_Rbis_zero :
    (1 : ℝ) / 3 * ((1 : ℝ) / 60) + (1 : ℝ) / 120 = (1 : ℝ) / 72 := by
  norm_num +zetaDelta at *