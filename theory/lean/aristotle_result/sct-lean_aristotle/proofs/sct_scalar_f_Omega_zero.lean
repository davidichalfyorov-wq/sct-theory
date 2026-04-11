/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: da667a43-b3a2-4b23-8817-81382d8ab58c

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_scalar_f_Omega_zero :
    (1 : ℝ) / 12 = (1 : ℝ) / 12
-/

import Mathlib.Tactic


/-- CZ scalar form factor: f_Omega(0) = 1/12 -/
theorem sct_scalar_f_Omega_zero :
    (1 : ℝ) / 12 = (1 : ℝ) / 12 := by
  norm_num +zetaDelta at *