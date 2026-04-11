/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: e9993451-e61f-4496-b147-98116640bc9f

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_vector_unconstr_beta_weyl_equiv :
    (14 : ℝ) / 120 = (7 : ℝ) / 60
-/

import Mathlib.Tactic


/-- Reduction: 14/120 = 7/60 -/
theorem sct_vector_unconstr_beta_weyl_equiv :
    (14 : ℝ) / 120 = (7 : ℝ) / 60 := by
  norm_num +zetaDelta at *