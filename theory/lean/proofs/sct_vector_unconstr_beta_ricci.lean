/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: eee9aeea-d7f3-46e1-99de-3a73ef55e7cb

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_vector_unconstr_beta_ricci :
    (1 : ℝ) / 36 = (1 : ℝ) / 36
-/

import Mathlib.Tactic


/-- Unconstrained vector Ricci beta: beta_R^(unconstr) = 1/36 -/
theorem sct_vector_unconstr_beta_ricci :
    (1 : ℝ) / 36 = (1 : ℝ) / 36 := by
  norm_num +zetaDelta at *