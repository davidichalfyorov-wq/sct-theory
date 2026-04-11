/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 67e75714-e19b-49e9-bbf7-de1f76e6ade9

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_dirac_beta_weyl :
    (1 : ℝ) / 20 = (1 : ℝ) / 20
-/

import Mathlib.Tactic


/-- Dirac Weyl beta: beta_W^(1/2) = 1/20 -/
theorem sct_dirac_beta_weyl :
    (1 : ℝ) / 20 = (1 : ℝ) / 20 := by
  norm_num +zetaDelta at *