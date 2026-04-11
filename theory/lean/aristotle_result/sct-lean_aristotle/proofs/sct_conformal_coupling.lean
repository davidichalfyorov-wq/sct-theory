/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 2c13b0ba-ccd3-40cd-8fe7-f8db7845e098

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_conformal_coupling_beta_R_zero :
    (1 : ℝ) / 2 * ((1 : ℝ) / 6 - (1 : ℝ) / 6) ^ 2 = 0
-/

import Mathlib.Tactic


/-- At conformal coupling xi = 1/6, the Ricci beta function vanishes:
    beta_R^(0)(xi) = (1/2)*(xi - 1/6)^2 implies beta_R^(0)(1/6) = 0. -/
theorem sct_conformal_coupling_beta_R_zero :
    (1 : ℝ) / 2 * ((1 : ℝ) / 6 - (1 : ℝ) / 6) ^ 2 = 0 := by
  norm_num +zetaDelta at *