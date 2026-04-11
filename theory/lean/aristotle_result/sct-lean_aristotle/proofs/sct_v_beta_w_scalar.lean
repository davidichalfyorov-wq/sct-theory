/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 6db92cd2-6c65-44e8-ac35-d03f94003bdf

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_v_beta_w_scalar :
    (1 : ℚ) / 120 = (1 : ℚ) / 120
-/

import Mathlib.Tactic


/-- beta_W_scalar -/
theorem sct_v_beta_w_scalar :
    (1 : ℚ) / 120 = (1 : ℚ) / 120 := by
  norm_num +zetaDelta at *