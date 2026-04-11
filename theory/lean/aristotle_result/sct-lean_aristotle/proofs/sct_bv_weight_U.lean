/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 0f224ad5-b02a-421a-9d5f-7c3b7469508d

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_bv_weight_U :
    (1 : ℝ) / 2 = (1 : ℝ) / 2
-/

import Mathlib.Tactic


/-- BV parametric weight: Phi_U = 1/2 -/
theorem sct_bv_weight_U :
    (1 : ℝ) / 2 = (1 : ℝ) / 2 := by
  -- This is trivially true.
  norm_num at *