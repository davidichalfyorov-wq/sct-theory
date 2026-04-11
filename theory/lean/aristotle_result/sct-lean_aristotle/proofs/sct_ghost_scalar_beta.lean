/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 9961054a-77cc-44b3-80aa-e9630f243b14

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_ghost_scalar_beta :
    (1 : ℝ) / 120 = (1 : ℝ) / 120
-/

import Mathlib.Tactic


/-- Single FP ghost = scalar contribution: beta_W^(ghost) = 1/120 -/
theorem sct_ghost_scalar_beta :
    (1 : ℝ) / 120 = (1 : ℝ) / 120 := by
  -- This is trivially true.
  norm_num at *