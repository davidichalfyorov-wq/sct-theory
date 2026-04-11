/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: cda26c74-2cf8-4ebe-817e-88150362264b

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_ghost_subtraction :
    (7 : ℝ) / 60 - 2 * ((1 : ℝ) / 120) = (1 : ℝ) / 10
-/

import Mathlib.Tactic


/-- Ghost subtraction: 7/60 - 2*(1/120) = 7/60 - 1/60 = 1/10 -/
theorem sct_ghost_subtraction :
    (7 : ℝ) / 60 - 2 * ((1 : ℝ) / 120) = (1 : ℝ) / 10 := by
  grind