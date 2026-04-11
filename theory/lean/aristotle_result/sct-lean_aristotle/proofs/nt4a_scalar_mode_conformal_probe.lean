/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 05a7c29e-9a9d-4e1d-afa8-8babaf267c27

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>
-/

import Mathlib.Tactic

/-- probe -/
theorem nt4a_scalar_mode_conformal_probe :
    6 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = (0 : ℚ) := by
  norm_num
