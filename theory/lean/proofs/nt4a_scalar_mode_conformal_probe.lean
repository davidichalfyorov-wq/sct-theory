/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 05a7c29e-9a9d-4e1d-afa8-8babaf267c27

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

Aristotle encountered an error processing this file.
Lean errors:
At line 6, column 16:
  unexpected token ')'; expected '_' or identifier
-/

import Mathlib.Tactic

/-- probe -/
theorem nt4a_scalar_mode_conformal_probe :
    6 * (((1 : ?) / 6) - 1 / 6) ^ 2 = (0 : ?) := by
  /-
  ERROR 1:
  unexpected token ')'; expected '_' or identifier
  -/
  sorry
