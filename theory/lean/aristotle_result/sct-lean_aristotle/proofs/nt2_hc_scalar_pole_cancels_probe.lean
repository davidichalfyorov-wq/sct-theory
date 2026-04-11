/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: c75287db-ca79-40bd-b725-b15d241ca253

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>
-/

import Mathlib.Tactic

/-- probe -/
theorem nt2_hc_scalar_pole_cancels_probe :
    (1 : ℚ) / 12 + (-(1 : ℚ) / 6) / 2 = (0 : ℚ) := by
  norm_num
