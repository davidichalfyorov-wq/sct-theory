/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 685f5c2d-97cd-47fa-818d-fa8cae297a84

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt2_total_weyl_beta :
    (13 : ℚ) / 120 = (13 : ℚ) / 120
-/

import Mathlib.Tactic


/-- NT-2 local Weyl coefficient inherited from Phase 3 -/
theorem nt2_total_weyl_beta :
    (13 : ℚ) / 120 = (13 : ℚ) / 120 := by
  -- This is trivially true.
  norm_num at *