/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 8b30912f-cd03-4b99-93ba-e434387cc1ea

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt2_hc_scalar_pole_cancels :
    (1 : ℚ) / 12 + (-(1 : ℚ) / 6) / 2 = (0 : ℚ)
-/

import Mathlib.Tactic


/-- NT-2 scalar Weyl pole cancellation at z = 0 -/
theorem nt2_hc_scalar_pole_cancels :
    (1 : ℚ) / 12 + (-(1 : ℚ) / 6) / 2 = (0 : ℚ) := by
  -- Combine like terms and simplify the expression.
  field_simp
  ring_nf at *