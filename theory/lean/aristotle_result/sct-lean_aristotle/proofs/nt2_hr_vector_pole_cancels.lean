/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: cf5c75f1-6d2a-4a34-b004-734c79154c6e

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt2_hr_vector_pole_cancels :
    (5 : ℚ) / 72 + 5 * (-(1 : ℚ) / 6) / 12 = (0 : ℚ)
-/

import Mathlib.Tactic


/-- NT-2 vector Ricci pole cancellation at z = 0 -/
theorem nt2_hr_vector_pole_cancels :
    (5 : ℚ) / 72 + 5 * (-(1 : ℚ) / 6) / 12 = (0 : ℚ) := by
  norm_num [ div_eq_mul_inv ] at *