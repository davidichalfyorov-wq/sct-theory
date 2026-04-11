/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 5a0e9771-d0e9-420f-8750-47591c71f67b

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt2_hc_dirac_pole_cancels :
    (1 : ℚ) / 3 + 2 * (-(1 : ℚ) / 6) = (0 : ℚ)
-/

import Mathlib.Tactic


/-- NT-2 Dirac Weyl pole cancellation at z = 0 -/
theorem nt2_hc_dirac_pole_cancels :
    (1 : ℚ) / 3 + 2 * (-(1 : ℚ) / 6) = (0 : ℚ) := by
  norm_num +zetaDelta at *