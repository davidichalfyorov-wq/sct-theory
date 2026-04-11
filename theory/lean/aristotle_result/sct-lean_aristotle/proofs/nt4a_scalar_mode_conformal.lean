/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 89dc5273-d5fe-45fc-b624-fb13f80d8c4b

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt4a_scalar_mode_conformal :
    6 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = (0 : ℚ)
-/

import Mathlib.Tactic


/-- NT-4a scalar mode decoupling at conformal coupling -/
theorem nt4a_scalar_mode_conformal :
    6 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = (0 : ℚ) := by
  norm_num +zetaDelta at *