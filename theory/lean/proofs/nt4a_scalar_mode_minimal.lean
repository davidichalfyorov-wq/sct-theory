/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: b06e7255-4afd-44b3-ae37-d2c1737c2acc

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt4a_scalar_mode_minimal :
    6 * ((0 : ℚ) - 1 / 6) ^ 2 = (1 : ℚ) / 6
-/

import Mathlib.Tactic


/-- NT-4a scalar mode coefficient at minimal coupling -/
theorem nt4a_scalar_mode_minimal :
    6 * ((0 : ℚ) - 1 / 6) ^ 2 = (1 : ℚ) / 6 := by
  norm_num +zetaDelta at *