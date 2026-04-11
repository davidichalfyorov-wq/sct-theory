/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 95c3bb29-cd8b-4e25-afc3-637324d6f389

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt4a_pi_tt_zero :
    (1 : ℚ) + 0 = (1 : ℚ)
-/

import Mathlib.Tactic


/-- NT-4a propagator normalization Pi_TT(0) = 1 -/
theorem nt4a_pi_tt_zero :
    (1 : ℚ) + 0 = (1 : ℚ) := by
  norm_num +zetaDelta at *