/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: b8227247-b34c-4d1f-8f20-1c496586be4c

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt4a_c2_local :
    2 * ((13 : ℚ) / 120) = (13 : ℚ) / 60
-/

import Mathlib.Tactic


/-- NT-4a spin-2 local coefficient c2 = 13/60 -/
theorem nt4a_c2_local :
    2 * ((13 : ℚ) / 120) = (13 : ℚ) / 60 := by
  norm_num +zetaDelta at *