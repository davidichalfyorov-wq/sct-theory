/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 006a3c28-cad0-412d-82a0-34298c31c31b

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem nt2_hc_vector_pole_cancels :
    (1 : ℚ) / 6 + (-(1 : ℚ) / 6) = (0 : ℚ)
-/

import Mathlib.Tactic


/-- NT-2 vector Weyl pole cancellation at z = 0 -/
theorem nt2_hc_vector_pole_cancels :
    (1 : ℚ) / 6 + (-(1 : ℚ) / 6) = (0 : ℚ) := by
  native_decide +revert