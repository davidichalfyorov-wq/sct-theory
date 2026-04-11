/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 2e17b11d-9bfe-40db-92fd-f2f66ed3c9df

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_sm_dof_scalar :
    (4 : ℝ) = (4 : ℝ)
-/

import Mathlib.Tactic


/-- SM scalar d.o.f.: N_s = 4 (real Higgs doublet) -/
theorem sct_sm_dof_scalar :
    (4 : ℝ) = (4 : ℝ) := by
  rfl