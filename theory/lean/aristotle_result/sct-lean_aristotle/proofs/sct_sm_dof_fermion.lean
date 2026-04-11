/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: f29d4f65-f398-44b2-a561-a67b452b34c7

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_sm_dof_fermion :
    (45 : ℝ) = (45 : ℝ)
-/

import Mathlib.Tactic


/-- SM fermion d.o.f.: N_f = 45 Dirac (Weyl->Dirac convention) -/
theorem sct_sm_dof_fermion :
    (45 : ℝ) = (45 : ℝ) := by
  rfl