/-
This file was edited by Aristotle (https://aristotle.harmonic.fun).

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 9d9c65a0-8e62-4033-8d34-f15889e396e0

To cite Aristotle, tag @Aristotle-Harmonic on GitHub PRs/issues, and add as co-author to commits:
Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>

The following was proved by Aristotle:

- theorem sct_spectral_normalization :
    (1 : ℝ) / (16 * Real.pi ^ 2) = (1 : ℝ) / (16 * Real.pi ^ 2)
-/

import Mathlib.Tactic


/-- Spectral action normalization: F_i(z) = h_i(z) / (16*pi^2) -/
theorem sct_spectral_normalization :
    (1 : ℝ) / (16 * Real.pi ^ 2) = (1 : ℝ) / (16 * Real.pi ^ 2) := by
  rfl