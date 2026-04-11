import Mathlib.Tactic

/-- The original statement is false: when C + g = 0 (e.g., C = 1, g = -1),
    LHS = 1/0 = 0 but RHS = 1/C - g/(C*0) = 1/C ≠ 0.
    The corrected version adds the hypothesis `hg : C + g ≠ 0`. -/
-- theorem inverse_decomposition {C : ℝ} (hC : C ≠ 0) (g : ℝ) :
--     1 / (C + g) = 1 / C - g / (C * (C + g)) := by
--   sorry

theorem inverse_decomposition {C : ℝ} (hC : C ≠ 0) (g : ℝ) (hg : C + g ≠ 0) :
    1 / (C + g) = 1 / C - g / (C * (C + g)) := by
  field_simp
  ring

theorem constant_propagator_decomposition {C : ℝ} (hC : C ≠ 0) (g : ℝ)
    (hg : C + g ≠ 0) :
    1 / (C + g) = 1 / C * (1 - g / (C + g)) := by
  field_simp
  ring
