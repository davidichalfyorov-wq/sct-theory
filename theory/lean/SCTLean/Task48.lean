import Mathlib.Tactic

theorem rank_one_quartic_unique (a : ℚ) (ha : a ≠ 0) :
    (9 * a ^ 4 / 8) / ((3 * a ^ 2 / 2) ^ 2) = 1 / 2 := by
  field_simp
  ring

theorem schwarzschild_quartic (M r : ℝ) (hr : r ≠ 0) :
    (48 : ℝ) * M ^ 2 / r ^ 6 = 48 * M ^ 2 * r⁻¹ ^ 6 := by
  rw [inv_pow]
  field_simp
