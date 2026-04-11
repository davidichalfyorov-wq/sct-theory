import Mathlib.Tactic

theorem scalaron_mass_formula (xi : ℝ) (hxi : xi ≠ 1/6) (Λ : ℝ) :
    Λ ^ 2 / (6 * (2 * (xi - 1/6) ^ 2)) = Λ ^ 2 / (12 * (xi - 1/6) ^ 2) := by
  ring

theorem scalaron_mass_minimal :
    (12 : ℝ) * ((0 : ℝ) - 1/6) ^ 2 = 12 * (1/36) := by
  norm_num

theorem scalaron_mass_minimal_value :
    (12 : ℝ) * (1 / 36) = 1 / 3 := by
  norm_num
