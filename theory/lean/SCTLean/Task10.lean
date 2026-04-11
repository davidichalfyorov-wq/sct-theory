import Mathlib.Tactic

theorem sm_alpha_C :
    (4 : ℚ) * (1 / 120) + (45 / 2) * (-1 / 20) + 12 * (1 / 10) = 13 / 120 := by
  norm_num

theorem sm_alpha_R_conformal :
    (2 : ℚ) * ((1 : ℚ) / 6 - 1 / 6) ^ 2 = 0 := by
  norm_num

theorem sm_c1c2_conformal :
    (-1 : ℚ) / 3 + 120 * ((1 : ℚ) / 6 - 1 / 6) ^ 2 / 13 = -1 / 3 := by
  norm_num

theorem sm_scalar_decoupling :
    (6 : ℚ) * ((1 : ℚ) / 6 - 1 / 6) ^ 2 = 0 := by
  norm_num
