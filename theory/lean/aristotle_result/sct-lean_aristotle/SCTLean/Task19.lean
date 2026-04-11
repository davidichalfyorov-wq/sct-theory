import Mathlib.Tactic

theorem starobinsky_mass_ratio :
    ((13 : ℚ) / 1000000) ^ 2 > (1 : ℚ) / 100000000000 := by norm_num

theorem inflation_mass_gap :
    ((13 : ℚ) / 1000000) / ((36 : ℚ) / 10^32) > 10^25 := by norm_num
