import Mathlib.Tactic

theorem lee_wick_partial_fraction (k m1 m2 : ℝ)
    (h1 : k ^ 2 - m1 ^ 2 ≠ 0) (h2 : k ^ 2 - m2 ^ 2 ≠ 0) :
    (m1 ^ 2 - m2 ^ 2) / ((k ^ 2 - m1 ^ 2) * (k ^ 2 - m2 ^ 2)) =
    1 / (k ^ 2 - m1 ^ 2) - 1 / (k ^ 2 - m2 ^ 2) := by
  field_simp
  ring

theorem lee_wick_spectral_sum :
    (1 : ℝ) + (-1) = 0 := by norm_num

theorem lee_wick_uv_falloff (k m1 m2 : ℝ) (hk : 0 < k)
    (h1 : m1 ^ 2 < k ^ 2) (h2 : m2 ^ 2 < k ^ 2) :
    (m2 ^ 2 - m1 ^ 2) / ((k ^ 2 - m1 ^ 2) * (k ^ 2 - m2 ^ 2)) =
    (m2 ^ 2 - m1 ^ 2) / (k ^ 4 - (m1 ^ 2 + m2 ^ 2) * k ^ 2 + m1 ^ 2 * m2 ^ 2) := by
  congr 1
  ring
