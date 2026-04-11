import Mathlib.Tactic

theorem spectral_dim_gr : (4 : ℝ) / 1 = 4 := by norm_num

theorem spectral_dim_as : (4 : ℝ) / 2 = 2 := by norm_num

theorem spectral_dim_sct : (4 : ℝ) / 1 = 4 := by norm_num

theorem dimensional_reduction_requires (d : ℝ) (hd : d = 4) :
    d / 2 = 2 := by rw [hd]; norm_num

theorem return_probability_exponent (d α : ℝ) (hα : α ≠ 0) :
    -2 * (-(d / (2 * α))) = d / α := by
  field_simp
