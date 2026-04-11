import Mathlib.Tactic

theorem fakeon_is_real (a b : ℝ) :
    let G_F : ℂ := ⟨a, b⟩
    let G_AF : ℂ := ⟨a, -b⟩
    (G_F + G_AF) / 2 = ↑a := by
  simp only
  apply Complex.ext <;> simp [Complex.add_re, Complex.add_im, Complex.div_ofReal] <;> ring

theorem real_amplitude_product (z w : ℂ) (hz : z.im = 0) (hw : w.im = 0) :
    (z * w).im = 0 := by
  simp [Complex.mul_im, hz, hw]

theorem real_amplitude_sum (z w : ℂ) (hz : z.im = 0) (hw : w.im = 0) :
    (z + w).im = 0 := by
  simp [Complex.add_im, hz, hw]

theorem fakeon_zero_discontinuity (a b : ℝ) :
    let G_F : ℂ := ⟨a, b⟩
    let G_AF : ℂ := ⟨a, -b⟩
    G_F - G_AF = ⟨0, 2 * b⟩ := by
  apply Complex.ext <;> simp [Complex.sub_re, Complex.sub_im] <;> ring
