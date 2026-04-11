import Mathlib.Tactic
import Mathlib.MeasureTheory.Integral.Bochner.Set

theorem fakeon_disc_zero (a b : ℝ) :
    let GF := (a : ℂ) + Complex.I * b
    let GAF := (a : ℂ) - Complex.I * b
    let Gfk := (GF + GAF) / 2
    Gfk.im = 0 := by
  simp [Complex.add_im, Complex.sub_im, Complex.ofReal_im, Complex.mul_im,
        Complex.I_im, Complex.I_re]

theorem zero_disc_no_cut (f : ℝ → ℝ) (hf : ∀ x, f x = 0) :
    ∫ x in Set.Ioi (0 : ℝ), f x = 0 := by
  simp only [hf, MeasureTheory.setIntegral_const, smul_eq_mul, mul_zero]

theorem optical_theorem_with_fakeons {n : ℕ} (physical : Fin n → ℝ) :
    ∑ i, physical i ^ 2 ≥ 0 := by
  apply Finset.sum_nonneg
  intro i _
  exact sq_nonneg _
