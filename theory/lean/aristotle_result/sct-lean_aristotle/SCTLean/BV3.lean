import Mathlib.Tactic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-- The eigenvalues of the Lyapunov operator are hᵢ + hⱼ, all positive. -/
theorem lyapunov_eigenvalues_positive {n : ℕ}
    (h : Fin n → ℝ) (hpos : ∀ i, 0 < h i)
    (i j : Fin n) :
    0 < h i + h j := by
  linarith [hpos i, hpos j]

/-- BV-3 for finite spectral triples: if H = diag(h₁,...,hₙ) with hᵢ > 0,
    then XH + HX = 0 implies X = 0.
    Proof strategy: (XH+HX)ᵢⱼ = (hᵢ+hⱼ)Xᵢⱼ = 0, and hᵢ+hⱼ > 0, so Xᵢⱼ = 0. -/
theorem bv3_finite {n : ℕ}
    (h : Fin n → ℝ) (hpos : ∀ i, 0 < h i) :
    ∀ X : Matrix (Fin n) (Fin n) ℝ,
      X * Matrix.diagonal h + Matrix.diagonal h * X = 0 → X = 0 := by
  intro X hX; ext i j
  replace hX := congr_fun (congr_fun hX i) j
  simp_all +decide [Matrix.mul_apply, add_eq_zero_iff_eq_neg]
  simp_all +decide [Matrix.diagonal]
  nlinarith [hpos i, hpos j]

/-- Injectivity form: the Lyapunov map is injective. -/
theorem lyapunov_injective {n : ℕ}
    (h : Fin n → ℝ) (hpos : ∀ i, 0 < h i) :
    Function.Injective (fun X : Matrix (Fin n) (Fin n) ℝ =>
      X * Matrix.diagonal h + Matrix.diagonal h * X) := by
  intros A B hAB
  have h_zero : (A - B) * (Matrix.diagonal h) + (Matrix.diagonal h) * (A - B) = 0 := by
    convert sub_eq_zero.mpr hAB using 1; simp +decide [sub_mul, mul_sub]; abel_nf
  have h_eq_zero : A - B = 0 := bv3_finite h hpos (A - B) h_zero
  simp_all [sub_eq_zero]

/-- Surjectivity: for any B, the equation XH + HX = B has a solution.
    Solution: Xᵢⱼ = Bᵢⱼ / (hᵢ + hⱼ). -/
theorem lyapunov_surjective {n : ℕ}
    (h : Fin n → ℝ) (hpos : ∀ i, 0 < h i)
    (B : Matrix (Fin n) (Fin n) ℝ) :
    ∃ X : Matrix (Fin n) (Fin n) ℝ,
      X * Matrix.diagonal h + Matrix.diagonal h * X = B := by
  use Matrix.of (fun i j => B i j / (h i + h j))
  ext i j
  by_cases hij : i = j <;> simp +decide [div_eq_mul_inv, *]; ring
  · rw [mul_assoc, mul_inv_cancel₀ (ne_of_gt (hpos j)), mul_one]
  · linarith [inv_mul_cancel_left₀ (ne_of_gt (add_pos (hpos i) (hpos j))) (B i j)]

/-- Combined: the Lyapunov map is bijective. The Jacobian is nonzero. BV-3 holds. -/
theorem lyapunov_bijective {n : ℕ}
    (h : Fin n → ℝ) (hpos : ∀ i, 0 < h i) :
    Function.Bijective (fun X : Matrix (Fin n) (Fin n) ℝ =>
      X * Matrix.diagonal h + Matrix.diagonal h * X) :=
  ⟨lyapunov_injective h hpos, lyapunov_surjective h hpos⟩
