import Mathlib

/-
PROVIDED SOLUTION
Use Summable.of_norm or AbsoluteValue.summable. The absolute summability implies summability for real-valued sequences.
-/
theorem trace_class_of_summable {a : ℕ → ℝ}
    (ha : Summable (fun n => |a n|)) :
    Summable a := by
  exact ha.of_abs

/-
PROVIDED SOLUTION
We need to show Summable (fun n => |a (n + 1)|). By comparison, |a (n+1)| ≤ C / (n+1)^α for all n (since n+1 > 0). The series ∑ C / (n+1)^α is summable since α > 1 (this is the p-series). Use Summable.of_nonneg_of_le with the bound, and Real.summable_nat_rpow_inv or summable_one_div_nat_pow for the comparison series.
-/
theorem summable_of_power_decay (C : ℝ) (α : ℝ) (hα : 1 < α) (hC : 0 < C)
    (a : ℕ → ℝ) (ha : ∀ n, 0 < n → |a n| ≤ C / ↑n ^ α) :
    Summable (fun n => |a (n + 1)|) := by
  exact Summable.of_nonneg_of_le ( fun n => abs_nonneg _ ) ( fun n => ha _ ( Nat.succ_pos _ ) ) ( Summable.mul_left _ <| by simpa using summable_nat_add_iff 1 |>.2 <| Real.summable_one_div_nat_rpow.2 hα )

/-
PROVIDED SOLUTION
This is trivially ⟨Matrix.det (1 + K), rfl⟩.
-/
theorem finite_fredholm_det {n : ℕ}
    (K : Matrix (Fin n) (Fin n) ℝ) :
    ∃ d : ℝ, d = Matrix.det (1 + K) := by
  use Matrix.det (1 + K)

/-
PROVIDED SOLUTION
Use Matrix.det_fin_two to expand the 2x2 determinant, then ring.
-/
theorem det_perturbation_2x2 (a b c d : ℝ) (ε : ℝ) :
    Matrix.det !![1 + ε * a, ε * b; ε * c, 1 + ε * d] =
    1 + ε * (a + d) + ε ^ 2 * (a * d - b * c) := by
  simpa [ Matrix.det_fin_two ] using by ring;