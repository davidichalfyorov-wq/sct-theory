import Mathlib.Tactic

theorem spectral_action_nonneg {n : ℕ}
    (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x)
    (eigenvalues : Fin n → ℝ) :
    0 ≤ ∑ i, f (eigenvalues i) := by
  apply Finset.sum_nonneg
  intro i _
  exact hf _

theorem exp_pos_of_real (x : ℝ) : 0 < Real.exp (-x) := by
  exact Real.exp_pos _

theorem spectral_action_exp_pos {n : ℕ} (hn : 0 < n)
    (eigenvalues : Fin n → ℝ) :
    0 < ∑ i, Real.exp (-(eigenvalues i)) := by
  apply Finset.sum_pos
  · intro i _
    exact Real.exp_pos _
  · exact Finset.univ_nonempty_iff.mpr ⟨⟨0, hn⟩⟩
