import Mathlib.Tactic

theorem regularity_from_propagator_decay (n : ℕ)
    (hn : n ≥ 3) : (2 * n - 3 : ℤ) ≥ 3 := by
  omega

theorem sct_regularity_index :
    (1 : ℤ) - 3 = -2 := by
  norm_num

theorem stelle_regularity_index :
    (2 : ℤ) - 3 = -1 := by
  norm_num

theorem six_deriv_regularity_index :
    (3 : ℤ) - 3 = 0 := by
  norm_num
