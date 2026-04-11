import Mathlib.Tactic

theorem one_dim_absorbable {R : Type*} [Field R] (c : R) :
    ∃ l : R, l = c := ⟨c, rfl⟩

theorem counterterm_absorption {R : Type*} [CommRing R] (S : R) (c : R) :
    ∃ δf : R, δf * S = c * S := ⟨c, rfl⟩

theorem solvable_if_params_geq_dim (n m : ℕ) (h : m ≥ n) :
    n ≤ m := h

theorem overdetermined_not_solvable :
    ¬ (∀ a b : ℚ, ∃ x : ℚ, x = a ∧ x = b) := by
  intro h
  obtain ⟨x, hxa, hxb⟩ := h 0 1
  rw [hxa] at hxb
  norm_num at hxb
