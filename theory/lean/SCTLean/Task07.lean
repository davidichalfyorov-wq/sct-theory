import Mathlib.Tactic

theorem comm_pow {R : Type*} [Ring R]
    (A B : R) (h : A * B = B * A) (n : ℕ) :
    A ^ n * B = B * A ^ n := by
  have hc : Commute A B := h
  exact (hc.pow_left n).eq

theorem comm_smul {R : Type*} [Ring R]
    (A B : R) (h : A * B = B * A) (c : R)
    (hc : c * B = B * c) :
    (c * A) * B = B * (c * A) := by
  rw [mul_assoc, h, ← mul_assoc, hc, mul_assoc]

theorem comm_sum {R : Type*} [Ring R] {n : ℕ}
    (B : R) (f : Fin n → R)
    (h : ∀ i, f i * B = B * f i) :
    (∑ i, f i) * B = B * (∑ i, f i) := by
  rw [Finset.sum_mul, Finset.mul_sum]
  congr 1
  ext i
  exact h i
