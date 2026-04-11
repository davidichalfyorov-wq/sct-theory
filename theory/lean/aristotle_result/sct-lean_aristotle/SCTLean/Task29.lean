import Mathlib.Tactic

theorem cross_term_vanishes {R : Type*} [Ring R]
    (P A : R) (hP : P * P = 1) (hcomm : P * A = A * P) :
    (A - P * A * P) = 0 := by
  have : P * A * P = A := by
    rw [hcomm, mul_assoc, hP, mul_one]
  rw [this, sub_self]

/-
PROVIDED SOLUTION
Same as off_diagonal_zero_of_comm_involution with K=M, C=P. Expand (1-C)*K*(1+C) with noncomm_ring, then use C*K*C = K (from hKC and hC_sq) and K*C = C*K.
-/
theorem heat_kernel_cross_term_zero {R : Type*} [Ring R]
    (K C : R) (hKC : K * C = C * K) (hC_sq : C * C = 1) :
    (1 - C) * K * (1 + C) = 0 := by
  simp +decide [ sub_mul, mul_add, hKC, hC_sq ];
  simp +decide [ mul_assoc, hKC, hC_sq ];
  simp +decide [ ← mul_assoc, hC_sq ]