import Mathlib.Tactic

/-
PROVIDED SOLUTION
Expand to M + M*P - P*M - P*M*P using noncomm_ring. Then P*M*P = (P*M)*P = (M*P)*P = M*(P*P) = M*1 = M by hcomm and hP_sq. And M*P = P*M by hcomm. So we get M + P*M - P*M - M = 0.
-/
theorem off_diagonal_zero_of_comm_involution {R : Type*} [Ring R]
    (M P : R)
    (hP_sq : P * P = 1)
    (hcomm : M * P = P * M) :
    (1 - P) * M * (1 + P) = 0 := by
  simp +decide [ sub_mul, mul_add, hP_sq, hcomm ];
  simp +decide [ mul_assoc, hP_sq, hcomm ];
  simp +decide [ ← mul_assoc, hP_sq ]

/-
PROVIDED SOLUTION
Same approach as the first theorem but for (1+P)*M*(1-P). Expand with noncomm_ring, rewrite P*M*P = M, and M*P = P*M, then simplify.
-/
theorem off_diagonal_zero_of_comm_involution' {R : Type*} [Ring R]
    (M P : R)
    (hP_sq : P * P = 1)
    (hcomm : M * P = P * M) :
    (1 + P) * M * (1 - P) = 0 := by
  simp +decide [ mul_sub, sub_mul, mul_assoc, add_mul, mul_add, hP_sq, hcomm ];
  grind