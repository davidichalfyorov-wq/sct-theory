import Mathlib.Tactic

/-
PROVIDED SOLUTION
Reassociate (P*A)*C = P*(A*C) = P*(-(C*A)) = -(P*C*A) = -(C*P*A) = -(C*(P*A)). Use mul_assoc, hA, neg_mul, mul_neg, hP.
-/
theorem comm_mul_anticomm_gives_anticomm {R : Type*} [Ring R]
    (P A C : R)
    (hP : P * C = C * P)
    (hA : A * C = -(C * A)) :
    (P * A) * C = -((C) * (P * A)) := by
  simp +decide [ mul_assoc, hA, hP ];
  rw [ ← mul_assoc, hP, mul_assoc ]

/-
PROVIDED SOLUTION
Reassociate (P*A)*C = P*(A*C) = P*(-(C*A)) = -(P*(C*A)) = -(P*C*A). Then use hP: P*C = -(C*P), so -(-(C*P)*A) = C*P*A = C*(P*A). Use mul_assoc, hA, mul_neg, neg_mul, hP, neg_neg.
-/
theorem anticomm_mul_anticomm_gives_comm {R : Type*} [Ring R]
    (P A C : R)
    (hP : P * C = -(C * P))
    (hA : A * C = -(C * A)) :
    (P * A) * C = C * (P * A) := by
  have h_comm : (P * A) * C = -((P * C) * A) := by
    simp +decide [ mul_assoc, hA ];
  simp_all +decide [ ← mul_assoc ]