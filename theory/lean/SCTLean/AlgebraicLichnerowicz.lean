import Mathlib

/-
PROVIDED SOLUTION
Expand LHS = (e1*a1 + e2*a2)*(e1*a1 + e2*a2) into 4 terms using mul_add and add_mul. Use commutativity hypotheses hc1-hc4 to move a's past e's: e1*a1*e1*a1 = e1*e1*a1*a1 = a1*a1 (using hc1 and he1). Similarly e2*a2*e2*a2 = a2*a2. For cross terms: e1*a1*e2*a2 = e1*e2*a1*a2 (using hc2) and e2*a2*e1*a1 = e2*e1*a2*a1 (using hc3). Then e1*e2*a1*a2 + e2*e1*a2*a1 = e1*e2*a1*a2 - e1*e2*a2*a1 (using he to get e2*e1 = -e1*e2) = e1*e2*(a1*a2 - a2*a1). The approach: use noncomm_ring-style manipulations or just rewrite with hypotheses step by step.
-/
theorem algebraic_lichnerowicz {R : Type*} [Ring R]
    (e1 e2 a1 a2 : R)
    (he : e1 * e2 + e2 * e1 = 0)
    (he1 : e1 * e1 = 1) (he2 : e2 * e2 = 1)
    (hc1 : a1 * e1 = e1 * a1) (hc2 : a1 * e2 = e2 * a1)
    (hc3 : a2 * e1 = e1 * a2) (hc4 : a2 * e2 = e2 * a2) :
    (e1 * a1 + e2 * a2) * (e1 * a1 + e2 * a2) =
    a1 * a1 + a2 * a2 + e1 * e2 * (a1 * a2 - a2 * a1) := by
  simp_all +decide [ mul_assoc, add_mul, mul_add ] ; abel_nf;
  simp_all +decide [ mul_add, mul_assoc ] ; abel_nf;
  simp_all +decide [ ← mul_assoc, eq_neg_of_add_eq_zero_right he ] ; abel_nf;