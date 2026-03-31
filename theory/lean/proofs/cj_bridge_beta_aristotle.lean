import Mathlib

/-!
# CJ Bridge: Beta Overlap — Aristotle Project 2 Proofs

Aristotle-generated proofs for the beta overlap identity in RATIONAL form:
  (d!)² / (2d+1)! = 1 / ((2d+1) × C(2d, d))

This is the DUAL form of the identity proved in cj_bridge_general_d.lean:
  (d!)² × C(2d,d) × (2d+1) = (2d+1)!

Both forms are equivalent; having both gives cross-validation.

Project ID: 3a6de30a-276d-45c2-87aa-c93a1d660d0a
Status: COMPLETE (8 sorry-free theorems)
-/

-- Concrete factorial identities (Aristotle)
theorem art2_factorial_3 : Nat.factorial 3 = 6 := by native_decide

theorem art2_factorial_4 : Nat.factorial 4 = 24 := by native_decide

theorem art2_factorial_5 : Nat.factorial 5 = 120 := by native_decide

-- Concrete choose identities (Aristotle)
theorem art2_choose_4_2 : Nat.choose 4 2 = 6 := by native_decide

theorem art2_choose_6_3 : Nat.choose 6 3 = 20 := by native_decide

-- Rational arithmetic identities (Aristotle)
theorem art2_rat_factorial_ratio :
    (Nat.factorial 3 : ℚ) / Nat.factorial 5 = 1 / 20 := by native_decide

theorem art2_rat_choose_identity :
    (1 : ℚ) / ((2 * 3 + 1) * Nat.choose (2 * 3) 3) =
    (Nat.factorial 3 * Nat.factorial 3 : ℚ) / Nat.factorial (2 * 3 + 1) := by native_decide

-- General beta overlap (RATIONAL FORM, Aristotle)
-- (d!)² / (2d+1)! = 1 / ((2d+1) × C(2d, d))
-- Uses: Nat.cast_choose, field_simp, norm_num, ring, grind +locals
theorem art2_beta_overlap_general (d : ℕ) :
    (Nat.factorial d : ℚ) * (Nat.factorial d : ℚ) / (Nat.factorial (2 * d + 1) : ℚ) =
    1 / ((2 * d + 1 : ℚ) * (Nat.choose (2 * d) d : ℚ)) := by
      rw [ Nat.cast_choose ];
      · field_simp;
        norm_num [ two_mul, Nat.factorial ] ; ring;
      · grind +locals
