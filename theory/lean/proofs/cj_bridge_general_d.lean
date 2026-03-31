import Mathlib.Tactic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Choose.Factorization

/-!
# CJ Bridge: General-d Beta-Function Identity

The key identity for general spacetime dimension d:

  B(d+1, d+1) = (d!)² / (2d+1)!

Equivalently:
  (d!)² × C(2d, d) × (2d+1) = (2d+1)!

This is the overlap integral of two d-th order retarded causal kernels:
  ∫₀¹ sᵈ(1-s)ᵈ ds = B(d+1, d+1) = Γ(d+1)²/Γ(2d+2) = (d!)²/(2d+1)!

Physical interpretation:
- k_d(s) = sᵈ/d! is the d-dim retarded propagation kernel
- The overlap ∫₀¹ k_d(s) × k_d(1-s) ds = 1/(2d+1)!
- This is the combinatorial origin of the 9! = (2×4+1)! in the CJ coefficient
-/

/-- Key helper: C(2d, d) × d! × d! = (2d)!
    Derived from Mathlib's `Nat.choose_mul_factorial_mul_factorial`. -/
theorem choose_times_factorial_sq (d : ℕ) :
    Nat.choose (2 * d) d * Nat.factorial d * Nat.factorial d =
    Nat.factorial (2 * d) := by
  have h := Nat.choose_mul_factorial_mul_factorial (Nat.le_add_left d d)
  -- h : C(d+d, d) * d! * (d+d-d)! = (d+d)!
  rw [Nat.add_sub_cancel] at h
  -- h : C(d+d, d) * d! * d! = (d+d)!
  rwa [show d + d = 2 * d from by omega] at h

/-- The beta-function identity at general d:
    (d!)² × C(2d, d) × (2d+1) = (2d+1)!

    Proof:
    1. C(2d, d) × (d!)² = (2d)!  [from Mathlib, via choose_times_factorial_sq]
    2. (2d+1)! = (2d+1) × (2d)!  [factorial recurrence]
    3. Combine and rearrange. -/
theorem beta_overlap_general (d : ℕ) :
    (Nat.factorial d) ^ 2 * Nat.choose (2 * d) d * (2 * d + 1) =
    Nat.factorial (2 * d + 1) := by
  have key := choose_times_factorial_sq d
  -- key : C(2d, d) * d! * d! = (2d)!
  rw [show 2 * d + 1 = (2 * d).succ from by omega, Nat.factorial_succ, sq]
  -- Goal: d! * d! * C(2d, d) * (2d).succ = (2d).succ * (2d)!
  -- Rearrange key: d! * d! * C(2d,d) = (2d)!
  have rearr : Nat.factorial d * Nat.factorial d * Nat.choose (2 * d) d =
      Nat.factorial (2 * d) := by linarith [key]
  -- Now substitute and use commutativity
  calc Nat.factorial d * Nat.factorial d * Nat.choose (2 * d) d * (2 * d).succ
      = Nat.factorial (2 * d) * (2 * d).succ := by rw [rearr]
    _ = (2 * d).succ * Nat.factorial (2 * d) := Nat.mul_comm _ _

/-- Corollary: the normalized kernel overlap = 1/(2d+1)!
    B(d+1,d+1) / (d!)² = (d!)²/((2d+1)!) / (d!)² = 1/(2d+1)!

    In ℚ: (d!)² / (2d+1)! / (d!)² = 1 / (2d+1)! -/
theorem normalized_overlap (d : ℕ) :
    (Nat.factorial d : ℚ) ^ 2 / (Nat.factorial (2 * d + 1) : ℚ) /
    (Nat.factorial d : ℚ) ^ 2 = 1 / (Nat.factorial (2 * d + 1) : ℚ) := by
  have hf : (Nat.factorial d : ℚ) ≠ 0 := by positivity
  have hf2 : (Nat.factorial d : ℚ) ^ 2 ≠ 0 := pow_ne_zero 2 hf
  field_simp

/-- Verification at d=1: (1!)² × C(2,1) × 3 = 1×2×3 = 6 = 3! ✓ -/
theorem beta_overlap_d1 :
    (Nat.factorial 1) ^ 2 * Nat.choose 2 1 * 3 = Nat.factorial 3 := by native_decide

/-- Verification at d=2: (2!)² × C(4,2) × 5 = 4×6×5 = 120 = 5! ✓ -/
theorem beta_overlap_d2 :
    (Nat.factorial 2) ^ 2 * Nat.choose 4 2 * 5 = Nat.factorial 5 := by native_decide

/-- Verification at d=3: (3!)² × C(6,3) × 7 = 36×20×7 = 5040 = 7! ✓ -/
theorem beta_overlap_d3 :
    (Nat.factorial 3) ^ 2 * Nat.choose 6 3 * 7 = Nat.factorial 7 := by native_decide

/-- Verification at d=4: (4!)² × C(8,4) × 9 = 576×70×9 = 362880 = 9! ✓
    THIS IS THE KEY CJ IDENTITY. -/
theorem beta_overlap_d4 :
    (Nat.factorial 4) ^ 2 * Nat.choose 8 4 * 9 = Nat.factorial 9 := by native_decide

/-- Verification at d=5: (5!)² × C(10,5) × 11 = 14400×252×11 = 39916800 = 11! ✓ -/
theorem beta_overlap_d5 :
    (Nat.factorial 5) ^ 2 * Nat.choose 10 5 * 11 = Nat.factorial 11 := by native_decide

/-- Verification at d=6: (6!)² × C(12,6) × 13 = 518400×924×13 = 6227020800 = 13! ✓ -/
theorem beta_overlap_d6 :
    (Nat.factorial 6) ^ 2 * Nat.choose 12 6 * 13 = Nat.factorial 13 := by native_decide

-- ============================================================================
-- Angular/Volume rational skeleton
-- ============================================================================

/-- The angular×volume factorization (rational part):
    1/45 = 8/360 = 8/(d!·(d²-1)) at d=4. -/
theorem angular_volume_rational : (1 : ℚ) / 45 = 8 / 360 := by norm_num

/-- The denominator 360 = d! × (d²-1) = 24 × 15 at d=4. -/
theorem denominator_360 : (24 : ℕ) * 15 = 360 := by norm_num

/-- So (2d)²/[(d-1)·(2d+1)!·d!·(d²-1)] = 64/[3·9!·360] = 1/6123600 -/
theorem full_decomposition_check :
    (64 : ℚ) / (3 * 362880 * 360) = 1 / 6123600 := by norm_num

/-- Same value from the CJ formula: 8/(3·9!·45) = 1/6123600 -/
theorem full_decomposition_alt :
    (8 : ℚ) / (3 * 362880 * 45) = 1 / 6123600 := by norm_num

/-- The two representations are equal -/
theorem representations_equal :
    (64 : ℚ) / (3 * 362880 * 360) = (8 : ℚ) / (3 * 362880 * 45) := by norm_num
