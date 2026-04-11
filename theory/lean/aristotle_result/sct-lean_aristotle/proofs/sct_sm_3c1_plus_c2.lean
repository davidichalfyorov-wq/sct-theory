import Mathlib.Tactic

/-!
# Scalar Mode Combination 3c₁ + c₂ = 6(ξ - 1/6)²

The combination 3c₁ + c₂ controls the massive scalar mode in the
linearized field equations:

  3c₁ + c₂ = 3(α_R - (2/3)α_C) + 2α_C
            = 3α_R - 2α_C + 2α_C
            = 3α_R
            = 6(ξ - 1/6)²

Key consequence: at conformal coupling ξ = 1/6, the scalar mode
DECOUPLES (3c₁ + c₂ = 0, so m₀ → ∞). Only the spin-2 massive
graviton survives.
-/

/-- **Main theorem:** 3c₁ + c₂ = 6(ξ-1/6)² (α_C cancels exactly). -/
theorem sct_3c1_plus_c2 (ξ : ℚ) :
    3 * (2 * (ξ - 1 / 6) ^ 2 - (2 : ℚ) / 3 * ((13 : ℚ) / 120)) +
    2 * ((13 : ℚ) / 120) = 6 * (ξ - 1 / 6) ^ 2 := by
  ring

/-- At conformal coupling: 3c₁ + c₂ = 0 (scalar mode decouples). -/
theorem sct_3c1_plus_c2_conformal :
    3 * (2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 - (2 : ℚ) / 3 * ((13 : ℚ) / 120)) +
    2 * ((13 : ℚ) / 120) = 0 := by
  norm_num

/-- At minimal coupling: 3c₁ + c₂ = 6(1/6)² = 1/6. -/
theorem sct_3c1_plus_c2_minimal :
    3 * (2 * ((0 : ℚ) - 1 / 6) ^ 2 - (2 : ℚ) / 3 * ((13 : ℚ) / 120)) +
    2 * ((13 : ℚ) / 120) = 1 / 6 := by
  norm_num
