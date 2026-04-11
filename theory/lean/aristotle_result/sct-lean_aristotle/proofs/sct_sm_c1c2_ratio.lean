import Mathlib.Tactic

/-!
# Wilson Coefficient Ratio c₁/c₂

From the Gauss-Bonnet decomposition (prediction_01):
  c₁ = α_R(ξ) - (2/3)α_C    (coefficient of R²)
  c₂ = 2α_C                  (coefficient of R_μν R^μν)

With α_C = 13/120 and α_R(ξ) = 2(ξ-1/6)²:

  c₁/c₂ = (α_R - (2/3)α_C) / (2α_C)
         = α_R/(2α_C) - 1/3
         = -1/3 + 120(ξ-1/6)²/13
-/

/-- **Main theorem:** c₁/c₂ = -1/3 + 120(ξ-1/6)²/13.
    Proved via field_simp (clearing denominators 13/60 and 13) then ring. -/
theorem sct_c1c2_ratio (ξ : ℚ) :
    (2 * (ξ - 1 / 6) ^ 2 - (2 : ℚ) / 3 * ((13 : ℚ) / 120)) /
    (2 * ((13 : ℚ) / 120)) =
    -(1 : ℚ) / 3 + 120 * (ξ - 1 / 6) ^ 2 / 13 := by
  field_simp
  ring

/-- At conformal coupling ξ = 1/6: c₁/c₂ = -1/3 (parameter-free). -/
theorem sct_c1c2_ratio_conformal :
    (2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 - (2 : ℚ) / 3 * ((13 : ℚ) / 120)) /
    (2 * ((13 : ℚ) / 120)) = -(1 : ℚ) / 3 := by
  norm_num

/-- At minimal coupling ξ = 0: c₁/c₂ = -1/3 + 120/(13·36) = -1/3 + 10/39. -/
theorem sct_c1c2_ratio_minimal :
    (2 * ((0 : ℚ) - 1 / 6) ^ 2 - (2 : ℚ) / 3 * ((13 : ℚ) / 120)) /
    (2 * ((13 : ℚ) / 120)) = -(1 : ℚ) / 3 + 10 / 39 := by
  norm_num
