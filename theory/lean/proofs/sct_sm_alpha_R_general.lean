import Mathlib.Tactic

/-!
# SM Total R² Coefficient α_R(ξ) = 2(ξ - 1/6)²

The R²-coefficient in the spectral action depends on the Higgs
non-minimal coupling ξ:

  α_R(ξ) = N_s · β_R^(0)(ξ) + N_D · 0 + N_v · 0
         = 4 · (1/2)(ξ - 1/6)² + 0 + 0
         = 2(ξ - 1/6)²

Only the scalar sector contributes (β_R = 0 for both Dirac and vector).
-/

/-- α_R(ξ) = 2(ξ-1/6)²: only scalar contributes to the R² coefficient. -/
theorem sct_sm_alpha_R_general (ξ : ℚ) :
    (4 : ℚ) * ((1 : ℚ) / 2 * (ξ - 1 / 6) ^ 2) = 2 * (ξ - 1 / 6) ^ 2 := by
  ring

/-- At conformal coupling ξ = 1/6: α_R vanishes identically. -/
theorem sct_sm_alpha_R_conformal :
    2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = 0 := by
  ring

/-- At minimal coupling ξ = 0: α_R = 2(1/6)² = 1/18. -/
theorem sct_sm_alpha_R_minimal :
    2 * ((0 : ℚ) - 1 / 6) ^ 2 = 1 / 18 := by
  ring
