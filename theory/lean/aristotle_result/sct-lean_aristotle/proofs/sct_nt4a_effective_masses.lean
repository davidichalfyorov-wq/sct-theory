import Mathlib.Tactic

/-!
# NT-4a Effective Masses from Propagator Zeros

The linearized propagators Π_TT(z) and Π_s(z,ξ) vanish at the
effective mass poles. In the local limit F̂ → 1:

  Π_TT(z) = 1 + (13/60)z = 0  →  z = -60/13  →  m₂² = (60/13)Λ²
  Π_s(z,ξ) = 1 + 6(ξ-1/6)²z = 0  →  z = -1/[6(ξ-1/6)²]

Special cases:
  ξ = 0 (minimal):    m₀² = 6Λ²       ≈ 2.449Λ
  ξ = 1/6 (conformal): m₀ → ∞ (scalar decouples)
  Spin-2:              m₂ = Λ√(60/13) ≈ 2.148Λ
-/

/-- Spin-2 effective mass: Π_TT vanishes at z = -60/13. -/
theorem nt4a_m2_squared :
    (1 : ℚ) + (13 : ℚ) / 60 * (-(60 : ℚ) / 13) = 0 := by
  norm_num

/-- Scalar effective mass at minimal coupling (ξ=0): Π_s vanishes at z = -6. -/
theorem nt4a_m0_squared_minimal :
    (1 : ℚ) + 6 * ((0 : ℚ) - 1 / 6) ^ 2 * (-(6 : ℚ)) = 0 := by
  norm_num

/-- The scalar mode coefficient 6(ξ-1/6)² equals c₂_scalar/60 = 1/6 at ξ=0. -/
theorem nt4a_scalar_coeff_minimal :
    6 * ((0 : ℚ) - 1 / 6) ^ 2 = 1 / 6 := by
  norm_num

/-- At conformal coupling: scalar propagator coefficient vanishes
    (scalar mode does not propagate, m₀ → ∞). -/
theorem nt4a_scalar_coeff_conformal :
    6 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = 0 := by
  norm_num

/-- Ratio of effective masses at minimal coupling:
    m₀²/m₂² = 6/(60/13) = 78/60 = 13/10. -/
theorem nt4a_mass_ratio_minimal :
    (6 : ℚ) / ((60 : ℚ) / 13) = 13 / 10 := by
  norm_num
