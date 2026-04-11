import Mathlib.Tactic

theorem alpha_R_conformal_zero :
    (2 : ℚ) * ((1 : ℚ) / 6 - 1 / 6) ^ 2 = 0 := by
  norm_num

/-
PROVIDED SOLUTION
The function is Λ²/(6*(2*(ξ-1/6)²)) = Λ²/(12*(ξ-1/6)²). As ξ → (1/6)⁺, (ξ-1/6)² → 0⁺, so the denominator → 0⁺ and the fraction → +∞ since Λ² > 0. Use Filter.Tendsto for division where numerator is positive constant and denominator tends to 0⁺.
-/
theorem scalar_mass_diverges_at_conformal (Λ : ℝ) (hΛ : 0 < Λ) :
    Filter.Tendsto (fun ξ : ℝ => Λ ^ 2 / (6 * (2 * (ξ - 1/6) ^ 2)))
      (nhdsWithin (1/6) (Set.Ioi (1/6))) Filter.atTop := by
  refine' Filter.Tendsto.const_mul_atTop ( by positivity ) _;
  refine' Filter.Tendsto.inv_tendsto_nhdsGT_zero _;
  refine' tendsto_nhdsWithin_iff.mpr _;
  exact ⟨ tendsto_nhdsWithin_of_tendsto_nhds ( Continuous.tendsto' ( by continuity ) _ _ <| by norm_num ), Filter.eventually_of_mem self_mem_nhdsWithin fun x hx => by norm_num; nlinarith [ hx.out ] ⟩