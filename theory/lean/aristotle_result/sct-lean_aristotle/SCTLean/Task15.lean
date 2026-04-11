import Mathlib.Tactic

theorem de_sitter_core_from_cubic_mass (C : ℝ) (r : ℝ) (hr : r ≠ 0) :
    1 - 2 * (C * r ^ 3) / r = 1 - 2 * C * r ^ 2 := by
  field_simp

theorem linear_mass_constant_lapse (C : ℝ) (r : ℝ) (hr : r ≠ 0) :
    1 - 2 * (C * r) / r = 1 - 2 * C := by
  field_simp

/-
PROVIDED SOLUTION
Show that 1 - 2/r → -∞ as r → 0⁺. As r → 0⁺, 2/r → +∞, so 1 - 2/r → -∞. Use Filter.Tendsto.const_sub or similar, and the fact that division by r tends to +∞ as r → 0⁺. Key: show 2/r → +∞ using Filter.Tendsto.div_atTop or tendsto_const_div_atTop_nhds_0_nat, then 1 - (2/r) → atBot.
-/
theorem schwarzschild_lapse_diverges :
    Filter.Tendsto (fun r : ℝ => 1 - 2 / r) (nhdsWithin 0 (Set.Ioi 0))
      Filter.atBot := by
  norm_num [ Filter.tendsto_atBot, Filter.eventually_inf_principal, nhdsWithin ] at *;
  intro b; exact Metric.eventually_nhds_iff.mpr ⟨ ( |b| + 1 ) ⁻¹, by positivity, fun x hx hx' => by cases abs_cases b <;> nlinarith [ mul_inv_cancel₀ ( by positivity : ( |b| + 1 : ℝ ) ≠ 0 ), abs_lt.mp hx, div_mul_cancel₀ 2 ( by positivity : x ≠ 0 ) ] ⟩ ;