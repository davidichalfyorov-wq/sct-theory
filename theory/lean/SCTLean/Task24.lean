import Mathlib.Tactic
import Mathlib.Topology.Order.IntermediateValue

/-
PROVIDED SOLUTION
Use the intermediate value theorem. f is continuous, f(0) > 0, f(L) < 0 with 0 < L. The interval [0,L] is connected, so there exists z in (0,L) with f(z) = 0. Use intermediate_value_Icc or IsPreconnected.intermediate_value₂_uIcc. Alternatively, use Continuous.ivt or the fact that f(0) > 0 > f(L) gives an intermediate value.
-/
theorem has_zero_of_sign_change {f : ℝ → ℝ} (hf : Continuous f)
    (h0 : 0 < f 0) {L : ℝ} (hL : 0 < L) (hfL : f L < 0) :
    ∃ z ∈ Set.Ioo 0 L, f z = 0 := by
  apply_rules [ intermediate_value_Ioo' ] ; linarith;
  · exact hf.continuousOn;
  · aesop

/-
PROVIDED SOLUTION
Given z₁, z₂ in (a,b) with f(z₁) = 0 and f(z₂) = 0, if z₁ ≠ z₂ then WLOG z₁ < z₂ (or z₂ < z₁), and strict monotonicity gives f(z₂) > f(z₁) = 0 (or f(z₁) > f(z₂) = 0), contradicting f(z₂) = 0 (or f(z₁) = 0). So z₁ = z₂. Use ExistsUnique from the existing witness.
-/
theorem unique_zero_of_strict_mono {f : ℝ → ℝ}
    {a b : ℝ} (hab : a < b)
    (hf : ContinuousOn f (Set.Ioo a b))
    (hmono : StrictMonoOn f (Set.Ioo a b))
    (hz : ∃ z ∈ Set.Ioo a b, f z = 0) :
    ∃! z ∈ Set.Ioo a b, f z = 0 := by
  exact ⟨ hz.choose, ⟨ hz.choose_spec.1, hz.choose_spec.2 ⟩, fun x hx => hmono.eq_iff_eq hx.1 hz.choose_spec.1 |>.1 <| hx.2.trans hz.choose_spec.2.symm ⟩