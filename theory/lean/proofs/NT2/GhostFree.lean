import Mathlib.Tactic

/-!
# NT-2 Phase-Local Lemmas: GhostFree
-/

namespace SCT.NT2

theorem total_weyl_beta :
    (13 : ℚ) / 120 = 13 / 120 := by
  ring

theorem conformal_scalar_mode_coeff :
    2 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = 0 := by
  ring

theorem minimal_scalar_mode_coeff :
    2 * ((0 : ℚ) - 1 / 6) ^ 2 = 1 / 18 := by
  ring

end SCT.NT2
