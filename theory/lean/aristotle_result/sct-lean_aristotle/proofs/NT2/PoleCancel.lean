import Mathlib.Tactic

/-!
# NT-2 Phase-Local Lemmas: PoleCancel
-/

namespace SCT.NT2

theorem hC_scalar_pole_cancels :
    (1 : ℚ) / 12 + (-(1 : ℚ) / 6) / 2 = 0 := by
  ring

theorem hC_dirac_pole_cancels :
    (1 : ℚ) / 3 + 2 * (-(1 : ℚ) / 6) = 0 := by
  ring

theorem hC_vector_pole_cancels :
    (1 : ℚ) / 6 + (-(1 : ℚ) / 6) = 0 := by
  ring

theorem hR_vector_pole_cancels :
    (5 : ℚ) / 72 + 5 * (-(1 : ℚ) / 6) / 12 = 0 := by
  ring

end SCT.NT2
