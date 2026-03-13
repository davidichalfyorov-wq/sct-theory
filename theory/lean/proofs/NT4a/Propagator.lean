import Mathlib.Tactic

/-!
# NT-4a Phase-Local Lemmas: Propagator
-/

namespace SCT.NT4a

theorem c2_local :
    2 * ((13 : ℚ) / 120) = 13 / 60 := by
  ring

theorem scalar_mode_minimal :
    6 * ((0 : ℚ) - 1 / 6) ^ 2 = 1 / 6 := by
  ring

theorem scalar_mode_conformal :
    6 * (((1 : ℚ) / 6) - 1 / 6) ^ 2 = 0 := by
  ring

theorem pi_tt_zero :
    (1 : ℚ) + 0 = 1 := by
  ring

end SCT.NT4a
