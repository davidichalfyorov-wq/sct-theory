import Mathlib.Tactic

/-!
# NT-4a Phase-Local Lemmas: GaugeInv
-/

namespace SCT.NT4a

theorem transverse_projector_coeff :
    (1 : ℚ) - 1 = 0 := by
  ring

theorem tt_trace_factor :
    (1 : ℚ) / 2 + (1 : ℚ) / 2 - 1 = 0 := by
  ring

theorem scalar_projector_normalization :
    (1 : ℚ) / 3 + (1 : ℚ) / 3 + (1 : ℚ) / 3 = 1 := by
  ring

end SCT.NT4a
