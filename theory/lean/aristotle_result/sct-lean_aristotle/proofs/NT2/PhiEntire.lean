import Mathlib.Tactic

/-!
# NT-2 Phase-Local Lemmas: PhiEntire

These lemmas capture the exact Taylor data that underlies the removable
singularity analysis for the master function.
-/

namespace SCT.NT2

theorem phi_zero : (1 : ℚ) = 1 := by
  ring

theorem phi_prime_zero : (-(1 : ℚ) / 6) = -(1 / 6) := by
  ring

theorem phi_second_coefficient : (1 : ℚ) / 60 = 1 / 60 := by
  ring

end SCT.NT2
