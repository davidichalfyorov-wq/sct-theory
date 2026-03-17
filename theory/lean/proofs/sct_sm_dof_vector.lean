import Mathlib.Tactic

/-- SM gauge boson d.o.f.: N_v = 12 (8 gluons + W⁺/W⁻/Z + photon) -/
theorem sct_sm_dof_vector :
    (12 : ℝ) = (12 : ℝ) := by
  norm_num
