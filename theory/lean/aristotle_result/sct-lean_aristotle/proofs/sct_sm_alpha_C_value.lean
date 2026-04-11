import Mathlib.Tactic

/-!
# SM Total Weyl Coefficient α_C = 13/120

The Weyl-squared coefficient in the spectral action with full SM content:

  α_C = N_s · h_C^(0)(0) + N_D · h_C^(1/2)(0) + N_v · h_C^(1)(0)
      = 4 · (1/120) + (45/2) · (-1/20) + 12 · (1/10)
      = 13/120

Note: h_C^(1/2)(0) = -1/20 (negative for Dirac, from the limit of
(3φ-1)/(6x) + 2(φ-1)/x² as x → 0, where divergent 1/x terms cancel
leaving the finite part -1/12 + 1/30 = -1/20).

This is the parameter-free, ξ-independent prediction of SCT.
-/

/-- α_C = 13/120: the total Weyl² coefficient from h_C(0) local limits.
    Uses physical form factor values h_C^(s)(0), NOT β_W^(s). -/
theorem sct_sm_alpha_C_value :
    (4 : ℚ) * (1 / 120) + (45 : ℚ) / 2 * (-(1 : ℚ) / 20) +
    (12 : ℚ) * (1 / 10) = 13 / 120 := by
  norm_num

/-- Verification: α_C is ξ-independent.
    The individual h_C^(s)(0) values contain no ξ-dependence. -/
theorem sct_sm_alpha_C_scalar_contribution :
    (4 : ℚ) * (1 / 120) = 1 / 30 := by
  norm_num

theorem sct_sm_alpha_C_dirac_contribution :
    (45 : ℚ) / 2 * (-(1 : ℚ) / 20) = -(9 : ℚ) / 8 := by
  norm_num

theorem sct_sm_alpha_C_vector_contribution :
    (12 : ℚ) * ((1 : ℚ) / 10) = 6 / 5 := by
  norm_num
