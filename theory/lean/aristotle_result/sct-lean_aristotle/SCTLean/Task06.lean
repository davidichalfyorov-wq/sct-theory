import Mathlib.Tactic

theorem involution_projections_orthogonal {R : Type*} [Ring R]
    (P : R) (hP : P * P = 1) :
    (1 + P) * (1 - P) = 0 := by
  have key : (1 + P) * (1 - P) = 1 - P * P := by noncomm_ring
  rw [key, hP, sub_self]

theorem involution_projections_orthogonal' {R : Type*} [Ring R]
    (P : R) (hP : P * P = 1) :
    (1 - P) * (1 + P) = 0 := by
  have key : (1 - P) * (1 + P) = 1 - P * P := by noncomm_ring
  rw [key, hP, sub_self]

theorem involution_projection_idempotent {R : Type*} [Ring R]
    (P : R) (hP : P * P = 1) :
    (1 + P) * (1 + P) = 2 * 1 + 2 * P := by
  have key : (1 + P) * (1 + P) = 1 + P + P + P * P := by noncomm_ring
  rw [key, hP]
  noncomm_ring
