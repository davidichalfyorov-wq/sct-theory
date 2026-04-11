import SCTLean.FND1.BoundaryPureTriangleOverlapCoherence

/-!
# FND-1 Boundary Pure Triangle Overlap Interfaces

This module collapses the remaining equivalent interface layers onto the pure
overlap formulation.

Once coherence is stated directly on canonical triangle overlaps, the older
edge-star, boundary-edge gluing, and extension-problem layers are no longer
independent assumptions. They are equivalent reformulations of the same
remaining condition.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem localTriangleEdgeStarCoherence_iff_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    LocalTriangleEdgeStarCoherence (beta := beta) m U tau ↔
      PureTriangleOverlapCoherence (beta := beta) m U tau := by
  rw [← triangleEdgeOrientationCoherence_iff_localTriangleEdgeStarCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)]
  exact triangleEdgeOrientationCoherence_iff_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)

theorem hasGluedTriangleBoundaryEdgeOrientations_iff_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau ↔
      PureTriangleOverlapCoherence (beta := beta) m U tau := by
  rw [← triangleEdgeOrientationCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)]
  exact triangleEdgeOrientationCoherence_iff_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)

theorem hasTriangleWitnessLocalExtension_iff_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    HasTriangleWitnessLocalExtension (beta := beta) m U tau ↔
      PureTriangleOverlapCoherence (beta := beta) m U tau := by
  rw [hasTriangleWitnessLocalExtension_iff_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)]
  exact triangleEdgeOrientationCoherence_iff_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)

end SCT.FND1
