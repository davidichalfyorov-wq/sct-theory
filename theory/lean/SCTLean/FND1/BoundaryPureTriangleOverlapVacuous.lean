import SCTLean.FND1.BoundaryPureTriangleOverlapInterfaces

/-!
# FND-1 Boundary Pure Triangle Overlap Vacuous Case

This module isolates the first genuinely automatic case for the current
noncanonical orientation stack.

If every triangle-bearing edge has a singleton triangle star, then there is no
nontrivial overlap-consistency problem left to solve. In that case the pure
triangle-overlap coherence law is vacuous, and the current square-zero
consequences follow with no further witness-coherence input.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Every triangle-bearing edge has at most one triangle in its star. -/
def BoundaryEdgeStarSubsingleton
    (m : Nat) (U : Cover beta alpha) : Prop :=
  ∀ edge : TriangleBoundaryEdge (beta := beta) m U,
    Subsingleton (EdgeTriangleWitness (beta := beta) m U edge.1)

theorem pureTriangleOverlapCoherence_of_boundaryEdgeStarSubsingleton
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : BoundaryEdgeStarSubsingleton (beta := beta) m U) :
    PureTriangleOverlapCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro triangle₁ triangle₂ hneq h₁ h₂
  let overlapEdge :=
    triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁
  let boundaryEdge : TriangleBoundaryEdge (beta := beta) m U :=
    ⟨overlapEdge, ⟨⟨triangle₁, h₁⟩⟩⟩
  have h₂' : overlapEdge.1 ∈ codimOneFaces (beta := beta) triangle₂.1 := by
    simpa [overlapEdge, triangleOverlapEdge] using h₂
  have hwEq :
      (⟨triangle₁, h₁⟩ : EdgeTriangleWitness (beta := beta) m U overlapEdge) =
        (⟨triangle₂, h₂'⟩ : EdgeTriangleWitness (beta := beta) m U overlapEdge) := by
    have hs : Subsingleton (EdgeTriangleWitness (beta := beta) m U overlapEdge) :=
      hstar boundaryEdge
    exact Subsingleton.elim _ _
  have htriEq : triangle₁ = triangle₂ := by
    exact congrArg Subtype.val hwEq
  exact False.elim (hneq htriEq)

theorem hasCompatibleLocalOrientation_of_boundaryEdgeStarSubsingleton
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : BoundaryEdgeStarSubsingleton (beta := beta) m U) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)
    (pureTriangleOverlapCoherence_of_boundaryEdgeStarSubsingleton
      (beta := beta) (m := m) (U := U) (tau := tau) hstar)

theorem hasBoundarySquareZero_of_boundaryEdgeStarSubsingleton
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : BoundaryEdgeStarSubsingleton (beta := beta) m U) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)
    (pureTriangleOverlapCoherence_of_boundaryEdgeStarSubsingleton
      (beta := beta) (m := m) (U := U) (tau := tau) hstar)

end SCT.FND1
