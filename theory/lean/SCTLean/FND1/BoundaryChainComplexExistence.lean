import SCTLean.FND1.BoundaryCompatibleExistence

/-!
# FND-1 Boundary Chain Complex Existence

This module records the minimal chain-level consequence currently justified by
the formal stack: the existence of oriented boundary data whose composition
vanishes.

This statement is intentionally weaker than any canonical-orientation theorem.
It does not say where the orientation data come from intrinsically; it only
isolates the exact existential boundary-square-zero content already proved.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- There exist oriented edge-to-vertex and triangle-to-edge boundary data on
the finite nerve whose composition vanishes. -/
def HasBoundarySquareZero
    (m : Nat) (U : Cover beta alpha) : Prop :=
  ∃ (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
      (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U),
    edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0

theorem hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (m : Nat) (U : Cover beta alpha)
    (hexists : HasCompatibleLocalOrientation (beta := beta) m U) :
    HasBoundarySquareZero (beta := beta) m U := by
  rcases hexists with ⟨theta, htheta⟩
  refine ⟨inducedEdgeVertexOrientationDatum (beta := beta) m U theta,
    inducedTriangleEdgeOrientationDatum (beta := beta) m U theta, ?_⟩
  exact edgeTriangleBoundaryComposition_matrix_eq_zero_of_local_compatible
    (beta := beta) (m := m) (U := U) (theta := theta) htheta

theorem hasBoundarySquareZero_of_globalChoiceCompatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_globalChoiceCompatible
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) hglobal)

end SCT.FND1
