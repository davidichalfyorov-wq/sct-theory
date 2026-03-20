import SCTLean.FND1.BoundaryChoiceToLocalCompatibility

/-!
# FND-1 Boundary Compatible Existence

This module packages the current noncanonical choice-based boundary theory into
the smallest existential form that is already justified.

Instead of exposing the explicit edge-choice datum `chi` and triangle-choice
datum `tau` downstream, we isolate the fact that what ultimately matters is the
existence of a compatible local orientation datum.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- There exists a local orientation datum on the finite nerve whose induced
boundary operators satisfy the current compatibility law. -/
def HasCompatibleLocalOrientation
    (m : Nat) (U : Cover beta alpha) : Prop :=
  ∃ theta : BoundaryLocalOrientationDatum (beta := beta) m U,
    BoundaryLocalOrientationCompatibility (beta := beta) m U theta

theorem hasCompatibleLocalOrientation_of_globalChoiceCompatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  refine ⟨choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau, ?_⟩
  exact choiceInduced_boundaryLocalOrientationCompatible
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) hglobal

theorem exists_boundarySquareZero_of_hasCompatibleLocalOrientation
    (m : Nat) (U : Cover beta alpha)
    (hexists : HasCompatibleLocalOrientation (beta := beta) m U) :
    ∃ theta : BoundaryLocalOrientationDatum (beta := beta) m U,
      BoundaryLocalOrientationCompatibility (beta := beta) m U theta ∧
      edgeTriangleBoundaryComposition (beta := beta) m U
        (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
        (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) = 0 := by
  rcases hexists with ⟨theta, htheta⟩
  refine ⟨theta, htheta, ?_⟩
  exact edgeTriangleBoundaryComposition_matrix_eq_zero_of_local_compatible
    (beta := beta) (m := m) (U := U) (theta := theta) htheta

theorem exists_boundarySquareZero_of_globalChoiceCompatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    ∃ theta : BoundaryLocalOrientationDatum (beta := beta) m U,
      BoundaryLocalOrientationCompatibility (beta := beta) m U theta ∧
      edgeTriangleBoundaryComposition (beta := beta) m U
        (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
        (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) = 0 := by
  exact exists_boundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_globalChoiceCompatible
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) hglobal)

end SCT.FND1
