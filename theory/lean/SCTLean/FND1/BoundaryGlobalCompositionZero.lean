import SCTLean.FND1.BoundaryWitnessAnyRow

/-!
# FND-1 Boundary Global Composition Zero

This module closes the current noncanonical boundary-composition program.

Under the existing global compatibility datum between edge choices and triangle
ordered witnesses, the composed boundary `∂₁ ∘ ∂₂` vanishes entrywise and hence
as a whole matrix for the current choice-induced boundary data.

This is still a theorem about the present noncanonical construction. It is not
yet a canonical intrinsic orientation theorem.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem choiceInduced_edgeTriangleBoundaryCompositionEntry_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (i : EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau))
    (k : TriangleEdgeColIndex (beta := beta) m U
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      i k = 0 := by
  exact anyRow_witnessTriangleCompositionEntry_eq_zero
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (hglobal := hglobal) (triangle := k) (i := i)

theorem choiceInduced_edgeTriangleBoundaryComposition_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau) = 0 := by
  ext i k
  exact choiceInduced_edgeTriangleBoundaryCompositionEntry_eq_zero
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (hglobal := hglobal) (i := i) (k := k)

end SCT.FND1
