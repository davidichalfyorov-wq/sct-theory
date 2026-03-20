import SCTLean.FND1.BoundaryWitnessCompositionBC

/-!
# FND-1 Boundary Witness Triangle Composition

This module packages the three witness-row entry vanishing theorems into one
triangle-level statement.

It does not yet claim a whole-nerve matrix theorem. It only records the exact
local result already justified for the three singleton rows attached to a fixed
ordered witness triangle.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem witnessTriangle_local_composition_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowA (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 ∧
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowB (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 ∧
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowC (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  constructor
  · exact witnessRowA_composition_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (triangle := triangle)
  constructor
  · exact witnessRowB_composition_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (triangle := triangle)
  · exact witnessRowC_composition_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (triangle := triangle)

theorem witnessTriangle_local_composition_zero_entries
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowA (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 ∧
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowB (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 ∧
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowC (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  constructor
  · exact witnessRowA_compositionEntry_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (triangle := triangle)
  constructor
  · exact witnessRowB_compositionEntry_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (triangle := triangle)
  · exact witnessRowC_compositionEntry_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (triangle := triangle)

end SCT.FND1
