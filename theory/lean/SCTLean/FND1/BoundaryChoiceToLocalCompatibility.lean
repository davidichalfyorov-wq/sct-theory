import SCTLean.FND1.BoundaryGlobalCompositionZero

/-!
# FND-1 Boundary Choice To Local Compatibility

This module removes one more layer of duplicated language.

The current global compatibility datum on noncanonical edge and triangle choices
already implies the local-face compatibility structure used elsewhere in the
formal stack.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem choiceInduced_boundaryLocalOrientationCompatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    BoundaryLocalOrientationCompatibility (beta := beta) m U
      (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) := by
  refine ⟨?_⟩
  refine ⟨?_⟩
  intro i k
  have hzero :=
    choiceInduced_edgeTriangleBoundaryCompositionEntry_eq_zero
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (hglobal := hglobal) (i := i) (k := k)
  simpa [edgeTriangleCompositeTerm, edgeTriangleBoundaryCompositionEntry] using hzero

end SCT.FND1
