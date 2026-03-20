import SCTLean.FND1.BoundaryTriangleEdgeCoherence

/-!
# FND-1 Boundary Triangle Edge To Local

This module removes one more layer of indirection.

Starting from coherent triangle-induced local edge orientations, it builds the
global edge-face part of a `BoundaryLocalOrientationDatum` directly, without
talking about primitive endpoint choices `chi`.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Global edge-local orientation rules induced by triangle witnesses. On edges
that appear in some triangle boundary, use any witness triangle. On isolated
edges, choose any local face and orient from it. -/
noncomputable def coherentGlobalEdgeLocalOrientationDatum
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    ∀ edge : ↑(edgeSimplices m U), LocalFaceOrientationDatum (beta := beta) edge.1 := by
  classical
  intro edge
  by_cases h : Nonempty (EdgeTriangleWitness (beta := beta) m U edge)
  · let witness : EdgeTriangleWitness (beta := beta) m U edge := Classical.choice h
    exact (tau witness.1).localEdgeOrientationOnBoundaryEdge
      (beta := beta) m U witness.1 edge witness.2
  · exact
      localOrientationFromChosenFace (beta := beta)
        (Classical.choice
          (localFaceSupport_nonempty_of_edge (beta := beta) (m := m) (U := U) edge))

theorem coherentGlobalEdgeLocalOrientationDatum_eq_triangle_edge_orientation
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau)
    (edge : ↑(edgeSimplices m U))
    (triangle : ↑(triangleSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1) :
    coherentGlobalEdgeLocalOrientationDatum (beta := beta) m U tau edge =
      (tau triangle).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle edge hedge := by
  classical
  let witness : EdgeTriangleWitness (beta := beta) m U edge := ⟨triangle, hedge⟩
  have hnonempty : Nonempty (EdgeTriangleWitness (beta := beta) m U edge) := ⟨witness⟩
  let selected : EdgeTriangleWitness (beta := beta) m U edge := Classical.choice hnonempty
  have horient :
      (tau selected.1).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U selected.1 edge selected.2 =
        (tau triangle).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle edge hedge :=
    hcoh.coherent edge selected.2 hedge
  simpa [coherentGlobalEdgeLocalOrientationDatum, hnonempty, selected] using horient

theorem coherentGlobalEdgeLocalOrientationDatum_eq_inducedEdgeLocalOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    coherentGlobalEdgeLocalOrientationDatum (beta := beta) m U tau =
      inducedEdgeLocalOrientations (beta := beta) m U
        (coherentEdgeEndpointChoiceDatum (beta := beta) m U tau) := by
  classical
  funext edge
  by_cases h : Nonempty (EdgeTriangleWitness (beta := beta) m U edge)
  · let witness : EdgeTriangleWitness (beta := beta) m U edge := Classical.choice h
    funext face
    simp [coherentGlobalEdgeLocalOrientationDatum, coherentEdgeEndpointChoiceDatum,
      inducedEdgeLocalOrientations, h, TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge]
  · let chosenFace := Classical.choice
      (localFaceSupport_nonempty_of_edge (beta := beta) (m := m) (U := U) edge)
    funext face
    simp [coherentGlobalEdgeLocalOrientationDatum, coherentEdgeEndpointChoiceDatum,
      inducedEdgeLocalOrientations, h]

/-- Direct boundary-local-orientation datum induced from triangle edge
orientations and triangle local orientations. -/
noncomputable def edgeOrientationChoiceInducedBoundaryLocalOrientationDatum
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    BoundaryLocalOrientationDatum (beta := beta) m U where
  edgeFaces := coherentGlobalEdgeLocalOrientationDatum (beta := beta) m U tau
  triangleFaces := fun triangle => (tau triangle).localTriangleOrientation

theorem edgeOrientationChoiceInducedBoundaryLocalOrientationDatum_eq_choiceInduced
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    edgeOrientationChoiceInducedBoundaryLocalOrientationDatum (beta := beta) m U tau =
      choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U
        (coherentEdgeEndpointChoiceDatum (beta := beta) m U tau) tau := by
  unfold edgeOrientationChoiceInducedBoundaryLocalOrientationDatum
  unfold choiceInducedBoundaryLocalOrientationDatum
  simp [coherentGlobalEdgeLocalOrientationDatum_eq_inducedEdgeLocalOrientations]

theorem boundaryLocalOrientationCompatible_of_triangleEdgeOrientationCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    BoundaryLocalOrientationCompatibility (beta := beta) m U
      (edgeOrientationChoiceInducedBoundaryLocalOrientationDatum
        (beta := beta) m U tau) := by
  have hchoiceCoh :=
    triangleChoiceCoherence_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh
  have hglobal :=
    globalChoiceCompatibility_of_triangleChoiceCoherent
      (beta := beta) (m := m) (U := U) (tau := tau) hchoiceCoh
  simpa [edgeOrientationChoiceInducedBoundaryLocalOrientationDatum_eq_choiceInduced
      (beta := beta) (m := m) (U := U) (tau := tau)] using
    choiceInduced_boundaryLocalOrientationCompatible
      (beta := beta) (m := m) (U := U)
      (chi := coherentEdgeEndpointChoiceDatum (beta := beta) m U tau)
      (tau := tau) hglobal

theorem hasCompatibleLocalOrientation_of_triangleEdgeOrientationCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact ⟨edgeOrientationChoiceInducedBoundaryLocalOrientationDatum
      (beta := beta) m U tau,
    boundaryLocalOrientationCompatible_of_triangleEdgeOrientationCoherent
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh⟩

theorem hasBoundarySquareZero_of_triangleEdgeOrientationCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_triangleEdgeOrientationCoherent
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh)

end SCT.FND1
