import SCTLean.FND1.BoundaryTriangleOverlapCoherence

/-!
# FND-1 Boundary Triangle Boundary-Edge Gluing

This module weakens the gluing interface by restricting it to the only edges
that matter for the current `∂₂`-layer: edges that occur in the boundary of at
least one triangle simplex.

The previous gluing layer asked for a global edge-local orientation datum on
*all* edge simplices, even isolated edges that never appear in any triangle
boundary. Here we remove that irrelevance:

- glue only on triangle-bearing edges,
- fill isolated edges arbitrarily only afterwards, when constructing a full
  local boundary-orientation datum.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Edge simplices that actually occur in the boundary of at least one
triangle simplex. -/
abbrev TriangleBoundaryEdge
    (m : Nat) (U : Cover beta alpha) :=
  { edge : ↑(edgeSimplices m U) // Nonempty (EdgeTriangleWitness (beta := beta) m U edge) }

/-- Gluing datum restricted only to edges that appear in some triangle boundary. -/
structure GluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) where
  edgeFaces :
    ∀ edge : TriangleBoundaryEdge (beta := beta) m U,
      LocalFaceOrientationDatum (beta := beta) edge.1.1
  agrees :
    ∀ (edge : TriangleBoundaryEdge (beta := beta) m U)
      {triangle : ↑(triangleSimplices m U)}
      (hedge : edge.1.1 ∈ codimOneFaces (beta := beta) triangle.1),
      edgeFaces edge =
        (tau triangle).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle edge.1 hedge

/-- Existence form of the boundary-edge-only gluing datum. -/
def HasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  Nonempty (GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau)

/-- Coherent triangle-induced edge orientations restrict to a gluing datum on
triangle-bearing edges. -/
noncomputable def gluedTriangleBoundaryEdgeOrientationsOfCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau where
  edgeFaces := fun edge =>
    coherentGlobalEdgeLocalOrientationDatum (beta := beta) m U tau edge.1
  agrees := fun edge triangle hedge =>
    coherentGlobalEdgeLocalOrientationDatum_eq_triangle_edge_orientation
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh edge.1 triangle hedge

theorem hasGluedTriangleBoundaryEdgeOrientations_of_triangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau := by
  exact ⟨gluedTriangleBoundaryEdgeOrientationsOfCoherent
    (beta := beta) (m := m) (U := U) (tau := tau) hcoh⟩

theorem triangleEdgeOrientationCoherence_of_gluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge triangle₁ triangle₂ h₁ h₂
  let boundaryEdge : TriangleBoundaryEdge (beta := beta) m U := ⟨edge, ⟨⟨triangle₁, h₁⟩⟩⟩
  calc
    (tau triangle₁).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle₁ edge h₁
      = eta.edgeFaces boundaryEdge := by
        simpa [boundaryEdge] using (eta.agrees boundaryEdge (triangle := triangle₁) h₁).symm
    _ = (tau triangle₂).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle₂ edge h₂ := by
        simpa [boundaryEdge] using eta.agrees boundaryEdge (triangle := triangle₂) h₂

theorem triangleEdgeOrientationCoherence_of_hasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  rcases hglue with ⟨eta⟩
  exact triangleEdgeOrientationCoherence_of_gluedTriangleBoundaryEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau) eta

theorem triangleEdgeOrientationCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau ↔
      HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau := by
  constructor
  · exact hasGluedTriangleBoundaryEdgeOrientations_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact triangleEdgeOrientationCoherence_of_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)

/-- Fill isolated edges arbitrarily, while preserving the glued data on
triangle-bearing edges. -/
noncomputable def filledEdgeFacesOfBoundaryEdgeGluing
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    ∀ edge : ↑(edgeSimplices m U), LocalFaceOrientationDatum (beta := beta) edge.1 := by
  classical
  intro edge
  by_cases h : Nonempty (EdgeTriangleWitness (beta := beta) m U edge)
  · exact eta.edgeFaces ⟨edge, h⟩
  · exact localOrientationFromChosenFace (beta := beta)
      (Classical.choice
        (localFaceSupport_nonempty_of_edge (beta := beta) (m := m) (U := U) edge))

/-- Extend a boundary-edge-only gluing datum to the old all-edges gluing datum
by filling isolated edges arbitrarily. -/
noncomputable def fullGluedTriangleEdgeOrientationsOfBoundaryEdgeGluing
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    GluedTriangleEdgeOrientations (beta := beta) m U tau where
  edgeFaces := filledEdgeFacesOfBoundaryEdgeGluing (beta := beta) m U tau eta
  agrees := by
    classical
    intro edge triangle hedge
    have h : Nonempty (EdgeTriangleWitness (beta := beta) m U edge) := ⟨⟨triangle, hedge⟩⟩
    unfold filledEdgeFacesOfBoundaryEdgeGluing
    simp [h]
    simpa using
      eta.agrees ⟨edge, h⟩ (triangle := triangle) hedge

theorem hasGluedTriangleEdgeOrientations_of_hasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    HasGluedTriangleEdgeOrientations (beta := beta) m U tau := by
  rcases hglue with ⟨eta⟩
  exact ⟨fullGluedTriangleEdgeOrientationsOfBoundaryEdgeGluing
    (beta := beta) (m := m) (U := U) (tau := tau) eta⟩

theorem hasCompatibleLocalOrientation_of_hasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_hasGluedTriangleEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau)
    (hasGluedTriangleEdgeOrientations_of_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau) hglue)

theorem hasBoundarySquareZero_of_hasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau) hglue)

end SCT.FND1
