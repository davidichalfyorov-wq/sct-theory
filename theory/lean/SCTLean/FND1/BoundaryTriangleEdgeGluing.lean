import SCTLean.FND1.BoundaryTriangleEdgeToLocal

/-!
# FND-1 Boundary Triangle Edge Gluing

This module reformulates the remaining noncanonical assumption as a genuine
gluing statement.

Instead of phrasing the gap as pairwise coherence between triangle-induced local
edge-orientation rules, we say:

- there exists one global edge-local orientation datum,
- and it agrees with every ordered triangle witness on every boundary edge of
  that triangle.

This gluing language is equivalent to the previous triangle-edge coherence
condition, but is more geometric and closer to a sheaf-style local-to-global
interface.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A global edge-local orientation datum glued from triangle witnesses. -/
structure GluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) where
  edgeFaces :
    ∀ edge : ↑(edgeSimplices m U),
      LocalFaceOrientationDatum (beta := beta) edge.1
  agrees :
    ∀ (edge : ↑(edgeSimplices m U))
      {triangle : ↑(triangleSimplices m U)}
      (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1),
      edgeFaces edge =
        (tau triangle).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle edge hedge

/-- Existence form of the gluing datum. -/
def HasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  Nonempty (GluedTriangleEdgeOrientations (beta := beta) m U tau)

/-- Coherent triangle-induced edge orientations directly define a glued global
edge-local orientation datum. -/
noncomputable def gluedTriangleEdgeOrientationsOfCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    GluedTriangleEdgeOrientations (beta := beta) m U tau where
  edgeFaces := coherentGlobalEdgeLocalOrientationDatum (beta := beta) m U tau
  agrees := coherentGlobalEdgeLocalOrientationDatum_eq_triangle_edge_orientation
    (beta := beta) (m := m) (U := U) (tau := tau) hcoh

theorem hasGluedTriangleEdgeOrientations_of_triangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    HasGluedTriangleEdgeOrientations (beta := beta) m U tau := by
  exact ⟨gluedTriangleEdgeOrientationsOfCoherent
    (beta := beta) (m := m) (U := U) (tau := tau) hcoh⟩

theorem triangleEdgeOrientationCoherence_of_gluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleEdgeOrientations (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge triangle₁ triangle₂ h₁ h₂
  calc
    (tau triangle₁).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle₁ edge h₁
      = eta.edgeFaces edge := by
        simpa using (eta.agrees edge (triangle := triangle₁) h₁).symm
    _ = (tau triangle₂).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle₂ edge h₂ := by
        simpa using eta.agrees edge (triangle := triangle₂) h₂

theorem triangleEdgeOrientationCoherence_of_hasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleEdgeOrientations (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  rcases hglue with ⟨eta⟩
  exact triangleEdgeOrientationCoherence_of_gluedTriangleEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau) eta

theorem triangleEdgeOrientationCoherence_iff_hasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau ↔
      HasGluedTriangleEdgeOrientations (beta := beta) m U tau := by
  constructor
  · exact hasGluedTriangleEdgeOrientations_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact triangleEdgeOrientationCoherence_of_hasGluedTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)

/-- Boundary-local orientation datum assembled from a glued edge-local datum and
the triangle-local orientations already carried by `tau`. -/
def gluedTriangleBoundaryLocalOrientationDatum
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleEdgeOrientations (beta := beta) m U tau) :
    BoundaryLocalOrientationDatum (beta := beta) m U where
  edgeFaces := eta.edgeFaces
  triangleFaces := fun triangle => (tau triangle).localTriangleOrientation

theorem hasCompatibleLocalOrientation_of_gluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleEdgeOrientations (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  have hcoh :=
    triangleEdgeOrientationCoherence_of_gluedTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau) eta
  exact hasCompatibleLocalOrientation_of_triangleEdgeOrientationCoherent
    (beta := beta) (m := m) (U := U) (tau := tau) hcoh

theorem hasCompatibleLocalOrientation_of_hasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleEdgeOrientations (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  rcases hglue with ⟨eta⟩
  exact hasCompatibleLocalOrientation_of_gluedTriangleEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau) eta

theorem hasBoundarySquareZero_of_hasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleEdgeOrientations (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_hasGluedTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau) hglue)

end SCT.FND1
