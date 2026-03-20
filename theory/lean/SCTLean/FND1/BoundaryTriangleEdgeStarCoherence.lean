import SCTLean.FND1.BoundaryTriangleBoundaryEdgeGluing

/-!
# FND-1 Boundary Triangle Edge-Star Coherence

This module rewrites the current remaining gap in a completely edge-local form.

Instead of asking for one glued family over all triangle-bearing edges, we ask:

- for each triangle-bearing edge separately,
- does there exist one local face-orientation rule on that edge,
- agreeing with every triangle witness in the star of that edge?

This is equivalent to the boundary-edge gluing layer, but is more pointwise and
strictly local over edge-stars.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Pointwise local coherence over the star of each triangle-bearing edge. -/
structure LocalTriangleEdgeStarCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop where
  coherent :
    ∀ edge : TriangleBoundaryEdge (beta := beta) m U,
      ∃ theta : LocalFaceOrientationDatum (beta := beta) edge.1.1,
        ∀ {triangle : ↑(triangleSimplices m U)}
          (hedge : edge.1.1 ∈ codimOneFaces (beta := beta) triangle.1),
          theta =
            (tau triangle).localEdgeOrientationOnBoundaryEdge
              (beta := beta) m U triangle edge.1 hedge

theorem localTriangleEdgeStarCoherence_of_gluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau) :
    LocalTriangleEdgeStarCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge
  refine ⟨eta.edgeFaces edge, ?_⟩
  intro triangle hedge
  exact eta.agrees edge (triangle := triangle) hedge

noncomputable def gluedTriangleBoundaryEdgeOrientationsOfLocalEdgeStarCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : LocalTriangleEdgeStarCoherence (beta := beta) m U tau) :
    GluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau where
  edgeFaces := fun edge => Classical.choose (hstar.coherent edge)
  agrees := by
    intro edge triangle hedge
    exact Classical.choose_spec (hstar.coherent edge) hedge

theorem localTriangleEdgeStarCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    LocalTriangleEdgeStarCoherence (beta := beta) m U tau ↔
      HasGluedTriangleBoundaryEdgeOrientations (beta := beta) m U tau := by
  constructor
  · intro hstar
    exact ⟨gluedTriangleBoundaryEdgeOrientationsOfLocalEdgeStarCoherence
      (beta := beta) (m := m) (U := U) (tau := tau) hstar⟩
  · intro hglue
    rcases hglue with ⟨eta⟩
    exact localTriangleEdgeStarCoherence_of_gluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau) eta

theorem localTriangleEdgeStarCoherence_of_triangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    LocalTriangleEdgeStarCoherence (beta := beta) m U tau := by
  exact (localTriangleEdgeStarCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)).2
    (hasGluedTriangleBoundaryEdgeOrientations_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh)

theorem triangleEdgeOrientationCoherence_of_localTriangleEdgeStarCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : LocalTriangleEdgeStarCoherence (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  exact triangleEdgeOrientationCoherence_of_hasGluedTriangleBoundaryEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((localTriangleEdgeStarCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)).1 hstar)

theorem triangleEdgeOrientationCoherence_iff_localTriangleEdgeStarCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau ↔
      LocalTriangleEdgeStarCoherence (beta := beta) m U tau := by
  constructor
  · exact localTriangleEdgeStarCoherence_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact triangleEdgeOrientationCoherence_of_localTriangleEdgeStarCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem hasCompatibleLocalOrientation_of_localTriangleEdgeStarCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : LocalTriangleEdgeStarCoherence (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_hasGluedTriangleBoundaryEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((localTriangleEdgeStarCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)).1 hstar)

theorem hasBoundarySquareZero_of_localTriangleEdgeStarCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hstar : LocalTriangleEdgeStarCoherence (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_localTriangleEdgeStarCoherence
      (beta := beta) (m := m) (U := U) (tau := tau) hstar)

end SCT.FND1
