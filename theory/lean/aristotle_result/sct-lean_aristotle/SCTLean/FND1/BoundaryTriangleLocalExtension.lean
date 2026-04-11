import SCTLean.FND1.BoundaryTriangleEdgeGluing

/-!
# FND-1 Boundary Triangle Local Extension

This module packages the remaining gap as an explicit extension problem.

Given a triangle-witness datum `tau`, ask whether it extends to a full local
boundary-orientation datum:

- the triangle-face part must equal the local triangle orientations already
  carried by `tau`,
- the edge-face part must agree with the local edge orientations induced by
  every triangle witness on each boundary edge.

This extension problem is equivalent to the gluing formulation and therefore to
the current triangle-edge coherence condition.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A full local boundary-orientation datum extending the triangle witness data
`tau`. -/
structure TriangleWitnessLocalExtension
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) where
  theta : BoundaryLocalOrientationDatum (beta := beta) m U
  triangleFaces_eq :
    ∀ triangle : ↑(triangleSimplices m U),
      theta.triangleFaces triangle = (tau triangle).localTriangleOrientation
  edgeFaces_agree :
    ∀ (edge : ↑(edgeSimplices m U))
      {triangle : ↑(triangleSimplices m U)}
      (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1),
      theta.edgeFaces edge =
        (tau triangle).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle edge hedge

/-- Existential form of the triangle-witness extension problem. -/
def HasTriangleWitnessLocalExtension
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  Nonempty (TriangleWitnessLocalExtension (beta := beta) m U tau)

/-- A glued edge-local orientation datum gives a full extension of the triangle
witness data. -/
def triangleWitnessLocalExtensionOfGlued
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedTriangleEdgeOrientations (beta := beta) m U tau) :
    TriangleWitnessLocalExtension (beta := beta) m U tau where
  theta := gluedTriangleBoundaryLocalOrientationDatum
    (beta := beta) (m := m) (U := U) (tau := tau) eta
  triangleFaces_eq := by
    intro triangle
    rfl
  edgeFaces_agree := by
    intro edge triangle hedge
    exact eta.agrees edge (triangle := triangle) hedge

/-- Any full extension yields the glued edge-local orientation datum by taking
its edge-face layer. -/
def gluedTriangleEdgeOrientationsOfLocalExtension
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (ext : TriangleWitnessLocalExtension (beta := beta) m U tau) :
    GluedTriangleEdgeOrientations (beta := beta) m U tau where
  edgeFaces := ext.theta.edgeFaces
  agrees := by
    intro edge triangle hedge
    exact ext.edgeFaces_agree edge (triangle := triangle) hedge

theorem hasTriangleWitnessLocalExtension_of_hasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedTriangleEdgeOrientations (beta := beta) m U tau) :
    HasTriangleWitnessLocalExtension (beta := beta) m U tau := by
  rcases hglue with ⟨eta⟩
  exact ⟨triangleWitnessLocalExtensionOfGlued
    (beta := beta) (m := m) (U := U) (tau := tau) eta⟩

theorem hasGluedTriangleEdgeOrientations_of_hasTriangleWitnessLocalExtension
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hext : HasTriangleWitnessLocalExtension (beta := beta) m U tau) :
    HasGluedTriangleEdgeOrientations (beta := beta) m U tau := by
  rcases hext with ⟨ext⟩
  exact ⟨gluedTriangleEdgeOrientationsOfLocalExtension
    (beta := beta) (m := m) (U := U) (tau := tau) ext⟩

theorem hasTriangleWitnessLocalExtension_iff_hasGluedTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    HasTriangleWitnessLocalExtension (beta := beta) m U tau ↔
      HasGluedTriangleEdgeOrientations (beta := beta) m U tau := by
  constructor
  · exact hasGluedTriangleEdgeOrientations_of_hasTriangleWitnessLocalExtension
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact hasTriangleWitnessLocalExtension_of_hasGluedTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem hasTriangleWitnessLocalExtension_iff_triangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    HasTriangleWitnessLocalExtension (beta := beta) m U tau ↔
      GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  rw [hasTriangleWitnessLocalExtension_iff_hasGluedTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)]
  rw [← triangleEdgeOrientationCoherence_iff_hasGluedTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)]

theorem hasCompatibleLocalOrientation_of_hasTriangleWitnessLocalExtension
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hext : HasTriangleWitnessLocalExtension (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_hasGluedTriangleEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau)
    (hasGluedTriangleEdgeOrientations_of_hasTriangleWitnessLocalExtension
      (beta := beta) (m := m) (U := U) (tau := tau) hext)

theorem hasBoundarySquareZero_of_hasTriangleWitnessLocalExtension
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hext : HasTriangleWitnessLocalExtension (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_hasTriangleWitnessLocalExtension
      (beta := beta) (m := m) (U := U) (tau := tau) hext)

end SCT.FND1
