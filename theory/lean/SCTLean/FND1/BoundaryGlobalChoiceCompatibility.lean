import SCTLean.FND1.BoundaryLocalOrientation
import SCTLean.FND1.BoundaryChoiceCompatibility

/-!
# FND-1 Boundary Global Choice Compatibility

This module packages the two noncanonical global data that are now known to
exist:

- one chosen codimension-one face for each edge,
- one ordered witness for each triangle.

It then records the honest global bridge currently justified by the previous
local work:

- if the edge choices agree triangle-by-triangle with the ordered witnesses,
  then each triangle enjoys the expected local cancellation pattern.

This is deliberately weaker than a whole-matrix `d_1 o d_2 = 0` theorem. It is
the correct next global layer before attempting that stronger statement.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Global compatibility between the edge-choice datum `chi` and the
triangle-choice datum `tau`: on every triangle, the chosen boundary faces on the
three boundary edges agree with the ordered witness carried by that triangle. -/
structure GlobalEdgeTriangleChoiceCompatibility
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop where
  compatible :
    ∀ triangle : ↑(triangleSimplices m U),
      EdgeChoicesCompatibleWithTriangleWitness
        (beta := beta) m U chi triangle (tau triangle)

/-- Assemble a global local-face orientation datum from:

- one chosen face on each edge,
- one ordered witness on each triangle.

This object is honest about its dependence on auxiliary global choices. -/
def choiceInducedBoundaryLocalOrientationDatum
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    BoundaryLocalOrientationDatum (beta := beta) m U where
  edgeFaces := inducedEdgeLocalOrientations (beta := beta) m U chi
  triangleFaces := fun triangle => (tau triangle).localTriangleOrientation

@[simp] theorem choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : ↑(edgeSimplices m U))
    (face : LocalFaceSupport (beta := beta) edge.1) :
    (choiceInducedBoundaryLocalOrientationDatum
      (beta := beta) m U chi tau).edgeFaces edge face =
      inducedEdgeLocalOrientations (beta := beta) m U chi edge face := by
  rfl

@[simp] theorem choiceInducedBoundaryLocalOrientationDatum_triangleFaces_apply
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (face : LocalFaceSupport (beta := beta) triangle.1) :
    (choiceInducedBoundaryLocalOrientationDatum
      (beta := beta) m U chi tau).triangleFaces triangle face =
      (tau triangle).localTriangleOrientation face := by
  rfl

theorem globalChoiceCompatibility_triangle_compatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle (tau triangle) :=
  hglobal.compatible triangle

theorem globalChoiceCompatibility_triangle_local_boundary_square_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    compatibleVertexSumAtA (beta := beta) m U chi triangle (tau triangle) = 0 ∧
    compatibleVertexSumAtB (beta := beta) m U chi triangle (tau triangle) = 0 ∧
    compatibleVertexSumAtC (beta := beta) m U chi triangle (tau triangle) = 0 := by
  exact compatible_triangle_local_boundary_square_zero
    (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)

end SCT.FND1
