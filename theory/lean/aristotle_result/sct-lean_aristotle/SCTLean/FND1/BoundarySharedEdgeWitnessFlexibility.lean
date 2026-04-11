import SCTLean.FND1.BoundaryTriangleWitnessFlexibility

/-!
# FND-1 Boundary Shared-Edge Witness Flexibility

This module lifts the single-triangle witness-order flexibility obstruction to a
shared-edge situation.

If two triangle witnesses use the same boundary `AB` edge, then swapping the
first two vertices in one witness keeps the same edge but flips the chosen
singleton face on that edge. Therefore one of the two versions of that witness
must disagree with any fixed comparison witness on the same shared edge.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem swapABTriangleOrderedWitness_sharedEdgeAB_eq
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    (swapABTriangleOrderedWitness w2).edgeAB (beta := beta) m U triangle2 =
      w1.edgeAB (beta := beta) m U triangle1 := by
  calc
    (swapABTriangleOrderedWitness w2).edgeAB (beta := beta) m U triangle2 =
        w2.edgeAB (beta := beta) m U triangle2 := by
          simpa using swapABTriangleOrderedWitness_edgeAB_eq
            (beta := beta) (m := m) (U := U) triangle2 w2
    _ = w1.edgeAB (beta := beta) m U triangle1 := hshared.symm

/-- The chosen singleton face on the `AB` edge from one witness. -/
def edgeABFace
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) : Finset beta :=
  ((w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
    (w.edgeAB (beta := beta) m U triangle)
    (w.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
      (triangle := triangle))).1 : Finset beta)

/-- The chosen singleton face on the `AB` edge after swapping the first two
vertices in the witness. -/
def swapABEdgeFace
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) : Finset beta :=
  ((((swapABTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
    (beta := beta) m U triangle
    ((swapABTriangleOrderedWitness w).edgeAB (beta := beta) m U triangle)
    ((swapABTriangleOrderedWitness w).edgeAB_mem_codimOneFaces_triangle
      (beta := beta) (m := m) (U := U) (triangle := triangle))).1 : Finset beta))

@[simp] theorem edgeABFace_eq
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    edgeABFace (beta := beta) triangle w = ({w.b} : Finset beta) := by
  simp [edgeABFace]

@[simp] theorem swapABEdgeFace_eq
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    swapABEdgeFace (beta := beta) triangle w = ({w.a} : Finset beta) := by
  simpa [swapABEdgeFace] using
    (chosenFace_edgeAB_val_of_swapABTriangleOrderedWitness
      (beta := beta) (m := m) (U := U) triangle w)

theorem edgeABFace_ne_or_swapped_face_ne
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1) :
    edgeABFace (beta := beta) triangle1 w1 ≠ edgeABFace (beta := beta) triangle2 w2 ∨
      edgeABFace (beta := beta) triangle1 w1 ≠ swapABEdgeFace (beta := beta) triangle2 w2 := by
  by_cases hne :
      edgeABFace (beta := beta) triangle1 w1 ≠ edgeABFace (beta := beta) triangle2 w2
  · exact Or.inl hne
  · right
    intro hEqSwap
    have hsingle : ({w2.b} : Finset beta) = ({w2.a} : Finset beta) := by
      calc
        ({w2.b} : Finset beta) = edgeABFace (beta := beta) triangle2 w2 := by
          symm
          exact edgeABFace_eq (beta := beta) triangle2 w2
        _ = edgeABFace (beta := beta) triangle1 w1 := (not_ne_iff.mp hne).symm
        _ = swapABEdgeFace (beta := beta) triangle2 w2 := hEqSwap
        _ = ({w2.a} : Finset beta) := swapABEdgeFace_eq (beta := beta) triangle2 w2
    have hba : w2.b = w2.a := by
      simpa using hsingle
    exact w2.hab hba.symm

theorem sharedEdgeAB_witness_flexibility
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    (swapABTriangleOrderedWitness w2).edgeAB (beta := beta) m U triangle2 =
        w1.edgeAB (beta := beta) m U triangle1 ∧
      (edgeABFace (beta := beta) triangle1 w1 ≠ edgeABFace (beta := beta) triangle2 w2 ∨
        edgeABFace (beta := beta) triangle1 w1 ≠ swapABEdgeFace (beta := beta) triangle2 w2) := by
  constructor
  · exact swapABTriangleOrderedWitness_sharedEdgeAB_eq
      (beta := beta) triangle1 triangle2 w1 w2 hshared
  · exact edgeABFace_ne_or_swapped_face_ne (beta := beta) triangle1 triangle2 w1 w2

theorem sharedEdgeAB_face_ne_or_swapped_face_ne
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    edgeABFace (beta := beta) triangle1 w1 ≠ edgeABFace (beta := beta) triangle2 w2 ∨
      edgeABFace (beta := beta) triangle1 w1 ≠ swapABEdgeFace (beta := beta) triangle2 w2 := by
  exact (sharedEdgeAB_witness_flexibility
    (beta := beta) triangle1 triangle2 w1 w2 hshared).2

end SCT.FND1
