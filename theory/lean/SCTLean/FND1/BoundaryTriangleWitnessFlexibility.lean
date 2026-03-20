import SCTLean.FND1.BoundaryTriangleEdgeCoherence

/-!
# FND-1 Boundary Triangle Witness Flexibility

This module isolates a simple but important obstruction signal.

For one fixed triangle, the same underlying boundary edge can receive different
chosen singleton faces purely by changing the ordered witness on that triangle.
So the current chosen-face law does not come from the triangle support alone.
-/

namespace SCT.FND1

universe v

variable {beta : Type v} [Fintype beta] [DecidableEq beta]

/-- Swap the first two vertices in an ordered triangle witness. -/
def swapABTriangleOrderedWitness
    {triangle : Finset beta}
    (w : TriangleOrderedWitness triangle) :
    TriangleOrderedWitness triangle where
  a := w.b
  b := w.a
  c := w.c
  hab := w.hab.symm
  hac := w.hbc
  hbc := w.hac
  triangle_eq := by
    calc
      triangle = ({w.a, w.b, w.c} : Finset beta) := w.triangle_eq
      _ = ({w.b, w.a, w.c} : Finset beta) := by
        ext x
        simp [or_left_comm]

theorem swapABTriangleOrderedWitness_edgeAB_eq
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (swapABTriangleOrderedWitness w).edgeAB (beta := beta) m U triangle =
      w.edgeAB (beta := beta) m U triangle := by
  apply Subtype.ext
  simp [TriangleOrderedWitness.edgeAB, swapABTriangleOrderedWitness, Finset.pair_comm]

theorem chosenFace_edgeAB_val_of_swapABTriangleOrderedWitness
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ((swapABTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
      (beta := beta) m U triangle
      ((swapABTriangleOrderedWitness w).edgeAB (beta := beta) m U triangle)
      ((swapABTriangleOrderedWitness w).edgeAB_mem_codimOneFaces_triangle
        (beta := beta) (m := m) (U := U) (triangle := triangle))).1 = ({w.a} : Finset beta) := by
  simp [swapABTriangleOrderedWitness]

theorem chosenFace_edgeAB_differs_under_swapAB
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ((w.chosenFaceOnBoundaryEdge
      (beta := beta) m U triangle
      (w.edgeAB (beta := beta) m U triangle)
      (w.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle))).1) ≠
      (((swapABTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle
        ((swapABTriangleOrderedWitness w).edgeAB (beta := beta) m U triangle)
        ((swapABTriangleOrderedWitness w).edgeAB_mem_codimOneFaces_triangle
          (beta := beta) (m := m) (U := U) (triangle := triangle))).1) := by
  intro hEq
  have hsingle : ({w.b} : Finset beta) = ({w.a} : Finset beta) := by
    simpa [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAB_val,
      chosenFace_edgeAB_val_of_swapABTriangleOrderedWitness] using hEq
  have hba : w.b = w.a := by
    simpa using hsingle
  exact w.hab hba.symm

/-- Swap the first and third vertices in an ordered triangle witness. -/
def swapACTriangleOrderedWitness
    {triangle : Finset beta}
    (w : TriangleOrderedWitness triangle) :
    TriangleOrderedWitness triangle where
  a := w.c
  b := w.b
  c := w.a
  hab := w.hbc.symm
  hac := w.hac.symm
  hbc := w.hab.symm
  triangle_eq := by
    calc
      triangle = ({w.a, w.b, w.c} : Finset beta) := w.triangle_eq
      _ = ({w.c, w.b, w.a} : Finset beta) := by
        ext x
        simp [Finset.mem_insert, Finset.mem_singleton, or_left_comm, or_comm]

theorem swapACTriangleOrderedWitness_edgeAC_eq
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (swapACTriangleOrderedWitness w).edgeAC (beta := beta) m U triangle =
      w.edgeAC (beta := beta) m U triangle := by
  apply Subtype.ext
  simp [TriangleOrderedWitness.edgeAC, swapACTriangleOrderedWitness, Finset.pair_comm]

theorem chosenFace_edgeAC_val_of_swapACTriangleOrderedWitness
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ((swapACTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
      (beta := beta) m U triangle
      ((swapACTriangleOrderedWitness w).edgeAC (beta := beta) m U triangle)
      ((swapACTriangleOrderedWitness w).edgeAC_mem_codimOneFaces_triangle
        (beta := beta) (m := m) (U := U) (triangle := triangle))).1 = ({w.a} : Finset beta) := by
  simp [swapACTriangleOrderedWitness]

theorem chosenFace_edgeAC_differs_under_swapAC
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ((w.chosenFaceOnBoundaryEdge
      (beta := beta) m U triangle
      (w.edgeAC (beta := beta) m U triangle)
      (w.edgeAC_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle))).1) ≠
      (((swapACTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle
        ((swapACTriangleOrderedWitness w).edgeAC (beta := beta) m U triangle)
        ((swapACTriangleOrderedWitness w).edgeAC_mem_codimOneFaces_triangle
          (beta := beta) (m := m) (U := U) (triangle := triangle))).1) := by
  intro hEq
  have hsingle : ({w.c} : Finset beta) = ({w.a} : Finset beta) := by
    simpa [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAC_val,
      chosenFace_edgeAC_val_of_swapACTriangleOrderedWitness] using hEq
  have hca : w.c = w.a := by
    simpa using hsingle
  exact w.hac hca.symm

/-- Swap the second and third vertices in an ordered triangle witness. -/
def swapBCTriangleOrderedWitness
    {triangle : Finset beta}
    (w : TriangleOrderedWitness triangle) :
    TriangleOrderedWitness triangle where
  a := w.a
  b := w.c
  c := w.b
  hab := w.hac
  hac := w.hab
  hbc := w.hbc.symm
  triangle_eq := by
    calc
      triangle = ({w.a, w.b, w.c} : Finset beta) := w.triangle_eq
      _ = ({w.a, w.c, w.b} : Finset beta) := by
        ext x
        simp [Finset.mem_insert, Finset.mem_singleton, or_comm]

theorem swapBCTriangleOrderedWitness_edgeBC_eq
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (swapBCTriangleOrderedWitness w).edgeBC (beta := beta) m U triangle =
      w.edgeBC (beta := beta) m U triangle := by
  apply Subtype.ext
  simp [TriangleOrderedWitness.edgeBC, swapBCTriangleOrderedWitness, Finset.pair_comm]

theorem chosenFace_edgeBC_val_of_swapBCTriangleOrderedWitness
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ((swapBCTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
      (beta := beta) m U triangle
      ((swapBCTriangleOrderedWitness w).edgeBC (beta := beta) m U triangle)
      ((swapBCTriangleOrderedWitness w).edgeBC_mem_codimOneFaces_triangle
        (beta := beta) (m := m) (U := U) (triangle := triangle))).1 = ({w.b} : Finset beta) := by
  simp [swapBCTriangleOrderedWitness]

theorem chosenFace_edgeBC_differs_under_swapBC
    {alpha : Type _} [DecidableEq alpha]
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ((w.chosenFaceOnBoundaryEdge
      (beta := beta) m U triangle
      (w.edgeBC (beta := beta) m U triangle)
      (w.edgeBC_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle))).1) ≠
      (((swapBCTriangleOrderedWitness w).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle
        ((swapBCTriangleOrderedWitness w).edgeBC (beta := beta) m U triangle)
        ((swapBCTriangleOrderedWitness w).edgeBC_mem_codimOneFaces_triangle
          (beta := beta) (m := m) (U := U) (triangle := triangle))).1) := by
  intro hEq
  have hsingle : ({w.c} : Finset beta) = ({w.b} : Finset beta) := by
    simpa [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeBC_val,
      chosenFace_edgeBC_val_of_swapBCTriangleOrderedWitness] using hEq
  have hcb : w.c = w.b := by
    simpa using hsingle
  exact w.hbc hcb.symm

end SCT.FND1
