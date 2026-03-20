import Mathlib.Data.Matrix.Basic
import SCTLean.FND1.BoundaryTable

/-!
# FND-1 Boundary Matrix

This module turns the already formalized finite boundary tables into honest
matrix-valued objects. The construction is intentionally minimal:

- row indices are exactly the elements of the row finset,
- column indices are exactly the elements of the column finset,
- entries are inherited from the boundary table,
- no extra geometric meaning is smuggled in at this stage.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

namespace BoundaryTableData

variable {row col : Type*} [DecidableEq row] [DecidableEq col]

/-- Row index type induced by the finite row set. -/
abbrev RowIndex (T : BoundaryTableData row col) := ↑T.rows

/-- Column index type induced by the finite column set. -/
abbrev ColIndex (T : BoundaryTableData row col) := ↑T.cols

/-- Matrix representation of a finite boundary table. -/
def toMatrix (T : BoundaryTableData row col) : Matrix (T.RowIndex) (T.ColIndex) Int :=
  fun i j => T.entry i.1 j.1

@[simp] theorem toMatrix_apply (T : BoundaryTableData row col)
    (i : T.RowIndex) (j : T.ColIndex) :
    T.toMatrix i j = T.entry i.1 j.1 := rfl

theorem toMatrix_entry_nonzero_iff (T : BoundaryTableData row col)
    {i : T.RowIndex} {j : T.ColIndex} :
    T.toMatrix i j ≠ 0 ↔ (i.1, j.1) ∈ T.incidence.support := by
  exact T.entry_nonzero_iff_support

theorem toMatrix_entry_eq_one_or_neg_one_of_nonzero (T : BoundaryTableData row col)
    {i : T.RowIndex} {j : T.ColIndex}
    (h : T.toMatrix i j ≠ 0) :
    T.toMatrix i j = 1 ∨ T.toMatrix i j = -1 := by
  exact T.entry_eq_one_or_neg_one_of_nonzero h

theorem rowIndex_val_mem (T : BoundaryTableData row col) (i : T.RowIndex) :
    i.1 ∈ T.rows := i.2

theorem colIndex_val_mem (T : BoundaryTableData row col) (j : T.ColIndex) :
    j.1 ∈ T.cols := j.2

end BoundaryTableData

/-- Edge-to-vertex boundary matrix. -/
abbrev edgeVertexBoundaryMatrix (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U) :=
  (edgeVertexBoundaryTable (beta := beta) m U omega).toMatrix

/-- Triangle-to-edge boundary matrix. -/
abbrev triangleEdgeBoundaryMatrix (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U) :=
  (triangleEdgeBoundaryTable (beta := beta) m U omega).toMatrix

theorem edgeVertexBoundaryMatrix_entry_nonzero_iff (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {i : (edgeVertexBoundaryTable (beta := beta) m U omega).RowIndex}
    {j : (edgeVertexBoundaryTable (beta := beta) m U omega).ColIndex} :
    edgeVertexBoundaryMatrix (beta := beta) m U omega i j ≠ 0 ↔
      (i.1, j.1) ∈ edgeVertexIncidences (beta := beta) m U := by
  exact (edgeVertexBoundaryTable (beta := beta) m U omega).toMatrix_entry_nonzero_iff

theorem triangleEdgeBoundaryMatrix_entry_nonzero_iff (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {i : (triangleEdgeBoundaryTable (beta := beta) m U omega).RowIndex}
    {j : (triangleEdgeBoundaryTable (beta := beta) m U omega).ColIndex} :
    triangleEdgeBoundaryMatrix (beta := beta) m U omega i j ≠ 0 ↔
      (i.1, j.1) ∈ triangleEdgeIncidences (beta := beta) m U := by
  exact (triangleEdgeBoundaryTable (beta := beta) m U omega).toMatrix_entry_nonzero_iff

theorem edgeVertexBoundaryMatrix_entry_eq_one_or_neg_one_of_nonzero
    (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {i : (edgeVertexBoundaryTable (beta := beta) m U omega).RowIndex}
    {j : (edgeVertexBoundaryTable (beta := beta) m U omega).ColIndex}
    (h : edgeVertexBoundaryMatrix (beta := beta) m U omega i j ≠ 0) :
    edgeVertexBoundaryMatrix (beta := beta) m U omega i j = 1 ∨
      edgeVertexBoundaryMatrix (beta := beta) m U omega i j = -1 := by
  exact (edgeVertexBoundaryTable (beta := beta) m U omega).toMatrix_entry_eq_one_or_neg_one_of_nonzero h

theorem triangleEdgeBoundaryMatrix_entry_eq_one_or_neg_one_of_nonzero
    (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {i : (triangleEdgeBoundaryTable (beta := beta) m U omega).RowIndex}
    {j : (triangleEdgeBoundaryTable (beta := beta) m U omega).ColIndex}
    (h : triangleEdgeBoundaryMatrix (beta := beta) m U omega i j ≠ 0) :
    triangleEdgeBoundaryMatrix (beta := beta) m U omega i j = 1 ∨
      triangleEdgeBoundaryMatrix (beta := beta) m U omega i j = -1 := by
  exact (triangleEdgeBoundaryTable (beta := beta) m U omega).toMatrix_entry_eq_one_or_neg_one_of_nonzero h

end SCT.FND1
