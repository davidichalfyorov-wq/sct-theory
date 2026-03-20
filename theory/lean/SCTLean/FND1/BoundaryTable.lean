import SCTLean.FND1.BoundarySignedIncidence

/-!
# FND-1 Boundary Table

This module turns signed incidence data into an explicit finite table with row
and column index sets. This is the last combinatorial stop before one starts
talking about actual linear operators or matrices.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A finite boundary table consists of:

- an explicit row index set,
- an explicit column index set,
- a signed incidence object whose support lives inside those indices.

This is intentionally still pre-linear-algebra. -/
structure BoundaryTableData (row col : Type*) [DecidableEq row] [DecidableEq col] where
  rows : Finset row
  cols : Finset col
  incidence : SignedIncidenceData row col
  support_row_mem : ∀ {r c}, (r, c) ∈ incidence.support -> r ∈ rows
  support_col_mem : ∀ {r c}, (r, c) ∈ incidence.support -> c ∈ cols

namespace BoundaryTableData

variable {row col : Type*} [DecidableEq row] [DecidableEq col]

/-- Table entry, exposed as a convenient wrapper around the underlying signed incidence coefficient. -/
def entry (T : BoundaryTableData row col) (r : row) (c : col) : Int :=
  T.incidence.coeff r c

theorem entry_nonzero_iff_support (T : BoundaryTableData row col) {r : row} {c : col} :
    T.entry r c ≠ 0 ↔ (r, c) ∈ T.incidence.support := by
  unfold entry
  exact T.incidence.coeff_nonzero_iff

theorem row_mem_of_nonzero (T : BoundaryTableData row col) {r : row} {c : col}
    (h : T.entry r c ≠ 0) :
    r ∈ T.rows := by
  exact T.support_row_mem ((T.entry_nonzero_iff_support).mp h)

theorem col_mem_of_nonzero (T : BoundaryTableData row col) {r : row} {c : col}
    (h : T.entry r c ≠ 0) :
    c ∈ T.cols := by
  exact T.support_col_mem ((T.entry_nonzero_iff_support).mp h)

theorem entry_eq_one_or_neg_one_of_nonzero (T : BoundaryTableData row col) {r : row} {c : col}
    (h : T.entry r c ≠ 0) :
    T.entry r c = 1 ∨ T.entry r c = -1 := by
  exact T.incidence.coeff_eq_one_or_neg_one_of_mem ((T.entry_nonzero_iff_support).mp h)

end BoundaryTableData

/-- Explicit edge-to-vertex boundary table. Rows are vertices, columns are edges. -/
def edgeVertexBoundaryTable (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U) :
    BoundaryTableData (Finset beta) (Finset beta) where
  rows := vertexSimplices (beta := beta)
  cols := edgeSimplices m U
  incidence := edgeVertexSignedIncidence (beta := beta) m U omega
  support_row_mem := by
    intro face edge h
    have h' : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U := by
      simpa [edgeVertexSignedIncidence] using h
    exact edgeVertexIncidences_target_vertices (beta := beta) (m := m) (U := U) h'
  support_col_mem := by
    intro face edge h
    have h' : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U := by
      simpa [edgeVertexSignedIncidence] using h
    exact (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp h' |>.1

/-- Explicit triangle-to-edge boundary table. Rows are edges, columns are triangles. -/
def triangleEdgeBoundaryTable (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U) :
    BoundaryTableData (Finset beta) (Finset beta) where
  rows := edgeSimplices m U
  cols := triangleSimplices m U
  incidence := triangleEdgeSignedIncidence (beta := beta) m U omega
  support_row_mem := by
    intro face triangle h
    have h' : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U := by
      simpa [triangleEdgeSignedIncidence] using h
    exact triangleEdgeIncidences_target_edges (beta := beta) (m := m) (U := U) h'
  support_col_mem := by
    intro face triangle h
    have h' : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U := by
      simpa [triangleEdgeSignedIncidence] using h
    exact (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp h' |>.1

theorem edgeVertexBoundaryTable_row_mem_of_nonzero (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (edgeVertexBoundaryTable (beta := beta) m U omega).entry face edge ≠ 0) :
    face ∈ vertexSimplices (beta := beta) := by
  exact BoundaryTableData.row_mem_of_nonzero _ h

theorem edgeVertexBoundaryTable_col_mem_of_nonzero (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (edgeVertexBoundaryTable (beta := beta) m U omega).entry face edge ≠ 0) :
    edge ∈ edgeSimplices m U := by
  exact BoundaryTableData.col_mem_of_nonzero _ h

theorem triangleEdgeBoundaryTable_row_mem_of_nonzero (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (triangleEdgeBoundaryTable (beta := beta) m U omega).entry face triangle ≠ 0) :
    face ∈ edgeSimplices m U := by
  exact BoundaryTableData.row_mem_of_nonzero _ h

theorem triangleEdgeBoundaryTable_col_mem_of_nonzero (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (triangleEdgeBoundaryTable (beta := beta) m U omega).entry face triangle ≠ 0) :
    triangle ∈ triangleSimplices m U := by
  exact BoundaryTableData.col_mem_of_nonzero _ h

end SCT.FND1
