import SCTLean.FND1.BoundaryOrientation

/-!
# FND-1 Boundary Signed Incidence

This module packages the already constructed unsigned support together with an
explicit auxiliary orientation datum into honest signed incidence objects.

Crucially:

- support remains canonical,
- signs remain external data,
- the packaged object only states what is actually available now.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A finite signed incidence object: support plus an integer-valued coefficient
function that is nonzero exactly on that support and takes only `±1` there. -/
structure SignedIncidenceData (row col : Type*) [DecidableEq row] [DecidableEq col] where
  support : Finset (row × col)
  coeff : row → col → Int
  coeff_nonzero_iff : ∀ {r c}, coeff r c ≠ 0 ↔ (r, c) ∈ support
  coeff_eq_one_or_neg_one_of_mem : ∀ {r c}, (r, c) ∈ support → coeff r c = 1 ∨ coeff r c = -1

namespace SignedIncidenceData

variable {row col : Type*} [DecidableEq row] [DecidableEq col]

theorem coeff_eq_zero_of_not_mem (M : SignedIncidenceData row col) {r : row} {c : col}
    (h : (r, c) ∉ M.support) :
    M.coeff r c = 0 := by
  by_contra hcoeff
  exact h ((M.coeff_nonzero_iff).mp hcoeff)

theorem coeff_nonzero_of_mem (M : SignedIncidenceData row col) {r : row} {c : col}
    (h : (r, c) ∈ M.support) :
    M.coeff r c ≠ 0 := by
  exact (M.coeff_nonzero_iff).mpr h

end SignedIncidenceData

/-- Packaged signed incidence from edges to vertices. -/
def edgeVertexSignedIncidence (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U) :
    SignedIncidenceData (Finset beta) (Finset beta) where
  support := edgeVertexIncidences (beta := beta) m U
  coeff := edgeVertexCoefficient (beta := beta) m U omega
  coeff_nonzero_iff := by
    intro face edge
    exact edgeVertexCoefficient_nonzero_iff (beta := beta) (m := m) (U := U) (omega := omega)
  coeff_eq_one_or_neg_one_of_mem := by
    intro face edge h
    exact edgeVertexCoefficient_eq_one_or_neg_one_of_mem
      (beta := beta) (m := m) (U := U) (omega := omega) h

/-- Packaged signed incidence from triangles to edges. -/
def triangleEdgeSignedIncidence (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U) :
    SignedIncidenceData (Finset beta) (Finset beta) where
  support := triangleEdgeIncidences (beta := beta) m U
  coeff := triangleEdgeCoefficient (beta := beta) m U omega
  coeff_nonzero_iff := by
    intro face triangle
    exact triangleEdgeCoefficient_nonzero_iff
      (beta := beta) (m := m) (U := U) (omega := omega)
  coeff_eq_one_or_neg_one_of_mem := by
    intro face triangle h
    exact triangleEdgeCoefficient_eq_one_or_neg_one_of_mem
      (beta := beta) (m := m) (U := U) (omega := omega) h

theorem edgeVertexSignedIncidence_target_vertices (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (edgeVertexSignedIncidence (beta := beta) m U omega).coeff face edge ≠ 0) :
    face ∈ vertexSimplices (beta := beta) := by
  have hmem :
      (face, edge) ∈ (edgeVertexSignedIncidence (beta := beta) m U omega).support :=
    ((edgeVertexSignedIncidence (beta := beta) m U omega).coeff_nonzero_iff).mp h
  exact edgeVertexIncidences_target_vertices (beta := beta) (m := m) (U := U) hmem

theorem edgeVertexSignedIncidence_source_edges (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (edgeVertexSignedIncidence (beta := beta) m U omega).coeff face edge ≠ 0) :
    edge ∈ edgeSimplices m U := by
  have hmem :
      (face, edge) ∈ (edgeVertexSignedIncidence (beta := beta) m U omega).support :=
    ((edgeVertexSignedIncidence (beta := beta) m U omega).coeff_nonzero_iff).mp h
  exact (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem |>.1

theorem triangleEdgeSignedIncidence_target_edges (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (triangleEdgeSignedIncidence (beta := beta) m U omega).coeff face triangle ≠ 0) :
    face ∈ edgeSimplices m U := by
  have hmem :
      (face, triangle) ∈ (triangleEdgeSignedIncidence (beta := beta) m U omega).support :=
    ((triangleEdgeSignedIncidence (beta := beta) m U omega).coeff_nonzero_iff).mp h
  exact triangleEdgeIncidences_target_edges (beta := beta) (m := m) (U := U) hmem

theorem triangleEdgeSignedIncidence_source_triangles (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (triangleEdgeSignedIncidence (beta := beta) m U omega).coeff face triangle ≠ 0) :
    triangle ∈ triangleSimplices m U := by
  have hmem :
      (face, triangle) ∈ (triangleEdgeSignedIncidence (beta := beta) m U omega).support :=
    ((triangleEdgeSignedIncidence (beta := beta) m U omega).coeff_nonzero_iff).mp h
  exact (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem |>.1

end SCT.FND1
