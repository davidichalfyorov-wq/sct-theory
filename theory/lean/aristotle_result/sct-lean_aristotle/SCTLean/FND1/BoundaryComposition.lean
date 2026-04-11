import Mathlib.Data.Matrix.Basic
import SCTLean.FND1.BoundaryMatrix

/-!
# FND-1 Boundary Composition

This module introduces the first genuine chain-level composition object:

- the edge-to-vertex boundary matrix `∂₁`,
- the triangle-to-edge boundary matrix `∂₂`,
- their composed entry formula `∂₁ ∘ ∂₂`.

At this stage we do **not** claim the composition is zero. Instead we prove the
honest support statement: any nonzero entry in the composition must pass through
an actual intermediate edge carrying nonzero contributions on both sides.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Vertex index type for the edge-to-vertex boundary table. -/
abbrev EdgeVertexRowIndex (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :=
  (edgeVertexBoundaryTable (beta := beta) m U omega₁).RowIndex

/-- Edge index type shared by the adjacent boundary levels. -/
abbrev EdgeBoundaryIndex (m : Nat) (U : Cover beta alpha) := ↑(edgeSimplices m U)

/-- Triangle index type for the triangle-to-edge boundary table. -/
abbrev TriangleEdgeColIndex (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :=
  (triangleEdgeBoundaryTable (beta := beta) m U omega₂).ColIndex

/-- Entrywise definition of the composed boundary `∂₁ ∘ ∂₂`. -/
def edgeTriangleBoundaryCompositionEntry (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (i : EdgeVertexRowIndex (beta := beta) m U omega₁)
    (k : TriangleEdgeColIndex (beta := beta) m U omega₂) : Int :=
  ∑ j : EdgeBoundaryIndex (beta := beta) m U,
    (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 *
      (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1

/-- Matrix whose entries are given by the explicit composition formula. -/
def edgeTriangleBoundaryComposition (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    Matrix
      (EdgeVertexRowIndex (beta := beta) m U omega₁)
      (TriangleEdgeColIndex (beta := beta) m U omega₂)
      Int :=
  fun i k => edgeTriangleBoundaryCompositionEntry (beta := beta) m U omega₁ omega₂ i k

theorem edgeTriangleBoundaryComposition_apply (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (i : EdgeVertexRowIndex (beta := beta) m U omega₁)
    (k : TriangleEdgeColIndex (beta := beta) m U omega₂) :
    edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ i k =
      ∑ j : EdgeBoundaryIndex (beta := beta) m U,
        (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 *
          (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1 := by
  rfl

theorem edgeTriangleBoundaryComposition_entry_nonzero_has_witness
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {i : EdgeVertexRowIndex (beta := beta) m U omega₁}
    {k : TriangleEdgeColIndex (beta := beta) m U omega₂}
    (h : edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ i k ≠ 0) :
    ∃ j : EdgeBoundaryIndex (beta := beta) m U,
      (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 ≠ 0 ∧
      (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1 ≠ 0 := by
  by_contra hw
  rw [edgeTriangleBoundaryComposition_apply (beta := beta) (m := m) (U := U)
    (omega₁ := omega₁) (omega₂ := omega₂) (i := i) (k := k)] at h
  have hzeroTerm :
      ∀ j : EdgeBoundaryIndex (beta := beta) m U,
        (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 *
          (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1 = 0 := by
    intro j
    by_cases hleft : (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 = 0
    · rw [hleft]
      simp
    · have hright :
          (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1 = 0 := by
        by_contra hrightNe
        exact hw ⟨j, hleft, hrightNe⟩
      rw [hright]
      simp
  have hsumZero :
      ∑ j : EdgeBoundaryIndex (beta := beta) m U,
        (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 *
          (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1 = 0 := by
    refine Finset.sum_eq_zero ?_
    intro j hj
    exact hzeroTerm j
  exact h hsumZero

theorem edgeTriangleBoundaryComposition_entry_nonzero_has_support
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {i : EdgeVertexRowIndex (beta := beta) m U omega₁}
    {k : TriangleEdgeColIndex (beta := beta) m U omega₂}
    (h : edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ i k ≠ 0) :
    ∃ j : EdgeBoundaryIndex (beta := beta) m U,
      (i.1, j.1) ∈ edgeVertexIncidences (beta := beta) m U ∧
      (j.1, k.1) ∈ triangleEdgeIncidences (beta := beta) m U := by
  rcases edgeTriangleBoundaryComposition_entry_nonzero_has_witness
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) h with
    ⟨j, hj₁, hj₂⟩
  refine ⟨j, ?_, ?_⟩
  · exact (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry_nonzero_iff_support.mp hj₁
  · exact (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry_nonzero_iff_support.mp hj₂

end SCT.FND1
