import SCTLean.FND1.BoundaryComposition

/-!
# FND-1 Boundary Compatibility

This module isolates the exact additional condition still missing for a genuine
chain-complex statement. The condition is kept explicit and honest:

- support is already canonical,
- orientation data are auxiliary,
- compatibility is an additional cancellation law on top of that data.

From this law one can prove the first genuine chain-level consequence:
`∂₁ ∘ ∂₂ = 0`.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A single term contributing to the composed entry `(vertex, triangle)`. -/
def edgeTriangleCompositeTerm (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (i : EdgeVertexRowIndex (beta := beta) m U omega₁)
    (j : EdgeBoundaryIndex (beta := beta) m U)
    (k : TriangleEdgeColIndex (beta := beta) m U omega₂) : Int :=
  (edgeVertexBoundaryTable (beta := beta) m U omega₁).entry i.1 j.1 *
    (triangleEdgeBoundaryTable (beta := beta) m U omega₂).entry j.1 k.1

/-- Explicit compatibility law needed for `∂₁ ∘ ∂₂ = 0`. -/
structure BoundaryOrientationCompatibility
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) : Prop where
  cancel :
    ∀ i : EdgeVertexRowIndex (beta := beta) m U omega₁,
      ∀ k : TriangleEdgeColIndex (beta := beta) m U omega₂,
        ∑ j : EdgeBoundaryIndex (beta := beta) m U,
          edgeTriangleCompositeTerm (beta := beta) m U omega₁ omega₂ i j k = 0

theorem edgeTriangleBoundaryComposition_eq_zero_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂)
    (i : EdgeVertexRowIndex (beta := beta) m U omega₁)
    (k : TriangleEdgeColIndex (beta := beta) m U omega₂) :
    edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ i k = 0 := by
  rw [edgeTriangleBoundaryComposition_apply (beta := beta) (m := m) (U := U)
    (omega₁ := omega₁) (omega₂ := omega₂) (i := i) (k := k)]
  simpa [edgeTriangleCompositeTerm] using hcompat.cancel i k

theorem edgeTriangleBoundaryComposition_matrix_eq_zero_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂) :
    edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0 := by
  ext i k
  exact edgeTriangleBoundaryComposition_eq_zero_of_compatible
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat i k

theorem no_nonzero_composed_entry_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂)
    {i : EdgeVertexRowIndex (beta := beta) m U omega₁}
    {k : TriangleEdgeColIndex (beta := beta) m U omega₂} :
    edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ i k ≠ 0 -> False := by
  intro h
  have hzero := edgeTriangleBoundaryComposition_eq_zero_of_compatible
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat i k
  exact h hzero

end SCT.FND1
