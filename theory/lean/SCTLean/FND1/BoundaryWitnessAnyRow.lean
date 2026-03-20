import SCTLean.FND1.BoundaryWitnessTriangleComposition

/-!
# FND-1 Boundary Witness Any Row

This module removes the last row-specific slop from the current witness-triangle
layer.

It proves that for a fixed witness triangle column, the composed boundary entry
vanishes for **any** row index, not just the three named witness rows `A/B/C`.
The proof is still local to a single witness triangle.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (i : EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau))
    {v : beta}
    (hv : i.1 = ({v} : Finset beta))
    (hnotmem : v ∉ triangle.1) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      i.1 ((tau triangle).edgeAB (beta := beta) m U triangle).1 = 0 := by
  have hnotmemAB : v ∉ ((tau triangle).edgeAB (beta := beta) m U triangle).1 := by
    rw [TriangleOrderedWitness.edgeAB_val]
    intro hmem
    have htri : v ∈ triangle.1 := by
      rw [(tau triangle).triangle_eq]
      simp at hmem ⊢
      rcases hmem with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
    exact hnotmem htri
  exact edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau)
    (hface := hv) hnotmemAB

theorem edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (i : EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau))
    {v : beta}
    (hv : i.1 = ({v} : Finset beta))
    (hnotmem : v ∉ triangle.1) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      i.1 ((tau triangle).edgeAC (beta := beta) m U triangle).1 = 0 := by
  have hnotmemAC : v ∉ ((tau triangle).edgeAC (beta := beta) m U triangle).1 := by
    rw [TriangleOrderedWitness.edgeAC_val]
    intro hmem
    have htri : v ∈ triangle.1 := by
      rw [(tau triangle).triangle_eq]
      simp at hmem ⊢
      rcases hmem with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inr h)
    exact hnotmem htri
  exact edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau)
    (hface := hv) hnotmemAC

theorem edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (i : EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau))
    {v : beta}
    (hv : i.1 = ({v} : Finset beta))
    (hnotmem : v ∉ triangle.1) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      i.1 ((tau triangle).edgeBC (beta := beta) m U triangle).1 = 0 := by
  have hnotmemBC : v ∉ ((tau triangle).edgeBC (beta := beta) m U triangle).1 := by
    rw [TriangleOrderedWitness.edgeBC_val]
    intro hmem
    have htri : v ∈ triangle.1 := by
      rw [(tau triangle).triangle_eq]
      simp at hmem ⊢
      rcases hmem with h | h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr h)
    exact hnotmem htri
  exact edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau)
    (hface := hv) hnotmemBC

theorem witnessTriangleCompositionEntry_eq_zero_of_row_vertex_not_mem_triangle
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (i : EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau))
    {v : beta}
    (hv : i.1 = ({v} : Finset beta))
    (hnotmem : v ∉ triangle.1) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      i
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  by_contra hne
  rcases edgeTriangleBoundaryComposition_entry_nonzero_has_witness
      (beta := beta) (m := m) (U := U)
      (omega₁ := choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (omega₂ := choiceInducedTriangleDatum (beta := beta) m U chi tau)
      (i := i)
      (k := witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) hne with ⟨j, hleft, hright⟩
  have htriInc :
      (j.1, triangle.1) ∈ triangleEdgeIncidences (beta := beta) m U := by
    exact (triangleEdgeBoundaryTable (beta := beta) m U
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)).entry_nonzero_iff_support.mp hright
  have hjFace :
      j.1 ∈ codimOneFaces (beta := beta) triangle.1 := by
    exact (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp htriInc |>.2
  rcases TriangleOrderedWitness.boundaryEdge_cases
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := tau triangle)
      (edge := j) hjFace with hAB | hAC | hBC
  · have hzero :
        edgeVertexCoefficient (beta := beta) m U
          (choiceInducedEdgeDatum (beta := beta) m U chi tau)
          i.1 j.1 = 0 := by
      rw [hAB]
      exact edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeAB
        (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
        (triangle := triangle) (i := i) hv hnotmem
    exact hleft (by
      simpa [BoundaryTableData.entry, edgeVertexBoundaryTable, edgeVertexSignedIncidence] using hzero)
  · have hzero :
        edgeVertexCoefficient (beta := beta) m U
          (choiceInducedEdgeDatum (beta := beta) m U chi tau)
          i.1 j.1 = 0 := by
      rw [hAC]
      exact edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeAC
        (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
        (triangle := triangle) (i := i) hv hnotmem
    exact hleft (by
      simpa [BoundaryTableData.entry, edgeVertexBoundaryTable, edgeVertexSignedIncidence] using hzero)
  · have hzero :
        edgeVertexCoefficient (beta := beta) m U
          (choiceInducedEdgeDatum (beta := beta) m U chi tau)
          i.1 j.1 = 0 := by
      rw [hBC]
      exact edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeBC
        (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
        (triangle := triangle) (i := i) hv hnotmem
    exact hleft (by
      simpa [BoundaryTableData.entry, edgeVertexBoundaryTable, edgeVertexSignedIncidence] using hzero)

theorem anyRow_witnessTriangleCompositionEntry_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U))
    (i : EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      i
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  rcases edgeVertexRowIndex_eq_singleton
      (beta := beta) (m := m) (U := U)
      (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) i with ⟨v, hv⟩
  by_cases hmem : v ∈ triangle.1
  · have hcases : v = (tau triangle).a ∨ v = (tau triangle).b ∨ v = (tau triangle).c := by
      rw [(tau triangle).triangle_eq] at hmem
      simpa using hmem
    rcases hcases with hA | hBC
    · have hi : i = (tau triangle).rowA (beta := beta) m U chi tau triangle := by
        apply Subtype.ext
        simp [hv, hA, TriangleOrderedWitness.rowA_val]
      rw [hi]
      exact witnessRowA_compositionEntry_eq_zero
        (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
        (hglobal := hglobal) (triangle := triangle)
    · rcases hBC with hB | hC
      · have hi : i = (tau triangle).rowB (beta := beta) m U chi tau triangle := by
          apply Subtype.ext
          simp [hv, hB, TriangleOrderedWitness.rowB_val]
        rw [hi]
        exact witnessRowB_compositionEntry_eq_zero
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)
      · have hi : i = (tau triangle).rowC (beta := beta) m U chi tau triangle := by
          apply Subtype.ext
          simp [hv, hC, TriangleOrderedWitness.rowC_val]
        rw [hi]
        exact witnessRowC_compositionEntry_eq_zero
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)
  · exact witnessTriangleCompositionEntry_eq_zero_of_row_vertex_not_mem_triangle
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
      (triangle := triangle) (i := i) hv hmem

end SCT.FND1
