import SCTLean.FND1.BoundaryTriangleCases

/-!
# FND-1 Boundary Entry Cases

This module turns the new finite case-split lemmas into coefficient-level
vanishing statements for the edge-to-vertex boundary.

The purpose is narrow and honest: before proving a whole-nerve cancellation
theorem, we need exact lemmas saying when a row singleton cannot contribute to a
given boundary edge.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem singletonVertex_mem_edge_of_edgeVertexSupport
    (m : Nat) (U : Cover beta alpha)
    {v : beta} {face edge : Finset beta}
    (hface : face = ({v} : Finset beta))
    (hsupport : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U) :
    v ∈ edge := by
  have hcodim :
      face ∈ codimOneFaces (beta := beta) edge :=
    (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hsupport |>.2
  rw [mem_codimOneFaces_iff] at hcodim
  have hsub := hcodim.1
  have hvFace : v ∈ face := by
    simp [hface]
  exact hsub hvFace

theorem singletonVertex_not_mem_edge_of_no_support
    (m : Nat) (U : Cover beta alpha)
    {v : beta} {face edge : Finset beta}
    (hface : face = ({v} : Finset beta))
    (hnotmem : v ∉ edge) :
    (face, edge) ∉ edgeVertexIncidences (beta := beta) m U := by
  intro hsupport
  exact hnotmem (singletonVertex_mem_edge_of_edgeVertexSupport
    (beta := beta) (m := m) (U := U) hface hsupport)

theorem edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {v : beta} {face edge : Finset beta}
    (hface : face = ({v} : Finset beta))
    (hnotmem : v ∉ edge) :
    edgeVertexCoefficient (beta := beta) m U omega face edge = 0 := by
  apply edgeVertexCoefficient_eq_zero_of_not_mem
  exact singletonVertex_not_mem_edge_of_no_support
    (beta := beta) (m := m) (U := U) hface hnotmem

theorem TriangleOrderedWitness.edgeBC_entry_zero_at_rowA
    (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face = ({w.a} : Finset beta)) :
    edgeVertexCoefficient (beta := beta) m U omega
      face (w.edgeBC (beta := beta) m U triangle).1 = 0 := by
  have hnotmem : w.a ∉ (w.edgeBC (beta := beta) m U triangle).1 := by
    rw [TriangleOrderedWitness.edgeBC_val]
    intro hmem
    simp at hmem
    rcases hmem with hab | hac
    · exact w.hab hab
    · exact w.hac hac
  apply edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (beta := beta) (m := m) (U := U) (omega := omega) hface hnotmem

theorem TriangleOrderedWitness.edgeAC_entry_zero_at_rowB
    (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face = ({w.b} : Finset beta)) :
    edgeVertexCoefficient (beta := beta) m U omega
      face (w.edgeAC (beta := beta) m U triangle).1 = 0 := by
  have hnotmem : w.b ∉ (w.edgeAC (beta := beta) m U triangle).1 := by
    rw [TriangleOrderedWitness.edgeAC_val]
    intro hmem
    simp at hmem
    rcases hmem with hba | hbc
    · exact w.hab hba.symm
    · exact w.hbc hbc
  apply edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (beta := beta) (m := m) (U := U) (omega := omega) hface hnotmem

theorem TriangleOrderedWitness.edgeAB_entry_zero_at_rowC
    (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face = ({w.c} : Finset beta)) :
    edgeVertexCoefficient (beta := beta) m U omega
      face (w.edgeAB (beta := beta) m U triangle).1 = 0 := by
  have hnotmem : w.c ∉ (w.edgeAB (beta := beta) m U triangle).1 := by
    rw [TriangleOrderedWitness.edgeAB_val]
    intro hmem
    simp at hmem
    rcases hmem with hca | hcb
    · exact w.hac hca.symm
    · exact w.hbc hcb.symm
  apply edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge
    (beta := beta) (m := m) (U := U) (omega := omega) hface hnotmem

end SCT.FND1
