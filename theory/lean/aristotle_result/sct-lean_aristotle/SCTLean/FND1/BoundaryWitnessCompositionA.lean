import SCTLean.FND1.BoundaryGlobalChoiceCompatibility
import SCTLean.FND1.BoundaryWitnessRowCoefficients

/-!
# FND-1 Boundary Witness Composition A

This module performs the first honest reduction of a global composition entry
to witness-level arithmetic.

It handles the row attached to vertex `a` of a chosen witness triangle. The
result is still deliberately local: it proves the `A`-row entry of
`∂₁ ∘ ∂₂` vanishes under the current global choice-compatibility hypothesis.
-/

namespace SCT.FND1

open scoped BigOperators

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- The contribution of a single intermediate edge to the `A`-row witness
composition entry. -/
def witnessRowAContribution
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (j : EdgeBoundaryIndex (beta := beta) m U) : Int :=
  edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      ((tau triangle).rowA (beta := beta) m U chi tau triangle).1
      j.1 *
    triangleEdgeCoefficient (beta := beta) m U
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      j.1 triangle.1

theorem witnessRowAContribution_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowAContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeAB (beta := beta) m U triangle) = -1 := by
  unfold witnessRowAContribution
  rw [choiceInduced_edgeVertexCoeff_rowA_edgeAB
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)]
  rw [choiceInduced_triangleEdgeCoeff_AB
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle) (hw := rfl)]
  norm_num

theorem witnessRowAContribution_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowAContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeAC (beta := beta) m U triangle) = 1 := by
  unfold witnessRowAContribution
  rw [choiceInduced_edgeVertexCoeff_rowA_edgeAC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)]
  rw [choiceInduced_triangleEdgeCoeff_AC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle) (hw := rfl)]
  norm_num

theorem witnessRowAContribution_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowAContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeBC (beta := beta) m U triangle) = 0 := by
  unfold witnessRowAContribution
  rw [choiceInduced_edgeVertexCoeff_rowA_edgeBC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)]
  norm_num

theorem witnessRowAContribution_eq_zero_of_not_witness_boundary_edge
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    {j : EdgeBoundaryIndex (beta := beta) m U}
    (hneAB : j ≠ (tau triangle).edgeAB (beta := beta) m U triangle)
    (hneAC : j ≠ (tau triangle).edgeAC (beta := beta) m U triangle)
    (hneBC : j ≠ (tau triangle).edgeBC (beta := beta) m U triangle) :
    witnessRowAContribution (beta := beta) m U chi tau triangle j = 0 := by
  unfold witnessRowAContribution
  have htri :
      triangleEdgeCoefficient (beta := beta) m U
        (choiceInducedTriangleDatum (beta := beta) m U chi tau)
        j.1 triangle.1 = 0 := by
    exact triangleEdgeCoefficient_eq_zero_of_not_witness_boundary_edge
      (beta := beta) (m := m) (U := U)
      (omega := choiceInducedTriangleDatum (beta := beta) m U chi tau)
      (triangle := triangle) (w := tau triangle) (edge := j)
      hneAB hneAC hneBC
  rw [htri]
  simp

theorem witnessRowA_compositionEntry_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowA (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  classical
  let T := EdgeBoundaryIndex (beta := beta) m U
  have hab_mem :
      (tau triangle).edgeAB (beta := beta) m U triangle ∈
        (Finset.univ : Finset T) := by
    simp [T]
  have hac_mem :
      (tau triangle).edgeAC (beta := beta) m U triangle ∈
        ((Finset.univ : Finset T).erase
          ((tau triangle).edgeAB (beta := beta) m U triangle)) := by
    have hne :
        (tau triangle).edgeAC (beta := beta) m U triangle ≠
          (tau triangle).edgeAB (beta := beta) m U triangle := by
      intro h
      exact (TriangleOrderedWitness.edgeAB_ne_edgeAC (beta := beta)
        (m := m) (U := U) (triangle := triangle) (w := tau triangle)) h.symm
    simp [T, hne]
  have hbc_mem :
      (tau triangle).edgeBC (beta := beta) m U triangle ∈
        (((Finset.univ : Finset T).erase
          ((tau triangle).edgeAB (beta := beta) m U triangle)).erase
            ((tau triangle).edgeAC (beta := beta) m U triangle)) := by
    have hneAB :
        (tau triangle).edgeBC (beta := beta) m U triangle ≠
          (tau triangle).edgeAB (beta := beta) m U triangle := by
      intro h
      exact (TriangleOrderedWitness.edgeAB_ne_edgeBC (beta := beta)
        (m := m) (U := U) (triangle := triangle) (w := tau triangle)) h.symm
    have hneAC :
        (tau triangle).edgeBC (beta := beta) m U triangle ≠
          (tau triangle).edgeAC (beta := beta) m U triangle := by
      intro h
      exact (TriangleOrderedWitness.edgeAC_ne_edgeBC (beta := beta)
        (m := m) (U := U) (triangle := triangle) (w := tau triangle)) h.symm
    simp [T, hneAB, hneAC]
  unfold edgeTriangleBoundaryCompositionEntry
  change
    ((Finset.univ : Finset T).sum
      (fun j => witnessRowAContribution (beta := beta) m U chi tau triangle j)) = 0
  rw [show (Finset.univ : Finset T) =
      insert ((tau triangle).edgeAB (beta := beta) m U triangle)
        ((Finset.univ : Finset T).erase
          ((tau triangle).edgeAB (beta := beta) m U triangle)) by
      exact (Finset.insert_erase hab_mem).symm]
  rw [Finset.sum_insert]
  · rw [show ((Finset.univ : Finset T).erase
          ((tau triangle).edgeAB (beta := beta) m U triangle)) =
        insert ((tau triangle).edgeAC (beta := beta) m U triangle)
          (((Finset.univ : Finset T).erase
            ((tau triangle).edgeAB (beta := beta) m U triangle)).erase
              ((tau triangle).edgeAC (beta := beta) m U triangle)) by
        exact (Finset.insert_erase hac_mem).symm]
    rw [Finset.sum_insert]
    · rw [show (((Finset.univ : Finset T).erase
            ((tau triangle).edgeAB (beta := beta) m U triangle)).erase
              ((tau triangle).edgeAC (beta := beta) m U triangle)) =
          insert ((tau triangle).edgeBC (beta := beta) m U triangle)
            ((((Finset.univ : Finset T).erase
              ((tau triangle).edgeAB (beta := beta) m U triangle)).erase
                ((tau triangle).edgeAC (beta := beta) m U triangle)).erase
                  ((tau triangle).edgeBC (beta := beta) m U triangle)) by
          exact (Finset.insert_erase hbc_mem).symm]
      rw [Finset.sum_insert]
      · have hrest :
            ((((Finset.univ : Finset T).erase
                ((tau triangle).edgeAB (beta := beta) m U triangle)).erase
                  ((tau triangle).edgeAC (beta := beta) m U triangle)).erase
                    ((tau triangle).edgeBC (beta := beta) m U triangle)).sum
              (fun j => witnessRowAContribution (beta := beta) m U chi tau triangle j) = 0 := by
          refine Finset.sum_eq_zero ?_
          intro j hj
          have hneBC :
              j ≠ (tau triangle).edgeBC (beta := beta) m U triangle := by
            exact (Finset.mem_erase.mp hj).1
          have hj' :
              j ∈ (((Finset.univ : Finset T).erase
                ((tau triangle).edgeAB (beta := beta) m U triangle)).erase
                  ((tau triangle).edgeAC (beta := beta) m U triangle)) := by
            exact (Finset.mem_erase.mp hj).2
          have hneAC :
              j ≠ (tau triangle).edgeAC (beta := beta) m U triangle := by
            exact (Finset.mem_erase.mp hj').1
          have hj'' :
              j ∈ ((Finset.univ : Finset T).erase
                ((tau triangle).edgeAB (beta := beta) m U triangle)) := by
            exact (Finset.mem_erase.mp hj').2
          have hneAB :
              j ≠ (tau triangle).edgeAB (beta := beta) m U triangle := by
            exact (Finset.mem_erase.mp hj'').1
          exact witnessRowAContribution_eq_zero_of_not_witness_boundary_edge
            (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
            (triangle := triangle) (j := j) hneAB hneAC hneBC
        rw [hrest]
        rw [witnessRowAContribution_edgeAB
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)]
        rw [witnessRowAContribution_edgeAC
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)]
        rw [witnessRowAContribution_edgeBC
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (triangle := triangle)]
        norm_num
      · simp
    · have hne :
          (tau triangle).edgeAC (beta := beta) m U triangle ≠
            (tau triangle).edgeAB (beta := beta) m U triangle := by
        intro h
        exact (TriangleOrderedWitness.edgeAB_ne_edgeAC (beta := beta)
          (m := m) (U := U) (triangle := triangle) (w := tau triangle)) h.symm
      simp [T, hne]
  · simp

theorem witnessRowA_composition_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowA (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  exact witnessRowA_compositionEntry_eq_zero
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (hglobal := hglobal) (triangle := triangle)

end SCT.FND1
