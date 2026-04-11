import SCTLean.FND1.BoundaryWitnessCompositionA

/-!
# FND-1 Boundary Witness Composition B/C

This module extends the first witness-entry reduction from `rowA` to the two
remaining singleton witness rows `rowB` and `rowC`.

The purpose is still narrow and honest: obtain the other two entrywise
vanishing theorems needed before a whole-nerve matrix theorem can be attempted.
-/

namespace SCT.FND1

open scoped BigOperators

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

def witnessRowBContribution
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (j : EdgeBoundaryIndex (beta := beta) m U) : Int :=
  edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      ((tau triangle).rowB (beta := beta) m U chi tau triangle).1
      j.1 *
    triangleEdgeCoefficient (beta := beta) m U
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      j.1 triangle.1

theorem witnessRowBContribution_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowBContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeAB (beta := beta) m U triangle) = 1 := by
  unfold witnessRowBContribution
  rw [choiceInduced_edgeVertexCoeff_rowB_edgeAB
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)]
  rw [choiceInduced_triangleEdgeCoeff_AB
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle) (hw := rfl)]
  norm_num

theorem witnessRowBContribution_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowBContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeAC (beta := beta) m U triangle) = 0 := by
  unfold witnessRowBContribution
  rw [choiceInduced_edgeVertexCoeff_rowB_edgeAC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)]
  norm_num

theorem witnessRowBContribution_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowBContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeBC (beta := beta) m U triangle) = -1 := by
  unfold witnessRowBContribution
  rw [choiceInduced_edgeVertexCoeff_rowB_edgeBC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)]
  rw [choiceInduced_triangleEdgeCoeff_BC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle) (hw := rfl)]
  norm_num

theorem witnessRowBContribution_eq_zero_of_not_witness_boundary_edge
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    {j : EdgeBoundaryIndex (beta := beta) m U}
    (hneAB : j ≠ (tau triangle).edgeAB (beta := beta) m U triangle)
    (hneAC : j ≠ (tau triangle).edgeAC (beta := beta) m U triangle)
    (hneBC : j ≠ (tau triangle).edgeBC (beta := beta) m U triangle) :
    witnessRowBContribution (beta := beta) m U chi tau triangle j = 0 := by
  unfold witnessRowBContribution
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

theorem witnessRowB_compositionEntry_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowB (beta := beta) m U chi tau triangle)
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
      (fun j => witnessRowBContribution (beta := beta) m U chi tau triangle j)) = 0
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
              (fun j => witnessRowBContribution (beta := beta) m U chi tau triangle j) = 0 := by
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
          exact witnessRowBContribution_eq_zero_of_not_witness_boundary_edge
            (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
            (triangle := triangle) (j := j) hneAB hneAC hneBC
        rw [hrest]
        rw [witnessRowBContribution_edgeAB
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)]
        rw [witnessRowBContribution_edgeAC
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (triangle := triangle)]
        rw [witnessRowBContribution_edgeBC
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)]
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

theorem witnessRowB_composition_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowB (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  exact witnessRowB_compositionEntry_eq_zero
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (hglobal := hglobal) (triangle := triangle)

def witnessRowCContribution
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (j : EdgeBoundaryIndex (beta := beta) m U) : Int :=
  edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      ((tau triangle).rowC (beta := beta) m U chi tau triangle).1
      j.1 *
    triangleEdgeCoefficient (beta := beta) m U
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      j.1 triangle.1

theorem witnessRowCContribution_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowCContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeAB (beta := beta) m U triangle) = 0 := by
  unfold witnessRowCContribution
  rw [choiceInduced_edgeVertexCoeff_rowC_edgeAB
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)]
  norm_num

theorem witnessRowCContribution_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowCContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeAC (beta := beta) m U triangle) = -1 := by
  unfold witnessRowCContribution
  rw [choiceInduced_edgeVertexCoeff_rowC_edgeAC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)]
  rw [choiceInduced_triangleEdgeCoeff_AC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle) (hw := rfl)]
  norm_num

theorem witnessRowCContribution_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    witnessRowCContribution (beta := beta) m U chi tau triangle
      ((tau triangle).edgeBC (beta := beta) m U triangle) = 1 := by
  unfold witnessRowCContribution
  rw [choiceInduced_edgeVertexCoeff_rowC_edgeBC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle)
    (hcompat := hglobal.compatible triangle)]
  rw [choiceInduced_triangleEdgeCoeff_BC
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (triangle := triangle) (w := tau triangle) (hw := rfl)]
  norm_num

theorem witnessRowCContribution_eq_zero_of_not_witness_boundary_edge
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    {j : EdgeBoundaryIndex (beta := beta) m U}
    (hneAB : j ≠ (tau triangle).edgeAB (beta := beta) m U triangle)
    (hneAC : j ≠ (tau triangle).edgeAC (beta := beta) m U triangle)
    (hneBC : j ≠ (tau triangle).edgeBC (beta := beta) m U triangle) :
    witnessRowCContribution (beta := beta) m U chi tau triangle j = 0 := by
  unfold witnessRowCContribution
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

theorem witnessRowC_compositionEntry_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryCompositionEntry (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowC (beta := beta) m U chi tau triangle)
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
      (fun j => witnessRowCContribution (beta := beta) m U chi tau triangle j)) = 0
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
              (fun j => witnessRowCContribution (beta := beta) m U chi tau triangle j) = 0 := by
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
          exact witnessRowCContribution_eq_zero_of_not_witness_boundary_edge
            (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
            (triangle := triangle) (j := j) hneAB hneAC hneBC
        rw [hrest]
        rw [witnessRowCContribution_edgeAB
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (triangle := triangle)]
        rw [witnessRowCContribution_edgeAC
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)]
        rw [witnessRowCContribution_edgeBC
          (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
          (hglobal := hglobal) (triangle := triangle)]
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

theorem witnessRowC_composition_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau)
    (triangle : ↑(triangleSimplices m U)) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (choiceInducedTriangleDatum (beta := beta) m U chi tau)
      ((tau triangle).rowC (beta := beta) m U chi tau triangle)
      (witnessTriangleCol (beta := beta) (m := m) (U := U)
        (chi := chi) (tau := tau) triangle) = 0 := by
  exact witnessRowC_compositionEntry_eq_zero
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)
    (hglobal := hglobal) (triangle := triangle)

end SCT.FND1
