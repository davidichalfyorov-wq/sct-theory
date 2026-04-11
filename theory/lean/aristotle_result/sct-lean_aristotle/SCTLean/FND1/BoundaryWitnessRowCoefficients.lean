import SCTLean.FND1.BoundaryWitnessIndices

/-!
# FND-1 Boundary Witness Row Coefficients

This module computes the exact edge-to-vertex coefficients of `∂₁` on the three
singleton rows attached to a fixed witness triangle.

This is the left-hand analogue of `BoundaryWitnessCoefficients`. Together, the
two files isolate the exact signed contributions that will appear in the future
reduction of a global composition entry to the local `A/B/C` cancellation sums.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem choiceInduced_edgeVertexCoeff_rowA_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowA (beta := beta) m U chi tau triangle).1
      (w.edgeAB (beta := beta) m U triangle).1 = -1 := by
  have hmem :
      ((w.rowA (beta := beta) m U chi tau triangle).1,
        (w.edgeAB (beta := beta) m U triangle).1) ∈
        edgeVertexIncidences (beta := beta) m U := by
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · exact (w.edgeAB (beta := beta) m U triangle).2
    · simpa [TriangleOrderedWitness.rowA_val, TriangleOrderedWitness.edgeAB_val] using
        (w.edgeAB_leftFace (beta := beta) m U triangle).2
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) hmem]
  unfold choiceInducedEdgeDatum
  rw [inducedEdgeVertexOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  have hface_eq :
      (⟨(w.rowA (beta := beta) m U chi tau triangle).1,
        ((mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2⟩ :
        LocalFaceSupport (beta := beta) (w.edgeAB (beta := beta) m U triangle).1) =
      w.edgeAB_leftFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simp [TriangleOrderedWitness.rowA_val, TriangleOrderedWitness.edgeAB_leftFace_val]
  rw [hface_eq]
  exact compatible_edgeAB_left_coeff (beta := beta) (m := m) (U := U)
    (chi := chi) (triangle := triangle) (w := w) hcompat

theorem choiceInduced_edgeVertexCoeff_rowA_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowA (beta := beta) m U chi tau triangle).1
      (w.edgeAC (beta := beta) m U triangle).1 = -1 := by
  have hmem :
      ((w.rowA (beta := beta) m U chi tau triangle).1,
        (w.edgeAC (beta := beta) m U triangle).1) ∈
        edgeVertexIncidences (beta := beta) m U := by
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · exact (w.edgeAC (beta := beta) m U triangle).2
    · simpa [TriangleOrderedWitness.rowA_val, TriangleOrderedWitness.edgeAC_val] using
        (w.edgeAC_leftFace (beta := beta) m U triangle).2
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) hmem]
  unfold choiceInducedEdgeDatum
  rw [inducedEdgeVertexOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  have hface_eq :
      (⟨(w.rowA (beta := beta) m U chi tau triangle).1,
        ((mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2⟩ :
        LocalFaceSupport (beta := beta) (w.edgeAC (beta := beta) m U triangle).1) =
      w.edgeAC_leftFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simp [TriangleOrderedWitness.rowA_val, TriangleOrderedWitness.edgeAC_leftFace_val]
  rw [hface_eq]
  exact compatible_edgeAC_left_coeff (beta := beta) (m := m) (U := U)
    (chi := chi) (triangle := triangle) (w := w) hcompat

theorem choiceInduced_edgeVertexCoeff_rowA_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowA (beta := beta) m U chi tau triangle).1
      (w.edgeBC (beta := beta) m U triangle).1 = 0 := by
  simpa [TriangleOrderedWitness.rowA_val] using
    TriangleOrderedWitness.edgeBC_entry_zero_at_rowA (beta := beta) (m := m) (U := U)
      (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (triangle := triangle) (w := w)
      (face := (w.rowA (beta := beta) m U chi tau triangle).1)
      (hface := by simp [TriangleOrderedWitness.rowA_val])

theorem choiceInduced_edgeVertexCoeff_rowB_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowB (beta := beta) m U chi tau triangle).1
      (w.edgeAB (beta := beta) m U triangle).1 = 1 := by
  have hmem :
      ((w.rowB (beta := beta) m U chi tau triangle).1,
        (w.edgeAB (beta := beta) m U triangle).1) ∈
        edgeVertexIncidences (beta := beta) m U := by
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · exact (w.edgeAB (beta := beta) m U triangle).2
    · simpa [TriangleOrderedWitness.rowB_val, TriangleOrderedWitness.edgeAB_val] using
        (w.edgeAB_rightFace (beta := beta) m U triangle).2
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) hmem]
  unfold choiceInducedEdgeDatum
  rw [inducedEdgeVertexOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  have hface_eq :
      (⟨(w.rowB (beta := beta) m U chi tau triangle).1,
        ((mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2⟩ :
        LocalFaceSupport (beta := beta) (w.edgeAB (beta := beta) m U triangle).1) =
      w.edgeAB_rightFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simp [TriangleOrderedWitness.rowB_val, TriangleOrderedWitness.edgeAB_rightFace_val]
  rw [hface_eq]
  exact compatible_edgeAB_right_coeff (beta := beta) (m := m) (U := U)
    (chi := chi) (triangle := triangle) (w := w) hcompat

theorem choiceInduced_edgeVertexCoeff_rowB_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowB (beta := beta) m U chi tau triangle).1
      (w.edgeAC (beta := beta) m U triangle).1 = 0 := by
  simpa [TriangleOrderedWitness.rowB_val] using
    TriangleOrderedWitness.edgeAC_entry_zero_at_rowB (beta := beta) (m := m) (U := U)
      (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (triangle := triangle) (w := w)
      (face := (w.rowB (beta := beta) m U chi tau triangle).1)
      (hface := by simp [TriangleOrderedWitness.rowB_val])

theorem choiceInduced_edgeVertexCoeff_rowB_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowB (beta := beta) m U chi tau triangle).1
      (w.edgeBC (beta := beta) m U triangle).1 = -1 := by
  have hmem :
      ((w.rowB (beta := beta) m U chi tau triangle).1,
        (w.edgeBC (beta := beta) m U triangle).1) ∈
        edgeVertexIncidences (beta := beta) m U := by
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · exact (w.edgeBC (beta := beta) m U triangle).2
    · simpa [TriangleOrderedWitness.rowB_val, TriangleOrderedWitness.edgeBC_val] using
        (w.edgeBC_leftFace (beta := beta) m U triangle).2
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) hmem]
  unfold choiceInducedEdgeDatum
  rw [inducedEdgeVertexOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  have hface_eq :
      (⟨(w.rowB (beta := beta) m U chi tau triangle).1,
        ((mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2⟩ :
        LocalFaceSupport (beta := beta) (w.edgeBC (beta := beta) m U triangle).1) =
      w.edgeBC_leftFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simp [TriangleOrderedWitness.rowB_val, TriangleOrderedWitness.edgeBC_leftFace_val]
  rw [hface_eq]
  exact compatible_edgeBC_left_coeff (beta := beta) (m := m) (U := U)
    (chi := chi) (triangle := triangle) (w := w) hcompat

theorem choiceInduced_edgeVertexCoeff_rowC_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowC (beta := beta) m U chi tau triangle).1
      (w.edgeAB (beta := beta) m U triangle).1 = 0 := by
  simpa [TriangleOrderedWitness.rowC_val] using
    TriangleOrderedWitness.edgeAB_entry_zero_at_rowC (beta := beta) (m := m) (U := U)
      (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (triangle := triangle) (w := w)
      (face := (w.rowC (beta := beta) m U chi tau triangle).1)
      (hface := by simp [TriangleOrderedWitness.rowC_val])

theorem choiceInduced_edgeVertexCoeff_rowC_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowC (beta := beta) m U chi tau triangle).1
      (w.edgeAC (beta := beta) m U triangle).1 = 1 := by
  have hmem :
      ((w.rowC (beta := beta) m U chi tau triangle).1,
        (w.edgeAC (beta := beta) m U triangle).1) ∈
        edgeVertexIncidences (beta := beta) m U := by
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · exact (w.edgeAC (beta := beta) m U triangle).2
    · simpa [TriangleOrderedWitness.rowC_val, TriangleOrderedWitness.edgeAC_val] using
        (w.edgeAC_rightFace (beta := beta) m U triangle).2
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) hmem]
  unfold choiceInducedEdgeDatum
  rw [inducedEdgeVertexOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  have hface_eq :
      (⟨(w.rowC (beta := beta) m U chi tau triangle).1,
        ((mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2⟩ :
        LocalFaceSupport (beta := beta) (w.edgeAC (beta := beta) m U triangle).1) =
      w.edgeAC_rightFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simp [TriangleOrderedWitness.rowC_val, TriangleOrderedWitness.edgeAC_rightFace_val]
  rw [hface_eq]
  exact compatible_edgeAC_right_coeff (beta := beta) (m := m) (U := U)
    (chi := chi) (triangle := triangle) (w := w) hcompat

theorem choiceInduced_edgeVertexCoeff_rowC_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    edgeVertexCoefficient (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau)
      (w.rowC (beta := beta) m U chi tau triangle).1
      (w.edgeBC (beta := beta) m U triangle).1 = 1 := by
  have hmem :
      ((w.rowC (beta := beta) m U chi tau triangle).1,
        (w.edgeBC (beta := beta) m U triangle).1) ∈
        edgeVertexIncidences (beta := beta) m U := by
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · exact (w.edgeBC (beta := beta) m U triangle).2
    · simpa [TriangleOrderedWitness.rowC_val, TriangleOrderedWitness.edgeBC_val] using
        (w.edgeBC_rightFace (beta := beta) m U triangle).2
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := choiceInducedEdgeDatum (beta := beta) m U chi tau) hmem]
  unfold choiceInducedEdgeDatum
  rw [inducedEdgeVertexOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_edgeFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  have hface_eq :
      (⟨(w.rowC (beta := beta) m U chi tau triangle).1,
        ((mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2⟩ :
        LocalFaceSupport (beta := beta) (w.edgeBC (beta := beta) m U triangle).1) =
      w.edgeBC_rightFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simp [TriangleOrderedWitness.rowC_val, TriangleOrderedWitness.edgeBC_rightFace_val]
  rw [hface_eq]
  exact compatible_edgeBC_right_coeff (beta := beta) (m := m) (U := U)
    (chi := chi) (triangle := triangle) (w := w) hcompat

end SCT.FND1
