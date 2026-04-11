import SCTLean.FND1.BoundaryTriangleCoherence

/-!
# FND-1 Boundary Triangle Coherence Equivalence

The previous module showed that coherent triangle witnesses are sufficient to
reconstruct the current edge-choice compatibility layer. This module adds the
converse direction.

Result: for a fixed triangle-choice datum `tau`, the remaining noncanonical
input is exactly the coherence law on shared edges. It is neither weaker nor
stronger than the existence of some compatible edge-choice datum `chi`.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem compatible_choice_eq_triangle_face_val
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w)
    (edge : ↑(edgeSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1) :
    ((w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle edge hedge).1) =
      (chi edge).1 := by
  rcases TriangleOrderedWitness.boundaryEdge_cases
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w)
      (edge := edge) hedge with hAB | hAC | hBC
  · cases hAB
    rw [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAB_val
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w)]
    simp [hcompat.edgeAB_choice]
  · cases hAC
    rw [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAC_val
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w)]
    simp [hcompat.edgeAC_choice]
  · cases hBC
    rw [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeBC_val
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w)]
    simp [hcompat.edgeBC_choice]

theorem triangleChoiceCoherence_of_globalChoiceCompatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    GlobalTriangleChoiceCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge triangle₁ triangle₂ h₁ h₂
  have hleft :
      ((tau triangle₁).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle₁ edge h₁).1 =
        (chi edge).1 :=
    compatible_choice_eq_triangle_face_val
      (beta := beta) (m := m) (U := U) (chi := chi)
      (triangle := triangle₁) (w := tau triangle₁)
      (hcompat := hglobal.compatible triangle₁)
      (edge := edge) (hedge := h₁)
  have hright :
      ((tau triangle₂).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle₂ edge h₂).1 =
        (chi edge).1 :=
    compatible_choice_eq_triangle_face_val
      (beta := beta) (m := m) (U := U) (chi := chi)
      (triangle := triangle₂) (w := tau triangle₂)
      (hcompat := hglobal.compatible triangle₂)
      (edge := edge) (hedge := h₂)
  exact hleft.trans hright.symm

theorem globalChoiceCompatibility_exists_iff_triangleChoiceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    (∃ chi : EdgeEndpointChoiceDatum (beta := beta) m U,
      GlobalEdgeTriangleChoiceCompatibility (beta := beta) m U chi tau) ↔
      GlobalTriangleChoiceCoherence (beta := beta) m U tau := by
  constructor
  · rintro ⟨chi, hglobal⟩
    exact triangleChoiceCoherence_of_globalChoiceCompatible
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) hglobal
  · intro hcoh
    exact ⟨coherentEdgeEndpointChoiceDatum (beta := beta) m U tau,
      globalChoiceCompatibility_of_triangleChoiceCoherent
        (beta := beta) (m := m) (U := U) (tau := tau) hcoh⟩

end SCT.FND1
