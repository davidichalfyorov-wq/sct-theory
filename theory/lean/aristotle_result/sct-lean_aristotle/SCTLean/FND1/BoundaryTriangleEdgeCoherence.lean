import SCTLean.FND1.BoundaryTriangleCoherenceEquiv

/-!
# FND-1 Boundary Triangle Edge Coherence

This module rewrites the remaining triangle-witness assumption in a more
operator-facing language.

Instead of talking about which singleton face a triangle witness chooses on a
shared edge, we can talk directly about the induced local edge-orientation rule
on that edge. The two viewpoints are equivalent.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

omit [Fintype beta] in
theorem localOrientationFromChosenFace_injective {simplex : Finset beta} :
    Function.Injective (fun chosen : LocalFaceSupport (beta := beta) simplex =>
      localOrientationFromChosenFace (beta := beta) chosen) := by
  intro chosen₁ chosen₂ hfun
  by_contra hne
  have hfalse :
      localOrientationFromChosenFace (beta := beta) chosen₂ chosen₁ = false := by
    simpa using congrArg (fun f => f chosen₁) hfun
  have hother :
      localOrientationFromChosenFace (beta := beta) chosen₂ chosen₁ = true := by
    exact localOrientationFromChosenFace_at_other (beta := beta) chosen₂ chosen₁ hne
  exact Bool.false_ne_true (hfalse.symm.trans hother)

/-- The local edge-orientation rule induced on a shared boundary edge by one
triangle witness. -/
def TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (edge : ↑(edgeSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1) :
    LocalFaceOrientationDatum (beta := beta) edge.1 :=
  localOrientationFromChosenFace (beta := beta)
    (w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle edge hedge)

/-- Local coherence on shared edges, phrased directly in terms of the induced
edge-orientation rules coming from triangle witnesses. -/
structure GlobalTriangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop where
  coherent :
    ∀ (edge : ↑(edgeSimplices m U))
      {triangle₁ triangle₂ : ↑(triangleSimplices m U)}
      (h₁ : edge.1 ∈ codimOneFaces (beta := beta) triangle₁.1)
      (h₂ : edge.1 ∈ codimOneFaces (beta := beta) triangle₂.1),
      (tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ edge h₁ =
        (tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂

theorem triangleEdgeOrientationCoherence_of_triangleChoiceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleChoiceCoherence (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge triangle₁ triangle₂ h₁ h₂
  have hchoiceVal :
      ((tau triangle₁).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle₁ edge h₁).1 =
        ((tau triangle₂).chosenFaceOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂).1 :=
    hcoh.coherent edge h₁ h₂
  have hchoice :
      (tau triangle₁).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle₁ edge h₁ =
        (tau triangle₂).chosenFaceOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂ := by
    apply Subtype.ext
    simpa using hchoiceVal
  exact congrArg
    (fun chosen =>
      localOrientationFromChosenFace (beta := beta) chosen)
    hchoice

theorem triangleChoiceCoherence_of_triangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    GlobalTriangleChoiceCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge triangle₁ triangle₂ h₁ h₂
  have horient :
      (tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ edge h₁ =
        (tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂ :=
    hcoh.coherent edge h₁ h₂
  have hchoice :
      (tau triangle₁).chosenFaceOnBoundaryEdge
        (beta := beta) m U triangle₁ edge h₁ =
        (tau triangle₂).chosenFaceOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂ :=
    localOrientationFromChosenFace_injective (beta := beta) horient
  exact congrArg Subtype.val hchoice

theorem globalTriangleChoiceCoherence_iff_edgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleChoiceCoherence (beta := beta) m U tau ↔
      GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  constructor
  · exact triangleEdgeOrientationCoherence_of_triangleChoiceCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact triangleChoiceCoherence_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)

end SCT.FND1
