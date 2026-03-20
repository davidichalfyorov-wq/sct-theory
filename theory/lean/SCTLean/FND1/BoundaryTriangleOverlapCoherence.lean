import Mathlib.Tactic
import SCTLean.FND1.BoundaryTriangleLocalExtension

/-!
# FND-1 Boundary Triangle Overlap Coherence

This module replaces one more externally indexed formulation by a purely overlap
formulation on the finite nerve.

For two distinct triangle simplices, a shared boundary edge is exactly their
set-theoretic intersection. This lets us restate the current coherence law
without an externally supplied edge index: coherence can be phrased directly on
pairs of triangles whose overlap is a codimension-one face of each.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem triangleSimplex_card_eq_three
    (m : Nat) (U : Cover beta alpha)
    {triangle : Finset beta}
    (htriangle : triangle ∈ triangleSimplices m U) :
    triangle.card = 3 := by
  exact (Finset.mem_powersetCard.mp (Finset.mem_filter.mp htriangle).1).2

theorem sharedBoundaryEdge_eq_intersection_of_ne
    (m : Nat) (U : Cover beta alpha)
    {triangle₁ triangle₂ : ↑(triangleSimplices m U)}
    (hneq : triangle₁ ≠ triangle₂)
    (edge : ↑(edgeSimplices m U))
    (h₁ : edge.1 ∈ codimOneFaces (beta := beta) triangle₁.1)
    (h₂ : edge.1 ∈ codimOneFaces (beta := beta) triangle₂.1) :
    edge.1 = triangle₁.1 ∩ triangle₂.1 := by
  rw [mem_codimOneFaces_iff] at h₁ h₂
  rcases h₁ with ⟨hsub₁, hcard₁⟩
  rcases h₂ with ⟨hsub₂, hcard₂⟩
  have htriCard₁ : triangle₁.1.card = 3 :=
    triangleSimplex_card_eq_three (beta := beta) (m := m) (U := U) triangle₁.2
  have htriCard₂ : triangle₂.1.card = 3 :=
    triangleSimplex_card_eq_three (beta := beta) (m := m) (U := U) triangle₂.2
  have hedgeCard : edge.1.card = 2 := by
    calc
      edge.1.card = triangle₁.1.card - 1 := hcard₁
      _ = 2 := by simp [htriCard₁]
  have hedgeSubInter : edge.1 ⊆ triangle₁.1 ∩ triangle₂.1 := by
    intro x hx
    rw [Finset.mem_inter]
    exact ⟨hsub₁ hx, hsub₂ hx⟩
  have hinterSub₁ : triangle₁.1 ∩ triangle₂.1 ⊆ triangle₁.1 := by
    intro x hx
    rw [Finset.mem_inter] at hx
    exact hx.1
  have hinterSub₂ : triangle₁.1 ∩ triangle₂.1 ⊆ triangle₂.1 := by
    intro x hx
    rw [Finset.mem_inter] at hx
    exact hx.2
  have hinterCardLeThree : (triangle₁.1 ∩ triangle₂.1).card ≤ 3 := by
    calc
      (triangle₁.1 ∩ triangle₂.1).card ≤ triangle₁.1.card := Finset.card_le_card hinterSub₁
      _ = 3 := htriCard₁
  have hinterCardNeThree : (triangle₁.1 ∩ triangle₂.1).card ≠ 3 := by
    intro hthree
    have hinterEq₁ : triangle₁.1 ∩ triangle₂.1 = triangle₁.1 := by
      apply Finset.eq_of_subset_of_card_le hinterSub₁
      simp [htriCard₁, hthree]
    have hinterEq₂ : triangle₁.1 ∩ triangle₂.1 = triangle₂.1 := by
      apply Finset.eq_of_subset_of_card_le hinterSub₂
      simp [htriCard₂, hthree]
    apply hneq
    apply Subtype.ext
    calc
      triangle₁.1 = triangle₁.1 ∩ triangle₂.1 := hinterEq₁.symm
      _ = triangle₂.1 := hinterEq₂
  have hinterCardLeTwo : (triangle₁.1 ∩ triangle₂.1).card ≤ 2 := by
    omega
  apply Finset.eq_of_subset_of_card_le hedgeSubInter
  simpa [hedgeCard] using hinterCardLeTwo

/-- Overlap-only version of the current coherence law. For distinct triangles,
their shared boundary edge is tied explicitly to their set-theoretic overlap. -/
structure GlobalTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop where
  coherent :
    ∀ {triangle₁ triangle₂ : ↑(triangleSimplices m U)}
      (_hneq : triangle₁ ≠ triangle₂)
      (edge : ↑(edgeSimplices m U))
      (_hedgeEq : edge.1 = triangle₁.1 ∩ triangle₂.1)
      (h₁ : edge.1 ∈ codimOneFaces (beta := beta) triangle₁.1)
      (h₂ : edge.1 ∈ codimOneFaces (beta := beta) triangle₂.1),
      (tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ edge h₁ =
        (tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂

theorem triangleOverlapCoherence_of_triangleEdgeOrientationCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau) :
    GlobalTriangleOverlapCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro triangle₁ triangle₂ _hneq edge _hedgeEq h₁ h₂
  exact hcoh.coherent edge h₁ h₂

theorem triangleEdgeOrientationCoherence_of_triangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleOverlapCoherence (beta := beta) m U tau) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro edge triangle₁ triangle₂ h₁ h₂
  by_cases hEq : triangle₁ = triangle₂
  · subst hEq
    have hproof : h₁ = h₂ := Subsingleton.elim _ _
    subst hproof
    rfl
  · have hedgeEq :
      edge.1 = triangle₁.1 ∩ triangle₂.1 :=
        sharedBoundaryEdge_eq_intersection_of_ne
          (beta := beta) (m := m) (U := U) hEq edge h₁ h₂
    exact hcoh.coherent hEq edge hedgeEq h₁ h₂

theorem triangleEdgeOrientationCoherence_iff_triangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau ↔
      GlobalTriangleOverlapCoherence (beta := beta) m U tau := by
  constructor
  · exact triangleOverlapCoherence_of_triangleEdgeOrientationCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact triangleEdgeOrientationCoherence_of_triangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)

end SCT.FND1
