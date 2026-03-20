import SCTLean.FND1.BoundaryTriangleEdgeStarCoherence

/-!
# FND-1 Boundary Pure Triangle Overlap Coherence

This module removes the last explicit edge parameter from the current overlap
language.

For two distinct triangles whose intersection is a codimension-one face of
each, the shared edge is built canonically from that intersection. Coherence is
then stated directly on the overlap itself, with no separate external edge
index.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- The shared boundary edge determined canonically by the overlap of two
triangles, built from the proof that the overlap is a codimension-one face of
the first triangle. -/
def triangleOverlapEdge
    (m : Nat) (U : Cover beta alpha)
    (triangle₁ triangle₂ : ↑(triangleSimplices m U))
    (h₁ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₁.1) :
    ↑(edgeSimplices m U) :=
  ⟨triangle₁.1 ∩ triangle₂.1,
    triangle_boundary_faces_are_edges
      (beta := beta) (m := m) (U := U) triangle₁.2 h₁⟩

@[simp] theorem triangleOverlapEdge_val
    (m : Nat) (U : Cover beta alpha)
    (triangle₁ triangle₂ : ↑(triangleSimplices m U))
    (h₁ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₁.1) :
    (triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁).1 =
      triangle₁.1 ∩ triangle₂.1 := rfl

theorem TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge_congr
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (edge₁ edge₂ : ↑(edgeSimplices m U))
    (h₁ : edge₁.1 ∈ codimOneFaces (beta := beta) triangle.1)
    (h₂ : edge₂.1 ∈ codimOneFaces (beta := beta) triangle.1)
    (hedgeEq : edge₁ = edge₂) :
    HEq
      (w.localEdgeOrientationOnBoundaryEdge (beta := beta) m U triangle edge₁ h₁)
      (w.localEdgeOrientationOnBoundaryEdge (beta := beta) m U triangle edge₂ h₂) := by
  cases hedgeEq
  have hproof : h₁ = h₂ := Subsingleton.elim _ _
  subst hproof
  rfl

/-- Fully intrinsic overlap law: for distinct triangles whose intersection is a
codimension-one face of each, the two induced overlap-orientation rules agree
on the canonical overlap edge. -/
structure PureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop where
  coherent :
    ∀ {triangle₁ triangle₂ : ↑(triangleSimplices m U)}
      (_hneq : triangle₁ ≠ triangle₂)
      (h₁ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₁.1)
      (h₂ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₂.1),
      let overlapEdge :=
        triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁
      (tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ overlapEdge h₁ =
        (tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ overlapEdge (by
            simpa [triangleOverlapEdge] using h₂)

theorem pureTriangleOverlapCoherence_of_globalTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleOverlapCoherence (beta := beta) m U tau) :
    PureTriangleOverlapCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro triangle₁ triangle₂ hneq h₁ h₂
  let overlapEdge :=
    triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁
  have h₂' : overlapEdge.1 ∈ codimOneFaces (beta := beta) triangle₂.1 := by
    simpa [overlapEdge, triangleOverlapEdge] using h₂
  exact hcoh.coherent hneq overlapEdge rfl h₁ h₂'

theorem globalTriangleOverlapCoherence_of_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : PureTriangleOverlapCoherence (beta := beta) m U tau) :
    GlobalTriangleOverlapCoherence (beta := beta) m U tau := by
  refine ⟨?_⟩
  intro triangle₁ triangle₂ hneq edge hedgeEq h₁ h₂
  have hInt₁ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₁.1 := by
    simpa [hedgeEq] using h₁
  have hInt₂ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₂.1 := by
    simpa [hedgeEq] using h₂
  let overlapEdge :=
    triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ hInt₁
  have hEdgeEq : edge = overlapEdge := by
    apply Subtype.ext
    simpa [overlapEdge, triangleOverlapEdge] using hedgeEq
  have hpure := hcoh.coherent hneq hInt₁ hInt₂
  have hleft :
      HEq
        ((tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ edge h₁)
        ((tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ overlapEdge hInt₁) :=
    TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge_congr
      (beta := beta) (m := m) (U := U) triangle₁ (tau triangle₁)
      edge overlapEdge h₁ hInt₁ hEdgeEq
  have hmid :
      HEq
        ((tau triangle₁).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₁ overlapEdge hInt₁)
        ((tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ overlapEdge hInt₂) :=
    hpure.heq
  have hright :
      HEq
        ((tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ overlapEdge hInt₂)
        ((tau triangle₂).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle₂ edge h₂) :=
    TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge_congr
      (beta := beta) (m := m) (U := U) triangle₂ (tau triangle₂)
      overlapEdge edge hInt₂ h₂ hEdgeEq.symm
  exact eq_of_heq (hleft.trans (hmid.trans hright))

theorem globalTriangleOverlapCoherence_iff_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleOverlapCoherence (beta := beta) m U tau ↔
      PureTriangleOverlapCoherence (beta := beta) m U tau := by
  constructor
  · exact pureTriangleOverlapCoherence_of_globalTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact globalTriangleOverlapCoherence_of_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem triangleEdgeOrientationCoherence_iff_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    GlobalTriangleEdgeOrientationCoherence (beta := beta) m U tau ↔
      PureTriangleOverlapCoherence (beta := beta) m U tau := by
  rw [triangleEdgeOrientationCoherence_iff_triangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)]
  rw [globalTriangleOverlapCoherence_iff_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)]

theorem hasCompatibleLocalOrientation_of_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : PureTriangleOverlapCoherence (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_localTriangleEdgeStarCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((triangleEdgeOrientationCoherence_iff_localTriangleEdgeStarCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)).1
      ((triangleEdgeOrientationCoherence_iff_pureTriangleOverlapCoherence
        (beta := beta) (m := m) (U := U) (tau := tau)).2 hcoh))

theorem hasBoundarySquareZero_of_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : PureTriangleOverlapCoherence (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh)

end SCT.FND1
