import SCTLean.FND1.BoundaryPureTriangleOverlapVacuous

/-!
# FND-1 Boundary Branching-Edge Gluing

This module removes one more vacuous part of the remaining assumption stack.

Singleton edge-stars no longer need any gluing data: their coherence is already
automatic. The only edges that still matter are branching triangle-bearing
edges, namely those whose star contains at least two different triangle
witnesses.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Triangle-bearing edges whose star is genuinely branching, i.e. not a
subsingleton. -/
abbrev BranchingTriangleBoundaryEdge
    (m : Nat) (U : Cover beta alpha) :=
  { edge : TriangleBoundaryEdge (beta := beta) m U //
      ¬ Subsingleton (EdgeTriangleWitness (beta := beta) m U edge.1) }

/-- A gluing datum restricted only to genuinely branching triangle-bearing
edges. Singleton stars need no data because they are automatically coherent. -/
structure GluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) where
  edgeFaces :
    ∀ edge : BranchingTriangleBoundaryEdge (beta := beta) m U,
      LocalFaceOrientationDatum (beta := beta) edge.1.1.1
  agrees :
    ∀ (edge : BranchingTriangleBoundaryEdge (beta := beta) m U)
      {triangle : ↑(triangleSimplices m U)}
      (hedge : edge.1.1.1 ∈ codimOneFaces (beta := beta) triangle.1),
      edgeFaces edge =
        (tau triangle).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U triangle edge.1.1 hedge

/-- Existence form of branching-edge-only gluing. -/
def HasGluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  Nonempty (GluedBranchingTriangleEdgeOrientations (beta := beta) m U tau)

theorem localEdgeOrientationOnBoundaryEdge_eq_of_witness_eq
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : ↑(edgeSimplices m U))
    {w₁ w₂ : EdgeTriangleWitness (beta := beta) m U edge}
    (hw : w₁ = w₂) :
    (tau w₁.1).localEdgeOrientationOnBoundaryEdge
      (beta := beta) m U w₁.1 edge w₁.2 =
      (tau w₂.1).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U w₂.1 edge w₂.2 := by
  cases hw
  rfl

theorem overlapEdge_not_subsingleton
    (m : Nat) (U : Cover beta alpha)
    {triangle₁ triangle₂ : ↑(triangleSimplices m U)}
    (hneq : triangle₁ ≠ triangle₂)
    (h₁ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₁.1)
    (h₂ : triangle₁.1 ∩ triangle₂.1 ∈ codimOneFaces (beta := beta) triangle₂.1) :
    ¬ Subsingleton (EdgeTriangleWitness (beta := beta) m U
      (triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁)) := by
  intro hsub
  let overlapEdge :=
    triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁
  have h₂' : overlapEdge.1 ∈ codimOneFaces (beta := beta) triangle₂.1 := by
    simpa [overlapEdge, triangleOverlapEdge] using h₂
  have hwEq :
      (⟨triangle₁, h₁⟩ : EdgeTriangleWitness (beta := beta) m U overlapEdge) =
        (⟨triangle₂, h₂'⟩ : EdgeTriangleWitness (beta := beta) m U overlapEdge) := by
    exact Subsingleton.elim _ _
  have htriEq : triangle₁ = triangle₂ := by
    exact congrArg Subtype.val hwEq
  exact hneq htriEq

theorem pureTriangleOverlapCoherence_of_hasGluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau) :
    PureTriangleOverlapCoherence (beta := beta) m U tau := by
  rcases hglue with ⟨eta⟩
  refine ⟨?_⟩
  intro triangle₁ triangle₂ hneq h₁ h₂
  let overlapEdge :=
    triangleOverlapEdge (beta := beta) (m := m) (U := U) triangle₁ triangle₂ h₁
  have h₂' : overlapEdge.1 ∈ codimOneFaces (beta := beta) triangle₂.1 := by
    simpa [overlapEdge, triangleOverlapEdge] using h₂
  let branchingEdge : BranchingTriangleBoundaryEdge (beta := beta) m U :=
    ⟨⟨overlapEdge, ⟨⟨triangle₁, h₁⟩⟩⟩,
      overlapEdge_not_subsingleton
        (beta := beta) (m := m) (U := U) hneq h₁ h₂⟩
  calc
    (tau triangle₁).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle₁ overlapEdge h₁
      = eta.edgeFaces branchingEdge := by
          simpa [branchingEdge] using (eta.agrees branchingEdge (triangle := triangle₁) h₁).symm
    _ = (tau triangle₂).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U triangle₂ overlapEdge h₂' := by
          simpa [branchingEdge] using eta.agrees branchingEdge (triangle := triangle₂) h₂'

noncomputable def gluedBranchingTriangleEdgeOrientationsOfPure
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : PureTriangleOverlapCoherence (beta := beta) m U tau) :
    GluedBranchingTriangleEdgeOrientations (beta := beta) m U tau where
  edgeFaces := by
    classical
    intro edge
    let selected : EdgeTriangleWitness (beta := beta) m U edge.1.1 := Classical.choice edge.1.2
    exact (tau selected.1).localEdgeOrientationOnBoundaryEdge
      (beta := beta) m U selected.1 edge.1.1 selected.2
  agrees := by
    classical
    intro edge triangle hedge
    let selected : EdgeTriangleWitness (beta := beta) m U edge.1.1 := Classical.choice edge.1.2
    by_cases hEq : selected.1 = triangle
    · have hwEq :
        selected = (⟨triangle, hedge⟩ : EdgeTriangleWitness (beta := beta) m U edge.1.1) := by
          apply Subtype.ext
          exact hEq
      exact localEdgeOrientationOnBoundaryEdge_eq_of_witness_eq
        (beta := beta) (m := m) (U := U) tau edge.1.1 hwEq
    · have hoverlap : GlobalTriangleOverlapCoherence (beta := beta) m U tau :=
        (globalTriangleOverlapCoherence_iff_pureTriangleOverlapCoherence
          (beta := beta) (m := m) (U := U) (tau := tau)).2 hcoh
      have hedgeEq :
          edge.1.1.1 = selected.1.1 ∩ triangle.1 :=
        sharedBoundaryEdge_eq_intersection_of_ne
          (beta := beta) (m := m) (U := U) hEq edge.1.1 selected.2 hedge
      exact hoverlap.coherent hEq edge.1.1 hedgeEq selected.2 hedge

theorem hasGluedBranchingTriangleEdgeOrientations_of_pureTriangleOverlapCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : PureTriangleOverlapCoherence (beta := beta) m U tau) :
    HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau := by
  exact ⟨gluedBranchingTriangleEdgeOrientationsOfPure
    (beta := beta) (m := m) (U := U) (tau := tau) hcoh⟩

theorem pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    PureTriangleOverlapCoherence (beta := beta) m U tau ↔
      HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau := by
  constructor
  · exact hasGluedBranchingTriangleEdgeOrientations_of_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact pureTriangleOverlapCoherence_of_hasGluedBranchingTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem hasCompatibleLocalOrientation_of_hasGluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)).2 hglue)

theorem hasBoundarySquareZero_of_hasGluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_pureTriangleOverlapCoherence
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)).2 hglue)

end SCT.FND1
