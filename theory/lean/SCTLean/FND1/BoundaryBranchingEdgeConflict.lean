import SCTLean.FND1.BoundaryBranchingEdgeGluing

/-!
# FND-1 Boundary Branching-Edge Conflict

This module isolates the first explicit local obstruction for the current
`tau`-driven route.

A disagreement between two triangle-induced local edge orientations on one
genuinely branching edge does not rule out *all possible* compatible boundary
data. But it does kill the current route that tries to build compatibility from
that specific triangle witness datum `tau`.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A concrete local conflict on one genuinely branching triangle-bearing edge
for the current `tau`-induced local edge orientations. -/
def BranchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  ∃ edge : BranchingTriangleBoundaryEdge (beta := beta) m U,
    ∃ triangle₁ : ↑(triangleSimplices m U),
      ∃ triangle₂ : ↑(triangleSimplices m U),
        ∃ h₁ : edge.1.1.1 ∈ codimOneFaces (beta := beta) triangle₁.1,
          ∃ h₂ : edge.1.1.1 ∈ codimOneFaces (beta := beta) triangle₂.1,
            (tau triangle₁).localEdgeOrientationOnBoundaryEdge
                (beta := beta) m U triangle₁ edge.1.1 h₁ ≠
              (tau triangle₂).localEdgeOrientationOnBoundaryEdge
                (beta := beta) m U triangle₂ edge.1.1 h₂

theorem not_pureTriangleOverlapCoherence_of_branchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hconflict : BranchingEdgeConflict (beta := beta) m U tau) :
    ¬ PureTriangleOverlapCoherence (beta := beta) m U tau := by
  intro hpure
  rcases hconflict with ⟨edge, triangle₁, triangle₂, h₁, h₂, hne⟩
  have hneq : triangle₁ ≠ triangle₂ := by
    intro hEq
    subst hEq
    exact hne rfl
  have hoverlap : GlobalTriangleOverlapCoherence (beta := beta) m U tau :=
    (globalTriangleOverlapCoherence_iff_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)).2 hpure
  have hedgeEq :
      edge.1.1.1 = triangle₁.1 ∩ triangle₂.1 :=
    sharedBoundaryEdge_eq_intersection_of_ne
      (beta := beta) (m := m) (U := U) hneq edge.1.1 h₁ h₂
  have heq :=
    hoverlap.coherent hneq edge.1.1 hedgeEq h₁ h₂
  exact hne heq

theorem pureTriangleOverlapCoherence_implies_no_branchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hpure : PureTriangleOverlapCoherence (beta := beta) m U tau) :
    ¬ BranchingEdgeConflict (beta := beta) m U tau := by
  intro hconflict
  exact not_pureTriangleOverlapCoherence_of_branchingEdgeConflict
    (beta := beta) (m := m) (U := U) (tau := tau) hconflict hpure

theorem not_hasGluedBranchingTriangleEdgeOrientations_of_branchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hconflict : BranchingEdgeConflict (beta := beta) m U tau) :
    ¬ HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau := by
  intro hglue
  have hpure :
      PureTriangleOverlapCoherence (beta := beta) m U tau :=
    (pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)).2 hglue
  exact not_pureTriangleOverlapCoherence_of_branchingEdgeConflict
    (beta := beta) (m := m) (U := U) (tau := tau) hconflict hpure

theorem not_localTriangleEdgeStarCoherence_of_branchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hconflict : BranchingEdgeConflict (beta := beta) m U tau) :
    ¬ LocalTriangleEdgeStarCoherence (beta := beta) m U tau := by
  intro hstar
  have hpure :
      PureTriangleOverlapCoherence (beta := beta) m U tau :=
    (localTriangleEdgeStarCoherence_iff_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)).1 hstar
  exact not_pureTriangleOverlapCoherence_of_branchingEdgeConflict
    (beta := beta) (m := m) (U := U) (tau := tau) hconflict hpure

theorem not_hasTriangleWitnessLocalExtension_of_branchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hconflict : BranchingEdgeConflict (beta := beta) m U tau) :
    ¬ HasTriangleWitnessLocalExtension (beta := beta) m U tau := by
  intro hext
  have hpure :
      PureTriangleOverlapCoherence (beta := beta) m U tau :=
    (hasTriangleWitnessLocalExtension_iff_pureTriangleOverlapCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)).1 hext
  exact not_pureTriangleOverlapCoherence_of_branchingEdgeConflict
    (beta := beta) (m := m) (U := U) (tau := tau) hconflict hpure

end SCT.FND1
