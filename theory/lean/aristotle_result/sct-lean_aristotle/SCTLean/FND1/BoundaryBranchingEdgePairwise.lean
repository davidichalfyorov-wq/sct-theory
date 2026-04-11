import SCTLean.FND1.BoundaryBranchingEdgeChoice

/-!
# FND-1 Boundary Branching-Edge Pairwise Coherence

This module removes the last existential phrasing from the minimal branching
edge layer.

On a genuinely branching edge, the current gluing problem is equivalent to a
pure pairwise law: every pair of triangle witnesses on that edge induces the
same chosen codimension-one face.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- On one genuinely branching edge, all triangle witnesses induce the same
chosen singleton face. -/
def PairwiseBranchingEdgeChosenFaceCoherenceAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : BranchingTriangleBoundaryEdge (beta := beta) m U) : Prop :=
  ∀ witness₁ witness₂ : EdgeTriangleWitness (beta := beta) m U edge.1.1,
    (tau witness₁.1).chosenFaceOnBoundaryEdge
        (beta := beta) m U witness₁.1 edge.1.1 witness₁.2 =
      (tau witness₂.1).chosenFaceOnBoundaryEdge
        (beta := beta) m U witness₂.1 edge.1.1 witness₂.2

/-- Pairwise chosen-face coherence on every genuinely branching edge. -/
def PairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  ∀ edge : BranchingTriangleBoundaryEdge (beta := beta) m U,
    PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge

theorem pairwiseBranchingEdgeChosenFaceCoherenceAt_of_gluedChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedBranchingTriangleEdgeChoices (beta := beta) m U tau)
    (edge : BranchingTriangleBoundaryEdge (beta := beta) m U) :
    PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge := by
  intro witness₁ witness₂
  calc
    (tau witness₁.1).chosenFaceOnBoundaryEdge
        (beta := beta) m U witness₁.1 edge.1.1 witness₁.2
      = eta.edgeFaces edge := by
          simpa using (eta.agrees edge (triangle := witness₁.1) witness₁.2).symm
    _ =
      (tau witness₂.1).chosenFaceOnBoundaryEdge
        (beta := beta) m U witness₂.1 edge.1.1 witness₂.2 := by
          simpa using eta.agrees edge (triangle := witness₂.1) witness₂.2

theorem pairwiseBranchingChosenFaceCoherence_of_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglue : HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau) :
    PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau := by
  rcases hglue with ⟨eta⟩
  intro edge
  exact pairwiseBranchingEdgeChosenFaceCoherenceAt_of_gluedChoices
    (beta := beta) (m := m) (U := U) (tau := tau) eta edge

noncomputable def gluedBranchingTriangleEdgeChoicesOfPairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hpair : PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau) :
    GluedBranchingTriangleEdgeChoices (beta := beta) m U tau where
  edgeFaces := by
    classical
    intro edge
    let selected : EdgeTriangleWitness (beta := beta) m U edge.1.1 := Classical.choice edge.1.2
    exact (tau selected.1).chosenFaceOnBoundaryEdge
      (beta := beta) m U selected.1 edge.1.1 selected.2
  agrees := by
    classical
    intro edge triangle hedge
    let selected : EdgeTriangleWitness (beta := beta) m U edge.1.1 := Classical.choice edge.1.2
    exact hpair edge selected ⟨triangle, hedge⟩

theorem hasGluedBranchingTriangleEdgeChoices_of_pairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hpair : PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau) :
    HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau := by
  exact ⟨gluedBranchingTriangleEdgeChoicesOfPairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U) (tau := tau) hpair⟩

theorem pairwiseBranchingChosenFaceCoherence_iff_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau ↔
      HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau := by
  constructor
  · exact hasGluedBranchingTriangleEdgeChoices_of_pairwiseBranchingChosenFaceCoherence
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact pairwiseBranchingChosenFaceCoherence_of_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    PureTriangleOverlapCoherence (beta := beta) m U tau ↔
      PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau := by
  rw [pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)]
  rw [← pairwiseBranchingChosenFaceCoherence_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)]

theorem hasCompatibleLocalOrientation_of_pairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hpair : PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_hasGluedBranchingTriangleEdgeChoices
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((pairwiseBranchingChosenFaceCoherence_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)).1 hpair)

theorem hasBoundarySquareZero_of_pairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hpair : PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasGluedBranchingTriangleEdgeChoices
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((pairwiseBranchingChosenFaceCoherence_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)).1 hpair)

/-- A local conflict on one genuinely branching edge. -/
def BranchingEdgeConflictAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : BranchingTriangleBoundaryEdge (beta := beta) m U) : Prop :=
  ∃ witness₁ witness₂ : EdgeTriangleWitness (beta := beta) m U edge.1.1,
    (tau witness₁.1).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U witness₁.1 edge.1.1 witness₁.2 ≠
      (tau witness₂.1).localEdgeOrientationOnBoundaryEdge
        (beta := beta) m U witness₂.1 edge.1.1 witness₂.2

theorem branchingEdgeConflict_of_branchingEdgeConflictAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    {edge : BranchingTriangleBoundaryEdge (beta := beta) m U}
    (hconf : BranchingEdgeConflictAt (beta := beta) m U tau edge) :
    BranchingEdgeConflict (beta := beta) m U tau := by
  rcases hconf with ⟨witness₁, witness₂, hne⟩
  exact ⟨edge, witness₁.1, witness₂.1, witness₁.2, witness₂.2, hne⟩

theorem not_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_branchingEdgeConflictAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    {edge : BranchingTriangleBoundaryEdge (beta := beta) m U}
    (hconf : BranchingEdgeConflictAt (beta := beta) m U tau edge) :
    ¬ PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge := by
  intro hpair
  rcases hconf with ⟨witness₁, witness₂, hne⟩
  apply hne
  exact congrArg
    (fun chosen => localOrientationFromChosenFace (beta := beta) chosen)
    (hpair witness₁ witness₂)

theorem branchingEdgeConflictAt_of_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    {edge : BranchingTriangleBoundaryEdge (beta := beta) m U}
    (hnot : ¬ PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge) :
    BranchingEdgeConflictAt (beta := beta) m U tau edge := by
  classical
  by_contra hconf
  apply hnot
  intro witness₁ witness₂
  by_contra hface
  have horientne :
      (tau witness₁.1).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U witness₁.1 edge.1.1 witness₁.2 ≠
        (tau witness₂.1).localEdgeOrientationOnBoundaryEdge
          (beta := beta) m U witness₂.1 edge.1.1 witness₂.2 := by
    intro horient
    apply hface
    exact localOrientationFromChosenFace_injective (beta := beta) (by
      simpa [TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge] using horient)
  exact hconf ⟨witness₁, witness₂, horientne⟩

theorem branchingEdgeConflictAt_iff_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : BranchingTriangleBoundaryEdge (beta := beta) m U) :
    BranchingEdgeConflictAt (beta := beta) m U tau edge ↔
      ¬ PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge := by
  constructor
  · exact not_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_branchingEdgeConflictAt
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact branchingEdgeConflictAt_of_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem pairwiseBranchingChosenFaceCoherence_implies_no_branchingEdgeConflict
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hpair : PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau) :
    ¬ BranchingEdgeConflict (beta := beta) m U tau := by
  intro hconf
  rcases hconf with ⟨edge, triangle₁, triangle₂, h₁, h₂, hne⟩
  have hlocalConf : BranchingEdgeConflictAt (beta := beta) m U tau edge := by
    exact ⟨⟨triangle₁, h₁⟩, ⟨triangle₂, h₂⟩, hne⟩
  exact ((branchingEdgeConflictAt_iff_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
    (beta := beta) (m := m) (U := U) (tau := tau) edge).1 hlocalConf) (hpair edge)

theorem branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    BranchingEdgeConflict (beta := beta) m U tau ↔
      ¬ PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau := by
  constructor
  · intro hconf hpair
    exact pairwiseBranchingChosenFaceCoherence_implies_no_branchingEdgeConflict
      (beta := beta) (m := m) (U := U) (tau := tau) hpair hconf
  · intro hnot
    classical
    rcases not_forall.mp hnot with ⟨edge, hnotedge⟩
    exact branchingEdgeConflict_of_branchingEdgeConflictAt
      (beta := beta) (m := m) (U := U) (tau := tau)
      (branchingEdgeConflictAt_of_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) (m := m) (U := U) (tau := tau) hnotedge)

end SCT.FND1
