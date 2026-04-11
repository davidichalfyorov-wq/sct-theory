import SCTLean.FND1.BoundaryBranchingEdgeConflict

/-!
# FND-1 Boundary Branching-Edge Binary Choice

This module puts the genuinely nontrivial local gap into a minimal normal form.

On a branching edge, every triangle-induced local orientation is of the form
"choose one codimension-one face of this edge". So the branching-edge gluing
problem is equivalent to the existence of one coherent chosen-face datum on each
branching edge.

For a star of size exactly two, this is literally a binary choice law on the
shared edge.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Binary choice data on genuinely branching triangle-bearing edges: choose one
singleton face of each branching edge, and require agreement with every
triangle witness in that edge-star. -/
structure GluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) where
  edgeFaces :
    ∀ edge : BranchingTriangleBoundaryEdge (beta := beta) m U,
      LocalFaceSupport (beta := beta) edge.1.1.1
  agrees :
    ∀ (edge : BranchingTriangleBoundaryEdge (beta := beta) m U)
      {triangle : ↑(triangleSimplices m U)}
      (hedge : edge.1.1.1 ∈ codimOneFaces (beta := beta) triangle.1),
      edgeFaces edge =
        (tau triangle).chosenFaceOnBoundaryEdge
          (beta := beta) m U triangle edge.1.1 hedge

/-- Existence form of the branching-edge binary choice datum. -/
def HasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop :=
  Nonempty (GluedBranchingTriangleEdgeChoices (beta := beta) m U tau)

noncomputable def gluedBranchingTriangleEdgeOrientationsOfChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedBranchingTriangleEdgeChoices (beta := beta) m U tau) :
    GluedBranchingTriangleEdgeOrientations (beta := beta) m U tau where
  edgeFaces := fun edge =>
    localOrientationFromChosenFace (beta := beta) (eta.edgeFaces edge)
  agrees := by
    intro edge triangle hedge
    have hchoice := eta.agrees edge (triangle := triangle) hedge
    simpa [TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge] using
      congrArg (fun chosen => localOrientationFromChosenFace (beta := beta) chosen) hchoice

theorem hasGluedBranchingTriangleEdgeOrientations_of_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hchoice : HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau) :
    HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau := by
  rcases hchoice with ⟨eta⟩
  exact ⟨gluedBranchingTriangleEdgeOrientationsOfChoices
    (beta := beta) (m := m) (U := U) (tau := tau) eta⟩

noncomputable def gluedBranchingTriangleEdgeChoicesOfOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (eta : GluedBranchingTriangleEdgeOrientations (beta := beta) m U tau) :
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
    have horient :
        (tau selected.1).localEdgeOrientationOnBoundaryEdge
            (beta := beta) m U selected.1 edge.1.1 selected.2 =
          (tau triangle).localEdgeOrientationOnBoundaryEdge
            (beta := beta) m U triangle edge.1.1 hedge := by
      calc
        (tau selected.1).localEdgeOrientationOnBoundaryEdge
            (beta := beta) m U selected.1 edge.1.1 selected.2
          = eta.edgeFaces edge := by
              simpa [selected] using (eta.agrees edge (triangle := selected.1) selected.2).symm
        _ = (tau triangle).localEdgeOrientationOnBoundaryEdge
            (beta := beta) m U triangle edge.1.1 hedge := by
              simpa using eta.agrees edge (triangle := triangle) hedge
    exact localOrientationFromChosenFace_injective (beta := beta) (by
      simpa [TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge, selected] using horient)

theorem hasGluedBranchingTriangleEdgeChoices_of_hasGluedBranchingTriangleEdgeOrientations
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (horient : HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau) :
    HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau := by
  rcases horient with ⟨eta⟩
  exact ⟨gluedBranchingTriangleEdgeChoicesOfOrientations
    (beta := beta) (m := m) (U := U) (tau := tau) eta⟩

theorem hasGluedBranchingTriangleEdgeOrientations_iff_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    HasGluedBranchingTriangleEdgeOrientations (beta := beta) m U tau ↔
      HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau := by
  constructor
  · exact hasGluedBranchingTriangleEdgeChoices_of_hasGluedBranchingTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)
  · exact hasGluedBranchingTriangleEdgeOrientations_of_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)

theorem pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    PureTriangleOverlapCoherence (beta := beta) m U tau ↔
      HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau := by
  rw [pureTriangleOverlapCoherence_iff_hasGluedBranchingTriangleEdgeOrientations
      (beta := beta) (m := m) (U := U) (tau := tau)]
  rw [hasGluedBranchingTriangleEdgeOrientations_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)]

theorem hasCompatibleLocalOrientation_of_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hchoice : HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_hasGluedBranchingTriangleEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((hasGluedBranchingTriangleEdgeOrientations_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)).2 hchoice)

theorem hasBoundarySquareZero_of_hasGluedBranchingTriangleEdgeChoices
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hchoice : HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasGluedBranchingTriangleEdgeOrientations
    (beta := beta) (m := m) (U := U) (tau := tau)
    ((hasGluedBranchingTriangleEdgeOrientations_iff_hasGluedBranchingTriangleEdgeChoices
      (beta := beta) (m := m) (U := U) (tau := tau)).2 hchoice)

end SCT.FND1
