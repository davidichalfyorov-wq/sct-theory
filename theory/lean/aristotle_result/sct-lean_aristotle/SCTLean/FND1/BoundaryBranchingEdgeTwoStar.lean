import SCTLean.FND1.BoundaryBranchingEdgeChoice

/-!
# FND-1 Boundary Branching-Edge Two-Star Split

This module isolates the first genuinely nontrivial branching regime:
one boundary edge whose triangle-star has exactly two witnesses.

In that case the entire local gluing question collapses to a binary law:
the two induced chosen codimension-one faces either agree or they do not.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- An edge whose triangle-star contains exactly two witnesses. -/
structure ExactTwoEdgeTriangleWitnesses
    (m : Nat) (U : Cover beta alpha)
    (edge : ↑(edgeSimplices m U)) where
  first : EdgeTriangleWitness (beta := beta) m U edge
  second : EdgeTriangleWitness (beta := beta) m U edge
  first_ne_second : first ≠ second
  exhaustive :
    ∀ witness : EdgeTriangleWitness (beta := beta) m U edge,
      witness = first ∨ witness = second

theorem exactTwoEdgeTriangleWitnesses_not_subsingleton
    (m : Nat) (U : Cover beta alpha)
    {edge : ↑(edgeSimplices m U)}
    (hexact : ExactTwoEdgeTriangleWitnesses (beta := beta) m U edge) :
    ¬ Subsingleton (EdgeTriangleWitness (beta := beta) m U edge) := by
  intro hsub
  exact hexact.first_ne_second (Subsingleton.elim _ _)

/-- Local binary-choice coherence on one edge: there is one chosen singleton
face agreeing with every triangle witness in that edge-star. -/
def LocalEdgeChosenFaceCoherenceAt
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : ↑(edgeSimplices m U)) : Prop :=
  ∃ face : LocalFaceSupport (beta := beta) edge.1,
    ∀ witness : EdgeTriangleWitness (beta := beta) m U edge,
      face =
        (tau witness.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U witness.1 edge witness.2

theorem localEdgeChosenFaceCoherenceAt_of_exactTwo_face_eq
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    {edge : ↑(edgeSimplices m U)}
    (hexact : ExactTwoEdgeTriangleWitnesses (beta := beta) m U edge)
    (hpair :
      (tau hexact.first.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.first.1 edge hexact.first.2 =
        (tau hexact.second.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.second.1 edge hexact.second.2) :
    LocalEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge := by
  refine ⟨(tau hexact.first.1).chosenFaceOnBoundaryEdge
      (beta := beta) m U hexact.first.1 edge hexact.first.2, ?_⟩
  intro witness
  rcases hexact.exhaustive witness with hw | hw
  · cases hw
    rfl
  · cases hw
    exact hpair

theorem localEdgeChosenFaceCoherenceAt_iff_pair_eq_of_exactTwo
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    {edge : ↑(edgeSimplices m U)}
    (hexact : ExactTwoEdgeTriangleWitnesses (beta := beta) m U edge) :
    LocalEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge ↔
      (tau hexact.first.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.first.1 edge hexact.first.2 =
        (tau hexact.second.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.second.1 edge hexact.second.2 := by
  constructor
  · intro hlocal
    rcases hlocal with ⟨face, hface⟩
    calc
      (tau hexact.first.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.first.1 edge hexact.first.2
        = face := by simpa using (hface hexact.first).symm
      _ =
        (tau hexact.second.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.second.1 edge hexact.second.2 := by
            simpa using hface hexact.second
  · intro hpair
    exact localEdgeChosenFaceCoherenceAt_of_exactTwo_face_eq
      (beta := beta) (m := m) (U := U) (tau := tau) hexact hpair

theorem not_localEdgeChosenFaceCoherenceAt_of_exactTwo_face_ne
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    {edge : ↑(edgeSimplices m U)}
    (hexact : ExactTwoEdgeTriangleWitnesses (beta := beta) m U edge)
    (hpairne :
      (tau hexact.first.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.first.1 edge hexact.first.2 ≠
        (tau hexact.second.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.second.1 edge hexact.second.2) :
    ¬ LocalEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge := by
  intro hlocal
  exact hpairne ((localEdgeChosenFaceCoherenceAt_iff_pair_eq_of_exactTwo
    (beta := beta) (m := m) (U := U) (tau := tau) hexact).1 hlocal)

theorem branchingEdgeConflict_of_exactTwo_face_ne
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (edge : BranchingTriangleBoundaryEdge (beta := beta) m U)
    (hexact : ExactTwoEdgeTriangleWitnesses (beta := beta) m U edge.1.1)
    (hpairne :
      (tau hexact.first.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.first.1 edge.1.1 hexact.first.2 ≠
        (tau hexact.second.1).chosenFaceOnBoundaryEdge
          (beta := beta) m U hexact.second.1 edge.1.1 hexact.second.2) :
    BranchingEdgeConflict (beta := beta) m U tau := by
  refine ⟨edge, hexact.first.1, hexact.second.1, hexact.first.2, hexact.second.2, ?_⟩
  intro horient
  apply hpairne
  exact localOrientationFromChosenFace_injective (beta := beta) (by
    simpa [TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge] using horient)

end SCT.FND1
