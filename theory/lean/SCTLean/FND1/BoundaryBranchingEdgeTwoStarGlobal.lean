import SCTLean.FND1.BoundaryBranchingEdgeTwoStar

/-!
# FND-1 Boundary Branching-Edge Two-Star Globalization

This module globalizes the local two-star split.

If every genuinely branching triangle-bearing edge has exactly two witnesses,
then one binary agreement law per branching edge is enough to build the full
branching-edge choice datum, hence recover the current square-zero route.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Exact-two witness data on every genuinely branching triangle-bearing edge. -/
abbrev ExactTwoBranchingStarDatum
    (m : Nat) (U : Cover beta alpha) :=
  ∀ edge : BranchingTriangleBoundaryEdge (beta := beta) m U,
    ExactTwoEdgeTriangleWitnesses (beta := beta) m U edge.1.1

/-- In the exact-two branching regime, each branching edge carries one binary
agreement condition between its two witness-induced chosen faces. -/
def ExactTwoBranchingBinaryAgreement
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (sigma : ExactTwoBranchingStarDatum (beta := beta) m U) : Prop :=
  ∀ edge : BranchingTriangleBoundaryEdge (beta := beta) m U,
    (tau (sigma edge).first.1).chosenFaceOnBoundaryEdge
        (beta := beta) m U (sigma edge).first.1 edge.1.1 (sigma edge).first.2 =
      (tau (sigma edge).second.1).chosenFaceOnBoundaryEdge
        (beta := beta) m U (sigma edge).second.1 edge.1.1 (sigma edge).second.2

noncomputable def gluedBranchingTriangleEdgeChoicesOfExactTwoBranchingBinaryAgreement
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (sigma : ExactTwoBranchingStarDatum (beta := beta) m U)
    (hagreement : ExactTwoBranchingBinaryAgreement (beta := beta) m U tau sigma) :
    GluedBranchingTriangleEdgeChoices (beta := beta) m U tau where
  edgeFaces := fun edge =>
    Classical.choose
      (localEdgeChosenFaceCoherenceAt_of_exactTwo_face_eq
        (beta := beta) (m := m) (U := U) (tau := tau) (sigma edge) (hagreement edge))
  agrees := by
    intro edge triangle hedge
    exact Classical.choose_spec
      (localEdgeChosenFaceCoherenceAt_of_exactTwo_face_eq
        (beta := beta) (m := m) (U := U) (tau := tau) (sigma edge) (hagreement edge))
      ⟨triangle, hedge⟩

theorem hasGluedBranchingTriangleEdgeChoices_of_exactTwoBranchingBinaryAgreement
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (sigma : ExactTwoBranchingStarDatum (beta := beta) m U)
    (hagreement : ExactTwoBranchingBinaryAgreement (beta := beta) m U tau sigma) :
    HasGluedBranchingTriangleEdgeChoices (beta := beta) m U tau := by
  exact ⟨gluedBranchingTriangleEdgeChoicesOfExactTwoBranchingBinaryAgreement
    (beta := beta) (m := m) (U := U) (tau := tau) sigma hagreement⟩

theorem hasCompatibleLocalOrientation_of_exactTwoBranchingBinaryAgreement
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (sigma : ExactTwoBranchingStarDatum (beta := beta) m U)
    (hagreement : ExactTwoBranchingBinaryAgreement (beta := beta) m U tau sigma) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_hasGluedBranchingTriangleEdgeChoices
    (beta := beta) (m := m) (U := U) (tau := tau)
    (hasGluedBranchingTriangleEdgeChoices_of_exactTwoBranchingBinaryAgreement
      (beta := beta) (m := m) (U := U) (tau := tau) sigma hagreement)

theorem hasBoundarySquareZero_of_exactTwoBranchingBinaryAgreement
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (sigma : ExactTwoBranchingStarDatum (beta := beta) m U)
    (hagreement : ExactTwoBranchingBinaryAgreement (beta := beta) m U tau sigma) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasGluedBranchingTriangleEdgeChoices
    (beta := beta) (m := m) (U := U) (tau := tau)
    (hasGluedBranchingTriangleEdgeChoices_of_exactTwoBranchingBinaryAgreement
      (beta := beta) (m := m) (U := U) (tau := tau) sigma hagreement)

end SCT.FND1
