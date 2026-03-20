import SCTLean.FND1.BoundarySharedEdgeWitnessFlexibility
import SCTLean.FND1.BoundaryBranchingEdgePairwise

/-!
# FND-1 Boundary Shared-Edge Tau Variants

This module packages the shared-edge witness-flexibility obstruction at the
level of global triangle-choice data.

On the same finite nerve support, changing only the ordered witness on one
triangle can flip the chosen singleton face on a shared edge. Therefore the
pairwise branching-edge coherence law is not determined by overlap support
alone.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Replace the ordered witness carried by one fixed triangle. -/
noncomputable def replaceTriangleWitness
    {m : Nat} {U : Cover beta alpha}
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    TriangleOrderedChoiceDatum (beta := beta) m U :=
  fun t =>
    if h : t = triangle then by
      cases h
      exact w
    else tau t

@[simp] theorem replaceTriangleWitness_self
    {m : Nat} {U : Cover beta alpha}
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    replaceTriangleWitness (beta := beta) tau triangle w triangle = w := by
  simp [replaceTriangleWitness]

theorem replaceTriangleWitness_of_ne
    {m : Nat} {U : Cover beta alpha}
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle target : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hne : target ≠ triangle) :
    replaceTriangleWitness (beta := beta) tau triangle w target = tau target := by
  simp [replaceTriangleWitness, hne]

/-- Replace the witnesses on two fixed triangles. -/
noncomputable def replaceTwoTriangleWitnesses
    {m : Nat} {U : Cover beta alpha}
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (triangle2 : ↑(triangleSimplices m U))
    (w2 : TriangleOrderedWitness triangle2.1) :
    TriangleOrderedChoiceDatum (beta := beta) m U :=
  replaceTriangleWitness (beta := beta)
    (replaceTriangleWitness (beta := beta) tau triangle1 w1)
    triangle2 w2

theorem replaceTwoTriangleWitnesses_left
    {m : Nat} {U : Cover beta alpha}
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hne : triangle1 ≠ triangle2) :
    replaceTwoTriangleWitnesses (beta := beta) tau triangle1 w1 triangle2 w2 triangle1 = w1 := by
  simp [replaceTwoTriangleWitnesses, replaceTriangleWitness, hne]

@[simp] theorem replaceTwoTriangleWitnesses_right
    {m : Nat} {U : Cover beta alpha}
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1) :
    replaceTwoTriangleWitnesses (beta := beta) tau triangle1 w1 triangle2 w2 triangle2 = w2 := by
  simp [replaceTwoTriangleWitnesses, replaceTriangleWitness]

theorem sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    (w1.edgeAB (beta := beta) m U triangle1).1 ∈ codimOneFaces (beta := beta) triangle2.1 := by
  simpa [hshared] using
    (w2.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
      (triangle := triangle2))

theorem chosenFaceOnAB_val_eq_singleton_b_of_eq
    {m : Nat} {U : Cover beta alpha}
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (edge : ↑(edgeSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1)
    (he : edge = w.edgeAB (beta := beta) m U triangle) :
    ((w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle edge hedge).1 : Finset beta) =
      ({w.b} : Finset beta) := by
  subst he
  simp

theorem chosenFaceOnSharedAB_val_eq_singleton_b
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    ((w2.chosenFaceOnBoundaryEdge (beta := beta) m U triangle2
        (w1.edgeAB (beta := beta) m U triangle1)
        (sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
          (beta := beta) triangle1 triangle2 w1 w2 hshared)).1 : Finset beta) =
      ({w2.b} : Finset beta) := by
  let edge := w1.edgeAB (beta := beta) m U triangle1
  let hedge :=
    sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
      (beta := beta) triangle1 triangle2 w1 w2 hshared
  have hgeneric :
      ∀ (edge : ↑(edgeSimplices m U))
        (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle2.1),
        edge = w2.edgeAB (beta := beta) m U triangle2 →
          ((w2.chosenFaceOnBoundaryEdge (beta := beta) m U triangle2 edge hedge).1 : Finset beta) =
            ({w2.b} : Finset beta) := by
    intro edge hedge he
    subst he
    simp
  exact hgeneric edge hedge hshared

theorem chosenFaceOnSharedAB_val_eq_singleton_a
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    (((swapABTriangleOrderedWitness w2).chosenFaceOnBoundaryEdge (beta := beta) m U triangle2
        (w1.edgeAB (beta := beta) m U triangle1)
        (sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
          (beta := beta) triangle1 triangle2 w1 w2 hshared)).1 : Finset beta) =
      ({w2.a} : Finset beta) := by
  let edge := w1.edgeAB (beta := beta) m U triangle1
  let hedge :=
    sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
      (beta := beta) triangle1 triangle2 w1 w2 hshared
  have heqSwap :
      edge = (swapABTriangleOrderedWitness w2).edgeAB (beta := beta) m U triangle2 := by
    calc
      edge = w2.edgeAB (beta := beta) m U triangle2 := hshared
      _ = (swapABTriangleOrderedWitness w2).edgeAB (beta := beta) m U triangle2 := by
        symm
        exact swapABTriangleOrderedWitness_edgeAB_eq
          (beta := beta) (m := m) (U := U) triangle2 w2
  have hgeneric :
      ∀ (edge : ↑(edgeSimplices m U))
        (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle2.1),
        edge = (swapABTriangleOrderedWitness w2).edgeAB (beta := beta) m U triangle2 →
          (((swapABTriangleOrderedWitness w2).chosenFaceOnBoundaryEdge
            (beta := beta) m U triangle2 edge hedge).1 : Finset beta) =
            ({w2.a} : Finset beta) := by
    intro edge hedge he
    subst he
    simpa using
      (chosenFace_edgeAB_val_of_swapABTriangleOrderedWitness
        (beta := beta) (m := m) (U := U) triangle2 w2)
  exact hgeneric edge hedge heqSwap

theorem sharedAB_not_subsingleton
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    ¬ Subsingleton (EdgeTriangleWitness (beta := beta) m U (w1.edgeAB (beta := beta) m U triangle1)) := by
  intro hsub
  let witness1 : EdgeTriangleWitness (beta := beta) m U (w1.edgeAB (beta := beta) m U triangle1) :=
    ⟨triangle1,
      w1.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U) (triangle := triangle1)⟩
  let witness2 : EdgeTriangleWitness (beta := beta) m U (w1.edgeAB (beta := beta) m U triangle1) :=
    ⟨triangle2,
      sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
        (beta := beta) triangle1 triangle2 w1 w2 hshared⟩
  have hwEq : witness1 = witness2 := Subsingleton.elim _ _
  have htriEq : triangle1 = triangle2 := congrArg Subtype.val hwEq
  exact hneq htriEq

noncomputable def sharedABBranchingEdge
    {m : Nat} {U : Cover beta alpha}
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    BranchingTriangleBoundaryEdge (beta := beta) m U :=
  ⟨⟨w1.edgeAB (beta := beta) m U triangle1,
      ⟨⟨triangle1,
        w1.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
          (triangle := triangle1)⟩⟩⟩,
    sharedAB_not_subsingleton (beta := beta) triangle1 triangle2 w1 w2 hneq hshared⟩

theorem not_both_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    let tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2
    let tauSwap := replaceTwoTriangleWitnesses
      (beta := beta) tau0 triangle1 w1 triangle2 (swapABTriangleOrderedWitness w2)
    let edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared
    ¬ (PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tau edge ∧
        PairwiseBranchingEdgeChosenFaceCoherenceAt (beta := beta) m U tauSwap edge) := by
  dsimp
  let tau :=
    replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2
  let tauSwap :=
    replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
      (swapABTriangleOrderedWitness w2)
  let edge : BranchingTriangleBoundaryEdge (beta := beta) m U :=
    sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared
  let witness1 : EdgeTriangleWitness (beta := beta) m U edge.1.1 :=
    ⟨triangle1,
      w1.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle1)⟩
  let witness2 : EdgeTriangleWitness (beta := beta) m U edge.1.1 :=
    ⟨triangle2,
      sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq
        (beta := beta) triangle1 triangle2 w1 w2 hshared⟩
  have hnotTau_of
      (hne : edgeABFace (beta := beta) triangle1 w1 ≠
        edgeABFace (beta := beta) triangle2 w2) :
      ¬ PairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) m U tau edge := by
    intro hpair
    have hEqRaw := congrArg Subtype.val (hpair witness1 witness2)
    have hEq1 :
        edgeABFace (beta := beta) triangle1 w1 =
          edgeABFace (beta := beta) triangle2 w2 := by
      have hleft :
          (((tau triangle1).chosenFaceOnBoundaryEdge (beta := beta) m U
            triangle1 edge.1.1 witness1.2).1 : Finset beta) = ({w1.b} : Finset beta) := by
        have hself :
            edge.1.1 = w1.edgeAB (beta := beta) m U triangle1 := by
          rfl
        dsimp [tau]
        rw [replaceTwoTriangleWitnesses_left (beta := beta) tau0 triangle1 triangle2 w1 w2 hneq]
        exact chosenFaceOnAB_val_eq_singleton_b_of_eq
          (beta := beta) (m := m) (U := U) triangle1 w1 edge.1.1 witness1.2 hself
      have hright :
          (((tau triangle2).chosenFaceOnBoundaryEdge (beta := beta) m U
            triangle2 edge.1.1 witness2.2).1 : Finset beta) = ({w2.b} : Finset beta) := by
        dsimp [tau]
        rw [replaceTwoTriangleWitnesses_right (beta := beta) tau0 triangle1 triangle2 w1 w2]
        exact chosenFaceOnSharedAB_val_eq_singleton_b
          (beta := beta) triangle1 triangle2 w1 w2 hshared
      calc
        edgeABFace (beta := beta) triangle1 w1 = ({w1.b} : Finset beta) := by
          simp [edgeABFace_eq]
        _ =
            (((tau triangle1).chosenFaceOnBoundaryEdge (beta := beta) m U
              triangle1 edge.1.1 witness1.2).1 : Finset beta) := hleft.symm
        _ = (((tau triangle2).chosenFaceOnBoundaryEdge (beta := beta) m U
            triangle2 edge.1.1 witness2.2).1 : Finset beta) := hEqRaw
        _ = ({w2.b} : Finset beta) := hright
        _ = edgeABFace (beta := beta) triangle2 w2 := by
          symm
          simp [edgeABFace_eq]
    exact hne hEq1
  have hnotTauSwap_of
      (hne : edgeABFace (beta := beta) triangle1 w1 ≠
        swapABEdgeFace (beta := beta) triangle2 w2) :
      ¬ PairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) m U tauSwap edge := by
    intro hpair
    have hEqRaw := congrArg Subtype.val (hpair witness1 witness2)
    have hEq2 :
        edgeABFace (beta := beta) triangle1 w1 =
          swapABEdgeFace (beta := beta) triangle2 w2 := by
      have hleft :
          (((tauSwap triangle1).chosenFaceOnBoundaryEdge (beta := beta) m U
            triangle1 edge.1.1 witness1.2).1 : Finset beta) = ({w1.b} : Finset beta) := by
        have hself :
            edge.1.1 = w1.edgeAB (beta := beta) m U triangle1 := by
          rfl
        dsimp [tauSwap]
        rw [replaceTwoTriangleWitnesses_left (beta := beta) tau0 triangle1 triangle2 w1
          (swapABTriangleOrderedWitness w2) hneq]
        exact chosenFaceOnAB_val_eq_singleton_b_of_eq
          (beta := beta) (m := m) (U := U) triangle1 w1 edge.1.1 witness1.2 hself
      have hright :
          (((tauSwap triangle2).chosenFaceOnBoundaryEdge (beta := beta) m U
            triangle2 edge.1.1 witness2.2).1 : Finset beta) = ({w2.a} : Finset beta) := by
        dsimp [tauSwap]
        rw [replaceTwoTriangleWitnesses_right (beta := beta) tau0 triangle1 triangle2 w1
          (swapABTriangleOrderedWitness w2)]
        exact chosenFaceOnSharedAB_val_eq_singleton_a
          (beta := beta) triangle1 triangle2 w1 w2 hshared
      calc
        edgeABFace (beta := beta) triangle1 w1 = ({w1.b} : Finset beta) := by
          simp [edgeABFace_eq]
        _ =
            (((tauSwap triangle1).chosenFaceOnBoundaryEdge (beta := beta) m U
              triangle1 edge.1.1 witness1.2).1 : Finset beta) := hleft.symm
        _ = (((tauSwap triangle2).chosenFaceOnBoundaryEdge (beta := beta) m U
            triangle2 edge.1.1 witness2.2).1 : Finset beta) := hEqRaw
        _ = ({w2.a} : Finset beta) := hright
        _ = swapABEdgeFace (beta := beta) triangle2 w2 := by
          symm
          simp [swapABEdgeFace_eq]
    exact hne hEq2
  rcases sharedEdgeAB_face_ne_or_swapped_face_ne
      (beta := beta) triangle1 triangle2 w1 w2 hshared with hne1 | hne2
  · intro hboth
    exact hnotTau_of hne1 hboth.1
  · intro hboth
    exact hnotTauSwap_of hne2 hboth.2

theorem branchingEdgeConflictAt_tau_or_tauSwap_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    let tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2
    let tauSwap := replaceTwoTriangleWitnesses
      (beta := beta) tau0 triangle1 w1 triangle2 (swapABTriangleOrderedWitness w2)
    let edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared
    BranchingEdgeConflictAt (beta := beta) m U tau edge ∨
      BranchingEdgeConflictAt (beta := beta) m U tauSwap edge := by
  dsimp
  by_contra hnone
  have hNoTau :
      ¬ BranchingEdgeConflictAt
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
        (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
    intro h
    exact hnone (Or.inl h)
  have hNoTauSwap :
      ¬ BranchingEdgeConflictAt
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))
        (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
    intro h
    exact hnone (Or.inr h)
  have hpair :
      PairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
        (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
    apply Classical.not_not.mp
    intro hnotPair
    exact hNoTau
      ((branchingEdgeConflictAt_iff_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) (m := m) (U := U)
        (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
        (edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared)).2 hnotPair)
  have hpairSwap :
      PairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))
        (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
    apply Classical.not_not.mp
    intro hnotPair
    exact hNoTauSwap
      ((branchingEdgeConflictAt_iff_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) (m := m) (U := U)
        (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))
        (edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared)).2 hnotPair)
  exact not_both_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflictAt_tauSwap_of_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpair :
      PairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
        (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared)) :
    BranchingEdgeConflictAt
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2))
      (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
  apply (branchingEdgeConflictAt_iff_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
      (swapABTriangleOrderedWitness w2))
    (edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared)).2
  intro hpairSwap
  exact not_both_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflictAt_tau_of_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpairSwap :
      PairwiseBranchingEdgeChosenFaceCoherenceAt
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))
        (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared)) :
    BranchingEdgeConflictAt
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
      (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
  apply (branchingEdgeConflictAt_iff_not_pairwiseBranchingEdgeChosenFaceCoherenceAt
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
    (edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared)).2
  intro hpair
  exact not_both_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflictAt_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpair :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)) :
    BranchingEdgeConflictAt
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2))
      (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
  exact branchingEdgeConflictAt_tauSwap_of_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    (hpair (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared))

theorem branchingEdgeConflictAt_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpairSwap :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))) :
    BranchingEdgeConflictAt
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
      (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
  exact branchingEdgeConflictAt_tau_of_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    (hpairSwap (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared))

theorem branchingEdgeConflictAt_tauSwap_of_pureTriangleOverlapCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpure :
      PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)) :
    BranchingEdgeConflictAt
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2))
      (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
  exact branchingEdgeConflictAt_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ((pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
      (beta := beta) (m := m) (U := U)
      (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)).1 hpure)

theorem branchingEdgeConflictAt_tau_of_pureTriangleOverlapCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpureSwap :
      PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))) :
    BranchingEdgeConflictAt
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)
      (sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared) := by
  exact branchingEdgeConflictAt_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ((pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
      (beta := beta) (m := m) (U := U)
      (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2))).1 hpureSwap)

theorem not_both_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    let tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2
    let tauSwap := replaceTwoTriangleWitnesses
      (beta := beta) tau0 triangle1 w1 triangle2 (swapABTriangleOrderedWitness w2)
    ¬ (PairwiseBranchingChosenFaceCoherence (beta := beta) m U tau ∧
        PairwiseBranchingChosenFaceCoherence (beta := beta) m U tauSwap) := by
  dsimp
  intro hboth
  let edge := sharedABBranchingEdge (beta := beta) triangle1 triangle2 w1 w2 hneq hshared
  exact not_both_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hboth.1 edge, hboth.2 edge⟩

theorem not_both_pureTriangleOverlapCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    let tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2
    let tauSwap := replaceTwoTriangleWitnesses
      (beta := beta) tau0 triangle1 w1 triangle2 (swapABTriangleOrderedWitness w2)
    ¬ (PureTriangleOverlapCoherence (beta := beta) m U tau ∧
        PureTriangleOverlapCoherence (beta := beta) m U tauSwap) := by
  dsimp
  intro hboth
  have hpair :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) :=
    (pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
      (beta := beta) (m := m) (U := U)
      (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)).1 hboth.1
  have hpairSwap :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2)) :=
    (pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
      (beta := beta) (m := m) (U := U)
      (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2))).1 hboth.2
  exact not_both_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflict_tau_or_tauSwap_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2) :
    let tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2
    let tauSwap := replaceTwoTriangleWitnesses
      (beta := beta) tau0 triangle1 w1 triangle2 (swapABTriangleOrderedWitness w2)
    BranchingEdgeConflict (beta := beta) m U tau ∨
      BranchingEdgeConflict (beta := beta) m U tauSwap := by
  dsimp
  by_contra hnone
  have hNoTau :
      ¬ BranchingEdgeConflict
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) := by
    intro h
    exact hnone (Or.inl h)
  have hNoTauSwap :
      ¬ BranchingEdgeConflict
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2)) := by
    intro h
    exact hnone (Or.inr h)
  have hpair :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) := by
    apply Classical.not_not.mp
    intro hnotPair
    exact hNoTau
      ((branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
        (beta := beta) (m := m) (U := U)
        (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)).2 hnotPair)
  have hpairSwap :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2)) := by
    apply Classical.not_not.mp
    intro hnotPair
    exact hNoTauSwap
      ((branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
        (beta := beta) (m := m) (U := U)
        (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))).2 hnotPair)
  exact not_both_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflict_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpair :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)) :
    BranchingEdgeConflict
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2)) := by
  apply (branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
      (swapABTriangleOrderedWitness w2))).2
  intro hpairSwap
  exact not_both_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflict_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpairSwap :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))) :
    BranchingEdgeConflict
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) := by
  apply (branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)).2
  intro hpair
  exact not_both_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpair, hpairSwap⟩

theorem branchingEdgeConflict_tauSwap_of_pureTriangleOverlapCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpure :
      PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)) :
    BranchingEdgeConflict
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
        (swapABTriangleOrderedWitness w2)) := by
  apply branchingEdgeConflict_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
  exact (pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)).1 hpure

theorem branchingEdgeConflict_tau_of_pureTriangleOverlapCoherence_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpureSwap :
      PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))) :
    BranchingEdgeConflict
      (beta := beta) m U
      (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) := by
  apply branchingEdgeConflict_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
  exact (pureTriangleOverlapCoherence_iff_pairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
      (swapABTriangleOrderedWitness w2))).1 hpureSwap

theorem not_pairwiseBranchingChosenFaceCoherence_tauSwap_of_pairwise_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpair :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)) :
    ¬ PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2)) := by
  intro hpairSwap
  exact (branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
      (swapABTriangleOrderedWitness w2))).1
      (branchingEdgeConflict_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
        (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared hpair)
      hpairSwap

theorem not_pairwiseBranchingChosenFaceCoherence_tau_of_pairwise_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpairSwap :
      PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))) :
    ¬ PairwiseBranchingChosenFaceCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) := by
  intro hpair
  exact (branchingEdgeConflict_iff_not_pairwiseBranchingChosenFaceCoherence
    (beta := beta) (m := m) (U := U)
    (tau := replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)).1
      (branchingEdgeConflict_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap
        (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared hpairSwap)
      hpair

theorem not_pureTriangleOverlapCoherence_tauSwap_of_pure_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpure :
      PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2)) :
    ¬ PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2)) := by
  intro hpureSwap
  exact not_both_pureTriangleOverlapCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpure, hpureSwap⟩

theorem not_pureTriangleOverlapCoherence_tau_of_pure_of_sharedEdge_swap
    {m : Nat} {U : Cover beta alpha}
    (tau0 : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle1 triangle2 : ↑(triangleSimplices m U))
    (w1 : TriangleOrderedWitness triangle1.1)
    (w2 : TriangleOrderedWitness triangle2.1)
    (hneq : triangle1 ≠ triangle2)
    (hshared : w1.edgeAB (beta := beta) m U triangle1 = w2.edgeAB (beta := beta) m U triangle2)
    (hpureSwap :
      PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2
          (swapABTriangleOrderedWitness w2))) :
    ¬ PureTriangleOverlapCoherence
        (beta := beta) m U
        (replaceTwoTriangleWitnesses (beta := beta) tau0 triangle1 w1 triangle2 w2) := by
  intro hpure
  exact not_both_pureTriangleOverlapCoherence_of_sharedEdge_swap
    (beta := beta) (m := m) (U := U) tau0 triangle1 triangle2 w1 w2 hneq hshared
    ⟨hpure, hpureSwap⟩

end SCT.FND1
