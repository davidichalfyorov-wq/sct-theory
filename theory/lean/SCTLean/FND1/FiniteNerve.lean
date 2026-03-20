import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Max
import Mathlib.Data.Fintype.Prod
import Mathlib.GroupTheory.Perm.Basic

/-!
# FND-1 Finite Nerve Core

This module formalizes the finite combinatorial invariants behind the current
`A1` nerve diagnostics: overlap size, adjacency, local degree, and total
directed adjacency count. The main goal is to show that these quantities are
invariant under relabeling of cover indices.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]

/-- A finite cover indexed by `beta` with cells in the finite universe `alpha`. -/
abbrev Cover (beta : Type v) (alpha : Type u) [DecidableEq alpha] :=
  beta -> Finset alpha

variable [DecidableEq alpha]

instance instFintypeProdBeta : Fintype (beta × beta) := inferInstance

/-- Relabel a cover by a permutation of its index set. -/
def relabelCover (sigma : Equiv.Perm beta) (U : Cover beta alpha) : Cover beta alpha :=
  fun b => U (sigma.symm b)

/-- Cardinality of the overlap between two cover cells. -/
def pairOverlap (U : Cover beta alpha) (i j : beta) : Nat :=
  ((U i) ∩ (U j)).card

/-- Two cover cells are adjacent when they are distinct and overlap enough. -/
def Adjacent (minOverlap : Nat) (U : Cover beta alpha) (i j : beta) : Prop :=
  i ≠ j ∧ minOverlap ≤ pairOverlap U i j

instance instDecidablePredAdjacentLeft (minOverlap : Nat) (U : Cover beta alpha) (i : beta) :
    DecidablePred (fun j => Adjacent minOverlap U i j) := by
  intro j
  unfold Adjacent
  infer_instance

instance instDecidablePredAdjacentProd (minOverlap : Nat) (U : Cover beta alpha) :
    DecidablePred (fun p : beta × beta => Adjacent minOverlap U p.1 p.2) := by
  intro p
  unfold Adjacent
  infer_instance

/-- Directed adjacency predicate on ordered index pairs. -/
def DirectedAdjacent (minOverlap : Nat) (U : Cover beta alpha) (p : beta × beta) : Prop :=
  Adjacent minOverlap U p.1 p.2

instance instDecidablePredDirectedAdjacent (minOverlap : Nat) (U : Cover beta alpha) :
    DecidablePred (DirectedAdjacent minOverlap U) := by
  intro p
  unfold DirectedAdjacent
  infer_instance

/-- Number of neighbors of a cover cell. -/
def degreeOf (minOverlap : Nat) (U : Cover beta alpha) (i : beta) : Nat :=
  Fintype.card {j : beta // Adjacent minOverlap U i j}

/-- Total number of directed adjacency incidences. -/
def sumDegrees (minOverlap : Nat) (U : Cover beta alpha) : Nat :=
  Fintype.card {p : beta × beta // DirectedAdjacent minOverlap U p}

/-- Maximum directed degree among cover cells. -/
def maxDegree (minOverlap : Nat) (U : Cover beta alpha) : Nat :=
  Finset.univ.sup (fun i => degreeOf minOverlap U i)

omit [Fintype beta] [DecidableEq beta] in
theorem pairOverlap_relabel (sigma : Equiv.Perm beta) (U : Cover beta alpha) (i j : beta) :
    pairOverlap (relabelCover sigma U) (sigma i) (sigma j) = pairOverlap U i j := by
  unfold pairOverlap relabelCover
  simp

omit [Fintype beta] [DecidableEq beta] in
theorem pairOverlap_symm (U : Cover beta alpha) (i j : beta) :
    pairOverlap U i j = pairOverlap U j i := by
  unfold pairOverlap
  rw [Finset.inter_comm]

omit [Fintype beta] [DecidableEq beta] in
theorem adjacent_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) (i j : beta) :
    Adjacent m (relabelCover sigma U) (sigma i) (sigma j) ↔ Adjacent m U i j := by
  unfold Adjacent
  rw [pairOverlap_relabel (sigma := sigma) (U := U) (i := i) (j := j)]
  constructor <;> rintro ⟨hneq, hcard⟩
  · constructor
    · exact fun hij => hneq (by simpa using hij)
    · exact hcard
  · constructor
    · exact fun hij => hneq (sigma.injective <| by simpa using hij)
    · exact hcard

omit [Fintype beta] [DecidableEq beta] in
theorem adjacent_symm (m : Nat) (U : Cover beta alpha) (i j : beta) :
    Adjacent m U i j ↔ Adjacent m U j i := by
  unfold Adjacent
  rw [pairOverlap_symm (U := U) (i := i) (j := j)]
  constructor <;> rintro ⟨hneq, hcard⟩
  · constructor
    · exact Ne.symm hneq
    · exact hcard
  · constructor
    · exact Ne.symm hneq
    · exact hcard

def adjacentFiberEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) (i : beta) :
    {j : beta // Adjacent m (relabelCover sigma U) (sigma i) j} ≃
      {j : beta // Adjacent m U i j} where
  toFun z := by
    refine ⟨sigma.symm z.1, ?_⟩
    have hz : Adjacent m (relabelCover sigma U) (sigma i) (sigma (sigma.symm z.1)) := by
      simpa using z.2
    simpa using (adjacent_relabel (sigma := sigma) (m := m) (U := U) (i := i) (j := sigma.symm z.1)).mp hz
  invFun z := by
    refine ⟨sigma z.1, ?_⟩
    simpa using (adjacent_relabel (sigma := sigma) (m := m) (U := U) (i := i) (j := z.1)).mpr z.2
  left_inv z := by
    ext
    simp
  right_inv z := by
    ext
    simp

theorem degreeOf_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) (i : beta) :
    degreeOf m (relabelCover sigma U) (sigma i) = degreeOf m U i := by
  unfold degreeOf
  exact Fintype.card_congr (adjacentFiberEquiv (sigma := sigma) (m := m) (U := U) (i := i))

def directedEdgeEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    {p : beta × beta // DirectedAdjacent m (relabelCover sigma U) p} ≃
      {p : beta × beta // DirectedAdjacent m U p} where
  toFun z := by
    refine ⟨(sigma.symm z.1.1, sigma.symm z.1.2), ?_⟩
    have hz :
        Adjacent m (relabelCover sigma U) (sigma (sigma.symm z.1.1)) (sigma (sigma.symm z.1.2)) := by
      simpa using z.2
    simpa using
      (adjacent_relabel (sigma := sigma) (m := m) (U := U)
        (i := sigma.symm z.1.1) (j := sigma.symm z.1.2)).mp hz
  invFun z := by
    refine ⟨(sigma z.1.1, sigma z.1.2), ?_⟩
    simpa using
      (adjacent_relabel (sigma := sigma) (m := m) (U := U) (i := z.1.1) (j := z.1.2)).mpr z.2
  left_inv z := by
    ext <;> simp
  right_inv z := by
    ext <;> simp

theorem sumDegrees_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    sumDegrees m (relabelCover sigma U) = sumDegrees m U := by
  unfold sumDegrees
  exact Fintype.card_congr (directedEdgeEquiv (sigma := sigma) (m := m) (U := U))

theorem maxDegree_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    maxDegree m (relabelCover sigma U) = maxDegree m U := by
  unfold maxDegree
  apply le_antisymm
  · refine Finset.sup_le ?_
    intro i _
    have hdeg : degreeOf m (relabelCover sigma U) i = degreeOf m U (sigma.symm i) := by
      simpa using degreeOf_relabel (sigma := sigma) (m := m) (U := U) (i := sigma.symm i)
    rw [hdeg]
    exact Finset.le_sup (Finset.mem_univ (sigma.symm i))
  · refine Finset.sup_le ?_
    intro i _
    have hdeg : degreeOf m U i = degreeOf m (relabelCover sigma U) (sigma i) := by
      simpa using (degreeOf_relabel (sigma := sigma) (m := m) (U := U) (i := i)).symm
    rw [hdeg]
    exact Finset.le_sup (Finset.mem_univ (sigma i))

end SCT.FND1
