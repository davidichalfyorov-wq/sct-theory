import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Data.Rat.Defs
import SCTLean.FND1.FiniteNerve

/-!
# FND-1 Nerve Diagnostics

This module builds the first normalized nerve diagnostics only after the raw
combinatorial invariants are fixed. The intended layering is:

1. `pairOverlap`, `Adjacent`, `degreeOf`, `sumDegrees` in `FiniteNerve`
2. `directedOverlapSum` here as the next raw weighted invariant
3. normalized diagnostics such as `meanDegree`, `edgeDensity`,
   and `meanEdgeOverlap`

The theorems in this file prove relabeling invariance for the weighted raw
invariant and then lift that invariance to the normalized diagnostics.
-/

open scoped BigOperators

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Number of cover cells. -/
def vertexCount : Nat := Fintype.card beta

/-- Total overlap weight over all directed adjacent ordered pairs. -/
def directedOverlapSum (m : Nat) (U : Cover beta alpha) : Nat :=
  ∑ p : {p : beta × beta // DirectedAdjacent m U p}, pairOverlap U p.1.1 p.1.2

/-- Mean degree of the directed nerve graph. -/
def meanDegree (m : Nat) (U : Cover beta alpha) : Rat :=
  let n := vertexCount (beta := beta)
  if _h : n = 0 then 0 else (sumDegrees m U : Rat) / n

/-- Directed edge density normalized by `n (n - 1)`. -/
def edgeDensity (m : Nat) (U : Cover beta alpha) : Rat :=
  let n := vertexCount (beta := beta)
  if _h : n <= 1 then 0 else (sumDegrees m U : Rat) / (n * (n - 1))

/-- Mean overlap size across directed adjacent ordered pairs. -/
def meanEdgeOverlap (m : Nat) (U : Cover beta alpha) : Rat :=
  if _h : sumDegrees m U = 0 then 0 else (directedOverlapSum m U : Rat) / sumDegrees m U

/-- Maximum overlap size across directed adjacent ordered pairs. -/
def maxEdgeOverlap (m : Nat) (U : Cover beta alpha) : Nat :=
  Finset.sup (s := (Finset.univ : Finset {p : beta × beta // DirectedAdjacent m U p}))
    (fun p => pairOverlap U p.1.1 p.1.2)

theorem directedOverlapSum_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    directedOverlapSum m (relabelCover sigma U) = directedOverlapSum m U := by
  unfold directedOverlapSum
  refine Fintype.sum_equiv (directedEdgeEquiv (sigma := sigma) (m := m) (U := U))
    (fun p => pairOverlap (relabelCover sigma U) p.1.1 p.1.2)
    (fun p => pairOverlap U p.1.1 p.1.2) ?_
  intro p
  change pairOverlap (relabelCover sigma U) p.1.1 p.1.2 =
    pairOverlap U ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p).1.1)
      ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p).1.2)
  simpa [directedEdgeEquiv] using
    (pairOverlap_relabel (sigma := sigma) (U := U)
      (i := sigma.symm p.1.1) (j := sigma.symm p.1.2))

theorem meanDegree_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    meanDegree m (relabelCover sigma U) = meanDegree m U := by
  unfold meanDegree
  rw [sumDegrees_relabel (sigma := sigma) (m := m) (U := U)]

theorem edgeDensity_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    edgeDensity m (relabelCover sigma U) = edgeDensity m U := by
  unfold edgeDensity
  rw [sumDegrees_relabel (sigma := sigma) (m := m) (U := U)]

theorem meanEdgeOverlap_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    meanEdgeOverlap m (relabelCover sigma U) = meanEdgeOverlap m U := by
  unfold meanEdgeOverlap
  rw [sumDegrees_relabel (sigma := sigma) (m := m) (U := U)]
  rw [directedOverlapSum_relabel (sigma := sigma) (m := m) (U := U)]

theorem maxEdgeOverlap_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    maxEdgeOverlap m (relabelCover sigma U) = maxEdgeOverlap m U := by
  unfold maxEdgeOverlap
  apply le_antisymm
  · refine Finset.sup_le_iff.mpr ?_
    intro p hp
    have hpair :
        pairOverlap (relabelCover sigma U) p.1.1 p.1.2 =
          pairOverlap U ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p).1.1)
            ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p).1.2) := by
      change pairOverlap (relabelCover sigma U) p.1.1 p.1.2 =
        pairOverlap U ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p).1.1)
          ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p).1.2)
      simpa [directedEdgeEquiv] using
        (pairOverlap_relabel (sigma := sigma) (U := U)
          (i := sigma.symm p.1.1) (j := sigma.symm p.1.2))
    rw [hpair]
    exact Finset.le_sup (s := (Finset.univ : Finset {p : beta × beta // DirectedAdjacent m U p}))
      (f := fun q => pairOverlap U q.1.1 q.1.2)
      (Finset.mem_univ ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U) p)))
  · refine Finset.sup_le_iff.mpr ?_
    intro p hp
    have hpair :
        pairOverlap U p.1.1 p.1.2 =
          pairOverlap (relabelCover sigma U)
            ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U)).symm p).1.1
            ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U)).symm p).1.2 := by
      change pairOverlap U p.1.1 p.1.2 =
        pairOverlap (relabelCover sigma U)
          ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U)).symm p).1.1
          ((directedEdgeEquiv (sigma := sigma) (m := m) (U := U)).symm p).1.2
      simpa [directedEdgeEquiv] using
        (pairOverlap_relabel (sigma := sigma) (U := U)
          (i := p.1.1) (j := p.1.2)).symm
    rw [hpair]
    exact Finset.le_sup
      (s := (Finset.univ : Finset {p : beta × beta // DirectedAdjacent m (relabelCover sigma U) p}))
      (f := fun q => pairOverlap (relabelCover sigma U) q.1.1 q.1.2)
      (Finset.mem_univ (((directedEdgeEquiv (sigma := sigma) (m := m) (U := U)).symm p)))

theorem NerveDiagnosticsRelabelingInvariance (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    directedOverlapSum m (relabelCover sigma U) = directedOverlapSum m U /\
      edgeDensity m (relabelCover sigma U) = edgeDensity m U /\
      meanDegree m (relabelCover sigma U) = meanDegree m U /\
      meanEdgeOverlap m (relabelCover sigma U) = meanEdgeOverlap m U /\
      maxEdgeOverlap m (relabelCover sigma U) = maxEdgeOverlap m U := by
  constructor
  · exact directedOverlapSum_relabel (sigma := sigma) (m := m) (U := U)
  constructor
  · exact edgeDensity_relabel (sigma := sigma) (m := m) (U := U)
  constructor
  · exact meanDegree_relabel (sigma := sigma) (m := m) (U := U)
  constructor
  · exact meanEdgeOverlap_relabel (sigma := sigma) (m := m) (U := U)
  · exact maxEdgeOverlap_relabel (sigma := sigma) (m := m) (U := U)

end SCT.FND1
