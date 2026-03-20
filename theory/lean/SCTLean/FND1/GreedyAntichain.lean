import Mathlib.Data.List.Pairwise

/-!
# FND-1 Greedy Antichain Core

This module isolates the first finite combinatorial theorem for the FND-1
program: a greedy antichain selector returns a subset of the input candidates
and the output is pairwise incomparable.

For the first formal target we deliberately model the greedy selector via
`List.pwFilter`, since Mathlib already proves the needed subset and pairwise
properties for pairwise-preserving filters.
-/

namespace SCT.FND1

universe u

variable {alpha : Type u}
variable (R : alpha -> alpha -> Prop)

/-- Two vertices are comparable if either one precedes the other. -/
def Comparable (x y : alpha) : Prop :=
  R x y \/ R y x

/-- Incomparability is the negation of comparability. -/
def Incomparable (x y : alpha) : Prop :=
  ¬ Comparable R x y

instance instDecidableRelIncomparable [DecidableRel R] : DecidableRel (Incomparable R) := by
  intro x y
  unfold Incomparable Comparable
  infer_instance

/-- Greedy antichain selection from an ordered candidate list. -/
def greedyAntichainFromCandidates [DecidableRel R] (candidates : List alpha) : List alpha :=
  List.pwFilter (Incomparable R) candidates

theorem greedy_subset_candidates [DecidableRel R] (candidates : List alpha) :
    greedyAntichainFromCandidates R candidates ⊆ candidates := by
  simpa [greedyAntichainFromCandidates] using
    (List.pwFilter_subset (R := Incomparable R) candidates)

theorem greedy_pairwise_incomparable [DecidableRel R] (candidates : List alpha) :
    List.Pairwise (Incomparable R) (greedyAntichainFromCandidates R candidates) := by
  simpa [greedyAntichainFromCandidates] using
    (List.pairwise_pwFilter (R := Incomparable R) candidates)

/-- First formal FND-1 theorem: the greedy selector stays inside the candidate
list and returns pairwise incomparable vertices. -/
theorem GreedyAntichainCorrectness [DecidableRel R] (candidates : List alpha) :
    (greedyAntichainFromCandidates R candidates ⊆ candidates) /\
      List.Pairwise (Incomparable R) (greedyAntichainFromCandidates R candidates) := by
  constructor
  · exact greedy_subset_candidates (R := R) candidates
  · exact greedy_pairwise_incomparable (R := R) candidates

end SCT.FND1
