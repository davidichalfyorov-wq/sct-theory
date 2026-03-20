import Mathlib.Data.Fintype.Card
import Mathlib.GroupTheory.Perm.Basic

namespace SCT.FND1

universe u

variable {alpha : Type u} [Fintype alpha]
variable (R : alpha -> alpha -> Prop) [DecidableRel R]

/-- Relabel a causal relation by a permutation of vertices. -/
def RelabelRel (pi : Equiv.Perm alpha) : alpha -> alpha -> Prop :=
  fun x y => R (pi.symm x) (pi.symm y)

instance instDecidableRelRelabelRel (pi : Equiv.Perm alpha) : DecidableRel (RelabelRel (R := R) pi) := by
  intro x y
  unfold RelabelRel
  infer_instance

/-- Number of predecessors of a vertex. -/
def pastCount (x : alpha) : Nat :=
  Fintype.card {y : alpha // R y x}

/-- Number of successors of a vertex. -/
def futureCount (x : alpha) : Nat :=
  Fintype.card {y : alpha // R x y}

/-- Order-native time surrogate given by past-future imbalance. -/
def orderTimeCoordinate (x : alpha) : Int :=
  Int.ofNat (pastCount R x) - Int.ofNat (futureCount R x)

/-- Candidate vertices whose order-time lies in the interval `[lo, hi]`. -/
def OrderTimeCandidateSet (lo hi : Int) : Set alpha :=
  fun x => lo <= orderTimeCoordinate R x /\ orderTimeCoordinate R x <= hi

def pastRelabelEquiv (pi : Equiv.Perm alpha) (x : alpha) :
    {y : alpha // RelabelRel (R := R) pi y (pi x)} ≃ {y : alpha // R y x} where
  toFun z := ⟨pi.symm z.1, by simpa [RelabelRel] using z.2⟩
  invFun z := ⟨pi z.1, by simpa [RelabelRel] using z.2⟩
  left_inv z := by
    ext
    simp
  right_inv z := by
    ext
    simp

def futureRelabelEquiv (pi : Equiv.Perm alpha) (x : alpha) :
    {y : alpha // RelabelRel (R := R) pi (pi x) y} ≃ {y : alpha // R x y} where
  toFun z := ⟨pi.symm z.1, by simpa [RelabelRel] using z.2⟩
  invFun z := ⟨pi z.1, by simpa [RelabelRel] using z.2⟩
  left_inv z := by
    ext
    simp
  right_inv z := by
    ext
    simp

theorem pastCount_relabel (pi : Equiv.Perm alpha) (x : alpha) :
    pastCount (RelabelRel (R := R) pi) (pi x) = pastCount R x := by
  unfold pastCount
  exact Fintype.card_congr (pastRelabelEquiv (R := R) pi x)

theorem futureCount_relabel (pi : Equiv.Perm alpha) (x : alpha) :
    futureCount (RelabelRel (R := R) pi) (pi x) = futureCount R x := by
  unfold futureCount
  exact Fintype.card_congr (futureRelabelEquiv (R := R) pi x)

theorem OrderTimeCoordinateEquivariance (pi : Equiv.Perm alpha) (x : alpha) :
    orderTimeCoordinate (RelabelRel (R := R) pi) (pi x) = orderTimeCoordinate R x := by
  unfold orderTimeCoordinate
  rw [pastCount_relabel (R := R) pi x, futureCount_relabel (R := R) pi x]

theorem OrderTimeCandidateSetEquivariance (pi : Equiv.Perm alpha) (lo hi : Int) :
    OrderTimeCandidateSet (R := RelabelRel (R := R) pi) lo hi =
      Set.image pi (OrderTimeCandidateSet (R := R) lo hi) := by
  ext x
  constructor
  · intro hx
    refine ⟨pi.symm x, ?_, by simp⟩
    have hcoord :
        orderTimeCoordinate (RelabelRel (R := R) pi) x =
          orderTimeCoordinate R (pi.symm x) := by
      simpa using OrderTimeCoordinateEquivariance (R := R) pi (pi.symm x)
    change lo <= orderTimeCoordinate R (pi.symm x) /\
      orderTimeCoordinate R (pi.symm x) <= hi
    rw [<- hcoord]
    simpa [OrderTimeCandidateSet] using hx
  · rintro ⟨y, hy, rfl⟩
    have hcoord :
        orderTimeCoordinate (RelabelRel (R := R) pi) (pi y) =
          orderTimeCoordinate R y := by
      simpa using OrderTimeCoordinateEquivariance (R := R) pi y
    change lo <= orderTimeCoordinate (RelabelRel (R := R) pi) (pi y) /\
      orderTimeCoordinate (RelabelRel (R := R) pi) (pi y) <= hi
    rw [hcoord]
    simpa [OrderTimeCandidateSet] using hy

end SCT.FND1
