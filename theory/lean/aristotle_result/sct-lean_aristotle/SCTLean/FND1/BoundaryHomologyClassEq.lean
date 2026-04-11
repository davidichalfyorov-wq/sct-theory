import SCTLean.FND1.BoundaryHomologyH1

/-!
# FND-1 First Homology Class Equality

This module adds the first useful relation on the quotient-level object `H₁`:
two cycles represent the same homology class iff their difference is a boundary.

We still stop short of Betti/rank statements.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Two `1`-cycles represent the same class in the current `H₁`. -/
def SameFirstHomologyClass
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (z w : oneCyclesSubmodule (beta := beta) m U omega₁) : Prop :=
  firstHomologyMkQ (beta := beta) m U omega₁ omega₂ z =
    firstHomologyMkQ (beta := beta) m U omega₁ omega₂ w

@[refl] theorem sameFirstHomologyClass_refl
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (z : oneCyclesSubmodule (beta := beta) m U omega₁) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z z := by
  rfl

@[symm] theorem sameFirstHomologyClass_symm
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z w : oneCyclesSubmodule (beta := beta) m U omega₁}
    (h : SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z w) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ w z := by
  exact h.symm

@[trans] theorem sameFirstHomologyClass_trans
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z w q : oneCyclesSubmodule (beta := beta) m U omega₁}
    (hzw : SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z w)
    (hwq : SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ w q) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z q := by
  exact hzw.trans hwq

theorem sameFirstHomologyClass_iff_sub_mem_boundariesInCycles
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (z w : oneCyclesSubmodule (beta := beta) m U omega₁) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z w ↔
      z - w ∈ oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂ := by
  show
    (Submodule.mkQ (oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂)) z =
      (Submodule.mkQ (oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂)) w ↔
      z - w ∈ oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂
  exact Submodule.Quotient.eq _

theorem sameFirstHomologyClass_iff_sub_mem_boundaries
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z w : OneChain (beta := beta) m U}
    (hz : z ∈ oneCyclesSubmodule (beta := beta) m U omega₁)
    (hw : w ∈ oneCyclesSubmodule (beta := beta) m U omega₁) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ ⟨z, hz⟩ ⟨w, hw⟩ ↔
      z - w ∈ oneBoundariesSubmodule (beta := beta) m U omega₂ := by
  rw [sameFirstHomologyClass_iff_sub_mem_boundariesInCycles
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)]
  change
    (((⟨z, hz⟩ : oneCyclesSubmodule (beta := beta) m U omega₁) - ⟨w, hw⟩ :
      oneCyclesSubmodule (beta := beta) m U omega₁) :
      OneChain (beta := beta) m U) ∈
        oneBoundariesSubmodule (beta := beta) m U omega₂ ↔
      z - w ∈ oneBoundariesSubmodule (beta := beta) m U omega₂
  rfl

theorem sameFirstHomologyClass_zero_iff_boundary
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (z : oneCyclesSubmodule (beta := beta) m U omega₁) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z 0 ↔
      z ∈ oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂ := by
  simpa using sameFirstHomologyClass_iff_sub_mem_boundariesInCycles
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) z 0

end SCT.FND1
