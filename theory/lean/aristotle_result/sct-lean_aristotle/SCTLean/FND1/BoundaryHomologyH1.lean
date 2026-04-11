import Mathlib.LinearAlgebra.Quotient.Basic
import SCTLean.FND1.BoundaryHomologyStructures

/-!
# FND-1 First Homology Interface

This module introduces the quotient-level interface

`H₁ = Z₁ / (B₁ ∩ Z₁)`

for the current finite boundary data. We deliberately stop at the level of the
quotient object and its canonical class map; no Betti/rank statements are made
here.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

@[simp] theorem edgeBoundaryLinearMap_apply
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (z : OneChain (beta := beta) m U) :
    edgeBoundaryLinearMap (beta := beta) m U omega₁ z =
      edgeBoundaryMap (beta := beta) m U omega₁ z := by
  rfl

@[simp] theorem triangleBoundaryLinearMap_apply
    (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (c : TwoChain (beta := beta) m U omega₂) :
    triangleBoundaryLinearMap (beta := beta) m U omega₂ c =
      triangleBoundaryMap (beta := beta) m U omega₂ c := by
  rfl

/-- `B₁ ∩ Z₁`, viewed as a submodule of `Z₁`, via the subtype map. -/
abbrev oneBoundariesInCyclesSubmodule
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    Submodule ℤ (oneCyclesSubmodule (beta := beta) m U omega₁) :=
  Submodule.comap
    (oneCyclesSubmodule (beta := beta) m U omega₁).subtype
    (oneBoundariesSubmodule (beta := beta) m U omega₂)

/-- The first homology interface for the current finite boundary data. -/
abbrev FirstHomology
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :=
  (oneCyclesSubmodule (beta := beta) m U omega₁) ⧸
    (oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂)

/-- The canonical projection `Z₁ -> H₁`. -/
abbrev firstHomologyMkQ
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    oneCyclesSubmodule (beta := beta) m U omega₁ →ₗ[ℤ]
      FirstHomology (beta := beta) m U omega₁ omega₂ :=
  Submodule.mkQ (oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂)

/-- A `2`-boundary, regarded as an element of `Z₁`, once `B₁ ≤ Z₁` is known. -/
def boundaryAsCycle
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    (c : TwoChain (beta := beta) m U omega₂) :
    oneCyclesSubmodule (beta := beta) m U omega₁ :=
  ⟨triangleBoundaryLinearMap (beta := beta) m U omega₂ c, hle ⟨c, rfl⟩⟩

@[simp] theorem boundaryAsCycle_val
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    (c : TwoChain (beta := beta) m U omega₂) :
    ((boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c :
      oneCyclesSubmodule (beta := beta) m U omega₁) :
      OneChain (beta := beta) m U) =
      triangleBoundaryLinearMap (beta := beta) m U omega₂ c := by
  rfl

theorem boundaryAsCycle_mem_oneBoundariesInCycles
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    (c : TwoChain (beta := beta) m U omega₂) :
    boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c ∈
      oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂ := by
  show (((boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c :
    oneCyclesSubmodule (beta := beta) m U omega₁) :
    OneChain (beta := beta) m U)) ∈
      oneBoundariesSubmodule (beta := beta) m U omega₂
  exact ⟨c, rfl⟩

theorem boundaryAsCycle_class_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    (c : TwoChain (beta := beta) m U omega₂) :
    firstHomologyMkQ (beta := beta) m U omega₁ omega₂
      (boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c) = 0 := by
  exact (Submodule.Quotient.mk_eq_zero
    (oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂)).2 <|
      boundaryAsCycle_mem_oneBoundariesInCycles
        (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
        hle c

theorem oneBoundary_class_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneBoundary (beta := beta) m U omega₂ z) :
    firstHomologyMkQ (beta := beta) m U omega₁ omega₂
      ⟨z, hle ((mem_oneBoundariesSubmodule_iff
        (beta := beta) (m := m) (U := U) (omega₂ := omega₂) z).2 hz)⟩ = 0 := by
  rcases hz with ⟨c, rfl⟩
  exact (Submodule.Quotient.mk_eq_zero
    (oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂)).2 <| by
      show triangleBoundaryMap (beta := beta) m U omega₂ c ∈
        oneBoundariesSubmodule (beta := beta) m U omega₂
      exact ⟨c, rfl⟩

theorem boundaryAsCycle_class_eq_zero_of_matrix_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hzero : edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0)
    (c : TwoChain (beta := beta) m U omega₂) :
    firstHomologyMkQ (beta := beta) m U omega₁ omega₂
      (boundaryAsCycle (beta := beta) m U omega₁ omega₂
        (oneBoundaries_le_oneCycles_of_matrix_zero
          (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hzero)
        c) = 0 := by
  exact boundaryAsCycle_class_eq_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
    (oneBoundaries_le_oneCycles_of_matrix_zero
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hzero)
    c

theorem boundaryAsCycle_class_eq_zero_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂)
    (c : TwoChain (beta := beta) m U omega₂) :
    firstHomologyMkQ (beta := beta) m U omega₁ omega₂
      (boundaryAsCycle (beta := beta) m U omega₁ omega₂
        (oneBoundaries_le_oneCycles_of_compatible
          (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat)
        c) = 0 := by
  exact boundaryAsCycle_class_eq_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
    (oneBoundaries_le_oneCycles_of_compatible
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat)
    c

theorem oneBoundary_class_eq_zero_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneBoundary (beta := beta) m U omega₂ z) :
    firstHomologyMkQ (beta := beta) m U omega₁ omega₂
      ⟨z, (oneBoundaries_le_oneCycles_of_compatible
        (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat)
        ((mem_oneBoundariesSubmodule_iff
          (beta := beta) (m := m) (U := U) (omega₂ := omega₂) z).2 hz)⟩ = 0 := by
  exact oneBoundary_class_eq_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
    (oneBoundaries_le_oneCycles_of_compatible
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat)
    hz

end SCT.FND1
