import SCTLean.FND1.BoundaryHomologyPrelude

/-!
# FND-1 Boundary Homology Structures

This module upgrades the current cycle/boundary predicates to the first
structural homology layer:

- `Z₁` as a kernel submodule,
- `B₁` as a range submodule,
- and the inclusion `B₁ ≤ Z₁` under the square-zero condition.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- The linear boundary map `∂₁ : C₁ -> C₀`. -/
abbrev edgeBoundaryLinearMap (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :
    OneChain (beta := beta) m U →ₗ[ℤ]
      ZeroChain (beta := beta) m U omega₁ :=
  (edgeVertexBoundaryMatrix (beta := beta) m U omega₁).mulVecLin

/-- The linear boundary map `∂₂ : C₂ -> C₁`. -/
abbrev triangleBoundaryLinearMap (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    TwoChain (beta := beta) m U omega₂ →ₗ[ℤ]
      OneChain (beta := beta) m U :=
  (triangleEdgeBoundaryMatrix (beta := beta) m U omega₂).mulVecLin

/-- `Z₁ = ker ∂₁`. -/
abbrev oneCyclesSubmodule (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :
    Submodule ℤ (OneChain (beta := beta) m U) :=
  LinearMap.ker (edgeBoundaryLinearMap (beta := beta) m U omega₁)

/-- `B₁ = im ∂₂`. -/
abbrev oneBoundariesSubmodule (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    Submodule ℤ (OneChain (beta := beta) m U) :=
  LinearMap.range (triangleBoundaryLinearMap (beta := beta) m U omega₂)

theorem mem_oneCyclesSubmodule_iff
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (z : OneChain (beta := beta) m U) :
    z ∈ oneCyclesSubmodule (beta := beta) m U omega₁ ↔
      IsOneCycle (beta := beta) m U omega₁ z := by
  rfl

theorem mem_oneBoundariesSubmodule_iff
    (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (z : OneChain (beta := beta) m U) :
    z ∈ oneBoundariesSubmodule (beta := beta) m U omega₂ ↔
      IsOneBoundary (beta := beta) m U omega₂ z := by
  constructor
  · intro hz
    rcases hz with ⟨c, rfl⟩
    exact ⟨c, rfl⟩
  · intro hz
    rcases hz with ⟨c, rfl⟩
    exact ⟨c, rfl⟩

theorem oneBoundaries_le_oneCycles_of_matrix_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hzero : edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0) :
    oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁ := by
  intro z hz
  exact (mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := U) (omega₁ := omega₁) z).2 <|
    oneBoundary_is_oneCycle_of_matrix_zero
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
      hzero <|
        (mem_oneBoundariesSubmodule_iff (beta := beta) (m := m) (U := U) (omega₂ := omega₂) z).1 hz

theorem oneBoundaries_le_oneCycles_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂) :
    oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁ := by
  exact oneBoundaries_le_oneCycles_of_matrix_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
    ((boundaryOrientationCompatibility_iff_matrix_eq_zero
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)).mp hcompat)

end SCT.FND1
