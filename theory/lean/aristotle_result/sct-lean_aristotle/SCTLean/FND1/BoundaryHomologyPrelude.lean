import SCTLean.FND1.BoundaryChainMaps

/-!
# FND-1 Boundary Homology Prelude

This module introduces the first explicit cycle/boundary predicates for the
current finite boundary maps.

At this stage we do not define quotient homology groups yet. We only prove the
first foundational implication: every `1`-boundary is a `1`-cycle whenever the
current square-zero boundary condition holds.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A `1`-chain is a cycle if its edge boundary vanishes. -/
def IsOneCycle
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (z : OneChain (beta := beta) m U) : Prop :=
  edgeBoundaryMap (beta := beta) m U omega₁ z = 0

/-- A `1`-chain is a boundary if it is the triangle boundary of some `2`-chain. -/
def IsOneBoundary
    (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (z : OneChain (beta := beta) m U) : Prop :=
  ∃ c : TwoChain (beta := beta) m U omega₂,
    triangleBoundaryMap (beta := beta) m U omega₂ c = z

theorem oneBoundary_is_oneCycle_of_matrix_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hzero : edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneBoundary (beta := beta) m U omega₂ z) :
    IsOneCycle (beta := beta) m U omega₁ z := by
  rcases hz with ⟨c, rfl⟩
  exact edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_matrix_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hzero c

theorem oneBoundary_is_oneCycle_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneBoundary (beta := beta) m U omega₂ z) :
    IsOneCycle (beta := beta) m U omega₁ z := by
  rcases hz with ⟨c, rfl⟩
  exact edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_compatible
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat c

theorem oneBoundary_is_oneCycle_of_local_compatible
    (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneBoundary (beta := beta) m U
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) z)
    (hcompat : BoundaryLocalOrientationCompatibility (beta := beta) m U theta) :
    IsOneCycle (beta := beta) m U
      (inducedEdgeVertexOrientationDatum (beta := beta) m U theta) z := by
  rcases hz with ⟨c, rfl⟩
  exact edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_local_compatible
    (beta := beta) (m := m) (U := U) (theta := theta) hcompat c

end SCT.FND1
