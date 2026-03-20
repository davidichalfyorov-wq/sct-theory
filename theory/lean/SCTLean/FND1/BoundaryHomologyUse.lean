import SCTLean.FND1.BoundaryHomologyClassEq

/-!
# FND-1 First Homology Use Layer

This module adds the first direct usage lemmas for `H₁`:

- if the difference of two cycles is a boundary, they are homologous;
- adding a current boundary does not change the class.

This is still a quotient-level use layer, not a Betti/rank layer.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem sameFirstHomologyClass_of_sub_mem_boundariesInCycles
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z w : oneCyclesSubmodule (beta := beta) m U omega₁}
    (hsub : z - w ∈ oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ z w := by
  exact (sameFirstHomologyClass_iff_sub_mem_boundariesInCycles
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) z w).2 hsub

theorem sameFirstHomologyClass_of_sub_mem_boundaries
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z w : OneChain (beta := beta) m U}
    (hz : z ∈ oneCyclesSubmodule (beta := beta) m U omega₁)
    (hw : w ∈ oneCyclesSubmodule (beta := beta) m U omega₁)
    (hsub : z - w ∈ oneBoundariesSubmodule (beta := beta) m U omega₂) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ ⟨z, hz⟩ ⟨w, hw⟩ := by
  exact (sameFirstHomologyClass_iff_sub_mem_boundaries
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
    hz hw).2 hsub

theorem sameFirstHomologyClass_add_boundary_right
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    (z : oneCyclesSubmodule (beta := beta) m U omega₁)
    (c : TwoChain (beta := beta) m U omega₂) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂
      z (z + boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c) := by
  apply sameFirstHomologyClass_of_sub_mem_boundariesInCycles
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
  have hmem :
      boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c ∈
        oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂ :=
    boundaryAsCycle_mem_oneBoundariesInCycles
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
      hle c
  have hneg :
      -boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c ∈
        oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂ :=
    Submodule.neg_mem _ hmem
  have hsub :
      z - (z + boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c) =
        -boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c := by
    abel
  rw [hsub]
  exact hneg

theorem sameFirstHomologyClass_add_boundary_left
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hle : oneBoundariesSubmodule (beta := beta) m U omega₂ ≤
      oneCyclesSubmodule (beta := beta) m U omega₁)
    (z : oneCyclesSubmodule (beta := beta) m U omega₁)
    (c : TwoChain (beta := beta) m U omega₂) :
    SameFirstHomologyClass (beta := beta) m U omega₁ omega₂
      (z + boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c) z := by
  apply sameFirstHomologyClass_of_sub_mem_boundariesInCycles
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
  have hmem :
      boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c ∈
        oneBoundariesInCyclesSubmodule (beta := beta) m U omega₁ omega₂ :=
    boundaryAsCycle_mem_oneBoundariesInCycles
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
      hle c
  have hsub :
      z + boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c - z =
        boundaryAsCycle (beta := beta) m U omega₁ omega₂ hle c := by
    abel
  rw [hsub]
  exact hmem

end SCT.FND1
