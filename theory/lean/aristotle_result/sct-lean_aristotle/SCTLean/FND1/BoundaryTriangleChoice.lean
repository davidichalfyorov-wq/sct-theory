import Mathlib.Data.Finset.Card
import SCTLean.FND1.BoundaryOrderedTriangle

/-!
# FND-1 Boundary Triangle Choice

This module is the triangle-level analogue of the edge endpoint-choice layer.
It does not produce a canonical ordering on triangles. It records that every
actual 2-simplex admits a noncanonical ordered witness.
-/
namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

structure TriangleOrderedWitness (triangle : Finset beta) where
  a : beta
  b : beta
  c : beta
  hab : a ≠ b
  hac : a ≠ c
  hbc : b ≠ c
  triangle_eq : triangle = ({a, b, c} : Finset beta)

theorem triangleOrderedWitness_exists
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U)) :
    Nonempty (TriangleOrderedWitness triangle.1) := by
  rcases Finset.mem_filter.mp triangle.2 with ⟨hpow, _⟩
  rw [Finset.mem_powersetCard] at hpow
  rcases hpow with ⟨_, hcard⟩
  rcases (Finset.card_eq_three.mp hcard) with ⟨a, b, c, hab, hac, hbc, htri⟩
  exact ⟨⟨a, b, c, hab, hac, hbc, htri⟩⟩

/-- A noncanonical ordered witness for each triangle simplex. -/
abbrev TriangleOrderedChoiceDatum (m : Nat) (U : Cover beta alpha) :=
  ∀ triangle : ↑(triangleSimplices m U), TriangleOrderedWitness triangle.1

/-- A global triangle ordering datum exists noncanonically by classical choice. -/
noncomputable def someTriangleOrderedChoiceDatum
    (m : Nat) (U : Cover beta alpha) :
    TriangleOrderedChoiceDatum (beta := beta) m U :=
  fun triangle => Classical.choice
    (triangleOrderedWitness_exists (beta := beta) (m := m) (U := U) triangle)

theorem triangleOrderedChoiceDatum_exists
    (m : Nat) (U : Cover beta alpha) :
    Nonempty (TriangleOrderedChoiceDatum (beta := beta) m U) := by
  exact ⟨someTriangleOrderedChoiceDatum (beta := beta) m U⟩

def TriangleOrderedWitness.localTriangleOrientation
    {triangle : Finset beta} (w : TriangleOrderedWitness triangle) :
    LocalFaceOrientationDatum (beta := beta) triangle := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  subst htri
  exact orderedTriangleOrientation (beta := beta) hab hac hbc

omit [Fintype beta] in
theorem TriangleOrderedWitness.local_boundary_square_zero
    {triangle : Finset beta} (w : TriangleOrderedWitness triangle) :
    orderedTriangleVertexSumAtA (beta := beta) w.hab w.hac w.hbc = 0 ∧
    orderedTriangleVertexSumAtB (beta := beta) w.hab w.hac w.hbc = 0 ∧
    orderedTriangleVertexSumAtC (beta := beta) w.hab w.hac w.hbc = 0 := by
  exact orderedTriangle_local_boundary_square_zero
    (beta := beta) w.hab w.hac w.hbc

end SCT.FND1
