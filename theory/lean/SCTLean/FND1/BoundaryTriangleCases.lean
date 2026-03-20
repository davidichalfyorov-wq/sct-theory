import Mathlib.Data.Finset.Card
import SCTLean.FND1.BoundaryGlobalChoiceCompatibility

/-!
# FND-1 Boundary Triangle Cases

This module isolates the finite case-splits needed before a genuine whole-nerve
compatibility theorem can be attempted.

It proves two kinds of facts:

- every row index in the edge-to-vertex boundary really is a singleton vertex,
- every codimension-one face of an ordered witness triangle is one of the three
  expected boundary edges `ab`, `ac`, or `bc`.

These are the exact combinatorial reduction lemmas needed to turn a matrix entry
of `∂₁ ∘ ∂₂` into one of the already formalized local cancellation sums.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

omit [DecidableEq beta] in
theorem vertexSimplex_eq_singleton {vertex : Finset beta}
    (hvertex : vertex ∈ vertexSimplices (beta := beta)) :
    ∃ v : beta, vertex = ({v} : Finset beta) := by
  rw [vertexSimplices, Finset.mem_powersetCard] at hvertex
  exact Finset.card_eq_one.mp hvertex.2

theorem edgeVertexRowIndex_eq_singleton
    (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    (i : EdgeVertexRowIndex (beta := beta) m U omega) :
    ∃ v : beta, i.1 = ({v} : Finset beta) := by
  exact vertexSimplex_eq_singleton (beta := beta) i.2

omit [Fintype beta] in
theorem pairCodimOneFace_cases
    {x y : beta} (hxy : x ≠ y)
    {face : Finset beta}
    (hface : face ∈ codimOneFaces (beta := beta) ({x, y} : Finset beta)) :
    face = ({x} : Finset beta) ∨ face = ({y} : Finset beta) := by
  rw [mem_codimOneFaces_iff] at hface
  rcases hface with ⟨hsub, hcard⟩
  have hcard1 : face.card = 1 := by
    calc
      face.card = ({x, y} : Finset beta).card - 1 := hcard
      _ = 1 := by simp [hxy]
  rcases Finset.card_eq_one.mp hcard1 with ⟨z, hz⟩
  subst hz
  have hz_mem : z ∈ ({x, y} : Finset beta) := hsub (by simp)
  have hz_cases : z = x ∨ z = y := by
    simpa using hz_mem
  rcases hz_cases with rfl | rfl
  · left
    simp
  · right
    simp

theorem TriangleOrderedWitness.codimOneFace_cases
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face ∈ codimOneFaces (beta := beta) triangle.1) :
    face = ({w.a, w.b} : Finset beta) ∨
      face = ({w.a, w.c} : Finset beta) ∨
      face = ({w.b, w.c} : Finset beta) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  rw [htri] at hface
  rw [mem_codimOneFaces_iff] at hface
  rcases hface with ⟨hsub, hcard⟩
  have hcard2 : face.card = 2 := by
    calc
      face.card = ({a, b, c} : Finset beta).card - 1 := hcard
      _ = 2 := by simp [hab, hac, hbc]
  rcases Finset.card_eq_two.mp hcard2 with ⟨x, y, hxy, hfaceEq⟩
  subst hfaceEq
  have hx_mem : x ∈ ({a, b, c} : Finset beta) := by
    exact hsub (by simp)
  have hy_mem : y ∈ ({a, b, c} : Finset beta) := by
    exact hsub (by simp)
  have hx_cases : x = a ∨ x = b ∨ x = c := by
    simpa using hx_mem
  have hy_cases : y = a ∨ y = b ∨ y = c := by
    simpa using hy_mem
  rcases hx_cases with rfl | rfl | rfl
  · rcases hy_cases with rfl | rfl | rfl
    · exfalso
      exact hxy rfl
    · left
      simp [Finset.pair_comm]
    · right
      left
      simp
  · rcases hy_cases with rfl | rfl | rfl
    · left
      simp [Finset.pair_comm]
    · exfalso
      exact hxy rfl
    · right
      right
      simp
  · rcases hy_cases with rfl | rfl | rfl
    · right
      left
      simp [Finset.pair_comm]
    · right
      right
      simp [Finset.pair_comm]
    · exfalso
      exact hxy rfl

theorem TriangleOrderedWitness.boundaryEdge_cases
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (edge : ↑(edgeSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1) :
    edge = w.edgeAB (beta := beta) m U triangle ∨
      edge = w.edgeAC (beta := beta) m U triangle ∨
      edge = w.edgeBC (beta := beta) m U triangle := by
  rcases TriangleOrderedWitness.codimOneFace_cases
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w) hedge with
    habEq | hacEq | hbcEq
  · left
    apply Subtype.ext
    simpa using habEq
  · right
    left
    apply Subtype.ext
    simpa using hacEq
  · right
    right
    apply Subtype.ext
    simpa using hbcEq

theorem TriangleOrderedWitness.edgeAB_face_cases
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face ∈ codimOneFaces (beta := beta) (w.edgeAB (beta := beta) m U triangle).1) :
    face = ({w.a} : Finset beta) ∨ face = ({w.b} : Finset beta) := by
  simpa using pairCodimOneFace_cases (beta := beta) w.hab hface

theorem TriangleOrderedWitness.edgeAC_face_cases
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face ∈ codimOneFaces (beta := beta) (w.edgeAC (beta := beta) m U triangle).1) :
    face = ({w.a} : Finset beta) ∨ face = ({w.c} : Finset beta) := by
  simpa using pairCodimOneFace_cases (beta := beta) w.hac hface

theorem TriangleOrderedWitness.edgeBC_face_cases
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    {face : Finset beta}
    (hface : face ∈ codimOneFaces (beta := beta) (w.edgeBC (beta := beta) m U triangle).1) :
    face = ({w.b} : Finset beta) ∨ face = ({w.c} : Finset beta) := by
  simpa using pairCodimOneFace_cases (beta := beta) w.hbc hface

end SCT.FND1
