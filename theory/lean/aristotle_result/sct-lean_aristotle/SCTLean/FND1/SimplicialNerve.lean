import Mathlib.Data.Finset.Powerset
import SCTLean.FND1.TriangleLayer

/-!
# FND-1 Simplicial Nerve

This module lifts the current graph-level and triangle-level invariants to an
explicit finite simplicial object layer:

- 0-simplices as singleton subsets,
- 1-simplices as 2-subsets that form an adjacency clique,
- 2-simplices as 3-subsets that form an adjacency clique.

This is the formal bridge needed before boundary matrices, Hodge Laplacians,
and Betti-type constructions can be introduced honestly.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Generic clique predicate on a finite subset of cover indices. -/
def IsCliqueSubset (m : Nat) (U : Cover beta alpha) (s : Finset beta) : Prop :=
  ∀ ⦃i j : beta⦄, i ∈ s -> j ∈ s -> i ≠ j -> Adjacent m U i j

instance instDecidablePredCliqueSubset (m : Nat) (U : Cover beta alpha) :
    DecidablePred (IsCliqueSubset m U) := by
  intro s
  unfold IsCliqueSubset
  infer_instance

/-- Relabel a finite simplex by a permutation of the cover index set. -/
def simplexImage (sigma : Equiv.Perm beta) (s : Finset beta) : Finset beta :=
  s.image sigma

omit [Fintype beta] in
theorem simplexImage_injective (sigma : Equiv.Perm beta) :
    Function.Injective (simplexImage (beta := beta) sigma) := by
  intro s t h
  ext x
  constructor
  · intro hx
    have hx' : sigma x ∈ simplexImage (beta := beta) sigma s := Finset.mem_image.mpr ⟨x, hx, rfl⟩
    have hx'' : sigma x ∈ simplexImage (beta := beta) sigma t := by
      simpa [h] using hx'
    simpa [simplexImage] using (mem_image_perm_iff (sigma := sigma) (s := t) (x := sigma x)).mp hx''
  · intro hx
    have hx' : sigma x ∈ simplexImage (beta := beta) sigma t := Finset.mem_image.mpr ⟨x, hx, rfl⟩
    have hx'' : sigma x ∈ simplexImage (beta := beta) sigma s := by
      simpa [h] using hx'
    simpa [simplexImage] using (mem_image_perm_iff (sigma := sigma) (s := s) (x := sigma x)).mp hx''

omit [Fintype beta] in
theorem simplexImage_card (sigma : Equiv.Perm beta) (s : Finset beta) :
    (simplexImage (beta := beta) sigma s).card = s.card := by
  unfold simplexImage
  exact Finset.card_image_of_injective _ sigma.injective

omit [Fintype beta] in
theorem isCliqueSubset_image (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) {s : Finset beta} :
    IsCliqueSubset m U s -> IsCliqueSubset m (relabelCover sigma U) (simplexImage (beta := beta) sigma s) := by
  intro hs i j hi hj hneq
  have hi' : sigma.symm i ∈ s := by
    simpa [simplexImage] using (mem_image_perm_iff (sigma := sigma) (s := s) (x := i)).mp hi
  have hj' : sigma.symm j ∈ s := by
    simpa [simplexImage] using (mem_image_perm_iff (sigma := sigma) (s := s) (x := j)).mp hj
  have hneq' : sigma.symm i ≠ sigma.symm j := by
    intro h
    apply hneq
    simpa using congrArg sigma h
  have hadj :
      Adjacent m (relabelCover sigma U) (sigma (sigma.symm i)) (sigma (sigma.symm j)) := by
    exact (adjacent_relabel (sigma := sigma) (m := m) (U := U)
      (i := sigma.symm i) (j := sigma.symm j)).mpr (hs hi' hj' hneq')
  simpa using hadj

/-- 0-simplices of the finite nerve. -/
def vertexSimplices : Finset (Finset beta) :=
  (Finset.univ : Finset beta).powersetCard 1

/-- 1-simplices of the finite nerve. -/
def edgeSimplices (m : Nat) (U : Cover beta alpha) : Finset (Finset beta) :=
  ((Finset.univ : Finset beta).powersetCard 2).filter (fun s => IsCliqueSubset m U s)

/-- 2-simplices of the finite nerve. -/
def triangleSimplices (m : Nat) (U : Cover beta alpha) : Finset (Finset beta) :=
  ((Finset.univ : Finset beta).powersetCard 3).filter (fun s => IsCliqueSubset m U s)

theorem triangleSimplices_eq_triangleFinset (m : Nat) (U : Cover beta alpha) :
    triangleSimplices m U = triangleFinset m U := by
  rfl

theorem mem_vertexSimplices_image (sigma : Equiv.Perm beta) {s : Finset beta}
    (hs : s ∈ vertexSimplices (beta := beta)) :
    simplexImage (beta := beta) sigma s ∈ vertexSimplices (beta := beta) := by
  rw [vertexSimplices, Finset.mem_powersetCard] at hs ⊢
  rcases hs with ⟨_, hsCard⟩
  constructor
  · intro x hx
    simp
  · rw [simplexImage_card (beta := beta) sigma s, hsCard]

theorem mem_edgeSimplices_image (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) {s : Finset beta}
    (hs : s ∈ edgeSimplices m U) :
    simplexImage (beta := beta) sigma s ∈ edgeSimplices m (relabelCover sigma U) := by
  rcases Finset.mem_filter.mp hs with ⟨hsPow, hsClique⟩
  refine Finset.mem_filter.mpr ?_
  constructor
  · rw [Finset.mem_powersetCard] at hsPow ⊢
    rcases hsPow with ⟨_, hsCard⟩
    constructor
    · intro x hx
      simp
    · rw [simplexImage_card (beta := beta) sigma s, hsCard]
  · exact isCliqueSubset_image (sigma := sigma) (m := m) (U := U) hsClique

theorem mem_triangleSimplices_image (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) {s : Finset beta}
    (hs : s ∈ triangleSimplices m U) :
    simplexImage (beta := beta) sigma s ∈ triangleSimplices m (relabelCover sigma U) := by
  rcases Finset.mem_filter.mp hs with ⟨hsPow, hsClique⟩
  refine Finset.mem_filter.mpr ?_
  constructor
  · rw [Finset.mem_powersetCard] at hsPow ⊢
    rcases hsPow with ⟨_, hsCard⟩
    constructor
    · intro x hx
      simp
    · rw [simplexImage_card (beta := beta) sigma s, hsCard]
  · exact isCliqueSubset_image (sigma := sigma) (m := m) (U := U) hsClique

end SCT.FND1
