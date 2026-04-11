import Mathlib.Data.Finset.Powerset
import SCTLean.FND1.FiniteNerve

/-!
# FND-1 Triangle Layer

This module adds the first genuinely order-free higher-order nerve invariant:
triangle counting. Triangles are defined as 3-element subsets of the cover
index set that form an undirected clique under the adjacency relation.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- A 3-element subset is triangular if every distinct pair inside it is adjacent. -/
def IsTriangleSubset (m : Nat) (U : Cover beta alpha) (s : Finset beta) : Prop :=
  ∀ ⦃i j : beta⦄, i ∈ s -> j ∈ s -> i ≠ j -> Adjacent m U i j

instance instDecidablePredTriangleSubset (m : Nat) (U : Cover beta alpha) :
    DecidablePred (IsTriangleSubset m U) := by
  intro s
  unfold IsTriangleSubset
  infer_instance

omit [Fintype beta] in
theorem mem_image_perm_iff (sigma : Equiv.Perm beta) {s : Finset beta} {x : beta} :
    x ∈ s.image sigma ↔ sigma.symm x ∈ s := by
  constructor
  · intro hx
    rw [Finset.mem_image] at hx
    rcases hx with ⟨y, hy, rfl⟩
    simpa
  · intro hx
    exact Finset.mem_image.mpr ⟨sigma.symm x, hx, by simp⟩

omit [Fintype beta] in
theorem image_symm_image (sigma : Equiv.Perm beta) (s : Finset beta) :
    (s.image sigma.symm).image sigma = s := by
  ext x
  constructor
  · intro hx
    have hx' : sigma.symm x ∈ s.image sigma.symm := by
      exact (mem_image_perm_iff (sigma := sigma) (s := s.image sigma.symm) (x := x)).mp hx
    have hx'' : ((sigma.symm).symm (sigma.symm x)) ∈ s := by
      exact (mem_image_perm_iff (sigma := sigma.symm) (s := s) (x := sigma.symm x)).mp hx'
    simpa using hx''
  · intro hx
    apply (mem_image_perm_iff (sigma := sigma) (s := s.image sigma.symm) (x := x)).mpr
    have hx' : (sigma.symm).symm (sigma.symm x) ∈ s := by
      simpa using hx
    exact (mem_image_perm_iff (sigma := sigma.symm) (s := s) (x := sigma.symm x)).mpr hx'

omit [Fintype beta] in
theorem isTriangleSubset_image (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) {s : Finset beta} :
    IsTriangleSubset m U s -> IsTriangleSubset m (relabelCover sigma U) (s.image sigma) := by
  intro hs i j hi hj hneq
  have hi' : sigma.symm i ∈ s := (mem_image_perm_iff (sigma := sigma) (s := s) (x := i)).mp hi
  have hj' : sigma.symm j ∈ s := (mem_image_perm_iff (sigma := sigma) (s := s) (x := j)).mp hj
  have hneq' : sigma.symm i ≠ sigma.symm j := by
    intro h
    apply hneq
    simpa using congrArg sigma h
  have hadj :
      Adjacent m (relabelCover sigma U) (sigma (sigma.symm i)) (sigma (sigma.symm j)) := by
    exact (adjacent_relabel (sigma := sigma) (m := m) (U := U)
      (i := sigma.symm i) (j := sigma.symm j)).mpr (hs hi' hj' hneq')
  simpa using hadj

/-- Finset of triangular 3-subsets of the cover index set. -/
def triangleFinset (m : Nat) (U : Cover beta alpha) : Finset (Finset beta) :=
  ((Finset.univ : Finset beta).powersetCard 3).filter (fun s => IsTriangleSubset m U s)

/-- Number of undirected triangles in the finite nerve. -/
def triangleCount (m : Nat) (U : Cover beta alpha) : Nat :=
  (triangleFinset m U).card

theorem mem_triangleFinset_image (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) {s : Finset beta}
    (hs : s ∈ triangleFinset m U) :
    s.image sigma ∈ triangleFinset m (relabelCover sigma U) := by
  rcases Finset.mem_filter.mp hs with ⟨hsPow, hsTri⟩
  refine Finset.mem_filter.mpr ?_
  constructor
  · rw [Finset.mem_powersetCard] at hsPow ⊢
    rcases hsPow with ⟨_, hsCard⟩
    constructor
    · intro x hx
      simp
    · rw [Finset.card_image_of_injective _ sigma.injective, hsCard]
  · exact isTriangleSubset_image (sigma := sigma) (m := m) (U := U) hsTri

theorem triangleCount_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    triangleCount m (relabelCover sigma U) = triangleCount m U := by
  unfold triangleCount triangleFinset
  refine Eq.symm <| Finset.card_bij
    (fun s _hs => s.image sigma)
    (fun s hs => mem_triangleFinset_image (sigma := sigma) (m := m) (U := U) hs)
    ?_ ?_
  · intro s hs t ht hEq
    ext x
    constructor
    · intro hx
      have hx' : sigma x ∈ s.image sigma := Finset.mem_image.mpr ⟨x, hx, rfl⟩
      have hx'' : sigma x ∈ t.image sigma := by simpa [hEq] using hx'
      simpa using (mem_image_perm_iff (sigma := sigma) (s := t) (x := sigma x)).mp hx''
    · intro hx
      have hx' : sigma x ∈ t.image sigma := Finset.mem_image.mpr ⟨x, hx, rfl⟩
      have hx'' : sigma x ∈ s.image sigma := by simpa [hEq] using hx'
      simpa using (mem_image_perm_iff (sigma := sigma) (s := s) (x := sigma x)).mp hx''
  · intro t ht
    refine ⟨t.image sigma.symm, ?_, ?_⟩
    · rcases Finset.mem_filter.mp ht with ⟨htPow, htTri⟩
      refine Finset.mem_filter.mpr ?_
      constructor
      · rw [Finset.mem_powersetCard] at htPow ⊢
        rcases htPow with ⟨_, htCard⟩
        constructor
        · intro x hx
          simp
        · rw [Finset.card_image_of_injective _ sigma.symm.injective, htCard]
      · have hrel :
            relabelCover sigma.symm (relabelCover sigma U) = U := by
          funext b
          simp [relabelCover]
        simpa [hrel] using
          (isTriangleSubset_image (sigma := sigma.symm) (m := m) (U := relabelCover sigma U) htTri)
    · exact image_symm_image (sigma := sigma) t

end SCT.FND1
