import Mathlib.Data.Finset.Powerset
import SCTLean.FND1.SimplicialNerve

/-!
# FND-1 Boundary Support

This module introduces the codimension-one face layer needed before any honest
boundary matrix can be defined. At this stage we deliberately stay unoriented:

- no hidden ordering on simplex vertices,
- no sign conventions,
- only the finite face relation and its compatibility with the current simplex
  families.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- `lower` is a codimension-one face of `upper`. The cardinality condition
already forces strict containment, so subset + one-step cardinal gap is the
cleanest formulation for the current finite layer. -/
def IsCodimOneFace (lower upper : Finset beta) : Prop :=
  lower ⊆ upper ∧ lower.card + 1 = upper.card

instance instDecidablePredCodimOneFaceLeft (upper : Finset beta) :
    DecidablePred (fun lower => IsCodimOneFace lower upper) := by
  intro lower
  unfold IsCodimOneFace
  infer_instance

omit [Fintype beta] in
theorem isCodimOneFace_image (sigma : Equiv.Perm beta) {lower upper : Finset beta} :
    IsCodimOneFace lower upper ->
      IsCodimOneFace (simplexImage (beta := beta) sigma lower)
        (simplexImage (beta := beta) sigma upper) := by
  intro h
  rcases h with ⟨hsub, hcard⟩
  constructor
  · intro x hx
    have hx' : sigma.symm x ∈ lower := by
      simpa [simplexImage] using
        (mem_image_perm_iff (sigma := sigma) (s := lower) (x := x)).mp hx
    exact (mem_image_perm_iff (sigma := sigma) (s := upper) (x := x)).mpr (hsub hx')
  · rw [simplexImage_card (beta := beta) sigma lower, simplexImage_card (beta := beta) sigma upper]
    exact hcard

/-- All codimension-one faces of a simplex. -/
def codimOneFaces (simplex : Finset beta) : Finset (Finset beta) :=
  simplex.powersetCard (simplex.card - 1)

omit [Fintype beta] [DecidableEq beta] in
theorem mem_codimOneFaces_iff {simplex face : Finset beta} :
    face ∈ codimOneFaces (beta := beta) simplex ↔
      face ⊆ simplex ∧ face.card = simplex.card - 1 := by
  unfold codimOneFaces
  rw [Finset.mem_powersetCard]

omit [Fintype beta] [DecidableEq beta] in
theorem card_codimOneFaces (simplex : Finset beta) :
    (codimOneFaces (beta := beta) simplex).card = Nat.choose simplex.card (simplex.card - 1) := by
  unfold codimOneFaces
  rw [Finset.card_powersetCard]

omit [Fintype beta] in
theorem codimOneFaces_image (sigma : Equiv.Perm beta) (simplex : Finset beta) :
    (codimOneFaces (beta := beta) simplex).image (simplexImage (beta := beta) sigma) =
      codimOneFaces (beta := beta) (simplexImage (beta := beta) sigma simplex) := by
  ext face
  constructor
  · intro hface
    rw [Finset.mem_image] at hface
    rcases hface with ⟨source, hsource, rfl⟩
    rw [mem_codimOneFaces_iff] at hsource ⊢
    rcases hsource with ⟨hsub, hcard⟩
    constructor
    · intro x hx
      have hx' : sigma.symm x ∈ source := by
        simpa [simplexImage] using
          (mem_image_perm_iff (sigma := sigma) (s := source) (x := x)).mp hx
      exact (mem_image_perm_iff (sigma := sigma) (s := simplex) (x := x)).mpr (hsub hx')
    · rw [simplexImage_card (beta := beta) sigma source, simplexImage_card (beta := beta) sigma simplex]
      exact hcard
  · intro hface
    rw [mem_codimOneFaces_iff] at hface
    rcases hface with ⟨hsub, hcard⟩
    refine Finset.mem_image.mpr ?_
    refine ⟨simplexImage (beta := beta) sigma.symm face, ?_, ?_⟩
    · rw [mem_codimOneFaces_iff]
      constructor
      · intro x hx
        have hxFace : sigma x ∈ face := by
          simpa [simplexImage] using
            (mem_image_perm_iff (sigma := sigma.symm) (s := face) (x := x)).mp hx
        have hxImg : sigma x ∈ simplexImage (beta := beta) sigma simplex := hsub hxFace
        simpa [simplexImage] using
          (mem_image_perm_iff (sigma := sigma) (s := simplex) (x := sigma x)).mp hxImg
      · calc
          (simplexImage (beta := beta) sigma.symm face).card
              = face.card := by rw [simplexImage_card (beta := beta) sigma.symm face]
          _ = (simplexImage (beta := beta) sigma simplex).card - 1 := hcard
          _ = simplex.card - 1 := by rw [simplexImage_card (beta := beta) sigma simplex]
    · simpa [simplexImage] using image_symm_image (sigma := sigma) face

omit [Fintype beta] [DecidableEq beta] in
theorem clique_subset {m : Nat} {U : Cover beta alpha} {small large : Finset beta} :
    small ⊆ large -> IsCliqueSubset m U large -> IsCliqueSubset m U small := by
  intro hsub hlarge i j hi hj hneq
  exact hlarge (hsub hi) (hsub hj) hneq

theorem edge_boundary_faces_are_vertices (m : Nat) (U : Cover beta alpha) {edge : Finset beta}
    (hedge : edge ∈ edgeSimplices m U) :
    ∀ {face : Finset beta}, face ∈ codimOneFaces (beta := beta) edge ->
      face ∈ vertexSimplices (beta := beta) := by
  intro face hface
  rcases Finset.mem_filter.mp hedge with ⟨hedgePow, _⟩
  rw [Finset.mem_powersetCard] at hedgePow
  rcases hedgePow with ⟨hedgeSub, hedgeCard⟩
  rw [mem_codimOneFaces_iff] at hface
  rcases hface with ⟨hfaceSub, hfaceCard⟩
  rw [vertexSimplices, Finset.mem_powersetCard]
  constructor
  · intro x hx
    exact hedgeSub (hfaceSub hx)
  · simpa [hedgeCard] using hfaceCard

theorem edge_boundary_face_count (m : Nat) (U : Cover beta alpha) {edge : Finset beta}
    (hedge : edge ∈ edgeSimplices m U) :
    (codimOneFaces (beta := beta) edge).card = 2 := by
  rcases Finset.mem_filter.mp hedge with ⟨hedgePow, _⟩
  rw [Finset.mem_powersetCard] at hedgePow
  rcases hedgePow with ⟨_, hedgeCard⟩
  rw [card_codimOneFaces (beta := beta) edge, hedgeCard]
  simp

theorem triangle_boundary_faces_are_edges (m : Nat) (U : Cover beta alpha) {triangle : Finset beta}
    (htriangle : triangle ∈ triangleSimplices m U) :
    ∀ {face : Finset beta}, face ∈ codimOneFaces (beta := beta) triangle ->
      face ∈ edgeSimplices m U := by
  intro face hface
  rcases Finset.mem_filter.mp htriangle with ⟨htriPow, htriClique⟩
  rw [Finset.mem_powersetCard] at htriPow
  rcases htriPow with ⟨htriSub, htriCard⟩
  rw [mem_codimOneFaces_iff] at hface
  rcases hface with ⟨hfaceSub, hfaceCard⟩
  refine Finset.mem_filter.mpr ?_
  constructor
  · rw [Finset.mem_powersetCard]
    constructor
    · intro x hx
      exact htriSub (hfaceSub hx)
    · simpa [htriCard] using hfaceCard
  · exact clique_subset (m := m) (U := U) hfaceSub htriClique

theorem triangle_boundary_face_count (m : Nat) (U : Cover beta alpha) {triangle : Finset beta}
    (htriangle : triangle ∈ triangleSimplices m U) :
    (codimOneFaces (beta := beta) triangle).card = 3 := by
  rcases Finset.mem_filter.mp htriangle with ⟨htriPow, _⟩
  rw [Finset.mem_powersetCard] at htriPow
  rcases htriPow with ⟨_, htriCard⟩
  rw [card_codimOneFaces (beta := beta) triangle, htriCard]
  simp

end SCT.FND1
