import SCTLean.FND1.BoundarySupport

/-!
# FND-1 Boundary Incidence

This module packages the already formalized codimension-one face relation into
explicit unsigned incidence supports. These supports are the honest precursor to
any later boundary matrix:

- no orientation,
- no signs,
- just the finite set of nonzero support pairs `(face, simplex)`.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Embed a codimension-one face into an unsigned incidence pair over a fixed simplex. -/
def faceToSimplexEmbedding (simplex : Finset beta) : Finset beta ↪ (Finset beta × Finset beta) where
  toFun face := (face, simplex)
  inj' := by
    intro a b hab
    simpa using congrArg Prod.fst hab

/-- Relabel an unsigned incidence pair by applying the simplex permutation to
both the face and the ambient simplex. -/
def incidenceImage (sigma : Equiv.Perm beta) : (Finset beta × Finset beta) ↪ (Finset beta × Finset beta) where
  toFun p := (simplexImage (beta := beta) sigma p.1, simplexImage (beta := beta) sigma p.2)
  inj' := by
    rintro ⟨face₁, simplex₁⟩ ⟨face₂, simplex₂⟩ h
    change ((simplexImage (beta := beta) sigma face₁), (simplexImage (beta := beta) sigma simplex₁)) =
      ((simplexImage (beta := beta) sigma face₂), (simplexImage (beta := beta) sigma simplex₂)) at h
    rcases Prod.mk.inj h with ⟨hface, hsimplex⟩
    have hface' := simplexImage_injective (beta := beta) sigma hface
    have hsimplex' := simplexImage_injective (beta := beta) sigma hsimplex
    cases hface'
    cases hsimplex'
    rfl

/-- Unsigned support of the boundary from edges to vertices. -/
def edgeVertexIncidences (m : Nat) (U : Cover beta alpha) : Finset (Finset beta × Finset beta) :=
  (edgeSimplices m U).biUnion fun edge =>
    (codimOneFaces (beta := beta) edge).image
      (faceToSimplexEmbedding edge)

/-- Unsigned support of the boundary from triangles to edges. -/
def triangleEdgeIncidences (m : Nat) (U : Cover beta alpha) : Finset (Finset beta × Finset beta) :=
  (triangleSimplices m U).biUnion fun triangle =>
    (codimOneFaces (beta := beta) triangle).image
      (faceToSimplexEmbedding triangle)

theorem mem_edgeVertexIncidences_iff (m : Nat) (U : Cover beta alpha)
    {face edge : Finset beta} :
    (face, edge) ∈ edgeVertexIncidences (beta := beta) m U ↔
      edge ∈ edgeSimplices m U ∧ face ∈ codimOneFaces (beta := beta) edge := by
  unfold edgeVertexIncidences
  constructor
  · intro h
    rw [Finset.mem_biUnion] at h
    rcases h with ⟨edge', hedge', hedgeFace⟩
    rw [Finset.mem_image] at hedgeFace
    rcases hedgeFace with ⟨face', hface', hp⟩
    change ((face', edge') : Finset beta × Finset beta) = (face, edge) at hp
    rcases Prod.mk.inj hp with ⟨hfaceEq, hedgeEq⟩
    subst hfaceEq
    subst hedgeEq
    exact ⟨hedge', hface'⟩
  · rintro ⟨hedge, hface⟩
    rw [Finset.mem_biUnion]
    refine ⟨edge, hedge, ?_⟩
    rw [Finset.mem_image]
    exact ⟨face, hface, rfl⟩

theorem mem_triangleEdgeIncidences_iff (m : Nat) (U : Cover beta alpha)
    {face triangle : Finset beta} :
    (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U ↔
      triangle ∈ triangleSimplices m U ∧ face ∈ codimOneFaces (beta := beta) triangle := by
  unfold triangleEdgeIncidences
  constructor
  · intro h
    rw [Finset.mem_biUnion] at h
    rcases h with ⟨triangle', htriangle', htriangleFace⟩
    rw [Finset.mem_image] at htriangleFace
    rcases htriangleFace with ⟨face', hface', hp⟩
    change ((face', triangle') : Finset beta × Finset beta) = (face, triangle) at hp
    rcases Prod.mk.inj hp with ⟨hfaceEq, htriangleEq⟩
    subst hfaceEq
    subst htriangleEq
    exact ⟨htriangle', hface'⟩
  · rintro ⟨htriangle, hface⟩
    rw [Finset.mem_biUnion]
    refine ⟨triangle, htriangle, ?_⟩
    rw [Finset.mem_image]
    exact ⟨face, hface, rfl⟩

theorem edgeVertexIncidences_target_vertices (m : Nat) (U : Cover beta alpha)
    {face edge : Finset beta}
    (h : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U) :
    face ∈ vertexSimplices (beta := beta) := by
  rw [mem_edgeVertexIncidences_iff] at h
  exact edge_boundary_faces_are_vertices (m := m) (U := U) h.1 h.2

theorem triangleEdgeIncidences_target_edges (m : Nat) (U : Cover beta alpha)
    {face triangle : Finset beta}
    (h : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U) :
    face ∈ edgeSimplices m U := by
  rw [mem_triangleEdgeIncidences_iff] at h
  exact triangle_boundary_faces_are_edges (m := m) (U := U) h.1 h.2

theorem edgeVertexIncidences_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    (edgeVertexIncidences (beta := beta) m U).image (incidenceImage (beta := beta) sigma) =
      edgeVertexIncidences (beta := beta) m (relabelCover sigma U) := by
  ext p
  rcases p with ⟨face, edge⟩
  constructor
  · intro h
    rw [Finset.mem_image] at h
    rcases h with ⟨source, hsource, hp⟩
    rcases source with ⟨face₀, edge₀⟩
    change ((simplexImage (beta := beta) sigma face₀), (simplexImage (beta := beta) sigma edge₀)) =
      (face, edge) at hp
    rcases Prod.mk.inj hp with ⟨hfaceEq, hedgeEq⟩
    rw [mem_edgeVertexIncidences_iff] at hsource
    rw [mem_edgeVertexIncidences_iff]
    constructor
    · simpa [hedgeEq] using
        (mem_edgeSimplices_image (sigma := sigma) (m := m) (U := U) (s := edge₀) hsource.1)
    · have hmem :
          simplexImage (beta := beta) sigma face₀ ∈
            (codimOneFaces (beta := beta) edge₀).image (simplexImage (beta := beta) sigma) := by
        exact Finset.mem_image.mpr ⟨face₀, hsource.2, rfl⟩
      simpa [hfaceEq.symm, hedgeEq.symm,
        codimOneFaces_image (beta := beta) (sigma := sigma) (simplex := edge₀)] using hmem
  · intro h
    rw [mem_edgeVertexIncidences_iff] at h
    refine Finset.mem_image.mpr ?_
    refine ⟨(simplexImage (beta := beta) sigma.symm face, simplexImage (beta := beta) sigma.symm edge), ?_, ?_⟩
    · rw [mem_edgeVertexIncidences_iff]
      constructor
      · have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
          funext b
          simp [relabelCover]
        simpa [hrel] using
          (mem_edgeSimplices_image (sigma := sigma.symm) (m := m) (U := relabelCover sigma U)
            (s := edge) h.1)
      · have hmem :
            simplexImage (beta := beta) sigma.symm face ∈
              (codimOneFaces (beta := beta) edge).image (simplexImage (beta := beta) sigma.symm) := by
          exact Finset.mem_image.mpr ⟨face, h.2, rfl⟩
        simpa [codimOneFaces_image (beta := beta) (sigma := sigma.symm) (simplex := edge)] using hmem
    · change
        (simplexImage (beta := beta) sigma (simplexImage (beta := beta) sigma.symm face),
          simplexImage (beta := beta) sigma (simplexImage (beta := beta) sigma.symm edge)) =
        (face, edge)
      simp [simplexImage, image_symm_image]

theorem triangleEdgeIncidences_relabel (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    (triangleEdgeIncidences (beta := beta) m U).image (incidenceImage (beta := beta) sigma) =
      triangleEdgeIncidences (beta := beta) m (relabelCover sigma U) := by
  ext p
  rcases p with ⟨face, triangle⟩
  constructor
  · intro h
    rw [Finset.mem_image] at h
    rcases h with ⟨source, hsource, hp⟩
    rcases source with ⟨face₀, triangle₀⟩
    change ((simplexImage (beta := beta) sigma face₀), (simplexImage (beta := beta) sigma triangle₀)) =
      (face, triangle) at hp
    rcases Prod.mk.inj hp with ⟨hfaceEq, htriangleEq⟩
    rw [mem_triangleEdgeIncidences_iff] at hsource
    rw [mem_triangleEdgeIncidences_iff]
    constructor
    · simpa [htriangleEq] using
        (mem_triangleSimplices_image (sigma := sigma) (m := m) (U := U) (s := triangle₀) hsource.1)
    · have hmem :
          simplexImage (beta := beta) sigma face₀ ∈
            (codimOneFaces (beta := beta) triangle₀).image (simplexImage (beta := beta) sigma) := by
        exact Finset.mem_image.mpr ⟨face₀, hsource.2, rfl⟩
      simpa [hfaceEq.symm, htriangleEq.symm,
        codimOneFaces_image (beta := beta) (sigma := sigma) (simplex := triangle₀)] using hmem
  · intro h
    rw [mem_triangleEdgeIncidences_iff] at h
    refine Finset.mem_image.mpr ?_
    refine ⟨(simplexImage (beta := beta) sigma.symm face, simplexImage (beta := beta) sigma.symm triangle), ?_, ?_⟩
    · rw [mem_triangleEdgeIncidences_iff]
      constructor
      · have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
          funext b
          simp [relabelCover]
        simpa [hrel] using
          (mem_triangleSimplices_image (sigma := sigma.symm) (m := m) (U := relabelCover sigma U)
            (s := triangle) h.1)
      · have hmem :
            simplexImage (beta := beta) sigma.symm face ∈
              (codimOneFaces (beta := beta) triangle).image (simplexImage (beta := beta) sigma.symm) := by
          exact Finset.mem_image.mpr ⟨face, h.2, rfl⟩
        simpa [codimOneFaces_image (beta := beta) (sigma := sigma.symm) (simplex := triangle)] using hmem
    · change
        (simplexImage (beta := beta) sigma (simplexImage (beta := beta) sigma.symm face),
          simplexImage (beta := beta) sigma (simplexImage (beta := beta) sigma.symm triangle)) =
        (face, triangle)
      simp [simplexImage, image_symm_image]

end SCT.FND1
