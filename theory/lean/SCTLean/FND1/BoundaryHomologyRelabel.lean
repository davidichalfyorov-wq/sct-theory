import SCTLean.FND1.BoundaryHomologyUse

/-!
# FND-1 Boundary Homology Relabeling

This module adds the first transport layer for the current finite homology
interface under relabeling of the cover index set.

We deliberately stay below Betti/rank level. The point here is structural:

- relabel simplices and boundary indices explicitly,
- transport `0/1/2`-chains by precomposition,
- prove that cycles, boundaries, and first-homology class equality are
  preserved under relabeling when the orientation data are transported
  transparently.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

def edgeBoundaryIndexEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    EdgeBoundaryIndex (beta := beta) m U ≃
      EdgeBoundaryIndex (beta := beta) m (relabelCover sigma U) where
  toFun j := ⟨simplexImage (beta := beta) sigma j.1,
    mem_edgeSimplices_image (beta := beta) (sigma := sigma) (m := m) (U := U) j.2⟩
  invFun j := by
    refine ⟨simplexImage (beta := beta) sigma.symm j.1, ?_⟩
    have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
      funext b
      simp [relabelCover]
    simpa [hrel] using
      (mem_edgeSimplices_image (beta := beta) (sigma := sigma.symm) (m := m)
        (U := relabelCover sigma U) (s := j.1) j.2)
  left_inv j := by
    apply Subtype.ext
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) j.1
  right_inv j := by
    apply Subtype.ext
    simpa [simplexImage] using image_symm_image (sigma := sigma) j.1

def edgeVertexRowIndexEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :
    EdgeVertexRowIndex (beta := beta) m U omega₁ ≃
      EdgeVertexRowIndex (beta := beta) m (relabelCover sigma U)
        (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) where
  toFun i := ⟨simplexImage (beta := beta) sigma i.1,
    mem_vertexSimplices_image (beta := beta) (sigma := sigma) i.2⟩
  invFun i := by
    refine ⟨simplexImage (beta := beta) sigma.symm i.1, ?_⟩
    simpa [simplexImage] using
      (mem_vertexSimplices_image (beta := beta) (sigma := sigma.symm) i.2)
  left_inv i := by
    apply Subtype.ext
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) i.1
  right_inv i := by
    apply Subtype.ext
    simpa [simplexImage] using image_symm_image (sigma := sigma) i.1

def triangleEdgeColIndexEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    TriangleEdgeColIndex (beta := beta) m U omega₂ ≃
      TriangleEdgeColIndex (beta := beta) m (relabelCover sigma U)
        (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) where
  toFun k := ⟨simplexImage (beta := beta) sigma k.1,
    mem_triangleSimplices_image (beta := beta) (sigma := sigma) (m := m) (U := U) k.2⟩
  invFun k := by
    refine ⟨simplexImage (beta := beta) sigma.symm k.1, ?_⟩
    have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
      funext b
      simp [relabelCover]
    simpa [hrel] using
      (mem_triangleSimplices_image (beta := beta) (sigma := sigma.symm) (m := m)
        (U := relabelCover sigma U) (s := k.1) k.2)
  left_inv k := by
    apply Subtype.ext
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) k.1
  right_inv k := by
    apply Subtype.ext
    simpa [simplexImage] using image_symm_image (sigma := sigma) k.1

/-- Transport `0`-chains by relabeling the row index set. -/
def relabelZeroChain (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (z : ZeroChain (beta := beta) m U omega₁) :
    ZeroChain (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) :=
  fun i => z ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁).symm i)

/-- Transport `1`-chains by relabeling the edge index set. -/
def relabelOneChain (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (z : OneChain (beta := beta) m U) :
    OneChain (beta := beta) m (relabelCover sigma U) :=
  fun j => z ((edgeBoundaryIndexEquiv (beta := beta) sigma m U).symm j)

/-- Transport `2`-chains by relabeling the triangle index set. -/
def relabelTwoChain (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (c : TwoChain (beta := beta) m U omega₂) :
    TwoChain (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) :=
  fun k => c ((triangleEdgeColIndexEquiv (beta := beta) sigma m U omega₂).symm k)

theorem relabelZeroChain_zero
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :
    relabelZeroChain (beta := beta) sigma m U omega₁ 0 = 0 := by
  funext i
  simp [relabelZeroChain]

theorem edgeVertexIncidences_mem_relabel_of_mem
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    {face edge : Finset beta}
    (h : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U) :
    (simplexImage (beta := beta) sigma face, simplexImage (beta := beta) sigma edge) ∈
      edgeVertexIncidences (beta := beta) m (relabelCover sigma U) := by
  have himg :
      incidenceImage (beta := beta) sigma (face, edge) ∈
        (edgeVertexIncidences (beta := beta) m U).image (incidenceImage (beta := beta) sigma) := by
    exact Finset.mem_image.mpr ⟨(face, edge), h, rfl⟩
  rw [edgeVertexIncidences_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)] at himg
  simpa [incidenceImage, simplexImage] using himg

theorem edgeVertexIncidences_mem_of_mem_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    {face edge : Finset beta}
    (h : (simplexImage (beta := beta) sigma face, simplexImage (beta := beta) sigma edge) ∈
      edgeVertexIncidences (beta := beta) m (relabelCover sigma U)) :
    (face, edge) ∈ edgeVertexIncidences (beta := beta) m U := by
  have h' := edgeVertexIncidences_mem_relabel_of_mem
      (beta := beta) (sigma := sigma.symm) (m := m) (U := relabelCover sigma U) h
  have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
    funext b
    simp [relabelCover]
  have hface :
      simplexImage (beta := beta) sigma.symm (simplexImage (beta := beta) sigma face) = face := by
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) face
  have hedge :
      simplexImage (beta := beta) sigma.symm (simplexImage (beta := beta) sigma edge) = edge := by
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) edge
  simpa [hrel, hface, hedge] using h'

theorem triangleEdgeIncidences_mem_relabel_of_mem
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    {edge triangle : Finset beta}
    (h : (edge, triangle) ∈ triangleEdgeIncidences (beta := beta) m U) :
    (simplexImage (beta := beta) sigma edge, simplexImage (beta := beta) sigma triangle) ∈
      triangleEdgeIncidences (beta := beta) m (relabelCover sigma U) := by
  have himg :
      incidenceImage (beta := beta) sigma (edge, triangle) ∈
        (triangleEdgeIncidences (beta := beta) m U).image (incidenceImage (beta := beta) sigma) := by
    exact Finset.mem_image.mpr ⟨(edge, triangle), h, rfl⟩
  rw [triangleEdgeIncidences_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)] at himg
  simpa [incidenceImage, simplexImage] using himg

theorem triangleEdgeIncidences_mem_of_mem_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    {edge triangle : Finset beta}
    (h : (simplexImage (beta := beta) sigma edge, simplexImage (beta := beta) sigma triangle) ∈
      triangleEdgeIncidences (beta := beta) m (relabelCover sigma U)) :
    (edge, triangle) ∈ triangleEdgeIncidences (beta := beta) m U := by
  have h' := triangleEdgeIncidences_mem_relabel_of_mem
      (beta := beta) (sigma := sigma.symm) (m := m) (U := relabelCover sigma U) h
  have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
    funext b
    simp [relabelCover]
  have hedge :
      simplexImage (beta := beta) sigma.symm (simplexImage (beta := beta) sigma edge) = edge := by
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) edge
  have htriangle :
      simplexImage (beta := beta) sigma.symm (simplexImage (beta := beta) sigma triangle) = triangle := by
    simpa [simplexImage] using image_symm_image (sigma := sigma.symm) triangle
  simpa [hrel, hedge, htriangle] using h'

theorem edgeVertexCoefficient_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (face edge : Finset beta) :
    edgeVertexCoefficient (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      (simplexImage (beta := beta) sigma face) (simplexImage (beta := beta) sigma edge) =
    edgeVertexCoefficient (beta := beta) m U omega₁ face edge := by
  by_cases hmem : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U
  · have hmem' := edgeVertexIncidences_mem_relabel_of_mem
        (beta := beta) (sigma := sigma) (m := m) (U := U) hmem
    rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m)
      (U := relabelCover sigma U)
      (omega := relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) hmem']
    rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m)
      (U := U) (omega := omega₁) hmem]
    let p : EdgeVertexSupport (beta := beta) m U := ⟨(face, edge), hmem⟩
    have hp :
        (relabelEdgeVertexSupportEquiv (beta := beta) sigma m U) p =
          ⟨(simplexImage (beta := beta) sigma face, simplexImage (beta := beta) sigma edge), hmem'⟩ := by
      apply Subtype.ext
      rfl
    rw [← hp]
    exact congrArg orientationSignValue <| by
      simpa [p] using
        (relabelEdgeVertexOrientationDatum_apply (beta := beta) (sigma := sigma)
          (m := m) (U := U) (omega := omega₁) p)
  · have hmem' :
      (simplexImage (beta := beta) sigma face, simplexImage (beta := beta) sigma edge) ∉
        edgeVertexIncidences (beta := beta) m (relabelCover sigma U) := by
      intro hcontra
      exact hmem <| edgeVertexIncidences_mem_of_mem_relabel
        (beta := beta) (sigma := sigma) (m := m) (U := U) hcontra
    rw [edgeVertexCoefficient_eq_zero_of_not_mem (beta := beta) (m := m)
      (U := relabelCover sigma U)
      (omega := relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) hmem']
    rw [edgeVertexCoefficient_eq_zero_of_not_mem (beta := beta) (m := m)
      (U := U) (omega := omega₁) hmem]

theorem triangleEdgeCoefficient_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (edge triangle : Finset beta) :
    triangleEdgeCoefficient (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
      (simplexImage (beta := beta) sigma edge) (simplexImage (beta := beta) sigma triangle) =
    triangleEdgeCoefficient (beta := beta) m U omega₂ edge triangle := by
  by_cases hmem : (edge, triangle) ∈ triangleEdgeIncidences (beta := beta) m U
  · have hmem' := triangleEdgeIncidences_mem_relabel_of_mem
        (beta := beta) (sigma := sigma) (m := m) (U := U) hmem
    rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m)
      (U := relabelCover sigma U)
      (omega := relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) hmem']
    rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m)
      (U := U) (omega := omega₂) hmem]
    let p : TriangleEdgeSupport (beta := beta) m U := ⟨(edge, triangle), hmem⟩
    have hp :
        (relabelTriangleEdgeSupportEquiv (beta := beta) sigma m U) p =
          ⟨(simplexImage (beta := beta) sigma edge, simplexImage (beta := beta) sigma triangle), hmem'⟩ := by
      apply Subtype.ext
      rfl
    rw [← hp]
    exact congrArg orientationSignValue <| by
      simpa [p] using
        (relabelTriangleEdgeOrientationDatum_apply (beta := beta) (sigma := sigma)
          (m := m) (U := U) (omega := omega₂) p)
  · have hmem' :
      (simplexImage (beta := beta) sigma edge, simplexImage (beta := beta) sigma triangle) ∉
        triangleEdgeIncidences (beta := beta) m (relabelCover sigma U) := by
      intro hcontra
      exact hmem <| triangleEdgeIncidences_mem_of_mem_relabel
        (beta := beta) (sigma := sigma) (m := m) (U := U) hcontra
    rw [triangleEdgeCoefficient_eq_zero_of_not_mem (beta := beta) (m := m)
      (U := relabelCover sigma U)
      (omega := relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) hmem']
    rw [triangleEdgeCoefficient_eq_zero_of_not_mem (beta := beta) (m := m)
      (U := U) (omega := omega₂) hmem]

theorem edgeVertexBoundaryMatrix_relabel_entry
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (i : EdgeVertexRowIndex (beta := beta) m U omega₁)
    (j : EdgeBoundaryIndex (beta := beta) m U) :
    edgeVertexBoundaryMatrix (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁) i)
      ((edgeBoundaryIndexEquiv (beta := beta) sigma m U) j) =
    edgeVertexBoundaryMatrix (beta := beta) m U omega₁ i j := by
  simpa [edgeVertexBoundaryMatrix, BoundaryTableData.toMatrix, BoundaryTableData.entry,
    edgeVertexBoundaryTable, edgeVertexSignedIncidence] using
    edgeVertexCoefficient_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
      (omega₁ := omega₁) i.1 j.1

theorem triangleEdgeBoundaryMatrix_relabel_entry
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (i : EdgeBoundaryIndex (beta := beta) m U)
    (k : TriangleEdgeColIndex (beta := beta) m U omega₂) :
    triangleEdgeBoundaryMatrix (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
      ((edgeBoundaryIndexEquiv (beta := beta) sigma m U) i)
      ((triangleEdgeColIndexEquiv (beta := beta) sigma m U omega₂) k) =
    triangleEdgeBoundaryMatrix (beta := beta) m U omega₂ i k := by
  simpa [triangleEdgeBoundaryMatrix, BoundaryTableData.toMatrix, BoundaryTableData.entry,
    triangleEdgeBoundaryTable, triangleEdgeSignedIncidence] using
    triangleEdgeCoefficient_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
      (omega₂ := omega₂) i.1 k.1

theorem relabelOneChain_zero (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    relabelOneChain (beta := beta) sigma m U 0 = 0 := by
  funext j
  simp [relabelOneChain]

theorem relabelOneChain_add (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (z w : OneChain (beta := beta) m U) :
    relabelOneChain (beta := beta) sigma m U (z + w) =
      relabelOneChain (beta := beta) sigma m U z +
        relabelOneChain (beta := beta) sigma m U w := by
  funext j
  simp [relabelOneChain]

theorem relabelOneChain_sub (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (z w : OneChain (beta := beta) m U) :
    relabelOneChain (beta := beta) sigma m U (z - w) =
      relabelOneChain (beta := beta) sigma m U z -
        relabelOneChain (beta := beta) sigma m U w := by
  funext j
  simp [relabelOneChain]

theorem edgeBoundaryMap_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (z : OneChain (beta := beta) m U) :
    edgeBoundaryMap (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      (relabelOneChain (beta := beta) sigma m U z) =
    relabelZeroChain (beta := beta) sigma m U omega₁
      (edgeBoundaryMap (beta := beta) m U omega₁ z) := by
  funext i
  change
    (∑ j : EdgeBoundaryIndex (beta := beta) m (relabelCover sigma U),
      edgeVertexBoundaryMatrix (beta := beta) m (relabelCover sigma U)
        (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) i j *
      relabelOneChain (beta := beta) sigma m U z j) =
    (edgeBoundaryMap (beta := beta) m U omega₁ z)
      ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁).symm i)
  let e := edgeBoundaryIndexEquiv (beta := beta) sigma m U
  symm
  refine Fintype.sum_equiv e
    (fun j : EdgeBoundaryIndex (beta := beta) m U =>
      edgeVertexBoundaryMatrix (beta := beta) m U omega₁
        ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁).symm i) j * z j)
    (fun j : EdgeBoundaryIndex (beta := beta) m (relabelCover sigma U) =>
      edgeVertexBoundaryMatrix (beta := beta) m (relabelCover sigma U)
        (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) i j *
      relabelOneChain (beta := beta) sigma m U z j) ?_
  intro j
  have hentry :=
    edgeVertexBoundaryMatrix_relabel_entry (beta := beta) (sigma := sigma) (m := m) (U := U)
      (omega₁ := omega₁) ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁).symm i) j
  have hentry' :
      edgeVertexBoundaryMatrix (beta := beta) m (relabelCover sigma U)
        (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) i (e j) =
      edgeVertexBoundaryMatrix (beta := beta) m U omega₁
        ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁).symm i) j := by
    simpa [e] using hentry
  change
    edgeVertexBoundaryMatrix (beta := beta) m U omega₁
      ((edgeVertexRowIndexEquiv (beta := beta) sigma m U omega₁).symm i) j * z j =
    edgeVertexBoundaryMatrix (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) i (e j) *
      relabelOneChain (beta := beta) sigma m U z (e j)
  rw [hentry']
  simp [relabelOneChain, e]

theorem triangleBoundaryMap_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (c : TwoChain (beta := beta) m U omega₂) :
    triangleBoundaryMap (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
      (relabelTwoChain (beta := beta) sigma m U omega₂ c) =
    relabelOneChain (beta := beta) sigma m U
      (triangleBoundaryMap (beta := beta) m U omega₂ c) := by
  funext i
  change
    (∑ k : TriangleEdgeColIndex (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂),
      triangleEdgeBoundaryMatrix (beta := beta) m (relabelCover sigma U)
        (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) i k *
      relabelTwoChain (beta := beta) sigma m U omega₂ c k) =
    (triangleBoundaryMap (beta := beta) m U omega₂ c)
      ((edgeBoundaryIndexEquiv (beta := beta) sigma m U).symm i)
  let e := triangleEdgeColIndexEquiv (beta := beta) sigma m U omega₂
  symm
  refine Fintype.sum_equiv e
    (fun k : TriangleEdgeColIndex (beta := beta) m U omega₂ =>
      triangleEdgeBoundaryMatrix (beta := beta) m U omega₂
        ((edgeBoundaryIndexEquiv (beta := beta) sigma m U).symm i) k * c k)
    (fun k : TriangleEdgeColIndex (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) =>
      triangleEdgeBoundaryMatrix (beta := beta) m (relabelCover sigma U)
        (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) i k *
      relabelTwoChain (beta := beta) sigma m U omega₂ c k) ?_
  intro k
  have hentry :=
    triangleEdgeBoundaryMatrix_relabel_entry (beta := beta) (sigma := sigma) (m := m) (U := U)
      (omega₂ := omega₂) ((edgeBoundaryIndexEquiv (beta := beta) sigma m U).symm i) k
  have hentry' :
      triangleEdgeBoundaryMatrix (beta := beta) m (relabelCover sigma U)
        (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) i (e k) =
      triangleEdgeBoundaryMatrix (beta := beta) m U omega₂
        ((edgeBoundaryIndexEquiv (beta := beta) sigma m U).symm i) k := by
    simpa [e] using hentry
  change
    triangleEdgeBoundaryMatrix (beta := beta) m U omega₂
      ((edgeBoundaryIndexEquiv (beta := beta) sigma m U).symm i) k * c k =
    triangleEdgeBoundaryMatrix (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) i (e k) *
      relabelTwoChain (beta := beta) sigma m U omega₂ c (e k)
  rw [hentry']
  simp [relabelTwoChain, e]

theorem isOneCycle_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneCycle (beta := beta) m U omega₁ z) :
    IsOneCycle (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      (relabelOneChain (beta := beta) sigma m U z) := by
  rw [IsOneCycle] at hz ⊢
  rw [edgeBoundaryMap_relabel (beta := beta) (sigma := sigma) (m := m) (U := U) (omega₁ := omega₁)
    (z := z)]
  rw [hz]
  exact relabelZeroChain_zero (beta := beta) (sigma := sigma) (m := m) (U := U) (omega₁ := omega₁)

theorem isOneBoundary_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z : OneChain (beta := beta) m U}
    (hz : IsOneBoundary (beta := beta) m U omega₂ z) :
    IsOneBoundary (beta := beta) m (relabelCover sigma U)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
      (relabelOneChain (beta := beta) sigma m U z) := by
  rcases hz with ⟨c, rfl⟩
  refine ⟨relabelTwoChain (beta := beta) sigma m U omega₂ c, ?_⟩
  exact triangleBoundaryMap_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
    (omega₂ := omega₂) (c := c)

theorem sameFirstHomologyClass_relabel
    (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    {z w : OneChain (beta := beta) m U}
    (hz : z ∈ oneCyclesSubmodule (beta := beta) m U omega₁)
    (hw : w ∈ oneCyclesSubmodule (beta := beta) m U omega₁)
    (hzw : SameFirstHomologyClass (beta := beta) m U omega₁ omega₂ ⟨z, hz⟩ ⟨w, hw⟩) :
    SameFirstHomologyClass (beta := beta) m (relabelCover sigma U)
      (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
      ⟨relabelOneChain (beta := beta) sigma m U z,
        isOneCycle_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
          (omega₁ := omega₁)
          ((mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := U) (omega₁ := omega₁) z).1 hz)⟩
      ⟨relabelOneChain (beta := beta) sigma m U w,
        isOneCycle_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
          (omega₁ := omega₁)
          ((mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := U) (omega₁ := omega₁) w).1 hw)⟩ := by
  have hz' :
      relabelOneChain (beta := beta) sigma m U z ∈
        oneCyclesSubmodule (beta := beta) m (relabelCover sigma U)
          (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) := by
    exact (mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := relabelCover sigma U)
      (omega₁ := relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      (relabelOneChain (beta := beta) sigma m U z)).2 <|
      isOneCycle_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
        (omega₁ := omega₁)
        ((mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := U) (omega₁ := omega₁) z).1 hz)
  have hw' :
      relabelOneChain (beta := beta) sigma m U w ∈
        oneCyclesSubmodule (beta := beta) m (relabelCover sigma U)
          (relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁) := by
    exact (mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := relabelCover sigma U)
      (omega₁ := relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
      (relabelOneChain (beta := beta) sigma m U w)).2 <|
      isOneCycle_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
        (omega₁ := omega₁)
        ((mem_oneCyclesSubmodule_iff (beta := beta) (m := m) (U := U) (omega₁ := omega₁) w).1 hw)
  have hsub :
      z - w ∈ oneBoundariesSubmodule (beta := beta) m U omega₂ := by
    exact (sameFirstHomologyClass_iff_sub_mem_boundaries
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
      (hz := hz) (hw := hw)).1 hzw
  have hsub' :
      relabelOneChain (beta := beta) sigma m U z -
        relabelOneChain (beta := beta) sigma m U w ∈
          oneBoundariesSubmodule (beta := beta) m (relabelCover sigma U)
            (relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂) := by
    rcases (mem_oneBoundariesSubmodule_iff (beta := beta) (m := m) (U := U)
      (omega₂ := omega₂) (z - w)).1 hsub with ⟨c, hc⟩
    refine (mem_oneBoundariesSubmodule_iff (beta := beta) (m := m)
      (U := relabelCover sigma U)
      (omega₂ := relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
      (relabelOneChain (beta := beta) sigma m U z -
        relabelOneChain (beta := beta) sigma m U w)).2 ?_
    refine ⟨relabelTwoChain (beta := beta) sigma m U omega₂ c, ?_⟩
    rw [triangleBoundaryMap_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)
      (omega₂ := omega₂) (c := c)]
    rw [hc, relabelOneChain_sub (beta := beta) (sigma := sigma) (m := m) (U := U) z w]
  exact sameFirstHomologyClass_of_sub_mem_boundaries
    (beta := beta) (m := m) (U := relabelCover sigma U)
    (omega₁ := relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega₁)
    (omega₂ := relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega₂)
    hz' hw' hsub'

end SCT.FND1
