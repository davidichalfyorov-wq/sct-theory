import Mathlib.Data.Int.Basic
import SCTLean.FND1.BoundaryIncidence

/-!
# FND-1 Boundary Orientation

This module introduces an explicit auxiliary orientation datum on top of the
already formalized unsigned incidence support. The point is architectural:

- the support of nonzero boundary entries is canonical,
- the signs are *not* yet canonical,
- so the signs are modeled as extra data rather than being smuggled in.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Boolean orientation sign, interpreted as `+1` / `-1`. -/
def orientationSignValue : Bool → Int
  | false => 1
  | true => -1

theorem orientationSignValue_eq_one_or_neg_one (b : Bool) :
    orientationSignValue b = 1 ∨ orientationSignValue b = -1 := by
  cases b <;> simp [orientationSignValue]

theorem orientationSignValue_ne_zero (b : Bool) :
    orientationSignValue b ≠ 0 := by
  cases b <;> simp [orientationSignValue]

/-- Support pairs for the unsigned boundary from edges to vertices. -/
abbrev EdgeVertexSupport (m : Nat) (U : Cover beta alpha) :=
  { p : Finset beta × Finset beta // p ∈ edgeVertexIncidences (beta := beta) m U }

/-- Support pairs for the unsigned boundary from triangles to edges. -/
abbrev TriangleEdgeSupport (m : Nat) (U : Cover beta alpha) :=
  { p : Finset beta × Finset beta // p ∈ triangleEdgeIncidences (beta := beta) m U }

/-- An auxiliary sign assignment on edge-to-vertex support pairs. -/
abbrev EdgeVertexOrientationDatum (m : Nat) (U : Cover beta alpha) :=
  EdgeVertexSupport (beta := beta) m U → Bool

/-- An auxiliary sign assignment on triangle-to-edge support pairs. -/
abbrev TriangleEdgeOrientationDatum (m : Nat) (U : Cover beta alpha) :=
  TriangleEdgeSupport (beta := beta) m U → Bool

/-- Signed coefficient induced by an explicit orientation datum on edge-to-vertex support. -/
def edgeVertexCoefficient (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    (face edge : Finset beta) : Int :=
  if h : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U then
    orientationSignValue (omega ⟨(face, edge), h⟩)
  else
    0

/-- Signed coefficient induced by an explicit orientation datum on triangle-to-edge support. -/
def triangleEdgeCoefficient (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    (face triangle : Finset beta) : Int :=
  if h : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U then
    orientationSignValue (omega ⟨(face, triangle), h⟩)
  else
    0

theorem edgeVertexCoefficient_eq_signValue_of_mem (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U) :
    edgeVertexCoefficient (beta := beta) m U omega face edge =
      orientationSignValue (omega ⟨(face, edge), h⟩) := by
  unfold edgeVertexCoefficient
  simp [h]

theorem edgeVertexCoefficient_eq_zero_of_not_mem (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (face, edge) ∉ edgeVertexIncidences (beta := beta) m U) :
    edgeVertexCoefficient (beta := beta) m U omega face edge = 0 := by
  unfold edgeVertexCoefficient
  simp [h]

theorem edgeVertexCoefficient_eq_one_or_neg_one_of_mem (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U) :
    edgeVertexCoefficient (beta := beta) m U omega face edge = 1 ∨
      edgeVertexCoefficient (beta := beta) m U omega face edge = -1 := by
  rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U) (omega := omega) h]
  exact orientationSignValue_eq_one_or_neg_one (omega ⟨(face, edge), h⟩)

theorem edgeVertexCoefficient_nonzero_iff (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    {face edge : Finset beta} :
    edgeVertexCoefficient (beta := beta) m U omega face edge ≠ 0 ↔
      (face, edge) ∈ edgeVertexIncidences (beta := beta) m U := by
  constructor
  · intro hcoeff
    by_contra hmem
    rw [edgeVertexCoefficient_eq_zero_of_not_mem (beta := beta) (m := m) (U := U) (omega := omega) hmem] at hcoeff
    exact hcoeff rfl
  · intro hmem
    rw [edgeVertexCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U) (omega := omega) hmem]
    exact orientationSignValue_ne_zero (omega ⟨(face, edge), hmem⟩)

theorem triangleEdgeCoefficient_eq_signValue_of_mem (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U) :
    triangleEdgeCoefficient (beta := beta) m U omega face triangle =
      orientationSignValue (omega ⟨(face, triangle), h⟩) := by
  unfold triangleEdgeCoefficient
  simp [h]

theorem triangleEdgeCoefficient_eq_zero_of_not_mem (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (face, triangle) ∉ triangleEdgeIncidences (beta := beta) m U) :
    triangleEdgeCoefficient (beta := beta) m U omega face triangle = 0 := by
  unfold triangleEdgeCoefficient
  simp [h]

theorem triangleEdgeCoefficient_eq_one_or_neg_one_of_mem (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U) :
    triangleEdgeCoefficient (beta := beta) m U omega face triangle = 1 ∨
      triangleEdgeCoefficient (beta := beta) m U omega face triangle = -1 := by
  rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U) (omega := omega) h]
  exact orientationSignValue_eq_one_or_neg_one (omega ⟨(face, triangle), h⟩)

theorem triangleEdgeCoefficient_nonzero_iff (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta} :
    triangleEdgeCoefficient (beta := beta) m U omega face triangle ≠ 0 ↔
      (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U := by
  constructor
  · intro hcoeff
    by_contra hmem
    rw [triangleEdgeCoefficient_eq_zero_of_not_mem (beta := beta) (m := m) (U := U) (omega := omega) hmem] at hcoeff
    exact hcoeff rfl
  · intro hmem
    rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U) (omega := omega) hmem]
    exact orientationSignValue_ne_zero (omega ⟨(face, triangle), hmem⟩)

omit [Fintype beta] in
theorem incidenceImage_symm_apply (sigma : Equiv.Perm beta) (p : Finset beta × Finset beta) :
    incidenceImage (beta := beta) sigma (incidenceImage (beta := beta) sigma.symm p) = p := by
  rcases p with ⟨face, simplex⟩
  apply Prod.ext
  · simpa [incidenceImage, simplexImage] using image_symm_image (sigma := sigma) face
  · simpa [incidenceImage, simplexImage] using image_symm_image (sigma := sigma) simplex

omit [Fintype beta] in
theorem incidenceImage_apply_symm (sigma : Equiv.Perm beta) (p : Finset beta × Finset beta) :
    incidenceImage (beta := beta) sigma.symm (incidenceImage (beta := beta) sigma p) = p := by
  rcases p with ⟨face, simplex⟩
  apply Prod.ext
  · simpa [incidenceImage, simplexImage] using image_symm_image (sigma := sigma.symm) face
  · simpa [incidenceImage, simplexImage] using image_symm_image (sigma := sigma.symm) simplex

/-- Relabel support pairs for the edge-to-vertex unsigned boundary. -/
def relabelEdgeVertexSupportEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    EdgeVertexSupport (beta := beta) m U ≃
      EdgeVertexSupport (beta := beta) m (relabelCover sigma U) where
  toFun p := ⟨incidenceImage (beta := beta) sigma p.1, by
    have hpImg :
        incidenceImage (beta := beta) sigma p.1 ∈
          (edgeVertexIncidences (beta := beta) m U).image (incidenceImage (beta := beta) sigma) := by
      exact Finset.mem_image.mpr ⟨p.1, p.2, rfl⟩
    simpa [edgeVertexIncidences_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)] using hpImg⟩
  invFun p := ⟨incidenceImage (beta := beta) sigma.symm p.1, by
    have hpImg :
        incidenceImage (beta := beta) sigma.symm p.1 ∈
          (edgeVertexIncidences (beta := beta) m (relabelCover sigma U)).image
            (incidenceImage (beta := beta) sigma.symm) := by
      exact Finset.mem_image.mpr ⟨p.1, p.2, rfl⟩
    have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
      funext b
      simp [relabelCover]
    simpa [edgeVertexIncidences_relabel (beta := beta) (sigma := sigma.symm) (m := m)
      (U := relabelCover sigma U), hrel] using hpImg⟩
  left_inv p := by
    apply Subtype.ext
    simpa using incidenceImage_apply_symm (beta := beta) (sigma := sigma) p.1
  right_inv p := by
    apply Subtype.ext
    simpa using incidenceImage_symm_apply (beta := beta) (sigma := sigma) p.1

/-- Relabel support pairs for the triangle-to-edge unsigned boundary. -/
def relabelTriangleEdgeSupportEquiv (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha) :
    TriangleEdgeSupport (beta := beta) m U ≃
      TriangleEdgeSupport (beta := beta) m (relabelCover sigma U) where
  toFun p := ⟨incidenceImage (beta := beta) sigma p.1, by
    have hpImg :
        incidenceImage (beta := beta) sigma p.1 ∈
          (triangleEdgeIncidences (beta := beta) m U).image (incidenceImage (beta := beta) sigma) := by
      exact Finset.mem_image.mpr ⟨p.1, p.2, rfl⟩
    simpa [triangleEdgeIncidences_relabel (beta := beta) (sigma := sigma) (m := m) (U := U)] using hpImg⟩
  invFun p := ⟨incidenceImage (beta := beta) sigma.symm p.1, by
    have hpImg :
        incidenceImage (beta := beta) sigma.symm p.1 ∈
          (triangleEdgeIncidences (beta := beta) m (relabelCover sigma U)).image
            (incidenceImage (beta := beta) sigma.symm) := by
      exact Finset.mem_image.mpr ⟨p.1, p.2, rfl⟩
    have hrel : relabelCover sigma.symm (relabelCover sigma U) = U := by
      funext b
      simp [relabelCover]
    simpa [triangleEdgeIncidences_relabel (beta := beta) (sigma := sigma.symm) (m := m)
      (U := relabelCover sigma U), hrel] using hpImg⟩
  left_inv p := by
    apply Subtype.ext
    simpa using incidenceImage_apply_symm (beta := beta) (sigma := sigma) p.1
  right_inv p := by
    apply Subtype.ext
    simpa using incidenceImage_symm_apply (beta := beta) (sigma := sigma) p.1

/-- Transport an edge-to-vertex orientation datum across relabeling. -/
def relabelEdgeVertexOrientationDatum (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U) :
    EdgeVertexOrientationDatum (beta := beta) m (relabelCover sigma U) :=
  fun p => omega ((relabelEdgeVertexSupportEquiv (beta := beta) sigma m U).symm p)

/-- Transport a triangle-to-edge orientation datum across relabeling. -/
def relabelTriangleEdgeOrientationDatum (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U) :
    TriangleEdgeOrientationDatum (beta := beta) m (relabelCover sigma U) :=
  fun p => omega ((relabelTriangleEdgeSupportEquiv (beta := beta) sigma m U).symm p)

theorem relabelEdgeVertexOrientationDatum_apply (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega : EdgeVertexOrientationDatum (beta := beta) m U)
    (p : EdgeVertexSupport (beta := beta) m U) :
    relabelEdgeVertexOrientationDatum (beta := beta) sigma m U omega
      ((relabelEdgeVertexSupportEquiv (beta := beta) sigma m U) p) = omega p := by
  unfold relabelEdgeVertexOrientationDatum
  exact congrArg omega ((relabelEdgeVertexSupportEquiv (beta := beta) sigma m U).left_inv p)

theorem relabelTriangleEdgeOrientationDatum_apply (sigma : Equiv.Perm beta) (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    (p : TriangleEdgeSupport (beta := beta) m U) :
    relabelTriangleEdgeOrientationDatum (beta := beta) sigma m U omega
      ((relabelTriangleEdgeSupportEquiv (beta := beta) sigma m U) p) = omega p := by
  unfold relabelTriangleEdgeOrientationDatum
  exact congrArg omega ((relabelTriangleEdgeSupportEquiv (beta := beta) sigma m U).left_inv p)

end SCT.FND1
