import Mathlib.LinearAlgebra.Matrix.ToLinearEquiv
import SCTLean.FND1.BoundaryCompatibilityEquiv

/-!
# FND-1 Boundary Chain Maps

This module turns the current boundary matrices into actual maps on finite
integer-valued chains and proves the first direct homological consequence:
if `∂₁ ∘ ∂₂ = 0`, then every `2`-boundary is a `1`-cycle.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Integer-valued `0`-chains for the current edge boundary. -/
abbrev ZeroChain (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :=
  EdgeVertexRowIndex (beta := beta) m U omega₁ → Int

/-- Integer-valued `1`-chains on the finite nerve. -/
abbrev OneChain (m : Nat) (U : Cover beta alpha) :=
  EdgeBoundaryIndex (beta := beta) m U → Int

/-- Integer-valued `2`-chains for the current triangle boundary. -/
abbrev TwoChain (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :=
  TriangleEdgeColIndex (beta := beta) m U omega₂ → Int

/-- Boundary map `∂₁ : C₁ -> C₀`. -/
def edgeBoundaryMap (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U) :
    OneChain (beta := beta) m U ->
      ZeroChain (beta := beta) m U omega₁ :=
  Matrix.mulVec
    (edgeVertexBoundaryMatrix (beta := beta) m U omega₁ :
      Matrix
        (EdgeVertexRowIndex (beta := beta) m U omega₁)
        (EdgeBoundaryIndex (beta := beta) m U)
        Int)

/-- Boundary map `∂₂ : C₂ -> C₁`. -/
def triangleBoundaryMap (m : Nat) (U : Cover beta alpha)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    TwoChain (beta := beta) m U omega₂ ->
      OneChain (beta := beta) m U :=
  Matrix.mulVec
    (triangleEdgeBoundaryMatrix (beta := beta) m U omega₂ :
      Matrix
        (EdgeBoundaryIndex (beta := beta) m U)
        (TriangleEdgeColIndex (beta := beta) m U omega₂)
        Int)

theorem edgeTriangleBoundaryComposition_eq_matrix_mul
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ =
      let M : Matrix
          (EdgeVertexRowIndex (beta := beta) m U omega₁)
          (EdgeBoundaryIndex (beta := beta) m U)
          Int := edgeVertexBoundaryMatrix (beta := beta) m U omega₁
      let N : Matrix
          (EdgeBoundaryIndex (beta := beta) m U)
          (TriangleEdgeColIndex (beta := beta) m U omega₂)
          Int := triangleEdgeBoundaryMatrix (beta := beta) m U omega₂
      M * N := by
  rfl

theorem edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_matrix_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hzero : edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0)
    (c : TwoChain (beta := beta) m U omega₂) :
    edgeBoundaryMap (beta := beta) m U omega₁
      (triangleBoundaryMap (beta := beta) m U omega₂ c) = 0 := by
  have hmul :
      let M : Matrix
          (EdgeVertexRowIndex (beta := beta) m U omega₁)
          (EdgeBoundaryIndex (beta := beta) m U)
          Int := edgeVertexBoundaryMatrix (beta := beta) m U omega₁
      let N : Matrix
          (EdgeBoundaryIndex (beta := beta) m U)
          (TriangleEdgeColIndex (beta := beta) m U omega₂)
          Int := triangleEdgeBoundaryMatrix (beta := beta) m U omega₂
      M * N = 0 := by
    simpa [edgeTriangleBoundaryComposition_eq_matrix_mul
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)] using hzero
  have hvec := congrArg (fun M => Matrix.mulVec M c) hmul
  simpa [edgeBoundaryMap, triangleBoundaryMap, Matrix.mulVec_mulVec] using hvec

theorem edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_compatible
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂)
    (c : TwoChain (beta := beta) m U omega₂) :
    edgeBoundaryMap (beta := beta) m U omega₁
      (triangleBoundaryMap (beta := beta) m U omega₂ c) = 0 := by
  exact edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_matrix_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)
    ((boundaryOrientationCompatibility_iff_matrix_eq_zero
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂)).mp hcompat)
    c

theorem edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_local_compatible
    (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryLocalOrientationCompatibility (beta := beta) m U theta)
    (c : TwoChain (beta := beta) m U
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)) :
    edgeBoundaryMap (beta := beta) m U
      (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
      (triangleBoundaryMap (beta := beta) m U
        (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) c) = 0 := by
  exact edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_compatible
    (beta := beta) (m := m) (U := U)
    (omega₁ := inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
    (omega₂ := inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)
    hcompat.compatible c

theorem exists_chainMaps_boundary_of_boundary_zero_of_hasBoundarySquareZero
    (m : Nat) (U : Cover beta alpha)
    (hexists : HasBoundarySquareZero (beta := beta) m U) :
    ∃ (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
      (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U),
      ∀ c : TwoChain (beta := beta) m U omega₂,
        edgeBoundaryMap (beta := beta) m U omega₁
          (triangleBoundaryMap (beta := beta) m U omega₂ c) = 0 := by
  rcases hexists with ⟨omega₁, omega₂, hzero⟩
  refine ⟨omega₁, omega₂, ?_⟩
  intro c
  exact edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_matrix_zero
    (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hzero c

theorem exists_chainMaps_boundary_of_boundary_zero_of_globalChoiceCompatible
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hglobal : GlobalEdgeTriangleChoiceCompatibility
      (beta := beta) m U chi tau) :
    ∃ (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
      (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U),
      ∀ c : TwoChain (beta := beta) m U omega₂,
        edgeBoundaryMap (beta := beta) m U omega₁
          (triangleBoundaryMap (beta := beta) m U omega₂ c) = 0 := by
  exact exists_chainMaps_boundary_of_boundary_zero_of_hasBoundarySquareZero
    (beta := beta) (m := m) (U := U)
    (hasBoundarySquareZero_of_globalChoiceCompatible
      (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) hglobal)

end SCT.FND1
