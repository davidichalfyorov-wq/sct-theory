import SCTLean.FND1.BoundaryChainComplexExistence

/-!
# FND-1 Boundary Compatibility Equivalences

This module removes one more layer of redundant language by proving that the
current compatibility predicates are exactly equivalent to the corresponding
matrix vanishing statements.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem boundaryOrientationCompatibility_iff_matrix_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (omega₁ : EdgeVertexOrientationDatum (beta := beta) m U)
    (omega₂ : TriangleEdgeOrientationDatum (beta := beta) m U) :
    BoundaryOrientationCompatibility (beta := beta) m U omega₁ omega₂ ↔
      edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ = 0 := by
  constructor
  · intro hcompat
    exact edgeTriangleBoundaryComposition_matrix_eq_zero_of_compatible
      (beta := beta) (m := m) (U := U) (omega₁ := omega₁) (omega₂ := omega₂) hcompat
  · intro hzero
    refine ⟨?_⟩
    intro i k
    have hentry :
        edgeTriangleBoundaryComposition (beta := beta) m U omega₁ omega₂ i k = 0 := by
      simpa using congrFun (congrFun hzero i) k
    simpa [edgeTriangleCompositeTerm, edgeTriangleBoundaryCompositionEntry] using hentry

theorem boundaryLocalOrientationCompatibility_iff_matrix_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U) :
    BoundaryLocalOrientationCompatibility (beta := beta) m U theta ↔
      edgeTriangleBoundaryComposition (beta := beta) m U
        (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
        (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) = 0 := by
  constructor
  · intro hcompat
    exact edgeTriangleBoundaryComposition_matrix_eq_zero_of_local_compatible
      (beta := beta) (m := m) (U := U) (theta := theta) hcompat
  · intro hzero
    refine ⟨?_⟩
    exact (boundaryOrientationCompatibility_iff_matrix_eq_zero
      (beta := beta) (m := m) (U := U)
      (omega₁ := inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
      (omega₂ := inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)).mpr hzero

theorem hasCompatibleLocalOrientation_iff_exists_localBoundarySquareZero
    (m : Nat) (U : Cover beta alpha) :
    HasCompatibleLocalOrientation (beta := beta) m U ↔
      ∃ theta : BoundaryLocalOrientationDatum (beta := beta) m U,
        edgeTriangleBoundaryComposition (beta := beta) m U
          (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
          (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) = 0 := by
  constructor
  · intro hcompat
    rcases hcompat with ⟨theta, htheta⟩
    refine ⟨theta, ?_⟩
    exact (boundaryLocalOrientationCompatibility_iff_matrix_eq_zero
      (beta := beta) (m := m) (U := U) (theta := theta)).mp htheta
  · intro hzero
    rcases hzero with ⟨theta, htheta⟩
    refine ⟨theta, ?_⟩
    exact (boundaryLocalOrientationCompatibility_iff_matrix_eq_zero
      (beta := beta) (m := m) (U := U) (theta := theta)).mpr htheta

end SCT.FND1
