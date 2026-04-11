import SCTLean.FND1.BoundaryCompatibility

/-!
# FND-1 Boundary Local Orientation

This module moves one step closer to a genuinely geometric orientation rule.
Instead of assigning signs directly to arbitrary global support pairs, we assign
signs locally to codimension-one faces *inside each simplex*:

- one local sign rule for each edge,
- one local sign rule for each triangle.

These local rules then induce the previously used global orientation data.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Local face support of a fixed simplex. -/
abbrev LocalFaceSupport (simplex : Finset beta) :=
  { face : Finset beta // face ∈ codimOneFaces (beta := beta) simplex }

/-- Local orientation rule on the codimension-one faces of a fixed simplex. -/
abbrev LocalFaceOrientationDatum (simplex : Finset beta) :=
  LocalFaceSupport (beta := beta) simplex → Bool

/-- Local orientation data for edges and triangles. -/
structure BoundaryLocalOrientationDatum (m : Nat) (U : Cover beta alpha) where
  edgeFaces :
    ∀ edge : ↑(edgeSimplices m U),
      LocalFaceOrientationDatum (beta := beta) edge.1
  triangleFaces :
    ∀ triangle : ↑(triangleSimplices m U),
      LocalFaceOrientationDatum (beta := beta) triangle.1

/-- Induced global edge-to-vertex orientation datum from local edge-face signs. -/
def inducedEdgeVertexOrientationDatum (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U) :
    EdgeVertexOrientationDatum (beta := beta) m U := by
  intro p
  let edgeMem : p.1.2 ∈ edgeSimplices m U :=
    (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp p.2 |>.1
  let faceMem : p.1.1 ∈ codimOneFaces (beta := beta) p.1.2 :=
    (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp p.2 |>.2
  exact theta.edgeFaces ⟨p.1.2, edgeMem⟩ ⟨p.1.1, faceMem⟩

/-- Induced global triangle-to-edge orientation datum from local triangle-face signs. -/
def inducedTriangleEdgeOrientationDatum (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U) :
    TriangleEdgeOrientationDatum (beta := beta) m U := by
  intro p
  let triMem : p.1.2 ∈ triangleSimplices m U :=
    (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp p.2 |>.1
  let faceMem : p.1.1 ∈ codimOneFaces (beta := beta) p.1.2 :=
    (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp p.2 |>.2
  exact theta.triangleFaces ⟨p.1.2, triMem⟩ ⟨p.1.1, faceMem⟩

theorem inducedEdgeVertexOrientationDatum_eq_local (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U)
    {face edge : Finset beta}
    (h : (face, edge) ∈ edgeVertexIncidences (beta := beta) m U) :
    inducedEdgeVertexOrientationDatum (beta := beta) m U theta ⟨(face, edge), h⟩ =
      theta.edgeFaces
        ⟨edge, (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp h |>.1⟩
        ⟨face, (mem_edgeVertexIncidences_iff (beta := beta) (m := m) (U := U)).mp h |>.2⟩ := by
  unfold inducedEdgeVertexOrientationDatum
  simp

theorem inducedTriangleEdgeOrientationDatum_eq_local (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U)
    {face triangle : Finset beta}
    (h : (face, triangle) ∈ triangleEdgeIncidences (beta := beta) m U) :
    inducedTriangleEdgeOrientationDatum (beta := beta) m U theta ⟨(face, triangle), h⟩ =
      theta.triangleFaces
        ⟨triangle, (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp h |>.1⟩
        ⟨face, (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp h |>.2⟩ := by
  unfold inducedTriangleEdgeOrientationDatum
  simp

/-- Local-face compatibility is stated on the induced global orientation data. -/
structure BoundaryLocalOrientationCompatibility
    (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U) : Prop where
  compatible :
    BoundaryOrientationCompatibility (beta := beta) m U
      (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)

theorem edgeTriangleBoundaryComposition_eq_zero_of_local_compatible
    (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryLocalOrientationCompatibility (beta := beta) m U theta)
    (i : EdgeVertexRowIndex (beta := beta) m U (inducedEdgeVertexOrientationDatum (beta := beta) m U theta))
    (k : TriangleEdgeColIndex (beta := beta) m U (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) i k = 0 := by
  exact edgeTriangleBoundaryComposition_eq_zero_of_compatible
    (beta := beta) (m := m) (U := U)
    (omega₁ := inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
    (omega₂ := inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)
    hcompat.compatible i k

theorem edgeTriangleBoundaryComposition_matrix_eq_zero_of_local_compatible
    (m : Nat) (U : Cover beta alpha)
    (theta : BoundaryLocalOrientationDatum (beta := beta) m U)
    (hcompat : BoundaryLocalOrientationCompatibility (beta := beta) m U theta) :
    edgeTriangleBoundaryComposition (beta := beta) m U
      (inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U theta) = 0 := by
  exact edgeTriangleBoundaryComposition_matrix_eq_zero_of_compatible
    (beta := beta) (m := m) (U := U)
    (omega₁ := inducedEdgeVertexOrientationDatum (beta := beta) m U theta)
    (omega₂ := inducedTriangleEdgeOrientationDatum (beta := beta) m U theta)
    hcompat.compatible

end SCT.FND1
