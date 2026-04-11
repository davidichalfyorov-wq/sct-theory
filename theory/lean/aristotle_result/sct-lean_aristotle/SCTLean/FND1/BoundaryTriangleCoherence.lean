import SCTLean.FND1.BoundaryChainComplexExistence

/-!
# FND-1 Boundary Triangle Coherence

This module weakens the current noncanonical boundary-square-zero assumptions.

Previously the whole chain-complex layer depended on two primitive global data:

- an edge endpoint-choice datum `chi`,
- a triangle ordered-witness datum `tau`.

Here we make `tau` primary and ask only for a coherence law saying that when
two ordered triangle witnesses share an edge, they induce the same chosen
codimension-one face on that edge. Under this coherence hypothesis we can
construct a compatible global edge-choice datum `chi` and recover the earlier
boundary-square-zero consequences.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- An edge together with a witness that it appears in the boundary of a
triangle simplex. -/
abbrev EdgeTriangleWitness
    (m : Nat) (U : Cover beta alpha)
    (edge : ↑(edgeSimplices m U)) :=
  { triangle : ↑(triangleSimplices m U) // edge.1 ∈ codimOneFaces (beta := beta) triangle.1 }

theorem TriangleOrderedWitness.edgeAB_mem_codimOneFaces_triangle
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAB (beta := beta) m U triangle).1 ∈ codimOneFaces (beta := beta) triangle.1 := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  simpa [TriangleOrderedWitness.edgeAB, htri] using
    (triangleFaceAB_mem_codimOneFaces (beta := beta) hab hac hbc)

theorem TriangleOrderedWitness.edgeAC_mem_codimOneFaces_triangle
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAC (beta := beta) m U triangle).1 ∈ codimOneFaces (beta := beta) triangle.1 := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  simpa [TriangleOrderedWitness.edgeAC, htri] using
    (triangleFaceAC_mem_codimOneFaces (beta := beta) hab hac hbc)

theorem TriangleOrderedWitness.edgeBC_mem_codimOneFaces_triangle
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeBC (beta := beta) m U triangle).1 ∈ codimOneFaces (beta := beta) triangle.1 := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  simpa [TriangleOrderedWitness.edgeBC, htri] using
    (triangleFaceBC_mem_codimOneFaces (beta := beta) hab hac hbc)

/-- The chosen codimension-one face on a boundary edge induced by one ordered
triangle witness. This keeps only the "positive endpoint" information carried
by the witness ordering. -/
def TriangleOrderedWitness.chosenFaceOnBoundaryEdge
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (edge : ↑(edgeSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1) :
    LocalFaceSupport (beta := beta) edge.1 := by
  classical
  by_cases hAB : edge = w.edgeAB (beta := beta) m U triangle
  · cases hAB
    exact w.edgeAB_rightFace (beta := beta) m U triangle
  · by_cases hAC : edge = w.edgeAC (beta := beta) m U triangle
    · cases hAC
      exact w.edgeAC_rightFace (beta := beta) m U triangle
    · have hBC : edge = w.edgeBC (beta := beta) m U triangle := by
        rcases TriangleOrderedWitness.boundaryEdge_cases
            (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w)
            (edge := edge) hedge with hAB' | hAC' | hBC'
        · exact False.elim (hAB hAB')
        · exact False.elim (hAC hAC')
        · exact hBC'
      cases hBC
      exact w.edgeBC_rightFace (beta := beta) m U triangle

@[simp] theorem TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAB
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
      (w.edgeAB (beta := beta) m U triangle)
      (w.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle)) =
      w.edgeAB_rightFace (beta := beta) m U triangle := by
  classical
  unfold TriangleOrderedWitness.chosenFaceOnBoundaryEdge
  simp

@[simp] theorem TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
      (w.edgeAC (beta := beta) m U triangle)
      (w.edgeAC_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle)) =
      w.edgeAC_rightFace (beta := beta) m U triangle := by
  classical
  unfold TriangleOrderedWitness.chosenFaceOnBoundaryEdge
  have hneAB :
      ¬w.edgeAC (beta := beta) m U triangle = w.edgeAB (beta := beta) m U triangle := by
    intro h
    exact (TriangleOrderedWitness.edgeAB_ne_edgeAC (beta := beta) (m := m) (U := U)
      (triangle := triangle) (w := w)) h.symm
  simp [hneAB]

@[simp] theorem TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
      (w.edgeBC (beta := beta) m U triangle)
      (w.edgeBC_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle)) =
      w.edgeBC_rightFace (beta := beta) m U triangle := by
  classical
  unfold TriangleOrderedWitness.chosenFaceOnBoundaryEdge
  have hneAB :
      ¬w.edgeBC (beta := beta) m U triangle = w.edgeAB (beta := beta) m U triangle := by
    intro h
    exact (TriangleOrderedWitness.edgeAB_ne_edgeBC (beta := beta) (m := m) (U := U)
      (triangle := triangle) (w := w)) h.symm
  have hneAC :
      ¬w.edgeBC (beta := beta) m U triangle = w.edgeAC (beta := beta) m U triangle := by
    intro h
    exact (TriangleOrderedWitness.edgeAC_ne_edgeBC (beta := beta) (m := m) (U := U)
      (triangle := triangle) (w := w)) h.symm
  simp [hneAB, hneAC]

@[simp] theorem TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAB_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
      (w.edgeAB (beta := beta) m U triangle)
      (w.edgeAB_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle))).1 = ({w.b} : Finset beta) := by
  rw [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAB (beta := beta)
    (m := m) (U := U) (triangle := triangle) (w := w)]
  simp

@[simp] theorem TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAC_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
      (w.edgeAC (beta := beta) m U triangle)
      (w.edgeAC_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle))).1 = ({w.c} : Finset beta) := by
  rw [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeAC (beta := beta)
    (m := m) (U := U) (triangle := triangle) (w := w)]
  simp

@[simp] theorem TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeBC_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.chosenFaceOnBoundaryEdge (beta := beta) m U triangle
      (w.edgeBC (beta := beta) m U triangle)
      (w.edgeBC_mem_codimOneFaces_triangle (beta := beta) (m := m) (U := U)
        (triangle := triangle))).1 = ({w.c} : Finset beta) := by
  rw [TriangleOrderedWitness.chosenFaceOnBoundaryEdge_edgeBC (beta := beta)
    (m := m) (U := U) (triangle := triangle) (w := w)]
  simp

/-- Coherence for the triangle witness datum alone: whenever two triangle
witnesses see the same edge in their boundaries, they induce the same chosen
codimension-one face on that shared edge. -/
structure GlobalTriangleChoiceCoherence
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) : Prop where
  coherent :
    ∀ (edge : ↑(edgeSimplices m U))
      {triangle₁ triangle₂ : ↑(triangleSimplices m U)}
      (h₁ : edge.1 ∈ codimOneFaces (beta := beta) triangle₁.1)
      (h₂ : edge.1 ∈ codimOneFaces (beta := beta) triangle₂.1),
      ((tau triangle₁).chosenFaceOnBoundaryEdge (beta := beta) m U triangle₁ edge h₁).1 =
        ((tau triangle₂).chosenFaceOnBoundaryEdge (beta := beta) m U triangle₂ edge h₂).1

/-- Construct a global edge endpoint-choice datum from coherent triangle
witnesses. On edges that appear in some triangle boundary, use any witness and
coherence guarantees independence of that choice. On isolated edges, fall back
to an arbitrary local face. -/
noncomputable def coherentEdgeEndpointChoiceDatum
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :
    EdgeEndpointChoiceDatum (beta := beta) m U := by
  classical
  intro edge
  by_cases h : Nonempty (EdgeTriangleWitness (beta := beta) m U edge)
  · let witness : EdgeTriangleWitness (beta := beta) m U edge := Classical.choice h
    exact (tau witness.1).chosenFaceOnBoundaryEdge (beta := beta) m U witness.1 edge witness.2
  · exact Classical.choice
      (localFaceSupport_nonempty_of_edge (beta := beta) (m := m) (U := U) edge)

theorem coherentEdgeEndpointChoiceDatum_val_eq_triangle_face
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleChoiceCoherence (beta := beta) m U tau)
    (edge : ↑(edgeSimplices m U))
    (triangle : ↑(triangleSimplices m U))
    (hedge : edge.1 ∈ codimOneFaces (beta := beta) triangle.1) :
    (coherentEdgeEndpointChoiceDatum (beta := beta) m U tau edge).1 =
      ((tau triangle).chosenFaceOnBoundaryEdge (beta := beta) m U triangle edge hedge).1 := by
  classical
  let witness : EdgeTriangleWitness (beta := beta) m U edge := ⟨triangle, hedge⟩
  have hnonempty : Nonempty (EdgeTriangleWitness (beta := beta) m U edge) := ⟨witness⟩
  let selected : EdgeTriangleWitness (beta := beta) m U edge := Classical.choice hnonempty
  have hval :
      ((tau selected.1).chosenFaceOnBoundaryEdge (beta := beta) m U selected.1 edge selected.2).1 =
        ((tau triangle).chosenFaceOnBoundaryEdge (beta := beta) m U triangle edge hedge).1 :=
    hcoh.coherent edge selected.2 hedge
  simpa [coherentEdgeEndpointChoiceDatum, hnonempty, selected] using hval

theorem globalChoiceCompatibility_of_triangleChoiceCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleChoiceCoherence (beta := beta) m U tau) :
    GlobalEdgeTriangleChoiceCompatibility (beta := beta) m U
      (coherentEdgeEndpointChoiceDatum (beta := beta) m U tau) tau := by
  refine ⟨?_⟩
  intro triangle
  refine ⟨?_, ?_, ?_⟩
  · have hval :=
      coherentEdgeEndpointChoiceDatum_val_eq_triangle_face
        (beta := beta) (m := m) (U := U) (tau := tau) (hcoh := hcoh)
        (edge := (tau triangle).edgeAB (beta := beta) m U triangle)
        (triangle := triangle)
        ((tau triangle).edgeAB_mem_codimOneFaces_triangle (beta := beta)
          (m := m) (U := U) (triangle := triangle))
    simpa using hval
  · have hval :=
      coherentEdgeEndpointChoiceDatum_val_eq_triangle_face
        (beta := beta) (m := m) (U := U) (tau := tau) (hcoh := hcoh)
        (edge := (tau triangle).edgeAC (beta := beta) m U triangle)
        (triangle := triangle)
        ((tau triangle).edgeAC_mem_codimOneFaces_triangle (beta := beta)
          (m := m) (U := U) (triangle := triangle))
    simpa using hval
  · have hval :=
      coherentEdgeEndpointChoiceDatum_val_eq_triangle_face
        (beta := beta) (m := m) (U := U) (tau := tau) (hcoh := hcoh)
        (edge := (tau triangle).edgeBC (beta := beta) m U triangle)
        (triangle := triangle)
        ((tau triangle).edgeBC_mem_codimOneFaces_triangle (beta := beta)
          (m := m) (U := U) (triangle := triangle))
    simpa using hval

theorem hasCompatibleLocalOrientation_of_triangleChoiceCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleChoiceCoherence (beta := beta) m U tau) :
    HasCompatibleLocalOrientation (beta := beta) m U := by
  exact hasCompatibleLocalOrientation_of_globalChoiceCompatible
    (beta := beta) (m := m) (U := U)
    (chi := coherentEdgeEndpointChoiceDatum (beta := beta) m U tau)
    (tau := tau)
    (globalChoiceCompatibility_of_triangleChoiceCoherent
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh)

theorem hasBoundarySquareZero_of_triangleChoiceCoherent
    (m : Nat) (U : Cover beta alpha)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (hcoh : GlobalTriangleChoiceCoherence (beta := beta) m U tau) :
    HasBoundarySquareZero (beta := beta) m U := by
  exact hasBoundarySquareZero_of_hasCompatibleLocalOrientation
    (beta := beta) (m := m) (U := U)
    (hasCompatibleLocalOrientation_of_triangleChoiceCoherent
      (beta := beta) (m := m) (U := U) (tau := tau) hcoh)

end SCT.FND1
