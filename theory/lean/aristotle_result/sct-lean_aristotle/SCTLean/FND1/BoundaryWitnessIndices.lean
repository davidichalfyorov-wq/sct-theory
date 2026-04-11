import SCTLean.FND1.BoundaryWitnessCoefficients

/-!
# FND-1 Boundary Witness Indices

This module packages the explicit row/column indices attached to a fixed witness
triangle under the current noncanonical choice-induced boundary data.

The purpose is narrow and structural:

- give canonical names to the three singleton vertex rows `A/B/C`,
- identify the column corresponding to the witness triangle itself,
- record the basic value and distinctness facts once, so later composition-entry
  reductions do not have to repeat `Subtype` boilerplate.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

/-- Edge orientation datum induced by the current global choice package. -/
abbrev choiceInducedEdgeDatum
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :=
  inducedEdgeVertexOrientationDatum (beta := beta) m U
    (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau)

/-- Triangle orientation datum induced by the current global choice package. -/
abbrev choiceInducedTriangleDatum
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U) :=
  inducedTriangleEdgeOrientationDatum (beta := beta) m U
    (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau)

/-- Singleton-vertex row corresponding to a chosen vertex `v`. -/
def witnessVertexRow
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (v : beta) :
    EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau) := by
  refine ⟨({v} : Finset beta), ?_⟩
  change ({v} : Finset beta) ∈ (Finset.univ : Finset beta).powersetCard 1
  rw [Finset.mem_powersetCard]
  constructor
  · intro x hx
    simp at hx
    simp [hx]
  · simp

/-- Singleton row for vertex `a` of the witness triangle. -/
def TriangleOrderedWitness.rowA
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau) :=
  witnessVertexRow (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) w.a

/-- Singleton row for vertex `b` of the witness triangle. -/
def TriangleOrderedWitness.rowB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau) :=
  witnessVertexRow (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) w.b

/-- Singleton row for vertex `c` of the witness triangle. -/
def TriangleOrderedWitness.rowC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    EdgeVertexRowIndex (beta := beta) m U
      (choiceInducedEdgeDatum (beta := beta) m U chi tau) :=
  witnessVertexRow (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) w.c

/-- Column index corresponding to the witness triangle itself. -/
def witnessTriangleCol
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U)) :
    TriangleEdgeColIndex (beta := beta) m U
      (choiceInducedTriangleDatum (beta := beta) m U chi tau) :=
  triangle

@[simp] theorem witnessVertexRow_val
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (v : beta) :
    (witnessVertexRow (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) v).1 =
      ({v} : Finset beta) := rfl

@[simp] theorem TriangleOrderedWitness.rowA_val
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.rowA (beta := beta) m U chi tau triangle).1 = ({w.a} : Finset beta) := by
  rfl

@[simp] theorem TriangleOrderedWitness.rowB_val
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.rowB (beta := beta) m U chi tau triangle).1 = ({w.b} : Finset beta) := by
  rfl

@[simp] theorem TriangleOrderedWitness.rowC_val
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.rowC (beta := beta) m U chi tau triangle).1 = ({w.c} : Finset beta) := by
  rfl

@[simp] theorem witnessTriangleCol_val
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U)) :
    (witnessTriangleCol (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau) triangle).1 =
      triangle.1 := rfl

theorem TriangleOrderedWitness.rowA_ne_rowB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.rowA (beta := beta) m U chi tau triangle ≠
      w.rowB (beta := beta) m U chi tau triangle := by
  intro h
  have hset : ({w.a} : Finset beta) = ({w.b} : Finset beta) := by
    simpa using congrArg Subtype.val h
  have ha : w.a ∈ ({w.a} : Finset beta) := by simp
  have ha' : w.a ∈ ({w.b} : Finset beta) := by simpa [hset] using ha
  exact w.hab (by simpa using ha')

theorem TriangleOrderedWitness.rowA_ne_rowC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.rowA (beta := beta) m U chi tau triangle ≠
      w.rowC (beta := beta) m U chi tau triangle := by
  intro h
  have hset : ({w.a} : Finset beta) = ({w.c} : Finset beta) := by
    simpa using congrArg Subtype.val h
  have ha : w.a ∈ ({w.a} : Finset beta) := by simp
  have ha' : w.a ∈ ({w.c} : Finset beta) := by simpa [hset] using ha
  exact w.hac (by simpa using ha')

theorem TriangleOrderedWitness.rowB_ne_rowC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↥(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.rowB (beta := beta) m U chi tau triangle ≠
      w.rowC (beta := beta) m U chi tau triangle := by
  intro h
  have hset : ({w.b} : Finset beta) = ({w.c} : Finset beta) := by
    simpa using congrArg Subtype.val h
  have hb : w.b ∈ ({w.b} : Finset beta) := by simp
  have hb' : w.b ∈ ({w.c} : Finset beta) := by simpa [hset] using hb
  exact w.hbc (by simpa using hb')

end SCT.FND1
