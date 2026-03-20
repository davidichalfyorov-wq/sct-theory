import SCTLean.FND1.BoundaryCanonicalObstruction

/-!
# FND-1 Boundary Endpoint Choice

This module records the constructive complement to the low-dimensional
orientation obstruction: although fully swap-invariant local rules fail on an
unordered edge, one explicit chosen codimension-one face is already enough to
recover an alternating local edge boundary.
-/
namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

def localOrientationFromChosenFace {simplex : Finset beta}
    (chosen : LocalFaceSupport (beta := beta) simplex) :
    LocalFaceOrientationDatum (beta := beta) simplex :=
  fun face => decide (face ≠ chosen)

omit [Fintype beta] in
@[simp] theorem localOrientationFromChosenFace_at_chosen {simplex : Finset beta}
    (chosen : LocalFaceSupport (beta := beta) simplex) :
    localOrientationFromChosenFace (beta := beta) chosen chosen = false := by
  simp [localOrientationFromChosenFace]

omit [Fintype beta] in
@[simp] theorem localOrientationFromChosenFace_at_other {simplex : Finset beta}
    (chosen face : LocalFaceSupport (beta := beta) simplex)
    (hface : face ≠ chosen) :
    localOrientationFromChosenFace (beta := beta) chosen face = true := by
  simp [localOrientationFromChosenFace, hface]

omit [Fintype beta] in
theorem leftPairFaceSupport_ne_rightPairFaceSupport
    {x y : beta} (hxy : x ≠ y) :
    leftPairFaceSupport (beta := beta) hxy ≠
      rightPairFaceSupport (beta := beta) hxy := by
  intro h
  have hset :
      (leftPairFaceSupport (beta := beta) hxy).1 =
        (rightPairFaceSupport (beta := beta) hxy).1 := by
    exact congrArg Subtype.val h
  have hset' : ({x} : Finset beta) = ({y} : Finset beta) := by
    simpa [leftPairFaceSupport, rightPairFaceSupport] using hset
  have hx : x ∈ ({x} : Finset beta) := by
    simp
  have hy : x ∈ ({y} : Finset beta) := by
    simpa [hset'] using hx
  exact hxy (by simpa using hy)

omit [Fintype beta] in
theorem rightPairFaceSupport_ne_leftPairFaceSupport
    {x y : beta} (hxy : x ≠ y) :
    rightPairFaceSupport (beta := beta) hxy ≠
      leftPairFaceSupport (beta := beta) hxy :=
  Ne.symm (leftPairFaceSupport_ne_rightPairFaceSupport (beta := beta) hxy)

def leftChosenPairOrientation {x y : beta} (hxy : x ≠ y) :
    LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta) :=
  localOrientationFromChosenFace (beta := beta) (leftPairFaceSupport (beta := beta) hxy)

def rightChosenPairOrientation {x y : beta} (hxy : x ≠ y) :
    LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta) :=
  localOrientationFromChosenFace (beta := beta) (rightPairFaceSupport (beta := beta) hxy)

omit [Fintype beta] in
theorem leftChosenPairOrientation_alternating
    {x y : beta} (hxy : x ≠ y) :
    PairAlternating (beta := beta) hxy (leftChosenPairOrientation (beta := beta) hxy) := by
  unfold PairAlternating pairFaceSignSum leftChosenPairOrientation
  have hneq :
      rightPairFaceSupport (beta := beta) hxy ≠
        leftPairFaceSupport (beta := beta) hxy :=
    rightPairFaceSupport_ne_leftPairFaceSupport (beta := beta) hxy
  simp [localOrientationFromChosenFace, hneq, orientationSignValue]

omit [Fintype beta] in
theorem rightChosenPairOrientation_alternating
    {x y : beta} (hxy : x ≠ y) :
    PairAlternating (beta := beta) hxy (rightChosenPairOrientation (beta := beta) hxy) := by
  unfold PairAlternating pairFaceSignSum rightChosenPairOrientation
  have hneq :
      leftPairFaceSupport (beta := beta) hxy ≠
        rightPairFaceSupport (beta := beta) hxy :=
    leftPairFaceSupport_ne_rightPairFaceSupport (beta := beta) hxy
  simp [localOrientationFromChosenFace, hneq, orientationSignValue]

omit [Fintype beta] in
theorem exists_pairAlternatingOrientation
    {x y : beta} (hxy : x ≠ y) :
    ∃ theta : LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta),
      PairAlternating (beta := beta) hxy theta := by
  exact ⟨leftChosenPairOrientation (beta := beta) hxy,
    leftChosenPairOrientation_alternating (beta := beta) hxy⟩

/-- One explicit chosen codimension-one face per edge is enough to define a
local edge orientation rule. This is a minimal honest asymmetry datum for the
edge-level boundary layer. -/
abbrev EdgeEndpointChoiceDatum (m : Nat) (U : Cover beta alpha) :=
  ∀ edge : ↑(edgeSimplices m U), LocalFaceSupport (beta := beta) edge.1

/-- Induce edge-local face signs from one chosen face per edge. -/
def inducedEdgeLocalOrientations (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U) :
    ∀ edge : ↑(edgeSimplices m U), LocalFaceOrientationDatum (beta := beta) edge.1 :=
  fun edge => localOrientationFromChosenFace (beta := beta) (chi edge)

@[simp] theorem inducedEdgeLocalOrientations_at_choice
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (edge : ↑(edgeSimplices m U)) :
    inducedEdgeLocalOrientations (beta := beta) m U chi edge (chi edge) = false := by
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace]

theorem localFaceSupport_nonempty_of_edge
    (m : Nat) (U : Cover beta alpha)
    (edge : ↑(edgeSimplices m U)) :
    Nonempty (LocalFaceSupport (beta := beta) edge.1) := by
  have hcard : (codimOneFaces (beta := beta) edge.1).card = 2 := by
    exact edge_boundary_face_count (beta := beta) (m := m) (U := U) edge.2
  have hpos : 0 < (codimOneFaces (beta := beta) edge.1).card := by
    rw [hcard]
    norm_num
  rcases Finset.card_pos.mp hpos with ⟨face, hface⟩
  exact ⟨⟨face, hface⟩⟩

/-- A global endpoint-choice datum always exists, but only noncanonically:
we use classical choice to pick one codimension-one face from each edge. -/
noncomputable def someEdgeEndpointChoiceDatum (m : Nat) (U : Cover beta alpha) :
    EdgeEndpointChoiceDatum (beta := beta) m U :=
  fun edge => Classical.choice
    (localFaceSupport_nonempty_of_edge (beta := beta) (m := m) (U := U) edge)

theorem edgeEndpointChoiceDatum_exists
    (m : Nat) (U : Cover beta alpha) :
    Nonempty (EdgeEndpointChoiceDatum (beta := beta) m U) := by
  exact ⟨someEdgeEndpointChoiceDatum (beta := beta) m U⟩

end SCT.FND1
