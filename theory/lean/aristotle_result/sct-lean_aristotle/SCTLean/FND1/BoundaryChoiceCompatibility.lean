import SCTLean.FND1.BoundaryTriangleChoice

/-!
# FND-1 Boundary Choice Compatibility

This module begins the real bridge between the global edge-choice layer and the
global triangle-choice layer. It does not yet prove a whole-nerve compatibility
theorem. It isolates the local notion that an edge-choice datum agrees with the
ordered witness carried by a specific triangle.
-/
namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

omit [Fintype beta] in
@[simp] theorem leftPairFaceSupport_val {x y : beta} (hxy : x ≠ y) :
    (leftPairFaceSupport (beta := beta) hxy).1 = ({x} : Finset beta) := rfl

omit [Fintype beta] in
@[simp] theorem rightPairFaceSupport_val {x y : beta} (hxy : x ≠ y) :
    (rightPairFaceSupport (beta := beta) hxy).1 = ({y} : Finset beta) := rfl

def TriangleOrderedWitness.edgeAB
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ↑(edgeSimplices m U) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  exact ⟨({a, b} : Finset beta),
    by
      have hface : ({a, b} : Finset beta) ∈ codimOneFaces (beta := beta) triangle.1 := by
        simpa [htri] using
          (triangleFaceAB_mem_codimOneFaces (beta := beta) hab hac hbc)
      simpa [htri] using
        (triangle_boundary_faces_are_edges (beta := beta) (m := m) (U := U) triangle.2
          hface)⟩

def TriangleOrderedWitness.edgeAC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ↑(edgeSimplices m U) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  exact ⟨({a, c} : Finset beta),
    by
      have hface : ({a, c} : Finset beta) ∈ codimOneFaces (beta := beta) triangle.1 := by
        simpa [htri] using
          (triangleFaceAC_mem_codimOneFaces (beta := beta) hab hac hbc)
      simpa [htri] using
        (triangle_boundary_faces_are_edges (beta := beta) (m := m) (U := U) triangle.2
          hface)⟩

def TriangleOrderedWitness.edgeBC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    ↑(edgeSimplices m U) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  exact ⟨({b, c} : Finset beta),
    by
      have hface : ({b, c} : Finset beta) ∈ codimOneFaces (beta := beta) triangle.1 := by
        simpa [htri] using
          (triangleFaceBC_mem_codimOneFaces (beta := beta) hab hac hbc)
      simpa [htri] using
        (triangle_boundary_faces_are_edges (beta := beta) (m := m) (U := U) triangle.2
          hface)⟩

@[simp] theorem TriangleOrderedWitness.edgeAB_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAB (beta := beta) m U triangle).1 = ({w.a, w.b} : Finset beta) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  simp [TriangleOrderedWitness.edgeAB]

@[simp] theorem TriangleOrderedWitness.edgeAC_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAC (beta := beta) m U triangle).1 = ({w.a, w.c} : Finset beta) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  simp [TriangleOrderedWitness.edgeAC]

@[simp] theorem TriangleOrderedWitness.edgeBC_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeBC (beta := beta) m U triangle).1 = ({w.b, w.c} : Finset beta) := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  simp [TriangleOrderedWitness.edgeBC]

structure EdgeChoicesCompatibleWithTriangleWitness
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) : Prop where
  edgeAB_choice : (chi (w.edgeAB (beta := beta) m U triangle)).1 = ({w.b} : Finset beta)
  edgeAC_choice : (chi (w.edgeAC (beta := beta) m U triangle)).1 = ({w.c} : Finset beta)
  edgeBC_choice : (chi (w.edgeBC (beta := beta) m U triangle)).1 = ({w.c} : Finset beta)

def TriangleOrderedWitness.edgeAB_leftFace
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    LocalFaceSupport (beta := beta) (w.edgeAB (beta := beta) m U triangle).1 := by
  simpa using leftPairFaceSupport (beta := beta) w.hab

def TriangleOrderedWitness.edgeAB_rightFace
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    LocalFaceSupport (beta := beta) (w.edgeAB (beta := beta) m U triangle).1 := by
  simpa using rightPairFaceSupport (beta := beta) w.hab

def TriangleOrderedWitness.edgeAC_leftFace
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    LocalFaceSupport (beta := beta) (w.edgeAC (beta := beta) m U triangle).1 := by
  simpa using leftPairFaceSupport (beta := beta) w.hac

def TriangleOrderedWitness.edgeAC_rightFace
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    LocalFaceSupport (beta := beta) (w.edgeAC (beta := beta) m U triangle).1 := by
  simpa using rightPairFaceSupport (beta := beta) w.hac

def TriangleOrderedWitness.edgeBC_leftFace
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    LocalFaceSupport (beta := beta) (w.edgeBC (beta := beta) m U triangle).1 := by
  simpa using leftPairFaceSupport (beta := beta) w.hbc

def TriangleOrderedWitness.edgeBC_rightFace
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    LocalFaceSupport (beta := beta) (w.edgeBC (beta := beta) m U triangle).1 := by
  simpa using rightPairFaceSupport (beta := beta) w.hbc

@[simp] theorem TriangleOrderedWitness.edgeAB_leftFace_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAB_leftFace (beta := beta) m U triangle).1 = ({w.a} : Finset beta) := by
  simp [TriangleOrderedWitness.edgeAB_leftFace]

@[simp] theorem TriangleOrderedWitness.edgeAB_rightFace_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAB_rightFace (beta := beta) m U triangle).1 = ({w.b} : Finset beta) := by
  simp [TriangleOrderedWitness.edgeAB_rightFace]

@[simp] theorem TriangleOrderedWitness.edgeAC_leftFace_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAC_leftFace (beta := beta) m U triangle).1 = ({w.a} : Finset beta) := by
  simp [TriangleOrderedWitness.edgeAC_leftFace]

@[simp] theorem TriangleOrderedWitness.edgeAC_rightFace_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeAC_rightFace (beta := beta) m U triangle).1 = ({w.c} : Finset beta) := by
  simp [TriangleOrderedWitness.edgeAC_rightFace]

@[simp] theorem TriangleOrderedWitness.edgeBC_leftFace_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeBC_leftFace (beta := beta) m U triangle).1 = ({w.b} : Finset beta) := by
  simp [TriangleOrderedWitness.edgeBC_leftFace]

@[simp] theorem TriangleOrderedWitness.edgeBC_rightFace_val
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    (w.edgeBC_rightFace (beta := beta) m U triangle).1 = ({w.c} : Finset beta) := by
  simp [TriangleOrderedWitness.edgeBC_rightFace]

theorem compatible_edgeAB_left_coeff
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAB (beta := beta) m U triangle))
        (w.edgeAB_leftFace (beta := beta) m U triangle)) = -1 := by
  have hneq :
      w.edgeAB_leftFace (beta := beta) m U triangle ≠ chi (w.edgeAB (beta := beta) m U triangle) := by
    intro h
    have hset :
        (w.edgeAB_leftFace (beta := beta) m U triangle).1 =
          (chi (w.edgeAB (beta := beta) m U triangle)).1 := by
      exact congrArg Subtype.val h
    have hsingleton : ({w.a} : Finset beta) = ({w.b} : Finset beta) := by
      simpa [hcompat.edgeAB_choice] using hset
    have ha : w.a ∈ ({w.a} : Finset beta) := by simp
    have ha' : w.a ∈ ({w.b} : Finset beta) := by simpa [hsingleton] using ha
    exact w.hab (by simpa using ha')
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace, hneq, orientationSignValue]

theorem compatible_edgeAB_right_coeff
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAB (beta := beta) m U triangle))
        (w.edgeAB_rightFace (beta := beta) m U triangle)) = 1 := by
  have hchoice :
      chi (w.edgeAB (beta := beta) m U triangle) =
        w.edgeAB_rightFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simpa using hcompat.edgeAB_choice
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace, hchoice, orientationSignValue]

theorem compatible_edgeAC_left_coeff
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAC (beta := beta) m U triangle))
        (w.edgeAC_leftFace (beta := beta) m U triangle)) = -1 := by
  have hneq :
      w.edgeAC_leftFace (beta := beta) m U triangle ≠ chi (w.edgeAC (beta := beta) m U triangle) := by
    intro h
    have hset :
        (w.edgeAC_leftFace (beta := beta) m U triangle).1 =
          (chi (w.edgeAC (beta := beta) m U triangle)).1 := by
      exact congrArg Subtype.val h
    have hsingleton : ({w.a} : Finset beta) = ({w.c} : Finset beta) := by
      simpa [hcompat.edgeAC_choice] using hset
    have ha : w.a ∈ ({w.a} : Finset beta) := by simp
    have ha' : w.a ∈ ({w.c} : Finset beta) := by simpa [hsingleton] using ha
    exact w.hac (by simpa using ha')
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace, hneq, orientationSignValue]

theorem compatible_edgeAC_right_coeff
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAC (beta := beta) m U triangle))
        (w.edgeAC_rightFace (beta := beta) m U triangle)) = 1 := by
  have hchoice :
      chi (w.edgeAC (beta := beta) m U triangle) =
        w.edgeAC_rightFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simpa using hcompat.edgeAC_choice
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace, hchoice, orientationSignValue]

theorem compatible_edgeBC_left_coeff
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeBC (beta := beta) m U triangle))
        (w.edgeBC_leftFace (beta := beta) m U triangle)) = -1 := by
  have hneq :
      w.edgeBC_leftFace (beta := beta) m U triangle ≠ chi (w.edgeBC (beta := beta) m U triangle) := by
    intro h
    have hset :
        (w.edgeBC_leftFace (beta := beta) m U triangle).1 =
          (chi (w.edgeBC (beta := beta) m U triangle)).1 := by
      exact congrArg Subtype.val h
    have hsingleton : ({w.b} : Finset beta) = ({w.c} : Finset beta) := by
      simpa [hcompat.edgeBC_choice] using hset
    have hb : w.b ∈ ({w.b} : Finset beta) := by simp
    have hb' : w.b ∈ ({w.c} : Finset beta) := by simpa [hsingleton] using hb
    exact w.hbc (by simpa using hb')
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace, hneq, orientationSignValue]

theorem compatible_edgeBC_right_coeff
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeBC (beta := beta) m U triangle))
        (w.edgeBC_rightFace (beta := beta) m U triangle)) = 1 := by
  have hchoice :
      chi (w.edgeBC (beta := beta) m U triangle) =
        w.edgeBC_rightFace (beta := beta) m U triangle := by
    apply Subtype.ext
    simpa using hcompat.edgeBC_choice
  simp [inducedEdgeLocalOrientations, localOrientationFromChosenFace, hchoice, orientationSignValue]

def compatibleVertexSumAtA
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) : Int :=
  orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAB (beta := beta) m U triangle))
        (w.edgeAB_leftFace (beta := beta) m U triangle)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc
        (triangleFaceABSupport (beta := beta) w.hab w.hac w.hbc)) +
  orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAC (beta := beta) m U triangle))
        (w.edgeAC_leftFace (beta := beta) m U triangle)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc
        (triangleFaceACSupport (beta := beta) w.hab w.hac w.hbc))

def compatibleVertexSumAtB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) : Int :=
  orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAB (beta := beta) m U triangle))
        (w.edgeAB_rightFace (beta := beta) m U triangle)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc
        (triangleFaceABSupport (beta := beta) w.hab w.hac w.hbc)) +
  orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeBC (beta := beta) m U triangle))
        (w.edgeBC_leftFace (beta := beta) m U triangle)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc
        (triangleFaceBCSupport (beta := beta) w.hab w.hac w.hbc))

def compatibleVertexSumAtC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) : Int :=
  orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeAC (beta := beta) m U triangle))
        (w.edgeAC_rightFace (beta := beta) m U triangle)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc
        (triangleFaceACSupport (beta := beta) w.hab w.hac w.hbc)) +
  orientationSignValue
      ((inducedEdgeLocalOrientations (beta := beta) m U chi
        (w.edgeBC (beta := beta) m U triangle))
        (w.edgeBC_rightFace (beta := beta) m U triangle)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc
        (triangleFaceBCSupport (beta := beta) w.hab w.hac w.hbc))

theorem compatibleVertexSumAtA_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    compatibleVertexSumAtA (beta := beta) m U chi triangle w = 0 := by
  unfold compatibleVertexSumAtA
  rw [compatible_edgeAB_left_coeff (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := w) hcompat]
  rw [orderedTriangleOrientation_coeff_ab (beta := beta) w.hab w.hac w.hbc]
  rw [compatible_edgeAC_left_coeff (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := w) hcompat]
  rw [orderedTriangleOrientation_coeff_ac (beta := beta) w.hab w.hac w.hbc]
  norm_num

theorem compatibleVertexSumAtB_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    compatibleVertexSumAtB (beta := beta) m U chi triangle w = 0 := by
  unfold compatibleVertexSumAtB
  rw [compatible_edgeAB_right_coeff (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := w) hcompat]
  rw [orderedTriangleOrientation_coeff_ab (beta := beta) w.hab w.hac w.hbc]
  rw [compatible_edgeBC_left_coeff (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := w) hcompat]
  rw [orderedTriangleOrientation_coeff_bc (beta := beta) w.hab w.hac w.hbc]
  norm_num

theorem compatibleVertexSumAtC_eq_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    compatibleVertexSumAtC (beta := beta) m U chi triangle w = 0 := by
  unfold compatibleVertexSumAtC
  rw [compatible_edgeAC_right_coeff (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := w) hcompat]
  rw [orderedTriangleOrientation_coeff_ac (beta := beta) w.hab w.hac w.hbc]
  rw [compatible_edgeBC_right_coeff (beta := beta) (m := m) (U := U) (chi := chi)
    (triangle := triangle) (w := w) hcompat]
  rw [orderedTriangleOrientation_coeff_bc (beta := beta) w.hab w.hac w.hbc]
  norm_num

theorem compatible_triangle_local_boundary_square_zero
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hcompat : EdgeChoicesCompatibleWithTriangleWitness
      (beta := beta) m U chi triangle w) :
    compatibleVertexSumAtA (beta := beta) m U chi triangle w = 0 ∧
    compatibleVertexSumAtB (beta := beta) m U chi triangle w = 0 ∧
    compatibleVertexSumAtC (beta := beta) m U chi triangle w = 0 := by
  constructor
  · exact compatibleVertexSumAtA_eq_zero (beta := beta) (m := m) (U := U)
      (chi := chi) (triangle := triangle) (w := w) hcompat
  constructor
  · exact compatibleVertexSumAtB_eq_zero (beta := beta) (m := m) (U := U)
      (chi := chi) (triangle := triangle) (w := w) hcompat
  · exact compatibleVertexSumAtC_eq_zero (beta := beta) (m := m) (U := U)
      (chi := chi) (triangle := triangle) (w := w) hcompat

end SCT.FND1
