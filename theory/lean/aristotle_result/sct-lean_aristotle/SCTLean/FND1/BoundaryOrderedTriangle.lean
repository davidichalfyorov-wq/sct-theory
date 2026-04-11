import SCTLean.FND1.BoundaryEndpointChoice

/-!
# FND-1 Boundary Ordered Triangle

This module gives a concrete local compatibility witness on a single ordered
triangle. It does not claim canonicality. It records a sufficient stronger
datum: an explicit ordered triple of distinct vertices.
-/
namespace SCT.FND1

universe v

variable {beta : Type v} [DecidableEq beta]

/-- On an ordered edge `(a,b)`, the positive endpoint is `b`. -/
def orderedEdgeOrientation {a b : beta} (hab : a ≠ b) :
    LocalFaceOrientationDatum (beta := beta) ({a, b} : Finset beta) :=
  localOrientationFromChosenFace (beta := beta) (rightPairFaceSupport (beta := beta) hab)

theorem orderedEdgeOrientation_left_coeff {a b : beta} (hab : a ≠ b) :
    orientationSignValue
      (orderedEdgeOrientation (beta := beta) hab (leftPairFaceSupport (beta := beta) hab)) = -1 := by
  unfold orderedEdgeOrientation
  simp [localOrientationFromChosenFace, leftPairFaceSupport_ne_rightPairFaceSupport, orientationSignValue]

theorem orderedEdgeOrientation_right_coeff {a b : beta} (hab : a ≠ b) :
    orientationSignValue
      (orderedEdgeOrientation (beta := beta) hab (rightPairFaceSupport (beta := beta) hab)) = 1 := by
  unfold orderedEdgeOrientation
  simp [localOrientationFromChosenFace, orientationSignValue]

theorem triangleFaceAB_mem_codimOneFaces
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    ({a, b} : Finset beta) ∈ codimOneFaces (beta := beta) ({a, b, c} : Finset beta) := by
  rw [mem_codimOneFaces_iff]
  constructor
  · intro z hz
    simp at hz ⊢
    rcases hz with rfl | rfl
    · simp
    · simp
  · simp [hab, hac, hbc]

theorem triangleFaceAC_mem_codimOneFaces
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    ({a, c} : Finset beta) ∈ codimOneFaces (beta := beta) ({a, b, c} : Finset beta) := by
  rw [mem_codimOneFaces_iff]
  constructor
  · intro z hz
    simp at hz ⊢
    rcases hz with rfl | rfl
    · simp
    · simp
  · simp [hab, hac, hbc]

theorem triangleFaceBC_mem_codimOneFaces
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    ({b, c} : Finset beta) ∈ codimOneFaces (beta := beta) ({a, b, c} : Finset beta) := by
  rw [mem_codimOneFaces_iff]
  constructor
  · intro z hz
    simp at hz ⊢
    rcases hz with rfl | rfl
    · simp
    · simp
  · simp [hab, hac, hbc]

def triangleFaceABSupport
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    LocalFaceSupport (beta := beta) ({a, b, c} : Finset beta) :=
  ⟨({a, b} : Finset beta), triangleFaceAB_mem_codimOneFaces (beta := beta) hab hac hbc⟩

def triangleFaceACSupport
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    LocalFaceSupport (beta := beta) ({a, b, c} : Finset beta) :=
  ⟨({a, c} : Finset beta), triangleFaceAC_mem_codimOneFaces (beta := beta) hab hac hbc⟩

def triangleFaceBCSupport
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    LocalFaceSupport (beta := beta) ({a, b, c} : Finset beta) :=
  ⟨({b, c} : Finset beta), triangleFaceBC_mem_codimOneFaces (beta := beta) hab hac hbc⟩

/-- Standard ordered-triangle local sign pattern: `bc - ac + ab`. -/
def orderedTriangleOrientation
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    LocalFaceOrientationDatum (beta := beta) ({a, b, c} : Finset beta) :=
  fun face => decide (face = triangleFaceACSupport (beta := beta) hab hac hbc)

theorem triangleFaceABSupport_ne_triangleFaceACSupport
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    triangleFaceABSupport (beta := beta) hab hac hbc ≠
      triangleFaceACSupport (beta := beta) hab hac hbc := by
  intro h
  have hset :
      (triangleFaceABSupport (beta := beta) hab hac hbc).1 =
        (triangleFaceACSupport (beta := beta) hab hac hbc).1 := by
    exact congrArg Subtype.val h
  have hset' : ({a, b} : Finset beta) = ({a, c} : Finset beta) := by
    simpa [triangleFaceABSupport, triangleFaceACSupport] using hset
  have hb : b ∈ ({a, b} : Finset beta) := by simp
  have hb' : b ∈ ({a, c} : Finset beta) := by simpa [hset'] using hb
  have : b = a ∨ b = c := by simpa using hb'
  rcases this with hba | hbc'
  · exact hab hba.symm
  · exact hbc hbc'

theorem triangleFaceBCSupport_ne_triangleFaceACSupport
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    triangleFaceBCSupport (beta := beta) hab hac hbc ≠
      triangleFaceACSupport (beta := beta) hab hac hbc := by
  intro h
  have hset :
      (triangleFaceBCSupport (beta := beta) hab hac hbc).1 =
        (triangleFaceACSupport (beta := beta) hab hac hbc).1 := by
    exact congrArg Subtype.val h
  have hset' : ({b, c} : Finset beta) = ({a, c} : Finset beta) := by
    simpa [triangleFaceBCSupport, triangleFaceACSupport] using hset
  have hb : b ∈ ({b, c} : Finset beta) := by simp
  have hb' : b ∈ ({a, c} : Finset beta) := by simpa [hset'] using hb
  have : b = a ∨ b = c := by simpa using hb'
  rcases this with hba | hbc'
  · exact hab hba.symm
  · exact hbc hbc'

theorem orderedTriangleOrientation_coeff_ab
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceABSupport (beta := beta) hab hac hbc)) = 1 := by
  unfold orderedTriangleOrientation
  have hneq :=
    triangleFaceABSupport_ne_triangleFaceACSupport (beta := beta) hab hac hbc
  simp [orientationSignValue, hneq]

theorem orderedTriangleOrientation_coeff_ac
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceACSupport (beta := beta) hab hac hbc)) = -1 := by
  unfold orderedTriangleOrientation
  simp [triangleFaceACSupport, orientationSignValue]

theorem orderedTriangleOrientation_coeff_bc
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceBCSupport (beta := beta) hab hac hbc)) = 1 := by
  unfold orderedTriangleOrientation
  have hneq :=
    triangleFaceBCSupport_ne_triangleFaceACSupport (beta := beta) hab hac hbc
  simp [orientationSignValue, hneq]

def orderedTriangleVertexSumAtA
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : Int :=
  orientationSignValue
      (orderedEdgeOrientation (beta := beta) hab (leftPairFaceSupport (beta := beta) hab)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceABSupport (beta := beta) hab hac hbc)) +
  orientationSignValue
      (orderedEdgeOrientation (beta := beta) hac (leftPairFaceSupport (beta := beta) hac)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceACSupport (beta := beta) hab hac hbc))

def orderedTriangleVertexSumAtB
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : Int :=
  orientationSignValue
      (orderedEdgeOrientation (beta := beta) hab (rightPairFaceSupport (beta := beta) hab)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceABSupport (beta := beta) hab hac hbc)) +
  orientationSignValue
      (orderedEdgeOrientation (beta := beta) hbc (leftPairFaceSupport (beta := beta) hbc)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceBCSupport (beta := beta) hab hac hbc))

def orderedTriangleVertexSumAtC
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : Int :=
  orientationSignValue
      (orderedEdgeOrientation (beta := beta) hac (rightPairFaceSupport (beta := beta) hac)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceACSupport (beta := beta) hab hac hbc)) +
  orientationSignValue
      (orderedEdgeOrientation (beta := beta) hbc (rightPairFaceSupport (beta := beta) hbc)) *
    orientationSignValue
      (orderedTriangleOrientation (beta := beta) hab hac hbc
        (triangleFaceBCSupport (beta := beta) hab hac hbc))

theorem orderedTriangleVertexSumAtA_eq_zero
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orderedTriangleVertexSumAtA (beta := beta) hab hac hbc = 0 := by
  unfold orderedTriangleVertexSumAtA
  rw [orderedEdgeOrientation_left_coeff (beta := beta) hab]
  rw [orderedTriangleOrientation_coeff_ab (beta := beta) hab hac hbc]
  rw [orderedEdgeOrientation_left_coeff (beta := beta) hac]
  rw [orderedTriangleOrientation_coeff_ac (beta := beta) hab hac hbc]
  norm_num

theorem orderedTriangleVertexSumAtB_eq_zero
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orderedTriangleVertexSumAtB (beta := beta) hab hac hbc = 0 := by
  unfold orderedTriangleVertexSumAtB
  rw [orderedEdgeOrientation_right_coeff (beta := beta) hab]
  rw [orderedTriangleOrientation_coeff_ab (beta := beta) hab hac hbc]
  rw [orderedEdgeOrientation_left_coeff (beta := beta) hbc]
  rw [orderedTriangleOrientation_coeff_bc (beta := beta) hab hac hbc]
  norm_num

theorem orderedTriangleVertexSumAtC_eq_zero
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orderedTriangleVertexSumAtC (beta := beta) hab hac hbc = 0 := by
  unfold orderedTriangleVertexSumAtC
  rw [orderedEdgeOrientation_right_coeff (beta := beta) hac]
  rw [orderedTriangleOrientation_coeff_ac (beta := beta) hab hac hbc]
  rw [orderedEdgeOrientation_right_coeff (beta := beta) hbc]
  rw [orderedTriangleOrientation_coeff_bc (beta := beta) hab hac hbc]
  norm_num

theorem orderedTriangle_local_boundary_square_zero
    {a b c : beta} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    orderedTriangleVertexSumAtA (beta := beta) hab hac hbc = 0 ∧
    orderedTriangleVertexSumAtB (beta := beta) hab hac hbc = 0 ∧
    orderedTriangleVertexSumAtC (beta := beta) hab hac hbc = 0 := by
  constructor
  · exact orderedTriangleVertexSumAtA_eq_zero (beta := beta) hab hac hbc
  constructor
  · exact orderedTriangleVertexSumAtB_eq_zero (beta := beta) hab hac hbc
  · exact orderedTriangleVertexSumAtC_eq_zero (beta := beta) hab hac hbc

end SCT.FND1
