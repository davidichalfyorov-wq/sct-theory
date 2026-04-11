import SCTLean.FND1.BoundaryLocalOrientation

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

def relabelLocalFaceSupport (sigma : Equiv.Perm beta) {simplex : Finset beta}
    (hfix : simplexImage (beta := beta) sigma simplex = simplex) :
    LocalFaceSupport (beta := beta) simplex → LocalFaceSupport (beta := beta) simplex := by
  intro face
  have hpImg :
      simplexImage (beta := beta) sigma face.1 ∈
        (codimOneFaces (beta := beta) simplex).image (simplexImage (beta := beta) sigma) := by
    exact Finset.mem_image.mpr ⟨face.1, face.2, rfl⟩
  have hcodim :
      simplexImage (beta := beta) sigma face.1 ∈
        codimOneFaces (beta := beta) (simplexImage (beta := beta) sigma simplex) := by
    simpa [codimOneFaces_image (beta := beta) (sigma := sigma) simplex] using hpImg
  refine ⟨simplexImage (beta := beta) sigma face.1, ?_⟩
  simpa [hfix] using hcodim

def LocalFaceDatumInvariantUnder (sigma : Equiv.Perm beta) {simplex : Finset beta}
    (hfix : simplexImage (beta := beta) sigma simplex = simplex)
    (theta : LocalFaceOrientationDatum (beta := beta) simplex) : Prop :=
  ∀ face, theta (relabelLocalFaceSupport (beta := beta) sigma hfix face) = theta face

omit [Fintype beta] in
theorem simplexImage_swap_pair {x y : beta} :
    simplexImage (beta := beta) (Equiv.swap x y) ({x, y} : Finset beta) =
      ({x, y} : Finset beta) := by
  ext z
  simp [simplexImage, eq_comm, or_comm]

omit [Fintype beta] in
theorem leftPairFace_mem_codimOneFaces {x y : beta} (hxy : x ≠ y) :
    ({x} : Finset beta) ∈ codimOneFaces (beta := beta) ({x, y} : Finset beta) := by
  rw [mem_codimOneFaces_iff]
  constructor
  · intro z hz
    simp at hz ⊢
    rcases hz with rfl
    simp
  · have hcard : ({x, y} : Finset beta).card = 2 := by
      simp [hxy]
    rw [hcard]
    simp

omit [Fintype beta] in
theorem rightPairFace_mem_codimOneFaces {x y : beta} (hxy : x ≠ y) :
    ({y} : Finset beta) ∈ codimOneFaces (beta := beta) ({x, y} : Finset beta) := by
  rw [mem_codimOneFaces_iff]
  constructor
  · intro z hz
    simp at hz ⊢
    rcases hz with rfl
    simp
  · have hcard : ({x, y} : Finset beta).card = 2 := by
      simp [hxy]
    rw [hcard]
    simp

def leftPairFaceSupport {x y : beta} (hxy : x ≠ y) :
    LocalFaceSupport (beta := beta) ({x, y} : Finset beta) :=
  ⟨({x} : Finset beta), leftPairFace_mem_codimOneFaces (beta := beta) hxy⟩

def rightPairFaceSupport {x y : beta} (hxy : x ≠ y) :
    LocalFaceSupport (beta := beta) ({x, y} : Finset beta) :=
  ⟨({y} : Finset beta), rightPairFace_mem_codimOneFaces (beta := beta) hxy⟩

omit [Fintype beta] in
theorem relabel_leftPairFaceSupport_swap {x y : beta} (hxy : x ≠ y) :
    relabelLocalFaceSupport (beta := beta) (Equiv.swap x y)
      (simplex := ({x, y} : Finset beta))
      (simplexImage_swap_pair (beta := beta))
      (leftPairFaceSupport (beta := beta) hxy) =
    rightPairFaceSupport (beta := beta) hxy := by
  apply Subtype.ext
  ext z
  simp [relabelLocalFaceSupport, leftPairFaceSupport, rightPairFaceSupport, simplexImage]

omit [Fintype beta] in
theorem relabel_rightPairFaceSupport_swap {x y : beta} (hxy : x ≠ y) :
    relabelLocalFaceSupport (beta := beta) (Equiv.swap x y)
      (simplex := ({x, y} : Finset beta))
      (simplexImage_swap_pair (beta := beta))
      (rightPairFaceSupport (beta := beta) hxy) =
    leftPairFaceSupport (beta := beta) hxy := by
  apply Subtype.ext
  ext z
  simp [relabelLocalFaceSupport, leftPairFaceSupport, rightPairFaceSupport, simplexImage]

def pairFaceSignSum {x y : beta} (hxy : x ≠ y)
    (theta : LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta)) : Int :=
  orientationSignValue (theta (leftPairFaceSupport (beta := beta) hxy)) +
    orientationSignValue (theta (rightPairFaceSupport (beta := beta) hxy))

def PairAlternating {x y : beta} (hxy : x ≠ y)
    (theta : LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta)) : Prop :=
  pairFaceSignSum (beta := beta) hxy theta = 0

omit [Fintype beta] in
theorem pairFaceSigns_equal_of_swap_invariant
    {x y : beta} (hxy : x ≠ y)
    (theta : LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta))
    (hinv : LocalFaceDatumInvariantUnder (beta := beta) (Equiv.swap x y)
      (simplex := ({x, y} : Finset beta))
      (simplexImage_swap_pair (beta := beta)) theta) :
    theta (rightPairFaceSupport (beta := beta) hxy) =
      theta (leftPairFaceSupport (beta := beta) hxy) := by
  simpa [relabel_leftPairFaceSupport_swap (beta := beta) hxy] using
    hinv (leftPairFaceSupport (beta := beta) hxy)

omit [Fintype beta] in
theorem pairFaceSignSum_ne_zero_of_swap_invariant
    {x y : beta} (hxy : x ≠ y)
    (theta : LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta))
    (hinv : LocalFaceDatumInvariantUnder (beta := beta) (Equiv.swap x y)
      (simplex := ({x, y} : Finset beta))
      (simplexImage_swap_pair (beta := beta)) theta) :
    pairFaceSignSum (beta := beta) hxy theta ≠ 0 := by
  have heq := pairFaceSigns_equal_of_swap_invariant (beta := beta) hxy theta hinv
  cases hleft : theta (leftPairFaceSupport (beta := beta) hxy) <;>
    simp [pairFaceSignSum, orientationSignValue, hleft, heq]

omit [Fintype beta] in
theorem no_swap_invariant_alternating_pair_orientation
    {x y : beta} (hxy : x ≠ y) :
    ¬ ∃ theta : LocalFaceOrientationDatum (beta := beta) ({x, y} : Finset beta),
        LocalFaceDatumInvariantUnder (beta := beta) (Equiv.swap x y)
          (simplex := ({x, y} : Finset beta))
          (simplexImage_swap_pair (beta := beta)) theta ∧
        PairAlternating (beta := beta) hxy theta := by
  intro h
  rcases h with ⟨theta, hinv, halt⟩
  exact pairFaceSignSum_ne_zero_of_swap_invariant (beta := beta) hxy theta hinv halt

end SCT.FND1
