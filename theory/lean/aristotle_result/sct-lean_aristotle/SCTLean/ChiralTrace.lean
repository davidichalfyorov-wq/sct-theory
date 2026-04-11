import Mathlib

/-
PROVIDED SOLUTION
We need D.mulVecLin (P.mulVecLin v) = (-ev) • (P.mulVecLin v). Key: mulVecLin corresponds to matrix-vector multiplication. D.mulVecLin (P.mulVecLin v) = D.mulVecLin (P *ᵥ v). Since mulVecLin is the linear map for mulVec, and (D*P) *ᵥ v = D *ᵥ (P *ᵥ v), we have D.mulVecLin (P.mulVecLin v) corresponds to (D * P) *ᵥ v. By hanti, D * P = -(P * D). So we get (-(P*D)) *ᵥ v = -(P *ᵥ (D *ᵥ v)). By hev, D *ᵥ v = D.mulVecLin v = ev • v. So -(P *ᵥ (ev • v)) = -(ev • (P *ᵥ v)) = (-ev) • (P *ᵥ v). Use ext to reduce to pointwise, then simp with Matrix.mulVecLin_apply, Matrix.mulVec_mulVec etc.
-/
theorem eigenvalue_pairing {n : ℕ} {R : Type*} [CommRing R]
    (D P : Matrix (Fin n) (Fin n) R)
    (hP_sq : P * P = 1)
    (hanti : D * P = -(P * D))
    (v : Fin n → R) (ev : R)
    (hev : D.mulVecLin v = ev • v) :
    D.mulVecLin (P.mulVecLin v) = (-ev) • (P.mulVecLin v) := by
  -- By Lemma 26.3, $D.mulVecLin (P.mulVecLin v) = -(P.mulVecLin (D.mulVecLin v))$
  have h_mul : D.mulVecLin (P.mulVecLin v) = -(P.mulVecLin (D.mulVecLin v)) := by
    convert congr_arg ( fun m => m.mulVec v ) hanti using 1 ; simp +decide [ Matrix.mulVec_mulVec, ← Matrix.mul_assoc ] ;
    simp +decide [ Matrix.neg_mulVec, Matrix.mulVec_neg ];
  simp_all +decide [ Matrix.mulVec_smul ]

/-
PROVIDED SOLUTION
Unfold the let bindings. P*M = fromBlocks(1,0,0,-1) * fromBlocks(L,0,0,L). By Matrix.fromBlocks_multiply this equals fromBlocks(1*L + 0*0, 1*0 + 0*L, 0*L + (-1)*0, 0*0 + (-1)*L) = fromBlocks(L, 0, 0, -L). The trace of a block diagonal fromBlocks(A, 0, 0, B) over Fin n ⊕ Fin n equals trace A + trace B = trace L + trace(-L) = trace L - trace L = 0. For the trace computation, unfold Matrix.trace as the sum of diagonal entries over Fin n ⊕ Fin n, split into Sum.inl and Sum.inr parts. Use Fintype.sum_sum_type and Matrix.fromBlocks_apply to reduce.
-/
theorem chiral_trace_zero_paired {n : ℕ}
    (L : Matrix (Fin n) (Fin n) ℚ) :
    let M := Matrix.fromBlocks L 0 0 L
    let P := Matrix.fromBlocks (1 : Matrix (Fin n) (Fin n) ℚ) 0 0 (-1)
    Matrix.trace (P * M) = 0 := by
  norm_num [ Matrix.trace, Matrix.fromBlocks_multiply ]