import Mathlib

/-
PROVIDED SOLUTION
All elements pass the positive filter since hpos says all eigenvalues are positive. Use Finset.filter_true_of_mem to show the filter keeps everything, then Finset.card_univ gives n. Alternatively: convert Finset.card_fin n, then show filter = univ via ext.
-/
theorem eta_positive_operator {n : ℕ}
    (eigenvalues : Fin n → ℝ) (hpos : ∀ i, 0 < eigenvalues i) :
    (Finset.univ.filter (fun i => 0 < eigenvalues i)).card = n := by
  simp +decide [ Finset.filter_true_of_mem, hpos ]

/-
PROVIDED SOLUTION
Partition univ into positive and negative eigenvalues. Since no eigenvalue is zero, every eigenvalue is either positive or negative. Show filter pos ∪ filter neg = univ and they're disjoint. Use Finset.filter_union_compl or similar. Key: for each i, eigenvalues i ≠ 0 implies either 0 < eigenvalues i or eigenvalues i < 0 (by lt_or_gt_of_ne (hne i).symm or similar). Then card of the union is the sum of cards, which equals card univ = n.
-/
theorem spectral_signature {n : ℕ}
    (eigenvalues : Fin n → ℝ) (hne : ∀ i, eigenvalues i ≠ 0) :
    (Finset.univ.filter (fun i => 0 < eigenvalues i)).card +
    (Finset.univ.filter (fun i => eigenvalues i < 0)).card = n := by
  rw [ ← Finset.card_union_of_disjoint ];
  · convert Finset.card_fin n ; ext i ; cases lt_or_gt_of_ne ( hne i ) <;> aesop;
  · exact Finset.disjoint_filter.mpr fun _ _ _ _ => by linarith;