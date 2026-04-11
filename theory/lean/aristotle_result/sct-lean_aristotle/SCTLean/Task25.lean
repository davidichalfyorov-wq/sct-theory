import Mathlib.Tactic
import Mathlib.RingTheory.Polynomial.Basic

/-
PROVIDED SOLUTION
Apply star (complex conjugation) to both sides of hz. star 0 = 0. For the LHS, use that aeval commutes with star for real polynomials mapped to ℂ. Key: p.map (algebraMap ℝ ℂ) has coefficients in the image of algebraMap ℝ ℂ, so star fixes each coefficient. Use Polynomial.eval₂_map, or manually show that star (aeval z (p.map f)) = aeval (star z) ((p.map f).map star) = aeval (star z) (p.map (star ∘ f)) = aeval (star z) (p.map f) since star ∘ algebraMap ℝ ℂ = algebraMap ℝ ℂ (because Complex.conj_ofReal).
-/
theorem conj_root_of_real_poly (p : Polynomial ℝ) (z : ℂ)
    (hz : Polynomial.aeval z (p.map (algebraMap ℝ ℂ)) = 0) :
    Polynomial.aeval (starRingEnd ℂ z) (p.map (algebraMap ℝ ℂ)) = 0 := by
  convert congr_arg Star.star hz using 1 ; norm_num [ Polynomial.aeval_def, Polynomial.eval₂_eq_sum_range ] ; ring;
  norm_num +zetaDelta at *

/-
PROBLEM
The original statement used `Finset.card` on `Multiset.filter`, which is a type error
    since `Polynomial.roots` returns a `Multiset`, not a `Finset`.
    Corrected to use `Multiset.card` instead.

PROVIDED SOLUTION
The non-real roots of a real polynomial come in conjugate pairs. For each root z with z.im ≠ 0, its conjugate conj(z) is also a root (by conj_root_of_real_poly) and conj(z).im = -z.im ≠ 0, and z ≠ conj(z). So the non-real roots pair up, giving an even count in the multiset.
-/
theorem even_nonreal_roots (p : Polynomial ℝ) (hp : p ≠ 0) :
    Even (Multiset.card (Polynomial.roots (p.map (algebraMap ℝ ℂ)) |>.filter
      (fun z => z.im ≠ 0))) := by
  -- Let $q(x)$ be the polynomial with real coefficients such that $q(x) = p(x)$.
  set q : Polynomial ℝ := p;
  -- Since $q$ is a polynomial with real coefficients, its roots come in conjugate pairs. Therefore, the number of non-real roots must be even.
  have h_conj_pairs : Multiset.card (Multiset.filter (fun z => z.im > 0) (Polynomial.map (algebraMap ℝ ℂ) q).roots) = Multiset.card (Multiset.filter (fun z => z.im < 0) (Polynomial.map (algebraMap ℝ ℂ) q).roots) := by
    -- Since $q$ is a polynomial with real coefficients, the roots of $q$ in the complex plane are symmetric with respect to the real axis.
    have h_symm : Multiset.map (fun z => starRingEnd ℂ z) (Polynomial.map (algebraMap ℝ ℂ) q).roots = (Polynomial.map (algebraMap ℝ ℂ) q).roots := by
      have h_conj_root : Polynomial.map (starRingEnd ℂ) (Polynomial.map (algebraMap ℝ ℂ) q) = Polynomial.map (algebraMap ℝ ℂ) q := by
        ext; simp [Polynomial.map_map];
      rw [ ← Polynomial.roots_map ] ; aesop;
      exact?;
    rw [ ← h_symm, Multiset.filter_map ] ; aesop;
  -- The non-real roots of $q$ are exactly those roots with non-zero imaginary part.
  have h_non_real_roots : Multiset.filter (fun z => z.im ≠ 0) (Polynomial.map (algebraMap ℝ ℂ) q).roots = Multiset.filter (fun z => z.im > 0) (Polynomial.map (algebraMap ℝ ℂ) q).roots + Multiset.filter (fun z => z.im < 0) (Polynomial.map (algebraMap ℝ ℂ) q).roots := by
    ext z;
    by_cases h : z.im = 0 <;> simp +decide [ h ];
    cases lt_or_gt_of_ne h <;> simp_all +decide [ Multiset.count_filter ];
    · intros; linarith;
    · intros; linarith;
  aesop