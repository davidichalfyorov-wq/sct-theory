import Mathlib

/-
PROVIDED SOLUTION
The function x ↦ exp(-(x²)) is the composition of exp with the smooth function x ↦ -(x²). Use ContDiff.exp (or Real.contDiff_exp.comp) composed with the ContDiff of negation and squaring. Specifically: contDiff_neg.comp (contDiff_pow 2) gives smoothness of -x², then contDiff_exp.comp gives the result. Or more directly, use that id is smooth, then pow, then neg, then exp.
-/
theorem gaussian_smooth : ContDiff ℝ ⊤ (fun x : ℝ => Real.exp (-(x ^ 2))) := by
  exact ContDiff.exp <| contDiff_neg.comp <| contDiff_id.pow 2

/-
PROVIDED SOLUTION
Show that exp(-|x|) is not differentiable at 0. The right derivative is d/dx exp(-x)|_{x=0} = -1 and left derivative is d/dx exp(x)|_{x=0} = 1. Since these don't match, the function isn't differentiable at 0. Use HasDerivAt or show that the difference quotient has different limits from left and right. One approach: assume differentiable, then the derivative composed with abs would need to exist, but abs is not differentiable at 0 and exp is a diffeomorphism. Alternatively: if f(x) = exp(-|x|) were differentiable at 0, then g(x) = log(f(x)) = -|x| would be differentiable at 0 (since exp is never zero, log is smooth on positive reals, and f(0) = 1 > 0), but |x| is not differentiable at 0, contradiction. Use Real.log_exp or that log ∘ exp(-|·|) = -|·| near 0.
-/
theorem exp_neg_abs_not_diff_at_zero :
    ¬ DifferentiableAt ℝ (fun x : ℝ => Real.exp (-|x|)) 0 := by
  exact fun h => not_differentiableAt_abs_zero ( by simpa using h.log ( by positivity ) )

/-
PROVIDED SOLUTION
The maximum of |x^n * exp(-x²)| over all x ∈ ℝ is achieved at x² = n/2, giving the value (n/(2e))^(n/2). The function f(x) = x^(2m) * exp(-x²) can be maximized by setting u = x², then f = u^m * exp(-u) which has max at u = m, giving m^m * exp(-m) = (m/e)^m. For odd n, similar analysis. The key calculus fact: for t ≥ 0, t^a * exp(-t) ≤ (a/e)^a (with the convention 0^0 = 1). This gives |x|^n * exp(-x²) = (x²)^(n/2) * exp(-x²) ≤ (n/(2e))^(n/2) when we substitute t = x² and a = n/2. Try using the AM-GM inequality or the fact that t/a ≤ exp(t/a - 1).
-/
theorem schwartz_times_poly_is_schwartz (n : ℕ) :
    ∀ x : ℝ, |x ^ n * Real.exp (-(x ^ 2))| ≤ (n / (2 * Real.exp 1)) ^ (n / 2 : ℝ) := by
  -- Set $y = x^2$, then consider the function $h(y) = y^{n/2} * e^{-y}$ and show that it's bounded above by $(n/(2e))^{n/2}$.
  have h_y : ∀ y : ℝ, y ≥ 0 → y ^ (n / 2 : ℝ) * Real.exp (-y) ≤ (n / (2 * Real.exp 1)) ^ (n / 2 : ℝ) := by
    -- We'll use that exponential functions grow faster than polynomial functions under the given conditions.
    intros y hy
    by_cases hn : n = 0;
    · aesop;
    · -- We can simplify the inequality by dividing both sides by $e^{-y}$.
      have h_simplified : y ^ (n / 2 : ℝ) ≤ (n / (2 * Real.exp 1)) ^ (n / 2 : ℝ) * Real.exp y := by
        -- We can simplify the inequality by dividing both sides by $e^{-y}$ and taking the $n/2$-th root.
        have h_simplified : y ≤ (n / (2 * Real.exp 1)) * Real.exp (2 * y / n) := by
          field_simp;
          rw [ show ( y * 2 / n : ℝ ) = 1 + ( y * 2 / n - 1 ) by ring, Real.exp_add ];
          nlinarith [ Real.add_one_le_exp 1, Real.add_one_le_exp ( y * 2 / n - 1 ), show ( n : ℝ ) ≥ 1 by exact Nat.one_le_cast.mpr ( Nat.pos_of_ne_zero hn ), mul_div_cancel₀ ( y * 2 ) ( Nat.cast_ne_zero.mpr hn ), mul_le_mul_of_nonneg_left ( Real.add_one_le_exp ( y * 2 / n - 1 ) ) ( Real.exp_nonneg 1 ) ];
        convert Real.rpow_le_rpow ( by positivity ) h_simplified ( by positivity : 0 ≤ ( n : ℝ ) / 2 ) using 1 ; rw [ Real.mul_rpow ( by positivity ) ( by positivity ) ] ; rw [ ← Real.exp_mul ] ; ring_nf ; norm_num [ hn ] ;
        exact Or.inl ( by rw [ mul_right_comm, mul_inv_cancel₀ ( by positivity ), one_mul ] );
      exact le_trans ( mul_le_mul_of_nonneg_right h_simplified <| Real.exp_nonneg _ ) <| by rw [ mul_assoc, ← Real.exp_add ] ; norm_num;
  intro x; specialize h_y ( x ^ 2 ) ( sq_nonneg x ) ; simp_all +decide [ abs_mul, abs_pow ] ;
  convert h_y using 1 ; rw [ ← Real.sqrt_sq_eq_abs ] ; rw [ Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul ( by positivity ) ] ; ring