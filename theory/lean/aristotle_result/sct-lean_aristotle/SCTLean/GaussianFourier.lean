import Mathlib

/-- The error function erf(x) = (2/√π) ∫₀ˣ exp(-t²) dt -/
noncomputable def Real.erf (x : ℝ) : ℝ :=
  (2 / Real.sqrt Real.pi) * ∫ t in (0)..x, Real.exp (-(t ^ 2))

/-
PROBLEM
Show that ∫_{k>0} exp(-k²) sin(kr)/k dk = (π/2) erf(r/2).

PROVIDED SOLUTION
This is a standard result from Fourier analysis. Define I(r) = ∫₀^∞ exp(-k²) sin(kr)/k dk.
Then I'(r) = ∫₀^∞ exp(-k²) cos(kr) dk = (√π/2) exp(-r²/4) (the Gaussian Fourier transform).
So I(r) = (√π/2) ∫₀^r exp(-t²/4) dt. Substituting u = t/2: = (√π/2) · 2 ∫₀^{r/2} exp(-u²) du
= √π · ∫₀^{r/2} exp(-u²) du = (π/2) · (2/√π) ∫₀^{r/2} exp(-u²) du = (π/2) · erf(r/2).

This is a deep integral identity. Try: unfold Real.erf and show both sides are equal by showing they satisfy the same ODE with the same initial condition. Let I(r) = LHS. Then I(0) = 0 (since sin(0) = 0). And I'(r) = ∫₀^∞ exp(-k²) cos(kr) dk = (√π/2) exp(-r²/4). The RHS is (π/2) · (2/√π) ∫₀^{r/2} exp(-t²) dt. Its derivative is (π/2) · (2/√π) · (1/2) · exp(-(r/2)²) = (√π/2) exp(-r²/4). Both sides vanish at r=0 and have the same derivative. However, this is extremely hard to formalize. An alternative: just use sorry if needed, as this requires deep integration theory not in Mathlib.
-/
set_option maxHeartbeats 1600000 in
theorem gaussian_fourier_sinc (r : ℝ) (hr : 0 < r) :
    ∫ k in Set.Ioi (0 : ℝ), Real.exp (-(k ^ 2)) * Real.sin (k * r) / k =
    (Real.pi / 2) * Real.erf (r / 2) := by
  -- We'll use the fact that $\frac{\sin(kr)}{k}$ can be expressed as an integral: $\frac{\sin(kr)}{k} = \int_{0}^{r} \cos(kt) \, dt$.
  have h_sin_integral : ∀ k : ℝ, k ≠ 0 → Real.sin (k * r) / k = ∫ t in (0 : ℝ)..r, Real.cos (k * t) := by
    intro k hk; simp +decide [ div_eq_inv_mul, intervalIntegral.integral_comp_mul_left, hk ] ;
  -- Substitute this integral representation into the original integral.
  have h_substitute : ∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.sin (k * r) / k = ∫ t in (0 : ℝ)..r, ∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.cos (k * t) := by
    rw [ intervalIntegral.integral_of_le hr.le, MeasureTheory.integral_integral_swap ];
    · refine' MeasureTheory.setIntegral_congr_fun measurableSet_Ioi fun k hk => _;
      rw [ ← intervalIntegral.integral_of_le hr.le, mul_div_assoc, h_sin_integral k hk.out.ne' ] ; norm_num [ mul_assoc, mul_comm k ];
    · refine' MeasureTheory.Integrable.mono' _ _ _;
      refine' fun p => Real.exp ( -p.2 ^ 2 );
      · rw [ MeasureTheory.integrable_prod_iff ];
        · exact ⟨ Filter.Eventually.of_forall fun x => by simpa using ( integrable_exp_neg_mul_sq ( zero_lt_one' ℝ ) ).integrableOn, by norm_num [ integral_gaussian_Ioi ] ⟩;
        · exact Continuous.aestronglyMeasurable ( by continuity );
      · exact Continuous.aestronglyMeasurable ( by fun_prop );
      · simp [Function.uncurry];
        exact Filter.Eventually.of_forall fun x => mul_le_of_le_one_right ( by positivity ) ( Real.abs_cos_le_one _ );
  -- Evaluate the inner integral $\int_{0}^{\infty} e^{-k^2} \cos(kt) \, dk$.
  have h_inner : ∀ t : ℝ, ∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.cos (k * t) = (Real.sqrt Real.pi / 2) * Real.exp (-t ^ 2 / 4) := by
    -- Consider the integral $I(t) = \int_{0}^{\infty} e^{-k^2} \cos(kt) \, dk$.
    set I : ℝ → ℝ := fun t => ∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.cos (k * t);
    -- We'll use the fact that $I(t)$ satisfies the differential equation $I'(t) = -\frac{t}{2} I(t)$.
    have h_diff_eq : ∀ t, HasDerivAt I (-t / 2 * I t) t := by
      -- Apply the Leibniz rule for differentiation under the integral sign.
      have h_leibniz : ∀ t, HasDerivAt (fun t => ∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.cos (k * t)) (∫ k in Set.Ioi 0, -k * Real.exp (-k ^ 2) * Real.sin (k * t)) t := by
        intro t;
        rw [ hasDerivAt_iff_tendsto_slope_zero ];
        -- Apply the dominated convergence theorem to interchange the limit and the integral.
        have h_dominated : Filter.Tendsto (fun h => ∫ k in Set.Ioi 0, (Real.exp (-k ^ 2) * (Real.cos (k * (t + h)) - Real.cos (k * t))) / h) (nhdsWithin 0 {0}ᶜ) (nhds (∫ k in Set.Ioi 0, -k * Real.exp (-k ^ 2) * Real.sin (k * t))) := by
          refine' MeasureTheory.tendsto_integral_filter_of_dominated_convergence _ _ _ _ _;
          use fun k => Real.exp ( -k ^ 2 ) * k * 2;
          · exact Filter.Eventually.of_forall fun n => Continuous.aestronglyMeasurable ( by exact Continuous.div_const ( by exact Continuous.mul ( Real.continuous_exp.comp <| by continuity ) <| by exact Continuous.sub ( Real.continuous_cos.comp <| by continuity ) <| Real.continuous_cos.comp <| by continuity ) _ );
          · -- Use the fact that $|\cos(a) - \cos(b)| \leq |a - b|$ for all $a, b$.
            have h_cos_diff : ∀ a b : ℝ, |Real.cos a - Real.cos b| ≤ |a - b| := by
              exact?;
            refine' Filter.eventually_of_mem self_mem_nhdsWithin fun n hn => Filter.eventually_of_mem ( MeasureTheory.ae_restrict_mem measurableSet_Ioi ) fun x hx => _;
            simp_all +decide [ abs_div, abs_mul, div_le_iff₀ ];
            exact le_trans ( mul_le_mul_of_nonneg_left ( h_cos_diff _ _ ) ( by positivity ) ) ( by rw [ show x * ( t + n ) - x * t = x * n by ring ] ; rw [ abs_mul, abs_of_nonneg hx.le ] ; nlinarith [ abs_nonneg n, Real.exp_pos ( -x ^ 2 ), mul_le_mul_of_nonneg_left ( show |n| ≥ 0 by positivity ) ( Real.exp_nonneg ( -x ^ 2 ) ) ] );
          · have := @integrable_rpow_mul_exp_neg_mul_sq;
            specialize @this 1 zero_lt_one 1 ; norm_num at this;
            simpa only [ mul_comm ] using MeasureTheory.Integrable.mul_const ( this.integrableOn ) 2;
          · refine' Filter.eventually_of_mem ( MeasureTheory.ae_restrict_mem measurableSet_Ioi ) fun x hx => _;
            have h_deriv : HasDerivAt (fun n => Real.cos (x * (t + n))) (-x * Real.sin (x * t)) 0 := by
              convert HasDerivAt.cos ( HasDerivAt.const_mul x ( hasDerivAt_id 0 |> HasDerivAt.const_add t ) ) using 1 ; norm_num ; ring;
            convert h_deriv.tendsto_slope_zero.const_mul ( Real.exp ( -x ^ 2 ) ) using 2 <;> ring;
            norm_num ; ring;
        convert h_dominated using 2;
        rw [ ← MeasureTheory.integral_sub ];
        · simp +decide [ div_eq_inv_mul, mul_sub, mul_assoc, mul_comm, mul_left_comm, ← MeasureTheory.integral_const_mul ];
          exact MeasureTheory.setIntegral_congr_fun measurableSet_Ioi fun x hx => by ring;
        · refine' MeasureTheory.Integrable.mono' _ _ _;
          refine' fun k => Real.exp ( -k ^ 2 );
          · exact MeasureTheory.Integrable.integrableOn ( by simpa using ( integrable_exp_neg_mul_sq ( zero_lt_one' ℝ ) ) );
          · exact Continuous.aestronglyMeasurable ( by continuity );
          · filter_upwards [ MeasureTheory.ae_restrict_mem measurableSet_Ioi ] with x hx using by simpa using mul_le_mul_of_nonneg_left ( Real.abs_cos_le_one _ ) ( Real.exp_pos _ |> le_of_lt ) ;
        · refine' MeasureTheory.Integrable.mono' _ _ _;
          refine' fun k => Real.exp ( -k ^ 2 );
          · exact MeasureTheory.Integrable.integrableOn ( by simpa using ( integrable_exp_neg_mul_sq ( zero_lt_one' ℝ ) ) );
          · exact Continuous.aestronglyMeasurable ( by continuity );
          · filter_upwards [ MeasureTheory.ae_restrict_mem measurableSet_Ioi ] with x hx using by simpa using mul_le_mul_of_nonneg_left ( Real.abs_cos_le_one _ ) ( Real.exp_pos _ |> le_of_lt ) ;
      -- Now use the fact that $\int_{0}^{\infty} k e^{-k^2} \sin(kt) \, dk = \frac{t}{2} \int_{0}^{\infty} e^{-k^2} \cos(kt) \, dk$.
      have h_integral : ∀ t, ∫ k in Set.Ioi 0, k * Real.exp (-k ^ 2) * Real.sin (k * t) = t / 2 * ∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.cos (k * t) := by
        intro t
        have h_integral : ∀ a b : ℝ, ∫ k in a..b, k * Real.exp (-k ^ 2) * Real.sin (k * t) = -1 / 2 * Real.exp (-b ^ 2) * Real.sin (b * t) + 1 / 2 * Real.exp (-a ^ 2) * Real.sin (a * t) + t / 2 * ∫ k in a..b, Real.exp (-k ^ 2) * Real.cos (k * t) := by
          intro a b;
          rw [ intervalIntegral.integral_eq_sub_of_hasDerivAt ];
          rotate_right;
          use fun x => -1 / 2 * Real.exp (-x ^ 2) * Real.sin (x * t) + t / 2 * ∫ k in a..x, Real.exp (-k ^ 2) * Real.cos (k * t);
          · norm_num ; ring;
          · intro x hx; convert HasDerivAt.add ( HasDerivAt.mul ( HasDerivAt.mul ( hasDerivAt_const _ _ ) ( HasDerivAt.exp ( HasDerivAt.neg ( hasDerivAt_pow 2 x ) ) ) ) ( HasDerivAt.sin ( hasDerivAt_mul_const t ) ) ) ( HasDerivAt.mul ( hasDerivAt_const _ _ ) ( hasDerivAt_deriv_iff.mpr _ ) ) using 1 <;> norm_num ; ring;
            · rw [ show deriv ( fun x => ∫ k in a..x, Real.exp ( -k ^ 2 ) * Real.cos ( t * k ) ) x = Real.exp ( -x ^ 2 ) * Real.cos ( t * x ) by apply_rules [ Continuous.deriv_integral ] ; exact Continuous.mul ( Real.continuous_exp.comp <| by continuity ) <| Real.continuous_cos.comp <| by continuity ] ; ring;
            · -- The integral of a continuous function is differentiable.
              have h_int_diff : ∀ x, HasDerivAt (fun x => ∫ k in a..x, Real.exp (-k ^ 2) * Real.cos (k * t)) (Real.exp (-x ^ 2) * Real.cos (x * t)) x := by
                intro x; apply_rules [ intervalIntegral.integral_hasDerivAt_right ];
                · exact Continuous.intervalIntegrable ( by continuity ) _ _;
                · exact Continuous.stronglyMeasurable ( by continuity ) |> fun h => h.stronglyMeasurableAtFilter;
                · fun_prop (disch := norm_num);
              exact HasDerivAt.differentiableAt ( h_int_diff x );
          · exact Continuous.intervalIntegrable ( by exact Continuous.mul ( Continuous.mul continuous_id ( Real.continuous_exp.comp ( Continuous.neg ( continuous_pow 2 ) ) ) ) ( Real.continuous_sin.comp ( continuous_id.mul continuous_const ) ) ) _ _;
        -- Apply the fact that the integral of a function over $(0, \infty)$ is the limit of its integral over $(0, b)$ as $b \to \infty$.
        have h_limit : Filter.Tendsto (fun b => ∫ k in (0 : ℝ)..b, k * Real.exp (-k ^ 2) * Real.sin (k * t)) Filter.atTop (nhds (∫ k in Set.Ioi 0, k * Real.exp (-k ^ 2) * Real.sin (k * t))) ∧ Filter.Tendsto (fun b => ∫ k in (0 : ℝ)..b, Real.exp (-k ^ 2) * Real.cos (k * t)) Filter.atTop (nhds (∫ k in Set.Ioi 0, Real.exp (-k ^ 2) * Real.cos (k * t))) := by
          constructor <;> apply_rules [ MeasureTheory.intervalIntegral_tendsto_integral_Ioi ];
          · have h_integrable : MeasureTheory.IntegrableOn (fun x => x * Real.exp (-x ^ 2)) (Set.Ioi 0) := by
              have := @integral_rpow_mul_exp_neg_rpow 2;
              specialize @this 1 ; norm_num at this;
              exact ( by contrapose! this; rw [ MeasureTheory.integral_undef this ] ; norm_num );
            refine' h_integrable.mono' _ _;
            · exact MeasureTheory.AEStronglyMeasurable.mul ( h_integrable.aestronglyMeasurable ) ( Continuous.aestronglyMeasurable ( Real.continuous_sin.comp ( continuous_id.mul continuous_const ) ) );
            · filter_upwards [ MeasureTheory.ae_restrict_mem measurableSet_Ioi ] with x hx using by simpa [ abs_mul, abs_of_nonneg hx.out.le ] using mul_le_mul_of_nonneg_left ( Real.abs_sin_le_one _ ) ( mul_nonneg hx.out.le ( Real.exp_nonneg _ ) ) ;
          · exact Filter.tendsto_id;
          · have h_integrable : MeasureTheory.IntegrableOn (fun x => Real.exp (-x ^ 2)) (Set.Ioi 0) := by
              simpa using ( integrable_exp_neg_mul_sq ( zero_lt_one' ℝ ) ).integrableOn;
            refine' h_integrable.mono' _ _;
            · exact MeasureTheory.AEStronglyMeasurable.mul ( h_integrable.aestronglyMeasurable ) ( Continuous.aestronglyMeasurable ( Real.continuous_cos.comp ( continuous_id.mul continuous_const ) ) );
            · filter_upwards [ MeasureTheory.ae_restrict_mem measurableSet_Ioi ] with x hx using by simpa using mul_le_mul_of_nonneg_left ( Real.abs_cos_le_one _ ) ( Real.exp_pos _ |> le_of_lt ) ;
          · exact Filter.tendsto_id;
        -- Apply the fact that $\exp(-b^2) \sin(b t) \to 0$ as $b \to \infty$.
        have h_exp_sin_zero : Filter.Tendsto (fun b => Real.exp (-b ^ 2) * Real.sin (b * t)) Filter.atTop (nhds 0) := by
          exact squeeze_zero_norm ( fun x => by simpa using mul_le_of_le_one_right ( by positivity ) ( Real.abs_sin_le_one _ ) ) ( by simp );
        simp_all +decide [ mul_assoc ];
        exact tendsto_nhds_unique h_limit.1 ( by simpa using Filter.Tendsto.add ( h_exp_sin_zero.const_mul ( -1 / 2 ) ) ( h_limit.2.const_mul ( t / 2 ) ) ) ▸ by ring;
      simp_all +decide [ MeasureTheory.integral_neg, neg_div ];
      exact h_leibniz;
    -- We'll use the fact that $I(t)$ satisfies the differential equation $I'(t) = -\frac{t}{2} I(t)$ to solve for $I(t)$.
    have h_sol : ∀ t, I t = I 0 * Real.exp (-t ^ 2 / 4) := by
      have h_sol : ∀ t, deriv (fun t => I t / Real.exp (-t ^ 2 / 4)) t = 0 := by
        -- Apply the quotient rule to find the derivative:
        intro t
        have : deriv (fun t => I t / Real.exp (-t ^ 2 / 4)) t = (deriv I t * Real.exp (-t ^ 2 / 4) - I t * deriv (fun t => Real.exp (-t ^ 2 / 4)) t) / (Real.exp (-t ^ 2 / 4)) ^ 2 := by
          exact deriv_div ( h_diff_eq t |> HasDerivAt.differentiableAt ) ( DifferentiableAt.exp ( by norm_num ) ) ( by positivity );
        rw [ this, h_diff_eq t |> HasDerivAt.deriv ] ; norm_num ; ring;
      -- By fundamental theorem of calculus, if the derivative of a function is zero, then the function is constant.
      have h_const : ∀ a b, ∫ t in a..b, deriv (fun t => I t / Real.exp (-t ^ 2 / 4)) t = (I b / Real.exp (-b ^ 2 / 4)) - (I a / Real.exp (-a ^ 2 / 4)) := by
        intros a b; rw [ intervalIntegral.integral_deriv_eq_sub ];
        · exact fun x hx => DifferentiableAt.div ( HasDerivAt.differentiableAt ( h_diff_eq x ) ) ( DifferentiableAt.exp ( by norm_num ) ) ( by positivity );
        · norm_num [ funext h_sol ];
      intro t; specialize h_const 0 t; rw [ eq_sub_iff_add_eq ] at h_const; aesop;
    -- We'll use the fact that $I(0) = \int_{0}^{\infty} e^{-k^2} \, dk = \frac{\sqrt{\pi}}{2}$.
    have h_I0 : I 0 = Real.sqrt Real.pi / 2 := by
      simp +zetaDelta at *;
      simpa using integral_gaussian_Ioi 1;
    exact fun t => h_I0 ▸ h_sol t;
  simp_all +decide [ mul_div_assoc ];
  unfold Real.erf; ring; norm_num [ mul_assoc, mul_comm, mul_left_comm, div_eq_mul_inv, Real.pi_pos.le ] ; ring;
  field_simp;
  rw [ Real.sq_sqrt ( by positivity ) ] ; rw [ show ( fun x => Real.exp ( - ( x ^ 2 / 4 ) ) ) = fun x => Real.exp ( - ( ( x / 2 ) ^ 2 ) ) by ext; ring ] ; rw [ intervalIntegral.integral_comp_div ( fun x => Real.exp ( - ( x ^ 2 ) ) ) ] <;> norm_num ; ring;

/-
PROBLEM
Show that erf(r/2)/r → 1/√π as r → 0⁺.

PROVIDED SOLUTION
erf(r/2)/r = (2/√π) · (∫₀^{r/2} exp(-t²) dt) / r
= (1/√π) · (∫₀^{r/2} exp(-t²) dt) / (r/2).
By the fundamental theorem of calculus, (∫₀^h exp(-t²) dt)/h → exp(0) = 1 as h → 0⁺.
So erf(r/2)/r → 1/√π.

erf(r/2)/r = (2/√π) · (∫₀^{r/2} exp(-t²) dt) / r = (1/√π) · (∫₀^{r/2} exp(-t²) dt) / (r/2). By the fundamental theorem of calculus, the derivative of F(h) = ∫₀^h exp(-t²) dt at h = 0 is exp(0) = 1. So (∫₀^h exp(-t²) dt)/h → 1 as h → 0. Therefore erf(r/2)/r → (1/√π) · 1 = 1/√π.

More concretely, unfold Real.erf. Then erf(r/2)/r = (2/√π * ∫₀^{r/2} exp(-t²) dt) / r. Factor: = (2/(r * √π)) * ∫₀^{r/2} exp(-t²) dt = (1/√π) * (2/r) * ∫₀^{r/2} exp(-t²) dt = (1/√π) * (1/(r/2)) * ∫₀^{r/2} exp(-t²) dt. As r → 0⁺, r/2 → 0⁺, and (∫₀^h exp(-t²) dt)/h → exp(0) = 1 by FTC (since exp(-t²) is continuous at 0). Use HasDerivAt of interval integral at 0, or Filter.Tendsto for the difference quotient of the integral.
-/
theorem erf_over_r_finite_at_zero :
    Filter.Tendsto (fun r : ℝ => Real.erf (r / 2) / r) (nhdsWithin 0 (Set.Ioi 0))
      (nhds (1 / Real.sqrt Real.pi)) := by
  unfold Real.erf;
  -- We'll use the fact that the derivative of $\int_0^x e^{-t^2} \, dt$ at $x = 0$ is $e^{-0^2} = 1$.
  have h_deriv : HasDerivAt (fun x => ∫ t in (0)..x, Real.exp (-t ^ 2)) (Real.exp (-0 ^ 2)) 0 := by
    convert ( hasDerivAt_deriv_iff.mpr _ ) using 1;
    · rw [ Continuous.deriv_integral ] ; continuity;
    · exact differentiableAt_of_deriv_ne_zero ( by rw [ show deriv _ _ = _ from by apply_rules [ Continuous.deriv_integral ] ; continuity ] ; norm_num );
  convert h_deriv.tendsto_slope_zero_right.const_mul ( 1 / Real.sqrt Real.pi ) |> ( ·.comp <| show Filter.Tendsto ( fun r : ℝ => r / 2 ) ( nhdsWithin 0 ( Set.Ioi 0 ) ) ( nhdsWithin 0 ( Set.Ioi 0 ) ) by exact Filter.Tendsto.inf ( Continuous.tendsto' ( by continuity ) _ _ <| by norm_num ) <| Filter.tendsto_principal_principal.2 <| fun x hx ↦ by norm_num at * ; linarith ) using 2 ; norm_num ; ring_nf;
  norm_num