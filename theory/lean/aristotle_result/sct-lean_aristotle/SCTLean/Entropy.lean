import Mathlib

/-
PROVIDED SOLUTION
Use Jensen's inequality for the concave function log. Since ∑ p_i = 1 and p_i > 0, by Jensen: ∑ p_i * log(p_i) ≥ log(∑ p_i * p_i)... Actually the standard proof uses log(n * p_i) ≤ n*p_i - 1 or the inequality ln(x) ≤ x - 1. Specifically: -∑ p_i log(p_i) = ∑ p_i log(1/p_i) ≤ log(∑ p_i * (1/p_i)) = log(n) by Jensen (log is concave) applied with weights p_i to values 1/p_i. More concretely, use the inequality -x*log(x) ≤ -x*log(1/n) = x*log(n) summed over i gives -∑ p_i log(p_i) ≤ log(n) * ∑ p_i = log(n). Wait, that's not right. The correct approach: use ln(x) ≤ x - 1 to get ∑ p_i * log(1/(n*p_i)) ≤ ∑ p_i * (1/(n*p_i) - 1) = ∑(1/n) - ∑ p_i = 1 - 1 = 0. So ∑ p_i * (log(1/p_i) - log(n)) ≤ 0, hence ∑ p_i * log(1/p_i) ≤ log(n). This gives the result.
-/
theorem shannon_entropy_max {n : ℕ} (hn : 0 < n)
    (p : Fin n → ℝ) (hp_pos : ∀ i, 0 < p i) (hp_sum : ∑ i, p i = 1) :
    -∑ i, p i * Real.log (p i) ≤ Real.log n := by
  -- Applying Jensen's inequality for the concave function $\log$ with weights $p_i$ and values $1/p_i$.
  have h_jensen : (∑ i, p i * Real.log (1 / p i)) ≤ Real.log (∑ i, p i * (1 / p i)) := by
    have h_jensen : ∀ (x : Fin n → ℝ), (∀ i, 0 < x i) → (∑ i, p i = 1) → (∑ i, p i * Real.log (x i)) ≤ Real.log (∑ i, p i * x i) := by
      intros x hx_pos hx_sum
      have h_jensen : (∑ i, p i * Real.log (x i)) ≤ Real.log (∑ i, p i * x i) := by
        have h_concave : ConcaveOn ℝ (Set.Ioi 0) Real.log := by
          exact ( StrictConcaveOn.concaveOn <| strictConcaveOn_log_Ioi )
        apply_rules [ h_concave.le_map_sum ];
        · exact fun i _ => le_of_lt ( hp_pos i );
        · aesop;
      exact h_jensen;
    exact h_jensen _ ( fun i => one_div_pos.mpr ( hp_pos i ) ) hp_sum;
  simp_all +decide [ ne_of_gt, mul_div_cancel₀ ]

/-
PROVIDED SOLUTION
-log(exp(-β*x)) = β*x by Real.log_exp. So the function simplifies to the linear function β*x, which is convex on any set. Use simp [Real.log_exp] to simplify, then show that a linear function is convex (convexOn_const or LinearMap convexity).
-/
theorem exp_neg_log_concave (β : ℝ) (hβ : 0 < β) :
    ConvexOn ℝ Set.univ (fun x : ℝ => -Real.log (Real.exp (-β * x))) := by
  exact ⟨ convex_univ, fun x _ y _ a b ha hb hab => by simpa using by nlinarith ⟩

/-
PROVIDED SOLUTION
Sum of positive terms over a nonempty Finset is positive. Each term exp(-β * E i) is positive by exp_pos. The index set Fin n is nonempty since 0 < n (use Fin.pos_iff_nonempty or hn). Use Finset.sum_pos with the fact that each term is positive and univ is nonempty.
-/
theorem partition_function_pos {n : ℕ} (hn : 0 < n) (β : ℝ)
    (E : Fin n → ℝ) :
    0 < ∑ i, Real.exp (-β * E i) := by
  exact Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) ⟨ ⟨ 0, hn ⟩, Finset.mem_univ _ ⟩