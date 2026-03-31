import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# CJ Bridge: Angular × Volume Factorization (ℝ with π)

The angular×volume factor in the CJ formula:
  π²/45 = (8π/15) × (π/24)

Where:
- 8π/15 comes from ∫_{S²} (E_{ij}n^in^j)² dΩ / E² = 8π/15
  (angular average of traceless symmetric tensor squared over unit sphere)
- π/24 = π/d! comes from the diamond volume V₄ = T⁴/(4!) = T⁴/24
  (the volume of a causal diamond in 4D, up to π factors)

The π²/90 = (1/2)·π²/45 is the Stefan-Boltzmann factor,
connecting CJ's geometric coefficient to thermal physics
via the diamond temperature T_d = 1/(πT).
-/

/-- The main factorization: π²/45 = (8π/15)·(π/24).
    Reduces to the rational identity 1/45 = 8/(15×24) = 8/360. -/
theorem angular_volume_pi :
    Real.pi ^ 2 / 45 = (8 * Real.pi / 15) * (Real.pi / 24) := by
  ring

/-- Stefan-Boltzmann connection: π²/90 = (1/2)·(π²/45). -/
theorem stefan_boltzmann_connection :
    Real.pi ^ 2 / 90 = (1 / 2 : ℝ) * (Real.pi ^ 2 / 45) := by
  ring

/-- Rational part: 8/15 = 8/(d²-1) at d=4. -/
theorem angular_factor_d4 : (8 : ℚ) / 15 = 8 / ((4:ℚ)^2 - 1) := by norm_num

/-- Volume part: 1/24 = 1/d! at d=4. -/
theorem volume_factor_d4 : (1 : ℚ) / 24 = 1 / (Nat.factorial 4 : ℚ) := by
  simp [Nat.factorial]

/-- Combined: (8/15)·(1/24) = 1/45. -/
theorem angular_times_volume : (8 : ℚ) / 15 * (1 / 24) = 1 / 45 := by norm_num

/-- The full CJ formula coefficient (with π²):
    8π²/(3·9!·45) = π² × 8/(3·362880·45) = π²/6123600.

    In terms of the decomposition:
    [8/(3·9!)] × [π²/45]
    = BD² coefficient × angular·volume factor
    = [BD_norm²/(2d+1)!] × [(8π/(d²-1))·(π/d!)] -/
theorem cj_full_coefficient_decomposition :
    (8 : ℝ) / (3 * 362880) * (Real.pi ^ 2 / 45) =
    8 * Real.pi ^ 2 / (3 * 362880 * 45) := by ring

-- The angular integral identity for S^{d-2} in general d.
-- The general formula for the 4th moment of unit vectors on S^{d-2} gives
-- coefficient 1/(d²-1) when contracted with traceless symmetric E.

/-- d²-1 values for d=2..5 -/
theorem dsq_minus_1_d2 : (2:ℕ)^2 - 1 = 3 := by norm_num
theorem dsq_minus_1_d3 : (3:ℕ)^2 - 1 = 8 := by norm_num
theorem dsq_minus_1_d4 : (4:ℕ)^2 - 1 = 15 := by norm_num
theorem dsq_minus_1_d5 : (5:ℕ)^2 - 1 = 24 := by norm_num

/-- The (2d)² factor (null directions squared) -/
theorem null_directions_general (d : ℕ) : (2 * d)^2 = 4 * d^2 := by ring

/-- For d=4: (2d)²/[(d-1)·(d²-1)·d!] = 64/(3·15·24) = 64/1080 = 8/135 -/
theorem null_over_all_d4 : (64 : ℚ) / (3 * 15 * 24) = 8 / 135 := by norm_num

/-- Cross-check: 8/135 × 1/(2d+1)! = 8/(135·9!) = 8/48988800 = 1/6123600 -/
theorem full_cj_rational :
    (8 : ℚ) / 135 * (1 / 362880) = 1 / 6123600 := by norm_num
