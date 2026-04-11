import Mathlib.Tactic
import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Data.Rat.Defs

open scoped BigOperators

/-!
# CJ Bridge Estimator Skeleton

This file formalizes the exact finite algebraic skeleton of the Paper 7 CJ estimator.

The goal is not to formalize the large-`N` probabilistic asymptotics yet.
Instead, we fix the exact finite objects that future asymptotic arguments talk about:

- fiber sets induced by a bin map,
- finite means on fibers,
- finite covariances on fibers,
- the Paper-7-style weighted sum over nonempty bins.
-/

namespace SCT.CJBridge

section FiniteStats

variable {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]

/-- The fiber of a finite bin map over a bin label. -/
def fiber (bin : α → β) (b : β) : Finset α :=
  Finset.univ.filter fun i => bin i = b

/-- The finite mean of a rational-valued function on a finite set.
Empty sets are assigned mean `0`. -/
def meanOn (S : Finset α) (f : α → ℚ) : ℚ :=
  if _h : S.card = 0 then 0 else (Finset.sum S f) / (S.card : ℚ)

/-- The finite covariance of two rational-valued functions on a finite set. -/
def covOn (S : Finset α) (U V : α → ℚ) : ℚ :=
  meanOn S (fun i => (U i - meanOn S U) * (V i - meanOn S V))

/-- The rational bin weight induced by a finite bin map. -/
def binWeight (bin : α → β) (b : β) : ℚ :=
  ((fiber bin b).card : ℚ) / (Fintype.card α : ℚ)

/-- The fiber of a finite bin map over a bin label, restricted to a finite support set. -/
def fiberIn (S : Finset α) (bin : α → β) (b : β) : Finset α :=
  S.filter fun i => bin i = b

/-- The rational bin weight induced by a finite bin map on a finite support set.
Empty support sets are assigned weight `0`. -/
def binWeightOn (S : Finset α) (bin : α → β) (b : β) : ℚ :=
  if _h : S.card = 0 then 0 else ((fiberIn S bin b).card : ℚ) / (S.card : ℚ)

/-- A rational-valued observable factors through the finite bin map if it is constant on each fiber. -/
def FactorsThroughBin (bin : α → β) (f : α → ℚ) : Prop :=
  ∃ g : β → ℚ, ∀ i, f i = g (bin i)

/-- The finite support set cut out by a decidable predicate. -/
def supportOf (P : α → Prop) [DecidablePred P] : Finset α :=
  Finset.univ.filter P

/-- The size of a finite predicate-defined support. -/
def Nsupport (P : α → Prop) [DecidablePred P] : Nat :=
  (supportOf P).card

/-- The finite bulk support defined by the slack inequality `slack >= ζ T`. -/
def bulkSupportSlack (slack : α → ℚ) (ζ T : ℚ) : Finset α :=
  supportOf (fun i => ζ * T ≤ slack i)

/-- A generic rational raw-rank skeleton built from a finite past-count and total size.
If `Ntot = 0`, return `0`. -/
def rawRankFromPastCount (pastCount : α → Nat) (Ntot : Nat) (i : α) : ℚ :=
  if _h : Ntot = 0 then 0 else (pastCount i : ℚ) / (Ntot : ℚ)

/-- Generic affine renormalization used by the S2 split-coordinate candidate.
If `rmax = rmin`, return `0`. -/
def affineRenorm (r rmin rmax : ℚ) : ℚ :=
  if h : rmax = rmin then 0 else (r - rmin) / (rmax - rmin)

/-- Minimum raw-rank value on a finite support set.
Empty supports are assigned `0`. -/
def supportMinVal (S : Finset α) (r : α → ℚ) : ℚ :=
  if h : S.card = 0 then 0 else
    have hS : S.Nonempty := Finset.card_ne_zero.mp h
    have hT : (S.image r).Nonempty := by
      rcases hS with ⟨i, hi⟩
      exact ⟨r i, Finset.mem_image.mpr ⟨i, hi, rfl⟩⟩
    (S.image r).min' hT

/-- Maximum raw-rank value on a finite support set.
Empty supports are assigned `0`. -/
def supportMaxVal (S : Finset α) (r : α → ℚ) : ℚ :=
  if h : S.card = 0 then 0 else
    have hS : S.Nonempty := Finset.card_ne_zero.mp h
    have hT : (S.image r).Nonempty := by
      rcases hS with ⟨i, hi⟩
      exact ⟨r i, Finset.mem_image.mpr ⟨i, hi, rfl⟩⟩
    (S.image r).max' hT

/-- The current exact finite S2 split-coordinate candidate attached to a support set.
Outside the support, the coordinate is defined to be `0`. -/
def s2CoordOnSupport (S : Finset α) (r : α → ℚ) (i : α) : ℚ :=
  if hi : i ∈ S then
    affineRenorm (r i) (supportMinVal S r) (supportMaxVal S r)
  else
    0

/-- The bulk-cut specialization of `s2CoordOnSupport`. -/
def s2CoordBulk (slack : α → ℚ) (ζ T : ℚ) (r : α → ℚ) (i : α) : ℚ :=
  s2CoordOnSupport (bulkSupportSlack slack ζ T) r i

/-- The finite pushforward support induced by a coordinate map on a support set. -/
def pushforwardSupportOn (S : Finset α) (coord : α → β) : Finset β :=
  S.image coord

/-- The finite pushforward weight induced by a coordinate map on a support set. -/
def pushforwardWeightOn (S : Finset α) (coord : α → β) (y : β) : ℚ :=
  binWeightOn S coord y

/-- The bulk-cut specialization of the finite pushforward support. -/
def pushforwardSupportBulk (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) : Finset β :=
  pushforwardSupportOn (bulkSupportSlack slack ζ T) coord

/-- The bulk-cut specialization of the finite pushforward weight. -/
def pushforwardWeightBulk (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (y : β) : ℚ :=
  pushforwardWeightOn (bulkSupportSlack slack ζ T) coord y

/-- A finite density attached to the baseline pushforward geometry reweights the pushforward
weight on a support set. -/
def densityPushforwardWeightOn (S : Finset α) (coord : α → β) (ρ : β → ℚ) (y : β) : ℚ :=
  ρ y * pushforwardWeightOn S coord y

/-- The bulk-cut specialization of the density-weighted pushforward weight. -/
def densityPushforwardWeightBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (y : β) : ℚ :=
  densityPushforwardWeightOn (bulkSupportSlack slack ζ T) coord ρ y

/-- The finite weighted-pushforward sum of a test function over a support set. -/
def densityPushforwardSumOn
    (S : Finset α) (coord : α → β) (ρ φ : β → ℚ) : ℚ :=
  Finset.sum (pushforwardSupportOn S coord)
    (fun y => φ y * densityPushforwardWeightOn S coord ρ y)

/-- The bulk-cut specialization of the finite weighted-pushforward sum. -/
def densityPushforwardSumBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ) : ℚ :=
  densityPushforwardSumOn (bulkSupportSlack slack ζ T) coord ρ φ

/-- The total weighted pushforward mass on a support set. -/
def densityPushforwardMassOn
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) : ℚ :=
  densityPushforwardSumOn S coord ρ (fun _ => (1 : ℚ))

/-- The bulk-cut specialization of the total weighted pushforward mass. -/
def densityPushforwardMassBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) : ℚ :=
  densityPushforwardMassOn (bulkSupportSlack slack ζ T) coord ρ

/-- The normalized density-weighted pushforward average of a test function on a support set.
If the total weighted pushforward mass vanishes, the average is defined to be `0`. -/
def densityPushforwardAvgOn
    (S : Finset α) (coord : α → β) (ρ φ : β → ℚ) : ℚ :=
  if hmass : densityPushforwardMassOn S coord ρ = 0 then 0
  else densityPushforwardSumOn S coord ρ φ / densityPushforwardMassOn S coord ρ

/-- The bulk-cut specialization of the normalized density-weighted pushforward average. -/
def densityPushforwardAvgBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ) : ℚ :=
  densityPushforwardAvgOn (bulkSupportSlack slack ζ T) coord ρ φ

/-- The finite baseline pushforward sum of a test function on a support set. -/
def pushforwardSumOn
    (S : Finset α) (coord : α → β) (φ : β → ℚ) : ℚ :=
  densityPushforwardSumOn S coord (fun _ => (1 : ℚ)) φ

/-- The bulk-cut specialization of the finite baseline pushforward sum. -/
def pushforwardSumBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) : ℚ :=
  pushforwardSumOn (bulkSupportSlack slack ζ T) coord φ

/-- The total baseline pushforward mass on a support set. -/
def pushforwardMassOn (S : Finset α) (coord : α → β) : ℚ :=
  densityPushforwardMassOn S coord (fun _ => (1 : ℚ))

/-- The bulk-cut specialization of the total baseline pushforward mass. -/
def pushforwardMassBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) : ℚ :=
  pushforwardMassOn (bulkSupportSlack slack ζ T) coord

/-- The normalized baseline pushforward average of a test function on a support set. -/
def pushforwardAvgOn
    (S : Finset α) (coord : α → β) (φ : β → ℚ) : ℚ :=
  densityPushforwardAvgOn S coord (fun _ => (1 : ℚ)) φ

/-- The bulk-cut specialization of the normalized baseline pushforward average. -/
def pushforwardAvgBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) : ℚ :=
  pushforwardAvgOn (bulkSupportSlack slack ζ T) coord φ

/-- The finite Paper 7 estimator skeleton: weighted sum of within-bin covariances
over the bins that are actually hit by the finite bin map. -/
def paperEstimator (bin : α → β) (U V : α → ℚ) : ℚ :=
  Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U V)

/-- The weighted sum of within-bin `mean(UV)` terms for the full estimator. -/
def productTerm (bin : α → β) (U V : α → ℚ) : ℚ :=
  Finset.sum (Finset.univ.image bin)
    (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i))

/-- The weighted sum of within-bin `mean(U)mean(V)` terms for the full estimator. -/
def meanProductTerm (bin : α → β) (U V : α → ℚ) : ℚ :=
  Finset.sum (Finset.univ.image bin)
    (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V))

/-- The restricted-domain Paper 7 estimator skeleton: weighted sum of within-bin covariances
over the bins actually hit inside a finite support set. -/
def paperEstimatorOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : ℚ :=
  Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U V)

/-- The weighted sum of within-bin `mean(UV)` terms for the restricted estimator. -/
def productTermOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : ℚ :=
  Finset.sum (S.image bin)
    (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i))

/-- The weighted sum of within-bin `mean(U)mean(V)` terms for the restricted estimator. -/
def meanProductTermOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : ℚ :=
  Finset.sum (S.image bin)
    (fun b => binWeightOn S bin b * (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V))

/-- The restricted mean-product channel on the image side. -/
def meanProdChannelOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : β → ℚ :=
  fun b => meanOn (fiberIn S bin b) (fun i => U i * V i)

/-- The restricted mean-means channel on the image side. -/
def meanMeansChannelOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : β → ℚ :=
  fun b => meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V

/-- The exact restricted centered covariance channel on the image side. -/
def covChannelOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : β → ℚ :=
  fun b => covOn (fiberIn S bin b) U V

/-- The bulk-cut specialization of the restricted product term. -/
def productTermBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : ℚ :=
  productTermOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the restricted mean-product term. -/
def meanProductTermBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : ℚ :=
  meanProductTermOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the restricted Paper 7 estimator. -/
def paperEstimatorBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : ℚ :=
  paperEstimatorOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the mean-product channel. -/
def meanProdChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : β → ℚ :=
  meanProdChannelOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the mean-means channel. -/
def meanMeansChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : β → ℚ :=
  meanMeansChannelOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the centered covariance channel. -/
def covChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : β → ℚ :=
  covChannelOn (bulkSupportSlack slack ζ T) bin U V

/-- The exact pairwise finite-difference object for the product term. -/
def deltaProductTermOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : ℚ :=
  productTermOn S bin U V - productTerm bin U V

/-- The exact pairwise finite-difference object for the mean-product term. -/
def deltaMeanProductTermOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : ℚ :=
  meanProductTermOn S bin U V - meanProductTerm bin U V

/-- The exact centered finite-difference object for the Paper estimator. -/
def deltaPaperEstimatorOn (S : Finset α) (bin : α → β) (U V : α → ℚ) : ℚ :=
  paperEstimatorOn S bin U V - paperEstimator bin U V

/-- The bulk-cut specialization of the exact pairwise finite-difference object for the product term. -/
def deltaProductTermBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : ℚ :=
  deltaProductTermOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the exact pairwise finite-difference object for the mean-product term. -/
def deltaMeanProductTermBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : ℚ :=
  deltaMeanProductTermOn (bulkSupportSlack slack ζ T) bin U V

/-- The bulk-cut specialization of the exact centered finite-difference object for the Paper estimator. -/
def deltaPaperEstimatorBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) : ℚ :=
  deltaPaperEstimatorOn (bulkSupportSlack slack ζ T) bin U V

@[simp] theorem binWeightOn_empty (bin : α → β) (b : β) :
    binWeightOn (∅ : Finset α) bin b = 0 := by
  simp [binWeightOn]

@[simp] theorem paperEstimatorOn_empty (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn (∅ : Finset α) bin U V = 0 := by
  simp [paperEstimatorOn]

theorem supportOf_false (P : α → Prop) [DecidablePred P]
    (hP : ∀ i, ¬ P i) :
    supportOf P = (∅ : Finset α) := by
  ext i
  simp [supportOf, hP i]

theorem supportOf_true (P : α → Prop) [DecidablePred P]
    (hP : ∀ i, P i) :
    supportOf P = Finset.univ := by
  ext i
  simp [supportOf, hP i]

@[simp] theorem rawRankFromPastCount_zero (pastCount : α → Nat) :
    rawRankFromPastCount pastCount 0 = fun _ => (0 : ℚ) := by
  funext i
  simp [rawRankFromPastCount]

@[simp] theorem affineRenorm_degenerate (r rmin : ℚ) :
    affineRenorm r rmin rmin = 0 := by
  simp [affineRenorm]

theorem affineRenorm_at_min (rmin rmax : ℚ) (h : rmax ≠ rmin) :
    affineRenorm rmin rmin rmax = 0 := by
  simp [affineRenorm, h]

theorem affineRenorm_at_max (rmin rmax : ℚ) (h : rmax ≠ rmin) :
    affineRenorm rmax rmin rmax = 1 := by
  have h' : rmax - rmin ≠ 0 := sub_ne_zero.mpr h
  simp [affineRenorm, h]
  have hdiv : (rmax - rmin) / (rmax - rmin) = (1 : ℚ) := by
    field_simp [h']
  simpa using hdiv

@[simp] theorem supportMinVal_empty (r : α → ℚ) :
    supportMinVal (∅ : Finset α) r = 0 := by
  simp [supportMinVal]

@[simp] theorem supportMaxVal_empty (r : α → ℚ) :
    supportMaxVal (∅ : Finset α) r = 0 := by
  simp [supportMaxVal]

@[simp] theorem supportMinVal_singleton (a : α) (r : α → ℚ) :
    supportMinVal ({a} : Finset α) r = r a := by
  simp [supportMinVal]

@[simp] theorem supportMaxVal_singleton (a : α) (r : α → ℚ) :
    supportMaxVal ({a} : Finset α) r = r a := by
  simp [supportMaxVal]

@[simp] theorem s2CoordOnSupport_not_mem (S : Finset α) (r : α → ℚ) (i : α)
    (hi : i ∉ S) :
    s2CoordOnSupport S r i = 0 := by
  simp [s2CoordOnSupport, hi]

theorem s2CoordOnSupport_singleton_self (a : α) (r : α → ℚ) :
    s2CoordOnSupport ({a} : Finset α) r a = 0 := by
  simp [s2CoordOnSupport, affineRenorm]

theorem s2CoordOnSupport_support_true (P : α → Prop) [DecidablePred P]
    (r : α → ℚ) (hP : ∀ i, P i) :
    s2CoordOnSupport (supportOf P) r = s2CoordOnSupport Finset.univ r := by
  rw [supportOf_true P hP]

theorem s2CoordOnSupport_support_false (P : α → Prop) [DecidablePred P]
    (r : α → ℚ) (hP : ∀ i, ¬ P i) :
    s2CoordOnSupport (supportOf P) r = fun _ => (0 : ℚ) := by
  funext i
  rw [supportOf_false P hP]
  simp [s2CoordOnSupport]

theorem s2CoordBulk_true (slack : α → ℚ) (ζ T : ℚ) (r : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    s2CoordBulk slack ζ T r = s2CoordOnSupport Finset.univ r := by
  unfold s2CoordBulk bulkSupportSlack
  rw [supportOf_true (fun i => ζ * T ≤ slack i) hbulk]

theorem s2CoordBulk_false (slack : α → ℚ) (ζ T : ℚ) (r : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    s2CoordBulk slack ζ T r = fun _ => (0 : ℚ) := by
  unfold s2CoordBulk bulkSupportSlack
  exact s2CoordOnSupport_support_false (fun i => ζ * T ≤ slack i) r hbulk

@[simp] theorem pushforwardSupportOn_empty (coord : α → β) :
    pushforwardSupportOn (∅ : Finset α) coord = (∅ : Finset β) := by
  simp [pushforwardSupportOn]

@[simp] theorem pushforwardWeightOn_empty (coord : α → β) (y : β) :
    pushforwardWeightOn (∅ : Finset α) coord y = 0 := by
  simp [pushforwardWeightOn]

theorem pushforwardSupportOn_univ (coord : α → β) :
    pushforwardSupportOn Finset.univ coord = Finset.univ.image coord := by
  rfl

theorem pushforwardWeightOn_univ (coord : α → β) (y : β) :
    pushforwardWeightOn Finset.univ coord y = binWeight coord y := by
  unfold pushforwardWeightOn binWeightOn binWeight
  by_cases hα : Fintype.card α = 0
  · simp [hα, fiberIn, fiber]
  · simp [hα, fiberIn, fiber]

theorem pushforwardSupportOn_support_true (P : α → Prop) [DecidablePred P]
    (coord : α → β) (hP : ∀ i, P i) :
    pushforwardSupportOn (supportOf P) coord = Finset.univ.image coord := by
  rw [supportOf_true P hP]
  rfl

theorem pushforwardSupportOn_support_false (P : α → Prop) [DecidablePred P]
    (coord : α → β) (hP : ∀ i, ¬ P i) :
    pushforwardSupportOn (supportOf P) coord = (∅ : Finset β) := by
  rw [supportOf_false P hP]
  simp [pushforwardSupportOn]

theorem pushforwardWeightOn_support_true (P : α → Prop) [DecidablePred P]
    (coord : α → β) (y : β) (hP : ∀ i, P i) :
    pushforwardWeightOn (supportOf P) coord y = binWeight coord y := by
  rw [supportOf_true P hP]
  exact pushforwardWeightOn_univ coord y

theorem pushforwardWeightOn_support_false (P : α → Prop) [DecidablePred P]
    (coord : α → β) (y : β) (hP : ∀ i, ¬ P i) :
    pushforwardWeightOn (supportOf P) coord y = 0 := by
  rw [supportOf_false P hP]
  simp [pushforwardWeightOn]

theorem pushforwardSupportBulk_true (slack : α → ℚ) (ζ T : ℚ) (coord : α → β)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    pushforwardSupportBulk slack ζ T coord = Finset.univ.image coord := by
  unfold pushforwardSupportBulk bulkSupportSlack
  exact pushforwardSupportOn_support_true (fun i => ζ * T ≤ slack i) coord hbulk

theorem pushforwardSupportBulk_false (slack : α → ℚ) (ζ T : ℚ) (coord : α → β)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    pushforwardSupportBulk slack ζ T coord = (∅ : Finset β) := by
  unfold pushforwardSupportBulk bulkSupportSlack
  exact pushforwardSupportOn_support_false (fun i => ζ * T ≤ slack i) coord hbulk

theorem pushforwardWeightBulk_true (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (y : β)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    pushforwardWeightBulk slack ζ T coord y = binWeight coord y := by
  unfold pushforwardWeightBulk bulkSupportSlack
  exact pushforwardWeightOn_support_true (fun i => ζ * T ≤ slack i) coord y hbulk

theorem pushforwardWeightBulk_false (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (y : β)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    pushforwardWeightBulk slack ζ T coord y = 0 := by
  unfold pushforwardWeightBulk bulkSupportSlack
  exact pushforwardWeightOn_support_false (fun i => ζ * T ≤ slack i) coord y hbulk

@[simp] theorem densityPushforwardWeightOn_empty (coord : α → β) (ρ : β → ℚ) (y : β) :
    densityPushforwardWeightOn (∅ : Finset α) coord ρ y = 0 := by
  simp [densityPushforwardWeightOn]

theorem densityPushforwardWeightOn_univ (coord : α → β) (ρ : β → ℚ) (y : β) :
    densityPushforwardWeightOn Finset.univ coord ρ y = ρ y * binWeight coord y := by
  unfold densityPushforwardWeightOn
  rw [pushforwardWeightOn_univ]

theorem densityPushforwardWeightOn_support_true (P : α → Prop) [DecidablePred P]
    (coord : α → β) (ρ : β → ℚ) (y : β) (hP : ∀ i, P i) :
    densityPushforwardWeightOn (supportOf P) coord ρ y = ρ y * binWeight coord y := by
  unfold densityPushforwardWeightOn
  rw [pushforwardWeightOn_support_true P coord y hP]

theorem densityPushforwardWeightOn_support_false (P : α → Prop) [DecidablePred P]
    (coord : α → β) (ρ : β → ℚ) (y : β) (hP : ∀ i, ¬ P i) :
    densityPushforwardWeightOn (supportOf P) coord ρ y = 0 := by
  unfold densityPushforwardWeightOn
  rw [pushforwardWeightOn_support_false P coord y hP]
  ring

theorem densityPushforwardWeightBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (y : β)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    densityPushforwardWeightBulk slack ζ T coord ρ y = ρ y * binWeight coord y := by
  unfold densityPushforwardWeightBulk bulkSupportSlack
  exact densityPushforwardWeightOn_support_true (fun i => ζ * T ≤ slack i) coord ρ y hbulk

theorem densityPushforwardWeightBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (y : β)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    densityPushforwardWeightBulk slack ζ T coord ρ y = 0 := by
  unfold densityPushforwardWeightBulk bulkSupportSlack
  exact densityPushforwardWeightOn_support_false (fun i => ζ * T ≤ slack i) coord ρ y hbulk

theorem densityPushforwardWeightOn_congr_density
    (S : Finset α) (coord : α → β) {ρ ρ' : β → ℚ} (y : β)
    (hρ : ∀ z, ρ z = ρ' z) :
    densityPushforwardWeightOn S coord ρ y = densityPushforwardWeightOn S coord ρ' y := by
  unfold densityPushforwardWeightOn
  rw [hρ y]

theorem densityPushforwardWeightOn_scale_density
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) (c : ℚ) (y : β) :
    densityPushforwardWeightOn S coord (fun z => c * ρ z) y =
      c * densityPushforwardWeightOn S coord ρ y := by
  unfold densityPushforwardWeightOn
  ring

theorem densityPushforwardWeightOn_add_density
    (S : Finset α) (coord : α → β) (ρ₁ ρ₂ : β → ℚ) (y : β) :
    densityPushforwardWeightOn S coord (fun z => ρ₁ z + ρ₂ z) y =
      densityPushforwardWeightOn S coord ρ₁ y +
      densityPushforwardWeightOn S coord ρ₂ y := by
  unfold densityPushforwardWeightOn
  ring

theorem densityPushforwardWeightBulk_congr_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) {ρ ρ' : β → ℚ} (y : β)
    (hρ : ∀ z, ρ z = ρ' z) :
    densityPushforwardWeightBulk slack ζ T coord ρ y =
      densityPushforwardWeightBulk slack ζ T coord ρ' y := by
  unfold densityPushforwardWeightBulk
  exact densityPushforwardWeightOn_congr_density _ _ y hρ

theorem densityPushforwardWeightBulk_scale_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (c : ℚ) (y : β) :
    densityPushforwardWeightBulk slack ζ T coord (fun z => c * ρ z) y =
      c * densityPushforwardWeightBulk slack ζ T coord ρ y := by
  unfold densityPushforwardWeightBulk
  exact densityPushforwardWeightOn_scale_density _ _ ρ c y

theorem densityPushforwardWeightBulk_add_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ₁ ρ₂ : β → ℚ) (y : β) :
    densityPushforwardWeightBulk slack ζ T coord (fun z => ρ₁ z + ρ₂ z) y =
      densityPushforwardWeightBulk slack ζ T coord ρ₁ y +
      densityPushforwardWeightBulk slack ζ T coord ρ₂ y := by
  unfold densityPushforwardWeightBulk
  exact densityPushforwardWeightOn_add_density _ _ ρ₁ ρ₂ y

@[simp] theorem densityPushforwardSumOn_empty
    (coord : α → β) (ρ φ : β → ℚ) :
    densityPushforwardSumOn (∅ : Finset α) coord ρ φ = 0 := by
  simp [densityPushforwardSumOn]

theorem densityPushforwardSumOn_univ
    (coord : α → β) (ρ φ : β → ℚ) :
    densityPushforwardSumOn Finset.univ coord ρ φ =
      Finset.sum (Finset.univ.image coord) (fun y => φ y * (ρ y * binWeight coord y)) := by
  unfold densityPushforwardSumOn
  apply Finset.sum_congr rfl
  intro y hy
  rw [densityPushforwardWeightOn_univ]

theorem densityPushforwardSumOn_support_true
    (P : α → Prop) [DecidablePred P] (coord : α → β) (ρ φ : β → ℚ)
    (hP : ∀ i, P i) :
    densityPushforwardSumOn (supportOf P) coord ρ φ =
      Finset.sum (Finset.univ.image coord) (fun y => φ y * (ρ y * binWeight coord y)) := by
  rw [supportOf_true P hP]
  exact densityPushforwardSumOn_univ coord ρ φ

theorem densityPushforwardSumOn_support_false
    (P : α → Prop) [DecidablePred P] (coord : α → β) (ρ φ : β → ℚ)
    (hP : ∀ i, ¬ P i) :
    densityPushforwardSumOn (supportOf P) coord ρ φ = 0 := by
  rw [supportOf_false P hP]
  simp [densityPushforwardSumOn]

theorem densityPushforwardSumBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    densityPushforwardSumBulk slack ζ T coord ρ φ =
      Finset.sum (Finset.univ.image coord) (fun y => φ y * (ρ y * binWeight coord y)) := by
  unfold densityPushforwardSumBulk bulkSupportSlack
  exact densityPushforwardSumOn_support_true (fun i => ζ * T ≤ slack i) coord ρ φ hbulk

theorem densityPushforwardSumBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    densityPushforwardSumBulk slack ζ T coord ρ φ = 0 := by
  unfold densityPushforwardSumBulk bulkSupportSlack
  exact densityPushforwardSumOn_support_false (fun i => ζ * T ≤ slack i) coord ρ φ hbulk

theorem densityPushforwardSumOn_scale_test
    (S : Finset α) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) :
    densityPushforwardSumOn S coord ρ (fun y => c * φ y) =
      c * densityPushforwardSumOn S coord ρ φ := by
  unfold densityPushforwardSumOn
  calc
    Finset.sum (pushforwardSupportOn S coord)
        (fun y => (c * φ y) * densityPushforwardWeightOn S coord ρ y)
        =
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => c * (φ y * densityPushforwardWeightOn S coord ρ y)) := by
            apply Finset.sum_congr rfl
            intro y hy
            ring
    _ = c *
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ y * densityPushforwardWeightOn S coord ρ y) := by
            rw [← Finset.mul_sum]

theorem densityPushforwardSumOn_add_test
    (S : Finset α) (coord : α → β) (ρ φ₁ φ₂ : β → ℚ) :
    densityPushforwardSumOn S coord ρ (fun y => φ₁ y + φ₂ y) =
      densityPushforwardSumOn S coord ρ φ₁ +
      densityPushforwardSumOn S coord ρ φ₂ := by
  unfold densityPushforwardSumOn
  calc
    Finset.sum (pushforwardSupportOn S coord)
        (fun y => (φ₁ y + φ₂ y) * densityPushforwardWeightOn S coord ρ y)
        =
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ₁ y * densityPushforwardWeightOn S coord ρ y +
            φ₂ y * densityPushforwardWeightOn S coord ρ y) := by
              apply Finset.sum_congr rfl
              intro y hy
              ring
    _ =
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ₁ y * densityPushforwardWeightOn S coord ρ y) +
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ₂ y * densityPushforwardWeightOn S coord ρ y) := by
            rw [Finset.sum_add_distrib]

theorem densityPushforwardSumOn_congr_test
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    densityPushforwardSumOn S coord ρ φ = densityPushforwardSumOn S coord ρ φ' := by
  unfold densityPushforwardSumOn
  apply Finset.sum_congr rfl
  intro y hy
  rw [hφ y]

theorem densityPushforwardSumOn_congr_density
    (S : Finset α) (coord : α → β) {ρ ρ' : β → ℚ} (φ : β → ℚ)
    (hρ : ∀ y, ρ y = ρ' y) :
    densityPushforwardSumOn S coord ρ φ = densityPushforwardSumOn S coord ρ' φ := by
  unfold densityPushforwardSumOn
  apply Finset.sum_congr rfl
  intro y hy
  rw [densityPushforwardWeightOn_congr_density S coord y hρ]

theorem densityPushforwardSumOn_scale_density
    (S : Finset α) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) :
    densityPushforwardSumOn S coord (fun y => c * ρ y) φ =
      c * densityPushforwardSumOn S coord ρ φ := by
  unfold densityPushforwardSumOn
  calc
    Finset.sum (pushforwardSupportOn S coord)
        (fun y => φ y * densityPushforwardWeightOn S coord (fun z => c * ρ z) y)
        =
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => c * (φ y * densityPushforwardWeightOn S coord ρ y)) := by
            apply Finset.sum_congr rfl
            intro y hy
            rw [densityPushforwardWeightOn_scale_density]
            ring
    _ = c *
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ y * densityPushforwardWeightOn S coord ρ y) := by
            rw [← Finset.mul_sum]

theorem densityPushforwardSumOn_add_density
    (S : Finset α) (coord : α → β) (ρ₁ ρ₂ φ : β → ℚ) :
    densityPushforwardSumOn S coord (fun y => ρ₁ y + ρ₂ y) φ =
      densityPushforwardSumOn S coord ρ₁ φ +
      densityPushforwardSumOn S coord ρ₂ φ := by
  unfold densityPushforwardSumOn
  calc
    Finset.sum (pushforwardSupportOn S coord)
        (fun y => φ y * densityPushforwardWeightOn S coord (fun z => ρ₁ z + ρ₂ z) y)
        =
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ y * densityPushforwardWeightOn S coord ρ₁ y +
            φ y * densityPushforwardWeightOn S coord ρ₂ y) := by
              apply Finset.sum_congr rfl
              intro y hy
              rw [densityPushforwardWeightOn_add_density]
              ring
    _ =
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ y * densityPushforwardWeightOn S coord ρ₁ y) +
        Finset.sum (pushforwardSupportOn S coord)
          (fun y => φ y * densityPushforwardWeightOn S coord ρ₂ y) := by
            rw [Finset.sum_add_distrib]

theorem densityPushforwardSumBulk_congr_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) {ρ ρ' : β → ℚ} (φ : β → ℚ)
    (hρ : ∀ y, ρ y = ρ' y) :
    densityPushforwardSumBulk slack ζ T coord ρ φ =
      densityPushforwardSumBulk slack ζ T coord ρ' φ := by
  unfold densityPushforwardSumBulk
  exact densityPushforwardSumOn_congr_density _ _ φ hρ

theorem densityPushforwardSumBulk_scale_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) :
    densityPushforwardSumBulk slack ζ T coord (fun y => c * ρ y) φ =
      c * densityPushforwardSumBulk slack ζ T coord ρ φ := by
  unfold densityPushforwardSumBulk
  exact densityPushforwardSumOn_scale_density _ _ ρ φ c

theorem densityPushforwardSumBulk_add_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ₁ ρ₂ φ : β → ℚ) :
    densityPushforwardSumBulk slack ζ T coord (fun y => ρ₁ y + ρ₂ y) φ =
      densityPushforwardSumBulk slack ζ T coord ρ₁ φ +
      densityPushforwardSumBulk slack ζ T coord ρ₂ φ := by
  unfold densityPushforwardSumBulk
  exact densityPushforwardSumOn_add_density _ _ ρ₁ ρ₂ φ

theorem densityPushforwardSumBulk_congr_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    densityPushforwardSumBulk slack ζ T coord ρ φ =
      densityPushforwardSumBulk slack ζ T coord ρ φ' := by
  unfold densityPushforwardSumBulk
  exact densityPushforwardSumOn_congr_test _ _ ρ hφ

theorem densityPushforwardSumBulk_scale_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) :
    densityPushforwardSumBulk slack ζ T coord ρ (fun y => c * φ y) =
      c * densityPushforwardSumBulk slack ζ T coord ρ φ := by
  unfold densityPushforwardSumBulk
  exact densityPushforwardSumOn_scale_test _ _ ρ φ c

theorem densityPushforwardSumBulk_add_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ₁ φ₂ : β → ℚ) :
    densityPushforwardSumBulk slack ζ T coord ρ (fun y => φ₁ y + φ₂ y) =
      densityPushforwardSumBulk slack ζ T coord ρ φ₁ +
      densityPushforwardSumBulk slack ζ T coord ρ φ₂ := by
  unfold densityPushforwardSumBulk
  exact densityPushforwardSumOn_add_test _ _ ρ φ₁ φ₂

@[simp] theorem densityPushforwardMassOn_empty
    (coord : α → β) (ρ : β → ℚ) :
    densityPushforwardMassOn (∅ : Finset α) coord ρ = 0 := by
  simp [densityPushforwardMassOn]

theorem densityPushforwardMassOn_univ
    (coord : α → β) (ρ : β → ℚ) :
    densityPushforwardMassOn Finset.univ coord ρ =
      Finset.sum (Finset.univ.image coord) (fun y => ρ y * binWeight coord y) := by
  unfold densityPushforwardMassOn
  rw [densityPushforwardSumOn_univ]
  apply Finset.sum_congr rfl
  intro y hy
  ring

theorem densityPushforwardMassOn_support_true
    (P : α → Prop) [DecidablePred P] (coord : α → β) (ρ : β → ℚ)
    (hP : ∀ i, P i) :
    densityPushforwardMassOn (supportOf P) coord ρ =
      Finset.sum (Finset.univ.image coord) (fun y => ρ y * binWeight coord y) := by
  rw [supportOf_true P hP]
  exact densityPushforwardMassOn_univ coord ρ

theorem densityPushforwardMassOn_support_false
    (P : α → Prop) [DecidablePred P] (coord : α → β) (ρ : β → ℚ)
    (hP : ∀ i, ¬ P i) :
    densityPushforwardMassOn (supportOf P) coord ρ = 0 := by
  rw [supportOf_false P hP]
  simp [densityPushforwardMassOn]

theorem densityPushforwardMassBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    densityPushforwardMassBulk slack ζ T coord ρ =
      Finset.sum (Finset.univ.image coord) (fun y => ρ y * binWeight coord y) := by
  unfold densityPushforwardMassBulk bulkSupportSlack
  exact densityPushforwardMassOn_support_true (fun i => ζ * T ≤ slack i) coord ρ hbulk

theorem densityPushforwardMassBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    densityPushforwardMassBulk slack ζ T coord ρ = 0 := by
  unfold densityPushforwardMassBulk bulkSupportSlack
  exact densityPushforwardMassOn_support_false (fun i => ζ * T ≤ slack i) coord ρ hbulk

theorem densityPushforwardMassOn_congr_density
    (S : Finset α) (coord : α → β) {ρ ρ' : β → ℚ}
    (hρ : ∀ y, ρ y = ρ' y) :
    densityPushforwardMassOn S coord ρ = densityPushforwardMassOn S coord ρ' := by
  unfold densityPushforwardMassOn
  exact densityPushforwardSumOn_congr_density S coord (fun _ => (1 : ℚ)) hρ

theorem densityPushforwardMassOn_scale_density
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) (c : ℚ) :
    densityPushforwardMassOn S coord (fun y => c * ρ y) =
      c * densityPushforwardMassOn S coord ρ := by
  unfold densityPushforwardMassOn
  exact densityPushforwardSumOn_scale_density S coord ρ (fun _ => (1 : ℚ)) c

theorem densityPushforwardMassOn_add_density
    (S : Finset α) (coord : α → β) (ρ₁ ρ₂ : β → ℚ) :
    densityPushforwardMassOn S coord (fun y => ρ₁ y + ρ₂ y) =
      densityPushforwardMassOn S coord ρ₁ +
      densityPushforwardMassOn S coord ρ₂ := by
  unfold densityPushforwardMassOn
  exact densityPushforwardSumOn_add_density S coord ρ₁ ρ₂ (fun _ => (1 : ℚ))

theorem densityPushforwardMassBulk_congr_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) {ρ ρ' : β → ℚ}
    (hρ : ∀ y, ρ y = ρ' y) :
    densityPushforwardMassBulk slack ζ T coord ρ =
      densityPushforwardMassBulk slack ζ T coord ρ' := by
  unfold densityPushforwardMassBulk
  exact densityPushforwardMassOn_congr_density _ _ hρ

theorem densityPushforwardMassBulk_scale_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (c : ℚ) :
    densityPushforwardMassBulk slack ζ T coord (fun y => c * ρ y) =
      c * densityPushforwardMassBulk slack ζ T coord ρ := by
  unfold densityPushforwardMassBulk
  exact densityPushforwardMassOn_scale_density _ _ ρ c

theorem densityPushforwardMassBulk_add_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ₁ ρ₂ : β → ℚ) :
    densityPushforwardMassBulk slack ζ T coord (fun y => ρ₁ y + ρ₂ y) =
      densityPushforwardMassBulk slack ζ T coord ρ₁ +
      densityPushforwardMassBulk slack ζ T coord ρ₂ := by
  unfold densityPushforwardMassBulk
  exact densityPushforwardMassOn_add_density _ _ ρ₁ ρ₂

@[simp] theorem densityPushforwardAvgOn_empty
    (coord : α → β) (ρ φ : β → ℚ) :
    densityPushforwardAvgOn (∅ : Finset α) coord ρ φ = 0 := by
  simp [densityPushforwardAvgOn]

theorem densityPushforwardAvgOn_support_true
    (P : α → Prop) [DecidablePred P] (coord : α → β) (ρ φ : β → ℚ)
    (hP : ∀ i, P i) :
    densityPushforwardAvgOn (supportOf P) coord ρ φ =
      densityPushforwardAvgOn Finset.univ coord ρ φ := by
  rw [supportOf_true P hP]

theorem densityPushforwardAvgOn_support_false
    (P : α → Prop) [DecidablePred P] (coord : α → β) (ρ φ : β → ℚ)
    (hP : ∀ i, ¬ P i) :
    densityPushforwardAvgOn (supportOf P) coord ρ φ = 0 := by
  rw [supportOf_false P hP]
  simp [densityPushforwardAvgOn]

theorem densityPushforwardAvgBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    densityPushforwardAvgBulk slack ζ T coord ρ φ =
      densityPushforwardAvgOn Finset.univ coord ρ φ := by
  unfold densityPushforwardAvgBulk bulkSupportSlack
  exact densityPushforwardAvgOn_support_true (fun i => ζ * T ≤ slack i) coord ρ φ hbulk

theorem densityPushforwardAvgBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    densityPushforwardAvgBulk slack ζ T coord ρ φ = 0 := by
  unfold densityPushforwardAvgBulk bulkSupportSlack
  exact densityPushforwardAvgOn_support_false (fun i => ζ * T ≤ slack i) coord ρ φ hbulk

theorem densityPushforwardAvgOn_congr_test
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    densityPushforwardAvgOn S coord ρ φ = densityPushforwardAvgOn S coord ρ φ' := by
  by_cases hmass : densityPushforwardMassOn S coord ρ = 0
  · simp [densityPushforwardAvgOn, hmass]
  · simp [densityPushforwardAvgOn, hmass]
    rw [densityPushforwardSumOn_congr_test S coord ρ hφ]

theorem densityPushforwardAvgOn_scale_test
    (S : Finset α) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) :
    densityPushforwardAvgOn S coord ρ (fun y => c * φ y) =
      c * densityPushforwardAvgOn S coord ρ φ := by
  by_cases hmass : densityPushforwardMassOn S coord ρ = 0
  · simp [densityPushforwardAvgOn, hmass]
  · simp [densityPushforwardAvgOn, hmass, densityPushforwardSumOn_scale_test]
    field_simp [hmass]

theorem densityPushforwardAvgOn_add_test
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) (φ₁ φ₂ : β → ℚ) :
    densityPushforwardAvgOn S coord ρ (fun y => φ₁ y + φ₂ y) =
      densityPushforwardAvgOn S coord ρ φ₁ +
      densityPushforwardAvgOn S coord ρ φ₂ := by
  by_cases hmass : densityPushforwardMassOn S coord ρ = 0
  · simp [densityPushforwardAvgOn, hmass]
  · simp [densityPushforwardAvgOn, hmass, densityPushforwardSumOn_add_test]
    field_simp [hmass]

theorem densityPushforwardAvgOn_scale_density
    (S : Finset α) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) (hc : c ≠ 0) :
    densityPushforwardAvgOn S coord (fun y => c * ρ y) φ =
      densityPushforwardAvgOn S coord ρ φ := by
  by_cases hmass : densityPushforwardMassOn S coord ρ = 0
  · have hmass' : densityPushforwardMassOn S coord (fun y => c * ρ y) = 0 := by
      rw [densityPushforwardMassOn_scale_density, hmass]
      ring
    simp [densityPushforwardAvgOn, hmass, hmass']
  · have hmass' : densityPushforwardMassOn S coord (fun y => c * ρ y) ≠ 0 := by
      rw [densityPushforwardMassOn_scale_density]
      exact mul_ne_zero hc hmass
    calc
      densityPushforwardAvgOn S coord (fun y => c * ρ y) φ
          = densityPushforwardSumOn S coord (fun y => c * ρ y) φ /
              densityPushforwardMassOn S coord (fun y => c * ρ y) := by
              simp [densityPushforwardAvgOn, hmass']
      _ = (c * densityPushforwardSumOn S coord ρ φ) /
            (c * densityPushforwardMassOn S coord ρ) := by
            rw [densityPushforwardSumOn_scale_density, densityPushforwardMassOn_scale_density]
      _ = densityPushforwardSumOn S coord ρ φ / densityPushforwardMassOn S coord ρ := by
            field_simp [hc, hmass]
      _ = densityPushforwardAvgOn S coord ρ φ := by
            simp [densityPushforwardAvgOn, hmass]

theorem densityPushforwardAvgOn_congr_density
    (S : Finset α) (coord : α → β) {ρ ρ' : β → ℚ} (φ : β → ℚ)
    (hρ : ∀ y, ρ y = ρ' y) :
    densityPushforwardAvgOn S coord ρ φ = densityPushforwardAvgOn S coord ρ' φ := by
  unfold densityPushforwardAvgOn
  have hmass : densityPushforwardMassOn S coord ρ = densityPushforwardMassOn S coord ρ' := by
    exact densityPushforwardMassOn_congr_density S coord hρ
  have hsum : densityPushforwardSumOn S coord ρ φ = densityPushforwardSumOn S coord ρ' φ := by
    exact densityPushforwardSumOn_congr_density S coord φ hρ
  by_cases h0 : densityPushforwardMassOn S coord ρ = 0
  · have h0' : densityPushforwardMassOn S coord ρ' = 0 := by
      rw [← hmass]
      exact h0
    simp [h0, h0']
  · have h0' : densityPushforwardMassOn S coord ρ' ≠ 0 := by
      rw [← hmass]
      exact h0
    simp [h0, h0', hsum, hmass]

theorem densityPushforwardAvgOn_const
    (S : Finset α) (coord : α → β) (ρ : β → ℚ) (c : ℚ)
    (hmass : densityPushforwardMassOn S coord ρ ≠ 0) :
    densityPushforwardAvgOn S coord ρ (fun _ => c) = c := by
  calc
    densityPushforwardAvgOn S coord ρ (fun _ => c)
        = densityPushforwardSumOn S coord ρ (fun _ => c) /
            densityPushforwardMassOn S coord ρ := by
              simp [densityPushforwardAvgOn, hmass]
    _ = (c * densityPushforwardSumOn S coord ρ (fun _ => (1 : ℚ))) /
          densityPushforwardMassOn S coord ρ := by
          rw [show densityPushforwardSumOn S coord ρ (fun _ => c) =
                c * densityPushforwardSumOn S coord ρ (fun _ => (1 : ℚ)) by
                  simpa using densityPushforwardSumOn_scale_test S coord ρ (fun _ => (1 : ℚ)) c]
    _ = (c * densityPushforwardMassOn S coord ρ) /
          densityPushforwardMassOn S coord ρ := by
          rfl
    _ = c := by
          field_simp [hmass]

theorem densityPushforwardAvgBulk_congr_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    densityPushforwardAvgBulk slack ζ T coord ρ φ =
      densityPushforwardAvgBulk slack ζ T coord ρ φ' := by
  unfold densityPushforwardAvgBulk
  exact densityPushforwardAvgOn_congr_test _ _ _ hφ

theorem densityPushforwardAvgBulk_scale_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) :
    densityPushforwardAvgBulk slack ζ T coord ρ (fun y => c * φ y) =
      c * densityPushforwardAvgBulk slack ζ T coord ρ φ := by
  unfold densityPushforwardAvgBulk
  exact densityPushforwardAvgOn_scale_test _ _ _ _ c

theorem densityPushforwardAvgBulk_add_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (φ₁ φ₂ : β → ℚ) :
    densityPushforwardAvgBulk slack ζ T coord ρ (fun y => φ₁ y + φ₂ y) =
      densityPushforwardAvgBulk slack ζ T coord ρ φ₁ +
      densityPushforwardAvgBulk slack ζ T coord ρ φ₂ := by
  unfold densityPushforwardAvgBulk
  exact densityPushforwardAvgOn_add_test _ _ _ _ _

theorem densityPushforwardAvgBulk_scale_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ φ : β → ℚ) (c : ℚ) (hc : c ≠ 0) :
    densityPushforwardAvgBulk slack ζ T coord (fun y => c * ρ y) φ =
      densityPushforwardAvgBulk slack ζ T coord ρ φ := by
  unfold densityPushforwardAvgBulk
  exact densityPushforwardAvgOn_scale_density _ _ _ _ c hc

theorem densityPushforwardAvgBulk_congr_density
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) {ρ ρ' : β → ℚ} (φ : β → ℚ)
    (hρ : ∀ y, ρ y = ρ' y) :
    densityPushforwardAvgBulk slack ζ T coord ρ φ =
      densityPushforwardAvgBulk slack ζ T coord ρ' φ := by
  unfold densityPushforwardAvgBulk
  exact densityPushforwardAvgOn_congr_density _ _ φ hρ

theorem densityPushforwardAvgBulk_const
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (ρ : β → ℚ) (c : ℚ)
    (hmass : densityPushforwardMassBulk slack ζ T coord ρ ≠ 0) :
    densityPushforwardAvgBulk slack ζ T coord ρ (fun _ => c) = c := by
  simpa [densityPushforwardAvgBulk, densityPushforwardMassBulk] using
    (densityPushforwardAvgOn_const
      (S := bulkSupportSlack slack ζ T) (coord := coord) (ρ := ρ) (c := c)
      (hmass := by simpa [densityPushforwardMassBulk] using hmass))

theorem densityPushforwardWeightOn_one
    (S : Finset α) (coord : α → β) (y : β) :
    densityPushforwardWeightOn S coord (fun _ => (1 : ℚ)) y =
      pushforwardWeightOn S coord y := by
  simp [densityPushforwardWeightOn, pushforwardWeightOn]

theorem densityPushforwardWeightBulk_one
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (y : β) :
    densityPushforwardWeightBulk slack ζ T coord (fun _ => (1 : ℚ)) y =
      pushforwardWeightBulk slack ζ T coord y := by
  simp [densityPushforwardWeightBulk, densityPushforwardWeightOn_one, pushforwardWeightBulk]

@[simp] theorem pushforwardSumOn_empty
    (coord : α → β) (φ : β → ℚ) :
    pushforwardSumOn (∅ : Finset α) coord φ = 0 := by
  simp [pushforwardSumOn]

theorem pushforwardSumOn_univ
    (coord : α → β) (φ : β → ℚ) :
    pushforwardSumOn Finset.univ coord φ =
      Finset.sum (Finset.univ.image coord) (fun y => φ y * binWeight coord y) := by
  unfold pushforwardSumOn
  simpa using densityPushforwardSumOn_univ coord (fun _ => (1 : ℚ)) φ

theorem pushforwardSumOn_support_true
    (P : α → Prop) [DecidablePred P] (coord : α → β) (φ : β → ℚ)
    (hP : ∀ i, P i) :
    pushforwardSumOn (supportOf P) coord φ =
      Finset.sum (Finset.univ.image coord) (fun y => φ y * binWeight coord y) := by
  unfold pushforwardSumOn
  simpa using densityPushforwardSumOn_support_true P coord (fun _ => (1 : ℚ)) φ hP

theorem pushforwardSumOn_support_false
    (P : α → Prop) [DecidablePred P] (coord : α → β) (φ : β → ℚ)
    (hP : ∀ i, ¬ P i) :
    pushforwardSumOn (supportOf P) coord φ = 0 := by
  unfold pushforwardSumOn
  simpa using densityPushforwardSumOn_support_false P coord (fun _ => (1 : ℚ)) φ hP

theorem pushforwardSumBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    pushforwardSumBulk slack ζ T coord φ =
      Finset.sum (Finset.univ.image coord) (fun y => φ y * binWeight coord y) := by
  unfold pushforwardSumBulk pushforwardSumOn
  simpa using densityPushforwardSumBulk_true slack ζ T coord (fun _ => (1 : ℚ)) φ hbulk

theorem pushforwardSumBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    pushforwardSumBulk slack ζ T coord φ = 0 := by
  unfold pushforwardSumBulk pushforwardSumOn
  simpa using densityPushforwardSumBulk_false slack ζ T coord (fun _ => (1 : ℚ)) φ hbulk

theorem pushforwardSumOn_congr_test
    (S : Finset α) (coord : α → β) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    pushforwardSumOn S coord φ = pushforwardSumOn S coord φ' := by
  unfold pushforwardSumOn
  simpa using densityPushforwardSumOn_congr_test S coord (fun _ => (1 : ℚ)) hφ

theorem pushforwardSumOn_scale_test
    (S : Finset α) (coord : α → β) (φ : β → ℚ) (c : ℚ) :
    pushforwardSumOn S coord (fun y => c * φ y) =
      c * pushforwardSumOn S coord φ := by
  unfold pushforwardSumOn
  simpa using densityPushforwardSumOn_scale_test S coord (fun _ => (1 : ℚ)) φ c

theorem pushforwardSumOn_add_test
    (S : Finset α) (coord : α → β) (φ₁ φ₂ : β → ℚ) :
    pushforwardSumOn S coord (fun y => φ₁ y + φ₂ y) =
      pushforwardSumOn S coord φ₁ +
      pushforwardSumOn S coord φ₂ := by
  unfold pushforwardSumOn
  simpa using densityPushforwardSumOn_add_test S coord (fun _ => (1 : ℚ)) φ₁ φ₂

theorem pushforwardSumBulk_congr_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    pushforwardSumBulk slack ζ T coord φ =
      pushforwardSumBulk slack ζ T coord φ' := by
  unfold pushforwardSumBulk
  exact pushforwardSumOn_congr_test _ _ hφ

theorem pushforwardSumBulk_scale_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) (c : ℚ) :
    pushforwardSumBulk slack ζ T coord (fun y => c * φ y) =
      c * pushforwardSumBulk slack ζ T coord φ := by
  unfold pushforwardSumBulk
  exact pushforwardSumOn_scale_test _ _ _ _

theorem pushforwardSumBulk_add_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ₁ φ₂ : β → ℚ) :
    pushforwardSumBulk slack ζ T coord (fun y => φ₁ y + φ₂ y) =
      pushforwardSumBulk slack ζ T coord φ₁ +
      pushforwardSumBulk slack ζ T coord φ₂ := by
  unfold pushforwardSumBulk
  exact pushforwardSumOn_add_test _ _ _ _

theorem pushforwardSumOn_const
    (S : Finset α) (coord : α → β) (c : ℚ) :
    pushforwardSumOn S coord (fun _ => c) = c * pushforwardMassOn S coord := by
  unfold pushforwardSumOn pushforwardMassOn
  simpa using densityPushforwardSumOn_scale_test S coord (fun _ => (1 : ℚ)) (fun _ => (1 : ℚ)) c

theorem pushforwardSumBulk_const
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (c : ℚ) :
    pushforwardSumBulk slack ζ T coord (fun _ => c) = c * pushforwardMassBulk slack ζ T coord := by
  simpa [pushforwardSumBulk, pushforwardMassBulk] using
    (pushforwardSumOn_const (S := bulkSupportSlack slack ζ T) (coord := coord) (c := c))

theorem pushforwardSumOn_eq_meanOn_comp
    (S : Finset α) (coord : α → β) (φ : β → ℚ) :
    pushforwardSumOn S coord φ = meanOn S (fun i => φ (coord i)) := by
  by_cases hS : S.card = 0
  · have hSe : S = ∅ := Finset.card_eq_zero.mp hS
    subst hSe
    simp [pushforwardSumOn, meanOn]
  · unfold pushforwardSumOn densityPushforwardSumOn densityPushforwardWeightOn
      pushforwardWeightOn binWeightOn meanOn
    simp [hS, pushforwardSupportOn]
    have hf : ∀ i ∈ S, coord i ∈ S.image coord := by
      intro i hi
      exact Finset.mem_image_of_mem coord hi
    calc
      ∑ x ∈ S.image coord, φ x * (↑(fiberIn S coord x).card / ↑S.card)
          = ∑ x ∈ S.image coord, ∑ i ∈ S with coord i = x, φ (coord i) / ↑S.card := by
              apply Finset.sum_congr rfl
              intro y hy
              have hcongr :
                  ∀ i ∈ S.filter (fun i => coord i = y), φ (coord i) / ↑S.card = φ y / ↑S.card := by
                intro i hi
                have hiy : coord i = y := by
                  simp at hi
                  exact hi.2
                rw [hiy]
              rw [Finset.sum_congr rfl hcongr, Finset.sum_const, nsmul_eq_mul]
              have hcard : ((fiberIn S coord y).card : ℚ) = ({i ∈ S | coord i = y}.card : ℚ) := by
                rfl
              rw [hcard]
              ring
      _ = ∑ i ∈ S, φ (coord i) / ↑S.card := by
            simpa using (Finset.sum_fiberwise_of_maps_to hf (fun i => φ (coord i) / ↑S.card))
      _ = (∑ i ∈ S, φ (coord i)) / ↑S.card := by
            rw [← Finset.sum_div]

theorem densityPushforwardSumOn_one
    (S : Finset α) (coord : α → β) (φ : β → ℚ) :
    densityPushforwardSumOn S coord (fun _ => (1 : ℚ)) φ = pushforwardSumOn S coord φ := by
  rfl

theorem densityPushforwardSumBulk_one
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) :
    densityPushforwardSumBulk slack ζ T coord (fun _ => (1 : ℚ)) φ =
      pushforwardSumBulk slack ζ T coord φ := by
  rfl

theorem densityPushforwardMassOn_one
    (S : Finset α) (coord : α → β) :
    densityPushforwardMassOn S coord (fun _ => (1 : ℚ)) = pushforwardMassOn S coord := by
  rfl

theorem densityPushforwardMassBulk_one
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) :
    densityPushforwardMassBulk slack ζ T coord (fun _ => (1 : ℚ)) =
      pushforwardMassBulk slack ζ T coord := by
  rfl

theorem densityPushforwardAvgOn_one
    (S : Finset α) (coord : α → β) (φ : β → ℚ) :
    densityPushforwardAvgOn S coord (fun _ => (1 : ℚ)) φ = pushforwardAvgOn S coord φ := by
  rfl

theorem densityPushforwardAvgBulk_one
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) :
    densityPushforwardAvgBulk slack ζ T coord (fun _ => (1 : ℚ)) φ =
      pushforwardAvgBulk slack ζ T coord φ := by
  rfl

@[simp] theorem pushforwardMassOn_empty
    (coord : α → β) :
    pushforwardMassOn (∅ : Finset α) coord = 0 := by
  simp [pushforwardMassOn]

theorem pushforwardMassOn_univ
    (coord : α → β) :
    pushforwardMassOn Finset.univ coord =
      Finset.sum (Finset.univ.image coord) (fun y => binWeight coord y) := by
  unfold pushforwardMassOn
  simpa using densityPushforwardMassOn_univ coord (fun _ => (1 : ℚ))

theorem pushforwardMassOn_support_true
    (P : α → Prop) [DecidablePred P] (coord : α → β)
    (hP : ∀ i, P i) :
    pushforwardMassOn (supportOf P) coord =
      Finset.sum (Finset.univ.image coord) (fun y => binWeight coord y) := by
  unfold pushforwardMassOn
  simpa using densityPushforwardMassOn_support_true P coord (fun _ => (1 : ℚ)) hP

theorem pushforwardMassOn_support_false
    (P : α → Prop) [DecidablePred P] (coord : α → β)
    (hP : ∀ i, ¬ P i) :
    pushforwardMassOn (supportOf P) coord = 0 := by
  simp [pushforwardMassOn, supportOf_false P hP]

theorem pushforwardMassBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    pushforwardMassBulk slack ζ T coord =
      Finset.sum (Finset.univ.image coord) (fun y => binWeight coord y) := by
  unfold pushforwardMassBulk pushforwardMassOn
  simpa using densityPushforwardMassBulk_true slack ζ T coord (fun _ => (1 : ℚ)) hbulk

theorem pushforwardMassBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    pushforwardMassBulk slack ζ T coord = 0 := by
  unfold pushforwardMassBulk pushforwardMassOn
  simpa using densityPushforwardMassBulk_false slack ζ T coord (fun _ => (1 : ℚ)) hbulk

theorem pushforwardMassOn_eq_one_of_card_ne_zero
    (S : Finset α) (coord : α → β) (hS : S.card ≠ 0) :
    pushforwardMassOn S coord = 1 := by
  have hsumNat : Finset.sum (S.image coord) (fun b => (fiberIn S coord b).card) = S.card := by
    simpa [fiberIn] using (Finset.card_eq_sum_card_image coord S).symm
  have hsumRat' : Finset.sum (S.image coord) (fun b => ((fiberIn S coord b).card : ℚ)) = (S.card : ℚ) := by
    exact_mod_cast hsumNat
  have hq : (S.card : ℚ) ≠ 0 := by
    exact_mod_cast hS
  have hsumRat : (↑(Finset.sum (S.image coord) fun b => (fiberIn S coord b).card) : ℚ) = S.card := by
    exact_mod_cast hsumNat
  unfold pushforwardMassOn densityPushforwardMassOn densityPushforwardSumOn
    densityPushforwardWeightOn pushforwardWeightOn
  simp [binWeightOn, hS, pushforwardSupportOn]
  calc
    (Finset.sum (S.image coord) (fun x => ↑(fiberIn S coord x).card / ↑S.card))
        = (Finset.sum (S.image coord) (fun x => ↑(fiberIn S coord x).card)) / ↑S.card := by
            rw [← Finset.sum_div]
    _ = (S.card : ℚ) / ↑S.card := by rw [hsumRat']
    _ = 1 := by field_simp [hq]

theorem pushforwardMassBulk_eq_one_of_card_ne_zero
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β)
    (hbulk : (bulkSupportSlack slack ζ T).card ≠ 0) :
    pushforwardMassBulk slack ζ T coord = 1 := by
  unfold pushforwardMassBulk
  exact pushforwardMassOn_eq_one_of_card_ne_zero _ _ hbulk

theorem pushforwardSumOn_const_of_card_ne_zero
    (S : Finset α) (coord : α → β) (c : ℚ)
    (hS : S.card ≠ 0) :
    pushforwardSumOn S coord (fun _ => c) = c := by
  rw [pushforwardSumOn_const, pushforwardMassOn_eq_one_of_card_ne_zero S coord hS]
  ring

theorem pushforwardSumBulk_const_of_card_ne_zero
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (c : ℚ)
    (hbulk : (bulkSupportSlack slack ζ T).card ≠ 0) :
    pushforwardSumBulk slack ζ T coord (fun _ => c) = c := by
  rw [pushforwardSumBulk_const, pushforwardMassBulk_eq_one_of_card_ne_zero slack ζ T coord hbulk]
  ring

@[simp] theorem pushforwardAvgOn_empty
    (coord : α → β) (φ : β → ℚ) :
    pushforwardAvgOn (∅ : Finset α) coord φ = 0 := by
  simp [pushforwardAvgOn]

theorem pushforwardAvgOn_support_true
    (P : α → Prop) [DecidablePred P] (coord : α → β) (φ : β → ℚ)
    (hP : ∀ i, P i) :
    pushforwardAvgOn (supportOf P) coord φ =
      pushforwardAvgOn Finset.univ coord φ := by
  unfold pushforwardAvgOn
  simpa using densityPushforwardAvgOn_support_true P coord (fun _ => (1 : ℚ)) φ hP

theorem pushforwardAvgOn_support_false
    (P : α → Prop) [DecidablePred P] (coord : α → β) (φ : β → ℚ)
    (hP : ∀ i, ¬ P i) :
    pushforwardAvgOn (supportOf P) coord φ = 0 := by
  unfold pushforwardAvgOn
  simpa using densityPushforwardAvgOn_support_false P coord (fun _ => (1 : ℚ)) φ hP

theorem pushforwardAvgBulk_true
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    pushforwardAvgBulk slack ζ T coord φ =
      pushforwardAvgOn Finset.univ coord φ := by
  unfold pushforwardAvgBulk pushforwardAvgOn
  simpa using densityPushforwardAvgBulk_true slack ζ T coord (fun _ => (1 : ℚ)) φ hbulk

theorem pushforwardAvgBulk_false
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    pushforwardAvgBulk slack ζ T coord φ = 0 := by
  unfold pushforwardAvgBulk pushforwardAvgOn
  simpa using densityPushforwardAvgBulk_false slack ζ T coord (fun _ => (1 : ℚ)) φ hbulk

theorem pushforwardAvgOn_congr_test
    (S : Finset α) (coord : α → β) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    pushforwardAvgOn S coord φ = pushforwardAvgOn S coord φ' := by
  unfold pushforwardAvgOn
  simpa using densityPushforwardAvgOn_congr_test S coord (fun _ => (1 : ℚ)) hφ

theorem pushforwardAvgOn_scale_test
    (S : Finset α) (coord : α → β) (φ : β → ℚ) (c : ℚ) :
    pushforwardAvgOn S coord (fun y => c * φ y) =
      c * pushforwardAvgOn S coord φ := by
  unfold pushforwardAvgOn
  simpa using densityPushforwardAvgOn_scale_test S coord (fun _ => (1 : ℚ)) φ c

theorem pushforwardAvgOn_add_test
    (S : Finset α) (coord : α → β) (φ₁ φ₂ : β → ℚ) :
    pushforwardAvgOn S coord (fun y => φ₁ y + φ₂ y) =
      pushforwardAvgOn S coord φ₁ +
      pushforwardAvgOn S coord φ₂ := by
  unfold pushforwardAvgOn
  simpa using densityPushforwardAvgOn_add_test S coord (fun _ => (1 : ℚ)) φ₁ φ₂

theorem pushforwardAvgOn_const
    (S : Finset α) (coord : α → β) (c : ℚ)
    (hmass : pushforwardMassOn S coord ≠ 0) :
    pushforwardAvgOn S coord (fun _ => c) = c := by
  simpa [pushforwardAvgOn, pushforwardMassOn] using
    (densityPushforwardAvgOn_const
      (S := S) (coord := coord) (ρ := fun _ => (1 : ℚ)) (c := c)
      (hmass := by simpa [pushforwardMassOn] using hmass))

theorem pushforwardAvgOn_eq_pushforwardSumOn_of_card_ne_zero
    (S : Finset α) (coord : α → β) (φ : β → ℚ)
    (hS : S.card ≠ 0) :
    pushforwardAvgOn S coord φ = pushforwardSumOn S coord φ := by
  have hmass1 : densityPushforwardMassOn S coord (fun _ => (1 : ℚ)) = 1 := by
    simpa [pushforwardMassOn] using pushforwardMassOn_eq_one_of_card_ne_zero S coord hS
  unfold pushforwardAvgOn pushforwardSumOn
  simp [densityPushforwardAvgOn, hmass1]

theorem pushforwardAvgOn_eq_pushforwardSumOn
    (S : Finset α) (coord : α → β) (φ : β → ℚ) :
    pushforwardAvgOn S coord φ = pushforwardSumOn S coord φ := by
  by_cases hS : S.card = 0
  · exact by
      have hSe : S = ∅ := Finset.card_eq_zero.mp hS
      subst hSe
      simp [pushforwardAvgOn, pushforwardSumOn]
  · exact pushforwardAvgOn_eq_pushforwardSumOn_of_card_ne_zero S coord φ hS

theorem pushforwardAvgOn_eq_meanOn_comp
    (S : Finset α) (coord : α → β) (φ : β → ℚ) :
    pushforwardAvgOn S coord φ = meanOn S (fun i => φ (coord i)) := by
  rw [pushforwardAvgOn_eq_pushforwardSumOn, pushforwardSumOn_eq_meanOn_comp]

theorem pushforwardAvgOn_const_of_card_ne_zero
    (S : Finset α) (coord : α → β) (c : ℚ)
    (hS : S.card ≠ 0) :
    pushforwardAvgOn S coord (fun _ => c) = c := by
  apply pushforwardAvgOn_const
  rw [pushforwardMassOn_eq_one_of_card_ne_zero S coord hS]
  norm_num

theorem pushforwardAvgBulk_congr_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) {φ φ' : β → ℚ}
    (hφ : ∀ y, φ y = φ' y) :
    pushforwardAvgBulk slack ζ T coord φ =
      pushforwardAvgBulk slack ζ T coord φ' := by
  unfold pushforwardAvgBulk
  exact pushforwardAvgOn_congr_test _ _ hφ

theorem pushforwardAvgBulk_scale_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) (c : ℚ) :
    pushforwardAvgBulk slack ζ T coord (fun y => c * φ y) =
      c * pushforwardAvgBulk slack ζ T coord φ := by
  unfold pushforwardAvgBulk
  exact pushforwardAvgOn_scale_test _ _ _ _

theorem pushforwardAvgBulk_add_test
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ₁ φ₂ : β → ℚ) :
    pushforwardAvgBulk slack ζ T coord (fun y => φ₁ y + φ₂ y) =
      pushforwardAvgBulk slack ζ T coord φ₁ +
      pushforwardAvgBulk slack ζ T coord φ₂ := by
  unfold pushforwardAvgBulk
  exact pushforwardAvgOn_add_test _ _ _ _

theorem pushforwardAvgBulk_const
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (c : ℚ)
    (hmass : pushforwardMassBulk slack ζ T coord ≠ 0) :
    pushforwardAvgBulk slack ζ T coord (fun _ => c) = c := by
  simpa [pushforwardAvgBulk, pushforwardMassBulk, pushforwardAvgOn, pushforwardMassOn] using
    (pushforwardAvgOn_const
      (S := bulkSupportSlack slack ζ T) (coord := coord) (c := c)
      (hmass := by simpa [pushforwardMassBulk, pushforwardMassOn] using hmass))

theorem pushforwardAvgBulk_eq_pushforwardSumBulk_of_card_ne_zero
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ)
    (hbulk : (bulkSupportSlack slack ζ T).card ≠ 0) :
    pushforwardAvgBulk slack ζ T coord φ = pushforwardSumBulk slack ζ T coord φ := by
  unfold pushforwardAvgBulk pushforwardSumBulk
  exact pushforwardAvgOn_eq_pushforwardSumOn_of_card_ne_zero _ _ _ hbulk

theorem pushforwardAvgBulk_eq_pushforwardSumBulk
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) :
    pushforwardAvgBulk slack ζ T coord φ = pushforwardSumBulk slack ζ T coord φ := by
  unfold pushforwardAvgBulk pushforwardSumBulk
  exact pushforwardAvgOn_eq_pushforwardSumOn _ _ _

theorem pushforwardAvgBulk_eq_meanOn_comp
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) :
    pushforwardAvgBulk slack ζ T coord φ =
      meanOn (bulkSupportSlack slack ζ T) (fun i => φ (coord i)) := by
  unfold pushforwardAvgBulk
  exact pushforwardAvgOn_eq_meanOn_comp _ _ _

theorem pushforwardSumBulk_eq_meanOn_comp
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (φ : β → ℚ) :
    pushforwardSumBulk slack ζ T coord φ =
      meanOn (bulkSupportSlack slack ζ T) (fun i => φ (coord i)) := by
  rw [← pushforwardAvgBulk_eq_pushforwardSumBulk]
  exact pushforwardAvgBulk_eq_meanOn_comp slack ζ T coord φ

theorem pushforwardAvgBulk_const_of_card_ne_zero
    (slack : α → ℚ) (ζ T : ℚ) (coord : α → β) (c : ℚ)
    (hbulk : (bulkSupportSlack slack ζ T).card ≠ 0) :
    pushforwardAvgBulk slack ζ T coord (fun _ => c) = c := by
  apply pushforwardAvgBulk_const
  rw [pushforwardMassBulk_eq_one_of_card_ne_zero slack ζ T coord hbulk]
  norm_num

theorem fiberIn_singleton_self (a : α) (bin : α → β) :
    fiberIn ({a} : Finset α) bin (bin a) = ({a} : Finset α) := by
  ext i
  constructor
  · intro hi
    simp [fiberIn] at hi
    simpa using hi.1
  · intro hi
    simp [fiberIn]
    constructor
    · simpa using hi
    · have hia : i = a := by simpa using hi
      simpa [hia]

theorem paperEstimatorOn_singleton (a : α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn ({a} : Finset α) bin U V = 0 := by
  unfold paperEstimatorOn
  have himage : ({a} : Finset α).image bin = ({bin a} : Finset β) := by
    ext b
    simp
  rw [himage]
  have hcov : covOn ({a} : Finset α) U V = 0 := by
    simp [covOn, meanOn]
  simp [binWeightOn, fiberIn_singleton_self, hcov]

theorem fiberIn_of_constant_bin_on_support (S : Finset α) (bin : α → β) (b0 : β)
    (hbin : ∀ i ∈ S, bin i = b0) :
    fiberIn S bin b0 = S := by
  ext i
  constructor
  · intro hi
    simp [fiberIn] at hi
    exact hi.1
  · intro hi
    simp [fiberIn]
    constructor
    · exact hi
    · exact hbin i hi

omit [Fintype α] [DecidableEq α] in
@[simp] theorem meanOn_empty (f : α → ℚ) : meanOn (∅ : Finset α) f = 0 := by
  simp [meanOn]

omit [Fintype α] [DecidableEq α] in
@[simp] theorem meanOn_singleton (a : α) (f : α → ℚ) :
    meanOn ({a} : Finset α) f = f a := by
  simp [meanOn]

omit [Fintype α] [DecidableEq α] in
@[simp] theorem covOn_singleton (a : α) (U V : α → ℚ) :
    covOn ({a} : Finset α) U V = 0 := by
  simp [covOn, meanOn]

omit [Fintype α] [DecidableEq α] in
theorem meanOn_scale_left (S : Finset α) (f : α → ℚ) (c : ℚ) :
    meanOn S (fun i => c * f i) = c * meanOn S f := by
  by_cases h : S.card = 0
  · simp [meanOn, h]
  · have hq : (S.card : ℚ) ≠ 0 := by
      exact_mod_cast h
    rw [meanOn, meanOn]
    simp [h]
    rw [← Finset.mul_sum]
    field_simp [hq]

omit [Fintype α] [DecidableEq α] in
theorem meanOn_scale_right (S : Finset α) (f : α → ℚ) (c : ℚ) :
    meanOn S (fun i => f i * c) = meanOn S f * c := by
  by_cases h : S.card = 0
  · simp [meanOn, h]
  · have hq : (S.card : ℚ) ≠ 0 := by
      exact_mod_cast h
    rw [meanOn, meanOn]
    simp [h]
    have hs : Finset.sum S (fun i => f i * c) = (Finset.sum S f) * c := by
      rw [Finset.sum_mul]
    rw [hs]
    field_simp [hq]

omit [Fintype α] [DecidableEq α] in
theorem meanOn_add (S : Finset α) (f g : α → ℚ) :
    meanOn S (fun i => f i + g i) = meanOn S f + meanOn S g := by
  by_cases h : S.card = 0
  · simp [meanOn, h]
  · have hq : (S.card : ℚ) ≠ 0 := by
      exact_mod_cast h
    rw [meanOn, meanOn, meanOn]
    simp [h]
    rw [Finset.sum_add_distrib]
    field_simp [hq]

omit [Fintype α] [DecidableEq α] in
theorem meanOn_neg (S : Finset α) (f : α → ℚ) :
    meanOn S (fun i => -f i) = - meanOn S f := by
  simpa using meanOn_scale_left S f (-1)

omit [Fintype α] [DecidableEq α] in
theorem meanOn_sub (S : Finset α) (f g : α → ℚ) :
    meanOn S (fun i => f i - g i) = meanOn S f - meanOn S g := by
  simp [sub_eq_add_neg, meanOn_add, meanOn_neg]

omit [Fintype α] [DecidableEq α] in
theorem meanOn_const_of_card_ne_zero (S : Finset α) (c : ℚ) (h : S.card ≠ 0) :
    meanOn S (fun _ => c) = c := by
  rw [meanOn]
  simp [h]

omit [Fintype α] [DecidableEq α] in
theorem meanOn_congr (S : Finset α) {f g : α → ℚ}
    (hfg : ∀ i ∈ S, f i = g i) :
    meanOn S f = meanOn S g := by
  by_cases h : S.card = 0
  · simp [meanOn, h]
  · rw [meanOn, meanOn]
    simp [h]
    have hsum : Finset.sum S f = Finset.sum S g := by
      apply Finset.sum_congr rfl
      intro i hi
      exact hfg i hi
    rw [hsum]

omit [Fintype α] [DecidableEq α] in
theorem abs_meanOn_le_meanOn_abs (S : Finset α) (f : α → ℚ) :
    |meanOn S f| ≤ meanOn S (fun i => |f i|) := by
  by_cases hS : S.card = 0
  · simp [meanOn, hS]
  · have hcard : 0 ≤ (S.card : ℚ) := by positivity
    rw [meanOn, meanOn]
    simp [hS]
    rw [abs_div, abs_of_nonneg hcard]
    exact div_le_div_of_nonneg_right (by simpa using Finset.abs_sum_le_sum_abs (s := S) f) hcard

omit [Fintype α] [DecidableEq α] in
theorem meanOn_nonneg (S : Finset α) (f : α → ℚ)
    (hf : ∀ i ∈ S, 0 ≤ f i) :
    0 ≤ meanOn S f := by
  by_cases hS : S.card = 0
  · simp [meanOn, hS]
  · rw [meanOn]
    simp [hS]
    refine div_nonneg ?_ (by positivity)
    exact Finset.sum_nonneg (by intro i hi; exact hf i hi)

omit [Fintype α] [DecidableEq α] in
theorem meanOn_mono (S : Finset α) {f g : α → ℚ}
    (hfg : ∀ i ∈ S, f i ≤ g i) :
    meanOn S f ≤ meanOn S g := by
  by_cases hS : S.card = 0
  · simp [meanOn, hS]
  · have hcard : 0 ≤ (S.card : ℚ) := by positivity
    have hsum : Finset.sum S f ≤ Finset.sum S g := by
      apply Finset.sum_le_sum
      intro i hi
      exact hfg i hi
    rw [meanOn, meanOn]
    simp [hS]
    exact div_le_div_of_nonneg_right hsum hcard

omit [Fintype α] [DecidableEq α] in
theorem meanOn_le_const (S : Finset α) (f : α → ℚ) (c : ℚ)
    (hc : 0 ≤ c)
    (hf : ∀ i ∈ S, f i ≤ c) :
    meanOn S f ≤ c := by
  by_cases hS : S.card = 0
  · simp [meanOn, hS, hc]
  · have hmono : meanOn S f ≤ meanOn S (fun _ => c) := by
      exact meanOn_mono S (fun i hi => hf i hi)
    have hconst : meanOn S (fun _ => c) = c := meanOn_const_of_card_ne_zero S c hS
    exact hmono.trans_eq hconst

omit [Fintype α] [DecidableEq α] in
theorem meanOn_abs_mul_le_half_sum_sq (S : Finset α) (U V : α → ℚ) :
    meanOn S (fun i => |U i * V i|) ≤
      (meanOn S (fun i => U i * U i) + meanOn S (fun i => V i * V i)) / 2 := by
  have hpt :
      ∀ i ∈ S, |U i * V i| ≤ ((U i * U i) + (V i * V i)) / 2 := by
    intro i hi
    have hsq : 0 ≤ (|U i| - |V i|) ^ 2 := sq_nonneg (|U i| - |V i|)
    have htwo : 2 * |U i * V i| ≤ U i * U i + V i * V i := by
      rw [abs_mul]
      nlinarith [hsq, sq_abs (U i), sq_abs (V i)]
    linarith
  have hmono := meanOn_mono S hpt
  calc
    meanOn S (fun i => |U i * V i|) ≤ meanOn S (fun i => ((U i * U i) + (V i * V i)) / 2) := hmono
    _ = (meanOn S (fun i => U i * U i) + meanOn S (fun i => V i * V i)) / 2 := by
      have hscale :
          meanOn S (fun i => ((U i * U i) + (V i * V i)) / 2) =
            meanOn S (fun i => ((U i * U i) + (V i * V i)) * ((1 : ℚ) / 2)) := by
              apply meanOn_congr S
              intro i hi
              ring
      rw [hscale, meanOn_scale_right, meanOn_add]
      ring

omit [Fintype α] [DecidableEq α] in
theorem meanOn_abs_sq_le_meanOn_sq (S : Finset α) (f : α → ℚ) :
    meanOn S (fun i => |f i|) * meanOn S (fun i => |f i|) ≤ meanOn S (fun i => f i * f i) := by
  by_cases hS : S.card = 0
  · simp [meanOn, hS]
  · let g : α → ℚ := fun i => |f i|
    let μ : ℚ := meanOn S g
    have hnonneg : 0 ≤ meanOn S (fun i => (g i - μ) * (g i - μ)) := by
      apply meanOn_nonneg
      intro i hi
      have hs : 0 ≤ (g i - μ) ^ 2 := sq_nonneg (g i - μ)
      nlinarith
    have hexpand :
        meanOn S (fun i => (g i - μ) * (g i - μ)) =
          meanOn S (fun i => g i * g i) - μ * μ := by
      have hrew :
          meanOn S (fun i => (g i - μ) * (g i - μ)) =
            meanOn S (fun i => g i * g i - (2 * μ) * g i + μ * μ) := by
              apply meanOn_congr S
              intro i hi
              ring
      rw [hrew, meanOn_add, meanOn_sub, meanOn_scale_left]
      rw [meanOn_const_of_card_ne_zero S (μ * μ) hS]
      dsimp [μ]
      ring
    have habssq : meanOn S (fun i => g i * g i) = meanOn S (fun i => f i * f i) := by
      apply meanOn_congr S
      intro i hi
      have hi' : |f i| * |f i| = f i * f i := by
        nlinarith [sq_abs (f i)]
      simpa [g] using hi'
    rw [hexpand] at hnonneg
    rw [habssq] at hnonneg
    dsimp [μ, g] at hnonneg
    linarith

omit [Fintype α] [DecidableEq α] in
theorem meanOn_abs_means_mul_le_sum_sq (S : Finset α) (U V : α → ℚ) :
    meanOn S (fun i => |U i|) * meanOn S (fun i => |V i|) ≤
      (meanOn S (fun i => U i * U i) + meanOn S (fun i => V i * V i)) / 2 := by
  let a : ℚ := meanOn S (fun i => |U i|)
  let b : ℚ := meanOn S (fun i => |V i|)
  have hab : 2 * (a * b) ≤ a * a + b * b := by
    have hs : 0 ≤ (a - b) ^ 2 := sq_nonneg (a - b)
    nlinarith
  have ha : a * a ≤ meanOn S (fun i => U i * U i) := by
    simpa [a] using meanOn_abs_sq_le_meanOn_sq S U
  have hb : b * b ≤ meanOn S (fun i => V i * V i) := by
    simpa [b] using meanOn_abs_sq_le_meanOn_sq S V
  nlinarith

omit [Fintype α] [DecidableEq α] in
theorem sq_meanOn_sq_le_meanOn_fourth (S : Finset α) (f : α → ℚ) :
    meanOn S (fun i => f i * f i) * meanOn S (fun i => f i * f i) ≤
      meanOn S (fun i => (f i * f i) * (f i * f i)) := by
  simpa using meanOn_abs_sq_le_meanOn_sq S (fun i => f i * f i)

omit [Fintype α] [DecidableEq α] in
theorem joint_abs_envelope_le_sum_sq (S : Finset α) (U V : α → ℚ) :
    meanOn S (fun i => |U i * V i|) +
      meanOn S (fun i => |U i|) * meanOn S (fun i => |V i|) ≤
        meanOn S (fun i => U i * U i) + meanOn S (fun i => V i * V i) := by
  have h1 := meanOn_abs_mul_le_half_sum_sq S U V
  have h2 := meanOn_abs_means_mul_le_sum_sq S U V
  nlinarith

theorem meanOn_abs_indicator_le_bound
    (S : Finset α) (f : α → ℚ) (P : α → Prop) [DecidablePred P] (M : ℚ)
    (hM : 0 ≤ M)
    (hbound : ∀ i ∈ S, P i → |f i| ≤ M) :
    meanOn S (fun i => |f i| * (if P i then (1 : ℚ) else 0)) ≤
      M * meanOn S (fun i => if P i then (1 : ℚ) else 0) := by
  let ind : α → ℚ := fun i => if P i then (1 : ℚ) else 0
  let g : α → ℚ := fun i => if P i then |f i| else 0
  let h : α → ℚ := fun i => if P i then M else 0
  have hleft :
      meanOn S (fun i => |f i| * (if P i then (1 : ℚ) else 0)) = meanOn S g := by
    apply meanOn_congr S
    intro i hi
    by_cases hPi : P i <;> simp [g, hPi]
  have hmean_le : meanOn S g ≤ meanOn S h := by
    by_cases hS : S.card = 0
    · simp [meanOn, hS, g, h]
    · have hcard : 0 ≤ (S.card : ℚ) := by positivity
      rw [meanOn, meanOn]
      simp [hS]
      refine div_le_div_of_nonneg_right ?_ hcard
      apply Finset.sum_le_sum
      intro i hi
      by_cases hPi : P i
      · simpa [g, h, hPi] using hbound i hi hPi
      · simp [g, h, hPi, hM]
  have hright : meanOn S h = M * meanOn S ind := by
    calc
      meanOn S h = meanOn S (fun i => M * ind i) := by
        apply meanOn_congr S
        intro i hi
        by_cases hPi : P i <;> simp [h, ind, hPi]
      _ = M * meanOn S ind := meanOn_scale_left S ind M
  calc
    meanOn S (fun i => |f i| * (if P i then (1 : ℚ) else 0)) = meanOn S g := hleft
    _ ≤ meanOn S h := hmean_le
    _ = M * meanOn S ind := hright
    _ = M * meanOn S (fun i => if P i then (1 : ℚ) else 0) := by rfl

theorem meanOn_weight_indicator_le_bound
    (S : Finset α) (w : α → ℚ) (P : α → Prop) [DecidablePred P] (M : ℚ)
    (hM : 0 ≤ M)
    (hbound : ∀ i ∈ S, 0 ≤ w i ∧ w i ≤ M) :
    meanOn S (fun i => w i * (if P i then (1 : ℚ) else 0)) ≤
      M * meanOn S (fun i => if P i then (1 : ℚ) else 0) := by
  have habs :
      meanOn S (fun i => |w i| * (if P i then (1 : ℚ) else 0)) ≤
        M * meanOn S (fun i => if P i then (1 : ℚ) else 0) := by
    apply meanOn_abs_indicator_le_bound S w P M hM
    intro i hi hPi
    rw [abs_of_nonneg (hbound i hi).1]
    exact (hbound i hi).2
  have hEq :
      meanOn S (fun i => w i * (if P i then (1 : ℚ) else 0)) =
        meanOn S (fun i => |w i| * (if P i then (1 : ℚ) else 0)) := by
    apply meanOn_congr S
    intro i hi
    have hnonneg : 0 ≤ w i := (hbound i hi).1
    rw [abs_of_nonneg hnonneg]
  rw [hEq]
  exact habs

theorem meanOn_weight_mul_indicator_le_bound
    (S : Finset α) (w f : α → ℚ) (P : α → Prop) [DecidablePred P] (M : ℚ)
    (hM : 0 ≤ M)
    (hf : ∀ i ∈ S, 0 ≤ f i)
    (hbound : ∀ i ∈ S, 0 ≤ w i ∧ w i ≤ M) :
    meanOn S (fun i => w i * f i * (if P i then (1 : ℚ) else 0)) ≤
      M * meanOn S (fun i => f i * (if P i then (1 : ℚ) else 0)) := by
  let g : α → ℚ := fun i => f i * (if P i then (1 : ℚ) else 0)
  have hgnonneg : ∀ i ∈ S, 0 ≤ g i := by
    intro i hi
    by_cases hPi : P i
    · simp [g, hPi, hf i hi]
    · simp [g, hPi]
  have hmono :
      meanOn S (fun i => w i * g i) ≤ meanOn S (fun i => M * g i) := by
    apply meanOn_mono
    intro i hi
    exact mul_le_mul_of_nonneg_right (hbound i hi).2 (hgnonneg i hi)
  have hleft :
      meanOn S (fun i => w i * f i * (if P i then (1 : ℚ) else 0)) =
        meanOn S (fun i => w i * g i) := by
    apply meanOn_congr S
    intro i hi
    by_cases hPi : P i <;> simp [g, hPi]
  have hright :
      meanOn S (fun i => M * g i) =
        M * meanOn S (fun i => f i * (if P i then (1 : ℚ) else 0)) := by
    rw [meanOn_scale_left]
  calc
    meanOn S (fun i => w i * f i * (if P i then (1 : ℚ) else 0))
        = meanOn S (fun i => w i * g i) := hleft
    _ ≤ meanOn S (fun i => M * g i) := hmono
    _ = M * meanOn S (fun i => f i * (if P i then (1 : ℚ) else 0)) := hright

theorem meanOn_weight_mul_indicator_le_weight_indicator_mul_bound
    (S : Finset α) (w f : α → ℚ) (P : α → Prop) [DecidablePred P] (M : ℚ)
    (hM : 0 ≤ M)
    (hw : ∀ i ∈ S, 0 ≤ w i)
    (hbound : ∀ i ∈ S, P i → 0 ≤ f i ∧ f i ≤ M) :
    meanOn S (fun i => w i * f i * (if P i then (1 : ℚ) else 0)) ≤
      M * meanOn S (fun i => w i * (if P i then (1 : ℚ) else 0)) := by
  let g : α → ℚ := fun i => w i * (if P i then (1 : ℚ) else 0)
  have hmono :
      meanOn S (fun i => w i * f i * (if P i then (1 : ℚ) else 0)) ≤
        meanOn S (fun i => M * g i) := by
    apply meanOn_mono
    intro i hi
    by_cases hPi : P i
    · rcases hbound i hi hPi with ⟨hf_nonneg, hf_le⟩
      have hw_nonneg : 0 ≤ w i := hw i hi
      have : w i * f i ≤ w i * M := mul_le_mul_of_nonneg_left hf_le hw_nonneg
      simp [g, hPi]
      linarith
    · simp [hPi, g]
  have hright :
      meanOn S (fun i => M * g i) =
        M * meanOn S (fun i => w i * (if P i then (1 : ℚ) else 0)) := by
    rw [meanOn_scale_left]
  exact hmono.trans_eq hright

theorem meanOn_mul_indicator_le_cutoff_add_tail
    (S : Finset α) (f : α → ℚ) (P : α → Prop) [DecidablePred P] (R : ℚ)
    (hR : 0 ≤ R)
    (hf : ∀ i ∈ S, 0 ≤ f i) :
    meanOn S (fun i => f i * (if P i then (1 : ℚ) else 0)) ≤
      R * meanOn S (fun i => if P i then (1 : ℚ) else 0) +
        meanOn S (fun i => f i * (if P i ∧ R < f i then (1 : ℚ) else 0)) := by
  let ind : α → ℚ := fun i => if P i then (1 : ℚ) else 0
  let tail : α → ℚ := fun i => if P i ∧ R < f i then (1 : ℚ) else 0
  have hmono :
      meanOn S (fun i => f i * ind i) ≤ meanOn S (fun i => R * ind i + f i * tail i) := by
    apply meanOn_mono
    intro i hi
    by_cases hPi : P i
    · by_cases htail : R < f i
      · simp [ind, tail, hPi, htail]
        nlinarith [hf i hi, hR]
      · have hle : f i ≤ R := not_lt.mp htail
        simp [ind, tail, hPi, htail]
        nlinarith [hf i hi, hR, hle]
    · simp [ind, tail, hPi]
  have hsplit :
      meanOn S (fun i => R * ind i + f i * tail i) =
        R * meanOn S (fun i => if P i then (1 : ℚ) else 0) +
          meanOn S (fun i => f i * (if P i ∧ R < f i then (1 : ℚ) else 0)) := by
    calc
      meanOn S (fun i => R * ind i + f i * tail i)
          = meanOn S (fun i => R * ind i) + meanOn S (fun i => f i * tail i) := by
              rw [meanOn_add]
      _ = R * meanOn S ind + meanOn S (fun i => f i * tail i) := by
            rw [meanOn_scale_left]
      _ = R * meanOn S (fun i => if P i then (1 : ℚ) else 0) +
            meanOn S (fun i => f i * (if P i ∧ R < f i then (1 : ℚ) else 0)) := by
              rfl
  calc
    meanOn S (fun i => f i * (if P i then (1 : ℚ) else 0))
        = meanOn S (fun i => f i * ind i) := by rfl
    _ ≤ meanOn S (fun i => R * ind i + f i * tail i) := hmono
    _ = R * meanOn S (fun i => if P i then (1 : ℚ) else 0) +
          meanOn S (fun i => f i * (if P i ∧ R < f i then (1 : ℚ) else 0)) := hsplit

theorem meanOn_fourth_sum_indicator_le_bound
    (S : Finset α) (U V : α → ℚ) (P : α → Prop) [DecidablePred P] (M : ℚ)
    (hM : 0 ≤ M)
    (hbound : ∀ i ∈ S, P i →
      0 ≤ ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)) ∧
      ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)) ≤ M) :
    meanOn S
        (fun i =>
          (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
            (if P i then (1 : ℚ) else 0)) ≤
      M * meanOn S (fun i => if P i then (1 : ℚ) else 0) := by
  simpa using
    (meanOn_weight_mul_indicator_le_weight_indicator_mul_bound
      (S := S)
      (w := fun _ => (1 : ℚ))
      (f := fun i => ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)))
      (P := P)
      (M := M)
      (hM := hM)
      (hw := by
        intro i hi
        norm_num)
      (hbound := by
        intro i hi hPi
        exact hbound i hi hPi))

theorem meanOn_fourth_sum_bulk_complement_le_bound
    (slack : α → ℚ) (ζ T : ℚ) (U V : α → ℚ) (M : ℚ)
    (hM : 0 ≤ M)
    (hbound : ∀ i,
      ¬ (ζ * T ≤ slack i) →
        0 ≤ ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)) ∧
        ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)) ≤ M) :
    meanOn Finset.univ
        (fun i =>
          (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
            (if ¬ (ζ * T ≤ slack i) then (1 : ℚ) else 0)) ≤
      M * meanOn Finset.univ (fun i => if ¬ (ζ * T ≤ slack i) then (1 : ℚ) else 0) := by
  simpa using
    (meanOn_fourth_sum_indicator_le_bound
      (S := Finset.univ)
      (U := U)
      (V := V)
      (P := fun i => ¬ (ζ * T ≤ slack i))
      (M := M)
      (hM := hM)
      (hbound := by
        intro i hi hPi
        exact hbound i hPi))

theorem meanOn_fourth_sum_bulk_complement_le_cutoff_add_tail
    (slack : α → ℚ) (ζ T : ℚ) (U V : α → ℚ) (R : ℚ)
    (hR : 0 ≤ R) :
    meanOn Finset.univ
        (fun i =>
          (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
            (if ¬ (ζ * T ≤ slack i) then (1 : ℚ) else 0)) ≤
      R * meanOn Finset.univ (fun i => if ¬ (ζ * T ≤ slack i) then (1 : ℚ) else 0) +
        meanOn Finset.univ
          (fun i =>
            (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
              (if ¬ (ζ * T ≤ slack i) ∧
                    R < ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))
               then (1 : ℚ) else 0)) := by
  simpa using
    (meanOn_mul_indicator_le_cutoff_add_tail
      (S := Finset.univ)
      (f := fun i => ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)))
      (P := fun i => ¬ (ζ * T ≤ slack i))
      (R := R)
      (hR := hR)
      (hf := by
        intro i hi
        have hU : 0 ≤ (U i * U i) * (U i * U i) := by
          nlinarith [sq_nonneg (U i * U i)]
        have hV : 0 ≤ (V i * V i) * (V i * V i) := by
          nlinarith [sq_nonneg (V i * V i)]
        linarith))

theorem meanOn_fourth_sum_bulk_complement_le_two_mul_fourth_bound
    (slack : α → ℚ) (ζ T : ℚ) (U V : α → ℚ) (B : ℚ)
    (hB : 0 ≤ B)
    (hU : ∀ i, ¬ (ζ * T ≤ slack i) → |U i| ≤ B)
    (hV : ∀ i, ¬ (ζ * T ≤ slack i) → |V i| ≤ B) :
    meanOn Finset.univ
        (fun i =>
          (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
            (if ¬ (ζ * T ≤ slack i) then (1 : ℚ) else 0)) ≤
      (2 * ((B * B) * (B * B))) *
        meanOn Finset.univ (fun i => if ¬ (ζ * T ≤ slack i) then (1 : ℚ) else 0) := by
  apply meanOn_fourth_sum_bulk_complement_le_bound
    (slack := slack) (ζ := ζ) (T := T) (U := U) (V := V)
    (M := 2 * ((B * B) * (B * B)))
  · nlinarith
  · intro i hi
    have hUabs : |U i| ≤ B := hU i hi
    have hVabs : |V i| ≤ B := hV i hi
    have hBBnonneg : 0 ≤ B * B := by nlinarith
    have hU2nonneg : 0 ≤ U i * U i := by nlinarith [sq_nonneg (U i)]
    have hV2nonneg : 0 ≤ V i * V i := by nlinarith [sq_nonneg (V i)]
    have hU2 : U i * U i ≤ B * B := by
      have hs : |U i| * |U i| ≤ B * B := by
        exact mul_le_mul hUabs hUabs (abs_nonneg _) hB
      simpa [sq_abs] using hs
    have hV2 : V i * V i ≤ B * B := by
      have hs : |V i| * |V i| ≤ B * B := by
        exact mul_le_mul hVabs hVabs (abs_nonneg _) hB
      simpa [sq_abs] using hs
    have hU4 : (U i * U i) * (U i * U i) ≤ (B * B) * (B * B) := by
      exact mul_le_mul hU2 hU2 hU2nonneg hBBnonneg
    have hV4 : (V i * V i) * (V i * V i) ≤ (B * B) * (B * B) := by
      exact mul_le_mul hV2 hV2 hV2nonneg hBBnonneg
    refine ⟨by nlinarith, ?_⟩
    nlinarith

theorem binWeightOn_nonneg (S : Finset α) (bin : α → β) (b : β) :
    0 ≤ binWeightOn S bin b := by
  by_cases hS : S.card = 0
  · simp [binWeightOn, hS]
  · unfold binWeightOn
    simp [hS]
    positivity

theorem binWeightOn_le_one (S : Finset α) (bin : α → β) (b : β) :
    binWeightOn S bin b ≤ 1 := by
  by_cases hS : S.card = 0
  · simp [binWeightOn, hS]
  · unfold binWeightOn
    simp [hS]
    have hcardNat : (fiberIn S bin b).card ≤ S.card := by
      simpa [fiberIn] using (Finset.card_filter_le (s := S) (p := fun i => bin i = b))
    have hcard : ((fiberIn S bin b).card : ℚ) ≤ (S.card : ℚ) := by
      exact_mod_cast hcardNat
    have hdenom : 0 < (S.card : ℚ) := by
      exact_mod_cast Nat.pos_of_ne_zero hS
    have hdiv :
        ((fiberIn S bin b).card : ℚ) / (S.card : ℚ) ≤ (S.card : ℚ) / (S.card : ℚ) := by
      exact div_le_div_of_nonneg_right hcard (le_of_lt hdenom)
    have hone : (S.card : ℚ) / (S.card : ℚ) = 1 := by
      field_simp [show (S.card : ℚ) ≠ 0 by exact_mod_cast hS]
    exact hdiv.trans_eq hone

theorem pushforwardWeightOn_nonneg (S : Finset α) (coord : α → β) (y : β) :
    0 ≤ pushforwardWeightOn S coord y := by
  simpa [pushforwardWeightOn] using binWeightOn_nonneg S coord y

theorem pushforwardWeightOn_le_one (S : Finset α) (coord : α → β) (y : β) :
    pushforwardWeightOn S coord y ≤ 1 := by
  simpa [pushforwardWeightOn] using binWeightOn_le_one S coord y

theorem meanOn_pushforwardWeight_indicator_le_indicator
    (S : Finset α) (coord : α → β) (P : α → Prop) [DecidablePred P] :
    meanOn Finset.univ
        (fun i => pushforwardWeightOn S coord (coord i) * (if P i then (1 : ℚ) else 0)) ≤
      meanOn Finset.univ (fun i => if P i then (1 : ℚ) else 0) := by
  simpa using
    (meanOn_weight_indicator_le_bound
      (S := Finset.univ)
      (w := fun i => pushforwardWeightOn S coord (coord i))
      (P := P)
      (M := (1 : ℚ))
      (hM := by norm_num)
      (hbound := by
        intro i hi
        exact ⟨pushforwardWeightOn_nonneg S coord (coord i), pushforwardWeightOn_le_one S coord (coord i)⟩))

theorem meanOn_pushforwardWeight_mul_indicator_le_indicator_mul
    (S : Finset α) (coord : α → β) (f : α → ℚ) (P : α → Prop) [DecidablePred P]
    (hf : ∀ i, 0 ≤ f i) :
    meanOn Finset.univ
        (fun i => pushforwardWeightOn S coord (coord i) * f i * (if P i then (1 : ℚ) else 0)) ≤
      meanOn Finset.univ (fun i => f i * (if P i then (1 : ℚ) else 0)) := by
  simpa using
    (meanOn_weight_mul_indicator_le_bound
      (S := Finset.univ)
      (w := fun i => pushforwardWeightOn S coord (coord i))
      (f := f)
      (P := P)
      (M := (1 : ℚ))
      (hM := by norm_num)
      (hf := by
        intro i hi
        exact hf i)
      (hbound := by
        intro i hi
        exact ⟨pushforwardWeightOn_nonneg S coord (coord i), pushforwardWeightOn_le_one S coord (coord i)⟩))

theorem meanOn_pushforwardWeight_fourth_indicator_le_fourth_indicator
    (S : Finset α) (coord : α → β) (U V : α → ℚ) (P : α → Prop) [DecidablePred P] :
    meanOn Finset.univ
        (fun i =>
          pushforwardWeightOn S coord (coord i) *
            (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
              (if P i then (1 : ℚ) else 0)) ≤
      meanOn Finset.univ
        (fun i =>
          (((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i))) *
            (if P i then (1 : ℚ) else 0)) := by
  simpa using
    (meanOn_pushforwardWeight_mul_indicator_le_indicator_mul
      (S := S)
      (coord := coord)
      (f := fun i => ((U i * U i) * (U i * U i)) + ((V i * V i) * (V i * V i)))
      (P := P)
      (hf := by
        intro i
        have hU : 0 ≤ (U i * U i) * (U i * U i) := by
          nlinarith [sq_nonneg (U i * U i)]
        have hV : 0 ≤ (V i * V i) * (V i * V i) := by
          nlinarith [sq_nonneg (V i * V i)]
        linarith))

omit [Fintype α] [DecidableEq α] in
theorem covOn_eq_mean_prod_sub_means (S : Finset α) (U V : α → ℚ) :
    covOn S U V = meanOn S (fun i => U i * V i) - meanOn S U * meanOn S V := by
  by_cases h : S.card = 0
  · simp [covOn, meanOn, h]
  · let μU : ℚ := meanOn S U
    let μV : ℚ := meanOn S V
    have hexpand :
        (fun i => (U i - μU) * (V i - μV)) =
        (fun i => U i * V i - μV * U i - μU * V i + μU * μV) := by
      funext i
      ring
    rw [covOn, hexpand, meanOn_add, meanOn_sub, meanOn_sub]
    rw [meanOn_scale_left, meanOn_scale_left]
    have hconst : meanOn S (fun _ => μU * μV) = μU * μV := by
      exact meanOn_const_of_card_ne_zero S (μU * μV) h
    rw [hconst]
    dsimp [μU, μV]
    ring

omit [Fintype α] [DecidableEq α] in
theorem meanOn_sq_dev_eq_meanOn_sq_sub_sq_meanOn (S : Finset α) (f : α → ℚ) :
    meanOn S (fun i => (f i - meanOn S f) * (f i - meanOn S f)) =
      meanOn S (fun i => f i * f i) - meanOn S f * meanOn S f := by
  simpa using covOn_eq_mean_prod_sub_means S f f

omit [Fintype α] [DecidableEq α] in
theorem covOn_congr (S : Finset α) {U U' V V' : α → ℚ}
    (hU : ∀ i ∈ S, U i = U' i) (hV : ∀ i ∈ S, V i = V' i) :
    covOn S U V = covOn S U' V' := by
  unfold covOn
  have hμU : meanOn S U = meanOn S U' := meanOn_congr S hU
  have hμV : meanOn S V = meanOn S V' := meanOn_congr S hV
  apply meanOn_congr S
  intro i hi
  rw [hU i hi, hV i hi, hμU, hμV]

omit [Fintype α] [DecidableEq α] in
theorem covOn_const_left (S : Finset α) (c : ℚ) (V : α → ℚ) :
    covOn S (fun _ => c) V = 0 := by
  by_cases h : S.card = 0
  · simp [covOn, meanOn, h]
  · rw [covOn_eq_mean_prod_sub_means]
    have hconst : meanOn S (fun _ => c) = c := meanOn_const_of_card_ne_zero S c h
    have hprod : meanOn S (fun i => c * V i) = c * meanOn S V := by
      simpa using meanOn_scale_left S V c
    rw [hconst, hprod]
    ring

omit [Fintype α] [DecidableEq α] in
theorem covOn_const_right (S : Finset α) (U : α → ℚ) (c : ℚ) :
    covOn S U (fun _ => c) = 0 := by
  by_cases h : S.card = 0
  · simp [covOn, meanOn, h]
  · rw [covOn_eq_mean_prod_sub_means]
    have hconst : meanOn S (fun _ => c) = c := meanOn_const_of_card_ne_zero S c h
    have hprod : meanOn S (fun i => U i * c) = meanOn S U * c := by
      simpa using meanOn_scale_right S U c
    rw [hconst, hprod]
    ring

omit [Fintype α] [DecidableEq α] in
theorem covOn_scale_left (S : Finset α) (U V : α → ℚ) (c : ℚ) :
    covOn S (fun i => c * U i) V = c * covOn S U V := by
  unfold covOn
  have hfun :
      (fun i => ((fun i => c * U i) i - meanOn S (fun i => c * U i)) * (V i - meanOn S V)) =
      (fun i => c * ((U i - meanOn S U) * (V i - meanOn S V))) := by
    funext i
    rw [meanOn_scale_left]
    ring
  rw [hfun, meanOn_scale_left]

omit [Fintype α] [DecidableEq α] in
theorem covOn_scale_right (S : Finset α) (U V : α → ℚ) (c : ℚ) :
    covOn S U (fun i => V i * c) = covOn S U V * c := by
  unfold covOn
  have hfun :
      (fun i => (U i - meanOn S U) * ((fun i => V i * c) i - meanOn S (fun i => V i * c))) =
      (fun i => ((U i - meanOn S U) * (V i - meanOn S V)) * c) := by
    funext i
    rw [meanOn_scale_right]
    ring
  rw [hfun, meanOn_scale_right]

omit [Fintype α] [DecidableEq α] in
theorem covOn_add_left (S : Finset α) (U₁ U₂ V : α → ℚ) :
    covOn S (fun i => U₁ i + U₂ i) V = covOn S U₁ V + covOn S U₂ V := by
  rw [covOn_eq_mean_prod_sub_means, covOn_eq_mean_prod_sub_means, covOn_eq_mean_prod_sub_means]
  have hprod :
      meanOn S (fun i => (U₁ i + U₂ i) * V i) =
      meanOn S (fun i => U₁ i * V i) + meanOn S (fun i => U₂ i * V i) := by
    have hfun :
        (fun i => (U₁ i + U₂ i) * V i) =
        (fun i => U₁ i * V i + U₂ i * V i) := by
      funext i
      ring
    rw [hfun, meanOn_add]
  rw [hprod, meanOn_add]
  ring

omit [Fintype α] [DecidableEq α] in
theorem covOn_add_right (S : Finset α) (U V₁ V₂ : α → ℚ) :
    covOn S U (fun i => V₁ i + V₂ i) = covOn S U V₁ + covOn S U V₂ := by
  rw [covOn_eq_mean_prod_sub_means, covOn_eq_mean_prod_sub_means, covOn_eq_mean_prod_sub_means]
  have hprod :
      meanOn S (fun i => U i * (V₁ i + V₂ i)) =
      meanOn S (fun i => U i * V₁ i) + meanOn S (fun i => U i * V₂ i) := by
    have hfun :
        (fun i => U i * (V₁ i + V₂ i)) =
        (fun i => U i * V₁ i + U i * V₂ i) := by
      funext i
      ring
    rw [hfun, meanOn_add]
  rw [hprod, meanOn_add]
  ring

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_scale_left (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    paperEstimator bin (fun i => c * U i) V = c * paperEstimator bin U V := by
  unfold paperEstimator
  simp [covOn_scale_left, mul_left_comm]
  rw [← Finset.mul_sum]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_scale_right (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    paperEstimator bin U (fun i => V i * c) = paperEstimator bin U V * c := by
  unfold paperEstimator
  calc
    Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U (fun i => V i * c))
        = Finset.sum (Finset.univ.image bin) (fun b => (binWeight bin b * covOn (fiber bin b) U V) * c) := by
            apply Finset.sum_congr rfl
            intro b hb
            rw [covOn_scale_right]
            ring
    _ = (Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U V)) * c := by
          rw [Finset.sum_mul]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_add_left (bin : α → β) (U₁ U₂ V : α → ℚ) :
    paperEstimator bin (fun i => U₁ i + U₂ i) V =
    paperEstimator bin U₁ V + paperEstimator bin U₂ V := by
  unfold paperEstimator
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b * covOn (fiber bin b) (fun i => U₁ i + U₂ i) V)
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * covOn (fiber bin b) U₁ V +
            binWeight bin b * covOn (fiber bin b) U₂ V) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [covOn_add_left]
              ring
    _ =
        Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U₁ V) +
        Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U₂ V) := by
          rw [Finset.sum_add_distrib]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_add_right (bin : α → β) (U V₁ V₂ : α → ℚ) :
    paperEstimator bin U (fun i => V₁ i + V₂ i) =
    paperEstimator bin U V₁ + paperEstimator bin U V₂ := by
  unfold paperEstimator
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b * covOn (fiber bin b) U (fun i => V₁ i + V₂ i))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * covOn (fiber bin b) U V₁ +
            binWeight bin b * covOn (fiber bin b) U V₂) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [covOn_add_right]
              ring
    _ =
        Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U V₁) +
        Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U V₂) := by
          rw [Finset.sum_add_distrib]

omit [DecidableEq α] [Fintype β] in
theorem fiber_of_constant_bin (bin : α → β) (b0 : β) (hbin : ∀ i, bin i = b0) :
    fiber bin b0 = Finset.univ := by
  ext i
  simp [fiber, hbin i]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_constant_bin [Nonempty α] (bin : α → β) (b0 : β) (U V : α → ℚ)
    (hbin : ∀ i, bin i = b0) :
    paperEstimator bin U V = covOn Finset.univ U V := by
  have himage : Finset.univ.image bin = ({b0} : Finset β) := by
    ext b
    simp only [Finset.mem_image, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro hb
      rcases hb with ⟨i, hi⟩
      simpa [hbin i] using hi.symm
    · intro hb
      subst hb
      let i0 : α := Classical.choice ‹Nonempty α›
      exact ⟨i0, hbin i0⟩
  unfold paperEstimator
  rw [himage]
  simp [binWeight, fiber_of_constant_bin, hbin]

omit [DecidableEq α] [Fintype β] in
theorem covOn_on_fiber_of_factorsThroughBin_left (bin : α → β) (U V : α → ℚ) (b : β)
    (hU : FactorsThroughBin bin U) :
    covOn (fiber bin b) U V = 0 := by
  rcases hU with ⟨g, hg⟩
  apply Eq.trans ?_ (covOn_const_left (fiber bin b) (g b) V)
  apply covOn_congr (fiber bin b)
  · intro i hi
    have hmem : bin i = b := by
      simp [fiber] at hi
      exact hi
    rw [hg i, hmem]
  · intro i hi
    rfl

omit [DecidableEq α] [Fintype β] in
theorem covOn_on_fiber_of_factorsThroughBin_right (bin : α → β) (U V : α → ℚ) (b : β)
    (hV : FactorsThroughBin bin V) :
    covOn (fiber bin b) U V = 0 := by
  rcases hV with ⟨g, hg⟩
  apply Eq.trans ?_ (covOn_const_right (fiber bin b) U (g b))
  apply covOn_congr (fiber bin b)
  · intro i hi
    rfl
  · intro i hi
    have hmem : bin i = b := by
      simp [fiber] at hi
      exact hi
    rw [hg i, hmem]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_zero_of_factorsThroughBin_left (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    paperEstimator bin U V = 0 := by
  unfold paperEstimator
  apply Finset.sum_eq_zero
  intro b hb
  rw [covOn_on_fiber_of_factorsThroughBin_left bin U V b hU]
  simp

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_zero_of_factorsThroughBin_right (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    paperEstimator bin U V = 0 := by
  unfold paperEstimator
  apply Finset.sum_eq_zero
  intro b hb
  rw [covOn_on_fiber_of_factorsThroughBin_right bin U V b hV]
  simp

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_congr (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    paperEstimator bin U V = paperEstimator bin U' V' := by
  unfold paperEstimator
  apply Finset.sum_congr rfl
  intro b hb
  congr 1
  apply covOn_congr (fiber bin b)
  · intro i hi
    exact hU i
  · intro i hi
    exact hV i

omit [DecidableEq α] [Fintype β] in
theorem paperEstimator_eq_sum_mean_prod_sub_sum_mean_means (bin : α → β) (U V : α → ℚ) :
    paperEstimator bin U V = productTerm bin U V - meanProductTerm bin U V := by
  unfold paperEstimator
  unfold productTerm meanProductTerm
  calc
    Finset.sum (Finset.univ.image bin) (fun b => binWeight bin b * covOn (fiber bin b) U V)
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b *
            (meanOn (fiber bin b) (fun i => U i * V i) -
             meanOn (fiber bin b) U * meanOn (fiber bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [covOn_eq_mean_prod_sub_means]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i) -
            binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i)) -
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V)) := by
            rw [Finset.sum_sub_distrib]

theorem fiberIn_univ (bin : α → β) (b : β) :
    fiberIn Finset.univ bin b = fiber bin b := by
  ext i
  simp [fiberIn, fiber]

theorem meanProdChannelOn_univ (bin : α → β) (U V : α → ℚ) :
    meanProdChannelOn Finset.univ bin U V =
      fun b => meanOn (fiber bin b) (fun i => U i * V i) := by
  funext b
  simp [meanProdChannelOn, fiberIn_univ]

theorem meanMeansChannelOn_univ (bin : α → β) (U V : α → ℚ) :
    meanMeansChannelOn Finset.univ bin U V =
      fun b => meanOn (fiber bin b) U * meanOn (fiber bin b) V := by
  funext b
  simp [meanMeansChannelOn, fiberIn_univ]

theorem covChannelOn_univ (bin : α → β) (U V : α → ℚ) :
    covChannelOn Finset.univ bin U V =
      fun b => covOn (fiber bin b) U V := by
  funext b
  simp [covChannelOn, fiberIn_univ]

theorem meanProdChannelOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    meanProdChannelOn (supportOf P) bin U V =
      fun b => meanOn (fiber bin b) (fun i => U i * V i) := by
  rw [supportOf_true P hP]
  exact meanProdChannelOn_univ bin U V

theorem meanProdChannelOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    meanProdChannelOn (supportOf P) bin U V = 0 := by
  funext b
  rw [supportOf_false P hP]
  simp [meanProdChannelOn, fiberIn]

theorem meanMeansChannelOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    meanMeansChannelOn (supportOf P) bin U V =
      fun b => meanOn (fiber bin b) U * meanOn (fiber bin b) V := by
  rw [supportOf_true P hP]
  exact meanMeansChannelOn_univ bin U V

theorem meanMeansChannelOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    meanMeansChannelOn (supportOf P) bin U V = 0 := by
  funext b
  rw [supportOf_false P hP]
  simp [meanMeansChannelOn, fiberIn]

theorem covChannelOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    covChannelOn (supportOf P) bin U V =
      fun b => covOn (fiber bin b) U V := by
  rw [supportOf_true P hP]
  exact covChannelOn_univ bin U V

theorem covChannelOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    covChannelOn (supportOf P) bin U V = 0 := by
  funext b
  rw [supportOf_false P hP]
  simp [covChannelOn, covOn, fiberIn, meanOn]

theorem binWeightOn_univ (bin : α → β) (b : β) :
    binWeightOn Finset.univ bin b = binWeight bin b := by
  by_cases hα : Fintype.card α = 0
  · unfold binWeightOn binWeight
    simp [hα, fiberIn_univ]
  · unfold binWeightOn binWeight
    simp [hα, fiberIn_univ]

theorem paperEstimatorOn_univ (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn Finset.univ bin U V = paperEstimator bin U V := by
  unfold paperEstimatorOn paperEstimator
  simp [binWeightOn_univ, fiberIn_univ]

theorem productTermOn_univ (bin : α → β) (U V : α → ℚ) :
    productTermOn Finset.univ bin U V = productTerm bin U V := by
  unfold productTermOn productTerm
  simp [binWeightOn_univ, fiberIn_univ]

theorem meanProductTermOn_univ (bin : α → β) (U V : α → ℚ) :
    meanProductTermOn Finset.univ bin U V = meanProductTerm bin U V := by
  unfold meanProductTermOn meanProductTerm
  simp [binWeightOn_univ, fiberIn_univ]

@[simp] theorem productTermOn_empty (bin : α → β) (U V : α → ℚ) :
    productTermOn (∅ : Finset α) bin U V = 0 := by
  simp [productTermOn]

@[simp] theorem meanProductTermOn_empty (bin : α → β) (U V : α → ℚ) :
    meanProductTermOn (∅ : Finset α) bin U V = 0 := by
  simp [meanProductTermOn]

theorem productTermOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    productTermOn (supportOf P) bin U V = productTerm bin U V := by
  rw [supportOf_true P hP, productTermOn_univ]

theorem productTermOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    productTermOn (supportOf P) bin U V = 0 := by
  rw [supportOf_false P hP, productTermOn_empty]

theorem meanProductTermOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    meanProductTermOn (supportOf P) bin U V = meanProductTerm bin U V := by
  rw [supportOf_true P hP, meanProductTermOn_univ]

theorem meanProductTermOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    meanProductTermOn (supportOf P) bin U V = 0 := by
  rw [supportOf_false P hP, meanProductTermOn_empty]

theorem paperEstimatorOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    paperEstimatorOn (supportOf P) bin U V = paperEstimator bin U V := by
  rw [supportOf_true P hP, paperEstimatorOn_univ]

theorem paperEstimatorOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    paperEstimatorOn (supportOf P) bin U V = 0 := by
  rw [supportOf_false P hP, paperEstimatorOn_empty]

theorem paperEstimatorOn_bulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    paperEstimatorOn (bulkSupportSlack slack ζ T) bin U V = paperEstimator bin U V := by
  unfold bulkSupportSlack
  exact paperEstimatorOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem paperEstimatorOn_bulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    paperEstimatorOn (bulkSupportSlack slack ζ T) bin U V = 0 := by
  unfold bulkSupportSlack
  exact paperEstimatorOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem productTermOn_bulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    productTermOn (bulkSupportSlack slack ζ T) bin U V = productTerm bin U V := by
  unfold bulkSupportSlack
  exact productTermOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem productTermOn_bulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    productTermOn (bulkSupportSlack slack ζ T) bin U V = 0 := by
  unfold bulkSupportSlack
  exact productTermOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem meanProductTermOn_bulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    meanProductTermOn (bulkSupportSlack slack ζ T) bin U V = meanProductTerm bin U V := by
  unfold bulkSupportSlack
  exact meanProductTermOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem meanProductTermOn_bulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    meanProductTermOn (bulkSupportSlack slack ζ T) bin U V = 0 := by
  unfold bulkSupportSlack
  exact meanProductTermOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem productTermBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    productTermBulk slack ζ T bin U V = productTerm bin U V := by
  unfold productTermBulk
  exact productTermOn_bulk_true slack ζ T bin U V hbulk

theorem productTermBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    productTermBulk slack ζ T bin U V = 0 := by
  unfold productTermBulk
  exact productTermOn_bulk_false slack ζ T bin U V hbulk

theorem meanProductTermBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    meanProductTermBulk slack ζ T bin U V = meanProductTerm bin U V := by
  unfold meanProductTermBulk
  exact meanProductTermOn_bulk_true slack ζ T bin U V hbulk

theorem meanProductTermBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    meanProductTermBulk slack ζ T bin U V = 0 := by
  unfold meanProductTermBulk
  exact meanProductTermOn_bulk_false slack ζ T bin U V hbulk

theorem paperEstimatorBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    paperEstimatorBulk slack ζ T bin U V = paperEstimator bin U V := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_bulk_true slack ζ T bin U V hbulk

theorem paperEstimatorBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    paperEstimatorBulk slack ζ T bin U V = 0 := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_bulk_false slack ζ T bin U V hbulk

theorem meanProdChannelBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    meanProdChannelBulk slack ζ T bin U V =
      fun b => meanOn (fiber bin b) (fun i => U i * V i) := by
  unfold meanProdChannelBulk bulkSupportSlack
  exact meanProdChannelOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem meanProdChannelBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    meanProdChannelBulk slack ζ T bin U V = 0 := by
  unfold meanProdChannelBulk bulkSupportSlack
  exact meanProdChannelOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem meanMeansChannelBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    meanMeansChannelBulk slack ζ T bin U V =
      fun b => meanOn (fiber bin b) U * meanOn (fiber bin b) V := by
  unfold meanMeansChannelBulk bulkSupportSlack
  exact meanMeansChannelOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem meanMeansChannelBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    meanMeansChannelBulk slack ζ T bin U V = 0 := by
  unfold meanMeansChannelBulk bulkSupportSlack
  exact meanMeansChannelOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem covChannelBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    covChannelBulk slack ζ T bin U V =
      fun b => covOn (fiber bin b) U V := by
  unfold covChannelBulk bulkSupportSlack
  exact covChannelOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem covChannelBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    covChannelBulk slack ζ T bin U V = 0 := by
  unfold covChannelBulk bulkSupportSlack
  exact covChannelOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

@[simp] theorem deltaProductTermOn_univ (bin : α → β) (U V : α → ℚ) :
    deltaProductTermOn Finset.univ bin U V = 0 := by
  unfold deltaProductTermOn
  rw [productTermOn_univ]
  ring

@[simp] theorem deltaMeanProductTermOn_univ (bin : α → β) (U V : α → ℚ) :
    deltaMeanProductTermOn Finset.univ bin U V = 0 := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_univ]
  ring

theorem deltaProductTermOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    deltaProductTermOn (supportOf P) bin U V = 0 := by
  unfold deltaProductTermOn
  rw [productTermOn_support_true P bin U V hP]
  ring

theorem deltaProductTermOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    deltaProductTermOn (supportOf P) bin U V = - productTerm bin U V := by
  unfold deltaProductTermOn
  rw [productTermOn_support_false P bin U V hP]
  ring

theorem deltaMeanProductTermOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    deltaMeanProductTermOn (supportOf P) bin U V = 0 := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_support_true P bin U V hP]
  ring

theorem deltaMeanProductTermOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    deltaMeanProductTermOn (supportOf P) bin U V = - meanProductTerm bin U V := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_support_false P bin U V hP]
  ring

theorem deltaProductTermOn_bulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    deltaProductTermOn (bulkSupportSlack slack ζ T) bin U V = 0 := by
  unfold bulkSupportSlack
  exact deltaProductTermOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem deltaProductTermOn_bulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    deltaProductTermOn (bulkSupportSlack slack ζ T) bin U V = - productTerm bin U V := by
  unfold bulkSupportSlack
  exact deltaProductTermOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem deltaMeanProductTermOn_bulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    deltaMeanProductTermOn (bulkSupportSlack slack ζ T) bin U V = 0 := by
  unfold bulkSupportSlack
  exact deltaMeanProductTermOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem deltaMeanProductTermOn_bulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    deltaMeanProductTermOn (bulkSupportSlack slack ζ T) bin U V = - meanProductTerm bin U V := by
  unfold bulkSupportSlack
  exact deltaMeanProductTermOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

@[simp] theorem deltaPaperEstimatorOn_univ (bin : α → β) (U V : α → ℚ) :
    deltaPaperEstimatorOn Finset.univ bin U V = 0 := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_univ]
  ring

theorem deltaPaperEstimatorOn_support_true (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, P i) :
    deltaPaperEstimatorOn (supportOf P) bin U V = 0 := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_support_true P bin U V hP]
  ring

theorem deltaPaperEstimatorOn_support_false (P : α → Prop) [DecidablePred P]
    (bin : α → β) (U V : α → ℚ) (hP : ∀ i, ¬ P i) :
    deltaPaperEstimatorOn (supportOf P) bin U V = - paperEstimator bin U V := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_support_false P bin U V hP]
  ring

theorem deltaPaperEstimatorOn_bulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    deltaPaperEstimatorOn (bulkSupportSlack slack ζ T) bin U V = 0 := by
  unfold bulkSupportSlack
  exact deltaPaperEstimatorOn_support_true (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem deltaPaperEstimatorOn_bulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    deltaPaperEstimatorOn (bulkSupportSlack slack ζ T) bin U V = - paperEstimator bin U V := by
  unfold bulkSupportSlack
  exact deltaPaperEstimatorOn_support_false (fun i => ζ * T ≤ slack i) bin U V hbulk

theorem deltaProductTermBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    deltaProductTermBulk slack ζ T bin U V = 0 := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_bulk_true slack ζ T bin U V hbulk

theorem deltaProductTermBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    deltaProductTermBulk slack ζ T bin U V = - productTerm bin U V := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_bulk_false slack ζ T bin U V hbulk

theorem deltaMeanProductTermBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    deltaMeanProductTermBulk slack ζ T bin U V = 0 := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_bulk_true slack ζ T bin U V hbulk

theorem deltaMeanProductTermBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    deltaMeanProductTermBulk slack ζ T bin U V = - meanProductTerm bin U V := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_bulk_false slack ζ T bin U V hbulk

theorem deltaPaperEstimatorBulk_true (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ζ * T ≤ slack i) :
    deltaPaperEstimatorBulk slack ζ T bin U V = 0 := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_bulk_true slack ζ T bin U V hbulk

theorem deltaPaperEstimatorBulk_false (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hbulk : ∀ i, ¬ (ζ * T ≤ slack i)) :
    deltaPaperEstimatorBulk slack ζ T bin U V = - paperEstimator bin U V := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_bulk_false slack ζ T bin U V hbulk

theorem productTerm_congr (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    productTerm bin U V = productTerm bin U' V' := by
  unfold productTerm
  apply Finset.sum_congr rfl
  intro b hb
  congr 1
  apply meanOn_congr (fiber bin b)
  intro i hi
  rw [hU i, hV i]

theorem meanProductTerm_congr (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanProductTerm bin U V = meanProductTerm bin U' V' := by
  unfold meanProductTerm
  apply Finset.sum_congr rfl
  intro b hb
  rw [meanOn_congr (fiber bin b) (fun i _ => hU i)]
  rw [meanOn_congr (fiber bin b) (fun i _ => hV i)]

theorem productTerm_scale_left (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    productTerm bin (fun i => c * U i) V = c * productTerm bin U V := by
  unfold productTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => (c * U i) * V i))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (c * meanOn (fiber bin b) (fun i => U i * V i))) := by
            apply Finset.sum_congr rfl
            intro b hb
            congr 1
            calc
              meanOn (fiber bin b) (fun i => (c * U i) * V i)
                  = meanOn (fiber bin b) (fun i => c * (U i * V i)) := by
                      apply meanOn_congr (fiber bin b)
                      intro i hi
                      ring
              _ = c * meanOn (fiber bin b) (fun i => U i * V i) := by
                    rw [meanOn_scale_left]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => c * (binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i))) := by
            apply Finset.sum_congr rfl
            intro b hb
            ring
    _ = c * Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i)) := by
            rw [← Finset.mul_sum]
    _ = c * productTerm bin U V := by
          rw [productTerm]

theorem productTerm_scale_right (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    productTerm bin U (fun i => V i * c) = productTerm bin U V * c := by
  unfold productTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * (V i * c)))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) (fun i => U i * V i) * c)) := by
            apply Finset.sum_congr rfl
            intro b hb
            congr 1
            calc
              meanOn (fiber bin b) (fun i => U i * (V i * c))
                  = meanOn (fiber bin b) (fun i => (U i * V i) * c) := by
                      apply meanOn_congr (fiber bin b)
                      intro i hi
                      ring
              _ = meanOn (fiber bin b) (fun i => U i * V i) * c := by
                    rw [meanOn_scale_right]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => (binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i)) * c) := by
            apply Finset.sum_congr rfl
            intro b hb
            ring
    _ = Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V i)) * c := by
            rw [Finset.sum_mul]
    _ = productTerm bin U V * c := by
          rw [productTerm]

theorem productTerm_add_left (bin : α → β) (U₁ U₂ V : α → ℚ) :
    productTerm bin (fun i => U₁ i + U₂ i) V =
    productTerm bin U₁ V + productTerm bin U₂ V := by
  unfold productTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => (U₁ i + U₂ i) * V i))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b *
            (meanOn (fiber bin b) (fun i => U₁ i * V i) +
             meanOn (fiber bin b) (fun i => U₂ i * V i))) := by
              apply Finset.sum_congr rfl
              intro b hb
              congr 1
              calc
                meanOn (fiber bin b) (fun i => (U₁ i + U₂ i) * V i)
                    = meanOn (fiber bin b) (fun i => U₁ i * V i + U₂ i * V i) := by
                        apply meanOn_congr (fiber bin b)
                        intro i hi
                        ring
                _ = meanOn (fiber bin b) (fun i => U₁ i * V i) +
                    meanOn (fiber bin b) (fun i => U₂ i * V i) := by
                        rw [meanOn_add]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U₁ i * V i) +
            binWeight bin b * meanOn (fiber bin b) (fun i => U₂ i * V i)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U₁ i * V i)) +
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U₂ i * V i)) := by
            rw [Finset.sum_add_distrib]

theorem productTerm_add_right (bin : α → β) (U V₁ V₂ : α → ℚ) :
    productTerm bin U (fun i => V₁ i + V₂ i) =
    productTerm bin U V₁ + productTerm bin U V₂ := by
  unfold productTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * (V₁ i + V₂ i)))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b *
            (meanOn (fiber bin b) (fun i => U i * V₁ i) +
             meanOn (fiber bin b) (fun i => U i * V₂ i))) := by
              apply Finset.sum_congr rfl
              intro b hb
              congr 1
              calc
                meanOn (fiber bin b) (fun i => U i * (V₁ i + V₂ i))
                    = meanOn (fiber bin b) (fun i => U i * V₁ i + U i * V₂ i) := by
                        apply meanOn_congr (fiber bin b)
                        intro i hi
                        ring
                _ = meanOn (fiber bin b) (fun i => U i * V₁ i) +
                    meanOn (fiber bin b) (fun i => U i * V₂ i) := by
                        rw [meanOn_add]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V₁ i) +
            binWeight bin b * meanOn (fiber bin b) (fun i => U i * V₂ i)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V₁ i)) +
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * meanOn (fiber bin b) (fun i => U i * V₂ i)) := by
            rw [Finset.sum_add_distrib]

theorem meanProductTerm_scale_left (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProductTerm bin (fun i => c * U i) V = c * meanProductTerm bin U V := by
  unfold meanProductTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b *
          (meanOn (fiber bin b) (fun i => c * U i) * meanOn (fiber bin b) V))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * ((c * meanOn (fiber bin b) U) * meanOn (fiber bin b) V)) := by
            apply Finset.sum_congr rfl
            intro b hb
            rw [meanOn_scale_left]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => c * (binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V))) := by
            apply Finset.sum_congr rfl
            intro b hb
            ring
    _ = c * Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V)) := by
            rw [← Finset.mul_sum]
    _ = c * meanProductTerm bin U V := by
          rw [meanProductTerm]

theorem meanProductTerm_scale_right (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProductTerm bin U (fun i => V i * c) = meanProductTerm bin U V * c := by
  unfold meanProductTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b *
          (meanOn (fiber bin b) U * meanOn (fiber bin b) (fun i => V i * c)))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * (meanOn (fiber bin b) V * c))) := by
            apply Finset.sum_congr rfl
            intro b hb
            rw [meanOn_scale_right]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => (binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V)) * c) := by
            apply Finset.sum_congr rfl
            intro b hb
            ring
    _ = Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V)) * c := by
            rw [Finset.sum_mul]
    _ = meanProductTerm bin U V * c := by
          rw [meanProductTerm]

theorem meanProductTerm_add_left (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanProductTerm bin (fun i => U₁ i + U₂ i) V =
    meanProductTerm bin U₁ V + meanProductTerm bin U₂ V := by
  unfold meanProductTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b *
          (meanOn (fiber bin b) (fun i => U₁ i + U₂ i) * meanOn (fiber bin b) V))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b *
            ((meanOn (fiber bin b) U₁ + meanOn (fiber bin b) U₂) * meanOn (fiber bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [meanOn_add]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U₁ * meanOn (fiber bin b) V) +
            binWeight bin b * (meanOn (fiber bin b) U₂ * meanOn (fiber bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U₁ * meanOn (fiber bin b) V)) +
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U₂ * meanOn (fiber bin b) V)) := by
            rw [Finset.sum_add_distrib]

theorem meanProductTerm_add_right (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanProductTerm bin U (fun i => V₁ i + V₂ i) =
    meanProductTerm bin U V₁ + meanProductTerm bin U V₂ := by
  unfold meanProductTerm
  calc
    Finset.sum (Finset.univ.image bin)
        (fun b => binWeight bin b *
          (meanOn (fiber bin b) U * meanOn (fiber bin b) (fun i => V₁ i + V₂ i)))
        =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b *
            (meanOn (fiber bin b) U * (meanOn (fiber bin b) V₁ + meanOn (fiber bin b) V₂))) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [meanOn_add]
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V₁) +
            binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V₂)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V₁)) +
        Finset.sum (Finset.univ.image bin)
          (fun b => binWeight bin b * (meanOn (fiber bin b) U * meanOn (fiber bin b) V₂)) := by
            rw [Finset.sum_add_distrib]

theorem productTermOn_congr (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    productTermOn S bin U V = productTermOn S bin U' V' := by
  unfold productTermOn
  apply Finset.sum_congr rfl
  intro b hb
  congr 1
  apply meanOn_congr (fiberIn S bin b)
  intro i hi
  rw [hU i, hV i]

theorem meanProductTermOn_congr (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanProductTermOn S bin U V = meanProductTermOn S bin U' V' := by
  unfold meanProductTermOn
  apply Finset.sum_congr rfl
  intro b hb
  rw [meanOn_congr (fiberIn S bin b) (fun i _ => hU i)]
  rw [meanOn_congr (fiberIn S bin b) (fun i _ => hV i)]

theorem productTermOn_scale_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    productTermOn S bin (fun i => c * U i) V = c * productTermOn S bin U V := by
  unfold productTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => (c * U i) * V i))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * (c * meanOn (fiberIn S bin b) (fun i => U i * V i))) := by
            apply Finset.sum_congr rfl
            intro b hb
            congr 1
            calc
              meanOn (fiberIn S bin b) (fun i => (c * U i) * V i)
                  = meanOn (fiberIn S bin b) (fun i => c * (U i * V i)) := by
                      apply meanOn_congr (fiberIn S bin b)
                      intro i hi
                      ring
              _ = c * meanOn (fiberIn S bin b) (fun i => U i * V i) := by
                    rw [meanOn_scale_left]
    _ =
        Finset.sum (S.image bin)
          (fun b => c * (binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i))) := by
            apply Finset.sum_congr rfl
            intro b hb
            ring
    _ = c * Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i)) := by
            rw [← Finset.mul_sum]
    _ = c * productTermOn S bin U V := by
          rw [productTermOn]

theorem productTermOn_scale_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    productTermOn S bin U (fun i => V i * c) = productTermOn S bin U V * c := by
  unfold productTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * (V i * c)))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * (meanOn (fiberIn S bin b) (fun i => U i * V i) * c)) := by
            apply Finset.sum_congr rfl
            intro b hb
            congr 1
            calc
              meanOn (fiberIn S bin b) (fun i => U i * (V i * c))
                  = meanOn (fiberIn S bin b) (fun i => (U i * V i) * c) := by
                      apply meanOn_congr (fiberIn S bin b)
                      intro i hi
                      ring
              _ = meanOn (fiberIn S bin b) (fun i => U i * V i) * c := by
                    rw [meanOn_scale_right]
    _ =
        Finset.sum (S.image bin)
          (fun b => (binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i)) * c) := by
            apply Finset.sum_congr rfl
            intro b hb
            ring
    _ = Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i)) * c := by
            rw [Finset.sum_mul]
    _ = productTermOn S bin U V * c := by
          rw [productTermOn]

theorem productTermOn_add_left (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    productTermOn S bin (fun i => U₁ i + U₂ i) V =
    productTermOn S bin U₁ V + productTermOn S bin U₂ V := by
  unfold productTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => (U₁ i + U₂ i) * V i))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) (fun i => U₁ i * V i) +
             meanOn (fiberIn S bin b) (fun i => U₂ i * V i))) := by
              apply Finset.sum_congr rfl
              intro b hb
              congr 1
              calc
                meanOn (fiberIn S bin b) (fun i => (U₁ i + U₂ i) * V i)
                    = meanOn (fiberIn S bin b) (fun i => U₁ i * V i + U₂ i * V i) := by
                        apply meanOn_congr (fiberIn S bin b)
                        intro i hi
                        ring
                _ = meanOn (fiberIn S bin b) (fun i => U₁ i * V i) +
                    meanOn (fiberIn S bin b) (fun i => U₂ i * V i) := by
                        rw [meanOn_add]
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U₁ i * V i) +
            binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U₂ i * V i)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U₁ i * V i)) +
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U₂ i * V i)) := by
            rw [Finset.sum_add_distrib]

theorem productTermOn_add_right (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    productTermOn S bin U (fun i => V₁ i + V₂ i) =
    productTermOn S bin U V₁ + productTermOn S bin U V₂ := by
  unfold productTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * (V₁ i + V₂ i)))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) (fun i => U i * V₁ i) +
             meanOn (fiberIn S bin b) (fun i => U i * V₂ i))) := by
              apply Finset.sum_congr rfl
              intro b hb
              congr 1
              calc
                meanOn (fiberIn S bin b) (fun i => U i * (V₁ i + V₂ i))
                    = meanOn (fiberIn S bin b) (fun i => U i * V₁ i + U i * V₂ i) := by
                        apply meanOn_congr (fiberIn S bin b)
                        intro i hi
                        ring
                _ = meanOn (fiberIn S bin b) (fun i => U i * V₁ i) +
                    meanOn (fiberIn S bin b) (fun i => U i * V₂ i) := by
                        rw [meanOn_add]
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V₁ i) +
            binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V₂ i)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V₁ i)) +
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V₂ i)) := by
            rw [Finset.sum_add_distrib]

theorem meanProductTermOn_scale_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProductTermOn S bin (fun i => c * U i) V = c * meanProductTermOn S bin U V := by
  unfold meanProductTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b *
          (meanOn (fiberIn S bin b) (fun i => c * U i) * meanOn (fiberIn S bin b) V))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            ((c * meanOn (fiberIn S bin b) U) * meanOn (fiberIn S bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [meanOn_scale_left]
    _ =
        Finset.sum (S.image bin)
          (fun b => c * (binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V))) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ = c * Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V)) := by
              rw [← Finset.mul_sum]
    _ = c * meanProductTermOn S bin U V := by
          rw [meanProductTermOn]

theorem meanProductTermOn_scale_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProductTermOn S bin U (fun i => V i * c) = meanProductTermOn S bin U V * c := by
  unfold meanProductTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b *
          (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) (fun i => V i * c)))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * (meanOn (fiberIn S bin b) V * c))) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [meanOn_scale_right]
    _ =
        Finset.sum (S.image bin)
          (fun b => (binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V)) * c) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ = Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V)) * c := by
              rw [Finset.sum_mul]
    _ = meanProductTermOn S bin U V * c := by
          rw [meanProductTermOn]

theorem meanProductTermOn_add_left (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanProductTermOn S bin (fun i => U₁ i + U₂ i) V =
    meanProductTermOn S bin U₁ V + meanProductTermOn S bin U₂ V := by
  unfold meanProductTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b *
          (meanOn (fiberIn S bin b) (fun i => U₁ i + U₂ i) * meanOn (fiberIn S bin b) V))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            ((meanOn (fiberIn S bin b) U₁ + meanOn (fiberIn S bin b) U₂) *
              meanOn (fiberIn S bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [meanOn_add]
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U₁ * meanOn (fiberIn S bin b) V) +
            binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U₂ * meanOn (fiberIn S bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U₁ * meanOn (fiberIn S bin b) V)) +
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U₂ * meanOn (fiberIn S bin b) V)) := by
            rw [Finset.sum_add_distrib]

theorem meanProductTermOn_add_right (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanProductTermOn S bin U (fun i => V₁ i + V₂ i) =
    meanProductTermOn S bin U V₁ + meanProductTermOn S bin U V₂ := by
  unfold meanProductTermOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b *
          (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) (fun i => V₁ i + V₂ i)))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U *
              (meanOn (fiberIn S bin b) V₁ + meanOn (fiberIn S bin b) V₂))) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [meanOn_add]
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V₁) +
            binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V₂)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V₁)) +
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V₂)) := by
            rw [Finset.sum_add_distrib]

theorem deltaProductTermOn_congr (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    deltaProductTermOn S bin U V = deltaProductTermOn S bin U' V' := by
  unfold deltaProductTermOn
  rw [productTermOn_congr S bin hU hV, productTerm_congr bin hU hV]

theorem deltaMeanProductTermOn_congr (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    deltaMeanProductTermOn S bin U V = deltaMeanProductTermOn S bin U' V' := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_congr S bin hU hV, meanProductTerm_congr bin hU hV]

theorem deltaProductTermOn_scale_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaProductTermOn S bin (fun i => c * U i) V = c * deltaProductTermOn S bin U V := by
  unfold deltaProductTermOn
  rw [productTermOn_scale_left, productTerm_scale_left]
  ring

theorem deltaProductTermOn_scale_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaProductTermOn S bin U (fun i => V i * c) = deltaProductTermOn S bin U V * c := by
  unfold deltaProductTermOn
  rw [productTermOn_scale_right, productTerm_scale_right]
  ring

theorem deltaMeanProductTermOn_scale_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaMeanProductTermOn S bin (fun i => c * U i) V = c * deltaMeanProductTermOn S bin U V := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_scale_left, meanProductTerm_scale_left]
  ring

theorem deltaMeanProductTermOn_scale_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaMeanProductTermOn S bin U (fun i => V i * c) = deltaMeanProductTermOn S bin U V * c := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_scale_right, meanProductTerm_scale_right]
  ring

theorem deltaProductTermOn_add_left (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    deltaProductTermOn S bin (fun i => U₁ i + U₂ i) V =
    deltaProductTermOn S bin U₁ V + deltaProductTermOn S bin U₂ V := by
  unfold deltaProductTermOn
  rw [productTermOn_add_left, productTerm_add_left]
  ring

theorem deltaProductTermOn_add_right (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    deltaProductTermOn S bin U (fun i => V₁ i + V₂ i) =
    deltaProductTermOn S bin U V₁ + deltaProductTermOn S bin U V₂ := by
  unfold deltaProductTermOn
  rw [productTermOn_add_right, productTerm_add_right]
  ring

theorem deltaMeanProductTermOn_add_left (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    deltaMeanProductTermOn S bin (fun i => U₁ i + U₂ i) V =
    deltaMeanProductTermOn S bin U₁ V + deltaMeanProductTermOn S bin U₂ V := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_add_left, meanProductTerm_add_left]
  ring

theorem deltaMeanProductTermOn_add_right (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    deltaMeanProductTermOn S bin U (fun i => V₁ i + V₂ i) =
    deltaMeanProductTermOn S bin U V₁ + deltaMeanProductTermOn S bin U V₂ := by
  unfold deltaMeanProductTermOn
  rw [meanProductTermOn_add_right, meanProductTerm_add_right]
  ring

theorem paperEstimatorOn_congr (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    paperEstimatorOn S bin U V = paperEstimatorOn S bin U' V' := by
  unfold paperEstimatorOn
  apply Finset.sum_congr rfl
  intro b hb
  congr 1
  apply covOn_congr (fiberIn S bin b)
  · intro i hi
    exact hU i
  · intro i hi
    exact hV i

theorem paperEstimatorOn_eq_sum_mean_prod_sub_sum_mean_means (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn S bin U V = productTermOn S bin U V - meanProductTermOn S bin U V := by
  unfold paperEstimatorOn
  unfold productTermOn meanProductTermOn
  calc
    Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U V)
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b *
            (meanOn (fiberIn S bin b) (fun i => U i * V i) -
             meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [covOn_eq_mean_prod_sub_means]
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i) -
            binWeightOn S bin b * (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V)) := by
              apply Finset.sum_congr rfl
              intro b hb
              ring
    _ =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * meanOn (fiberIn S bin b) (fun i => U i * V i)) -
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * (meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V)) := by
            rw [Finset.sum_sub_distrib]

theorem productTermOn_eq_pushforwardSumOn_mean_prod
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    productTermOn S bin U V =
      pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) (fun i => U i * V i)) := by
  unfold productTermOn pushforwardSumOn densityPushforwardSumOn
  unfold densityPushforwardWeightOn pushforwardWeightOn pushforwardSupportOn
  apply Finset.sum_congr rfl
  intro b hb
  ring

theorem pushforwardSumOn_fiberMean_eq_meanOn
    (S : Finset α) (bin : α → β) (F : α → ℚ) :
    pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) F) = meanOn S F := by
  by_cases hS : S.card = 0
  · have hSe : S = ∅ := Finset.card_eq_zero.mp hS
    subst hSe
    simp [pushforwardSumOn, meanOn]
  · unfold pushforwardSumOn densityPushforwardSumOn densityPushforwardWeightOn
      pushforwardWeightOn binWeightOn
    simp [hS, pushforwardSupportOn]
    have hf : ∀ i ∈ S, bin i ∈ S.image bin := by
      intro i hi
      exact Finset.mem_image_of_mem bin hi
    calc
      ∑ x ∈ S.image bin, meanOn (fiberIn S bin x) F * (↑(fiberIn S bin x).card / ↑S.card)
          = ∑ x ∈ S.image bin, ∑ i ∈ S with bin i = x, F i / ↑S.card := by
              apply Finset.sum_congr rfl
              intro y hy
              have hfiber_nonzero : (fiberIn S bin y).card ≠ 0 := by
                rcases Finset.mem_image.mp hy with ⟨i, hiS, hiy⟩
                apply Finset.card_ne_zero.mpr
                refine ⟨i, ?_⟩
                simp [fiberIn, hiS, hiy]
              have hq : ((fiberIn S bin y).card : ℚ) ≠ 0 := by
                exact_mod_cast hfiber_nonzero
              rw [meanOn]
              simp [hfiber_nonzero]
              calc
                (∑ i ∈ fiberIn S bin y, F i) / ↑(fiberIn S bin y).card * (↑(fiberIn S bin y).card / ↑S.card)
                    = (∑ i ∈ fiberIn S bin y, F i) / ↑S.card := by
                        field_simp [hq]
                _ = ∑ i ∈ fiberIn S bin y, F i / ↑S.card := by
                      rw [← Finset.sum_div]
                _ = ∑ i ∈ S with bin i = y, F i / ↑S.card := by
                      rfl
      _ = ∑ i ∈ S, F i / ↑S.card := by
            simpa using (Finset.sum_fiberwise_of_maps_to hf (fun i => F i / ↑S.card))
      _ = meanOn S F := by
            simpa [meanOn, hS, ← Finset.sum_div]

theorem pushforwardSumOn_weightedFiberMean_eq_meanOn_comp
    (S : Finset α) (bin : α → β) (F : α → ℚ) (c : β → ℚ) :
    pushforwardSumOn S bin (fun b => c b * meanOn (fiberIn S bin b) F) =
      meanOn S (fun i => c (bin i) * F i) := by
  by_cases hS : S.card = 0
  · have hSe : S = ∅ := Finset.card_eq_zero.mp hS
    subst hSe
    simp [pushforwardSumOn, meanOn]
  · unfold pushforwardSumOn densityPushforwardSumOn densityPushforwardWeightOn
      pushforwardWeightOn binWeightOn
    simp [hS, pushforwardSupportOn]
    have hf : ∀ i ∈ S, bin i ∈ S.image bin := by
      intro i hi
      exact Finset.mem_image_of_mem bin hi
    calc
      ∑ x ∈ S.image bin,
          c x * meanOn (fiberIn S bin x) F *
            (↑(fiberIn S bin x).card / ↑S.card)
          = ∑ x ∈ S.image bin, ∑ i ∈ S with bin i = x, c (bin i) * F i / ↑S.card := by
              apply Finset.sum_congr rfl
              intro y hy
              have hfiber_nonzero : (fiberIn S bin y).card ≠ 0 := by
                rcases Finset.mem_image.mp hy with ⟨i, hiS, hiy⟩
                apply Finset.card_ne_zero.mpr
                refine ⟨i, ?_⟩
                simp [fiberIn, hiS, hiy]
              have hq : ((fiberIn S bin y).card : ℚ) ≠ 0 := by
                exact_mod_cast hfiber_nonzero
              rw [meanOn]
              simp [hfiber_nonzero]
              calc
                c y * ((∑ i ∈ fiberIn S bin y, F i) / ↑(fiberIn S bin y).card) *
                    (↑(fiberIn S bin y).card / ↑S.card)
                    = c y * ((∑ i ∈ fiberIn S bin y, F i) / ↑S.card) := by
                        field_simp [hq]
                _ = (c y * (∑ i ∈ fiberIn S bin y, F i)) / ↑S.card := by
                      ring
                _ = (∑ i ∈ fiberIn S bin y, c y * F i) / ↑S.card := by
                      rw [← Finset.mul_sum]
                _ = ∑ i ∈ fiberIn S bin y, c y * F i / ↑S.card := by
                      rw [← Finset.sum_div]
                _ = ∑ i ∈ S with bin i = y, c (bin i) * F i / ↑S.card := by
                      have hcongr :
                          ∑ i ∈ fiberIn S bin y, c y * F i / ↑S.card =
                            ∑ i ∈ fiberIn S bin y, c (bin i) * F i / ↑S.card := by
                              apply Finset.sum_congr rfl
                              intro i hi
                              have hiy : bin i = y := by
                                simp [fiberIn] at hi
                                exact hi.2
                              simp [hiy]
                      simpa [fiberIn] using hcongr
      _ = ∑ i ∈ S, c (bin i) * F i / ↑S.card := by
            simpa using (Finset.sum_fiberwise_of_maps_to hf (fun i => c (bin i) * F i / ↑S.card))
      _ = meanOn S (fun i => c (bin i) * F i) := by
            simpa [meanOn, hS, ← Finset.sum_div]

theorem meanProductTermOn_eq_pushforwardSumOn_mean_means
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    meanProductTermOn S bin U V =
      pushforwardSumOn S bin
        (fun b => meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V) := by
  unfold meanProductTermOn pushforwardSumOn densityPushforwardSumOn
  unfold densityPushforwardWeightOn pushforwardWeightOn pushforwardSupportOn
  apply Finset.sum_congr rfl
  intro b hb
  ring

theorem paperEstimatorOn_eq_pushforwardSumOn_mean_prod_sub_meanProductTermOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn S bin U V =
      pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) (fun i => U i * V i)) -
        meanProductTermOn S bin U V := by
  rw [paperEstimatorOn_eq_sum_mean_prod_sub_sum_mean_means,
    productTermOn_eq_pushforwardSumOn_mean_prod]

theorem paperEstimatorOn_eq_pushforwardSumOn_cov_channel
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn S bin U V =
      pushforwardSumOn S bin (fun b => covOn (fiberIn S bin b) U V) := by
  unfold paperEstimatorOn pushforwardSumOn densityPushforwardSumOn
  unfold densityPushforwardWeightOn pushforwardWeightOn pushforwardSupportOn
  apply Finset.sum_congr rfl
  intro b hb
  ring

theorem pushforwardSumOn_cov_channel_eq_pushforwardSumOn_mean_prod_sub_pushforwardSumOn_mean_means
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    pushforwardSumOn S bin (fun b => covOn (fiberIn S bin b) U V) =
      pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) (fun i => U i * V i)) -
        pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V) := by
  rw [← paperEstimatorOn_eq_pushforwardSumOn_cov_channel,
    paperEstimatorOn_eq_sum_mean_prod_sub_sum_mean_means,
    productTermOn_eq_pushforwardSumOn_mean_prod,
    meanProductTermOn_eq_pushforwardSumOn_mean_means]

theorem paperEstimatorOn_eq_pushforwardSumOn_mean_prod_sub_pushforwardSumOn_mean_means
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn S bin U V =
      pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) (fun i => U i * V i)) -
        pushforwardSumOn S bin (fun b => meanOn (fiberIn S bin b) U * meanOn (fiberIn S bin b) V) := by
  rw [paperEstimatorOn_eq_pushforwardSumOn_cov_channel,
    pushforwardSumOn_cov_channel_eq_pushforwardSumOn_mean_prod_sub_pushforwardSumOn_mean_means]

theorem productTermOn_eq_pushforwardSumOn_meanProdChannelOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    productTermOn S bin U V = pushforwardSumOn S bin (meanProdChannelOn S bin U V) := by
  simpa [meanProdChannelOn] using
    (productTermOn_eq_pushforwardSumOn_mean_prod (S := S) (bin := bin) (U := U) (V := V))

theorem meanProductTermOn_eq_pushforwardSumOn_meanMeansChannelOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    meanProductTermOn S bin U V = pushforwardSumOn S bin (meanMeansChannelOn S bin U V) := by
  simpa [meanMeansChannelOn] using
    (meanProductTermOn_eq_pushforwardSumOn_mean_means (S := S) (bin := bin) (U := U) (V := V))

theorem paperEstimatorOn_eq_pushforwardSumOn_covChannelOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn S bin U V = pushforwardSumOn S bin (covChannelOn S bin U V) := by
  simpa [covChannelOn] using
    (paperEstimatorOn_eq_pushforwardSumOn_cov_channel (S := S) (bin := bin) (U := U) (V := V))

omit [Fintype α] [DecidableEq α] [Fintype β] in
theorem covChannelOn_eq_meanProdChannelOn_sub_meanMeansChannelOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    covChannelOn S bin U V =
      fun b => meanProdChannelOn S bin U V b - meanMeansChannelOn S bin U V b := by
  funext b
  simp [covChannelOn, meanProdChannelOn, meanMeansChannelOn, covOn_eq_mean_prod_sub_means]

theorem pushforwardSumOn_covChannelOn_eq_pushforwardSumOn_meanProdChannelOn_sub_pushforwardSumOn_meanMeansChannelOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    pushforwardSumOn S bin (covChannelOn S bin U V) =
      pushforwardSumOn S bin (meanProdChannelOn S bin U V) -
        pushforwardSumOn S bin (meanMeansChannelOn S bin U V) := by
  simpa [covChannelOn, meanProdChannelOn, meanMeansChannelOn] using
    (pushforwardSumOn_cov_channel_eq_pushforwardSumOn_mean_prod_sub_pushforwardSumOn_mean_means
      (S := S) (bin := bin) (U := U) (V := V))

theorem paperEstimatorOn_eq_pushforwardSumOn_meanProdChannelOn_sub_pushforwardSumOn_meanMeansChannelOn
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorOn S bin U V =
      pushforwardSumOn S bin (meanProdChannelOn S bin U V) -
        pushforwardSumOn S bin (meanMeansChannelOn S bin U V) := by
  simpa [meanProdChannelOn, meanMeansChannelOn] using
    (paperEstimatorOn_eq_pushforwardSumOn_mean_prod_sub_pushforwardSumOn_mean_means
      (S := S) (bin := bin) (U := U) (V := V))

theorem productTerm_eq_pushforwardSumOn_meanProdChannelOn_univ
    (bin : α → β) (U V : α → ℚ) :
    productTerm bin U V =
      pushforwardSumOn Finset.univ bin (meanProdChannelOn Finset.univ bin U V) := by
  simpa [productTermOn_univ] using
    (productTermOn_eq_pushforwardSumOn_meanProdChannelOn
      (S := Finset.univ) (bin := bin) (U := U) (V := V))

theorem meanProductTerm_eq_pushforwardSumOn_meanMeansChannelOn_univ
    (bin : α → β) (U V : α → ℚ) :
    meanProductTerm bin U V =
      pushforwardSumOn Finset.univ bin (meanMeansChannelOn Finset.univ bin U V) := by
  simpa [meanProductTermOn_univ] using
    (meanProductTermOn_eq_pushforwardSumOn_meanMeansChannelOn
      (S := Finset.univ) (bin := bin) (U := U) (V := V))

theorem paperEstimator_eq_pushforwardSumOn_covChannelOn_univ
    (bin : α → β) (U V : α → ℚ) :
    paperEstimator bin U V =
      pushforwardSumOn Finset.univ bin (covChannelOn Finset.univ bin U V) := by
  simpa [paperEstimatorOn_univ] using
    (paperEstimatorOn_eq_pushforwardSumOn_covChannelOn
      (S := Finset.univ) (bin := bin) (U := U) (V := V))

theorem deltaProductTermBulk_eq_pushforwardSumBulk_meanProdChannelBulk_sub_pushforwardSumOn_meanProdChannelOn_univ
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaProductTermBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (meanProdChannelBulk slack ζ T bin U V) -
        pushforwardSumOn Finset.univ bin (meanProdChannelOn Finset.univ bin U V) := by
  unfold deltaProductTermBulk deltaProductTermOn
  rw [productTermOn_eq_pushforwardSumOn_meanProdChannelOn,
      productTerm_eq_pushforwardSumOn_meanProdChannelOn_univ]
  rfl

theorem deltaMeanProductTermBulk_eq_pushforwardSumBulk_meanMeansChannelBulk_sub_pushforwardSumOn_meanMeansChannelOn_univ
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaMeanProductTermBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (meanMeansChannelBulk slack ζ T bin U V) -
        pushforwardSumOn Finset.univ bin (meanMeansChannelOn Finset.univ bin U V) := by
  unfold deltaMeanProductTermBulk deltaMeanProductTermOn
  rw [meanProductTermOn_eq_pushforwardSumOn_meanMeansChannelOn,
      meanProductTerm_eq_pushforwardSumOn_meanMeansChannelOn_univ]
  rfl

theorem deltaPaperEstimatorBulk_eq_pushforwardSumBulk_covChannelBulk_sub_pushforwardSumOn_covChannelOn_univ
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (covChannelBulk slack ζ T bin U V) -
        pushforwardSumOn Finset.univ bin (covChannelOn Finset.univ bin U V) := by
  unfold deltaPaperEstimatorBulk deltaPaperEstimatorOn
  rw [paperEstimatorOn_eq_pushforwardSumOn_covChannelOn,
      paperEstimator_eq_pushforwardSumOn_covChannelOn_univ]
  rfl

theorem productTermBulk_eq_pushforwardSumBulk_meanProdChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    productTermBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (meanProdChannelBulk slack ζ T bin U V) := by
  simpa [productTermBulk, pushforwardSumBulk, meanProdChannelBulk] using
    (productTermOn_eq_pushforwardSumOn_meanProdChannelOn
      (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V))

theorem meanProductTermBulk_eq_pushforwardSumBulk_meanMeansChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    meanProductTermBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (meanMeansChannelBulk slack ζ T bin U V) := by
  simpa [meanProductTermBulk, pushforwardSumBulk, meanMeansChannelBulk] using
    (meanProductTermOn_eq_pushforwardSumOn_meanMeansChannelOn
      (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V))

theorem paperEstimatorBulk_eq_pushforwardSumBulk_covChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (covChannelBulk slack ζ T bin U V) := by
  simpa [paperEstimatorBulk, pushforwardSumBulk, covChannelBulk] using
    (paperEstimatorOn_eq_pushforwardSumOn_covChannelOn
      (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V))

theorem covChannelBulk_eq_meanProdChannelBulk_sub_meanMeansChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    covChannelBulk slack ζ T bin U V =
      fun b =>
        meanProdChannelBulk slack ζ T bin U V b -
          meanMeansChannelBulk slack ζ T bin U V b := by
  simpa [covChannelBulk, meanProdChannelBulk, meanMeansChannelBulk] using
    (covChannelOn_eq_meanProdChannelOn_sub_meanMeansChannelOn
      (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V))

theorem pushforwardSumBulk_covChannelBulk_eq_pushforwardSumBulk_meanProdChannelBulk_sub_pushforwardSumBulk_meanMeansChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    pushforwardSumBulk slack ζ T bin (covChannelBulk slack ζ T bin U V) =
      pushforwardSumBulk slack ζ T bin (meanProdChannelBulk slack ζ T bin U V) -
        pushforwardSumBulk slack ζ T bin (meanMeansChannelBulk slack ζ T bin U V) := by
  simpa [pushforwardSumBulk, covChannelBulk, meanProdChannelBulk, meanMeansChannelBulk] using
    (pushforwardSumOn_covChannelOn_eq_pushforwardSumOn_meanProdChannelOn_sub_pushforwardSumOn_meanMeansChannelOn
      (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V))

theorem paperEstimatorBulk_eq_pushforwardSumBulk_meanProdChannelBulk_sub_pushforwardSumBulk_meanMeansChannelBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    paperEstimatorBulk slack ζ T bin U V =
      pushforwardSumBulk slack ζ T bin (meanProdChannelBulk slack ζ T bin U V) -
        pushforwardSumBulk slack ζ T bin (meanMeansChannelBulk slack ζ T bin U V) := by
  simpa [paperEstimatorBulk, pushforwardSumBulk, meanProdChannelBulk, meanMeansChannelBulk] using
    (paperEstimatorOn_eq_pushforwardSumOn_meanProdChannelOn_sub_pushforwardSumOn_meanMeansChannelOn
      (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V))

theorem deltaPaperEstimatorOn_eq_deltaProduct_minus_deltaMean
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    deltaPaperEstimatorOn S bin U V =
      deltaProductTermOn S bin U V - deltaMeanProductTermOn S bin U V := by
  unfold deltaPaperEstimatorOn deltaProductTermOn deltaMeanProductTermOn
  rw [paperEstimatorOn_eq_sum_mean_prod_sub_sum_mean_means,
      paperEstimator_eq_sum_mean_prod_sub_sum_mean_means]
  ring

theorem deltaPaperEstimatorBulk_eq_deltaProductBulk_sub_deltaMeanBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin U V =
      deltaProductTermBulk slack ζ T bin U V - deltaMeanProductTermBulk slack ζ T bin U V := by
  unfold deltaPaperEstimatorBulk deltaProductTermBulk deltaMeanProductTermBulk
  exact deltaPaperEstimatorOn_eq_deltaProduct_minus_deltaMean
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V)

theorem abs_deltaPaperEstimatorOn_le_abs_deltaProduct_add_abs_deltaMean
    (S : Finset α) (bin : α → β) (U V : α → ℚ) :
    |deltaPaperEstimatorOn S bin U V| ≤
      |deltaProductTermOn S bin U V| + |deltaMeanProductTermOn S bin U V| := by
  rw [deltaPaperEstimatorOn_eq_deltaProduct_minus_deltaMean]
  simpa [sub_eq_add_neg, abs_neg] using
    (abs_add_le (deltaProductTermOn S bin U V) (-deltaMeanProductTermOn S bin U V))

theorem abs_deltaPaperEstimatorBulk_le_abs_deltaProductBulk_add_abs_deltaMeanBulk
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    |deltaPaperEstimatorBulk slack ζ T bin U V| ≤
      |deltaProductTermBulk slack ζ T bin U V| + |deltaMeanProductTermBulk slack ζ T bin U V| := by
  rw [deltaPaperEstimatorBulk_eq_deltaProductBulk_sub_deltaMeanBulk]
  simpa [sub_eq_add_neg, abs_neg] using
    (abs_add_le (deltaProductTermBulk slack ζ T bin U V) (-deltaMeanProductTermBulk slack ζ T bin U V))

theorem deltaPaperEstimatorBulk_eq_pushforwardDifference_meanProd_minus_pushforwardDifference_meanMeans
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin U V =
      (pushforwardSumBulk slack ζ T bin (meanProdChannelBulk slack ζ T bin U V) -
        pushforwardSumOn Finset.univ bin (meanProdChannelOn Finset.univ bin U V)) -
      (pushforwardSumBulk slack ζ T bin (meanMeansChannelBulk slack ζ T bin U V) -
        pushforwardSumOn Finset.univ bin (meanMeansChannelOn Finset.univ bin U V)) := by
  rw [deltaPaperEstimatorBulk_eq_deltaProductBulk_sub_deltaMeanBulk,
      deltaProductTermBulk_eq_pushforwardSumBulk_meanProdChannelBulk_sub_pushforwardSumOn_meanProdChannelOn_univ,
      deltaMeanProductTermBulk_eq_pushforwardSumBulk_meanMeansChannelBulk_sub_pushforwardSumOn_meanMeansChannelOn_univ]

theorem deltaProductTermBulk_eq_bulkMean_minus_univMean_meanProdChannel
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaProductTermBulk slack ζ T bin U V =
      meanOn (bulkSupportSlack slack ζ T) (fun i => meanProdChannelBulk slack ζ T bin U V (bin i)) -
        meanOn Finset.univ (fun i => meanProdChannelOn Finset.univ bin U V (bin i)) := by
  rw [deltaProductTermBulk_eq_pushforwardSumBulk_meanProdChannelBulk_sub_pushforwardSumOn_meanProdChannelOn_univ,
      pushforwardSumBulk_eq_meanOn_comp, pushforwardSumOn_eq_meanOn_comp]

theorem deltaMeanProductTermBulk_eq_bulkMean_minus_univMean_meanMeansChannel
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaMeanProductTermBulk slack ζ T bin U V =
      meanOn (bulkSupportSlack slack ζ T) (fun i => meanMeansChannelBulk slack ζ T bin U V (bin i)) -
        meanOn Finset.univ (fun i => meanMeansChannelOn Finset.univ bin U V (bin i)) := by
  rw [deltaMeanProductTermBulk_eq_pushforwardSumBulk_meanMeansChannelBulk_sub_pushforwardSumOn_meanMeansChannelOn_univ,
      pushforwardSumBulk_eq_meanOn_comp, pushforwardSumOn_eq_meanOn_comp]

theorem deltaPaperEstimatorBulk_eq_bulkMean_minus_univMean_covChannel
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin U V =
      meanOn (bulkSupportSlack slack ζ T) (fun i => covChannelBulk slack ζ T bin U V (bin i)) -
        meanOn Finset.univ (fun i => covChannelOn Finset.univ bin U V (bin i)) := by
  rw [deltaPaperEstimatorBulk_eq_pushforwardSumBulk_covChannelBulk_sub_pushforwardSumOn_covChannelOn_univ,
      pushforwardSumBulk_eq_meanOn_comp, pushforwardSumOn_eq_meanOn_comp]

omit [Fintype α] [Fintype β] in
theorem paperEstimatorOn_scale_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    paperEstimatorOn S bin (fun i => c * U i) V = c * paperEstimatorOn S bin U V := by
  unfold paperEstimatorOn
  simp [covOn_scale_left, mul_left_comm]
  rw [← Finset.mul_sum]

omit [Fintype α] [Fintype β] in
theorem paperEstimatorOn_scale_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    paperEstimatorOn S bin U (fun i => V i * c) = paperEstimatorOn S bin U V * c := by
  unfold paperEstimatorOn
  calc
    Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U (fun i => V i * c))
        = Finset.sum (S.image bin) (fun b => (binWeightOn S bin b * covOn (fiberIn S bin b) U V) * c) := by
            apply Finset.sum_congr rfl
            intro b hb
            rw [covOn_scale_right]
            ring
    _ = (Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U V)) * c := by
          rw [Finset.sum_mul]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimatorOn_add_left (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    paperEstimatorOn S bin (fun i => U₁ i + U₂ i) V =
    paperEstimatorOn S bin U₁ V + paperEstimatorOn S bin U₂ V := by
  unfold paperEstimatorOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) (fun i => U₁ i + U₂ i) V)
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U₁ V +
            binWeightOn S bin b * covOn (fiberIn S bin b) U₂ V) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [covOn_add_left]
              ring
    _ =
        Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U₁ V) +
        Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U₂ V) := by
          rw [Finset.sum_add_distrib]

omit [DecidableEq α] [Fintype β] in
theorem paperEstimatorOn_add_right (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    paperEstimatorOn S bin U (fun i => V₁ i + V₂ i) =
    paperEstimatorOn S bin U V₁ + paperEstimatorOn S bin U V₂ := by
  unfold paperEstimatorOn
  calc
    Finset.sum (S.image bin)
        (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U (fun i => V₁ i + V₂ i))
        =
        Finset.sum (S.image bin)
          (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U V₁ +
            binWeightOn S bin b * covOn (fiberIn S bin b) U V₂) := by
              apply Finset.sum_congr rfl
              intro b hb
              rw [covOn_add_right]
              ring
    _ =
        Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U V₁) +
        Finset.sum (S.image bin) (fun b => binWeightOn S bin b * covOn (fiberIn S bin b) U V₂) := by
          rw [Finset.sum_add_distrib]

theorem deltaPaperEstimatorOn_congr (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    deltaPaperEstimatorOn S bin U V = deltaPaperEstimatorOn S bin U' V' := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_congr S bin hU hV, paperEstimator_congr bin hU hV]

theorem deltaPaperEstimatorOn_scale_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaPaperEstimatorOn S bin (fun i => c * U i) V = c * deltaPaperEstimatorOn S bin U V := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_scale_left, paperEstimator_scale_left]
  ring

theorem deltaPaperEstimatorOn_scale_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaPaperEstimatorOn S bin U (fun i => V i * c) = deltaPaperEstimatorOn S bin U V * c := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_scale_right, paperEstimator_scale_right]
  ring

theorem deltaPaperEstimatorOn_add_left (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    deltaPaperEstimatorOn S bin (fun i => U₁ i + U₂ i) V =
    deltaPaperEstimatorOn S bin U₁ V + deltaPaperEstimatorOn S bin U₂ V := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_add_left, paperEstimator_add_left]
  ring

theorem deltaPaperEstimatorOn_add_right (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    deltaPaperEstimatorOn S bin U (fun i => V₁ i + V₂ i) =
    deltaPaperEstimatorOn S bin U V₁ + deltaPaperEstimatorOn S bin U V₂ := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_add_right, paperEstimator_add_right]
  ring

theorem deltaProductTermBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    deltaProductTermBulk slack ζ T bin U V =
      deltaProductTermBulk slack ζ T bin U' V' := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_congr
    (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem deltaMeanProductTermBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    deltaMeanProductTermBulk slack ζ T bin U V =
      deltaMeanProductTermBulk slack ζ T bin U' V' := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_congr
    (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem deltaProductTermBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaProductTermBulk slack ζ T bin (fun i => c * U i) V =
      c * deltaProductTermBulk slack ζ T bin U V := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_scale_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem deltaProductTermBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaProductTermBulk slack ζ T bin U (fun i => V i * c) =
      deltaProductTermBulk slack ζ T bin U V * c := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_scale_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem deltaMeanProductTermBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaMeanProductTermBulk slack ζ T bin (fun i => c * U i) V =
      c * deltaMeanProductTermBulk slack ζ T bin U V := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_scale_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem deltaMeanProductTermBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaMeanProductTermBulk slack ζ T bin U (fun i => V i * c) =
      deltaMeanProductTermBulk slack ζ T bin U V * c := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_scale_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem deltaProductTermBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    deltaProductTermBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      deltaProductTermBulk slack ζ T bin U₁ V +
        deltaProductTermBulk slack ζ T bin U₂ V := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_add_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem deltaProductTermBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    deltaProductTermBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      deltaProductTermBulk slack ζ T bin U V₁ +
        deltaProductTermBulk slack ζ T bin U V₂ := by
  unfold deltaProductTermBulk
  exact deltaProductTermOn_add_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem deltaMeanProductTermBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    deltaMeanProductTermBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      deltaMeanProductTermBulk slack ζ T bin U₁ V +
        deltaMeanProductTermBulk slack ζ T bin U₂ V := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_add_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem deltaMeanProductTermBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    deltaMeanProductTermBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      deltaMeanProductTermBulk slack ζ T bin U V₁ +
        deltaMeanProductTermBulk slack ζ T bin U V₂ := by
  unfold deltaMeanProductTermBulk
  exact deltaMeanProductTermOn_add_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem productTermBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    productTermBulk slack ζ T bin U V = productTermBulk slack ζ T bin U' V' := by
  unfold productTermBulk
  exact productTermOn_congr (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem productTermBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    productTermBulk slack ζ T bin (fun i => c * U i) V = c * productTermBulk slack ζ T bin U V := by
  unfold productTermBulk
  exact productTermOn_scale_left (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem productTermBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    productTermBulk slack ζ T bin U (fun i => V i * c) = productTermBulk slack ζ T bin U V * c := by
  unfold productTermBulk
  exact productTermOn_scale_right (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem productTermBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    productTermBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      productTermBulk slack ζ T bin U₁ V + productTermBulk slack ζ T bin U₂ V := by
  unfold productTermBulk
  exact productTermOn_add_left (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem productTermBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    productTermBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      productTermBulk slack ζ T bin U V₁ + productTermBulk slack ζ T bin U V₂ := by
  unfold productTermBulk
  exact productTermOn_add_right (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem meanProductTermBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanProductTermBulk slack ζ T bin U V = meanProductTermBulk slack ζ T bin U' V' := by
  unfold meanProductTermBulk
  exact meanProductTermOn_congr (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem meanProductTermBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProductTermBulk slack ζ T bin (fun i => c * U i) V = c * meanProductTermBulk slack ζ T bin U V := by
  unfold meanProductTermBulk
  exact meanProductTermOn_scale_left (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem meanProductTermBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProductTermBulk slack ζ T bin U (fun i => V i * c) = meanProductTermBulk slack ζ T bin U V * c := by
  unfold meanProductTermBulk
  exact meanProductTermOn_scale_right (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem meanProductTermBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanProductTermBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      meanProductTermBulk slack ζ T bin U₁ V + meanProductTermBulk slack ζ T bin U₂ V := by
  unfold meanProductTermBulk
  exact meanProductTermOn_add_left (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem meanProductTermBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanProductTermBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      meanProductTermBulk slack ζ T bin U V₁ + meanProductTermBulk slack ζ T bin U V₂ := by
  unfold meanProductTermBulk
  exact meanProductTermOn_add_right (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem paperEstimatorBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    paperEstimatorBulk slack ζ T bin U V = paperEstimatorBulk slack ζ T bin U' V' := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_congr (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem paperEstimatorBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    paperEstimatorBulk slack ζ T bin (fun i => c * U i) V = c * paperEstimatorBulk slack ζ T bin U V := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_scale_left (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem paperEstimatorBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    paperEstimatorBulk slack ζ T bin U (fun i => V i * c) = paperEstimatorBulk slack ζ T bin U V * c := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_scale_right (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem paperEstimatorBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    paperEstimatorBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      paperEstimatorBulk slack ζ T bin U₁ V + paperEstimatorBulk slack ζ T bin U₂ V := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_add_left (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem paperEstimatorBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    paperEstimatorBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      paperEstimatorBulk slack ζ T bin U V₁ + paperEstimatorBulk slack ζ T bin U V₂ := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_add_right (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem deltaPaperEstimatorBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    deltaPaperEstimatorBulk slack ζ T bin U V =
      deltaPaperEstimatorBulk slack ζ T bin U' V' := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_congr
    (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem deltaPaperEstimatorBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin (fun i => c * U i) V =
      c * deltaPaperEstimatorBulk slack ζ T bin U V := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_scale_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem deltaPaperEstimatorBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin U (fun i => V i * c) =
      deltaPaperEstimatorBulk slack ζ T bin U V * c := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_scale_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem deltaPaperEstimatorBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      deltaPaperEstimatorBulk slack ζ T bin U₁ V +
        deltaPaperEstimatorBulk slack ζ T bin U₂ V := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_add_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem deltaPaperEstimatorBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    deltaPaperEstimatorBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      deltaPaperEstimatorBulk slack ζ T bin U V₁ +
        deltaPaperEstimatorBulk slack ζ T bin U V₂ := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_add_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem covOn_on_fiberIn_of_factorsThroughBin_left (S : Finset α) (bin : α → β) (U V : α → ℚ) (b : β)
    (hU : FactorsThroughBin bin U) :
    covOn (fiberIn S bin b) U V = 0 := by
  rcases hU with ⟨g, hg⟩
  apply Eq.trans ?_ (covOn_const_left (fiberIn S bin b) (g b) V)
  apply covOn_congr (fiberIn S bin b)
  · intro i hi
    have hmem : bin i = b := by
      simp [fiberIn] at hi
      exact hi.2
    rw [hg i, hmem]
  · intro i hi
    rfl

theorem covOn_on_fiberIn_of_factorsThroughBin_right (S : Finset α) (bin : α → β) (U V : α → ℚ) (b : β)
    (hV : FactorsThroughBin bin V) :
    covOn (fiberIn S bin b) U V = 0 := by
  rcases hV with ⟨g, hg⟩
  apply Eq.trans ?_ (covOn_const_right (fiberIn S bin b) U (g b))
  apply covOn_congr (fiberIn S bin b)
  · intro i hi
    rfl
  · intro i hi
    have hmem : bin i = b := by
      simp [fiberIn] at hi
      exact hi.2
    rw [hg i, hmem]

theorem covChannelOn_zero_of_factorsThroughBin_left
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    covChannelOn S bin U V = 0 := by
  funext b
  simpa [covChannelOn] using
    (covOn_on_fiberIn_of_factorsThroughBin_left (S := S) (bin := bin) (U := U) (V := V) (b := b) hU)

theorem covChannelOn_zero_of_factorsThroughBin_right
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    covChannelOn S bin U V = 0 := by
  funext b
  simpa [covChannelOn] using
    (covOn_on_fiberIn_of_factorsThroughBin_right (S := S) (bin := bin) (U := U) (V := V) (b := b) hV)

theorem meanProdChannelOn_congr
    (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanProdChannelOn S bin U V = meanProdChannelOn S bin U' V' := by
  funext b
  apply meanOn_congr (fiberIn S bin b)
  intro i hi
  rw [hU i, hV i]

theorem meanProdChannelOn_scale_left
    (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProdChannelOn S bin (fun i => c * U i) V =
      fun b => c * meanProdChannelOn S bin U V b := by
  funext b
  unfold meanProdChannelOn
  calc
    meanOn (fiberIn S bin b) (fun i => (c * U i) * V i)
        = meanOn (fiberIn S bin b) (fun i => c * (U i * V i)) := by
            apply meanOn_congr (fiberIn S bin b)
            intro i hi
            ring
    _ = c * meanOn (fiberIn S bin b) (fun i => U i * V i) := by
          rw [meanOn_scale_left]

theorem meanProdChannelOn_scale_right
    (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProdChannelOn S bin U (fun i => V i * c) =
      fun b => meanProdChannelOn S bin U V b * c := by
  funext b
  unfold meanProdChannelOn
  calc
    meanOn (fiberIn S bin b) (fun i => U i * (V i * c))
        = meanOn (fiberIn S bin b) (fun i => (U i * V i) * c) := by
            apply meanOn_congr (fiberIn S bin b)
            intro i hi
            ring
    _ = meanOn (fiberIn S bin b) (fun i => U i * V i) * c := by
          rw [meanOn_scale_right]

theorem meanProdChannelOn_add_left
    (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanProdChannelOn S bin (fun i => U₁ i + U₂ i) V =
      fun b => meanProdChannelOn S bin U₁ V b + meanProdChannelOn S bin U₂ V b := by
  funext b
  unfold meanProdChannelOn
  calc
    meanOn (fiberIn S bin b) (fun i => (U₁ i + U₂ i) * V i)
        = meanOn (fiberIn S bin b) (fun i => U₁ i * V i + U₂ i * V i) := by
            apply meanOn_congr (fiberIn S bin b)
            intro i hi
            ring
    _ = meanOn (fiberIn S bin b) (fun i => U₁ i * V i) +
        meanOn (fiberIn S bin b) (fun i => U₂ i * V i) := by
          rw [meanOn_add]

theorem meanProdChannelOn_add_right
    (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanProdChannelOn S bin U (fun i => V₁ i + V₂ i) =
      fun b => meanProdChannelOn S bin U V₁ b + meanProdChannelOn S bin U V₂ b := by
  funext b
  unfold meanProdChannelOn
  calc
    meanOn (fiberIn S bin b) (fun i => U i * (V₁ i + V₂ i))
        = meanOn (fiberIn S bin b) (fun i => U i * V₁ i + U i * V₂ i) := by
            apply meanOn_congr (fiberIn S bin b)
            intro i hi
            ring
    _ = meanOn (fiberIn S bin b) (fun i => U i * V₁ i) +
        meanOn (fiberIn S bin b) (fun i => U i * V₂ i) := by
          rw [meanOn_add]

theorem meanMeansChannelOn_congr
    (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanMeansChannelOn S bin U V = meanMeansChannelOn S bin U' V' := by
  funext b
  unfold meanMeansChannelOn
  rw [meanOn_congr (fiberIn S bin b) (fun i _ => hU i),
      meanOn_congr (fiberIn S bin b) (fun i _ => hV i)]

theorem meanMeansChannelOn_scale_left
    (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanMeansChannelOn S bin (fun i => c * U i) V =
      fun b => c * meanMeansChannelOn S bin U V b := by
  funext b
  unfold meanMeansChannelOn
  rw [meanOn_scale_left]
  ring

theorem meanMeansChannelOn_scale_right
    (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanMeansChannelOn S bin U (fun i => V i * c) =
      fun b => meanMeansChannelOn S bin U V b * c := by
  funext b
  unfold meanMeansChannelOn
  rw [meanOn_scale_right]
  ring

theorem meanMeansChannelOn_add_left
    (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanMeansChannelOn S bin (fun i => U₁ i + U₂ i) V =
      fun b => meanMeansChannelOn S bin U₁ V b + meanMeansChannelOn S bin U₂ V b := by
  funext b
  unfold meanMeansChannelOn
  rw [meanOn_add]
  ring

theorem meanMeansChannelOn_add_right
    (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanMeansChannelOn S bin U (fun i => V₁ i + V₂ i) =
      fun b => meanMeansChannelOn S bin U V₁ b + meanMeansChannelOn S bin U V₂ b := by
  funext b
  unfold meanMeansChannelOn
  rw [meanOn_add]
  ring

theorem covChannelOn_congr
    (S : Finset α) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    covChannelOn S bin U V = covChannelOn S bin U' V' := by
  funext b
  exact covOn_congr (fiberIn S bin b) (fun i _ => hU i) (fun i _ => hV i)

theorem covChannelOn_scale_left
    (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    covChannelOn S bin (fun i => c * U i) V = fun b => c * covChannelOn S bin U V b := by
  funext b
  simp [covChannelOn, covOn_scale_left]

theorem covChannelOn_scale_right
    (S : Finset α) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    covChannelOn S bin U (fun i => V i * c) = fun b => covChannelOn S bin U V b * c := by
  funext b
  simp [covChannelOn, covOn_scale_right]

theorem covChannelOn_add_left
    (S : Finset α) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    covChannelOn S bin (fun i => U₁ i + U₂ i) V =
      fun b => covChannelOn S bin U₁ V b + covChannelOn S bin U₂ V b := by
  funext b
  simp [covChannelOn, covOn_add_left]

theorem covChannelOn_add_right
    (S : Finset α) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    covChannelOn S bin U (fun i => V₁ i + V₂ i) =
      fun b => covChannelOn S bin U V₁ b + covChannelOn S bin U V₂ b := by
  funext b
  simp [covChannelOn, covOn_add_right]

theorem covChannelBulk_zero_of_factorsThroughBin_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    covChannelBulk slack ζ T bin U V = 0 := by
  unfold covChannelBulk
  exact covChannelOn_zero_of_factorsThroughBin_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hU

theorem covChannelBulk_zero_of_factorsThroughBin_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    covChannelBulk slack ζ T bin U V = 0 := by
  unfold covChannelBulk
  exact covChannelOn_zero_of_factorsThroughBin_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hV

theorem meanProdChannelBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanProdChannelBulk slack ζ T bin U V = meanProdChannelBulk slack ζ T bin U' V' := by
  unfold meanProdChannelBulk
  exact meanProdChannelOn_congr (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem meanProdChannelBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProdChannelBulk slack ζ T bin (fun i => c * U i) V =
      fun b => c * meanProdChannelBulk slack ζ T bin U V b := by
  unfold meanProdChannelBulk
  exact meanProdChannelOn_scale_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem meanProdChannelBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanProdChannelBulk slack ζ T bin U (fun i => V i * c) =
      fun b => meanProdChannelBulk slack ζ T bin U V b * c := by
  unfold meanProdChannelBulk
  exact meanProdChannelOn_scale_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem meanProdChannelBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanProdChannelBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      fun b => meanProdChannelBulk slack ζ T bin U₁ V b + meanProdChannelBulk slack ζ T bin U₂ V b := by
  unfold meanProdChannelBulk
  exact meanProdChannelOn_add_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem meanProdChannelBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanProdChannelBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      fun b => meanProdChannelBulk slack ζ T bin U V₁ b + meanProdChannelBulk slack ζ T bin U V₂ b := by
  unfold meanProdChannelBulk
  exact meanProdChannelOn_add_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem meanMeansChannelBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    meanMeansChannelBulk slack ζ T bin U V = meanMeansChannelBulk slack ζ T bin U' V' := by
  unfold meanMeansChannelBulk
  exact meanMeansChannelOn_congr (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem meanMeansChannelBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanMeansChannelBulk slack ζ T bin (fun i => c * U i) V =
      fun b => c * meanMeansChannelBulk slack ζ T bin U V b := by
  unfold meanMeansChannelBulk
  exact meanMeansChannelOn_scale_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem meanMeansChannelBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    meanMeansChannelBulk slack ζ T bin U (fun i => V i * c) =
      fun b => meanMeansChannelBulk slack ζ T bin U V b * c := by
  unfold meanMeansChannelBulk
  exact meanMeansChannelOn_scale_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem meanMeansChannelBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    meanMeansChannelBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      fun b => meanMeansChannelBulk slack ζ T bin U₁ V b + meanMeansChannelBulk slack ζ T bin U₂ V b := by
  unfold meanMeansChannelBulk
  exact meanMeansChannelOn_add_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem meanMeansChannelBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    meanMeansChannelBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      fun b => meanMeansChannelBulk slack ζ T bin U V₁ b + meanMeansChannelBulk slack ζ T bin U V₂ b := by
  unfold meanMeansChannelBulk
  exact meanMeansChannelOn_add_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem covChannelBulk_congr
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) {U U' V V' : α → ℚ}
    (hU : ∀ i, U i = U' i) (hV : ∀ i, V i = V' i) :
    covChannelBulk slack ζ T bin U V = covChannelBulk slack ζ T bin U' V' := by
  unfold covChannelBulk
  exact covChannelOn_congr (S := bulkSupportSlack slack ζ T) (bin := bin) hU hV

theorem covChannelBulk_scale_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    covChannelBulk slack ζ T bin (fun i => c * U i) V =
      fun b => c * covChannelBulk slack ζ T bin U V b := by
  unfold covChannelBulk
  exact covChannelOn_scale_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem covChannelBulk_scale_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ) (c : ℚ) :
    covChannelBulk slack ζ T bin U (fun i => V i * c) =
      fun b => covChannelBulk slack ζ T bin U V b * c := by
  unfold covChannelBulk
  exact covChannelOn_scale_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) (c := c)

theorem covChannelBulk_add_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U₁ U₂ V : α → ℚ) :
    covChannelBulk slack ζ T bin (fun i => U₁ i + U₂ i) V =
      fun b => covChannelBulk slack ζ T bin U₁ V b + covChannelBulk slack ζ T bin U₂ V b := by
  unfold covChannelBulk
  exact covChannelOn_add_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U₁ := U₁) (U₂ := U₂) (V := V)

theorem covChannelBulk_add_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V₁ V₂ : α → ℚ) :
    covChannelBulk slack ζ T bin U (fun i => V₁ i + V₂ i) =
      fun b => covChannelBulk slack ζ T bin U V₁ b + covChannelBulk slack ζ T bin U V₂ b := by
  unfold covChannelBulk
  exact covChannelOn_add_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V₁ := V₁) (V₂ := V₂)

theorem pushforwardSumOn_covChannelOn_zero_of_factorsThroughBin_left
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    pushforwardSumOn S bin (covChannelOn S bin U V) = 0 := by
  rw [covChannelOn_zero_of_factorsThroughBin_left (S := S) (bin := bin) (U := U) (V := V) hU]
  simp [pushforwardSumOn, densityPushforwardSumOn]

theorem pushforwardSumOn_covChannelOn_zero_of_factorsThroughBin_right
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    pushforwardSumOn S bin (covChannelOn S bin U V) = 0 := by
  rw [covChannelOn_zero_of_factorsThroughBin_right (S := S) (bin := bin) (U := U) (V := V) hV]
  simp [pushforwardSumOn, densityPushforwardSumOn]

theorem paperEstimatorOn_zero_of_factorsThroughBin_left'
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    paperEstimatorOn S bin U V = 0 := by
  rw [paperEstimatorOn_eq_pushforwardSumOn_covChannelOn]
  exact pushforwardSumOn_covChannelOn_zero_of_factorsThroughBin_left
    (S := S) (bin := bin) (U := U) (V := V) hU

theorem paperEstimatorOn_zero_of_factorsThroughBin_right'
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    paperEstimatorOn S bin U V = 0 := by
  rw [paperEstimatorOn_eq_pushforwardSumOn_covChannelOn]
  exact pushforwardSumOn_covChannelOn_zero_of_factorsThroughBin_right
    (S := S) (bin := bin) (U := U) (V := V) hV

theorem pushforwardSumBulk_covChannelBulk_zero_of_factorsThroughBin_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    pushforwardSumBulk slack ζ T bin (covChannelBulk slack ζ T bin U V) = 0 := by
  unfold pushforwardSumBulk covChannelBulk
  rw [covChannelOn_zero_of_factorsThroughBin_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hU]
  simp [pushforwardSumOn, densityPushforwardSumOn]

theorem pushforwardSumBulk_covChannelBulk_zero_of_factorsThroughBin_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    pushforwardSumBulk slack ζ T bin (covChannelBulk slack ζ T bin U V) = 0 := by
  unfold pushforwardSumBulk covChannelBulk
  rw [covChannelOn_zero_of_factorsThroughBin_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hV]
  simp [pushforwardSumOn, densityPushforwardSumOn]

theorem paperEstimatorBulk_zero_of_factorsThroughBin_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    paperEstimatorBulk slack ζ T bin U V = 0 := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_zero_of_factorsThroughBin_left'
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hU

theorem paperEstimatorBulk_zero_of_factorsThroughBin_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    paperEstimatorBulk slack ζ T bin U V = 0 := by
  unfold paperEstimatorBulk
  exact paperEstimatorOn_zero_of_factorsThroughBin_right'
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hV

theorem paperEstimatorOn_zero_of_factorsThroughBin_left (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    paperEstimatorOn S bin U V = 0 := by
  unfold paperEstimatorOn
  apply Finset.sum_eq_zero
  intro b hb
  rw [covOn_on_fiberIn_of_factorsThroughBin_left S bin U V b hU]
  simp

theorem paperEstimatorOn_zero_of_factorsThroughBin_right (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    paperEstimatorOn S bin U V = 0 := by
  unfold paperEstimatorOn
  apply Finset.sum_eq_zero
  intro b hb
  rw [covOn_on_fiberIn_of_factorsThroughBin_right S bin U V b hV]
  simp

theorem deltaPaperEstimatorOn_zero_of_factorsThroughBin_left
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    deltaPaperEstimatorOn S bin U V = 0 := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_zero_of_factorsThroughBin_left S bin U V hU,
    paperEstimator_zero_of_factorsThroughBin_left bin U V hU]
  ring

theorem deltaPaperEstimatorOn_zero_of_factorsThroughBin_right
    (S : Finset α) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    deltaPaperEstimatorOn S bin U V = 0 := by
  unfold deltaPaperEstimatorOn
  rw [paperEstimatorOn_zero_of_factorsThroughBin_right S bin U V hV,
    paperEstimator_zero_of_factorsThroughBin_right bin U V hV]
  ring

theorem deltaPaperEstimatorBulk_zero_of_factorsThroughBin_left
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hU : FactorsThroughBin bin U) :
    deltaPaperEstimatorBulk slack ζ T bin U V = 0 := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_zero_of_factorsThroughBin_left
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hU

theorem deltaPaperEstimatorBulk_zero_of_factorsThroughBin_right
    (slack : α → ℚ) (ζ T : ℚ) (bin : α → β) (U V : α → ℚ)
    (hV : FactorsThroughBin bin V) :
    deltaPaperEstimatorBulk slack ζ T bin U V = 0 := by
  unfold deltaPaperEstimatorBulk
  exact deltaPaperEstimatorOn_zero_of_factorsThroughBin_right
    (S := bulkSupportSlack slack ζ T) (bin := bin) (U := U) (V := V) hV

end FiniteStats

end SCT.CJBridge
