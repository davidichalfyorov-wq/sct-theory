# Path combinatorics detect spacetime curvature on causal sets

## Technical note — David Alfyorov, March 2026

---

## Abstract

We introduce path_kurtosis, an observable on causal sets that detects spacetime curvature through the statistical organization of directed path counts in the Hasse diagram. On pp-wave spacetimes — where all polynomial scalar curvature invariants vanish and the Benincasa-Dowker action is silent — path_kurtosis gives a stable, statistically significant response proportional to ε². We establish a scaling law E[Δκ] = A_eff × ε² × √N × C₂ with A_eff of order 0.06, verified computationally across N = 2000–10,000 at ε = 2–3. An exact geodesic-derived causal relation for pp-wave spacetimes replaces the commonly used midpoint surrogate, correcting a systematic underestimate of approximately 19% at N = 5000. A battery of eight robustness tests (with acceptance criteria specified before computation) demonstrates that the signal survives coarse-graining, thickening, and structured decimation, while revealing intrinsic boundary sensitivity and mesoscopic structure beyond the perturbation profile.

---

## 1. Introduction

The Benincasa-Dowker (BD) action [arXiv:1209.0777] discretizes ∫R on causal sets, recovering the Ricci scalar in the continuum limit. No established discrete counterpart exists for the Weyl tensor.

This note reports a candidate observable, path_kurtosis, that detects curvature information beyond scalar invariants. On pp-wave spacetimes — VSI (Vanishing Scalar Invariants), hence BD = 0 — path_kurtosis gives a nonzero, stable, ε²-proportional signal with Cohen's d = 9–14 at N = 10,000.

## 2. Definition

Let C = (X, ≺) be a causal set obtained by Poisson sprinkling N points into a 4-dimensional causal diamond of duration T. The Hasse diagram H(C) is the transitive reduction of the causal order. For each element x ∈ X, define:

- p_down(x) = number of directed paths from any source to x in H(C)
- p_up(x) = number of directed paths from x to any sink in H(C)
- Y(x) = log₂(p_down(x) × p_up(x) + 1)

**Definition.** path_kurtosis(C) = excess kurtosis of {Y(x) : x ∈ X}, i.e.,

κ = μ₄/σ⁴ − 3

where μ₄ = E[(Y − μ)⁴] and σ² = Var(Y).

**Novelty.** No prior study of this specific observable was found in searches across arXiv and INSPIRE-HEP, including comparison with Roy-Sinha-Surya chain abundances, Johnston path-sum propagators, Ilie-Thompson-Reid longest paths, and Gudder maximal path counts. The combination Y(x) = log₂(p_down × p_up + 1) and its kurtosis appear to be new.

## 3. Method: Common Random Numbers (CRN)

To isolate the curvature effect from Poisson sampling noise, we use paired experiments:

1. Sprinkle N points into a causal diamond (seed s).
2. Compute the Hasse diagram under **flat** Minkowski causal relation → κ₀.
3. Compute the Hasse diagram under **curved** causal relation (same points) → κ_ε.
4. The CRN difference Δκ = κ_ε − κ₀ cancels seed-to-seed scatter.

**Necessity.** Without CRN, the inter-realization kurtosis scatter is ±0.10 — much larger than the perturbative signal at ε ≤ 3. The CRN pairing reduces the effective noise by an order of magnitude.

## 4. Exact pp-wave causal relation

The pp-wave metric in Brinkmann coordinates is

ds² = −du dv + dx² + dy² + (ε/2)(x² − y²) du²

where u = t + z, v = t − z. The standard midpoint surrogate approximates the causal condition as

s²_Mink − (ε/2)(x_m² − y_m²)(Δu)² > 0

We derived the **exact** geodesic-based causal condition: A ≺ B if and only if U > 0 and V ≥ V_needed, where U = Δu, V = Δv, and for ε > 0 with ω = √(ε/2):

V_needed = ω [(x_A² + x_B²)cosh(ωU) − 2x_A x_B] / sinh(ωU)
         + ω [(y_A² + y_B²)cos(ωU) − 2y_A y_B] / sin(ωU)

The midpoint surrogate has a pairwise error of order O(ε), not O(ε²):

V_needed − V_mid = (εU/24)(Δx² − Δy²) + O(ε²)

This error averages to zero by x ↔ y symmetry, so the ensemble-level bias begins at O(ε²). Nevertheless, it systematically underestimates A_eff by approximately 19% at N = 5000.

**Validity.** The formula is valid when ωU < π, which holds for all pairs in a T = 1 diamond with |ε| ≤ 5 (since ωU ≤ √(5/2) ≈ 1.58 < π). Sorting by t remains correct for |ε| < 8.

## 5. Scaling law

The perturbative Taylor expansion of kurtosis under the CRN pairing, verified by independent computational checks at multiple N and ε values, yields the scaling law:

**E[Δκ] = A_eff × ε² × √N × C₂**

where C₂ = ⟨f²⟩_diamond = T⁴/1120 (analytically derived), f(x,y) = (x² − y²)/2 is the pp-wave profile, and A_eff is a dimensionless coefficient.

### 5.1. Why ε²

The ensemble average of any relabeling-invariant scalar on a symmetric diamond satisfies ⟨O(ε)⟩ = ⟨O(−ε)⟩ by the x ↔ y symmetry of the diamond combined with f being odd under x ↔ y. Therefore Δκ is even in ε, and the leading term is O(ε²).

### 5.2. A_eff ensemble

A_eff values extracted from CRN ensembles (M = 15–30 seeds per cell). The causal relation used is noted for each row:

| N | ε = 2.0 | ε = 3.0 | Causal relation |
|---|---------|---------|-----------------|
| 2000 | 0.069 ± 0.019 | 0.074 ± 0.012 | midpoint |
| 3000 | 0.052 ± 0.016 | 0.048 ± 0.011 | midpoint |
| 5000 | 0.051 ± 0.015 | 0.068 ± 0.008 | exact |
| 5000 | 0.043 ± 0.013 | 0.057 ± 0.008 | midpoint |
| 10000 | 0.076 ± 0.008 | 0.075 ± 0.005 | midpoint |

*[Figure 1: A_eff vs N at ε = 3, with exact and midpoint values distinguished.]*

The scatter across N (0.048–0.076 at ε = 3) exceeds the per-cell standard errors and is not yet fully understood. Possible sources: slow finite-size corrections, residual midpoint bias at N = 2000–3000, or genuine N-dependence of A_eff. At this stage, A_eff is of order 0.06 but should not be quoted as a sharp universal constant.

The exact causal relation increases A_eff by approximately 19% relative to midpoint at N = 5000 (ratio 1.19 at both ε = 2 and ε = 3). The N = 10000 values (midpoint) would increase correspondingly if recomputed with the exact relation.

### 5.3. Key theorems

1. **Exact cancellation** (β_int + β_bdy = 0): In an infinite translation-invariant volume, the link sensitivity β decomposes into interior and boundary contributions that exactly cancel. The net signal in a finite diamond is the boundary residual.

2. **Double-counting identity** (Σ_v G_x(v) = E[|γ|]): The sum of the path-occupation kernel equals the expected path length, bounded by the longest chain H ~ N^{1/4}. This constrains the perturbative coefficient.

3. **Conjectured factorization** (A_eff = β² × Ξ₄): Numerical evidence suggests the effective coefficient factorizes into the link sensitivity squared (β ≈ 0.30, hence β² ≈ 0.09) and a flat-space structural constant Ξ₄ ≈ 0.7 (with ~50% uncertainty given the scatter in A_eff). This factorization is not proven; the analytical derivation of Ξ₄ remains open.

## 6. K = 0: Detection beyond scalar curvature invariants

The pp-wave spacetime is VSI (Vanishing Scalar Invariants): all polynomial scalar curvature invariants — including the Kretschner scalar K = R_abcd R^abcd — vanish identically. The Ricci scalar R = 0, so the Benincasa-Dowker action gives zero.

However, the Bel-Robinson superenergy W = E² + B² = ε² is nonzero (where E and B are the electric and magnetic parts of the Weyl tensor). path_kurtosis detects a signal proportional to ε², which numerically equals W for the specific case of pp-wave.

**Caveat.** The coincidence Δκ ∝ ε² = W holds only because both quantities happen to scale as ε² for pp-wave by the quadrupole structure of the profile function. This does not establish that path_kurtosis measures W in general. Universality across different vacuum spacetimes (e.g. Schwarzschild) has not been tested. The correct interpretation at this stage is that path_kurtosis detects order-deformation through path-structure reorganization on pp-wave backgrounds.

## 7. Robustness battery

Eight tests were run with acceptance criteria specified before computation (N = 5000, ε = 3, M = 8–10 seeds, exact causal relation).

### Q1. Coarse-graining stability

| Test | Deformation | Signal retention | Verdict |
|------|-------------|-----------------|---------|
| Edge deletion 20% | Remove 20% of Hasse links | 0.80 | ROBUST |
| Poisson thinning 30% | Remove 30% of elements, rebuild Hasse | 0.76 (median) | ROBUST |
| Thickening 5% | Add 5% "almost-direct" links | 1.01 [0.98, 1.03] | ROBUST (no significant effect) |
| Layer decimation | Remove one depth-residue class (20%) | 0.84 (mean), 0.50 (worst) | ROBUST |

*[Figure 2: Signal retention ratio vs deformation strength for all four Q1 tests.]*

The signal degrades gracefully under all four deformation types. Thickening has no statistically significant effect (R = 1.01, 95% CI [0.98, 1.03]), showing the observable does not depend on the exact minimality of the transitive reduction.

### Q2. Boundary sensitivity

Boundary exclusion amplifies the signal: restricting to the 80% most interior elements increases Δκ by a factor of 3.4× (after correcting for the normalization artifact from empirical C₂ vs fixed C₂). This amplification does not diminish with N (tested at N = 2000, 3000, 5000), indicating intrinsic rather than finite-size boundary dependence.

The exact cancellation theorem (β_int + β_bdy = 0 in infinite volume) provides a theoretical framework: the link sensitivity β is a boundary residual. Whether this extends directly to the kurtosis signal (which involves higher-order moments, not just β) requires further analysis. In a finite diamond, the observable definition must specify the windowing prescription.

### Q3. Distributional structure

| Statistic | Cohen's d | Type |
|-----------|-----------|------|
| Kurtosis | +5.9 | Shape |
| Skewness | −5.9 | Shape |
| Median | +3.5 | Location |
| IQR | −2.2 | Scale |
| MAD | −2.5 | Scale |

Shape statistics (kurtosis, skewness; d ≈ 6) are more sensitive than location/scale statistics (d ≈ 2–3.5), but all five statistics detect the perturbation under CRN. The effect is predominantly a shape response, though location and scale shifts are also present at lower significance.

### Q4. Mesoscopic localization

After conditioning on the perturbation profile f(x,y) via 10-quantile binning, graph coordinates (total Hasse degree, flat path centrality Y₀) explain an additional 17% of the residual variance in δY (stratified permutation test, p = 0.005 for all seeds). This indicates structure in the response beyond the quadrupole perturbation profile.

**Caveat.** This test conditions on f = (x² − y²)/2 alone, not on the full spatial position (t, x, y, z). The residual localization by degree and Y₀ may therefore reflect spatial structure orthogonal to the quadrupole pattern (e.g., radial or temporal dependence) rather than purely graph-theoretic structure. A follow-up test conditioning on full spatial position is needed to distinguish these possibilities.

The signal is carried by approximately 3 out of 10 path-centrality strata (effective number of contributing deciles m_eff = 3.3 median, jackknife max|J| = 1.9 median). It is localized rather than globally distributed.

*[Figure 3: Jackknife drop J_q across 10 Y₀-deciles, showing which strata carry the signal.]*

## 8. Per-element mechanism

The per-element perturbation is tightly coupled to the metric profile: corr(δY, f) ≈ −0.9 (ranging from −0.88 to −0.92 across seeds). The extreme contributors sit at the temporal midplane (t ≈ 0.5) and outer radial shell (r ≈ 0.9–1.0), where the pp-wave perturbation flips causal relations: elements on the f > 0 side lose connections (Y drops from ~7 to ~1), while elements on the f < 0 side gain connections (Y jumps from ~1 to ~7).

The kurtosis is sensitive to this spatially organized redistribution of connectivity — specifically, to the asymmetric tails generated by the connection flipping.

## 9. Implementation

All computations use a bitset-packed Hasse diagram algorithm with O(N² × N/64) time complexity, implemented in the `sct_tools.hasse` module. Performance: N = 5000 in 4.4s, N = 10000 in 10.4s per CRN trial (single core, Intel i9-12900KS).

The exact pp-wave causal relation adds approximately 20% overhead compared to the midpoint surrogate, while correcting a systematic underestimate in A_eff of approximately 19% at N = 5000 (measured at M = 15; the correction magnitude at other N values has not yet been systematically measured).

## 10. Errors caught and corrected

Three errors were found during the analysis and corrected before reporting:

1. **Normalization artifact.** Per-element kurtosis decomposition using different σ for flat and pp-wave distributions created a spurious "edges oppose" pattern in temporal stratification. Corrected by using a single baseline σ.

2. **Regime-dependent tail claim.** A "tail mechanism" interpretation based on ε = 5 data (where |z| > 2.5σ tail fraction shifts by 8.5%) does not hold at perturbative amplitudes (ε = 2: shift < 0.5%). The claim was retracted.

3. **Boundary windowing normalization.** Normalizing by empirical C₂ of each window subset inflated A_eff at small windows (interior elements have smaller |f|, hence smaller C₂). Corrected by reporting with both empirical and fixed C₂.

## 11. Open problems

1. **Analytical derivation of Ξ₄.** The flat-space structural constant Ξ₄ ≈ 0.72 in the factorization A_eff = β² × Ξ₄ requires the two-replica path kernel E[G_x(u)G_x(v)], which is a new mathematical object on Poisson DAGs.

2. **Universality across vacuum families.** The observable has been tested only on pp-wave spacetimes. Testing on Schwarzschild (a Petrov type D vacuum with nonzero scalar invariants) is the natural next step. The scaling may differ (ε vs ε² leading term) due to the absence of the x ↔ y symmetry that ensures ε²-scaling for pp-wave.

3. **Boundary correction.** The intrinsic boundary sensitivity must be resolved, either by defining a boundary-corrected version of path_kurtosis or by specifying a canonical windowing prescription that survives the continuum limit.

4. **Exact midpoint correction at N = 10000.** The N = 10000 ensemble was computed with the midpoint surrogate. Re-running with the exact causal relation would provide the cleanest A_∞ estimate.

5. **Continuum limit.** A proof (or disproof) that A_eff converges to a finite nonzero limit as N → ∞ at fixed ε would place the observable on firm theoretical footing.

---

**Computational resources.** Intel i9-12900KS (16C/24T), 64 GB DDR5, RTX 3090 Ti. Benchmark: one CRN trial at N = 5000 takes approximately 8s with exact causal relation (two Hasse builds + path counting, single core). With the midpoint surrogate: approximately 4.4s.

**Code availability.** The `sct_tools.hasse` module (bitset Hasse construction, path counting, CRN trials, exact pp-wave causal relation) is available as part of the SCT Theory repository.

---

## Figures

All figures are in `docs/figures/technical_note/`.

1. `fig1_aeff_vs_N` — A_eff vs N at ε = 2, 3. Exact (red squares) and midpoint (blue circles) distinguished.
2. `fig2_eps2_scaling` — Δκ vs ε² at N = 5000, with linear fit through ε = 2, 3 (perturbative window).
3. `fig3_edge_deletion` — Signal retention ratio vs fraction of Hasse edges deleted.
4. `fig4_boundary_collapse` — Left: A_eff(fixed C₂) vs window α. Right: raw Δκ vs α showing 3.4× amplification.
5. `fig5_scatter_dY_vs_f` — Per-element δY vs f(x,y), colored by boundary slack. Single seed, N = 5000, ε = 3.
6. `fig6_statistics_ladder` — Cohen's d for five summary statistics, colored by type (location/scale/shape).
