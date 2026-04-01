# Changelog

## [3.2] - 2026-04-01

### Paper 7: Factor-4 Definitive Investigation

**The factor 4 in the CJ bridge formula is now fully characterised.**

The coefficient C_0 = 32pi^2/(3*9!*45) decomposes as:

```
Factor 4 = 2 (algebraic) * 2 (dynamical)
```

- **Algebraic 2**: from delta_Y^2 = 2*g_s^2 (the CJ observable depends only on the symmetric channel g_s = (g_- + g_+)/sqrt(2) and is blind to the antisymmetric channel g_a)
- **Dynamical 2**: the symmetric-channel covariance M_ss/(C_AN * N^{8/9} * E^2 * T^4) converges to 2.0 as N -> infinity

**Symmetric-channel decomposition (new Section 5.5):**

| N      | M_ss  | M_aa  | M_sa   | R_full |
|--------|-------|-------|--------|--------|
| 1,000  | 2.008 | 0.898 | 0.100  | 1.00   |
| 3,000  | 1.974 | 0.592 | 0.162  | 0.99   |
| 5,000  | 2.011 | 0.565 | 0.067  | 1.01   |
| 8,000  | 2.018 | 0.539 | 0.056  | 1.01   |
| 10,000 | 2.059 | 0.522 | -0.013 | 1.03   |

Three convergence properties verified:
1. M_ss -> 2 (the dynamical 2, confirming factor 4)
2. M_sa -> 0 (past/future symmetry restored in continuum limit)
3. M_aa -> 0.52 (persistent nonzero; equalization hypothesis permanently excluded)

**Six definitive findings from the investigation:**
1. |X|-weighted past-future correlation rho_eff ~ 0.5 (not 1.0)
2. One-leg CJ is NEGATIVE at all tested N (the product p_down*p_up is constitutive)
3. Interval-counting CJ = 0 (CJ is fundamentally a Hasse-path phenomenon)
4. CJ kernel K(s) ~ [s(1-s)]^18 (not k_4(s)*k_4(1-s); the five-factor decomposition is an integral identity, not pointwise)
5. Coefficient is score-dependent ((p_down*p_up)^{1/4} gives ~7x different C_0)
6. The "+1" correction in log_2(p_down*p_up+1) is negligible (<0.001% at N >= 5000)

**Extended epsilon range:**
- Six values eps = 0.5, 1, 2, 3, 5, 10 (two orders of magnitude in E^2)
- chi^2/dof = 0.48 (5 dof), confirming CJ proportional to E^2

**Statistical diagnostics:**
- Shapiro-Wilk normality test: W = 0.988, p = 0.88 (M=50 trials)
- Bootstrap 95% CI for R: [0.89, 1.09] (contains 1.0)

**Open problem sharpened:** Prove M_ss -> 2*C_AN as N -> infinity on a Poisson DAG in a d=4 Alexandrov diamond. This requires a theory of Hasse-path occupation kernels that does not yet exist in the literature.

**Fixed broken references:** prop:factor4 -> rem:factor4, sec:outlook -> sec:open

### New analysis scripts (6)

- `factor4_decisive_test.py`: |X|-weighted correlation rho_eff at N=1000-5000
- `factor4_kernel_profile.py`: CJ kernel K(s) profile, Hasse vs interval comparison
- `factor4_score_variants.py`: 5 link-score variants (log, root, one-leg)
- `cj_three_kernel_decomposition.py`: full three-kernel decomposition N=1000-10000, M=30
- `paper7_mini_tests.py`: Tier-A mini-test battery (de Sitter, epsilon range, normality, interval)
- `check_paper7.py`: pre-push verification (references, figures, bibliography, AI markers)

### New data files (2)

- `analysis/fnd1_data/cj_three_kernel_decomposition.json`: 180 MC trials, 6 N values
- `analysis/fnd1_data/paper7_mini_tests.json`: mini-test results

### Previous v3.1 changes (2026-04-01, same day)

- Lorentz boost test: CJ proportional to E^2, blind to B^2 (<1%)
- T^4 scaling test: ratio/T^4 = 0.998 at T=0.70
- Factor-4 characterised as "indivisible coefficient"
- C_0 confirmed to 0.05 sigma at M=50, N=5000
- Paper 7 submitted to Classical and Quantum Gravity
- Zenodo DOI: 10.5281/zenodo.19364212

## [3.0] - 2026-03-31

### Paper 7: CJ Bridge Formula

**New paper:** "Weyl curvature from the Hasse diagram: a parameter-free bridge formula for causal sets" (34 pages, 8 figures, 105 Lean theorems).

First parameter-free relationship between a discrete causal-set observable and a Weyl-curvature invariant:

```
CJ = 4 * (8/3) * (1/9!) * (8pi/15) * (pi/24) * N^{8/9} * E^2 * T^4
```

Each factor has a distinct geometric origin: the two-leg structure of the link score (4), the squared Benincasa-Dowker normalisation (8/3), the nine-simplex beta overlap (1/9!), the angular average of the squared tidal deformation (8pi/15), and the four-dimensional diamond volume (pi/24). The combined coefficient C_0 = 32pi^2/(3*9!*45) ~ 6.45e-6 contains no continuous free parameters.

**Numerical verification:**
- Ratio data/prediction R = 1.016 +/- 0.015 across N = 500-15000 (pp-wave, 8 data points)
- Curvature-strength independence confirmed (CV 1.4% for epsilon >= 3)
- De Sitter null test: CJ = 0 exactly (200 seeds, 4 T values)
- Kottler Ricci-reduction: 22% suppression consistent with R*E^2 cross-term
- Polarisation independence: CJ_cross/CJ_plus = 0.98 +/- 0.13 with exact predicate

**Formal verification:**
- 105 sorry-free Lean 4 theorems across 6 proof files
- General-d identity (d!)^2 C(2d,d)(2d+1) = (2d+1)! proven by induction
- Beta chain B(5,5) = Gamma(5)^2/Gamma(10) = (4!)^2/9! fully formalised
- Angular-volume identity pi^2/45 = (8pi/15)(pi/24) verified
- Three independent proof backends: local Lean, Aristotle, WSL SciLean

**Negative results documented:**
- 5 failed approaches with detailed analysis
- 30+ closed spectral/QFT routes (Appendix D)
- 3 structural obstructions closing entire families of approaches
- d = 2 test: formula fails (alpha_meas = 1.53 vs predicted 0.80)

**Open problems:**
- Conditions A and B remain unproven
- N^{8/9} exponent is empirical (alpha_meas = 0.955 +/- 0.027)
- Kottler cross-term coefficient not derived quantitatively
- Extension to non-vacuum spacetimes

### New Lean proofs

Six new proof files in `theory/lean/proofs/`:
- `cj_bridge_local.lean` (28 theorems): all d=4 identities
- `cj_bridge_general_d.lean` (14 theorems): general-d beta overlap
- `cj_bridge_angular.lean` (13 theorems): angular-volume factorisation
- `cj_bridge_beta_integral.lean` (6 theorems): beta chain, positivity
- `cj_bridge_rigorous.lean` (6 theorems): Gamma/Beta connection
- `cj_bridge_*_aristotle.lean` (38 theorems): dual verification

### New figures

Eight figures for Paper 7:
- `fig_cj_ratio.pdf`: ratio R_N vs N (parameter-free prediction)
- `fig_cj_nscaling.pdf`: log-log N-scaling
- `fig_cj_epsilon.pdf`: epsilon-independence plateau
- `fig_cj_diagnostics.pdf`: de Sitter / Kottler / polarisation summary
- `fig_cj_derivation.pdf`: coefficient decomposition schematic
- `fig_cj_validity.pdf`: domain of validity
- `fig_cj_residuals.pdf`: residuals (R_N - 1) vs N
- `fig_cj_stratification.pdf`: 45-stratum diamond stratification schematic

### Infrastructure

- Updated `.gitignore` to exclude AI session artifacts
- Updated roadmap chart (41 tasks: 26 complete, 5 conditional, 6 negative, 2 pending)

## [2.0] - 2026-03-20

Papers 1-4, 6 published with Zenodo DOIs. Repository v2.0 archived.
Full 8-layer verification pipeline. 4445 pytest tests. 71 TeX targets.

## [1.0] - 2026-03-12

Initial release. Papers 1-4 complete. Core form factors and field equations derived.
