# Changelog

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
