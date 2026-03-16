# MR-6: Curvature Expansion Convergence -- Handoff Certificate

**Task:** MR-6 (Curvature Expansion Convergence)
**Agent:** MR6-V (Verification)
**Date:** 2026-03-15
**Status:** CERTIFIED

---

## Result

The Seeley-DeWitt curvature expansion of the spectral action is **ASYMPTOTIC** (Gevrey-1 class, zero radius of convergence).

## Key Findings

### Expansion Classification
| Property | Value |
|----------|-------|
| Expansion type | ASYMPTOTIC |
| Gevrey class | 1 (factorial growth) |
| Radius of convergence | 0 (zero) |
| Borel summable (on S^4) | Likely (positive Borel radius ~84) |
| Laplace representation | EXACT (verified to 168+ digits) |

### Seeley-DeWitt Coefficients on S^4 (Dirac, a=1)
| k | a_{2k} | Source |
|---|--------|--------|
| 0 | 2/3 = 0.66667 | Integral term (Vassilevich convention) |
| 1 | -2/3 = -0.66667 | Integral term (Einstein-Hilbert) |
| 2 | 11/90 = 0.12222 | Bernoulli correction |
| 3 | 0.01640 | Bernoulli correction |
| 4 | 0.00542 | Bernoulli correction |
| 5 | 0.00261 | Bernoulli correction |
| 6 | 0.00159 | Bernoulli correction |
| 7 | 0.00116 | Bernoulli correction |

Growth ratios |a_{2k}/a_{2(k-1)}| exceed 1 at k=10 and grow linearly thereafter, confirming factorial (Gevrey-1) divergence.

### Optimal Truncation K*
| Lambda^2 a^2 | K* | Min |S_exact - S_trunc| |
|--------------|----|-----------------------------|
| 1 | 21 | 8.6e-03 |
| 10 | 12 | 2.9e-05 |
| 100 | 9 | 2.7e-07 |
| 1000 | 8 | 2.7e-09 |

### Reliability Domain
| Regime | R/Lambda^2 | K* | Error (log10) |
|--------|-----------|-----|---------------|
| Solar system | 10^{-50} | 9 | < -999 |
| Neutron star | 10^{-20} | 17 | < -999 |
| Inflation | 10^{-10} | 29 | < -999 |
| Near-Planck | 0.1 | 30 | -85 |
| Planck | 1 | 30 | -57 |
| Strong curvature | 10 | 30 | -29 |

## Borel Summability

Borel radius R_B ~ 84 (related to 2*(2*pi)^2 from the Bernoulli number singularity structure). The Borel transform of the tail coefficients converges in a disc of this radius. The Borel sum does not recover the full exact action because the leading Lambda^4 and Lambda^2 terms (from a_0, a_2) dominate and are not part of the asymptotic tail.

Status: Borel summable on S^4 (likely); general manifolds unknown.

## Laplace Representation

The exact spectral action S = Tr(f(D^2/Lambda^2)) was verified to agree with the heat-trace-based Laplace representation to:
- 170.5 digits at la2 = 10
- 168.8 digits at la2 = 100
- 167.9 digits at la2 = 1000

This provides a rigorous non-perturbative definition of the spectral action that bypasses the divergent SD expansion.

## Normalization Issue: RESOLVED

### Problem
Agent D's original Bernoulli formula computed SD coefficients via the formal identity:
```
sum_{m=0}^inf m^q exp(-s m^2) "=" sum_j (-s)^j/j! * zeta(-q-2j)
```
This formally replaces divergent sums by zeta-function values. However, it captures only the Bernoulli correction terms in the Euler-Maclaurin formula, missing the INTEGRAL TERM which produces the leading coefficients a_0 and a_2.

### Root Cause
The Euler-Maclaurin formula for the heat trace sum has three parts:
1. **Integral term**: int_0^inf (m^3-m) exp(-s m^2) dm = 1/(2s^2) - 1/(2s)
2. **Boundary term**: f(0) = 0 (vanishes since 0^3-0 = 0)
3. **Bernoulli corrections**: formal series with B_{2k} coefficients

The zeta-regularization approach captures only part (3). Parts (1) gives:
- a_0 = (4/3) * (1/2) = 2/3 (from the s^{-2} term)
- a_2 = (4/3) * (-1/2) = -2/3 (from the s^{-1} term)

### Resolution
The corrected `sdw_coefficients_bernoulli()` now computes:
- k=0: a_0 = 2/3 * a^4 (integral term)
- k=1: a_2 = -2/3 * a^2 (integral term)
- k >= 2: a_{2k} from Bernoulli formula with shifted index j = k-2

### Verification
a_0 = 2/3 confirmed by:
1. Vassilevich formula: (4pi)^{-2} Vol(S^4) tr(Id) = (1/(16pi^2)) * (8/3)pi^2 * 4 = 2/3
2. Numerical heat trace: lim_{t->0+} t^2 K(t) = 0.66660... -> 2/3
3. Polynomial fitting (sdw_coefficients_numerical): 0.66667 to 15+ digits

a_2 = -2/3 confirmed by:
1. Vassilevich formula: (4pi)^{-2} (1/6) int tr(6E+R) with E=-3, R=12, tr=4
2. Numerical extraction: -0.66665... -> -2/3

### Impact
- Growth rate analysis: UNAFFECTED (Gevrey-1 classification depends on Bernoulli tail ratios)
- Truncation errors: NOW CORRECT (previously overestimated by orders of magnitude)
- Laplace verification: UNAFFECTED (compares two independent exact sums)
- Borel radius: SLIGHTLY CHANGED (~84 vs ~77, from using corrected tail coefficients)
- Qualitative conclusions: UNCHANGED (asymptotic, Gevrey-1, R=0)

## Implications for SCT

1. **The curvature expansion does not converge.** The SD series is a useful asymptotic approximation, not a convergent representation.

2. **The SCT framework is unaffected.** The theory is defined by:
   - The Laplace representation Tr(f(D^2/Lambda^2)), which is exact.
   - The entire form factors F_1(z), F_2(z), which converge for ALL z.
   - The curvature expansion is a derived approximation, not the definition.

3. **The a_4-level truncation is reliable for all sub-Planckian physics.** For R/Lambda^2 << 1, the truncation error is exponentially small.

4. **Two distinct expansions must not be confused:**
   - Form factor expansion (in z = Box/Lambda^2): CONVERGENT (entire functions, proved in NT-2)
   - Curvature expansion (in R/Lambda^2): ASYMPTOTIC (Gevrey-1, proved here in MR-6)

## Tests

74 PASS (0 FAIL) in `test_mr6_convergence.py`

## Figures

4 publication-quality PDFs in `analysis/figures/mr6/`:
1. `mr6_sdw_growth.pdf` -- SD coefficient growth with factorial comparison
2. `mr6_s4_convergence.pdf` -- Truncation error vs K for multiple la2 values
3. `mr6_borel_plane.pdf` -- Borel transform along real axis
4. `mr6_reliability_domain.pdf` -- Optimal truncation error vs R/Lambda^2

## Deliverables

| Deliverable | Location |
|-------------|----------|
| Main script | `analysis/scripts/mr6_convergence.py` |
| DR verification script | `analysis/scripts/mr6_convergence_dr.py` |
| Test file | `analysis/sct_tools/tests/test_mr6_convergence.py` |
| Results JSON | `analysis/results/mr6/mr6_convergence_results.json` |
| DR results JSON | `analysis/results/mr6/mr6_convergence_dr_results.json` |
| Figures (4 PDFs) | `analysis/figures/mr6/` |
| Literature review | `docs/reviews/MR6_L_literature.md` |
| Literature audit | `docs/reviews/MR6_LR_audit.md` |
| Derivation report | `docs/reviews/MR6_D_derivation.md` |
| Derivation review | `docs/reviews/MR6_DR_review.md` |
| Verification report | `docs/reviews/MR6_V_verification.md` |
| This handoff | `theory/consistency-checks/MR6_convergence_handoff.md` |
