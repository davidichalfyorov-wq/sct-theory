# GZ: Entire Part g(z) -- Handoff Certificate

**Task:** GZ (Entire Part of ML Expansion)
**Date:** 2026-03-15
**Status:** PROVEN
**Pipeline:** GZ-L -> GZ-LR -> GZ-D -> GZ-DR -> GZ-V

---

## Result

The entire part g_A(z) in the Mittag-Leffler expansion of the SCT physical propagator H(z) = 1/(z * Pi_TT(z)) is a constant:

    g_A(z) = -c_2 = -13/60 = -2 * alpha_C     for all z in C.

**Corollary (Sum Rule):** Sum_{n >= 1} R_n / z_n = c_2 = 13/60.

---

## Key Quantitative Results

| Quantity | Value |
|----------|-------|
| g_A(z) | -13/60 (constant, proven) |
| c_2 = 2*alpha_C | 13/60 = 0.21667 |
| Sum R_n/z_n (16 poles) | 0.21655 (99.95% of target) |
| Max deviation from constancy | 4.48e-07 (at z = 3+15i, truncation) |
| Propagator reconstruction error | 7.97e-08 (at z = 3+i) |
| Genus of Pi_TT | p = 1 |
| Order of Pi_TT | rho = 1, type sigma = 1/4 |
| g_B(z) for 1/Pi_TT | NOT constant (linear in z) |

---

## Proof Method

**Route A (Genus Bound + Asymptotic Limit):**
1. z*Pi_TT(z) is entire of order 1, genus 1.
2. By Hadamard-Mittag-Leffler theory, g_A(z) = a + bz (at most linear).
3. On positive real axis: H(x) -> 0, S(x) -> Sum R_n/z_n (by dominated convergence). Since g_A(x) must be bounded, b = 0.
4. g_A(0) = lim_{z->0}[1/(z*Pi_TT) - 1/z] = -c_2 = -13/60.
5. Therefore g_A(z) = -c_2 for all z in C.

**Independent confirmation (GZ-DR, Method B):**
- 46-point polynomial fitting at dps=200: |a_1| = 1.3e-08 (truncation noise)
- Derivative vanishing: |g_A'(z)| < 3e-08 at 6 test points
- Cauchy integral: a_1 = -2.8e-08 (consistent with zero)

---

## Verification Summary

| Layer | Tests | Result |
|-------|-------|--------|
| 1. Analytic (g_A(0) = -c_2, sum rule, genus) | 8 | PASS |
| 2. Numerical (19 points, 100-digit) | 18 | PASS |
| 3. Cross-consistency (FK, CL, OT) | 6 | PASS |
| 4. Physical (reconstruction, UV/IR limits) | 5 | PASS |
| 5. Edge cases (near poles, complex, large) | 6 | PASS |
| 6. Regression (canonical values, catalogue) | 12 | PASS |
| **Total** | **55** | **ALL PASS** |
| Full regression suite | 2825+ | ALL PASS |

---

## Physical Significance

1. **Pure pole decomposition:** The SCT graviton propagator is entirely determined by its poles (graviton + ghost tower) plus a single constant contact term. No smooth entire-function background exists.

2. **Contact term:** g_A = -c_2 = -13/60 is a momentum-independent delta-function interaction in position space, absorbable into a counterterm.

3. **Lee-Wick connection:** Extends the pure-pole structure of finite-pole Lee-Wick models to the infinite-pole SCT case, with a nonzero constant arising from the SM Weyl-squared coefficient.

4. **Unitarity simplification:** The spectral function Im[G(s)] is a pure sum of delta/PV distributions, strengthening the OT optical theorem result.

5. **Sum rule:** Sum R_n/z_n = c_2 = 13/60 provides an independent constraint on the ghost catalogue.

---

## Artifacts

| File | Path |
|------|------|
| Computation script | analysis/scripts/gz_entire_part.py |
| Test suite (55 tests) | analysis/sct_tools/tests/test_gz_entire_part.py |
| Results JSON | analysis/results/gz/gz_entire_part_results.json |
| LaTeX document | theory/consistency-checks/GZ_entire_part.tex |
| Literature report | docs/reviews/GZ_L_literature.md |
| Literature audit | docs/reviews/GZ_LR_audit.md |
| Derivation report | docs/reviews/GZ_D_derivation.md |
| Re-derivation report | docs/reviews/GZ_DR_review.md |
| Verification report | docs/reviews/GZ_V_verification.md |
| This handoff | theory/consistency-checks/GZ_entire_part_handoff.md |

---

## Scope

**Covers:**
- Entire part of H(z) = 1/(z*Pi_TT(z)) -- the physical propagator (Function A)
- Sum rule Sum R_n/z_n = c_2 = 13/60
- Genus determination: order 1, genus 1
- g_B(z) analysis for 1/Pi_TT(z) (Function B): shown to be linear, not constant
- Physical interpretation: pure pole decomposition, contact term

**Does NOT cover:**
- Scalar sector Pi_s(z, xi) at xi != 1/6 (expected: g_{scalar} = -6(xi-1/6)^2)
- Higher-order (multi-loop) corrections to the ML expansion
- Whether the contact term -c_2 is renormalized at higher loops
- Kubo-Kugo objection (MR-2 Condition 3)

---

## Impact on SCT Programme

- **MR-2:** Strengthens Condition 1 (fakeon prescription convergence) by showing the non-pole part is trivially constant.
- **OT:** Confirms that all spectral weight is in poles, validating the pure-pole spectral positivity theorem.
- **CL:** Provides the underlying structural reason for commutativity: the N-independent part is just 1/z - c_2.
- **FK:** The sum rule Sum R_n/z_n = c_2 adds a new independent consistency check on the ghost catalogue.
