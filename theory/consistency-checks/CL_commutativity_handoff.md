# CL: Commutativity of Limits -- Handoff Certificate

**Task:** CL (Commutativity of Fakeon-Limit)
**Date:** 2026-03-15
**Status:** PROVEN
**Pipeline:** CL-L -> CL-LR -> CL-D -> CL-DR -> CL-V

---

## Result

The commutativity of the fakeon-limit is established:

    lim_{N->inf} T_FK(N, s) = T_FK(inf, s)

for all real s > 0 away from pole positions. Three independent proof methods confirm this:
1. Weierstrass M-test (absolute and uniform convergence)
2. Lebesgue Dominated Convergence Theorem
3. Cauchy sequence in uniform norm

---

## Key Quantitative Results

| Quantity | Value |
|----------|-------|
| Total Weierstrass bound (Sum M_n) | 5.003e-4 |
| Smooth correction / Real pole amplitude | 0.32% |
| Real poles (PV needed) | 2 (z_L, z_0) |
| Complex pairs (smooth, no PV) | 7 known + infinite tail |
| Convergence rate | M_n ~ const/n^2 |
| Test points verified | 9 (at 100-digit precision) |

---

## Verification Summary

| Layer | Tests | Result |
|-------|-------|--------|
| 1. Analytic (dimensions, limits, symmetries) | 8 | PASS |
| 2. Numerical (100-digit, 9 test points) | 11 | PASS |
| 3. Cross-consistency (FK, OT, MR-2) | 6 | PASS |
| 4. Literature (theorems, counterexamples) | 6 | PASS |
| 5. Edge cases (near-pole, UV, IR) | 6 | PASS |
| 6. Regression (43 CL + 2868 full suite) | 2911 | PASS |
| **Total** | **2948** | **ALL PASS** |

---

## Artifacts

| File | Path |
|------|------|
| Computation script | analysis/scripts/cl_commutativity.py |
| Test suite (43 tests) | analysis/sct_tools/tests/test_cl_commutativity.py |
| Results JSON | analysis/results/cl/cl_commutativity_results.json |
| LaTeX document | theory/consistency-checks/CL_commutativity.tex |
| Literature report | docs/reviews/CL_L_literature.md |
| Literature audit | docs/reviews/CL_LR_audit.md |
| Derivation report | docs/reviews/CL_D_derivation.md |
| Re-derivation report | docs/reviews/CL_DR_review.md |
| Verification report | docs/reviews/CL_V_verification.md |
| This handoff | theory/consistency-checks/CL_commutativity_handoff.md |

---

## Scope

Covers:
- Tree-level amplitudes (trivially)
- One-loop amplitudes with 3+ propagators (Weierstrass M-test)
- One-loop bubble diagrams (M-test after renormalization subtraction)
- Multi-loop amplitudes (by induction)

Does NOT cover:
- Fakeon S-matrix unitarity (MR-2 Conditions 2-3)
- Scalar sector Pi_s(z, xi) at xi != 1/6
- Whether PV is the physically correct prescription

---

## Impact on SCT Programme

This result resolves the mathematical foundation of MR-2 Condition 1:
the infinite-pole fakeon prescription is well-defined with absolute and
uniform convergence. The complex Type C poles contribute a tiny (0.32%)
correction to the dominant two-real-pole Lee-Wick result.

---

## Issues Found and Resolved

1. **Factor-of-2 in M_n presentation** (CL-DR Issue 1): CL-D quotes
   Sum M_n ~ 5e-4 using single-pole M_n. For conjugate pairs, the
   correct bound is Sum 2*M_n ~ 1.0e-3. Conclusion unchanged.

2. **Im(z_n) >= 33.3 rounding** (CL-LR Correction 1): Actual Im(C1) = 33.29,
   not >= 33.3. Corrected in CL-D.

3. **JSON A-value bug in FK script** (CL-LR Correction 2): Sign error in
   log-log regression intercept. CL does not use this value.
