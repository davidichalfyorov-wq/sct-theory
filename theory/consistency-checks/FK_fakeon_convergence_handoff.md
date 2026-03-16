# FK Handoff Certificate: Fakeon Prescription Convergence for the SCT Propagator

**Task:** FK (Fakeon Extension to Infinite-Pole Propagators)
**Date:** March 14, 2026
**Status:** CONDITIONALLY CONVERGENT
**Pipeline:** FK-L -> FK-LR -> FK-D -> FK-DR -> FK-V -> FK-VR
**Total verification checks:** 56 FK-dedicated + 194 regression = 250/250 PASS

---

## 1. Task Description

The MR-2 unitarity analysis identified three conditions for SCT viability. Condition 1 stated: "The fakeon prescription extends to nonlocal propagators with entire-function form factors." This was listed as "plausible but unproven."

The FK pipeline investigates whether the Anselmi-Piva fakeon prescription, proven for polynomial propagators with finitely many poles, can be extended to the SCT propagator Pi_TT(z) = 1 + (13/60) z F_hat_1(z), which is an entire function with infinitely many zeros. Each zero produces a ghost pole in the graviton propagator G(k^2) = 1/(k^2 Pi_TT(z)).

---

## 2. Pipeline Summary

| Agent | Role | Key Output |
|-------|------|------------|
| FK-L | Literature extraction | No proof exists for infinite poles; mainstream nonlocal gravity avoids the problem; Boos-Carone N-pole program identified as promising |
| FK-LR | Literature audit | Corrected FK-L overstatements (triangle diagram is fine-tuning, not failure; KL representation generalizes, not fails); revised probability upward (30% -> 40%); identified 7 missing papers |
| FK-D | Derivation | Extended ghost catalogue to 16 zeros; proved |R_n| ~ 0.289/|z_n|; established convergence classification; one-subtraction Mittag-Leffler converges absolutely |
| FK-DR | Derivation review | Verified all FK-D claims independently at 80-100 dps; corrected bubble diagram loop integral claim; confirmed CONDITIONALLY CONVERGENT |
| FK-V | Verification | 56-test suite across 6 layers; all pass; 194 regression tests pass |
| FK-VR | Final certification | Cross-agent consistency verified; all deliverables confirmed; this handoff certificate |

---

## 3. Key Result: Fakeon Prescription Is Mathematically Well-Defined with One Subtraction

The fakeon prescription replaces each ghost pole with a principal-value distribution:

    R_n / (k^2 - m_n^2 + i*eps) --> R_n * P[1/(k^2 - m_n^2)]

For the SCT propagator with infinitely many poles, the Mittag-Leffler theorem with one subtraction gives:

    1/Pi_TT(z) = g(z) + sum_n R_n * [1/(z - z_n) + 1/z_n]

where g(z) is an entire function and the subtracted series converges absolutely. This is guaranteed by the order-1 nature of Pi_TT(z) and the verified decay |R_n| ~ 0.289/|z_n|.

---

## 4. Convergence Classification: CONDITIONALLY CONVERGENT

| Condition | Requirement | Status |
|-----------|-------------|--------|
| Absolute convergence of sum|R_n| | alpha > 1 | **NOT SATISFIED** (alpha = 1; logarithmic divergence ~0.023 ln N) |
| Conditional convergence of sum R_n | Modified sum rule | **SATISFIED** (1 + sum R_n = -6/83) |
| Subtracted absolute convergence of sum|R_n/z_n| | alpha > 0 | **SATISFIED** (p-series with p = 2; total ~ 0.625) |

The unsubtracted series is conditionally (not absolutely) convergent. One Mittag-Leffler subtraction yields absolute convergence. This is the expected and standard behavior for an order-1 entire function with genus 1.

---

## 5. Extended Ghost Catalogue: 16 Zeros (2 Real + 7 Conjugate Pairs)

| # | Label | Location (Euclidean z) | |z| | |R_n| | Type |
|---|-------|------------------------|-----|-------|------|
| 1 | z_L | -1.2807 + 0i | 1.28 | 0.5378 | Real (Lorentzian) |
| 2 | z_0 | +2.4148 + 0i | 2.41 | 0.4931 | Real (Euclidean) |
| 3,4 | C1 | 6.051 +/- 33.290i | 33.84 | 0.00856 | Complex pair |
| 5,6 | C2 | 7.144 +/- 58.931i | 59.36 | 0.00487 | Complex pair |
| 7,8 | C3 | 7.842 +/- 84.274i | 84.64 | 0.00342 | Complex pair |
| 9,10 | C4 | 8.357 +/- 109.524i | 109.84 | 0.00263 | Complex pair |
| 11,12 | C5 | 8.767 +/- 134.731i | 135.02 | 0.00214 | Complex pair |
| 13,14 | C6 | 9.107 +/- 159.916i | 160.17 | 0.00181 | Complex pair |
| 15,16 | C7 | 9.397 +/- 185.087i | 185.33 | 0.00156 | Complex pair |

First 8 zeros (|z| <= 100) match MR-2 catalogue to 10+ significant figures. New zeros C4-C7 (100 < |z| <= 200) verified as genuine zeros with |Pi_TT(z_n)| < 10^{-15}.

Pattern: Im(z_n) increases linearly with spacing Delta ~ 25.3; Re(z_n) increases logarithmically (6 to 9.4). Consistent with order-1 entire function.

---

## 6. Sum Rule: 1 + sum R_n = -6/83

The modified sum rule from MR-2:

    1 + sum_i R_i = 1/Pi_TT(+inf) = -6/83 ~ -0.0723

With 16 zeros catalogued: partial sum = -0.035, deficit = -0.038. The deficit is filled by zeros at |z| > 200 and converges as sum 1/|z_n|^2.

This is NOT a superconvergence relation (the sum is not zero) because Pi_TT(z -> +inf) = -83/6 is a finite nonzero constant.

---

## 7. Asymptotic: |R_n| ~ 0.289/|z_n|

The product |R_n| * |z_n| converges monotonically to C_R ~ 0.2892:

| Pair | |R_n| * |z_n| |
|------|---------------|
| C1 | 0.28962 |
| C2 | 0.28937 |
| C3 | 0.28930 |
| C4 | 0.28926 |
| C5 | 0.28924 |
| C6 | 0.28922 |
| C7 | 0.28921 |

Analytic prediction: C_R = 60/(13 * D) where D = lim |F_hat_1'(z_n)| * |z_n| ~ 15.96, giving C_R = 0.2889. This agrees with the numerical data to 0.1%.

The fitted power law exponent alpha = 1.0008 +/- 0.0005 is indistinguishable from 1, with the sub-leading correction attributable to logarithmic growth of Re(z_n).

---

## 8. What This Resolves and What It Does Not

### RESOLVED

1. **The pole series converges with one subtraction.** The one-subtraction Mittag-Leffler series for 1/Pi_TT(z) converges absolutely. This is a rigorous mathematical result following from standard complex analysis (Hadamard factorization for order-1 entire functions).

2. **The principal-value integral converges for the multi-pole sum.** Tested numerically on [0.01, 20] with up to 12 poles; successive differences decrease monotonically.

3. **The N-pole approximation converges at generic points.** At z = 5 and z = 10i, the N-pole partial fraction reconstruction improves monotonically with N.

4. **The entire part g(z) is smooth and unambiguous.** Since Pi_TT(z) is exactly known, g(z) is uniquely determined. It requires no principal-value regularization (it has no poles).

5. **The logarithmic divergence of sum|R_n| is extremely slow.** Even at N = 10^20 pairs, sum|R_n| ~ 2.1 (compared to ~1.08 at 7 pairs). The practical impact is negligible.

### NOT RESOLVED

1. **Limit commutativity.** Whether lim_{N->inf}[fakeon(N-pole)] = fakeon(lim_{N->inf}[N-pole]) is not proven. This is a well-posed mathematical question, not an open conjecture.

2. **Bubble diagram convergence.** FK-DR identified that the loop integral claim in FK-D is incorrect for bubble (two-propagator) diagrams: the contribution per pole decays as log(|z_n|)/|z_n|, not 1/|z_n|^3. After standard renormalization, convergence is restored.

3. **S-matrix unitarity.** Whether the fakeon S-matrix satisfies the optical theorem at all orders is a separate question, not addressed by the convergence analysis.

4. **Physical correctness of the prescription.** The analysis shows the fakeon prescription is mathematically consistent for SCT. Whether it is the physically correct prescription (vs. Feynman, average continuation, etc.) is a physics question, not a mathematics question.

5. **Scalar sector at xi != 1/6.** The convergence analysis covers only the spin-2 propagator Pi_TT. The scalar propagator Pi_s(z, xi) at xi != 1/6 has its own zero structure requiring analogous analysis. At xi = 1/6, Pi_s = 1 identically (no poles).

---

## 9. Impact on MR-2 Conditions

### Before FK

**Condition 1 (Fakeon extension):** "Plausible but unproven. No rigorous proof exists for propagators with infinitely many poles."

### After FK

**Condition 1 (Fakeon extension):** Upgraded to: "Mathematically well-posed. The pole series converges absolutely with one Mittag-Leffler subtraction. The remaining steps (computing g(z) explicitly, proving limit commutativity, and verifying S-matrix unitarity) are concrete, achievable tasks rather than open conjectures."

The mathematical obstruction identified in MR-2 (infinite number of Type C poles as a new subtlety) is shown to be mild: only logarithmic divergence of sum|R_n|, fully resolved by one subtraction. The SCT propagator's pole structure is as well-behaved as one could hope for an order-1 entire function.

### Conditions 2 and 3 (Unchanged)

Condition 2 (Donoghue-Menezes applicability) and Condition 3 (Kubo-Kugo resolution) are not affected by the FK analysis. They remain plausible but unproven.

---

## 10. Test Counts

| Suite | Tests | Status | Time |
|-------|-------|--------|------|
| test_fk_fakeon_convergence.py | 56 | 56/56 PASS | 3.4s |

Test layers:
| Layer | Focus | Tests | Status |
|-------|-------|-------|--------|
| Layer 1 | Ghost Catalogue | 17 | 17/17 PASS |
| Layer 2 | Convergence | 13 | 13/13 PASS |
| Layer 3 | Cross-Consistency | 7 | 7/7 PASS |
| Layer 4 | Physical Properties | 8 | 8/8 PASS |
| Layer 5 | Verdict and Regression | 6 | 6/6 PASS |
| Layer HP | High-Precision (100 dps) | 5 | 5/5 PASS |

Regression (run during FK-V):
| Suite | Tests | Status |
|-------|-------|--------|
| test_mr2_unitarity.py | 79 | 79/79 PASS |
| test_mr2_residue.py | 31 | 31/31 PASS |
| test_gp_dressed_propagator.py | 84 | 84/84 PASS |
| **Regression total** | **194** | **194/194 PASS** |

**Grand total: 250/250 PASS, 0 FAIL, 0 SKIP**

---

## 11. All Artifacts

| File | Description |
|------|-------------|
| `analysis/scripts/fk_fakeon_convergence.py` | Complete analysis script (1145 lines) |
| `analysis/results/fk/fk_fakeon_convergence_results.json` | Full numerical results |
| `analysis/sct_tools/tests/test_fk_fakeon_convergence.py` | 56-test verification suite |
| `docs/reviews/FK_L_fakeon_extension_literature.md` | Literature extraction report |
| `docs/reviews/FK_LR_fakeon_extension_audit.md` | Literature audit and probability calibration |
| `docs/reviews/FK_D_fakeon_convergence_derivation.md` | Derivation report |
| `docs/reviews/FK_DR_fakeon_convergence_review.md` | Independent derivation review |
| `docs/reviews/FK_V_fakeon_convergence_verification.md` | Verification report |
| `docs/reviews/FK_VR_final_certification.md` | Final certification and verdict |
| `theory/consistency-checks/FK_fakeon_convergence_handoff.md` | This handoff certificate |

---

## 12. Verdict

### Classification: CONDITIONALLY CONVERGENT -- PARTIALLY RESOLVES MR-2 CONDITION 1

The fakeon prescription for the SCT propagator with infinitely many poles is:

1. **NOT absolutely convergent** in the unsubtracted form (sum|R_n| diverges logarithmically with coefficient ~0.023 ln N).

2. **Absolutely convergent** with one Mittag-Leffler subtraction (sum|R_n/z_n| ~ 0.625, dominated by the two real poles). This is guaranteed by the order-1 nature of Pi_TT(z).

3. **Convergent for physical amplitudes** after standard renormalization. Diagrams with three or more propagators are absolutely convergent without subtraction; the bubble diagram requires one renormalization subtraction (which is needed independently for UV divergences).

MR-2 Condition 1 is upgraded from "plausible but unproven" to "mathematically well-posed with one subtraction." The remaining open questions are concrete and achievable rather than vaguely conjectural.

### Cross-Agent Consistency: VERIFIED

All six agents (FK-L through FK-VR) agree on:
- z_L = -1.2807, R_L = -0.538
- z_0 = +2.4148, R_0 = -0.493
- Sum rule 1 + sum R_n = -6/83
- Classification: CONDITIONALLY CONVERGENT
- 16 zeros total (12 catalogue entries)
- |R_n| ~ 0.289/|z_n| (alpha = 1 exactly)

One correction applied: FK-D's bubble diagram loop integral claim (1/|z_n|^3 per pole) corrected to log(|z_n|)/|z_n| per pole, convergent after renormalization (FK-DR finding, confirmed by FK-V).

### Survival Probability Update

The FK analysis does not change the overall MR-2 survival probability estimate (60-70%), because Conditions 2 and 3 remain unchanged. However, it strengthens Condition 1 from a speculative hope to a well-posed mathematical program. The literature probability for full fakeon extension is revised from FK-L's 30% to FK-LR's 40% plausible estimate.
