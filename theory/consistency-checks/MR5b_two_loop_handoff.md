# MR-5b Handoff Certificate: Two-Loop Divergence Analysis in SCT

**Task:** MR-5b: Two-loop D=0 in the background field method
**Date:** 2026-03-16
**Pipeline:** 6-agent (L -> LR -> D -> DR -> V -> VR)
**Classification:** CONDITIONAL (Option C)

---

## Verdict

**Physical (on-shell) D=0:** ESTABLISHED at leading perturbative order O(alpha_C^2).

The perturbative on-shell reduction shows that only the CCC invariant (Goroff-Sagnotti cubic Weyl contraction) survives at leading order. With 1 constraint and 1 adjustable parameter (delta f_6 from spectral function deformation), absorption is possible.

**Formal (off-shell) D=0:** OPEN. The off-shell effective action contains 5-6 independent dimension-6 counterterms with unknown individual coefficients. A Goroff-Sagnotti-level computation with the SCT propagator would be needed to resolve this.

---

## Key Results

| Result | Value | Verification |
|--------|-------|-------------|
| SM CCC coefficient | -740.5/3 = -1481/6 ~ -246.83 | 120-digit exact |
| Scalar CCC | -16/3 | Gilkey (1975), Vassilevich (2003) |
| Dirac CCC | -109/3 = -64/3 - 15 | Avramidi (2000) |
| Vector CCC | 148/3 = 116/3 + 32/3 | 2 FP ghost subtraction |
| f_6 | Gamma(3) = 2 | Exact |
| GS coefficient (GR) | 209/2880 | Goroff-Sagnotti (1986) |
| NGFP theta_3 | -79.39 | Gies et al. (2016) |
| DR ratio |R| | 0.00469 | Finite, nonzero |

## Agent Pipeline Summary

| Agent | Tests | Key Contribution |
|-------|-------|-----------------|
| L | -- | 24 references, 8 objectives, FKWC basis |
| LR | -- | 1 CRITICAL correction (theta_3), a_6 formula audit |
| D | 74 PASS | a_6 computation, perturbative on-shell reduction |
| DR | 49 PASS | 5 independent methods, CCC verification |
| V | 34 PASS | 120-digit verification, 4-layer check, LaTeX |

## Artifacts

| File | Location |
|------|----------|
| Main script | `analysis/scripts/mr5b_two_loop.py` |
| DR script | `analysis/scripts/_mr5b_dr_rederivation.py` |
| V script | `analysis/scripts/mr5b_v_verification.py` |
| Tests | `analysis/sct_tools/tests/test_mr5b_two_loop.py` |
| D results | `analysis/results/mr5b/mr5b_two_loop_results.json` |
| DR results | `analysis/results/mr5b/mr5b_dr_rederivation_results.json` |
| V results | `analysis/results/mr5b/mr5b_v_verification_results.json` |
| Figures | `analysis/figures/mr5b/mr5b_ccc_per_spin.pdf`, `mr5b_a6_ratio_comparison.pdf` |
| LaTeX | `theory/derivations/MR5b_two_loop.tex` |
| Literature | `docs/reviews/MR5b_L_literature.md` |
| LR audit | `docs/reviews/MR5b_LR_audit.md` |
| D review | `docs/reviews/MR5b_D_derivation.md` |
| DR review | `docs/reviews/MR5b_DR_review.md` |
| V review | `docs/reviews/MR5b_V_verification.md` |

## Limitations

1. Off-shell D=0 not proven (requires person-years of computation)
2. Non-vacuum backgrounds not treated
3. NLO corrections O(alpha_C^3) not guaranteed absorbable
4. Order-of-limits commutativity assumed (standard in perturbative QFT)
5. SCT fails Modesto-Rachwal finiteness conditions (q=0, form factor order 1/2)

## Impact on Roadmap

- MR-5 status updated to include two-loop sub-result
- Survival probability: 62-72% (unchanged)
- Consistent with MR-4 CONDITIONAL and MR-5 CONDITIONAL

---

*Handoff certified: 2026-03-16*
*Total tests: 74 (D) + 49 (DR) + 34 (V) = 157 PASS*
*Full regression: 4094 PASS*
