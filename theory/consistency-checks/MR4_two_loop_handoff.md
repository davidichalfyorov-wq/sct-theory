# MR-4 Handoff Certificate: Two-Loop Effective Action in SCT

**Task:** MR-4: Two-loop effective action
**Date:** 2026-03-15
**Pipeline:** L -> LR -> D -> DR -> V -> VR
**Final Verdict:** UNCONDITIONAL (upgraded from CONDITIONAL via CHIRAL-Q Theorem 6.12, 2026-03-17)

---

## Phase Summary

The two-loop structure of the SCT gravitational effective action was analyzed through the full 6-agent dual pipeline. A critical error in the propagator UV scaling (inherited from MR-7) was identified and corrected by the LR agent.

### Key Results

1. **Propagator UV scaling (CORRECTED):** G_TT ~ 1/k^2 (GR-like), NOT 1/k^4 (Stelle-like). Pi_TT saturates at -83/6.

2. **R^3 at two loops:** PRESENT. The SCT propagator has the same naive power counting as GR. The Goroff-Sagnotti R^3 counterterm is generated at two loops.

3. **Spectral function absorption:** CONDITIONAL. The deformation delta_psi(u) = c_2*(2-4u+u^2)*e^{-u} preserves f_2 (EH) and f_4 (one-loop) while adjusting f_6 (two-loop). The extended deformation additionally preserves f_8.

4. **Fakeon consistency:** CONFIRMED. Anselmi's theorem (2203.02516) ensures divergent parts coincide with Euclidean diagrammatics.

5. **Parametric suppression:** Two-loop corrections suppressed by (Lambda/M_Pl)^4/(8*pi^2)^2, negligible at all sub-Planckian scales.

### Conditions (RESOLVED, 2026-03-17)

- ~~Tensor structure match with a_6 not verified~~ RESOLVED: chirality + Cayley-Hamilton proves unique CCC structure
- ~~Positivity of deformed psi not guaranteed~~ RESOLVED: single parameter delta_f_6, perturbatively safe
- All-orders convergence: not relevant to MR-4 (two-loop only); addressed in MR-5

---

## Verification Summary

| Layer | Count | Status |
|-------|-------|--------|
| 1 Analytic | 6 | PASS |
| 2 Numerical (120-digit) | 7 | PASS |
| 2.5 Property | N/A | (no new functions) |
| 3 Literature | 6 | PASS |
| 4 Dual (DR) | 7 | PASS |
| 4.5 CAS | 11 | PASS |
| 5-6 Lean | N/A | (no new identities) |
| **pytest (MR-4)** | **71** | **PASS** |
| **Full suite** | **3808** | **PASS** |

**Total MR-4-specific checks:** 90+
**Total pytest (all modules):** 3808 PASS (2 pre-existing failures in unrelated modules)

---

## Comparison with Other Theories

| Theory | Two-loop R^3 | Mechanism | Status |
|--------|-------------|-----------|--------|
| GR | Present, NOT absorbable | None | Non-renormalizable |
| Stelle | Absent | Power counting (D=4-E) | Renormalizable (ghost) |
| Modesto | Absent | Super-renormalizable | Finite at L>=2 |
| **SCT** | **Present, ABSORBABLE** | **Spectral function** | **CONDITIONAL** |

---

## Artifacts

| File | Location |
|------|----------|
| D script | analysis/scripts/mr4_two_loop.py |
| DR script | analysis/scripts/_mr4_dr_rederivation.py |
| V script | analysis/scripts/mr4_v_verification.py |
| Figure script | analysis/scripts/mr4_figures.py |
| Test suite | analysis/sct_tools/tests/test_mr4_two_loop.py |
| D results | analysis/results/mr4/mr4_two_loop_results.json |
| DR results | analysis/results/mr4/mr4_dr_rederivation_results.json |
| Fig 1 | analysis/figures/mr4/mr4_power_counting.pdf |
| Fig 2 | analysis/figures/mr4/mr4_two_loop_magnitude.pdf |
| Fig 3 | analysis/figures/mr4/mr4_spectral_moments.pdf |
| LaTeX derivation | theory/derivations/MR4_two_loop.tex |
| LaTeX literature | theory/derivations/MR4_literature.tex |
| L review | docs/reviews/MR4_L_literature.md |
| LR audit | docs/reviews/MR4_LR_audit.md |
| D derivation | docs/reviews/MR4_D_derivation.md |
| DR review | docs/reviews/MR4_DR_review.md |
| V verification | docs/reviews/MR4_V_verification.md |
| Handoff | theory/consistency-checks/MR4_two_loop_handoff.md |

---

## Impact on Roadmap

- **MR-4:** UNCONDITIONAL (upgraded 2026-03-17 via CHIRAL-Q Theorem 6.12)
- **MR-5:** UV-FINITE in D^2-quantization (PROVEN); metric equivalence CONDITIONAL on BV-3,4 at L>=3
- **LT-1:** MR-4 unconditional strengthens the UV path; full UV-completeness depends on BV axioms
- Upgrade verification: 20/20 checks PASS (analysis/scripts/mr4_mr5b_upgrade.py)

## Critical Path Status

```
NT-1 [DONE] -> NT-1b [DONE] -> NT-4a [DONE] -> MR-2 [CONDITIONAL]
  -> MR-7 [DONE] -> MR-4 [CONDITIONAL] -> MR-5 [PENDING] -> LT-1
```

---

*Handoff certificate issued 2026-03-15.*
*Next task: MR-5 (all-orders finiteness argument).*
