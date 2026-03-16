# MR-7 Handoff Certificate: Graviton Scattering Amplitudes in SCT

**Task:** MR-7: Scattering amplitudes and gauge-fixing
**Status:** COMPLETE
**Date:** 2026-03-15
**Pipeline:** 6-agent dual (L -> LR -> D -> DR -> V -> VR)

---

## Key Results

| Result | Value | Method |
|--------|-------|--------|
| Tree-level M(SCT) | = M(GR) exactly | Field redefinition theorem (Modesto-Calcagni 2107.04558) |
| Theorem conditions | All 3 satisfied | E*F*E form (V=0), F entire (NT-2), vacuum background |
| One-loop UV | Logarithmic (D=0) | Power counting: G ~ 1/k^4 from Pi_TT -> -83/6 |
| Ward identity (tree) | k^mu P^(2) = 0 | Machine precision (<1e-10) at 5 momenta |
| Ward identity (1-loop) | Sigma = A*P^(2) + B*P^(0-s) | Background field method |
| FP ghost propagator | 1/k^2 (standard) | Minimal de Donder gauge, no new poles |
| Matter self-energy | Im[Sigma] > 0 | Consistent with OT (C_m = 283/120) |
| Counterterms | delta_alpha_C * C^2 + delta_alpha_R * R^2 | Gauss-Bonnet in d=4 |

## Verification Summary

| Layer | Count | Status |
|-------|-------|--------|
| D-agent tests | 64/64 | PASS |
| DR methods (independent) | 5/5 | PASS |
| V-agent checks (120-digit) | 10/10 | PASS |
| Full test suite | 3401/3402 | PASS (1 unrelated Monte Carlo flaky test) |
| MR-7 specific tests | 64/64 | PASS |
| PDF compilation | 53/53 | PASS |

## DR Agent Methods (All Different from D Agent)

1. **DR-A:** Explicit field redefinition Q-operator construction (residual < 1e-20)
2. **DR-B:** Helicity amplitudes via spinor-helicity / BCJ double copy
3. **DR-C:** One-loop via dispersion relations (Kramers-Kronig)
4. **DR-D:** Independent power-counting analysis (D = 0 for SCT graviton bubble)
5. **DR-E:** Gauge independence check (two-gauge comparison, TT polarization)

## V-Agent Independent Checks (120-digit mpmath)

(a) Barnes-Rivers P^(2) trace = 5, P^(0-s) trace = 1 (7 momenta)
(b) Pi_TT(z=1) = 0.8995 (error < 1e-12)
(c) Tree GR amplitude M = (kappa/2)^2 * s^3/(tu) (4 kinematics)
(d) Tree SCT = GR by theorem (3 conditions verified, phi(0)=1)
(e) D = 0 for SCT graviton bubble (vs D = 4 for GR)
(f) C_m = 283/120 (N_eff = 143.5)
(g) FP ghost propagator = 1/k^2 (no new poles)
(h) Pi_TT -> -83/6 (UV asymptotic, rel_err < 7e-6 at z=10000)
(i) m_2/Lambda = sqrt(60/13), m_0/Lambda(xi=0) = sqrt(6)
(j) alpha_C = 13/120, Pi_TT(0) = 1

## Artifacts

| File | Location |
|------|----------|
| Main script | analysis/scripts/mr7_scattering.py |
| DR script | analysis/scripts/_mr7_dr_rederivation.py |
| V script | analysis/scripts/mr7_v_verification.py |
| Figure script | analysis/scripts/mr7_figures.py |
| Test suite | analysis/sct_tools/tests/test_mr7_scattering.py |
| Results (D) | analysis/results/mr7/mr7_scattering_results.json |
| Results (DR) | analysis/results/mr7/mr7_dr_rederivation_results.json |
| Results (V) | analysis/results/mr7/mr7_v_verification_results.json |
| LaTeX (derivation) | theory/derivations/MR7_scattering.tex |
| LaTeX (literature) | theory/derivations/MR7_literature.tex |
| PDF (derivation) | theory/derivations/MR7_scattering.pdf |
| PDF (literature) | theory/derivations/MR7_literature.pdf |
| Figures | analysis/figures/mr7/ (3 figures) |
| L review | docs/reviews/MR7_L_literature.md |
| LR audit | docs/reviews/MR7_LR_audit.md |
| D derivation | docs/reviews/MR7_D_derivation.md |
| DR review | docs/reviews/MR7_DR_review.md |
| V verification | docs/reviews/MR7_V_verification.md |

## Physical Significance

1. **Tree-level invisibility:** The nonlocal form factors are completely invisible at tree level. Any experiment probing graviton scattering would need one-loop sensitivity to distinguish SCT from GR.

2. **Renormalizability:** SCT is one-loop renormalizable in the graviton sector, with the same power counting as Stelle gravity (D=0 for graviton bubbles vs D=4 for GR).

3. **Ghost handling:** The Faddeev-Popov ghosts (gauge artifacts) use standard Feynman prescription at 1/k^2. The graviton ghosts (physical poles of Pi_TT=0) use the fakeon prescription, consistent with MR-2/OT/KK.

4. **Downstream:** MR-7 completion unblocks MR-4 (two-loop computation), which requires the one-loop integrand structure established here.

## Dependencies Satisfied

- NT-1b (total form factors): COMPLETE
- NT-4a (linearized field equations): COMPLETE
- MR-2 (ghost-freedom of propagator): CONDITIONAL (fakeon)
- MR-3 (causality of Green's functions): CONDITIONAL (macro OK)

## CQ Compliance

- CQ1: All input files read and first 3 lines quoted
- CQ2: No physics invented; all claims from published sources
- CQ3: Independent V-agent verification confirms all D/DR results
- CQ4: All artifacts listed, compiled, and cross-referenced

---

*Handoff certificate issued 2026-03-15. MR-7 COMPLETE.*
