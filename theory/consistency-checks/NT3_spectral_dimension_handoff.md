# NT-3 Spectral Dimension: Handoff Certificate

**Task:** NT-3 Spectral Dimension Flow in Spectral Causal Theory
**Agent:** NT3-V (Verification)
**Date:** 2026-03-16
**Status:** COMPLETE (DEFINITION-DEPENDENT)
**Verification:** 32/32 checks across 5 layers

---

## Key Results

### Primary Finding: Ghost-Induced P < 0

The Mittag-Leffler return probability P_ML(sigma) = W(sigma)/(16 pi^2 sigma^2) becomes negative for sigma < sigma* ~ 0.010 Lambda^{-2} because:

    Sum R_n = -1.034    (sum of 8 ghost residues, 100-digit verified)
    W(0) = 1 + Sum R_n = -0.034 < 0

This is NOT a truncation artifact: additional (uncatalogued) poles have negative real residues, making W(0) more negative.

### Physical Spectral Dimension Flow

In the physical region (sigma > sigma*):

| sigma [Lambda^{-2}] | W(sigma) | d_S |
|---------------------|----------|-----|
| 0.02 | 0.027 | 0.62 |
| 0.05 | 0.069 | 2.47 |
| 0.10 | 0.139 | 2.06 |
| 0.50 | 0.568 | 2.78 |
| 1.0  | 0.807 | 3.26 |
| 5.0  | 0.999 | 3.99 |
| 10.0 | 1.000 | 4.00 |

### Definition Dependence

| Method | d_S(UV) | d_S(IR) |
|--------|---------|---------|
| Propagator (CMN) | 2 | 2 |
| Heat kernel (std) | 4 | 4 |
| ASZ/Fakeon | 0 | 4 |
| ML (physical region) | ~2 at sigma* | 4 |

### Final Prediction

**Option E: d_S depends on definition.** Recommended primary result:
- d_S ~ 2 at sigma ~ 0.05-0.1 Lambda^{-2} (ghost scale)
- d_S = 4 at sigma >> Lambda^{-2} (IR)
- d_S not defined for sigma < sigma* ~ 0.01 Lambda^{-2} (ghost dominance)

---

## Verification Summary

| Layer | Name | Result |
|-------|------|--------|
| 1 | Analytic | 7/7 PASS |
| 2 | Numerical (100-digit) | 9/9 PASS |
| 3 | Literature | 8/8 PASS |
| 4 | Dual Derivation (DR) | 6/6 PASS |
| 4.5 | Triple CAS | 2/2 PASS |
| **Total** | | **32/32 PASS** |

### Cross-Agent Agreement

- D agent vs DR agent: 5/5 numerical cross-checks AGREE
- DR Method A (direct ratio), B (Kallen-Lehmann), C (Weyl counting), D (saddle-point), E (Stelle validation): ALL PASS
- Critical finding P < 0: independently confirmed by both DR agent and V agent

---

## Artifacts

| Deliverable | Location |
|-------------|----------|
| Main script | analysis/scripts/nt3_spectral_dimension.py |
| DR script | analysis/scripts/_nt3_dr_rederivation.py |
| V script | analysis/scripts/nt3_v_verification.py |
| Test suite | analysis/sct_tools/tests/test_nt3_spectral_dimension.py |
| V results | analysis/results/nt3/nt3_v_verification_results.json |
| D results | analysis/results/nt3/nt3_spectral_dimension_results.json |
| DR results | analysis/results/nt3/nt3_dr_rederivation_results.json |
| Figure 1 | analysis/figures/nt3/nt3_spectral_dimension.pdf |
| Figure 2 | analysis/figures/nt3/nt3_return_probability.pdf |
| Figure 3 | analysis/figures/nt3/nt3_uv_comparison.pdf |
| LaTeX (main) | theory/derivations/NT3_spectral_dimension.tex |
| LaTeX (lit) | theory/derivations/NT3_literature.tex |
| L review | docs/reviews/NT3_L_literature.md |
| LR audit | docs/reviews/NT3_LR_audit.md |
| D review | docs/reviews/NT3_D_derivation.md |
| DR review | docs/reviews/NT3_DR_review.md |
| V review | docs/reviews/NT3_V_verification.md |
| Handoff | theory/consistency-checks/NT3_spectral_dimension_handoff.md |

---

## Canonical Values (DO NOT MODIFY)

- Sum R_n = -1.034218 (8 catalogued poles)
- W(0) = 1 + Sum R_n = -0.034218
- sigma* = 0.01002 Lambda^{-2} (W = 0 crossing)
- Pi_TT(inf) = -83/6 = -13.8333...
- Pi_TT(z_0 = 2.4148) = 0 (ghost zero)
- d_S(ML, sigma=0.1) = 2.061
- d_S(ML, sigma=1.0) = 3.264
- d_S(ML, IR) = 4.000
- d_S(ASZ, UV) = 0
- d_S(propagator) = 2

---

## Impact on Theory

1. SCT does NOT cleanly predict d_S(UV) = 2 from modified propagator scaling. The 1/k^2 UV behavior would give d_S = 4, but ghost interference produces a transient d_S ~ 2 near the ghost scale.

2. The P < 0 issue is the heat-kernel manifestation of the ghost problem. It reinforces the importance of the fakeon prescription (MR-2) and the ghost width analysis (A3).

3. The definition-dependent nature of d_S in SCT is itself a physically meaningful result: it shows that the spectral dimension concept requires careful treatment in theories with ghost poles.

4. The d_S ~ 2 value at intermediate scales provides qualitative consistency with the CDT/AS universality conjecture, but through a distinct mechanism.
