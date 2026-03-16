# NT-4b Handoff Certificate

**Phase:** NT-4b (Full Nonlinear Variational Field Equations)
**Date:** 2026-03-15
**Status:** PASS

---

## Pipeline Trace

| Agent | Role | Status | Key Output |
|-------|------|--------|------------|
| NT4b-L | Literature extraction | PASS | `NT4b_literature.tex`, 11 references, 7 gaps identified |
| NT4b-LR | Literature audit | PASS WITH CORRECTIONS | 6 errors found and corrected (factor-of-2, trace, variations) |
| NT4b-D | Derivation (Method A) | PASS | `nt4b_nonlinear.py`, 98 tests, field equations in {C^2, R^2} basis |
| NT4b-DR | Re-derivation (Method B1) | PASS | 12 new cross-check tests, exact agreement via Gauss-Bonnet |
| NT4b-V | Verification | PASS | 8 layers verified, 110/110 tests, 4 figures |
| NT4b-VR | Final review | **PASS** | LaTeX document, handoff certificate, all outputs verified |

---

## Key Results

### The Canonical Field Equations

```
(1/kappa^2) G_{mn}
  + (1/(16pi^2)) [2 F_1(Box/Lambda^2) B_{mn} + Theta^(C)_{mn}]
  + (1/(16pi^2)) [F_2(Box/Lambda^2, xi) H_{mn} + Theta^(R)_{mn}]
  = (1/2) T_{mn}
```

where:
- G_{mn}: Einstein tensor from S_EH
- B_{mn}: Bach tensor from C^2 variation (traceless, divergence-free)
- H_{mn}: R^2 sector tensor, trace g^mn H_mn = -6 Box R
- Theta^(C), Theta^(R): BV alpha-insertion corrections (O(h^2) on flat)
- F_1, F_2: entire form factors from NT-1b Phase 3

### Basis
{C^2, R^2} (Weyl basis), maintaining spin-2/spin-0 separation.

### Canonical Coefficients

| Quantity | Value | Precision |
|----------|-------|-----------|
| alpha_C | 13/120 | exact (100-digit) |
| alpha_R(xi) | 2(xi - 1/6)^2 | exact |
| c_2 = 2 alpha_C | 13/60 | exact |
| 3c_1 + c_2 | 6(xi - 1/6)^2 | exact |
| m_2/Lambda | sqrt(60/13) ~ 2.148 | 100-digit |
| m_0/Lambda (xi=0) | sqrt(6) ~ 2.449 | 100-digit |

### Consistency Checks (6/6 PASS)

1. **Local limit:** Recovers Stelle: G + 2*alpha*B + beta*H = kappa^2 T
2. **Linearized limit:** Recovers NT-4a Pi_TT and Pi_s (28 checks, err < 1e-25)
3. **Bianchi identity:** Verified on 20+ momenta (residual < 1e-16)
4. **Trace:** Bach traceless, H trace = -6 Box R (corrected from LR audit)
5. **De Sitter:** Exact solution for all R_0 (H=0, Theta=0, C=0)
6. **Ricci-flat:** Only Weyl sector contributes

### Verification Summary

| Layer | Status | Tests/Cases |
|-------|--------|-------------|
| L1 Analytic | PASS | 8 checks |
| L2 Numerical (100-digit) | PASS | 10 checks |
| L2.5 Property fuzzing | PASS | 12 properties, 3611+ cases |
| L3 Literature | PASS | 6 cross-checks |
| L4 Dual derivation | PASS | 12 Method B1 tests |
| L4.5 Triple CAS (rational) | PASS | 9 identity checks |
| L5/6 Lean | SKIP | Justified (nonlocal tensors not formalizable) |
| L7 Regression | PASS | 110 NT-4b tests, full suite clean |

### Method B1 Confirmation

Method B1 (variation in {Riem^2, Ric^2, R^2} basis then convert via Gauss-Bonnet) independently confirmed:
- Ric^2 sector drops out: 2*gamma_4 + gamma_2 = 0
- GB identity verified for all 7 tensor structures
- Pi_TT and Pi_s match to machine precision at 16 test points
- Theta terms recombine identically
- The agreement is exact, not approximate

---

## Files Produced (Complete Pipeline)

### Theory Documents
| File | Description |
|------|-------------|
| `theory/derivations/NT4b_literature.tex` | Literature extraction with conventions |
| `theory/derivations/NT4b_literature.pdf` | Compiled literature review |
| `theory/derivations/NT4b_nonlinear.tex` | Full derivation document |
| `theory/derivations/NT4b_nonlinear.pdf` | Compiled derivation |
| `theory/derivations/NT4b_handoff.md` | This handoff certificate |

### Review Documents
| File | Description |
|------|-------------|
| `docs/reviews/NT4b_L_literature_review.md` | L agent summary |
| `docs/reviews/NT4b_LR_audit.md` | LR agent audit (6 corrections) |
| `docs/reviews/NT4b_D_derivation.md` | D agent derivation notes |
| `docs/reviews/NT4b_DR_rederivation.md` | DR agent re-derivation |
| `docs/reviews/NT4b_V_verification.md` | V agent verification report |
| `docs/reviews/NT4b_VR_final.md` | VR agent final review |

### Code and Data
| File | Description |
|------|-------------|
| `analysis/scripts/nt4b_nonlinear.py` | Main derivation/verification script |
| `analysis/sct_tools/tests/test_nt4b_nonlinear.py` | 110 tests |
| `analysis/results/nt4b/nt4b_nonlinear_results.json` | All check results |
| `analysis/results/nt4b/nt4b_wald_variation.json` | Wald entropy data |

### Figures
| File | Description |
|------|-------------|
| `analysis/figures/nt4b/nt4b_linearized_recovery.pdf` | Propagator denominators |
| `analysis/figures/nt4b/nt4b_local_limit_verification.pdf` | Stelle coefficients |
| `analysis/figures/nt4b/nt4b_field_eq_structure.pdf` | Field equation structure |
| `analysis/figures/nt4b/nt4b_effective_masses.pdf` | Mass scales and potential |

---

## Notes for Downstream Tasks

### NT-4c (FLRW Reduction)
- Start from Eq. (full-eom) in NT4b_nonlinear.tex
- On FLRW: R_mn = diag(a(t)), C_mnrs simplifies, H_mn involves H(t) and its derivatives
- Theta^(R) involves nabla R terms which are nonzero on FLRW (R depends on t)
- The de Sitter result (H=0 when R=const) provides the static limit
- Boundary terms (Gap G4) become relevant for FLRW topology

### MT-1 (Black Hole Entropy)
- Use the Wald variation from Eq. (wald-variation) and nt4b_wald_variation.json
- On-horizon evaluation: form factors reduce to z=0 values (alpha_C, alpha_R)
- Dominant term: S = A/(4G) + O(alpha_C * A/ell_Lambda^2)
- The Weyl-sector correction is antisymmetric in [mn] and [rs]

### MR-9 (Singularity Resolution)
- The nonlocal form factors may regularize the Schwarzschild singularity
- Start from the Ricci-flat sector of the field equations
- Key: the entire-function property of F_1 (NT-2) prevents new poles

### FUND-1 (QFT Recovery)
- Verify that the spectral action formalism gives standard QFT on curved backgrounds
- The local limit (Stelle) is the starting point
- The alpha-insertion structure is the BV framework, which is standard

---

## Known Limitations

1. **Gap G1:** Explicit closed-form Theta^(C) on general backgrounds not derived (tensor alpha-insertion). Properties verified without explicit formula.
2. **FLRW reduction:** Deferred to NT-4c.
3. **Lean formalization:** Nonlocal tensorial content not currently formalizable in Lean 4.
4. **Conformal float artifact:** Python float 1/6 differs from exact 1/6 by O(1e-16), producing a tiny scalar mode coefficient at xi=1/6. Not a physics error.

---

## Numerical Values (VR-Verified)

| Quantity | Value |
|----------|-------|
| alpha_C | 0.10833333333... = 13/120 |
| alpha_R(xi=0) | 0.05555555555... = 1/18 |
| alpha_R(xi=1/6) | 0 (exact) |
| c_2 = LOCAL_C2 | 0.21666666666... = 13/60 |
| F_1(0) | 6.860288475783287e-04 |
| F_2(0, xi=0) | 3.518096654247839e-04 |
| Pi_TT(1) | 0.8994734408563 |
| Pi_s(1, xi=0) | 1.2043061094801 |
| m_2/Lambda | 2.1483446221183 |
| m_0/Lambda (xi=0) | 2.4494897427832 |
| Bach trace | 0 (exact) |
| H trace coefficient | -6 (exact) |

---

## Build Status

- **pytest:** 110/110 NT-4b tests PASS
- **Full suite:** All tests clean (no regressions)
- **PDF build:** 45/45 .tex compiled successfully
- **ruff:** clean (0 errors)
