# NT-1b Phase 3: Handoff Certificate

**Task:** Combined SM Form Factors and c₁/c₂ Re-derivation
**Date:** 2026-03-12
**Pipeline:** Six-Stage Sequential Review (L → L-R → D → D-R → V → V-R)
**Status:** ✅ COMPLETE — ALL CHECKS PASS

---

## 1. Pipeline Execution Summary

| Stage | Role | Status | Key Findings |
|-------|------|--------|-------------|
| L | Derive combined SM form factors | ✅ PASS | All analytic results derived |
| L-R | Review L-stage derivation | ✅ PASS | Confirmed N_f convention, formulas correct |
| D | Implement code + 2046/2046 tests | ✅ PASS | form_factors.py updated, all tests green |
| D-R | Independent re-derivation | ✅ PASS | 1 UV asymptotic error caught and corrected |
| V | 8-layer verification | ✅ PASS | 354/354 checks across all 8 layers |
| V-R | Final review + handoff | ✅ PASS | This document |

---

## 2. Phase 3 Verified Results

### 2.1 SM Particle Counting Convention

| Sector | Symbol | Value | Convention |
|--------|--------|-------|-----------|
| Real scalars | N_s | 4 | Higgs doublet (4 real components) |
| Dirac fermions | N_D = N_f/2 | 22.5 | 45 Weyl spinors ÷ 2 (CPR 0805.2909) |
| Gauge bosons | N_v | 12 | SU(3)×SU(2)×U(1): 8+3+1 |

### 2.2 Total Weyl² Coefficient (ξ-independent)

```
α_C = N_s·h_C^(0)(0) + N_D·h_C^(1/2)(0) + N_v·h_C^(1)(0)
    = 4·(1/120) + 22.5·(-1/20) + 12·(1/10)
    = 4/120 - 135/120 + 144/120
    = 13/120
```

**Verified**: mpmath 100-digit, Triple CAS (SymPy × GiNaC × mpmath), Lean 4 formal proof.

### 2.3 Total R² Coefficient (ξ-dependent)

```
α_R(ξ) = N_s·h_R^(0)(0,ξ) + N_D·h_R^(1/2)(0) + N_v·h_R^(1)(0)
        = 4·(1/2)(ξ - 1/6)² + 22.5·0 + 12·0
        = 2(ξ - 1/6)²
```

Only the scalar sector contributes (Dirac and vector are conformal: β_R = 0).

### 2.4 Basis Conversion {C², R²} → {R², R_μν²}

```
c₁ = α_R(ξ) - (2/3)α_C     [coefficient of R²]
c₂ = 2α_C = 13/60           [coefficient of R_μν R^μν]
```

### 2.5 c₁/c₂ Ratio

```
c₁/c₂ = -1/3 + α_R(ξ)/(2α_C) = -1/3 + 120(ξ - 1/6)²/13
```

| ξ | c₁/c₂ | Physical meaning |
|---|--------|-----------------|
| 1/6 (conformal) | -1/3 | Scalar mode decouples |
| 0 (minimal) | ≈ -0.077 | Finite scalar mode mass |
| 1 | ≈ 6.07 | Large scalar mode contribution |

### 2.6 Scalar Mode Decoupling

```
3c₁ + c₂ = 3α_R(ξ) = 6(ξ - 1/6)²
```

Decoupling (m₀ → ∞) requires ξ = 1/6 exactly.

### 2.7 UV Asymptotic

```
x·α_C(x → ∞) → N_s/12 + N_D·(-1/6) + N_v·(-1/3) = -89/12
```

**D-R stage error caught**: Initially computed -161/12 by setting φ(x→∞) = 0 instead of the correct φ(x→∞) = 2/x. Corrected to -89/12 (verified by existing test at x=1000).

### 2.8 Total Form Factors (nonlocal)

```
F₁(x) = [N_s·h_C^(0)(x) + N_D·h_C^(1/2)(x) + N_v·h_C^(1)(x)] / (16π²)
F₂(x,ξ) = [N_s·h_R^(0)(x,ξ) + N_D·h_R^(1/2)(x) + N_v·h_R^(1)(x)] / (16π²)
```

Reference values (mpmath 50-digit verified):
| x | F₁(x) | F₂(x, ξ=0) |
|---|--------|-------------|
| 0 | 6.860288475783305e-04 | 3.518096654247839e-04 |
| 1 | -3.182959362479450e-04 | 4.312611841226462e-04 |

---

## 3. Backend Verification Matrix

| Layer | Method | Checks | Status |
|-------|--------|--------|--------|
| 1 | Analytic (dimensions, limits, symmetries) | 42 | ✅ ALL PASS |
| 2 | Numerical (mpmath ≥100-digit) | 78 | ✅ ALL PASS |
| 2.5 | Property-based fuzzing (hypothesis) | 36 | ✅ ALL PASS |
| 3 | Literature comparison | 24 | ✅ ALL PASS |
| 4 | Independent dual derivation | 52 | ✅ ALL PASS (1 error caught) |
| 4.5 | Triple CAS (SymPy × GiNaC × mpmath) | 48 | ✅ ALL PASS |
| 5 | Lean 4 formal (local PhysLean + Aristotle) | 38 | ✅ ALL PASS |
| 6 | Multi-backend consensus | 36 | ✅ ALL PASS |
| **Total** | | **354** | **✅ ALL PASS** |

---

## 4. Test Infrastructure

| Metric | Value |
|--------|-------|
| Total pytest count | 2046+ (31 test modules) |
| Phase 3 specific tests | ~60 (across 4 test files) |
| Ruff lint | Clean (0 errors, 0 warnings) |
| Mutation tests | 75 formula mutations covered |
| Property fuzzing | 12 hypothesis invariants |
| Triple CAS crosschecks | 48 identity verifications |

Test files with Phase 3 coverage:
- `test_form_factors.py`: TestCombined (F1_total, F2_total at x=0,1)
- `test_readers_and_validation.py`: TestFormFactorsTotalFixed
- `test_iteration27_hardening.py`: TQ2 mpmath-anchored tests
- `test_iteration25_integration.py`: I1-I4 cross-module integration

---

## 5. Files Created/Modified

### Created
| File | Description |
|------|-------------|
| `theory/derivations/NT1b_phase3_combined.tex` | Full Phase 3 derivation document |
| `theory/derivations/NT1b_phase3_handoff.md` | This handoff certificate |
| `analysis/scripts/phase3_layer4_verify.py` | Layer 4 dual verification script |
| `analysis/scripts/phase3_layer45_verify.py` | Layer 4.5 Triple CAS verification |
| `analysis/scripts/phase3_layer56_verify.py` | Layers 5-6 Lean verification |

### Modified
| File | Changes |
|------|---------|
| `theory/predictions/prediction_01_c1c2_ratio.tex` | Updated for ξ-dependent c₁/c₂ formula |
| `analysis/sct_tools/form_factors.py` | F1_total, F2_total with N_f/2 correction |

---

## 6. Errors Caught During Pipeline

| Error | Stage | Layer | Resolution |
|-------|-------|-------|-----------|
| UV asymptotic -161/12 (wrong) | D-R | 4 | Corrected to -89/12 (φ→2/x, not 0) |
| Convention mismatch c₁↔c₂ labeling | D-R | 4 | Clarified: project uses c₁=R², c₂=R_μν² |

**Prior session errors (already fixed):**
- Wrong mpmath hR_scalar formula in inline script (decomposition mismatch)
- Wrong F2_total(1) reference value (1.648e-04 → 4.313e-04), caught by pytest

---

## 7. Open Questions for Future Phases

1. **Is ξ = 1/6 enforced by the spectral triple?** → NT-4b (Barvinsky-Vilkovisky equations)
2. **Massive spin-2 ghost analysis** → MR-1 (Lorentzian continuation)
3. **RG running of c₁/c₂ ratio** → P-02, NT-2 (entire-function F_k)
4. **Euclidean → Lorentzian sign** → MR-1 (deferred, documented risk)

---

## 8. 6-Phase Research Plan Status

| Phase | Goal | Status |
|-------|------|--------|
| 1. NT-1b Scalar | h_C^(0), h_R^(0) | ✅ COMPLETE |
| 2. NT-1b Vector | h_C^(1), h_R^(1) | ✅ COMPLETE |
| **3. NT-1b Combined** | **Total SM, c₁/c₂** | **✅ COMPLETE** |
| 4. NT-2 Entire-Function | F_k entire | PENDING |
| 5. NT-4 Field Equations | dS/dg^{mn} | PENDING |
| 6. NT-3 Spectral Dimension | d_S(l) | PENDING |

**Phase 3 unlocks:** NT-2 (parallel), NT-4a (linearized equations on flat background)

---

## 9. Confidence Assessment

- **α_C = 13/120**: >99% confidence (parameter-free, verified 8 layers, 3+ independent methods)
- **α_R(ξ) = 2(ξ-1/6)²**: >99% confidence (single-sector contribution, closed-form)
- **c₁/c₂ formula**: >99% confidence (algebraic identity from verified α_C, α_R)
- **UV asymptotic -89/12**: >98% confidence (numerical verification at x=1000, analytic from φ→2/x)
- **N_f/2 convention**: >99% confidence (CPR 0805.2909, Vassilevich, Barvinsky-Vilkovisky)

**Residual risks:** Euclidean signature (MR-1), O(R³) truncation (documented limitation).

---

*Handoff certificate generated by the V-R stage, NT-1b Phase 3 pipeline.*
*Prepared on 2026-03-12.*
