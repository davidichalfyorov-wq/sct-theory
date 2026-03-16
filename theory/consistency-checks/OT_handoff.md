# OT Handoff Certificate: One-Loop Optical Theorem with Fakeon Prescription

**Task:** OT (One-Loop Optical Theorem)
**Date:** March 14, 2026
**Status:** PASS
**Pipeline:** OT-L → OT-LR → OT-D → OT-DR → OT-V → OT-VR
**Verification checks:** 67 new + 2758 regression = 2825 total, 0 failures

---

## 1. Scope

| Sub-question | Verdict |
|---|---|
| Is the one-loop optical theorem satisfied in SCT? | **YES** |
| Does the fakeon prescription exclude ghosts from the unitarity sum? | **YES** |
| Is Im[G_dressed(s)] > 0 for all s > 0? | **YES** (spectral positivity) |
| Does the central charge C_m = 283/120 match Anselmi? | **YES** (exact) |
| Is the N_eff reconciliation with HLZ correct? | **YES** (algebraic identity) |
| Does the 2-pole truncation capture the dominant physics? | **YES** (Type C < 0.2%) |
| Are all cross-checks with A3/GP consistent? | **YES** (4/4 PASS) |

## 2. Key Results

### 2.1 Spectral Positivity Theorem
```
Im[G_dressed(s)] = Im[Σ_matter(s)] / |s·Π_TT - Σ|² > 0   for all s > 0
```
This follows from:
1. Π_TT(z) entire with real coefficients ⟹ real for real z
2. Im[Σ_matter(s)] > 0 from Cutkosky rules (SM matter cuts)
3. Fakeon prescription excludes ghost cuts from Im[Σ]

### 2.2 Central Charge
- **C_m = 283/120** (Anselmi convention)
- **N_eff_width = 143.5** (HLZ convention)
- Algebraic identity: C_m = (2·N_eff - N_s)/120 ✓

### 2.3 Anselmi Absorptive Polynomials
| Species | P_Φ(0) | Weight ratio |
|---|---|---|
| Scalar | 1/120 | 1 |
| Dirac | 1/20 | 6 |
| Vector | 1/10 | 12 |

### 2.4 Ghost Exclusion Mechanism
- Fakeon prescription: ghost does NOT appear in unitarity sum
- Im[Σ_FK] = Im[Σ_matter] > 0 always (ghost cut removed)
- Ghost/matter ratio at high s: ~4.5% (negligible even without fakeon)

### 2.5 Stelle Comparison
| Property | Stelle | SCT |
|---|---|---|
| Ghost mass m/Λ | 2.148 | 1.132 |
| Residue |R| | 1.000 | 0.538 |
| Width ratio | 1.000 | 0.149 |

### 2.6 N-Pole Convergence
- Unitarity independent of N (structural theorem)
- Ghost width depends only on z_L, R_L (N-independent)
- Type C fraction: 0.12% of total residue sum
- Type C correction at s ≤ 2: < 0.1%

## 3. Cross-Check Matrix

| Cross-check | Pipeline | Result |
|---|---|---|
| Width Γ_OT / Γ_A3 | A3 | 1.0000000000 |
| Im[k²_pole] < 0 | GP | Matches sign chain |
| C_m / N_eff identity | Analytic | Verified |
| C_Γ_OT / C_Γ_A3 | A3 | 1.0000000000 |

## 4. Artifacts

| Artifact | Location |
|---|---|
| Computation script | `analysis/scripts/ot_optical_theorem.py` |
| Test suite | `analysis/sct_tools/tests/test_ot_optical_theorem.py` |
| Results JSON | `analysis/results/ot/ot_optical_theorem_results.json` |
| LaTeX document | `theory/consistency-checks/OT_optical_theorem.tex` |
| Compiled PDF | `theory/consistency-checks/OT_optical_theorem.pdf` |
| Handoff certificate | `theory/consistency-checks/OT_handoff.md` |

## 5. Test Summary

| Layer | Count | Status |
|---|---|---|
| Layer 1 (Analytic) | 14 | PASS |
| Layer 2 (Numerical) | 8 | PASS |
| Layer 3 (Cross-consistency) | 7 | PASS |
| Layer 4 (Physical) | 8 | PASS |
| Layer 5 (Fakeon) | 8 | PASS |
| Layer 6 (Convergence) | 6 | PASS |
| Layer 7 (Regression) | 11 | PASS |
| Layer 8 (Input stability) | 5 | PASS |
| **Total OT** | **67** | **PASS** |
| Full regression | 2825 | PASS |

## 6. Impact on MR-2 Conditions

### Condition 2: Donoghue-Menezes mechanism for nonlocal theories
**Partially resolved.** The one-loop optical theorem is satisfied. The ghost
is "purely virtual" — it contributes to Re[T] but not Im[T]. The spectral
positivity theorem (Section 2.1) is independent of the pole structure,
relying only on the entire-function property of Π_TT and the positivity
of matter cuts.

### Survival Probability Update
**60-70% → 70-80%** (one-loop unitarity confirmed)

### Remaining for full resolution of Condition 2
- Higher-loop unitarity (two-loop and beyond)
- Non-perturbative unitarity (if applicable)
- Explicit all-orders proof following Anselmi's ACE framework

## 7. Verdict

**PASS.** The one-loop optical theorem is verified for the SCT graviton
propagator with the fakeon prescription. Unitarity is maintained at one loop.
The spectral positivity condition Im[G_dressed] > 0 holds at all tested points
and is proven structurally. All cross-checks with A3, GP, and FK pipelines
are consistent to machine precision.
