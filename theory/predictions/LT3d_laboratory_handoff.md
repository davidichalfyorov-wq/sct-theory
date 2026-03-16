# LT-3d Handoff Certificate: Laboratory and Solar System Tests

**Task:** LT-3d (Laboratory and Solar System Tests)
**Pipeline:** L -> L-R -> D -> D-R -> V -> V-R
**Agent:** V (Verification)
**Date:** March 15, 2026
**Status:** COMPLETE

---

## 1. Primary Result

**Lambda_min = 2.565 x 10^{-3} eV** (95% CL, Eot-Wash torsion balance, Lee+ 2020)

Corresponding spin-2 Yukawa range: **lambda_1 < 35.8 um**

This is the tightest laboratory constraint on the SCT spectral scale.

---

## 2. SCT Modified Potential (Parameter-Free)

    V(r)/V_N(r) = 1 - (4/3) exp(-m_2 r) + (1/3) exp(-m_0 r)

- **alpha_1 = -4/3** (spin-2 massive ghost, repulsive) -- FIXED by SM field content
- **alpha_2 = +1/3** (scalar mode, attractive) -- FIXED; decouples at xi = 1/6
- **m_2 = Lambda * sqrt(60/13) = 2.1483 * Lambda**
- **m_0 = Lambda * sqrt(6) = 2.4495 * Lambda** (at xi = 0)
- **m_2/m_0 = sqrt(10/13) = 0.8771** (parameter-free)
- **V(0)/V_N(0) = 0** (finite potential at origin)
- **V(inf)/V_N(inf) = 1** (Newtonian recovery)
- V(r) is monotonically increasing for all r > 0

---

## 3. Experimental Constraints Summary

| Experiment | Constrains SCT? | Lambda_min (eV) | Notes |
|---|---|---|---|
| **Eot-Wash (Lee 2020)** | **YES** | **2.565 x 10^{-3}** | Only constraining experiment |
| Casimir (Chen 2016) | No | -- | |alpha|_95% >> 4/3 at sub-um scales |
| Atom interferometry | No | -- | |alpha|_95% ~ 10^8 at cm-m scales |
| LLR | No | -- | exp(-1.07 x 10^{13}) = 0 |
| GP-B | No | -- | exp(-1.95 x 10^{11}) = 0 |
| MICROSCOPE | No | -- | Universal coupling: no WEP violation |
| Cassini (PPN-1) | No | 6.3 x 10^{-18} | Vastly weaker than Eot-Wash |

---

## 4. xi-Dependence

Lambda_min is independent of the Higgs non-minimal coupling parameter xi because:
- The bound is set by the spin-2 Yukawa coupling |alpha_1| = 4/3
- This coupling is xi-independent (determined solely by the Weyl^2 sector)
- The scalar Yukawa coupling |alpha_2| = 1/3 < 4/3 always gives a weaker bound

At xi = 1/6 (conformal coupling): scalar decouples entirely, only one Yukawa term remains.

---

## 5. Consistency with PPN-1

| Check | Result |
|---|---|
| LT-3d bound tighter than PPN-1? | YES (2.565 > 2.38 meV, +7.8%) |
| PPN corrections at 1 AU | 0 (exp(-4.2 x 10^{15}) = 0) |
| Passes Cassini? | YES (trivially) |

The improvement over PPN-1 is because PPN-1 used the conservative |alpha| = 1 crossing (at 38.6 um), while LT-3d uses the correct SCT coupling |alpha_1| = 4/3 (crossing at 35.8 um).

---

## 6. Verification Summary

- **Tests:** 85/85 PASS (test_lt3d_laboratory.py)
- **Regression:** 3474 PASS (1 pre-existing Lean timeout, unrelated to LT-3d)
- **8-Layer verification:** All layers PASS (see docs/reviews/LT3d_V_verification.md)
- **Dual derivation:** Agent D (interpolation) and Agent DR (slope + minimal interp) agree to < 0.001%
- **Figures:** 5 PDFs generated (exclusion, exclusion_dr, lambda_min_vs_xi, potential_deviation, casimir_correction)
- **Ruff:** All 3 scripts clean

---

## 7. Deliverables

| File | Path | Status |
|---|---|---|
| Main script | analysis/scripts/lt3d_laboratory.py | COMPLETE (1079 lines) |
| DR script | analysis/scripts/lt3d_laboratory_dr.py | COMPLETE (831 lines) |
| Test file | analysis/sct_tools/tests/test_lt3d_laboratory.py | 85/85 PASS |
| Results JSON | analysis/results/lt3d/lt3d_laboratory_results.json | COMPLETE |
| DR Results JSON | analysis/results/lt3d/lt3d_laboratory_dr_results.json | COMPLETE |
| Exclusion plot | analysis/figures/lt3d/lt3d_exclusion_unified.pdf | 30 KB |
| DR Exclusion plot | analysis/figures/lt3d/lt3d_exclusion_dr.pdf | 27 KB |
| Lambda_min vs xi | analysis/figures/lt3d/lt3d_lambda_min_vs_xi.pdf | 19 KB |
| Potential deviation | analysis/figures/lt3d/lt3d_potential_deviation.pdf | 18 KB |
| Casimir correction | analysis/figures/lt3d/lt3d_casimir_correction.pdf | 20 KB |
| D derivation | docs/reviews/LT3d_D_derivation.md | COMPLETE |
| DR review | docs/reviews/LT3d_DR_review.md | COMPLETE |
| V verification | docs/reviews/LT3d_V_verification.md | COMPLETE |
| Handoff | theory/predictions/LT3d_laboratory_handoff.md | This document |

---

## 8. Key Numbers (for downstream tasks)

- Lambda_min = 2.5645 x 10^{-3} eV (from Eot-Wash)
- lambda_1 crossing = 35.82 um
- hbar*c = 1.97326980459 x 10^{-7} eV*m
- m_2/Lambda = sqrt(60/13) = 2.14834462211829
- m_0/Lambda(xi=0) = sqrt(6) = 2.44948974278318
- m_2/m_0(xi=0) = sqrt(10/13) = 0.87705801930703
- alpha_1/alpha_2 = -4 (exact)
- V(0)/V_N = 0 (exact, at general xi)
- V(0)/V_N = -1/3 (exact, at xi = 1/6)
- dV/dr|_{r=0} coefficient = 2.048 * Lambda (positive, monotonicity guaranteed)
