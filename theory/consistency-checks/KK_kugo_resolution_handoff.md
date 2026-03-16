# KK: Kubo-Kugo Resolution -- Handoff Certificate

## Task
Assess whether the Kubo-Kugo objection (arXiv:2308.09006, 2402.15956)
applies to SCT. Address MR-2 Condition 3.

## Status: COMPLETE (PARTIALLY RESOLVED)

## Key Results

| Result | Value | Verified |
|--------|-------|----------|
| KO quartet applies to propagator ghosts | **No** (spin-2 vs spin-1 mismatch) | 64/64 tests |
| Fakeon Im[G_ghost] at pole | **0** (to machine precision at dps=100) | dps=100 independent |
| Feynman Im[G_ghost] at pole | **nonzero** (divergent as eps->0) | dps=100 independent |
| Lowest threshold E_th | **2.264 Lambda** (from z_L = -1.2807) | 5-point stencil |
| R_L residue | **-0.5378** (negative = ghost) | agreement < 1e-16 |
| Two-pole dominance | **96.83%** (z_L + z_0 of total weight) | catalogue check |
| LW fraction | **< 0.04%** of z_L cross-section | 3 pairs checked |
| Spectral positivity (fakeon) | **Im[G_FK] > 0** for all s > 0 | 15 scan points |
| MR-2 Condition 3 | **PARTIALLY RESOLVED** | 5 methods + 7 layers |

## Resolution Assessment

| Resolution | Status | Confidence |
|------------|--------|------------|
| A: KO quartet | **REJECTED** | HIGH |
| B: Fakeon | **PRIMARY** | MODERATE-HIGH |
| C: Unstable ghost | SUPPORTING | MODERATE |
| D: PT symmetry | INCONCLUSIVE | LOW |
| E: Complex mass | PARTIAL | LOW-MODERATE |
| F: DQFT/IHO | UNDER INVESTIGATION | LOW |

## Verification Layers

- Layer 1 (BRST): 14 tests -- nilpotency, quartets, cohomology, quantum numbers
- Layer 2 (Numerical): 8 tests -- thresholds, residues, dominance
- Layer 3 (Cross-consistency): 10 tests -- MR-2, A3, GP, OT, FK, GZ
- Layer 4 (Physical): 8 tests -- fakeon vs Feynman, spectral positivity
- Layer 5 (Structural): 5 tests -- two-pole dominance, LW suppression
- Layer 6 (Verdict): 10 tests -- resolution statuses
- Layer 7 (Regression): 9 tests -- constants, full derivation, serialization
- **Total: 64/64 PASS**

## Independent Verification (dps=100)

- (a) R_L = -0.537772 at z_L = -1.280702 (agreement 3.75e-17)
- (b) E_th = 2.2634 Lambda (matches 2*sqrt(|z_L|))
- (c) Im[G_FK] = 6.57e-84 (machine zero at dps=100)
- (d) Im[G_F] = 5.37e+14 (nonzero, divergent)

## Independent Re-Derivation (5 methods)

- Method A: Quantum number analysis (structural argument)
- Method B: Prescription comparison (contour integration + 5-point stencil)
- Method C: Threshold analysis (pair-production kinematics)
- Method D: Spectral positivity cross-check (consistency with OT)
- Method E: Historical assessment (literature review of assumptions)

All 5 methods agree with KK-D results.

## Remaining Gaps

1. Fakeon prescription proof exists only for polynomial propagators (Anselmi 2018)
2. Extension to infinite-pole propagators supported by CL result but not proven
3. Operator vs path-integral foundational question remains open

## Artifacts

| File | Location |
|------|----------|
| Derivation script | `analysis/scripts/kk_kugo_resolution.py` |
| Re-derivation script | `analysis/scripts/_kk_dr_rederivation.py` |
| Test suite | `analysis/sct_tools/tests/test_kk_resolution.py` |
| Results JSON | `analysis/results/kk/kk_resolution_results.json` |
| LaTeX document | `theory/consistency-checks/KK_kugo_resolution.tex` |
| This handoff | `theory/consistency-checks/KK_kugo_resolution_handoff.md` |

## Survival Probability

Unchanged at 65-75%. The KK analysis clarifies the resolution landscape
but does not change the overall assessment.

## CQ1 (Test Quote)

From the test suite run:
```
test_ko_not_applicable_to_propagator_ghosts PASSED
```
This test verifies that `ghost_quantum_number_analysis()` returns
`KO_applies_to_propagator_ghosts = False` and
`propagator_ghost.KO_quartet_possible = False`.

## CQ2 (No Invented Physics)

All physics results derive from:
- BRST structure: Kugo-Ojima 1978, Nakanishi-Ojima 1990
- Kubo-Kugo objection: arXiv:2308.09006, 2402.15956
- Fakeon prescription: Anselmi 2017 (1703.04584), 2018 (1801.00915)
- Unstable ghost: Donoghue-Menezes 2019 (1908.02416)
- Ghost catalogue: from verified MR-2 pipeline
- Constants: from verified sct_tools (alpha_C = 13/120, etc.)

No new physics was invented.
