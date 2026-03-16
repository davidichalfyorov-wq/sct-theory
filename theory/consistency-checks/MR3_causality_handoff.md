# MR-3 Causality: Handoff Certificate

**Date:** 2026-03-15
**Author:** David Alfyorov
**Task:** MR-3 (Causality in the Nonlocal Theory)
**Overall Verdict:** CONDITIONAL

## Summary

MR-3 analyzes the causality properties of the SCT nonlocal graviton propagator
across five sub-tasks. The theory preserves macrocausality (signal propagation
at v = c, exponential suppression of acausal effects) but violates microcausality
at scale ~1/Lambda, which is an inherent feature of the fakeon prescription.

## Sub-task Results

| Sub-task | Verdict | Key Result |
|----------|---------|------------|
| (a) Retarded G.F. | PASS | G_ret vanishes for spacelike; G_FK has acausal tail at 1/Lambda |
| (b) Kramers-Kronig | PASS | Trivially satisfied at tree level (Im[Pi_TT] = 0 on real axis) |
| (c) Classification | Class III | Entire with zeros; closest comparison is Anselmi's fakeon gravity |
| (d) IVP | PASS | Classicized IVP well-posed with 2 initial conditions (Anselmi-Calcagni 2025) |
| (e) Macrocausality | PASS | v_front = c; acausal effects decay as exp(-1.13*Lambda*r) |
| Microcausality | VIOLATED | At scale ~0.884/Lambda; inherent to fakeon (unobservable for Lambda >> accessible energies) |

## Key Numerical Results

- **Front velocity:** v_front = c = 1 (Paley-Wiener theorem)
- **Lorentzian ghost mass:** m_ghost = sqrt(1.2807)*Lambda = 1.1317*Lambda
- **Acausal decay length:** l_a = 0.884/Lambda
- **Ghost residue:** |R_L| = 0.538 (suppressed 46% vs Stelle)
- **UV limit:** Pi_TT(z -> +inf) = -83/6
- **KK:** Im[Pi_TT] = 0 on entire real axis (tree level)
- **IVP:** 2 initial conditions (classicized, Anselmi-Calcagni 2025)
- **Type-C correction bound:** < 5.003e-4 (CL M-test)

## Verification

- **Test suite:** 83/83 PASS (test_mr3_causality.py)
- **Independent re-derivation:** MR3-DR confirmed all results, 0 discrepancies
- **Spot-checks at dps=100:** 4/4 PASS
- **Cross-checks:** Ghost catalogue matches MR-2, GZ, CL, FK pipelines

## Artifacts

| File | Location |
|------|----------|
| Main script | analysis/scripts/mr3_causality.py |
| DR script | analysis/scripts/_mr3_dr_rederivation.py |
| Test suite | analysis/sct_tools/tests/test_mr3_causality.py |
| Results (D) | analysis/results/mr3/mr3_causality_results.json |
| Results (DR) | analysis/results/mr3/mr3_dr_rederivation_results.json |
| LaTeX | theory/consistency-checks/MR3_causality.tex |
| Verification report | docs/reviews/MR3_V_verification.md |

## Dependencies Satisfied

- NT-1, NT-1b, NT-2, NT-4a (all COMPLETE)
- MR-1 (Lorentzian formulation, COMPLETE)
- MR-2 (Unitarity, CONDITIONAL)

## Downstream Unblocked

- MR-7 (Scattering amplitudes) -- now has all prerequisites (NT-1b, NT-4a, MR-2, MR-3)
