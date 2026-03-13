# NT-2 Handoff Certificate

**Phase:** NT-2  
**Date:** 2026-03-12  
**Status:** PASS

## Trace
- `NT2-L`: literature note recorded in `theory/derivations/NT2_literature.tex`.
- `NT2-LR`: literature audit recorded in `docs/reviews/NT2_L_review.md`.
- `NT2-D`: derivation and numerical artifacts recorded in `theory/derivations/NT2_entire_function.tex` and `analysis/results/nt2/`.
- `NT2-DR`: derivation audit recorded in `docs/reviews/NT2_D_review.md`.
- `NT2-V`: target pytest, growth scan, zero search, figures, and Lean-backed algebraic checks executed on the NT-2 artifacts.
- `NT2-VR`: handoff assembled after the target test slice and shared Lean build gate passed.

## Summary
- Complex-domain evaluators implemented in `analysis/scripts/nt2_entire_function.py`.
- Growth scan and Hadamard-style zero search implemented and executed.
- Lean phase-local algebraic lemmas added under `theory/lean/proofs/NT2/`.
- Public wrappers integrated in `analysis/sct_tools/entire_function.py`.
- The numerical zero search is recorded as bounded evidence over the scanned region, not as a standalone global ghost-freedom theorem.
- After the repaired NT-4a denominator scan, NT-2 should be read as an entire-function / analytic-control phase, not as a completed ghost-freedom proof.

## Key Results
- Pole-cancellation errors at `z = 1e-12` are at the `1e-15` level or smaller for all monitored quantities.
- `F1_total(0) = 6.860288475783286e-04`.
- `F2_total(0, xi=0) = 3.518096654247839e-04`.
- Growth scan:
  - finite-radius effective slope proxy `rho[F1] ~= 1.1859831641`
  - finite-radius effective slope proxy `rho[F2] ~= 1.3545030143`
  - finite-radius type proxies are near `0.24`
- Zero search:
  - no positive-real-axis zeros found for the legacy NT-2 proxy in the implemented scan window
  - first negative real zero near `-30.4816385005`
  - complex pair near `-30.47000817 +/- 12.96416990 i`

## Files Produced
- `analysis/results/nt2/nt2_snapshot.json`
- `analysis/results/nt2/nt2_growth_rate.json`
- `analysis/results/nt2/nt2_hadamard.json`
- `analysis/results/nt2/nt2_verify_phase_local.json`
- `analysis/results/nt2/nt2_verify_phase_cloud.json`
- `analysis/figures/nt2_growth_rate.pdf`
- `analysis/figures/nt2_complex_plane.pdf`
- `theory/lean/proofs/NT2/PhiEntire.lean`
- `theory/lean/proofs/NT2/PoleCancel.lean`
- `theory/lean/proofs/NT2/GhostFree.lean`

## Verification Notes
- Targeted pytest coverage for NT-2 passes.
- Shared Lean registry updated so `verify_phase_deep("NT-2")` is a first-class package check.
- Shared `build_sctlean()` gate passes after the NT-2 theorem registration.
- Independent external review completed and archived with the phase review record.
- Local-only phase deep verification passes with `5/5` identities and is recorded in `analysis/results/nt2/nt2_verify_phase_local.json`.
- Aristotle-backed cloud verification passes with `5/5` identities and is recorded in `analysis/results/nt2/nt2_verify_phase_cloud.json`.
- Full `analysis/run_ci.py` now passes with repository-wide lint, pytest, and canonical checks green.
