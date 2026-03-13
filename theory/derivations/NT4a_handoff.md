# NT-4a Handoff Certificate

**Phase:** NT-4a  
**Date:** 2026-03-12  
**Status:** PASS (with propagator-spectrum warning)

## Trace
- `NT4a-L`: literature note recorded in `theory/derivations/NT4a_literature.tex`.
- `NT4a-LR`: literature audit recorded in `docs/reviews/NT4a_L_review.md`.
- `NT4a-D`: derivation, projector code, propagator code, and Newtonian artifacts recorded in `theory/derivations/NT4a_linearized.tex` and `analysis/results/nt4a/`.
- `NT4a-DR`: derivation audit recorded in `docs/reviews/NT4a_D_review.md`.
- `NT4a-V`: standalone verifier executed, figures generated, and target pytest slice passed.
- `NT4a-VR`: handoff assembled after the target test slice and shared Lean build gate passed.

## Summary
- Symbolic linearization, propagator, Newtonian-potential, and standalone verification scripts implemented.
- Phase-local Lean proof files added under `theory/lean/proofs/NT4a/`.
- Public wrappers integrated in `analysis/sct_tools/propagator.py`.
- Shared Lean registry updated so `verify_phase_deep("NT-4a")` is available after integration.
- Off-shell gauge-invariance and Bianchi checks are executed before the TT-gauge propagator reduction.

## Key Results
- `checks_passed = 24`, `checks_failed = 0` in `analysis/results/nt4a/nt4a_verify.json`.
- `Pi_TT(0) = 1`, `Pi_scalar(0) = 1`.
- `Pi_TT(1) ~= 0.8994734409`.
- `Pi_TT(z)` crosses zero on the positive real axis at `z ~= 2.4148388899`.
- `Pi_scalar(1, xi=0) ~= 1.2043061095`.
- Scalar mode coefficient:
  - `6 (xi - 1/6)^2 = 1/6` at `xi = 0`
  - `0` at `xi = 1/6`
- Newtonian-potential ratios:
  - at `r = 10`, all sampled `xi` values are already within `0.3%` of the Newtonian limit
  - at `r >= 100`, the ratio is numerically `1.0` in the generated report
  - at exact conformal coupling `xi = 1/6`, the local short-distance `1/r` behavior reappears because the scalar Yukawa cancellation is absent

## Files Produced
- `analysis/results/nt4a/nt4a_linearized_identities.json`
- `analysis/results/nt4a/nt4a_propagator_snapshot.json`
- `analysis/results/nt4a/nt4a_newtonian.json`
- `analysis/results/nt4a/nt4a_verify.json`
- `analysis/results/nt4a/nt4a_verify_phase_local.json`
- `analysis/results/nt4a/nt4a_verify_phase_cloud.json`
- `analysis/figures/nt4a_propagator.pdf`
- `analysis/figures/nt4a_scalar_mode.pdf`
- `analysis/figures/nt4a_newtonian_potential.pdf`
- `theory/lean/proofs/NT4a/Propagator.lean`
- `theory/lean/proofs/NT4a/GaugeInv.lean`

## Verification Notes
- Targeted pytest coverage for NT-4a passes.
- Gauge-invariance and Bianchi checks pass off shell on deterministic random-momentum and symmetric-tensor samples used by the standalone verifier.
- The standalone verifier now fails hard on any nonzero failure count instead of silently writing a JSON report.
- The verifier explicitly records the positive-real TT zero rather than assuming ghost-freedom.
- Shared `build_sctlean()` gate passes after the NT-4a theorem registration.
- Independent external review completed and archived with the phase review record.
- Local-only phase deep verification passes with `4/4` identities and is recorded in `analysis/results/nt4a/nt4a_verify_phase_local.json`.
- Aristotle-backed cloud verification passes with `4/4` identities and is recorded in `analysis/results/nt4a/nt4a_verify_phase_cloud.json`.
- Full `analysis/run_ci.py` now passes with repository-wide lint, pytest, and canonical checks green.
