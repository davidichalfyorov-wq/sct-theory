# MR-8 Nonperturbative Selector Endpoint

Date: 2026-04-09

## Purpose

This note records the current endpoint of the MR-8 program after the working-note chain `v45-v56`.

The aim is to state, in one place and with clean status labels, what is already proven about nonperturbative external selector channels for SCT, what remains conditional, and what the remaining loophole actually is.

## Background

The prior archive established:

- the naive spectral-action path integral diverges;
- corrected two-sided Gaussian local completions exist sectorwise;
- global Gaussian gluing fails;
- the correct global object is a pro-torsor of local completed measure classes with density-valued observables;
- internal selectors fail;
- external selectors, when they exist, are classified by sector weights plus densities;
- the unresolved post-`v44` issue is whether a physically motivated state-independent finite-rank channel tower can realize those external states levelwise.

The present note concerns only that last issue.

## Theorem Chain

Let `\Omega_N` be one finite-rank truncation space, `\mathcal F_N = \mathcal B(\Omega_N)`, and let

\[
B_N:\Omega_N\to Y_N
\]

be a measurable finite-rank channel with standard Borel codomain.

### Theorem 1

If `B_N` is separating,

\[
\sigma(B_N)=\mathcal F_N,
\]

then `B_N` is injective.

Status: `PROVEN` in `MR8_working_note_v51.md`.

### Theorem 2

If `B_N` is injective and `Y_N` is standard Borel, then `B_N` is a Borel isomorphism between `\Omega_N` and its image. In particular, there exists a measurable inverse on `B_N(\Omega_N)`, so `B_N` is a lossless re-encoding of the full finite-rank state.

Status: `PROVEN` in `MR8_working_note_v52.md`.

### Theorem 3

If `B_N` is injective, then for every ambient external channel `B:\Omega\to Y` with standard Borel codomain, the `v44` sufficiency condition is automatic:

\[
\operatorname{Law}_\mu(B\mid \mathcal F_N)
\]

depends measurably only on `B_N\circ \pi_N`.

Status: `PROVEN` in `MR8_working_note_v55.md`.

### Corollary 4

The only finite-rank channel towers that can possibly solve the `v42-v44` selector burden are the tautological ones: those equivalent, rank by rank, to the identity truncation.

Status: `PROVEN`.

This is the main unconditional mathematical endpoint.

## Eliminated Natural Routes

The following natural finite-rank selector patterns have theorem-level negative results:

- natural conjugacy-invariant spectral channels;
- compression-only boundary channels;
- boundary restrictions of equivariant operator functionals;
- standard SJ boundary truncations under full basis covariance;
- proper-subspace SJ-type truncations under the weaker relative phase-covariance hypothesis;
- finite families of correlators on a proper test subspace;
- finite continuous feature maps of output dimension strictly smaller than `\dim_\mathbb R \Omega_N`.

Status: `PROVEN` in `MR8_working_note_v46.md` through `MR8_working_note_v51.md`.

## Conditional No-Go

Introduce the finite-rank non-tautology axiom:

> an admissible selector channel may not be Borel-equivalent to the identity truncation.

Under this axiom, the tautological loophole is removed. Therefore no admissible external selector tower can satisfy the `v42-v44` program.

Status: `CONDITIONAL`, proved in `MR8_working_note_v53.md`.

Equivalently, under non-tautology, `Option C` holds.

## Interpretive Reading

The archive language of `v44-v45` already treats:

- state-dependent tautological realizations as vacuous, and
- the intended selector channel as a genuine coarse external readout.

On that reading, the non-tautology principle is not a foreign extra axiom but an explicit formulation of the admissibility criterion already implicit in the program.

Status: `INTERPRETIVE BUT ARCHIVE-GROUNDED`, argued in `MR8_working_note_v54.md`.

## Current Endpoint

The current MR-8 endpoint is:

- `PROVEN`: `Option A` is mathematically complete; every nontrivial coarse finite-rank selector investigated so far fails; the only finite-rank channels that can survive are tautological re-encodings of the full truncation.
- `CONDITIONAL`: if tautological re-encodings are forbidden, then `Option C` is proven.
- `INTERPRETIVE`: under the intended `v44-v45` reading, the physically meaningful `Option B` space is exhausted, so the negative closure is likely the correct reading of MR-8.

## Practical Conclusion

There are only two scientifically honest final presentations left:

1. Adopt non-tautology explicitly and state a formal no-go theorem for nontrivial external selector towers.
2. Remain maximally cautious and state that `Option B` survives only tautologically, while `Option C` is conditionally closed.

The original search for a natural coarse physical selector among `B1-B5` is effectively finished.

## Sources Inside This Project

- `speculative/MR8_working_note_v45.md`
- `speculative/MR8_working_note_v46.md`
- `speculative/MR8_working_note_v47.md`
- `speculative/MR8_working_note_v48.md`
- `speculative/MR8_working_note_v49.md`
- `speculative/MR8_working_note_v50.md`
- `speculative/MR8_working_note_v51.md`
- `speculative/MR8_working_note_v52.md`
- `speculative/MR8_working_note_v53.md`
- `speculative/MR8_working_note_v54.md`
- `speculative/MR8_working_note_v55.md`
- `speculative/MR8_working_note_v56.md`
