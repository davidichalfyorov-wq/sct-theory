# FND-1 Bounded Obstruction

> Archive status: WORKING NOTE / QUARANTINED
> This note restates the strongest bounded obstruction currently supported inside `speculative/`. It is narrower than the historical verdict files and should be read together with [FND1_STATEMENT.md](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/FND1_STATEMENT.md), [FND1_SUCCESS_CRITERIA.md](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/FND1_SUCCESS_CRITERIA.md), [FND1_STATUS.md](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/FND1_STATUS.md), and [FND1_OPERATOR_MATRIX.md](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/FND1_OPERATOR_MATRIX.md).

## Claim Covered

This note covers the following scope only:

- finite matrices built directly from a causal set
- the standard Chamseddine-Connes spectral-action machinery based on heat-kernel / zeta-function asymptotics
- the reviewed candidate routes in this archive: SJ-style surrogates, BD-style scalar operators, truncated-SJ rescue attempts, and BD-like Hermitianization ideas

Within that scope, the archive supports a bounded obstruction:

- a finite causal-set matrix does not carry the pole structure used by standard Seeley-DeWitt extraction
- the reviewed SJ-style route keeps the desired self-adjoint flavor only by moving to an operator that is not established as a curvature-sensitive Dirac replacement
- the reviewed BD-style route carries curvature information in the scalar retarded sense but does not fit the standard self-adjoint spectral-triple role

## Claim Not Covered

This note does not automatically cover:

- ensemble observables defined across many causal sets
- emergent or coarse-grained spectral triples built after a separate continuum reconstruction step
- alternative Lorentzian spectral observables that are not the standard Chamseddine-Connes spectral action
- future constructions that derive extra spin, framing, or Wick data directly from causal-set input rather than importing them by hand

## Finite-Matrix Obstruction

The cleanest bounded result in the archive is the finite-matrix obstruction.

For a causal set with `N` elements, any directly constructed operator in the reviewed finite route is an `N x N` matrix with a finite spectrum. In that setting, the associated spectral zeta function is a finite sum over eigenvalues. A finite sum of analytic terms does not generate the pole structure that the standard heat-kernel / zeta-function machinery uses to read off Seeley-DeWitt coefficients.

That is enough to support the following bounded statement:

- on a single finite matrix, the standard spectral-action extraction of local curvature data is not available in the way it is on a continuum spectral triple

This is an obstruction to the reviewed direct finite-matrix route. It is not, by itself, a statement about every large-`N`, ensemble, or emergent construction.

## SJ Route Obstruction

The reviewed SJ route starts from the Pauli-Jordan matrix or an SJ-inspired Dirac surrogate built from it. In the archive, this route is attractive because it keeps the operator close to the self-adjoint side of the spectral-triple story.

The obstruction is not that the route is meaningless. The obstruction is narrower:

- the reviewed SJ object behaves like a propagator / on-shell object rather than a demonstrated curvature-sensitive Dirac operator
- the archive numerics tied to this route do not establish stable recovery of the declared continuum targets
- the route relies on an inverse or square-root style reinterpretation to become Dirac-like in the first place

In the language of [FND1_STATUS.md](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/FND1_STATUS.md), this leaves SJ-as-direct-Dirac at `HEURISTIC / NOT ESTABLISHED`.

## BD Route Obstruction

The reviewed BD route is the clearest split case in the archive.

On the positive side:

- the BD d'Alembertian is the strongest curvature-sensitive scalar operator in the archive
- its continuum target is exactly why it remains relevant to FND-1

On the obstructed side:

- the direct retarded operator is non-Hermitian and therefore does not slot cleanly into the standard self-adjoint spectral-triple role
- Hermitianization, symmetrization, or Wick-style rescue moves change the operator being used and are not verified here to preserve the desired gravitational content
- the archive does not provide a successful direct path from bare BD to a standard spectral-action computation that meets the current FND-1 criteria

That is why the current status is best summarized as `PLAUSIBLE OBSTRUCTION`, not as a settled impossibility statement.

## Open Loopholes

The archive still leaves several live loopholes outside the bounded obstruction above:

- an ensemble-level observable that recovers geometric information statistically rather than from one finite matrix
  Evidence needed to keep this loophole alive: a benchmarkable ensemble observable with finite-size scaling and seed-stable behavior tied back to the FND-1 success criteria.
- a coarse-grained or emergent spectral triple reconstructed after the causal-set stage
  Evidence needed: an explicit reconstruction map showing what data survives coarse-graining and why the resulting spectral object is still meaningfully causal-set-derived.
- a Lorentzian or Krein-style observable that can be mapped back to the FND-1 statement without pretending to be the unchanged Euclidean spectral action
  Evidence needed: a precise replacement observable plus a clear argument for how it answers the same direct-synthesis question rather than changing the problem.
- a future derivation of extra spin or framing data from causal-set structure itself, rather than from an imported background scaffold
  Evidence needed: a derivation of the extra structure from causal-set input, not an external coordinate choice or hand-added framing.

## Working Verdict

The strongest current verdict supported by `speculative/` is therefore:

- the reviewed direct finite-matrix spectral-action route faces a plausible obstruction
- the reviewed SJ and BD candidate routes each fail one side of the current FND-1 criteria
- the broader question remains open only in loopholes that lie outside the direct finite-matrix route

This is enough to guide the next step of the decision program: benchmark the remaining loopholes honestly, without treating the historical archive language as already settled theory.
