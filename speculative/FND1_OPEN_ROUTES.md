# FND-1 Open Routes

> Archive status: WORKING NOTE / QUARANTINED
> These are the remaining live routes after the bounded obstruction note and the first benchmark batch. They are not equal-strength candidates, and none of them preserves the original strongest form of direct finite-matrix synthesis.

## Ensemble Spectral Observables

### Why It Escapes The Current Bounded Obstruction

The bounded obstruction hits single finite matrices and the attempt to read standard spectral-action data directly from them. An ensemble observable weakens that demand by asking whether geometry is encoded only after averaging across many causal sets rather than inside one matrix.

### What New Mathematics It Would Require

- a precise ensemble-level observable tied to causal-set data rather than a post-hoc continuum surrogate
- concentration or stability estimates showing the observable survives seed variation and finite-size scaling
- a map from the ensemble observable back to the FND-1 success criteria, not just a visually smoother plot

### What Would Falsify It Quickly

- no ensemble observable can be found that improves on the current single-matrix failures
- the observable changes qualitatively under nearby normalization or cutoff choices
- the observable reproduces only rescaling artifacts rather than target geometry

## Coarse-Grained / Emergent Spectral Triple

### Why It Escapes The Current Bounded Obstruction

The bounded obstruction does not automatically cover a route where the causal set first produces coarse-grained geometric data and only then supports a spectral triple. This route abandons the strongest direct-synthesis claim, but it does escape the single-finite-matrix wall.

### What New Mathematics It Would Require

- an explicit reconstruction map from causal-set data to emergent geometric data
- a proof that the emergent algebra, Hilbert space, and operator are still causally sourced rather than imported by hand
- stability under changes of coarse-graining scale, so the emergent triple is not just a tuned reconstruction artifact

### What Would Falsify It Quickly

- the emergent triple requires a manifold, framing, or spin structure that cannot be derived from the causal set
- the coarse-graining map is non-robust or highly non-unique
- the recovered spectral data depends mainly on manual reconstruction choices rather than causal-set invariants

## Lorentzian/Krein Reformulation

### Why It Escapes The Current Bounded Obstruction

The bounded obstruction is aimed at the standard Chamseddine-Connes spectral action in a finite-matrix / self-adjoint setting. A Lorentzian or Krein reformulation changes the spectral observable and may handle retarded or indefinite structures more honestly than forced Euclideanization.

### What New Mathematics It Would Require

- a precise Lorentzian or Krein spectral observable that answers the same physical question as FND-1 rather than replacing it silently
- a controlled treatment of reality, positivity, and action functional meaning in the indefinite setting
- a benchmarkable finite or ensemble proxy that lets the route be tested numerically without collapsing back into ad hoc Wick tricks

### What Would Falsify It Quickly

- the reformulation only works after importing preferred time-splitting, framing, or Wick data from outside the causal set
- the resulting observable cannot be related back to curvature-sensitive targets
- the route never produces a real, stable action functional with a clear geometric interpretation

## Working Comparison

- `Ensemble spectral observables` is the most direct escape from the current finite-matrix obstruction, but it weakens the claim from single-object geometry to statistical geometry.
- `Coarse-grained / emergent spectral triple` is the cleanest route for eventually unblocking ALG-1, but only by admitting that the spectral triple lives after an emergent reconstruction step.
- `Lorentzian/Krein reformulation` is mathematically ambitious and still live, but it requires the largest amount of new formalism before it becomes testable.
