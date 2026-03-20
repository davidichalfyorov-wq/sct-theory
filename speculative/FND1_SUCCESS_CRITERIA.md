# FND-1 Success Criteria

## Analytic Criteria
- The construction must be stated precisely enough that the source of the operator, the algebra, and the representation are all explicit.
- The result must show how the spectral-action target is recovered or replaced, and the replacement must be named if it is not the standard Chamseddine-Connes action.
- The argument must separate finite-matrix behavior, continuum behavior, and ensemble behavior instead of conflating them.
- Any claimed obstruction must specify which class of constructions it covers and which it does not cover.

## Operator-Theoretic Criteria
- The candidate operator must have a well-defined mathematical status: self-adjoint, Krein-self-adjoint, or explicitly non-Hermitian, with consequences stated.
- Curvature sensitivity must be demonstrated or rigorously ruled out for the operator class under review.
- The construction must not depend on hidden background structure that is not part of the stated causal-set input.
- If truncation, symmetrization, or Wick rotation is used, the effect on the spectral-action target must be spelled out, not assumed.

## Numerical Criteria
- Single-run numerics do not count as success.
- Any numerical claim must be reproducible across multiple seeds and sizes.
- The result must include finite-size scaling behavior, sensitivity to normalization, and an explicit comparison against the declared continuum target.
- Error bars or equivalent stability indicators are required whenever a numerical claim is used to support a decision.

## What Does Not Count As Solving FND-1
- A toy operator that is only heuristically related to the causal set.
- A continuum approximation that drops the causal set from the definition.
- A truncation argument that preserves only the desired limit while changing the problem being solved.
- A claim of success that depends on one preferred normalization, one seed, or one size.
- A generalized Lorentzian observable that is not clearly mapped back to the FND-1 statement.
