# Evaluation Standards for Proposed Solutions

Any proposed solution to an open problem in this collection must satisfy
the standards below before it can be considered for integration into the
theory. These standards exist to maintain the level of rigour established
throughout the SCT programme (4000+ quantitative checks, 105 formally
verified theorems, 8-layer verification pipeline).

---

## 1. Consistency with established results

Every proposed solution must be checked against the verified results
registry (`VERIFIED-RESULTS.md`). If a proposed solution contradicts
any entry in that registry, it is rejected unless the contradiction
is explicitly identified and the registry entry is shown to be in error
with a concrete counterexample.

In particular:
- alpha_C = 13/120 is parameter-free and formally verified. Any
  derivation that obtains a different value is wrong.
- The master function phi(x) = e^{-x/4} sqrt(pi/x) erfi(sqrt(x)/2)
  is exact for the cutoff f(u) = e^{-u}. Do not approximate it.
- CJ = 0 on de Sitter is exact (50-seed verification). Any formula
  that predicts nonzero CJ on conformally flat vacuum is incorrect.
- (d!)^2 C(2d,d)(2d+1) = (2d+1)! for all d. This is formally
  verified in Lean 4. Do not re-derive it with errors.

## 2. Engagement with failed approaches

Each problem file lists approaches that have been tried and failed.
A proposed solution must explicitly address each relevant failed
approach and explain either:

(a) why the proposed approach differs structurally from the failed one, or
(b) where the failed approach made a specific error that the new
    approach avoids.

A solution that re-applies a documented failed approach without
addressing the stated failure mode is rejected.

## 3. Concrete verification at test points

Any analytical result must be accompanied by at least one independent
numerical check. Specifically:

- If the result is a formula: evaluate it at three specific parameter
  values and compare with independent computation (not derived from
  the same formula).
- If the result is an identity: verify at d = 1, 2, 3, 4, 5 by
  direct substitution.
- If the result involves an integral: provide a numerical quadrature
  to at least 6 significant figures.
- If the result is a bound: verify that known exact values satisfy
  the bound.

Results without numerical checks are treated as conjectures, not proofs.

## 4. Precise references

Every cited result from the literature must include:

- The specific paper (arXiv ID or DOI).
- The specific equation, theorem, or proposition number.
- The exact statement being used (not a paraphrase).

If a cited paper does not contain the claimed result, the solution is
rejected. Referencing a paper "for related ideas" without a specific
equation is not acceptable as evidence.

## 5. Dimensional and limit consistency

Every formula must pass:

- **Dimensional analysis:** all terms have consistent dimensions.
  State the unit convention (natural units, geometrized, SI) and
  verify.
- **Known limits:** the formula must reduce correctly in at least
  two limiting cases (e.g., flat space, large r, weak field,
  d = 4 specialisation from general d).
- **Sign convention:** state the metric signature and Riemann
  convention used. Verify compatibility with the SCT convention
  (see `GLOSSARY.md`).

## 6. Scope of validity

Every result must state explicitly:

- What assumptions are required (vacuum, d = 4, perturbative
  curvature, specific cutoff function f, etc.).
- What the domain of validity is (range of N, range of epsilon,
  range of coupling xi).
- Whether the result is exact or approximate, and if approximate,
  what the error estimate is.

Unqualified universal claims ("for all spacetimes," "at all orders")
are rejected unless proven with the corresponding generality.

## 7. Distinction between proof and evidence

Solutions must clearly separate:

- **Proven:** rigorous mathematical derivation with every step justified.
- **Verified:** confirmed numerically or by independent computation,
  but without a complete proof.
- **Conjectured:** supported by evidence but not proven.
- **Heuristic:** motivated by analogy or dimensional analysis.

Presenting a heuristic argument as a proof, or numerical evidence as
a derivation, is grounds for rejection.

## 8. Independence from the problem statement

A solution must not be circular. Specifically:

- The conclusion must not appear among the premises.
- Fitting a model to data and then claiming the data "confirms"
  the model is not a derivation (it is a consistency check).
- If the solution uses the CJ bridge formula to derive a property
  of the CJ bridge formula, the circularity must be identified.

## 9. Reproducibility

- All numerical computations must specify: random seed, N, epsilon,
  T, number of seeds, and the exact predicate used (flat, jet, exact).
- All analytical computations must be reproducible step-by-step
  from the stated premises.
- "It can be shown that..." without showing it is not acceptable.

## 10. Falsifiability

A proposed solution must identify at least one prediction that could
falsify it. If the solution is unfalsifiable (consistent with any
possible outcome), it has no scientific content.

---

## Evaluation checklist

Before submitting a proposed solution, verify:

- [ ] Checked against VERIFIED-RESULTS.md (no contradictions)
- [ ] Addressed all relevant failed approaches from the problem file
- [ ] Provided numerical verification at specific test points
- [ ] All references include specific equation/theorem numbers
- [ ] Dimensional analysis passes
- [ ] At least two limiting cases checked
- [ ] Scope of validity explicitly stated
- [ ] Proof/evidence/conjecture distinction clear
- [ ] No circularity
- [ ] Reproducible (seeds, parameters, or step-by-step derivation)
- [ ] At least one falsifiable prediction identified
