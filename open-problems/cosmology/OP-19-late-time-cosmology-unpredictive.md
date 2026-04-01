---
id: OP-19
title: "Late-time cosmology: SCT corrections suppressed by 64 orders of magnitude"
domain: [cosmology]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: []
blocked-by: []
roadmap-tasks: [MT-2]
papers: ["1305.3034", "1404.0726"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-19: Late-time cosmology unpredictive

## 1. Statement

The SCT corrections to the Friedmann equations are suppressed by

  delta H^2 / H^2 ~ 1.3 * 10^{-64}

at the PPN-1 lower bound Lambda >= 2.565 meV. This makes SCT
predictions for late-time cosmology (dark energy, H_0 tension, S_8
tension, structure growth) unobservable by any foreseeable experiment.
Determine whether any mechanism within the spectral action framework
can generate cosmologically relevant corrections at IR scales, or
prove that this is impossible.

## 2. Context

The result delta H^2 / H^2 ~ 10^{-64} is a firmly established
NEGATIVE result (MT-2, verified to 100 digits with mpmath). It arises
because the SCT form factors F_1(Box/Lambda^2) and F_2(Box/Lambda^2)
are entire functions of Box: they have Taylor expansions convergent on
the entire complex plane. At cosmological momenta k ~ H_0 ~ 10^{-33}
eV, the argument k^2/Lambda^2 is of order 10^{-59} (for Lambda ~ meV),
and the form factors evaluate to essentially 1 + O(10^{-59}).

The suppression is not an artifact of perturbation theory. The entire
function property guarantees that the corrections are analytic at
k^2 = 0, so no non-perturbative IR enhancement is possible within the
framework.

The effective equation of state is w_eff = -1 to 63 decimal places.
The SCT Friedmann equations are indistinguishable from the standard
Lambda-CDM equations at all observable scales.

## 3. Known Results

- **Quantitative suppression (MT-2, verified):**
  delta H^2 / H^2 = 1.3 * 10^{-64} at Lambda = 2.565 meV.

- **w_eff = -1 to 63 digits:** the effective dark energy equation of
  state is indistinguishable from a cosmological constant.

- **Maggiore's argument (adapted to SCT):** UV nonlocality (entire
  functions) cannot generate IR nonlocality (inverse powers of Box).
  The spectral action produces form factors that are entire in
  Box/Lambda^2. To affect late-time cosmology, one needs terms like
  R * Box^{-1} * R (Deser-Woodard model) or m^2 * Box^{-1} * R
  (Maggiore-Mancarella model). These are structurally absent from
  the spectral action.

- **De Sitter stability (NT-4c):** de Sitter is an exact solution.
  Perturbations about de Sitter are stable and decay exponentially.

- **c_T = c (proven):** gravitational wave speed equals the speed of
  light, consistent with GW170817.

## 4. Failed Approaches

1. **Resummation of higher-loop terms.** Hoped that summing the
   perturbative series in alpha_C and alpha_R to all orders might
   produce non-analytic IR terms. This is excluded by the
   Barvinsky-Vilkovisky covariant expansion: each order in the
   curvature expansion contributes entire functions of Box, and
   a convergent series of entire functions is entire.

2. **Nonlocal action from spectral flow.** Investigated whether
   Postulate 5 in the V2 (spectral flow) formulation could generate
   non-entire corrections to the action. No concrete computation was
   possible because V2 is not formulated beyond a schematic level (OP-02).

3. **Causal set IR effects.** The FND-1 programme investigated whether
   the discrete structure of causal sets could produce macroscopic
   cosmological effects. The result (MT-2 rederivation via causal set
   methods) confirms the 10^{-64} suppression.

## 5. Success Criteria

- A rigorous proof that no modification of the spectral action (within
  the Chamseddine-Connes framework) can produce delta H^2 / H^2 > 10^{-10}
  at late times. This would close the problem as a structural limitation.
- OR: identification of a non-perturbative mechanism (e.g., from
  Postulate 5, or from the measure in the spectral path integral) that
  generates IR-singular contributions to the Friedmann equations.
- Any claimed mechanism must be consistent with GW170817 (c_T = c)
  and solar-system tests (PPN-1).

## 6. Suggested Directions

1. **No-go theorem.** Formalize the argument that entire functions
   cannot generate IR nonlocality into a rigorous mathematical
   statement. Key ingredients: the spectral action is a Laplace
   transform of the spectral density; the spectral density has compact
   support (cutoff at Lambda); the Laplace transform of a compactly
   supported distribution is entire.

2. **Modified spectral triples.** Consider spectral triples where the
   Dirac operator D has a continuous spectrum extending to zero (as in
   non-compact manifolds). In this case, the spectral action
   Tr(f(D^2/Lambda^2)) may receive contributions from the zero-mode
   sector that are not entire functions.

3. **Emergent dark energy.** Accept that the spectral action does not
   predict dark energy and investigate whether the cosmological
   constant arises from a different sector of the theory (e.g., the
   vacuum energy of the finite spectral triple, or the topological
   contribution from the Euler density E_4).

4. **Comparison with other UV-complete frameworks.** String theory,
   asymptotic safety, and loop quantum gravity all face similar
   difficulties in connecting UV-complete gravitational physics to
   late-time cosmology. Study whether SCT's negative result is
   generic or specific to the spectral action.

## 7. References

1. Maggiore, M. and Mancarella, M. (2014). "Nonlocal gravity and dark
   energy." Phys. Rev. D 90, 023005. arXiv:1305.3034.
2. Dirian, Y. et al. (2014). "Non-local gravity. II. Exact solutions
   and perturbations." arXiv:1404.0726.
3. Deser, S. and Woodard, R. P. (2007). "Nonlocal cosmology." Phys.
   Rev. Lett. 99, 111301. arXiv:0706.2151.
4. Alfyorov, D. (2026). "Nonlinear field equations and FLRW cosmology
   from the spectral action." DOI:10.5281/zenodo.19098027.

## 8. Connections

- **OP-17 (scalaron mass):** independent problem (UV vs IR), but both
  illustrate the difficulty of cosmological predictions in SCT.
- **OP-18 (dilaton inflation):** if dilaton inflation is also excluded,
  SCT has no internal mechanism for either early-universe or late-time
  cosmology without BSM extensions.
- **OP-02 (Postulate 5):** the only remaining hope for IR effects lies
  in non-perturbative contributions from the path integral measure,
  which requires a formulated dynamical principle.
- Independent of Gap G1 (OP-01): cosmology uses FLRW backgrounds where
  C = 0.
