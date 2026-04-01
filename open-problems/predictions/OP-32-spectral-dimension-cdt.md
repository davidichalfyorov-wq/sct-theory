---
id: OP-32
title: "Spectral dimension comparison with CDT lattice simulations"
domain: [predictions, spectral-dimension]
difficulty: medium
status: open
deep-research-tier: B
blocks: []
blocked-by: [OP-24]
roadmap-tasks: [COMP-1]
papers: ["0505113", "0903.1024", "1205.3637"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-32: Spectral dimension comparison with CDT lattice simulations

## 1. Statement

Compute the full spectral dimension flow profile d_S(sigma) in SCT
from the UV (sigma -> 0) to the IR (sigma -> infinity) and compare
quantitatively with CDT lattice simulation data. Extract the
modified dispersion relation parameter eta and compare the SCT
prediction with CDT measurements across the full diffusion time range.

## 2. Context

The spectral dimension d_S(sigma) measures the effective dimensionality
of spacetime as probed by a diffusion process at scale sigma:

  d_S(sigma) = -2 d ln P(sigma) / d ln sigma

where P(sigma) = Tr(e^{-sigma D^2}) is the return probability. In
classical 4D spacetime, d_S = 4 for all sigma. In quantum gravity
programs, d_S flows from d_S ~ 2 in the UV to d_S = 4 in the IR.

This dimensional reduction was first observed in CDT lattice
simulations (Ambjorn et al., 0505113), where d_S ~ 1.80 +/- 0.25
was measured in the UV. It was subsequently found in asymptotic
safety (Lauscher-Reuter), causal set theory (Eichhorn-Mizera),
and loop quantum gravity (Modesto).

In SCT, the NT-3 computation established that d_S is definition-
dependent: four distinct definitions (heat kernel trace, modified
Laplacian, momentum space, and spectral zeta function) give
different UV values. The physically preferred definition (modified
Laplacian with SCT propagator) gives d_S ~ 2 in the UV, consistent
with CDT, but the full profile d_S(sigma) has not been compared
quantitatively with CDT data.

## 3. Known Results

- NT-3 spectral dimension computation: COMPLETE. Four definitions
  computed. Physical ML (modified Laplacian) definition gives
  d_S flow from approximately 2 in the UV to 4 in the IR.
- CDT lattice data (Ambjorn et al., 0505113, 0903.1024):
  d_S(UV) ~ 1.80 +/- 0.25, d_S(IR) = 4.02 +/- 0.10.
- CDT fit function: d_S(sigma) = 4 a sigma / (1 + a sigma) with
  a ~ 0.037 (arbitrary lattice units).
- AS prediction: d_S(UV) = 2 exactly (from anomalous dimension
  eta_N* = -2 at the Reuter fixed point).
- Causal set prediction: d_S(UV) = 2 from the Myrheim-Meyer
  dimension estimator.
- SCT P(sigma) < 0 issue: the return probability computed from
  the dressed propagator can become negative at small sigma due
  to ghost contributions. This is flagged in NT-3 as a potential
  problem requiring the fakeon prescription.

## 4. Failed Approaches

The NT-3 computation established the qualitative picture but stopped
short of a quantitative comparison with CDT. The main obstacles are:

1. The CDT data are in lattice units, requiring a non-trivial
   matching of the lattice scale a to the SCT cutoff Lambda. This
   matching has not been performed.

2. The SCT d_S(sigma) depends on the definition, and the CDT
   measurement corresponds to a specific lattice diffusion process.
   The correspondence between the lattice diffusion operator and
   the continuum SCT operator is not established.

3. The P(sigma) < 0 issue at small sigma complicates the UV
   extraction. The fakeon prescription modifies the spectral sum
   in a way that has not been computed for d_S.

## 5. Success Criteria

- Full d_S(sigma) profile in SCT using the ML definition, tabulated
  at 50+ sigma values spanning UV to IR.
- Quantitative comparison with CDT data (Ambjorn et al. 0903.1024,
  Table 1): chi-squared or Kolmogorov-Smirnov test.
- Extraction of the modified dispersion parameter eta defined by
  omega^2 = k^2 + alpha k^{2+2eta} at short distances. In SCT,
  eta is determined by the UV asymptotic of Pi_TT.
- Comparison of eta with CDT lattice extraction and AS prediction.
- Statement of whether the CDT d_S(UV) ~ 1.80 is consistent with
  the SCT ML-definition value.

## 6. Suggested Directions

1. Lattice-continuum matching: use the CDT lattice spacing a and
   the dimensionful diffusion time sigma to define sigma/a^2. Match
   a to 1/Lambda by requiring d_S(IR) = 4 in both systems. This
   gives a one-parameter fit to the CDT data.

2. Fakeon spectral dimension: compute d_S(sigma) with the fakeon
   prescription applied to the ghost poles. Replace the delta-function
   spectral contributions at z_L and z_0 with their fakeon (principal
   value) counterparts and evaluate the resulting P(sigma).

3. eta extraction: from the SCT propagator Pi_TT(z), the modified
   dispersion relation is omega^2 = k^2 Pi_TT(k^2/Lambda^2). For
   k >> Lambda, Pi_TT ~ -(83/6), giving omega^2 ~ -(83/6) k^2
   (Euclidean), which implies eta = 0 with a modified coefficient.
   Compare this with the CDT eta.

4. d=2 test: compute d_S(sigma) in d = 2 dimensions (where exact
   results are available for CDT) and verify agreement before
   tackling d = 4.

## 7. References

1. Ambjorn, J., Jurkiewicz, J. and Loll, R. (2005). "Spectral
   dimension of the universe." Phys. Rev. Lett. 95, 171301.
   arXiv:0505113.
2. Ambjorn, J., Jurkiewicz, J. and Loll, R. (2009). "The spectral
   dimension of the quantum universe is scale dependent." Phys. Rev.
   Lett. 95, 171301. arXiv:0903.1024.
3. Carlip, S. (2017). "Dimension and dimensional reduction in quantum
   gravity." Class. Quant. Grav. 34, 193001. arXiv:1705.05417.
4. Modesto, L. (2009). "Fractal spacetime from the area spectrum
   and dimension." Class. Quant. Grav. 26, 242002. arXiv:1205.3637.

## 8. Connections

- **Blocked by OP-24** (d_S physical definition): the comparison
  requires choosing among the four NT-3 definitions.
- Related to **OP-31** (form factor comparison with AS): spectral
  dimension is determined by the form factor in the UV.
- Related to **OP-33** (cross-program table): d_S is a key row
  in the comparison table.
- Related to **OP-23** (P(sigma) positivity): the P < 0 issue
  at small sigma could invalidate the d_S comparison.
