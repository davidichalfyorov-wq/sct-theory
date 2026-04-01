---
id: OP-24
title: "Spectral dimension definition dependence: which definition is physically correct?"
domain: [spectral-dimension, formal-theory]
difficulty: hard
status: open
deep-research-tier: A
blocks: []
blocked-by: [OP-07]
roadmap-tasks: [NT-3]
papers: ["1203.4515", "1304.7247", "1507.00330"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-24: Spectral dimension definition dependence resolution

## 1. Statement

Four physically motivated definitions of the spectral dimension d_S
give different UV predictions in SCT:

  CMN (propagator scaling): d_S = 2
  Heat kernel (standard):   d_S = 4
  ASZ/fakeon (projected):   d_S = 0 -> 4
  Mittag-Leffler (ML):      d_S ~ 2 at sigma* -> 4

Determine which definition is physically correct, or prove that the
spectral dimension is not a well-defined observable in SCT and should
be replaced by an alternative quantity.

## 2. Context

The spectral dimension has been proposed as a universal observable
that distinguishes quantum gravity theories. Most approaches predict
d_S -> 2 in the UV. SCT breaks this pattern because the dressed
propagator has ghost poles that modify the return probability.

The definition dependence arises from the treatment of ghost
contributions:

- **CMN:** uses the overall UV scaling of the propagator, ignoring
  individual poles. The result d_S = 2 follows from G_TT ~ 1/k^4
  at large k (since Pi_TT ~ k^2/Lambda^2).

- **Heat kernel:** uses Tr(exp(-sigma D^2)) directly. The operator D^2
  is a standard second-order differential operator on a 4-manifold,
  so d_S = 4 at all scales. The nonlocal corrections enter only through
  the effective potential, not through the kinetic structure.

- **ASZ/fakeon:** projects out the ghost states from the Hilbert space
  using the Anselmi prescription. The spectral function then has only
  the physical graviton pole, and d_S = 4 at all scales. The transient
  d_S = 0 at very short distances reflects the cancellation between
  the physical and projected-out poles.

- **ML (most detailed):** uses the full Mittag-Leffler expansion of
  the propagator, including all 8 poles. The return probability is a
  sum of exponentials with both positive and negative coefficients
  (the negative ones from ghost poles). The physical region sigma >
  sigma* gives d_S ~ 2 near sigma*, flowing to 4 in the IR. The
  unphysical region sigma < sigma* has P < 0 (negative return
  probability).

The residue sum is sum R_n = -1.034, which determines the location
of sigma* and the overall strength of the ghost-induced P < 0.

## 3. Known Results

- **ML residues (100-digit verified):**
  sum R_n = -1.034 (sum of 8 catalogued ghost residues).
  W(0) = 1 + sum R_n = -0.034 < 0.
  This is NOT a truncation artifact: additional poles (if they exist)
  have negative real residues, making W(0) more negative.

- **sigma* ~ 0.01 Lambda^{-2}:** the crossover scale where P changes
  sign. Below sigma*, the ML return probability is unphysical.

- **Physical flow (sigma > sigma*):**
  d_S(sigma = 0.05/Lambda^2) = 2.47
  d_S(sigma = 0.10/Lambda^2) = 2.06
  d_S(sigma = 1.0/Lambda^2) = 3.26
  d_S(sigma = 10.0/Lambda^2) = 4.00

- **Comparison with other QG approaches:** asymptotic safety (d_S = 2),
  CDT (d_S ~ 1.8-2.0), Horava-Lifshitz (d_S = 2), spin foams
  (d_S ~ 2), all predict d_S -> 2 universally. SCT is the first
  exception if the CMN definition is adopted, and the only case with
  definition dependence.

- **Belenchia et al. (2015):** showed that the causal set
  d'Alembertian gives d_S -> 2 in the UV universally in all dimensions.
  This is a different computation (discrete rather than continuum)
  and uses a specific nonlocal operator.

## 4. Failed Approaches

1. **Wick rotation argument.** Attempted to resolve the ambiguity by
   performing a Wick rotation to Euclidean signature, where the heat
   kernel definition is unambiguous. The difficulty is that the ghost
   poles move in the complex plane under Wick rotation, and the
   Euclidean spectral dimension differs from the Lorentzian one.

2. **Operational definition.** Tried to define d_S through a physical
   diffusion experiment (e.g., photon propagation). This reduces to
   the propagator definition (CMN) for free fields but becomes
   ambiguous when interactions are included.

3. **Truncation to different numbers of poles.** Computed d_S using
   2, 4, 6, and 8 poles in the ML expansion. The result is unstable:
   sigma* shifts by 67% when going from 2 to 8 poles, and the UV
   value of d_S changes qualitatively.

## 5. Success Criteria

- Identify a physically unambiguous definition of d_S that does not
  depend on the treatment of ghost poles, OR prove that no such
  definition exists.
- If a definition is selected: compute d_S(sigma) for all sigma > 0,
  including the deep UV, and compare with other QG predictions.
- If no definition exists: identify an alternative observable that
  captures the effective dimensionality of SCT spacetime at short
  distances and is independent of the ghost prescription.
- The result must be consistent with the ghost resolution (OP-07):
  the physical d_S must use the same ghost treatment as the unitarity
  analysis (MR-2).

## 6. Suggested Directions

1. **Fakeon-consistent definition.** If the fakeon prescription
   (Anselmi) resolves the ghost problem (OP-07), adopt the
   ASZ/fakeon definition as the physical one. This gives d_S = 4
   at all scales (no dimensional reduction). Verify that this is
   consistent with the fakeon propagator rules.

2. **Spectral action definition.** Define d_S through the spectral
   action itself: d_S(Lambda) = -2 d(ln Tr(f(D^2/Lambda^2))) /
   d(ln Lambda). This uses the cutoff function f directly and avoids
   the propagator ambiguity. Compute this quantity and check whether
   it gives a scale-dependent dimension.

3. **Causal set d_S.** Use the Belenchia-Benincasa-Liberati discrete
   definition of d_S on Poisson sprinklings. This is independent of
   the continuum propagator and may resolve the ambiguity. The FND-1
   programme provides the computational infrastructure for this.

4. **Analytic continuation.** Study the ML spectral dimension in the
   complex sigma-plane. The P < 0 region may correspond to a physical
   regime in Lorentzian signature (timelike random walks) rather than
   an unphysical artifact. Investigate the Wick rotation of the
   random walk problem.

## 7. References

1. Calcagni, G., Modesto, L. and Nardelli, G. (2012). "Quantum
   spectral dimension in quantum field theory." Int. J. Mod. Phys.
   D 25, 1650058. arXiv:1203.4515.
2. Modesto, L. and Rachwal, L. (2013). "Super-renormalizable and
   finite gravitational theories." Nucl. Phys. B 889, 228.
   arXiv:1304.7247.
3. Belenchia, A., Benincasa, D. M. T. and Liberati, S. (2015).
   "Nonlocal scalar quantum field theory from causal sets." JHEP 03,
   036. arXiv:1507.00330.
4. Reuter, M. and Saueressig, F. (2012). "Asymptotic safety,
   fractals, and cosmology." Lect. Notes Phys. 863, 185.
   arXiv:1205.5431.
5. Carlip, S. (2017). "Dimension and dimensional reduction in quantum
   gravity." Class. Quant. Grav. 34, 193001. arXiv:1705.05417.

## 8. Connections

- **Blocked by OP-07 (Lorentzian ghost):** the ghost resolution
  determines which definition of d_S is physical. If fakeon: d_S = 4.
  If Lee-Wick: d_S ~ 2 near sigma*. If dark matter: d_S undefined.
- **OP-25 (two-loop corrections):** two-loop contributions modify the
  residues R_n and shift sigma*, potentially changing the ML result
  qualitatively.
- **Independent of OP-01 (Gap G1):** the spectral dimension is computed
  from the propagator on flat backgrounds; no Weyl curvature is
  involved.
- **Independent of cosmological problems (OP-17 through OP-20):** the
  spectral dimension is a UV quantity; cosmology probes IR scales.
