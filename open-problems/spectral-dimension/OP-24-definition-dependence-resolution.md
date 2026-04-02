---
id: OP-24
title: "Spectral dimension definition dependence: which definition is physically correct?"
domain: [spectral-dimension, formal-theory]
difficulty: hard
status: resolved
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: [NT-3]
papers: ["1705.05417", "1507.00330", "hep-th/0505113", "hep-th/0508202", "0812.2214", "0902.3657", "1408.0199", "1204.2550", "1304.2709", "1806.03605"]
date-opened: 2026-03-31
date-updated: 2026-04-02
date-resolved: 2026-04-02
resolved-by: literature-analysis + independent-verification
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

## 3b. Resolution (2026-04-02)

**VERDICT: RESOLVED. d_S is not a universal observable — it is a probe-dependent dimensional estimator. In the physical fakeon-projected sector, SCT stays effectively 4D.**

### Core finding: different QG programs use different definitions

Six QG programs use six DIFFERENT constructions under the same symbol d_S:

| Program | Definition | UV d_S | What is probed |
|---------|-----------|--------|----------------|
| CDT | Ensemble random walk on triangulations, eqs.(3),(6),(8) of hep-th/0505113 | 1.80±0.25 | Genuinely probabilistic lattice walk |
| AS/QEG | RG-improved fictitious diffusion, eqs.(3.5),(3.12) of hep-th/0508202 | 2 | Euclidean diffusion for effective Laplacian |
| Horava-Lifshitz | Anisotropic diffusion from UV dispersion, eqs.(2.1),(4.2)-(4.5) of 0902.3657 | 2 | Modified-diffusion, not propagator scaling |
| LQG/Modesto | AS-like heat kernel on effective geometry, eqs.(11)-(14) of 0812.2214 | 2 | Effective metric, not spin-foam random walk |
| CMN/Stelle QFT | Schwinger kernel from inverse propagator, eqs.(2)-(7) of 1408.0199 | 2 | Full inverse propagator (off-shell content) |
| Causal sets (BBL) | Spectral dim. of nonlocal scalar d'Alembertian, eqs.(5)-(12) of 1507.00330 | 2 | Specific scalar operator, not tensor |

The statement "all QG approaches predict d_S→2" is correct only in a weak
sense: many different Euclidean/diffusion probes give 2, but they measure
different physical quantities. Carlip (1705.05417) explicitly warns that
different dimension definitions need not coincide.

### Three-level classification for SCT

| Level | Definition | Value | Physical status |
|-------|-----------|-------|-----------------|
| d_S^{kin} | Kinematic dimension of background operator | 4 | Trivial (standard 2nd-order D^2) |
| d_S^{full} | Full unprojected ML kernel (all 8+ poles) | ~2 near sigma* | Diagnostic (includes ghost/fakeon modes) |
| d_S^{phys} | Fakeon-projected physical sector | 4 | Observable (only physical spectrum) |

Rationale for d_S^{phys} = 4: Fakeons by definition do not belong to
the physical spectrum and cannot be external observable states (Anselmi-
Piva 1806.03605). Including them in d_S is including non-physical degrees
of freedom in a putatively physical observable. The ASZ/fakeon definition
projects them out, leaving d_S = 4.

### Implication: "SCT predicts d_S→2" is incorrect

The correct statement: "The unprojected auxiliary spectral probe shows
dimensional reduction in a transient regime near sigma*, but the physical
fakeon-projected sector remains effectively four-dimensional."

The ML d_S ~ 2 near sigma* is useful as a diagnostic of ghost/pole
structure, not as a physical prediction.

### P<0 problem in context

The sign change of P(sigma) below sigma* is not unique to SCT (Calcagni,
Eichhorn, Saueressig 1304.7247 showed many QG-improved kernels are not
positive semidefinite). But SCT's version is more severe (traced return
probability itself changes sign, not just density). Interpretation: the
probe enters the fakeon sector, which has no probabilistic meaning.

CDT by contrast has P > 0 by construction (genuine random walk); its
short-scale issues are lattice artifacts, not negative probability.

### Causal set d_S and SCT

BBL (1507.00330) compute d_S for a SCALAR causal-set-inspired nonlocal
d'Alembertian. This is NOT the same as the SCT tensor propagator Pi_TT.
Transfer to SCT requires matching the continuum symbol of the discrete
operator to the SCT kinetic operator — which has not been done.

### Key literature not in original problem file

- Calcagni "Quantum spectral dimension in QFT" (1408.0199): QFT/Schwinger
  interpretation, eqs.(2)-(7)
- Reuter-Saueressig review: eqs.(34),(35),(49)-(52) give full AS definition
- Horava "Spectral dimension of the universe in QG at a Lifshitz point"
  (0902.3657): anisotropic diffusion
- Amelino-Camelia, Arzano, Magueijo: criticize d_S as not fully physical
  (sensitive to off-shell sectors and Euclideanisation)

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
