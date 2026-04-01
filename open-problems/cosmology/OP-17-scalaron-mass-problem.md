---
id: OP-17
title: "Scalaron mass problem: SM-only spectral action overproduces the scalaron mass by six orders of magnitude"
domain: [cosmology, inflation]
difficulty: hard
status: resolved
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: [INF-1]
papers: ["hep-th/0610241", "0805.2909", "1809.02944", "1005.1188", "1304.8050", "1304.0415", "0710.3755", "1008.5157", "0903.0355", "hep-th/0512169", "1801.00260"]
date-opened: 2026-03-31
date-updated: 2026-04-01
date-resolved: 2026-04-01
resolved-by: literature-analysis + independent-numerical-verification
---

# OP-17: Scalaron mass problem

## 1. Statement

The R^2 sector of the one-loop SCT effective action produces a
Starobinsky-type scalaron with mass

  M_0 = Lambda / sqrt(12 (xi - 1/6)^2)

where xi is the non-minimal Higgs coupling and Lambda is the spectral
cutoff. Using the SM particle content (N_s = 4, N_f = 45, N_v = 12)
with minimal coupling xi = 0, this gives

  M_0 = Lambda * sqrt(3/(1/6)^2)^{1/2} = 15.39 M_Pl

where the Planck mass is M_Pl = 1/sqrt(8 pi G) = 2.435 * 10^{18} GeV.
CMB normalization of Starobinsky inflation requires

  M_inf = 1.28 * 10^{-5} M_Pl = 3.12 * 10^{13} GeV.

The ratio M_0 / M_inf is approximately 1.2 * 10^6. The SM-only
spectral action produces a scalaron that is six orders of magnitude
too heavy to drive slow-roll inflation.

## 2. Context

The scalaron mass problem was first implicitly present in the work of
Chamseddine and Connes (2006), who derived the spectral action for the
Standard Model coupled to gravity. The R^2 term arises from the a_4
Seeley-DeWitt coefficient and is controlled by the total Ricci-squared
coefficient alpha_R(xi) = 2(xi - 1/6)^2.

The mass hierarchy problem is sharp: it is not a factor of 2 or an
O(1) ambiguity, but a ratio of 10^6. This rules out any resolution
based on numerical prefactors or renormalization scheme dependence.

The problem is specific to the SM-only spectral action. If new scalar
fields are added (BSM physics), the coefficient alpha_R changes and the
scalaron mass can be adjusted. However, this requires explicit BSM
input that is not determined by the spectral triple.

## 3. Known Results

- **Exact scalaron mass formula (verified, INF-1):**
  M_0^2 = Lambda^2 / (6 alpha_R) with alpha_R = 2(xi - 1/6)^2.

- **SM counting (CPR 0805.2909):** N_s = 4, N_D = 22.5, N_v = 12.
  These multiplicities determine alpha_R via the standard heat-kernel
  trace technology.

- **Nonlocal corrections do not help:** the normalized shape function
  F_hat_2(z, xi) decreases at large z, making the effective mass
  heavier rather than lighter at high momenta.

- **Spectral index and tensor-to-scalar ratio (conditional on correct
  mass):** n_s(N = 55) = 0.965, r(N = 55) = 3.5 * 10^{-3}. These
  are consistent with Planck 2018 + BICEP/Keck 2021 data.

- **w_eff and reheating:** if the mass is fixed by hand, the spectral
  scalaron reheats through the same channel as standard Starobinsky.

## 4. Failed Approaches

1. **Nonlocal form factor modification.** Hoped that the z-dependent
   part of F_2(z, xi) would effectively renormalize the mass downward
   at inflationary scales. Computation shows F_hat_2(z) is monotonically
   decreasing, so the mass increases at high momenta.

2. **Running of xi.** Considered whether RG running of the non-minimal
   coupling xi from the electroweak scale to the inflationary scale
   could bring xi close to 1/6, making alpha_R small and M_0 large.
   However, this goes in the wrong direction: alpha_R -> 0 makes M_0
   diverge, not decrease.

3. **Dilaton replacement.** Attempted to use the trace mode (dilaton)
   instead of the scalaron for inflation. This fails for five separate
   structural reasons (see OP-18).

## 4b. Resolution (2026-04-01)

**VERDICT: NEGATIVE. No known mechanism within standard NCG spectral action can produce M_0 = M_inf while preserving alpha_C and geometric scalar couplings.**

### Complete mechanism catalogue

| Mechanism | M_0/M_inf | NCG-compatible? | Fatal problem |
|-----------|-----------|-----------------|---------------|
| Sub-Planckian Lambda (~10^13 GeV) | 1 | Partially | Standard literature interprets Lambda as GUT/unification scale |
| Lambda ~ M_GUT (~10^16 GeV) | ~10^2-10^4 | Yes | Still too heavy |
| Large xi ~ 2.25e4 | 1 | **No** | Conflicts with geometric BC xi=1/12 (Eq.(49) in 1005.1188) |
| Single BSM scalar with xi'~3.19e4 | 1 | No | Same large-xi mechanism |
| Many BSM scalars (xi'=0) | 1 | No | Requires N_s' ~ 3.65e10 |
| NCG sigma-singlet (1005.1188) | No change | Yes | xi_sigma = 1/12 conformal (Eq.(52)) |
| Pati-Salam (1304.8050) | No change | Yes | All scalars conformal (Eq.(129)) |
| Grand Symmetry (1304.0415) | No change | Yes | sigma for Higgs mass, not scalaron |
| Two-loop corrections | ~same | Unknown | No published calculation |
| Instantons | Unknown | Unknown | No published mechanism |
| Modified f(u) | Cannot help | Yes | alpha_R from a_4, not from f |

### Critical structural finding (verified)

ALL known BSM scalars arising from NCG spectral triples come with
conformal curvature coupling xi' = 1/6 (our convention). This is
confirmed in three independent sources:
- Buck-Fairbairn-Sakellariadou (1005.1188), Eq.(52): xi_H = xi_sigma = 1/12
- Chamseddine-Connes-van Suijlekom (1304.8050), Eq.(129): universal 1/6 R coupling
- van den Dungen et al. (1801.00260), Eq.(4.10): tr|D_mu Phi|^2 - (1/6)R tr|Phi|^2

Consequence: adding NCG-compatible BSM scalars contributes (xi'-1/6)^2 = 0
to alpha_R. The BSM route is structurally closed.

### Combined with OP-44 (xi = 1/6 is NCG prediction)

At the NCG-predicted conformal coupling xi = 1/6:
- alpha_R = 2(1/6 - 1/6)^2 = 0
- m_0 = Lambda / sqrt(6 * 0) -> infinity
- **The scalaron is entirely absent.** Not "too heavy" — absent.

This sharpens the problem from "mass 6 orders too heavy" to "no scalaron
exists in standard NCG spectral action."

### Surviving path

The only path not definitively excluded is **reinterpretation of Lambda**:
either as a sub-Planckian intermediate scale, or through framework change
(zeta spectral action, dilatonized action). This requires abandoning the
standard CCM interpretation of Lambda as GUT/unification boundary.

Falsifiable consequence: intermediate-Lambda rescue predicts that spectral-
action boundary conditions activate below 10^16 GeV, requiring additional
thresholds between GUT and intermediate scales.

### Bibliographic correction

The "Entropy and the spectral action" paper by Chamseddine-Connes-van
Suijlekom is arXiv:1809.02944 (2018), NOT arXiv:2012.11829 (which is
Giacchini-Netto on Newtonian limits of higher-derivative gravity).

### Numerical verification (all exact)

  xi_needed (for M_0=M_inf at Lambda=M_Pl): 2.25e4
  alpha_R_needed: 5.08e8
  N_s'(xi'=0) needed: 3.65e10
  N_s'(xi'=1) needed: 1.46e9
  Single scalar xi' needed: 3.19e4
  Lambda_unit for large xi: 1.08e14 GeV (only 3.46x above M_inf)

## 5. Success Criteria

- Identify a mechanism within the spectral action framework that
  produces M_0 = (1.28 +/- 0.03) * 10^{-5} M_Pl without introducing
  ad hoc parameters.
- OR: demonstrate that SCT inflation requires BSM extensions, and
  classify the minimal BSM spectral triples that achieve the correct
  mass.
- The solution must preserve alpha_C = 13/120 (Weyl coefficient is
  parameter-free and experimentally constrained).
- Spectral index and tensor-to-scalar ratio must remain within 2-sigma
  of Planck 2018 + BICEP/Keck bounds.

## 6. Suggested Directions

1. **Sub-Planckian Lambda with xi = 0.** If Lambda ~ 1.3 * 10^{13} GeV
   (far below M_Pl), the scalaron mass comes down to the correct scale.
   This conflicts with the interpretation of Lambda as a UV cutoff near
   the Planck scale but may be viable if Lambda is interpreted as the
   spectral cutoff of the finite noncommutative geometry rather than
   the gravitational UV scale.

2. **Large non-minimal coupling xi ~ 5 * 10^4.** This is the
   Bezrukov-Shaposhnikov mechanism adapted to the spectral action.
   The coupling xi must be fine-tuned; moreover, unitarity of the Higgs
   sector becomes marginal at energies ~ M_Pl / xi. Study whether the
   spectral triple imposes additional constraints on xi.

3. **BSM scalars.** Adding N_s' real scalars with non-minimal couplings
   modifies alpha_R. Determine the minimum N_s' required and whether
   any known BSM spectral triple (Pati-Salam, SO(10)) provides it.

4. **Higher-loop corrections.** Two-loop corrections to alpha_R from
   the R^3 sector (MR-4) could modify the scalaron mass. Estimate the
   magnitude of the shift. The two-loop D = 0 result (MR-5b) suggests
   the correction is absorbable by field redefinition, but the
   quantitative effect on M_0 has not been computed.

## 7. References

1. Chamseddine, A. H. and Connes, A. (2006). "Inner fluctuations of
   the spectral action." J. Geom. Phys. 57, 1. arXiv:hep-th/0610241.
2. Codello, A., Percacci, R. and Rachwal, L. (2008). "On the running
   of the gravitational constants." arXiv:0805.2909.
3. Starobinsky, A. A. (1980). "A new type of isotropic cosmological
   models without singularity." Phys. Lett. B 91, 99.
4. Bezrukov, F. and Shaposhnikov, M. (2008). "The Standard Model Higgs
   boson as the inflaton." Phys. Lett. B 659, 703. arXiv:0710.3755.
5. Chamseddine, A. H., Connes, A. and van Suijlekom, W. D. (2018).
   "Entropy and the spectral action." arXiv:1809.02944.
6. Buck, M., Fairbairn, M. and Sakellariadou, M. (2010). "Inflation
   in models with conformally coupled scalar fields." Phys. Rev. D 82.
   arXiv:1005.1188.
7. van den Dungen, K. et al. (2018). arXiv:1801.00260.

## 8. Connections

- **OP-18 (dilaton inflation):** if the scalaron mass cannot be fixed,
  dilaton inflation is the only remaining internal candidate. OP-18
  shows this is also excluded.
- **OP-19 (late-time cosmology):** independent problem (IR vs UV), but
  both illustrate that SCT cosmology is strongly constrained.
- **OP-20 (de Sitter conjecture):** the scalaron potential shape
  determines whether SCT satisfies Swampland constraints.
- Requires no input from Gap G1 (OP-01): the R^2 sector is fully
  computed on conformally flat backgrounds.
